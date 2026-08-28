"""Search over curb allocations: grid, random, and differential evolution.

Why three algorithms
--------------------
The objective is a stochastic simulation with no gradient and a budget measured
in simulation-hours, so the interesting question is not "which finds the best
point" but "which finds a good point per objective evaluation". Reporting all
three with their evaluation counts is the honest way to answer that, and makes
the central tradeoff visible: differential evolution needs no gradients and
handles a rugged surface, but it pays for that in evaluations, and every
evaluation here is several full simulations.

Because the objective is noisy, the reported optimum is re-evaluated on a larger,
independent seed set before it is compared with the baselines. Selecting a point
and reporting the *search-time* estimate of its value is a classic simulation-
optimization error - the winner's estimate is biased low by selection.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from src.config import VEHICLE_CLASSES, load_optimization_config
from src.experiments.provenance import Manifest
from src.optimization.baseline import evaluate_baselines
from src.optimization.objective import (
    CurbObjective,
    Evaluation,
    allocation_dict,
    project_to_simplex,
)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


@dataclass
class SearchResult:
    method: str
    allocation: dict[str, float]
    objective: float
    stderr: float
    n_evaluations: int
    n_simulations: int
    wall_time_s: float
    trace: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Search algorithms
# ---------------------------------------------------------------------------
def grid_search(objective: CurbObjective, step: float = 0.05, verbose: bool = True) -> SearchResult:
    """Exhaustive search over a simplex lattice.

    Complete within its resolution, and completely infeasible at any finer one:
    the number of points grows as O(1/step^2) for a 3-simplex and each point
    costs several simulations. It is included as the reference the cheaper
    methods are judged against.
    """
    t0 = time.perf_counter()
    lo = {c: objective.bounds[c][0] for c in VEHICLE_CLASSES}
    hi = {c: objective.bounds[c][1] for c in VEHICLE_CLASSES}
    values = {c: np.arange(lo[c], hi[c] + 1e-9, step) for c in VEHICLE_CLASSES}
    best: Evaluation | None = None
    trace = []
    for xp, xd in itertools.product(values["passenger"], values["delivery"]):
        xr = 1.0 - xp - xd
        if not (lo["ridehail"] - 1e-9 <= xr <= hi["ridehail"] + 1e-9):
            continue
        ev = objective.evaluate([xp, xd, xr])
        trace.append(ev.as_row())
        if best is None or ev.value < best.value:
            best = ev
            if verbose:
                print(f"  grid   J={ev.value:10.1f}  {_fmt(ev.allocation)}", flush=True)
    assert best is not None
    return SearchResult(
        "grid",
        best.allocation,
        best.value,
        best.stderr,
        objective.n_evaluations,
        objective.n_simulations,
        time.perf_counter() - t0,
        trace,
    )


def random_search(
    objective: CurbObjective, n_samples: int = 120, seed: int = 20260401, verbose: bool = True
) -> SearchResult:
    """Dirichlet sampling over the simplex, projected into the feasible box.

    A surprisingly strong baseline in low dimension, and the fairest control for
    "did the clever algorithm actually help?".
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    best: Evaluation | None = None
    trace = []
    for i in range(n_samples):
        x = rng.dirichlet(np.ones(len(VEHICLE_CLASSES)))
        ev = objective.evaluate(x)
        trace.append(ev.as_row())
        if best is None or ev.value < best.value:
            best = ev
            if verbose:
                print(
                    f"  random [{i + 1}/{n_samples}] J={ev.value:10.1f}  {_fmt(ev.allocation)}",
                    flush=True,
                )
    assert best is not None
    return SearchResult(
        "random",
        best.allocation,
        best.value,
        best.stderr,
        objective.n_evaluations,
        objective.n_simulations,
        time.perf_counter() - t0,
        trace,
    )


def differential_evolution_search(
    objective: CurbObjective,
    maxiter: int = 18,
    popsize: int = 8,
    tol: float = 0.01,
    mutation: tuple[float, float] = (0.5, 1.0),
    recombination: float = 0.7,
    seed: int = 20260401,
    workers: int = 1,
    verbose: bool = True,
) -> SearchResult:
    """Global, gradient-free search with :func:`scipy.optimize.differential_evolution`.

    The search runs on an unconstrained box in ``[0, 1]^3``; every candidate is
    projected onto the feasible simplex inside the objective, so the equality
    constraint holds by construction rather than by penalty. Polishing is off:
    the local polish step assumes a smooth deterministic function, which a
    Monte-Carlo objective is not.
    """
    t0 = time.perf_counter()
    trace: list[dict[str, Any]] = []

    def f(x: Sequence[float]) -> float:
        ev = objective.evaluate(x)
        trace.append(ev.as_row())
        return ev.value

    res = differential_evolution(
        f,
        bounds=[(0.0, 1.0)] * len(VEHICLE_CLASSES),
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        mutation=mutation,
        recombination=recombination,
        seed=seed,
        polish=False,
        disp=verbose,
        workers=workers,
        updating="deferred" if workers != 1 else "immediate",
    )
    alloc = allocation_dict(project_to_simplex(res.x, objective.bounds))
    best = objective.evaluate([alloc[c] for c in VEHICLE_CLASSES])
    return SearchResult(
        "differential_evolution",
        best.allocation,
        best.value,
        best.stderr,
        int(getattr(res, "nfev", objective.n_evaluations)),
        objective.n_simulations,
        time.perf_counter() - t0,
        trace,
    )


def _fmt(alloc: dict[str, float]) -> str:
    return " ".join(f"{c[:3]}={alloc[c]:.3f}" for c in VEHICLE_CLASSES)


# ---------------------------------------------------------------------------
# Multi-objective analysis
# ---------------------------------------------------------------------------
PARETO_OBJECTIVES = {
    "passenger_search_min": ("passenger_search_time_min", "min"),
    "delivery_delay_min": ("delivery_delay_min", "min"),
    "ridehail_wait_min": ("ridehail_wait_min", "min"),
    "illegal_rate": ("illegal_parking_rate", "min"),
    "revenue": ("revenue_usd", "max"),
}


def is_pareto_efficient(costs: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated rows (all objectives minimised)."""
    n = costs.shape[0]
    efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not efficient[i]:
            continue
        dominated = np.all(costs <= costs[i], axis=1) & np.any(costs < costs[i], axis=1)
        if dominated.any():
            efficient[i] = False
    return efficient


def pareto_front(
    objective: CurbObjective,
    n_samples: int = 90,
    objectives: Sequence[str] | None = None,
    seed: int = 6060,
    verbose: bool = True,
) -> pd.DataFrame:
    """Sample the allocation simplex and label the non-dominated allocations.

    There is no single best curb allocation, only a frontier: pushing passenger
    search time down past a point necessarily pushes freight delay up. Producing
    the frontier - rather than one number - is what makes the result usable as
    policy advice.
    """
    names = list(objectives or PARETO_OBJECTIVES)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_samples):
        x = rng.dirichlet(np.ones(len(VEHICLE_CLASSES)))
        ev = objective.evaluate(x)
        row = ev.as_row()
        rows.append(row)
        if verbose and (i + 1) % 10 == 0:
            print(f"  pareto [{i + 1}/{n_samples}]", flush=True)
    df = pd.DataFrame(rows)
    costs = np.column_stack(
        [
            df[PARETO_OBJECTIVES[n][0]].to_numpy(dtype=float)
            * (1.0 if PARETO_OBJECTIVES[n][1] == "min" else -1.0)
            for n in names
        ]
    )
    df["pareto_efficient"] = is_pareto_efficient(costs)
    return df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def optimize(
    scenario: str = "baseline",
    methods: Sequence[str] = ("random", "differential_evolution"),
    workers: int = 1,
    out_dir: Path | None = None,
    confirm_seeds: int = 20,
    do_pareto: bool = True,
    opt_cfg: dict | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the full optimization study and write results."""
    opt_cfg = opt_cfg or load_optimization_config()
    search_cfg = opt_cfg.get("search", {})
    out = Path(out_dir) if out_dir is not None else RESULTS_DIR / "optimization"
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    objective = CurbObjective(scenario=scenario, opt_cfg=opt_cfg, workers=workers)

    if verbose:
        print("Reference allocations")
    baselines = evaluate_baselines(objective, scenario, verbose=verbose)

    results: dict[str, SearchResult] = {}
    for method in methods:
        n0, s0 = objective.n_evaluations, objective.n_simulations
        if verbose:
            print(f"\n{method} search")
        if method == "grid":
            r = grid_search(
                objective, step=float(search_cfg.get("grid", {}).get("step", 0.05)), verbose=verbose
            )
        elif method == "random":
            r = random_search(
                objective,
                n_samples=int(search_cfg.get("random", {}).get("n_samples", 120)),
                verbose=verbose,
            )
        elif method == "differential_evolution":
            de = search_cfg.get("differential_evolution", {})
            r = differential_evolution_search(
                objective,
                maxiter=int(de.get("maxiter", 18)),
                popsize=int(de.get("popsize", 8)),
                tol=float(de.get("tol", 0.01)),
                mutation=tuple(de.get("mutation", (0.5, 1.0))),
                recombination=float(de.get("recombination", 0.7)),
                # Parallelism lives *inside* the objective (across seeds), not in
                # the DE population loop: nesting the two would spawn child
                # processes from daemonic workers, and would also throw away the
                # objective's memo cache, which is per-process.
                workers=1,
                verbose=verbose,
            )
        else:
            raise ValueError(f"unknown search method: {method}")
        r.n_evaluations = objective.n_evaluations - n0
        r.n_simulations = objective.n_simulations - s0
        results[method] = r

    # Re-evaluate every candidate optimum on an independent, larger seed set.
    # Search-time values are optimistically biased by selection; these are the
    # numbers that get reported.
    confirm = list(range(9001, 9001 + confirm_seeds))
    if verbose:
        print(f"\nConfirmation runs on {confirm_seeds} independent seeds")
    confirmed: dict[str, Evaluation] = {}
    for name, alloc in [(k, v.allocation) for k, v in baselines.items()] + [
        (f"opt_{k}", r.allocation) for k, r in results.items()
    ]:
        ev = objective.evaluate([alloc[c] for c in VEHICLE_CLASSES], seeds=confirm)
        confirmed[name] = ev
        if verbose:
            print(
                f"  {name:24s} J = {ev.value:10.1f} +/- {ev.stderr:.1f}   {_fmt(ev.allocation)}",
                flush=True,
            )

    pareto = None
    if do_pareto:
        if verbose:
            print("\nPareto sampling")
        pareto = pareto_front(
            objective,
            n_samples=int(opt_cfg.get("pareto", {}).get("n_samples", 90)),
            verbose=verbose,
        )
        pareto.to_csv(out / "pareto_samples.csv", index=False)

    wall = time.perf_counter() - t0

    # -- write outputs ---------------------------------------------------------
    summary = _build_summary(baselines, results, confirmed, objective, wall)
    (out / "optimization_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    pd.DataFrame([ev.as_row() | {"label": name} for name, ev in confirmed.items()]).to_csv(
        out / "confirmed_allocations.csv", index=False
    )
    for name, r in results.items():
        pd.DataFrame(r.trace).to_csv(out / f"trace_{name}.csv", index=False)
    Manifest.create(
        experiment="optimization",
        scenario=scenario,
        seeds=objective.seeds,
        config=opt_cfg,
        n_workers=workers,
        methods=list(methods),
        confirm_seeds=confirm,
    ).finish(wall).write(out / "manifest.json")

    if verbose:
        print_report(summary)
        print(f"\nWritten to {out}")
    return summary


def _build_summary(
    baselines: dict[str, Evaluation],
    results: dict[str, SearchResult],
    confirmed: dict[str, Evaluation],
    objective: CurbObjective,
    wall: float,
) -> dict[str, Any]:
    current = confirmed["current"]
    best_name = min(
        (k for k in confirmed if k.startswith("opt_")), key=lambda k: confirmed[k].value
    )
    best = confirmed[best_name]

    def improvement(metric: str) -> float:
        b = current.metrics.get(metric)
        s = best.metrics.get(metric)
        if not b:
            return float("nan")
        return (s - b) / b * 100.0

    return {
        "scenario": objective.scenario,
        "seeds_per_evaluation": len(objective.seeds),
        "total_objective_evaluations": objective.n_evaluations,
        "total_simulations": objective.n_simulations,
        "wall_time_s": wall,
        "search": {
            k: {
                "allocation": r.allocation,
                "objective_search_estimate": r.objective,
                "n_evaluations": r.n_evaluations,
                "n_simulations": r.n_simulations,
                "wall_time_s": r.wall_time_s,
            }
            for k, r in results.items()
        },
        "confirmed": {
            k: {
                "allocation": v.allocation,
                "objective": v.value,
                "stderr": v.stderr,
                "components": v.components,
                "metrics": {
                    m: v.metrics.get(m)
                    for m in (
                        "passenger_search_time_min",
                        "delivery_delay_min",
                        "ridehail_wait_min",
                        "illegal_parking_rate",
                        "vmt_miles",
                        "revenue_usd",
                        "passenger_abandonment_rate",
                        "curb_occupancy_passenger",
                        "system_social_cost_usd",
                    )
                },
            }
            for k, v in confirmed.items()
        },
        "best_method": best_name,
        "improvement_vs_current_pct": {
            "objective": (best.value - current.value) / current.value * 100.0,
            "passenger_search_time_min": improvement("passenger_search_time_min"),
            "delivery_delay_min": improvement("delivery_delay_min"),
            "ridehail_wait_min": improvement("ridehail_wait_min"),
            "illegal_parking_rate": improvement("illegal_parking_rate"),
            "vmt_miles": improvement("vmt_miles"),
            "revenue_usd": improvement("revenue_usd"),
        },
        "significant": abs(best.value - current.value)
        > 1.96 * float(np.hypot(best.stderr, current.stderr)),
    }


def print_report(summary: dict[str, Any]) -> None:
    cur = summary["confirmed"]["current"]
    best = summary["confirmed"][summary["best_method"]]
    print("\n" + "=" * 64)
    print("CURB ALLOCATION OPTIMIZATION")
    print("=" * 64)
    print("\nCurrent allocation")
    for c in VEHICLE_CLASSES:
        print(f"  {c:10s} {cur['allocation'][c] * 100:6.1f}%")
    print(f"  objective  {cur['objective']:10.1f} +/- {cur['stderr']:.1f}")
    print(f"\nOptimized allocation ({summary['best_method'].replace('opt_', '')})")
    for c in VEHICLE_CLASSES:
        print(f"  {c:10s} {best['allocation'][c] * 100:6.1f}%")
    print(f"  objective  {best['objective']:10.1f} +/- {best['stderr']:.1f}")
    print("\nImprovement vs current (negative = better)")
    for k, v in summary["improvement_vs_current_pct"].items():
        print(f"  {k:28s} {v:+8.2f}%")
    sig = "yes" if summary["significant"] else "NO - within noise"
    print(f"\nDifference distinguishable from Monte-Carlo noise: {sig}")
    print(
        f"Objective evaluations: {summary['total_objective_evaluations']}  "
        f"simulations: {summary['total_simulations']}  wall: {summary['wall_time_s']:.0f}s"
    )


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m src.optimization.optimizer",
        description="Optimize the district-wide curb allocation.",
    )
    p.add_argument("--scenario", default="baseline")
    p.add_argument(
        "--methods",
        nargs="+",
        default=["random", "differential_evolution"],
        choices=["grid", "random", "differential_evolution"],
    )
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--confirm-seeds", type=int, default=20)
    p.add_argument("--no-pareto", action="store_true")
    p.add_argument("--out", default=str(RESULTS_DIR / "optimization"))
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    optimize(
        scenario=args.scenario,
        methods=args.methods,
        workers=args.workers,
        out_dir=Path(args.out),
        confirm_seeds=args.confirm_seeds,
        do_pareto=not args.no_pareto,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
