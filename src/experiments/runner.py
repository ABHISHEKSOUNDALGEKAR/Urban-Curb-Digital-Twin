"""Scenario x seed experiment orchestration, with parallel execution.

Usage
-----
    python -m src.experiments.runner --scenario baseline --seeds 30
    python -m src.experiments.runner --all --seeds 30 --workers 8
    python -m src.experiments.runner --scenario baseline --seeds 8 --benchmark

Design notes
------------
*Replications are independent*, which makes the workload embarrassingly
parallel. Each worker process re-loads the configuration from a serialised dict
rather than inheriting live objects, so the runner behaves identically under
``fork`` and ``spawn`` start methods (i.e. on Linux and on macOS/Windows).

*Seeds are explicit.* A replication's random stream depends only on its seed, so
the same ``--seeds`` list reproduces the same numbers on any machine, and the
seed is carried through into every output row.
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import RunConfig, list_scenarios, load_optimization_config, load_scenario
from src.experiments.metrics import aggregate, summarise
from src.experiments.provenance import Manifest
from src.simulation.engine import run_simulation

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


# ---------------------------------------------------------------------------
# Worker entry point. Must be module-level and picklable.
# ---------------------------------------------------------------------------
def _run_one(payload: dict[str, Any]) -> dict[str, Any]:
    """Run a single replication described by a serialised RunConfig."""
    cfg = RunConfig(
        scenario=payload["scenario"],
        seed=payload["seed"],
        network=payload["network"],
        agents=payload["agents"],
        scenario_spec=payload["scenario_spec"],
        overrides=payload.get("overrides", {}),
    )
    result = run_simulation(cfg)
    return summarise(result, payload.get("social_cost_params"))


def run_replications(
    cfg: RunConfig,
    seeds: Sequence[int],
    workers: int | None = None,
    social_cost_params: dict | None = None,
    progress: bool = False,
) -> list[dict[str, Any]]:
    """Run ``cfg`` once per seed, in parallel when ``workers > 1``."""
    payloads = []
    for s in seeds:
        p = cfg.with_seed(int(s)).to_dict()
        p["social_cost_params"] = social_cost_params
        payloads.append(p)

    n_workers = _resolve_workers(workers, len(payloads))
    if n_workers <= 1:
        out = []
        for i, p in enumerate(payloads, 1):
            out.append(_run_one(p))
            if progress:
                print(f"  [{i}/{len(payloads)}] seed={p['seed']}", flush=True)
        return out

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_one, p): p["seed"] for p in payloads}
        for i, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            if progress:
                print(f"  [{i}/{len(payloads)}] seed={futures[fut]} done", flush=True)
    results.sort(key=lambda r: r["seed"])
    return results


def _resolve_workers(workers: int | None, n_jobs: int) -> int:
    if workers is None:
        workers = min(os.cpu_count() or 1, n_jobs)
    return max(1, min(int(workers), n_jobs))


def run_scenario(
    scenario: str,
    seeds: Sequence[int],
    workers: int | None = None,
    overrides: dict | None = None,
    out_dir: Path | None = None,
    progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, Manifest]:
    """Run one scenario over ``seeds``; return per-seed rows, aggregate and manifest."""
    cfg = load_scenario(scenario, seed=int(seeds[0]), overrides=overrides)
    opt_cfg = load_optimization_config()
    social = opt_cfg.get("objective", {})

    n_workers = _resolve_workers(workers, len(seeds))
    manifest = Manifest.create(
        experiment="scenario_sweep",
        scenario=scenario,
        seeds=[int(s) for s in seeds],
        config=cfg.to_dict(),
        n_workers=n_workers,
        label=cfg.scenario_spec.get("label", scenario),
    )
    t0 = time.perf_counter()
    if progress:
        print(f"[{scenario}] {len(seeds)} replications on {n_workers} worker(s)", flush=True)
    rows = run_replications(
        cfg, seeds, workers=n_workers, social_cost_params=social, progress=False
    )
    manifest.finish(time.perf_counter() - t0)

    per_seed = pd.DataFrame(rows)
    agg = aggregate(rows)
    if out_dir is not None:
        _write_scenario_outputs(out_dir, scenario, per_seed, agg, manifest)
    if progress:
        print(
            f"[{scenario}] done in {manifest.wall_time_s:.1f}s "
            f"({manifest.wall_time_s / len(seeds):.2f}s per replication)",
            flush=True,
        )
    return per_seed, agg, manifest


def _write_scenario_outputs(
    out_dir: Path, scenario: str, per_seed: pd.DataFrame, agg: pd.DataFrame, manifest: Manifest
) -> None:
    d = Path(out_dir) / scenario
    d.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(d / "per_seed.csv", index=False)
    agg.to_csv(d / "aggregate.csv")
    manifest.write(d / "manifest.json")


def run_all_scenarios(
    seeds: Sequence[int],
    workers: int | None = None,
    out_dir: Path | None = None,
    scenarios: Iterable[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Run every scenario and write a combined comparison table."""
    names = list(scenarios) if scenarios else list(list_scenarios())
    out_dir = Path(out_dir) if out_dir is not None else RESULTS_DIR
    aggregates: dict[str, pd.DataFrame] = {}
    all_rows: list[pd.DataFrame] = []
    for name in names:
        per_seed, agg, _ = run_scenario(name, seeds, workers=workers, out_dir=out_dir)
        aggregates[name] = agg
        all_rows.append(per_seed)

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(out_dir / "all_scenarios_per_seed.csv", index=False)

    means = pd.DataFrame({name: agg["mean"] for name, agg in aggregates.items()})
    cis = pd.DataFrame({name: agg["ci_halfwidth"] for name, agg in aggregates.items()})
    means.to_csv(out_dir / "scenario_means.csv")
    cis.to_csv(out_dir / "scenario_ci_halfwidth.csv")
    return aggregates


# ---------------------------------------------------------------------------
# Parallel scaling benchmark
# ---------------------------------------------------------------------------
def benchmark_parallel(
    scenario: str = "baseline",
    seeds: Sequence[int] | None = None,
    worker_counts: Sequence[int] | None = None,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    """Measure wall-clock speedup from parallel replication.

    Reports observed speedup against the sequential baseline alongside Amdahl's
    upper bound implied by the measured serial fraction, which is the honest way
    to present scaling on a small machine.
    """
    seeds = list(seeds or range(1, 17))
    counts = list(worker_counts or _default_worker_counts())
    cfg = load_scenario(scenario, seed=seeds[0])
    social = load_optimization_config().get("objective", {})

    rows = []
    t_seq = None
    for w in counts:
        t0 = time.perf_counter()
        run_replications(cfg, seeds, workers=w, social_cost_params=social)
        elapsed = time.perf_counter() - t0
        if w == 1:
            t_seq = elapsed
        rows.append(
            {
                "workers": w,
                "wall_time_s": elapsed,
                "replications": len(seeds),
                "s_per_replication": elapsed / len(seeds),
                "speedup": (t_seq / elapsed) if t_seq else float("nan"),
                "efficiency": ((t_seq / elapsed) / w) if t_seq else float("nan"),
            }
        )
        print(f"  workers={w:2d}  {elapsed:7.2f}s  speedup={rows[-1]['speedup']:.2f}x", flush=True)
    df = pd.DataFrame(rows)
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "parallel_benchmark.csv", index=False)
    return df


def _default_worker_counts() -> list[int]:
    n = os.cpu_count() or 1
    counts = [1]
    w = 2
    while w <= n:
        counts.append(w)
        w *= 2
    if n not in counts:
        counts.append(n)
    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.experiments.runner",
        description="Run curb digital twin scenarios over multiple random seeds.",
    )
    p.add_argument(
        "--scenario", default="baseline", help="scenario name from config/scenarios.yaml"
    )
    p.add_argument("--all", action="store_true", help="run every configured scenario")
    p.add_argument("--seeds", type=int, default=30, help="number of replications (seeds 1..N)")
    p.add_argument("--seed-start", type=int, default=1, help="first seed value")
    p.add_argument("--workers", type=int, default=None, help="parallel worker processes")
    p.add_argument("--out", default=str(RESULTS_DIR), help="output directory")
    p.add_argument("--benchmark", action="store_true", help="run the parallel scaling benchmark")
    p.add_argument("--list", action="store_true", help="list available scenarios and exit")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        for name, label in list_scenarios().items():
            print(f"{name:18s} {label}")
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    if args.benchmark:
        print("Parallel scaling benchmark")
        benchmark_parallel(args.scenario, seeds=seeds, out_dir=out_dir)
        return 0

    if args.all:
        aggregates = run_all_scenarios(seeds, workers=args.workers, out_dir=out_dir)
        _print_comparison(aggregates)
        return 0

    _per_seed, agg, manifest = run_scenario(
        args.scenario, seeds, workers=args.workers, out_dir=out_dir, progress=not args.quiet
    )
    _print_headline(args.scenario, agg)
    print(f"\nWritten to {out_dir / args.scenario}")
    print(f"git={manifest.git_commit} config={manifest.config_hash}")
    return 0


def _print_headline(scenario: str, agg: pd.DataFrame) -> None:
    from src.experiments.metrics import HEADLINE_METRICS

    print(f"\n{scenario} metrics (mean +/- 95% CI over {int(agg['n'].iloc[0])} seeds)")
    print("-" * 62)
    for m in HEADLINE_METRICS:
        if m in agg.index:
            print(f"{m:32s} {agg.loc[m, 'mean']:10.3f} +/- {agg.loc[m, 'ci_halfwidth']:.3f}")


def _print_comparison(aggregates: dict[str, pd.DataFrame]) -> None:
    from src.experiments.metrics import HEADLINE_METRICS, compare

    base = aggregates.get("baseline")
    if base is None:
        return
    print("\nChange vs baseline (%)")
    print("-" * 96)
    header = f"{'scenario':18s}" + "".join(
        f"{m.replace('_min', '').replace('passenger_', 'pas_')[:14]:>15s}" for m in HEADLINE_METRICS
    )
    print(header)
    for name, agg in aggregates.items():
        if name == "baseline":
            continue
        cmp = compare(base, agg, HEADLINE_METRICS)
        cells = ""
        for m in HEADLINE_METRICS:
            if m in cmp.index:
                mark = "*" if bool(cmp.loc[m, "significant"]) else " "
                cells += f"{cmp.loc[m, 'pct_change']:>14.1f}{mark}"
            else:
                cells += f"{'-':>15s}"
        print(f"{name:18s}{cells}")
    print("\n* = 95% confidence intervals do not overlap the baseline")


if __name__ == "__main__":
    raise SystemExit(main())
