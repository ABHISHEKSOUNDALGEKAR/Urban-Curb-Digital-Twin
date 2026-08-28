"""Cross-modal sensitivity: how each mode's demand spills onto the others.

The policy question this project exists to answer is not "how bad is parking
search" but "what happens to freight and to waiting passengers when ridehail
volume grows". That is an elasticity question, and it is only answerable in a
model where the three modes share one capacity constraint.

For each vehicle class ``c`` and each outcome metric ``m``:

    elasticity(c -> m) = %change in m / %change in demand for c

estimated by a symmetric two-sided shock around the baseline, which removes the
first-order curvature that a one-sided difference would pick up. Every point
uses common random numbers, so the difference between the up-shock and the
down-shock is not contaminated by sampling noise.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import VEHICLE_CLASSES, load_demand_levels, load_optimization_config, load_scenario
from src.experiments.metrics import aggregate
from src.experiments.provenance import Manifest
from src.experiments.runner import run_replications

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

#: Outcomes reported in the elasticity table.
ELASTICITY_METRICS = (
    "passenger_search_time_min",
    "passenger_abandonment_rate",
    "delivery_delay_min",
    "delivery_illegal_rate",
    "ridehail_wait_min",
    "illegal_parking_rate",
    "vmt_miles",
    "system_social_cost_usd",
)


def _run(scenario: str, seeds: Sequence[int], overrides: dict | None, workers: int) -> pd.DataFrame:
    cfg = load_scenario(scenario, seed=int(seeds[0]), overrides=overrides)
    social = load_optimization_config().get("objective", {})
    rows = run_replications(cfg, seeds, workers=workers, social_cost_params=social)
    return aggregate(rows)


def cross_modal_elasticities(
    seeds: Sequence[int] = tuple(range(1, 11)),
    shock: float = 0.20,
    scenario: str = "baseline",
    workers: int = 1,
    verbose: bool = True,
) -> pd.DataFrame:
    """Two-sided elasticity of every outcome to every mode's demand."""
    base = _run(scenario, seeds, None, workers)
    rows: list[dict[str, Any]] = []
    for cls in VEHICLE_CLASSES:
        if verbose:
            print(f"  shocking {cls} demand by +/-{shock:.0%}", flush=True)
        up = _run(scenario, seeds, {"demand_scale": {cls: 1.0 + shock}}, workers)
        down = _run(scenario, seeds, {"demand_scale": {cls: 1.0 - shock}}, workers)
        for m in ELASTICITY_METRICS:
            if m not in base.index:
                continue
            b = float(base.loc[m, "mean"])
            if b == 0:
                continue
            d_metric = (float(up.loc[m, "mean"]) - float(down.loc[m, "mean"])) / b
            d_demand = 2.0 * shock
            rows.append(
                {
                    "shocked_class": cls,
                    "metric": m,
                    "baseline": b,
                    "value_up": float(up.loc[m, "mean"]),
                    "value_down": float(down.loc[m, "mean"]),
                    "elasticity": d_metric / d_demand,
                    "cross_modal": not m.startswith(cls),
                }
            )
    return pd.DataFrame(rows)


def demand_level_sweep(
    seeds: Sequence[int] = tuple(range(1, 11)),
    scenario: str = "baseline",
    workers: int = 1,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the configured low / normal / peak / extreme demand levels."""
    levels = load_demand_levels()
    rows = []
    for name, scale in levels.items():
        if verbose:
            print(f"  demand level: {name}", flush=True)
        agg = _run(scenario, seeds, {"demand_scale": scale}, workers)
        row = {"level": name, **{k: float(v) for k, v in scale.items()}}
        for m in ELASTICITY_METRICS + ("curb_occupancy_passenger", "curb_saturated_share"):
            if m in agg.index:
                row[m] = float(agg.loc[m, "mean"])
                row[f"{m}_ci"] = float(agg.loc[m, "ci_halfwidth"])
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m src.experiments.sensitivity",
        description="Cross-modal elasticities and demand-level sweep.",
    )
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--shock", type=float, default=0.20)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--scenario", default="baseline")
    p.add_argument("--out", default=str(RESULTS_DIR / "sensitivity"))
    p.add_argument("--skip-levels", action="store_true")
    args = p.parse_args(argv)

    seeds = list(range(1, args.seeds + 1))
    workers = args.workers if args.workers is not None else 0
    workers = workers or None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Cross-modal elasticities")
    el = cross_modal_elasticities(seeds, args.shock, args.scenario, workers or 1)
    el.to_csv(out / "cross_modal_elasticities.csv", index=False)

    print("\nElasticity table (%change in outcome per 1% change in demand)")
    pivot = el.pivot_table(index="shocked_class", columns="metric", values="elasticity")
    print(pivot.round(3).to_string())

    if not args.skip_levels:
        print("\nDemand level sweep")
        levels = demand_level_sweep(seeds, args.scenario, workers or 1)
        levels.to_csv(out / "demand_levels.csv", index=False)
        cols = [
            "level",
            "passenger_search_time_min",
            "delivery_delay_min",
            "ridehail_wait_min",
            "illegal_parking_rate",
            "curb_occupancy_passenger",
        ]
        print(levels[[c for c in cols if c in levels.columns]].round(3).to_string(index=False))

    Manifest.create(
        experiment="sensitivity",
        scenario=args.scenario,
        seeds=seeds,
        config={"shock": args.shock},
        n_workers=workers or 1,
    ).finish(0.0).write(out / "manifest.json")
    print(f"\nWritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
