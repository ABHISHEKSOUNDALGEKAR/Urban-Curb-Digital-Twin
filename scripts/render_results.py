#!/usr/bin/env python3
"""Regenerate the results section of the README from the files in ``results/``.

Numbers in a README rot the moment someone re-runs an experiment and forgets to
update the prose. This script removes that failure mode: everything between the
``<!-- BEGIN:RESULTS -->`` and ``<!-- END:RESULTS -->`` markers is generated from
the CSV and JSON the pipeline writes, together with the git commit those files
were produced at.

Usage::

    python scripts/render_results.py            # rewrite README.md in place
    python scripts/render_results.py --check    # fail if the README is stale
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BEGIN = "<!-- BEGIN:RESULTS -->"
END = "<!-- END:RESULTS -->"

HEADLINE = [
    ("passenger_search_time_min", "Passenger search time (min)", "{:.2f}"),
    ("passenger_walk_distance_m", "Passenger walk (m, round trip)", "{:.0f}"),
    ("passenger_abandonment_rate", "Passenger trips abandoned", "{:.1%}"),
    ("delivery_delay_min", "Delivery delay (min)", "{:.2f}"),
    ("delivery_illegal_rate", "Delivery double-parking", "{:.1%}"),
    ("ridehail_wait_min", "Ridehail passenger wait (min)", "{:.2f}"),
    ("illegal_parking_rate", "Illegal parking rate (all classes)", "{:.1%}"),
    ("curb_occupancy_passenger", "Metered occupancy", "{:.1%}"),
    ("vmt_miles", "Vehicle miles travelled", "{:,.0f}"),
    ("cruising_vmt_share", "of which cruising for curb", "{:.1%}"),
    ("meter_revenue_usd", "Meter revenue ($)", "{:,.0f}"),
    ("system_social_cost_usd", "System social cost ($)", "{:,.0f}"),
]


def _fmt(value: float, fmt: str) -> str:
    try:
        return fmt.format(value)
    except (ValueError, TypeError):
        return "-"


def baseline_table(results: Path) -> str:
    agg = pd.read_csv(results / "baseline" / "aggregate.csv", index_col=0)
    n = int(agg["n"].iloc[0])
    lines = [
        f"Baseline, mean over {n} independent seeds, 95% confidence interval half-width in brackets.",
        "",
        "| metric | value | 95% CI |",
        "|---|---:|---:|",
    ]
    for key, label, fmt in HEADLINE:
        if key not in agg.index:
            continue
        mean = float(agg.loc[key, "mean"])
        ci = float(agg.loc[key, "ci_halfwidth"])
        lines.append(f"| {label} | {_fmt(mean, fmt)} | ±{_fmt(ci, fmt)} |")
    return "\n".join(lines)


def scenario_table(results: Path) -> str:
    means = pd.read_csv(results / "scenario_means.csv", index_col=0)
    cis = pd.read_csv(results / "scenario_ci_halfwidth.csv", index_col=0)
    metrics = [
        ("passenger_search_time_min", "pass. search (min)", "{:.2f}"),
        ("delivery_delay_min", "delivery delay (min)", "{:.2f}"),
        ("ridehail_wait_min", "ridehail wait (min)", "{:.2f}"),
        ("illegal_parking_rate", "illegal parking", "{:.1%}"),
        ("vmt_miles", "VMT", "{:,.0f}"),
        ("system_social_cost_usd", "social cost ($)", "{:,.0f}"),
    ]
    from src.config import list_scenarios

    labels = list_scenarios()
    header = "| scenario | " + " | ".join(m[1] for m in metrics) + " |"
    sep = "|---|" + "---:|" * len(metrics)
    lines = [header, sep]
    for scenario in means.columns:
        cells = []
        for key, _label, fmt in metrics:
            if key not in means.index:
                cells.append("-")
                continue
            value = float(means.loc[key, scenario])
            cell = _fmt(value, fmt)
            if scenario != "baseline" and "baseline" in means.columns:
                base = float(means.loc[key, "baseline"])
                if base:
                    pct = (value - base) / base * 100
                    star = _significant(means, cis, key, scenario)
                    cell += f" ({pct:+.1f}%{star})"
            cells.append(cell)
        lines.append(f"| **{labels.get(scenario, scenario)}** | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "`*` marks a change whose 95% confidence interval does not overlap the baseline's."
    )
    return "\n".join(lines)


def _significant(means, cis, key, scenario) -> str:
    try:
        m_s, c_s = float(means.loc[key, scenario]), float(cis.loc[key, scenario])
        m_b, c_b = float(means.loc[key, "baseline"]), float(cis.loc[key, "baseline"])
    except KeyError:
        return ""
    overlap = (m_s - c_s) <= (m_b + c_b) and (m_b - c_b) <= (m_s + c_s)
    return "" if overlap else "*"


def optimization_section(results: Path) -> str:
    path = results / "optimization" / "optimization_summary.json"
    if not path.exists():
        return "_Not yet run._"
    s = json.loads(path.read_text())
    cur = s["confirmed"]["current"]
    best = s["confirmed"][s["best_method"]]
    lines = [
        "| curb allocation | passenger | delivery | ridehail |",
        "|---|---:|---:|---:|",
    ]
    for label, ev in (
        ("Current (posted)", cur),
        (f"Optimized ({s['best_method'].replace('opt_', '')})", best),
    ):
        lines.append(
            f"| {label} | {ev['allocation']['passenger']:.1%} | "
            f"{ev['allocation']['delivery']:.1%} | {ev['allocation']['ridehail']:.1%} |"
        )
    lines += [
        "",
        "| outcome | current | optimized | change |",
        "|---|---:|---:|---:|",
    ]
    rows = [
        ("objective (weighted social cost, $)", "objective", "{:,.0f}"),
        ("passenger search time (min)", "passenger_search_time_min", "{:.2f}"),
        ("delivery delay (min)", "delivery_delay_min", "{:.2f}"),
        ("ridehail wait (min)", "ridehail_wait_min", "{:.2f}"),
        ("illegal parking rate", "illegal_parking_rate", "{:.1%}"),
        ("VMT", "vmt_miles", "{:,.0f}"),
        ("meter + fine revenue ($)", "revenue_usd", "{:,.0f}"),
    ]
    for label, key, fmt in rows:
        a = cur["objective"] if key == "objective" else cur["metrics"].get(key)
        b = best["objective"] if key == "objective" else best["metrics"].get(key)
        if a in (None, 0) or b is None:
            continue
        lines.append(f"| {label} | {_fmt(a, fmt)} | {_fmt(b, fmt)} | {(b - a) / a * 100:+.1f}% |")
    sig = "yes" if s.get("significant") else "**no — within Monte-Carlo noise**"
    lines += [
        "",
        f"Objective evaluations: {s['total_objective_evaluations']} "
        f"({s['total_simulations']} simulation replications, {s['seeds_per_evaluation']} per evaluation). "
        f"Wall clock {s['wall_time_s']:.0f}s.",
        "",
        f"Difference from the current allocation distinguishable from noise: {sig}.",
        "",
        "Search-method comparison, at the budget each was given:",
        "",
        "| method | evaluations | replications | wall (s) | best objective found |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, r in s.get("search", {}).items():
        conf = s["confirmed"].get(f"opt_{name}", {})
        lines.append(
            f"| {name} | {r['n_evaluations']} | {r['n_simulations']} | "
            f"{r['wall_time_s']:.0f} | {conf.get('objective', float('nan')):,.0f} |"
        )
    lines.append("")
    lines.append(
        "The *best objective found* column is the re-evaluation on 20 independent "
        "confirmation seeds, not the value the search itself reported: a search's own "
        "estimate of its winner is biased low by selection."
    )
    return "\n".join(lines)


def calibration_section(results: Path) -> str:
    path = results / "calibration" / "calibration_report.json"
    if not path.exists():
        return "_Not yet run._"
    r = json.loads(path.read_text())
    lines = [
        "| | occupancy RMSE | occupancy MAE | bias | corr |",
        "|---|---:|---:|---:|---:|",
        f"| Uncalibrated (config defaults), fitting window | {r['baseline_error']['calibration_rmse']:.4f} | - | - | - |",
        f"| Calibrated, fitting window | {r['calibration_error']['rmse']:.4f} | {r['calibration_error']['mae']:.4f} | {r['calibration_error']['bias']:+.4f} | {r['calibration_error']['corr']:.3f} |",
        f"| Uncalibrated, **held-out** window | {r['baseline_error']['validation_rmse']:.4f} | - | - | - |",
        f"| Calibrated, **held-out** window | {r['validation_error']['rmse']:.4f} | {r['validation_error']['mae']:.4f} | {r['validation_error']['bias']:+.4f} | {r['validation_error']['corr']:.3f} |",
    ]
    oracle = r.get("oracle_error") or {}
    if oracle:
        lines += [
            f"| *Oracle* (the true generating parameters), fitting window | {oracle['calibration_rmse']:.4f} | - | - | - |",
            f"| *Oracle*, held-out window | {oracle['validation_rmse']:.4f} | - | - | - |",
        ]
    lines += [
        "",
        f"Measurement noise on the synthetic sensors has SD {oracle.get('measurement_noise_sd', 0.035):.3f}, "
        "which is the floor below which any in-sample improvement is the estimator fitting noise.",
        "",
        f"Fitted on {r['calibration_error']['n_segments']} instrumented curb segments over "
        f"{r['n_evaluations']} objective evaluations ({r['wall_time_s']:.0f}s).",
        "",
        "Parameter recovery — the synthetic observations were generated at a hidden "
        "parameter vector, so the estimate can be scored directly:",
        "",
        "| parameter | true | estimated | error |",
        "|---|---:|---:|---:|",
    ]
    for name, true_value in r.get("ground_truth", {}).items():
        est = r["theta"][name]
        lines.append(f"| {name} | {true_value:.4g} | {est:.4g} | {est - true_value:+.4g} |")
    heldout = r.get("heldout_behavioural", {})
    if "observed_illegal_parking_rate" in heldout:
        lines += [
            "",
            "Behavioural hold-out — the illegal-parking rate was never part of the "
            "objective, and stands in for validating against citation records:",
            "",
            "| series | observed | simulated | abs. error |",
            "|---|---:|---:|---:|",
        ]
        for key in ("illegal_parking_rate", "delivery_illegal_rate", "passenger_search_time_min"):
            o, s_ = heldout.get(f"observed_{key}"), heldout.get(f"sim_{key}")
            if o is None or s_ is None:
                continue
            lines.append(f"| {key} | {o:.4f} | {s_:.4f} | {abs(s_ - o):.4f} |")
    return "\n".join(lines)


def elasticity_section(results: Path) -> str:
    path = results / "sensitivity" / "cross_modal_elasticities.csv"
    if not path.exists():
        return "_Not yet run._"
    el = pd.read_csv(path)
    pivot = el.pivot_table(index="shocked_class", columns="metric", values="elasticity")
    cols = [
        ("passenger_search_time_min", "pass. search"),
        ("delivery_delay_min", "delivery delay"),
        ("ridehail_wait_min", "ridehail wait"),
        ("illegal_parking_rate", "illegal parking"),
        ("vmt_miles", "VMT"),
        ("system_social_cost_usd", "social cost"),
    ]
    cols = [c for c in cols if c[0] in pivot.columns]
    lines = [
        "Elasticity: percent change in the outcome per 1% change in that mode's demand, "
        "estimated with a symmetric ±20% shock and common random numbers.",
        "",
        "| demand shocked ↓ / outcome → | " + " | ".join(c[1] for c in cols) + " |",
        "|---|" + "---:|" * len(cols),
    ]
    for cls in pivot.index:
        cells = [f"{pivot.loc[cls, k]:+.3f}" for k, _ in cols]
        lines.append(f"| **{cls}** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def demand_levels_section(results: Path) -> str:
    path = results / "sensitivity" / "demand_levels.csv"
    if not path.exists():
        return ""
    df = pd.read_csv(path)
    cols = [
        ("passenger_search_time_min", "pass. search (min)", "{:.2f}"),
        ("delivery_delay_min", "delivery delay (min)", "{:.2f}"),
        ("ridehail_wait_min", "ridehail wait (min)", "{:.2f}"),
        ("illegal_parking_rate", "illegal parking", "{:.1%}"),
        ("curb_occupancy_passenger", "metered occupancy", "{:.1%}"),
    ]
    cols = [c for c in cols if c[0] in df.columns]
    lines = [
        "",
        "Demand levels (arrival rates scaled together):",
        "",
        "| level | " + " | ".join(c[1] for c in cols) + " |",
        "|---|" + "---:|" * len(cols),
    ]
    for _, row in df.iterrows():
        cells = [_fmt(row[k], fmt) for k, _lab, fmt in cols]
        lines.append(f"| {row['level']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def parallel_section(results: Path) -> str:
    path = results / "parallel_benchmark.csv"
    if not path.exists():
        return "_Not yet run._"
    df = pd.read_csv(path)
    lines = [
        "| workers | wall (s) | s / replication | speedup | efficiency |",
        "|---:|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {int(r['workers'])} | {r['wall_time_s']:.1f} | {r['s_per_replication']:.2f} | "
            f"{r['speedup']:.2f}× | {r['efficiency']:.0%} |"
        )
    n = int(df["replications"].iloc[0])
    lines.append("")
    lines.append(f"{n} replications of the baseline scenario per row.")
    return "\n".join(lines)


def sumo_section(results: Path) -> str:
    files = (
        sorted((results / "sumo").glob("sumo_summary_seed*.json"))
        if (results / "sumo").exists()
        else []
    )
    if not files:
        return "_Not yet run._"
    s = json.loads(files[0].read_text())
    d = s.get("ridehail_dispatch", {})
    lines = [
        "| quantity | SUMO backend |",
        "|---|---:|",
        f"| vehicles simulated | {s.get('n_vehicles', 0):,} |",
        f"| passenger parked rate | {s.get('passenger_parked_rate', 0):.1%} |",
        f"| passenger search time (min) | {s.get('passenger_search_time_min', 0):.2f} |",
        f"| delivery search time (min) | {s.get('delivery_search_time_min', 0):.2f} |",
        f"| illegal parking rate | {s.get('illegal_parking_rate', 0):.1%} |",
        f"| taxi reservations dispatched | {d.get('dispatched', 0):,} |",
        f"| pickups completed | {d.get('completed_pickups', 0):,} |",
        f"| ridehail wait (min) | {d.get('mean_wait_min', 0):.2f} |",
        f"| wall clock (s) | {s.get('wall_time_s', 0):.0f} |",
    ]
    return "\n".join(lines)


def provenance_line(results: Path) -> str:
    m = results / "baseline" / "manifest.json"
    if not m.exists():
        return ""
    d = json.loads(m.read_text())
    dirty = " (working tree dirty)" if d.get("git_dirty") else ""
    return (
        f"Produced at commit `{d['git_commit']}`{dirty} on Python {d['python_version']}, "
        f"{d['cpu_count']} CPUs, {d['platform']}, finished {d['finished_utc']}."
    )


def render(results: Path) -> str:
    parts = [
        BEGIN,
        "",
        "### Baseline",
        "",
        baseline_table(results),
        "",
        "### Policy scenarios",
        "",
        scenario_table(results),
        "",
        "### Optimized curb allocation",
        "",
        optimization_section(results),
        "",
        "### Calibration and held-out validation",
        "",
        calibration_section(results),
        "",
        "### Cross-modal elasticities",
        "",
        elasticity_section(results),
        demand_levels_section(results),
        "",
        "### Parallel scaling",
        "",
        parallel_section(results),
        "",
        "### SUMO backend cross-check",
        "",
        sumo_section(results),
        "",
        provenance_line(results),
        "",
        END,
    ]
    return "\n".join(parts)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default=str(ROOT / "results"))
    p.add_argument("--readme", default=str(ROOT / "README.md"))
    p.add_argument("--check", action="store_true", help="exit non-zero if the README is stale")
    args = p.parse_args()

    readme = Path(args.readme)
    text = readme.read_text()
    if BEGIN not in text or END not in text:
        print(f"markers {BEGIN} / {END} not found in {readme}", file=sys.stderr)
        return 2

    block = render(Path(args.results))
    head, rest = text.split(BEGIN, 1)
    _old, tail = rest.split(END, 1)
    updated = head + block + tail

    if args.check:
        if updated != text:
            print("README results section is stale; run scripts/render_results.py", file=sys.stderr)
            return 1
        print("README results section is up to date")
        return 0

    readme.write_text(updated)
    print(f"Updated results section in {readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
