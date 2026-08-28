"""Streamlit dashboard for exploring scenarios and the optimized allocation.

Run with::

    streamlit run src/viz/dashboard.py

Reads the CSVs written by the experiment pipeline, so it never runs a simulation
itself: the dashboard is a view over committed results, not a second source of
truth. If results are missing it says so instead of inventing them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.viz.report import (
    elasticity_figure,
    occupancy_figure,
    pareto_figure,
    scenario_comparison_figure,
)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
HEADLINE = [
    "passenger_search_time_min",
    "delivery_delay_min",
    "ridehail_wait_min",
    "illegal_parking_rate",
    "system_social_cost_usd",
]


@st.cache_data
def load(results_dir: str):
    rd = Path(results_dir)
    out = {}
    for key, rel in {
        "means": "scenario_means.csv",
        "cis": "scenario_ci_halfwidth.csv",
        "per_seed": "all_scenarios_per_seed.csv",
        "pareto": "optimization/pareto_samples.csv",
        "elasticities": "sensitivity/cross_modal_elasticities.csv",
        "benchmark": "parallel_benchmark.csv",
    }.items():
        p = rd / rel
        if p.exists():
            out[key] = pd.read_csv(p, index_col=0 if key in ("means", "cis") else None)
    p = rd / "optimization" / "optimization_summary.json"
    if p.exists():
        out["optimization"] = json.loads(p.read_text())
    return out


def main() -> None:
    st.set_page_config(page_title="Urban Curb Digital Twin", layout="wide")
    st.title("Urban Curb Digital Twin")
    st.caption(
        "Passenger, freight and ridehail vehicles competing for curb space in a "
        "synthetic downtown district. All figures are model output, not empirical data."
    )

    results_dir = st.sidebar.text_input("Results directory", str(RESULTS_DIR))
    data = load(results_dir)
    if not data:
        st.warning(
            "No results found. Generate them with:\n\n"
            "`python -m src.experiments.runner --all --seeds 30`"
        )
        return

    means = data.get("means")
    if means is not None:
        scenarios = list(means.columns)
        scenario = st.sidebar.selectbox(
            "Scenario",
            scenarios,
            index=scenarios.index("baseline") if "baseline" in scenarios else 0,
        )
        cis = data["cis"]

        st.subheader(f"Headline metrics - {scenario}")
        cols = st.columns(len(HEADLINE))
        for col, metric in zip(cols, HEADLINE, strict=True):
            if metric not in means.index:
                continue
            value = float(means.loc[metric, scenario])
            ci = float(cis.loc[metric, scenario])
            delta = None
            if "baseline" in means.columns and scenario != "baseline":
                base = float(means.loc[metric, "baseline"])
                delta = f"{(value - base) / base * 100:+.1f}% vs baseline" if base else None
            fmt = f"{value:,.3f}" if abs(value) < 100 else f"{value:,.0f}"
            col.metric(metric.replace("_", " "), fmt, delta, delta_color="inverse")
            col.caption(f"+/- {ci:,.3f} (95% CI)")

        st.plotly_chart(scenario_comparison_figure(means, cis, HEADLINE), use_container_width=True)

    if "per_seed" in data:
        st.subheader("Curb occupancy")
        st.plotly_chart(occupancy_figure(data["per_seed"]), use_container_width=True)

    if "optimization" in data:
        st.subheader("Optimized curb allocation")
        s = data["optimization"]
        cur, best = s["confirmed"]["current"], s["confirmed"][s["best_method"]]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Current**")
            st.dataframe(
                pd.DataFrame(
                    {"share": {k: f"{v * 100:.1f}%" for k, v in cur["allocation"].items()}}
                )
            )
        with c2:
            st.markdown(f"**Optimized** ({s['best_method'].replace('opt_', '')})")
            st.dataframe(
                pd.DataFrame(
                    {"share": {k: f"{v * 100:.1f}%" for k, v in best["allocation"].items()}}
                )
            )
        st.dataframe(
            pd.DataFrame(
                {
                    "change vs current (%)": {
                        k: round(v, 2) for k, v in s["improvement_vs_current_pct"].items()
                    }
                }
            )
        )

    if "pareto" in data:
        st.subheader("Multi-objective tradeoff")
        st.plotly_chart(pareto_figure(data["pareto"]), use_container_width=True)
        st.caption(
            "Each point is one curb allocation. Optimising for a single mode moves "
            "along the frontier at another mode's expense."
        )

    if "elasticities" in data:
        st.subheader("Cross-modal elasticities")
        st.plotly_chart(elasticity_figure(data["elasticities"]), use_container_width=True)

    with st.sidebar:
        st.markdown("---")
        st.markdown(
            "**Reproduce**\n\n"
            "```\npython -m src.experiments.runner --all --seeds 30\n"
            "python -m src.experiments.sensitivity --seeds 10\n"
            "python -m src.optimization.optimizer\n```"
        )


if __name__ == "__main__":
    main()
