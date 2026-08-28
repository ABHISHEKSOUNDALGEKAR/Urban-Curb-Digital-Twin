"""Static HTML report: every headline result in one self-contained file.

Generated from the CSVs the experiment pipeline writes, so it can be rebuilt
without re-running anything, and opened with no server and no dependencies
beyond a browser.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

# A restrained, colour-blind-safe palette; one hue per vehicle class, used
# consistently across every figure so a colour always means the same thing.
CLASS_COLOURS = {
    "passenger": "#3b6ea5",
    "delivery": "#d1793a",
    "ridehail": "#3f8f6b",
}
ACCENT = "#8a3a5c"
GRID = "#e4e2dd"
INK = "#25262b"

pio.templates["curb"] = go.layout.Template(
    layout=dict(
        font=dict(family="Inter, Helvetica, Arial, sans-serif", size=13, color=INK),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
        margin=dict(l=60, r=30, t=50, b=50),
        colorway=list(CLASS_COLOURS.values()) + [ACCENT],
    )
)
pio.templates.default = "curb"


def _fig_html(fig: go.Figure, include_js: bool = False) -> str:
    return fig.to_html(full_html=False, include_plotlyjs="cdn" if include_js else False)


def scenario_comparison_figure(
    means: pd.DataFrame, cis: pd.DataFrame, metrics: Sequence[str]
) -> go.Figure:
    """Grouped bars with 95% CI error bars, one panel per headline metric."""
    metrics = [m for m in metrics if m in means.index]
    fig = make_subplots(
        rows=1,
        cols=len(metrics),
        subplot_titles=[m.replace("_", " ") for m in metrics],
        horizontal_spacing=0.06,
    )
    scenarios = list(means.columns)
    for i, m in enumerate(metrics, start=1):
        fig.add_trace(
            go.Bar(
                x=scenarios,
                y=means.loc[m].to_numpy(dtype=float),
                error_y=dict(
                    type="data", array=cis.loc[m].to_numpy(dtype=float), color=INK, thickness=1
                ),
                marker_color=[
                    ACCENT if s == "baseline" else CLASS_COLOURS["passenger"] for s in scenarios
                ],
                showlegend=False,
                hovertemplate="%{x}<br>%{y:.3f}<extra></extra>",
            ),
            row=1,
            col=i,
        )
    fig.update_layout(height=340, title="Scenario comparison (mean +/- 95% CI over seeds)")
    fig.update_xaxes(tickangle=-35)
    return fig


def occupancy_figure(per_seed: pd.DataFrame) -> go.Figure:
    """Mean curb occupancy by regulation type, per scenario."""
    fig = go.Figure()
    cols = {c: f"curb_occupancy_{c}" for c in CLASS_COLOURS}
    grouped = per_seed.groupby("scenario")
    for cls, col in cols.items():
        if col not in per_seed.columns:
            continue
        fig.add_trace(
            go.Bar(
                name=cls,
                x=list(grouped.groups),
                y=[float(grouped.get_group(s)[col].mean()) for s in grouped.groups],
                marker_color=CLASS_COLOURS[cls],
            )
        )
    fig.add_hline(
        y=0.85,
        line_dash="dot",
        line_color=ACCENT,
        annotation_text="0.85 target occupancy",
        annotation_position="top left",
    )
    fig.update_layout(
        barmode="group",
        height=360,
        title="Curb occupancy by regulation type",
        yaxis_title="occupancy",
    )
    fig.update_xaxes(tickangle=-35)
    return fig


def pareto_figure(pareto: pd.DataFrame) -> go.Figure:
    """Passenger search time against delivery delay, non-dominated points marked."""
    dom = pareto.loc[~pareto["pareto_efficient"]]
    eff = pareto.loc[pareto["pareto_efficient"]].sort_values("passenger_search_time_min")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dom["passenger_search_time_min"],
            y=dom["delivery_delay_min"],
            mode="markers",
            name="dominated",
            marker=dict(size=7, color="#c8c6c0", line=dict(width=0)),
            customdata=dom[["x_passenger", "x_delivery", "x_ridehail"]].to_numpy(),
            hovertemplate="search %{x:.2f} min<br>delay %{y:.2f} min<br>alloc %{customdata[0]:.2f}/%{customdata[1]:.2f}/%{customdata[2]:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=eff["passenger_search_time_min"],
            y=eff["delivery_delay_min"],
            mode="markers+lines",
            name="Pareto frontier",
            marker=dict(size=11, color=ACCENT, line=dict(width=1, color="white")),
            line=dict(color=ACCENT, width=1.5, dash="dot"),
            customdata=eff[["x_passenger", "x_delivery", "x_ridehail"]].to_numpy(),
            hovertemplate="search %{x:.2f} min<br>delay %{y:.2f} min<br>alloc %{customdata[0]:.2f}/%{customdata[1]:.2f}/%{customdata[2]:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=430,
        title="No single best allocation: passenger search vs. freight delay",
        xaxis_title="passenger search time (min)",
        yaxis_title="delivery delay (min)",
    )
    return fig


def elasticity_figure(elasticities: pd.DataFrame) -> go.Figure:
    """Cross-modal elasticity heatmap: demand shock (rows) x outcome (cols)."""
    pivot = elasticities.pivot_table(index="shocked_class", columns="metric", values="elasticity")
    fig = go.Figure(
        go.Heatmap(
            z=pivot.to_numpy(),
            x=[c.replace("_", " ") for c in pivot.columns],
            y=list(pivot.index),
            colorscale=[[0, "#3b6ea5"], [0.5, "#f4f2ee"], [1, "#b5432f"]],
            zmid=0,
            hovertemplate="%{y} demand +1%% -> %{x} %{z:+.2f}%%<extra></extra>",
            colorbar=dict(title="elasticity"),
        )
    )
    fig.update_layout(height=330, title="Cross-modal curb demand elasticities")
    fig.update_xaxes(tickangle=-30)
    return fig


def parallel_figure(bench: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=bench["workers"],
            y=bench["speedup"],
            mode="markers+lines",
            name="observed",
            marker=dict(size=10, color=ACCENT),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bench["workers"],
            y=bench["workers"],
            mode="lines",
            name="linear",
            line=dict(dash="dot", color="#9a9892"),
        )
    )
    fig.update_layout(
        height=330,
        title="Parallel replication speedup",
        xaxis_title="worker processes",
        yaxis_title="speedup vs. sequential",
    )
    return fig


def build_report(results_dir: Path | str = RESULTS_DIR, out_file: Path | str | None = None) -> Path:
    """Assemble every available result into one HTML file."""
    rd = Path(results_dir)
    out = Path(out_file) if out_file else rd / "report.html"
    parts: list[str] = []
    first = True

    def add(fig: go.Figure) -> None:
        nonlocal first
        parts.append(_fig_html(fig, include_js=first))
        first = False

    means_p, cis_p = rd / "scenario_means.csv", rd / "scenario_ci_halfwidth.csv"
    if means_p.exists() and cis_p.exists():
        means = pd.read_csv(means_p, index_col=0)
        cis = pd.read_csv(cis_p, index_col=0)
        add(
            scenario_comparison_figure(
                means,
                cis,
                [
                    "passenger_search_time_min",
                    "delivery_delay_min",
                    "ridehail_wait_min",
                    "illegal_parking_rate",
                    "system_social_cost_usd",
                ],
            )
        )
    per_seed_p = rd / "all_scenarios_per_seed.csv"
    if per_seed_p.exists():
        add(occupancy_figure(pd.read_csv(per_seed_p)))
    pareto_p = rd / "optimization" / "pareto_samples.csv"
    if pareto_p.exists():
        add(pareto_figure(pd.read_csv(pareto_p)))
    elast_p = rd / "sensitivity" / "cross_modal_elasticities.csv"
    if elast_p.exists():
        add(elasticity_figure(pd.read_csv(elast_p)))
    bench_p = rd / "parallel_benchmark.csv"
    if bench_p.exists():
        add(parallel_figure(pd.read_csv(bench_p)))

    summary_p = rd / "optimization" / "optimization_summary.json"
    summary_html = ""
    if summary_p.exists():
        s = json.loads(summary_p.read_text())
        cur = s["confirmed"]["current"]
        best = s["confirmed"][s["best_method"]]
        rows = "".join(
            f"<tr><td>{c}</td><td>{cur['allocation'][c] * 100:.1f}%</td>"
            f"<td>{best['allocation'][c] * 100:.1f}%</td></tr>"
            for c in ("passenger", "delivery", "ridehail")
        )
        imp = "".join(
            f"<tr><td>{k.replace('_', ' ')}</td><td>{v:+.2f}%</td></tr>"
            for k, v in s["improvement_vs_current_pct"].items()
        )
        summary_html = (
            "<h2>Optimized curb allocation</h2>"
            f"<table><tr><th>class</th><th>current</th><th>optimized</th></tr>{rows}</table>"
            f"<h3>Change vs current</h3><table><tr><th>metric</th><th>change</th></tr>{imp}</table>"
        )

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Urban Curb Digital Twin - results</title>
<style>
 body {{ font-family: Inter, Helvetica, Arial, sans-serif; color:{INK};
        max-width: 1180px; margin: 0 auto; padding: 32px 20px 80px; line-height:1.55; }}
 h1 {{ font-size: 1.7rem; margin-bottom: .2rem; }}
 h2 {{ font-size: 1.15rem; margin-top: 2.2rem; }}
 .sub {{ color:#6b6a66; margin-top:0; }}
 table {{ border-collapse: collapse; margin: 10px 0 20px; }}
 th, td {{ border-bottom: 1px solid {GRID}; padding: 6px 18px 6px 0; text-align: left; }}
 .note {{ background:#faf8f4; border-left:3px solid {ACCENT}; padding:10px 16px; margin:22px 0; }}
</style></head><body>
<h1>Urban Curb Digital Twin</h1>
<p class="sub">Multi-agent simulation of competition for curb space between passenger,
freight and ridehail vehicles, with calibration and allocation optimization.</p>
<div class="note"><strong>Synthetic model.</strong> The network, demand and
"observed" occupancy are generated, not empirical. Numbers describe the model's
behaviour, not any real district.</div>
{"".join(parts)}
{summary_html}
</body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.viz.report")
    p.add_argument("--results", default=str(RESULTS_DIR))
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    path = build_report(args.results, args.out)
    print(f"Report written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
