"""Report generation: figures build from whatever results exist, and degrade gracefully."""

from __future__ import annotations

import pandas as pd
import pytest

from src.viz.report import (
    build_report,
    elasticity_figure,
    occupancy_figure,
    parallel_figure,
    pareto_figure,
    scenario_comparison_figure,
)


@pytest.fixture
def fake_results(tmp_path):
    metrics = [
        "passenger_search_time_min",
        "delivery_delay_min",
        "ridehail_wait_min",
        "illegal_parking_rate",
        "system_social_cost_usd",
    ]
    means = pd.DataFrame(
        {
            "baseline": [2.1, 2.4, 1.8, 0.05, 11000.0],
            "loading_zones": [2.4, 2.0, 1.8, 0.04, 10500.0],
        },
        index=metrics,
    )
    cis = means * 0.02
    means.to_csv(tmp_path / "scenario_means.csv")
    cis.to_csv(tmp_path / "scenario_ci_halfwidth.csv")
    pd.DataFrame(
        {
            "scenario": ["baseline", "baseline", "loading_zones", "loading_zones"],
            "curb_occupancy_passenger": [0.84, 0.85, 0.88, 0.87],
            "curb_occupancy_delivery": [0.42, 0.44, 0.30, 0.31],
            "curb_occupancy_ridehail": [0.24, 0.23, 0.25, 0.24],
        }
    ).to_csv(tmp_path / "all_scenarios_per_seed.csv", index=False)
    (tmp_path / "optimization").mkdir()
    pd.DataFrame(
        {
            "x_passenger": [0.8, 0.7, 0.6],
            "x_delivery": [0.1, 0.2, 0.3],
            "x_ridehail": [0.1, 0.1, 0.1],
            "passenger_search_time_min": [2.0, 2.4, 2.9],
            "delivery_delay_min": [2.8, 2.2, 1.9],
            "pareto_efficient": [True, True, False],
        }
    ).to_csv(tmp_path / "optimization" / "pareto_samples.csv", index=False)
    return tmp_path


class TestFigures:
    def test_scenario_comparison_builds(self, fake_results):
        means = pd.read_csv(fake_results / "scenario_means.csv", index_col=0)
        cis = pd.read_csv(fake_results / "scenario_ci_halfwidth.csv", index_col=0)
        fig = scenario_comparison_figure(means, cis, list(means.index))
        assert len(fig.data) == len(means.index)

    def test_occupancy_figure_has_one_trace_per_class(self, fake_results):
        df = pd.read_csv(fake_results / "all_scenarios_per_seed.csv")
        assert len(occupancy_figure(df).data) == 3

    def test_pareto_figure_separates_frontier(self, fake_results):
        df = pd.read_csv(fake_results / "optimization" / "pareto_samples.csv")
        fig = pareto_figure(df)
        assert {t.name for t in fig.data} == {"dominated", "Pareto frontier"}

    def test_elasticity_figure_builds(self):
        el = pd.DataFrame(
            {
                "shocked_class": ["passenger", "ridehail"],
                "metric": ["delivery_delay_min", "delivery_delay_min"],
                "elasticity": [0.3, 0.2],
            }
        )
        assert elasticity_figure(el).data

    def test_parallel_figure_builds(self):
        bench = pd.DataFrame({"workers": [1, 2, 4], "speedup": [1.0, 1.9, 3.5]})
        assert len(parallel_figure(bench).data) == 2


class TestReport:
    def test_report_is_self_contained_html(self, fake_results):
        out = build_report(fake_results)
        html = out.read_text()
        assert html.startswith("<!doctype html>")
        assert "Urban Curb Digital Twin" in html
        assert "plotly" in html.lower()

    def test_report_states_the_data_are_synthetic(self, fake_results):
        html = build_report(fake_results).read_text()
        assert "Synthetic model" in html

    def test_missing_results_do_not_crash(self, tmp_path):
        out = build_report(tmp_path)
        assert out.exists()
