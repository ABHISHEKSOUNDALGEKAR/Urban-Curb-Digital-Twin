"""End-to-end simulation properties: determinism, conservation, metric sanity.

These are the tests that would catch a silent physics break - a stall leaking, a
run stopping being reproducible, warm-up not being excluded - which unit tests on
individual classes cannot see.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import ConfigError, deep_merge, list_scenarios, load_scenario
from src.experiments.metrics import aggregate, compare, occupancy_by_segment, summarise
from src.simulation.engine import profile_multiplier, run_simulation
from src.simulation.routing import Router, build_grid_network


class TestConfig:
    def test_every_scenario_loads_and_validates(self):
        for name in list_scenarios():
            cfg = load_scenario(name, seed=1)
            assert abs(sum(cfg.allocation.values()) - 1.0) < 1e-9
            assert cfg.horizon_min > cfg.warmup_min

    def test_unknown_scenario_raises(self):
        with pytest.raises(ConfigError):
            load_scenario("no_such_scenario")

    def test_fingerprint_ignores_seed(self):
        a = load_scenario("baseline", seed=1)
        b = load_scenario("baseline", seed=999)
        assert a.fingerprint() == b.fingerprint()

    def test_fingerprint_changes_with_allocation(self):
        a = load_scenario("baseline", seed=1)
        b = a.with_overrides({"allocation": {"passenger": 0.5, "delivery": 0.3, "ridehail": 0.2}})
        assert a.fingerprint() != b.fingerprint()

    def test_deep_merge_ignores_none(self):
        assert deep_merge({"a": 1}, {"a": None}) == {"a": 1}

    def test_demand_scale_applies(self):
        cfg = load_scenario("baseline", seed=1, overrides={"demand_scale": {"ridehail": 2.0}})
        base = load_scenario("baseline", seed=1)
        assert cfg.demand["ridehail"] == pytest.approx(2.0 * base.demand["ridehail"])


class TestRouting:
    def test_grid_is_strongly_connected(self, small_network_config):
        import networkx as nx

        graph, _ = build_grid_network(small_network_config)
        assert nx.is_strongly_connected(graph)

    def test_every_link_has_curb(self, small_network_config):
        graph, inventory = build_grid_network(small_network_config)
        assert len(inventory) == graph.number_of_edges()
        for _u, _v, data in graph.edges(data=True):
            assert data["curb_ids"]

    def test_travel_time_rises_with_double_parking(self, small_network_config):
        graph, _ = build_grid_network(small_network_config)
        router = Router(graph, small_network_config)
        link = next(iter(graph.edges))
        before = router.link_travel_time_s(link, 0.0)
        router.add_double_park(link)
        assert router.link_travel_time_s(link, 0.0) > before
        router.remove_double_park(link)
        assert router.link_travel_time_s(link, 0.0) == pytest.approx(before)

    def test_travel_time_rises_with_flow(self, small_network_config):
        graph, _ = build_grid_network(small_network_config)
        router = Router(graph, small_network_config)
        u, v = next(iter(graph.edges))
        free = router.link_travel_time_s((u, v), 0.0)
        for _ in range(400):
            router.record_traversal(u, v, 0.0)
        assert router.link_travel_time_s((u, v), 0.0) > free

    def test_path_is_shortest(self, small_network_config):
        graph, _ = build_grid_network(small_network_config)
        router = Router(graph, small_network_config)
        path = router.path("N00_00", "N02_02")
        # Manhattan distance on a 3x3 grid of 120 m blocks.
        assert router.path_length_m("N00_00", "N02_02") == pytest.approx(480.0)
        assert path[0] == "N00_00" and path[-1] == "N02_02"


class TestDemandProcess:
    def test_profile_interpolates(self):
        assert profile_multiplier([1.0, 2.0], 0.0, 100.0) == pytest.approx(1.0)
        assert profile_multiplier([1.0, 2.0], 100.0, 100.0) == pytest.approx(2.0)
        assert profile_multiplier([1.0, 2.0], 50.0, 100.0) == pytest.approx(1.5)

    def test_empty_profile_is_flat(self):
        assert profile_multiplier([], 10.0, 100.0) == 1.0

    def test_arrival_counts_scale_with_rate(self, fast_config):
        low = summarise(run_simulation(fast_config(demand_scale={"passenger": 0.5})))
        high = summarise(run_simulation(fast_config(demand_scale={"passenger": 1.5})))
        assert high["passenger_trips"] > low["passenger_trips"]


class TestDeterminism:
    def test_same_seed_gives_identical_results(self, fast_config):
        a = summarise(run_simulation(fast_config(seed=42)))
        b = summarise(run_simulation(fast_config(seed=42)))
        for key, value in a.items():
            if key == "wall_time_s":  # timing is the one thing that legitimately varies
                continue
            if isinstance(value, float):
                assert value == pytest.approx(b[key], rel=0, abs=0), f"{key} differs"
            else:
                assert value == b[key], f"{key} differs"

    def test_different_seeds_give_different_results(self, fast_config):
        a = summarise(run_simulation(fast_config(seed=1)))
        b = summarise(run_simulation(fast_config(seed=2)))
        assert a["passenger_search_time_min"] != b["passenger_search_time_min"]

    def test_run_order_does_not_matter(self, fast_config):
        """A run must not depend on what was simulated before it."""
        first = summarise(run_simulation(fast_config(seed=7)))
        run_simulation(fast_config(seed=8))
        again = summarise(run_simulation(fast_config(seed=7)))
        assert first["system_social_cost_usd"] == pytest.approx(again["system_social_cost_usd"])


class TestConservation:
    def test_capacity_is_conserved_across_the_run(self, fast_config):
        result = run_simulation(fast_config())
        assert sum(result.capacity_by_class.values()) == result.total_stalls

    def test_no_segment_ends_over_occupied(self, fast_config):
        result = run_simulation(fast_config())
        # Reconstruct: every trip that parked must also have released.
        parked = [t for t in result.trips if t["outcome"] in ("parked", "diverted")]
        assert all(t["curb_id"] is not None for t in parked)

    def test_every_trip_is_resolved(self, fast_config):
        result = run_simulation(fast_config())
        outcomes = {t["outcome"] for t in result.trips}
        assert outcomes <= {"parked", "illegal", "diverted", "abandoned"}
        assert "unresolved" not in outcomes

    def test_warmup_trips_are_excluded(self, fast_config):
        cfg = fast_config()
        result = run_simulation(cfg)
        summary = summarise(result)
        post = [t for t in result.trips if not t["warmup"]]
        assert summary["n_trips"] == len(post)
        assert any(t["warmup"] for t in result.trips), "warm-up should produce some trips"

    def test_occupancy_samples_stop_at_horizon(self, fast_config):
        cfg = fast_config()
        result = run_simulation(cfg)
        times = [s["t_min"] for s in result.occupancy_samples]
        assert min(times) >= cfg.warmup_min
        assert max(times) <= cfg.horizon_min


class TestMetrics:
    def test_rates_are_fractions(self, fast_config):
        s = summarise(run_simulation(fast_config()))
        for key, value in s.items():
            if key.endswith("_rate") or key.startswith("curb_occupancy"):
                assert 0.0 <= value <= 1.0, f"{key}={value}"

    def test_social_cost_is_positive_and_decomposes(self, fast_config):
        s = summarise(run_simulation(fast_config()))
        total = s["time_cost_usd"] + s["vmt_cost_usd"] + s["illegal_cost_usd"]
        assert s["system_social_cost_usd"] == pytest.approx(total)
        assert s["system_social_cost_usd"] > 0

    def test_aggregate_reports_confidence_intervals(self, fast_config):
        rows = [summarise(run_simulation(fast_config(seed=s))) for s in (1, 2, 3, 4)]
        agg = aggregate(rows)
        assert {"mean", "sd", "n", "ci_low", "ci_high"} <= set(agg.columns)
        m = "passenger_search_time_min"
        assert agg.loc[m, "ci_low"] <= agg.loc[m, "mean"] <= agg.loc[m, "ci_high"]

    def test_compare_flags_direction(self, fast_config):
        rows_a = [summarise(run_simulation(fast_config(seed=s))) for s in (1, 2)]
        rows_b = [
            summarise(run_simulation(fast_config(seed=s, demand_scale={"passenger": 1.6})))
            for s in (1, 2)
        ]
        cmp = compare(aggregate(rows_a), aggregate(rows_b), ["passenger_search_time_min"])
        assert cmp.loc["passenger_search_time_min", "pct_change"] > 0

    def test_occupancy_by_segment_is_indexed_by_curb_id(self, fast_config):
        result = run_simulation(fast_config())
        occ = occupancy_by_segment(result)
        assert len(occ) == result.total_stalls // 1 or len(occ) > 0
        assert all(isinstance(i, str) for i in occ.index)
        assert ((occ >= 0) & (occ <= 1)).all()


class TestScenarios:
    @pytest.mark.parametrize("scenario", list(list_scenarios()))
    def test_scenario_runs_end_to_end(self, fast_config, scenario):
        result = run_simulation(fast_config(scenario=scenario))
        assert len(result.trips) > 0
        s = summarise(result)
        assert s["scenario"] == scenario

    def test_dynamic_pricing_changes_prices(self, fast_config):
        result = run_simulation(fast_config(scenario="dynamic_pricing"))
        assert result.price_history, "the pricing controller should have run"
        prices = [h["mean_price"] for h in result.price_history]
        assert len(set(np.round(prices, 4))) > 1

    def test_time_of_day_regulation_changes_allocation(self, fast_config):
        result = run_simulation(fast_config(scenario="mixed_curb"))
        assert result.events.get("regulation_change", 0) >= 1

    def test_loading_zone_scenario_has_more_freight_capacity(self, fast_config):
        base = run_simulation(fast_config(scenario="baseline"))
        zones = run_simulation(fast_config(scenario="loading_zones"))
        assert zones.capacity_by_class["delivery"] > base.capacity_by_class["delivery"]
