"""Optimization: constraint handling, objective decomposition, search behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import VEHICLE_CLASSES, load_optimization_config
from src.optimization.baseline import (
    current_allocation,
    demand_share_allocation,
    equal_allocation,
)
from src.optimization.objective import (
    COMPONENTS,
    CurbObjective,
    allocation_dict,
    project_to_simplex,
)
from src.optimization.optimizer import is_pareto_efficient, random_search

BOUNDS = {"passenger": (0.30, 0.85), "delivery": (0.05, 0.45), "ridehail": (0.05, 0.40)}


def fast_opt_cfg(seeds: int = 1) -> dict:
    """A tiny optimization config so the tests exercise the machinery, not the CPU."""
    return {
        "decision": {"bounds": {k: list(v) for k, v in BOUNDS.items()}},
        "objective": {
            "weights": {c: 1.0 for c in COMPONENTS},
            "vmt_external_cost_per_mile": 0.62,
            "illegal_event_social_cost": 14.0,
        },
        "search": {"seeds_per_evaluation": seeds, "common_random_numbers": True},
    }


class TestSimplexProjection:
    def test_result_sums_to_one(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            x = rng.random(3) * rng.integers(1, 10)
            v = project_to_simplex(x, BOUNDS)
            assert v.sum() == pytest.approx(1.0, abs=1e-6)

    def test_result_respects_bounds(self):
        rng = np.random.default_rng(1)
        for _ in range(200):
            v = project_to_simplex(rng.random(3), BOUNDS)
            for i, c in enumerate(VEHICLE_CLASSES):
                assert BOUNDS[c][0] - 1e-6 <= v[i] <= BOUNDS[c][1] + 1e-6

    def test_extreme_input_is_clipped_into_the_box(self):
        v = project_to_simplex([1e6, 1e-9, 1e-9], BOUNDS)
        assert v[0] == pytest.approx(BOUNDS["passenger"][1], abs=1e-6)

    def test_zero_vector_does_not_blow_up(self):
        v = project_to_simplex([0.0, 0.0, 0.0], BOUNDS)
        assert np.isfinite(v).all() and v.sum() == pytest.approx(1.0)

    def test_infeasible_bounds_raise(self):
        bad = {"passenger": (0.9, 1.0), "delivery": (0.9, 1.0), "ridehail": (0.9, 1.0)}
        with pytest.raises(ValueError):
            project_to_simplex([1, 1, 1], bad)

    def test_no_bounds_still_normalises(self):
        v = project_to_simplex([2.0, 1.0, 1.0])
        assert v.sum() == pytest.approx(1.0)
        assert v[0] == pytest.approx(0.5)

    def test_allocation_dict_keys(self):
        d = allocation_dict([0.5, 0.3, 0.2])
        assert set(d) == set(VEHICLE_CLASSES)
        assert sum(d.values()) == pytest.approx(1.0)


class TestBaselines:
    def test_equal_allocation_is_uniform(self):
        alloc = equal_allocation()
        assert all(v == pytest.approx(1 / 3) for v in alloc.values())

    def test_demand_share_is_a_distribution(self):
        alloc = demand_share_allocation()
        assert sum(alloc.values()) == pytest.approx(1.0)
        assert all(v >= 0 for v in alloc.values())

    def test_demand_share_favours_the_longest_dwell(self):
        """Passenger cars dominate curb-*time* demand even at moderate arrival rates."""
        alloc = demand_share_allocation()
        assert alloc["passenger"] > alloc["delivery"]
        assert alloc["passenger"] > alloc["ridehail"]

    def test_current_allocation_matches_config(self):
        assert current_allocation() == pytest.approx(
            {k: v for k, v in current_allocation().items()}
        )


class TestObjective:
    @pytest.fixture(scope="class")
    def objective(self):
        return CurbObjective(opt_cfg=fast_opt_cfg(seeds=1))

    def test_evaluation_is_finite_and_positive(self, objective):
        ev = objective.evaluate([0.8, 0.13, 0.07])
        assert np.isfinite(ev.value) and ev.value > 0

    def test_components_sum_to_the_objective_under_unit_weights(self, objective):
        ev = objective.evaluate([0.8, 0.13, 0.07])
        assert sum(ev.components.values()) == pytest.approx(ev.value, rel=1e-6)

    def test_evaluation_is_cached(self, objective):
        objective.evaluate([0.7, 0.2, 0.1])
        before = objective.n_simulations
        objective.evaluate([0.7, 0.2, 0.1])
        assert objective.n_simulations == before, "repeat evaluations must be free"

    def test_allocation_is_always_feasible(self, objective):
        ev = objective.evaluate([5.0, 0.0, 0.0])
        assert sum(ev.allocation.values()) == pytest.approx(1.0)
        assert ev.allocation["passenger"] <= BOUNDS["passenger"][1] + 1e-6

    def test_common_random_numbers_are_reused(self, objective):
        a = objective.evaluate([0.8, 0.12, 0.08])
        b = objective.evaluate([0.6, 0.25, 0.15])
        assert a.seeds == b.seeds

    @pytest.mark.slow
    def test_more_seeds_reduce_the_standard_error(self):
        few = CurbObjective(opt_cfg=fast_opt_cfg(seeds=2)).evaluate([0.8, 0.13, 0.07])
        many = CurbObjective(opt_cfg=fast_opt_cfg(seeds=8)).evaluate([0.8, 0.13, 0.07])
        assert many.stderr < few.stderr


class TestPareto:
    def test_single_point_is_efficient(self):
        assert is_pareto_efficient(np.array([[1.0, 2.0]])).tolist() == [True]

    def test_dominated_point_is_excluded(self):
        costs = np.array([[1.0, 1.0], [2.0, 2.0]])
        assert is_pareto_efficient(costs).tolist() == [True, False]

    def test_incomparable_points_are_both_efficient(self):
        costs = np.array([[1.0, 3.0], [3.0, 1.0]])
        assert is_pareto_efficient(costs).tolist() == [True, True]

    def test_frontier_is_non_empty(self):
        rng = np.random.default_rng(3)
        costs = rng.random((60, 3))
        mask = is_pareto_efficient(costs)
        assert mask.any() and not mask.all()


@pytest.mark.slow
class TestSearch:
    def test_random_search_returns_a_feasible_allocation(self):
        objective = CurbObjective(opt_cfg=fast_opt_cfg(seeds=1))
        result = random_search(objective, n_samples=4, verbose=False)
        assert sum(result.allocation.values()) == pytest.approx(1.0)
        assert result.n_evaluations >= 1

    def test_random_search_reports_the_best_point_it_saw(self):
        objective = CurbObjective(opt_cfg=fast_opt_cfg(seeds=1))
        result = random_search(objective, n_samples=5, verbose=False)
        seen = [e.value for e in objective.history]
        assert result.objective == pytest.approx(min(seen))


class TestOptimizationConfig:
    def test_shipped_config_is_valid(self):
        cfg = load_optimization_config()
        bounds = cfg["decision"]["bounds"]
        lo = sum(b[0] for b in bounds.values())
        hi = sum(b[1] for b in bounds.values())
        assert lo <= 1.0 <= hi, "the allocation bounds must admit a feasible point"
        assert set(cfg["objective"]["weights"]) == set(COMPONENTS)
