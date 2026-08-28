"""Curb inventory invariants: capacity, apportionment, occupancy accounting."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import VEHICLE_CLASSES, ConfigError, validate_allocation
from src.simulation.curb import CurbInventory, CurbSegment, PricingPolicy, largest_remainder


def make_segment(stalls: int = 10, price: float = 4.0) -> CurbSegment:
    seg = CurbSegment("C1", ("A", "B"), (0.0, 0.0), stalls, price, 120.0)
    seg.set_allocation({"passenger": 0.6, "delivery": 0.25, "ridehail": 0.15})
    return seg


class TestLargestRemainder:
    def test_sums_to_total(self):
        for total in range(0, 40):
            counts = largest_remainder([0.7, 0.18, 0.12], total)
            assert sum(counts) == total

    def test_never_negative(self):
        assert all(c >= 0 for c in largest_remainder([0.5, 0.3, 0.2], 7))

    def test_degenerate_shares(self):
        assert sum(largest_remainder([0.0, 0.0, 0.0], 5)) == 5

    def test_proportional(self):
        counts = largest_remainder([0.5, 0.5], 100)
        assert counts == [50, 50]


class TestCurbSegment:
    def test_allocation_conserves_stalls(self):
        seg = make_segment(13)
        assert sum(p.capacity for p in seg.pools.values()) == 13

    def test_occupancy_never_exceeds_capacity(self):
        seg = make_segment(10)
        cap = seg.capacity("passenger")
        for _ in range(cap):
            assert seg.occupy("passenger", 0.0)
        # One more request must be refused, not silently over-fill the block.
        assert not seg.occupy("passenger", 0.0)
        assert seg.pools["passenger"].occupied == cap

    def test_availability_never_negative(self):
        seg = make_segment(10)
        for _ in range(seg.capacity("delivery") + 5):
            seg.occupy("delivery", 0.0)
        assert seg.available("delivery") >= 0

    def test_release_restores_capacity(self):
        seg = make_segment(10)
        seg.occupy("passenger", 0.0)
        before = seg.available("passenger")
        seg.release("passenger", 1.0)
        assert seg.available("passenger") == before + 1

    def test_release_without_occupancy_raises(self):
        seg = make_segment(10)
        with pytest.raises(RuntimeError):
            seg.release("passenger", 1.0)

    def test_zero_capacity_pool_rejects(self):
        seg = CurbSegment("C2", ("A", "B"), (0.0, 0.0), 4, 3.0, 60.0)
        seg.set_allocation({"passenger": 1.0, "delivery": 0.0, "ridehail": 0.0})
        assert seg.capacity("delivery") == 0
        assert not seg.occupy("delivery", 0.0)

    def test_time_weighted_occupancy(self):
        seg = make_segment(10)
        seg.occupy("passenger", 0.0)
        seg.occupy("passenger", 0.0)
        # Two of ten stalls held for the whole window -> 20% occupancy.
        assert seg.time_weighted_occupancy(60.0, since=0.0) == pytest.approx(0.2, abs=1e-9)

    def test_statistics_reset_drops_history(self):
        seg = make_segment(10)
        seg.occupy("passenger", 0.0)
        seg.reset_statistics(30.0)
        assert seg.time_weighted_occupancy(60.0, since=30.0) == pytest.approx(0.1, abs=1e-9)

    def test_rejections_are_counted(self):
        seg = make_segment(10)
        for _ in range(seg.capacity("ridehail") + 3):
            seg.occupy("ridehail", 0.0)
        assert seg.rejections["ridehail"] == 3


class TestAllocationValidation:
    def test_rejects_shares_that_do_not_sum_to_one(self):
        with pytest.raises(ConfigError):
            validate_allocation({"passenger": 0.5, "delivery": 0.2, "ridehail": 0.2})

    def test_rejects_negative_shares(self):
        with pytest.raises(ConfigError):
            validate_allocation({"passenger": 1.2, "delivery": -0.1, "ridehail": -0.1})

    def test_rejects_missing_class(self):
        with pytest.raises(ConfigError):
            validate_allocation({"passenger": 0.8, "delivery": 0.2})


class TestInventoryApportionment:
    def test_district_shares_are_exact(self, small_network_config):
        from src.simulation.routing import build_grid_network

        _graph, inventory = build_grid_network(small_network_config)
        total = inventory.total_stalls
        for alloc in (
            {"passenger": 0.82, "delivery": 0.13, "ridehail": 0.05},
            {"passenger": 0.50, "delivery": 0.30, "ridehail": 0.20},
            {"passenger": 0.9, "delivery": 0.06, "ridehail": 0.04},
        ):
            inventory.set_allocation(alloc)
            caps = inventory.capacity_by_class()
            assert sum(caps.values()) == total, "re-allocation must conserve stalls"
            for cls in VEHICLE_CLASSES:
                # Within one stall of the requested district-wide share.
                assert abs(caps[cls] / total - alloc[cls]) <= 1.5 / total

    def test_per_segment_totals_preserved(self, small_network_config):
        from src.simulation.routing import build_grid_network

        _graph, inventory = build_grid_network(small_network_config)
        before = [s.total_stalls for s in inventory]
        inventory.set_allocation({"passenger": 0.4, "delivery": 0.35, "ridehail": 0.25})
        for seg, n in zip(inventory, before, strict=True):
            assert sum(p.capacity for p in seg.pools.values()) == n

    def test_within_radius_is_symmetric(self, small_network_config):
        from src.simulation.routing import build_grid_network

        _graph, inventory = build_grid_network(small_network_config)
        near = inventory.within((0.0, 0.0), 70.0)
        assert all(np.hypot(*s.xy) <= 70.0 for s in near)


class TestPricingPolicy:
    def test_static_policy_never_changes_price(self):
        seg = make_segment(10, price=3.0)
        inv = CurbInventory([seg])
        PricingPolicy("static").review(inv, 30.0)
        assert seg.price_per_hour == 3.0

    def test_dynamic_policy_raises_price_when_full(self):
        seg = make_segment(10, price=3.0)
        for _ in range(seg.capacity("passenger")):
            seg.occupy("passenger", 0.0)
        for _ in range(seg.capacity("delivery")):
            seg.occupy("delivery", 0.0)
        for _ in range(seg.capacity("ridehail")):
            seg.occupy("ridehail", 0.0)
        inv = CurbInventory([seg])
        policy = PricingPolicy("dynamic", {"review_interval_min": 30, "step_up": 1.25})
        policy.review(inv, 30.0)
        assert seg.price_per_hour > 3.0

    def test_dynamic_policy_respects_bounds(self):
        seg = make_segment(10, price=11.5)
        for _ in range(seg.total_stalls):
            for cls in VEHICLE_CLASSES:
                seg.occupy(cls, 0.0)
        inv = CurbInventory([seg])
        policy = PricingPolicy(
            "dynamic", {"max_price": 12.0, "step_up": 2.0, "review_interval_min": 30}
        )
        policy.review(inv, 30.0)
        assert seg.price_per_hour <= 12.0
