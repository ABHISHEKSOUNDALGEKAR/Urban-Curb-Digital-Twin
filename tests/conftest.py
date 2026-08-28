"""Shared fixtures. Tests run on a deliberately small, fast configuration.

The point of the fast config is that the whole suite stays under a minute so it
can run on every push. Anything that needs the full-size district is marked
``slow`` and excluded from the default run.
"""

from __future__ import annotations

import pytest

from src.config import load_scenario

#: A short horizon with a proportionate warm-up: big enough to exercise every
#: code path, small enough to run in about a second.
FAST = {"horizon_min": 70.0, "warmup_min": 20.0}


@pytest.fixture
def fast_config():
    def _make(scenario: str = "baseline", seed: int = 1, **overrides):
        merged = dict(FAST)
        merged.update(overrides)
        return load_scenario(scenario, seed=seed, overrides=merged)

    return _make


@pytest.fixture
def small_network_config():
    """A 3x3 grid: small enough to assert on individual segments."""
    return {
        "network": {
            "grid_rows": 3,
            "grid_cols": 3,
            "block_length_m": 120.0,
            "free_flow_speed_kph": 30.0,
            "origin_xy": [0.0, 0.0],
        },
        "curb": {
            "segments_per_link": 2,
            "stall_length_m": 6.0,
            "usable_fraction": 0.5,
            "capacity_jitter": 0.0,
            "baseline_allocation": {"passenger": 0.6, "delivery": 0.25, "ridehail": 0.15},
            "pricing": {
                "core_price_per_hour": 4.0,
                "edge_price_per_hour": 2.0,
                "core_radius_m": 100.0,
            },
            "time_limit_min": 120,
        },
        "congestion": {
            "bpr_alpha": 0.15,
            "bpr_beta": 4.0,
            "link_capacity_vph": 600.0,
            "double_park_delay_s": 20.0,
        },
        "demand_geography": {
            "core_bias": {"passenger": 1.0, "delivery": 1.0, "ridehail": 1.0},
            "commercial_node_share": 0.5,
        },
    }
