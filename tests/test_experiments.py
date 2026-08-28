"""Experiment orchestration, calibration machinery and provenance."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.calibration import parameters as P
from src.calibration.calibrate import (
    CalibrationObjective,
    Windows,
    make_windows,
    occupancy_error,
)
from src.config import load_scenario
from src.experiments.metrics import mae, rmse
from src.experiments.provenance import Manifest, config_hash
from src.experiments.runner import _resolve_workers, run_replications, run_scenario

FAST = {"horizon_min": 70.0, "warmup_min": 20.0}


class TestRunner:
    def test_replications_are_labelled_with_their_seed(self):
        cfg = load_scenario("baseline", seed=1, overrides=FAST)
        rows = run_replications(cfg, [1, 2, 3], workers=1)
        assert [r["seed"] for r in rows] == [1, 2, 3]

    def test_parallel_and_sequential_agree_exactly(self):
        """Parallelism must not change a single number."""
        cfg = load_scenario("baseline", seed=1, overrides=FAST)
        seq = run_replications(cfg, [1, 2], workers=1)
        par = run_replications(cfg, [1, 2], workers=2)
        for a, b in zip(seq, par, strict=True):
            assert a["seed"] == b["seed"]
            assert a["system_social_cost_usd"] == pytest.approx(b["system_social_cost_usd"])
            assert a["passenger_search_time_min"] == pytest.approx(b["passenger_search_time_min"])

    def test_worker_count_is_clamped_to_the_job_count(self):
        assert _resolve_workers(64, 3) == 3
        assert _resolve_workers(0, 10) == 1
        assert _resolve_workers(None, 1) == 1

    def test_scenario_run_writes_expected_artefacts(self, tmp_path):
        per_seed, agg, manifest = run_scenario(
            "baseline", [1, 2], workers=1, overrides=FAST, out_dir=tmp_path, progress=False
        )
        d = tmp_path / "baseline"
        assert (d / "per_seed.csv").exists()
        assert (d / "aggregate.csv").exists()
        assert (d / "manifest.json").exists()
        assert len(per_seed) == 2
        assert "passenger_search_time_min" in agg.index
        saved = json.loads((d / "manifest.json").read_text())
        assert saved["seeds"] == [1, 2]
        assert saved["config_hash"] == manifest.config_hash


class TestProvenance:
    def test_manifest_records_the_environment(self):
        m = Manifest.create("test", "baseline", [1, 2], {"a": 1})
        assert m.seeds == [1, 2]
        assert m.python_version and m.platform
        assert m.config_hash == config_hash({"a": 1})

    def test_config_hash_is_order_independent(self):
        assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})

    def test_config_hash_detects_change(self):
        assert config_hash({"a": 1}) != config_hash({"a": 2})

    def test_manifest_round_trips(self, tmp_path):
        m = Manifest.create("test", "baseline", [1], {"a": 1}).finish(1.5)
        path = m.write(tmp_path / "m.json")
        loaded = json.loads(path.read_text())
        assert loaded["wall_time_s"] == 1.5
        assert loaded["finished_utc"]


class TestCalibrationParameters:
    def test_every_default_is_inside_its_bounds(self):
        for p in P.PARAMETERS:
            assert p.low <= p.default <= p.high, p.name

    def test_overrides_write_to_the_right_config_path(self):
        theta = P.defaults()
        ov = P.to_overrides(theta)
        assert ov["agents"]["passenger"]["alpha_walk_per_m"] == P.PARAMETERS[0].default
        assert "delivery" in ov["agents"] and "ridehail" in ov["agents"]

    def test_overrides_actually_change_the_config(self):
        theta = P.defaults().copy()
        theta[0] = 0.019
        cfg = load_scenario("baseline", seed=1, overrides=P.to_overrides(theta))
        assert cfg.agents["passenger"]["alpha_walk_per_m"] == pytest.approx(0.019)

    def test_out_of_range_values_are_clipped(self):
        theta = P.defaults().copy()
        theta[0] = 99.0
        ov = P.to_overrides(theta)
        assert ov["agents"]["passenger"]["alpha_walk_per_m"] == P.PARAMETERS[0].high

    def test_wrong_length_vector_raises(self):
        with pytest.raises(ValueError):
            P.to_overrides([0.1, 0.2])

    def test_dict_round_trip(self):
        theta = P.defaults()
        assert np.allclose(P.from_dict(P.to_dict(theta)), theta)

    def test_normalisation_maps_to_unit_box(self):
        n = P.normalise(P.defaults())
        assert ((n >= 0) & (n <= 1)).all()


class TestCalibrationMachinery:
    def test_error_metrics(self):
        a = np.array([0.5, 0.5, 0.5])
        b = np.array([0.6, 0.4, 0.5])
        assert rmse(a, b) == pytest.approx(np.sqrt((0.01 + 0.01 + 0) / 3))
        assert mae(a, b) == pytest.approx(0.2 / 3)

    def test_occupancy_error_aligns_on_shared_segments(self):
        obs = pd.Series({"a": 0.5, "b": 0.7, "c": 0.9})
        sim = pd.Series({"a": 0.5, "b": 0.7})
        err = occupancy_error(obs, sim)
        assert err["n_segments"] == 2
        assert err["rmse"] == pytest.approx(0.0)

    def test_perfect_fit_has_zero_error(self):
        s = pd.Series({"a": 0.4, "b": 0.8})
        err = occupancy_error(s, s)
        assert err["rmse"] == pytest.approx(0.0)
        assert err["bias"] == pytest.approx(0.0)

    def test_windows_do_not_overlap_and_cover_the_run(self):
        w = Windows(warmup_min=90.0, horizon_min=330.0)
        assert w.calibration[1] == w.validation[0]
        assert w.calibration[0] == 90.0 and w.validation[1] == 330.0

    def test_windows_come_from_the_scenario(self):
        w = make_windows("baseline")
        cfg = load_scenario("baseline", seed=0)
        assert w.warmup_min == cfg.warmup_min
        assert w.horizon_min == cfg.horizon_min

    def test_objective_is_deterministic_for_fixed_seeds(self):
        observed = pd.Series(
            {s: 0.8 for s in ("N00_00_N00_01_S0", "N00_00_N01_00_S0")}, dtype=float
        )
        obj = CalibrationObjective(observed, [1], Windows(20.0, 70.0), extra_overrides=FAST)
        first = obj(P.defaults())
        obj._cache.clear()
        assert obj(P.defaults()) == pytest.approx(first)


@pytest.mark.slow
class TestSensitivity:
    def test_elasticity_sign_for_own_demand(self):
        """More passenger demand must not reduce passenger search time."""
        from src.experiments.sensitivity import cross_modal_elasticities

        el = cross_modal_elasticities(seeds=[1, 2], shock=0.35, workers=1, verbose=False)
        own = el[(el.shocked_class == "passenger") & (el.metric == "passenger_search_time_min")]
        assert float(own["elasticity"].iloc[0]) > 0

    def test_table_covers_every_class(self):
        from src.experiments.sensitivity import cross_modal_elasticities

        el = cross_modal_elasticities(seeds=[1], shock=0.3, workers=1, verbose=False)
        assert set(el["shocked_class"]) == {"passenger", "delivery", "ridehail"}
