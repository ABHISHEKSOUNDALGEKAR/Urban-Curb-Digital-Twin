"""SUMO backend tests.

Every test here skips cleanly when SUMO is not installed: the microsimulation
backend is optional, and a missing optional dependency is not a test failure.
Install it with ``pip install eclipse-sumo traci sumolib``.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from src.config import load_scenario
from src.sumo import sumo_available

pytestmark = [
    pytest.mark.sumo,
    pytest.mark.skipif(not sumo_available(), reason="SUMO / TraCI not installed"),
]


@pytest.fixture(scope="module")
def net(tmp_path_factory):
    from src.sumo.network import build_sumo_network

    cfg = load_scenario("baseline", seed=1)
    return build_sumo_network(cfg, tmp_path_factory.mktemp("sumonet"), force=True)


class TestNetworkGeneration:
    def test_network_file_is_written(self, net):
        assert net.net_file.exists() and net.net_file.stat().st_size > 0

    def test_curb_capacity_matches_the_simpy_inventory(self, net):
        """Both backends must model exactly the same curb, or comparison is meaningless."""
        assert sum(net.parking_capacity.values()) == net.inventory.total_stalls

    def test_one_parking_area_per_regulation_with_capacity(self, net):
        for seg in net.inventory:
            for reg in ("passenger", "delivery", "ridehail"):
                has_area = reg in net.parking_areas.get(seg.id, {})
                assert has_area == (seg.capacity(reg) > 0)

    def test_parking_areas_fit_inside_their_lane(self, net):
        root = ET.parse(net.additional_file).getroot()
        for pa in root.findall("parkingArea"):
            start, end = float(pa.get("startPos")), float(pa.get("endPos"))
            assert 0 <= start < end

    def test_vehicle_types_are_declared(self, net):
        root = ET.parse(net.additional_file).getroot()
        ids = {v.get("id") for v in root.findall("vType")}
        assert {"passenger", "delivery", "ridehail"} <= ids

    def test_taxi_device_is_enabled_on_ridehail(self, net):
        root = ET.parse(net.additional_file).getroot()
        rh = next(v for v in root.findall("vType") if v.get("id") == "ridehail")
        params = {p.get("key"): p.get("value") for p in rh.findall("param")}
        assert params.get("has.taxi.device") == "true"

    def test_rerouters_reference_real_parking_areas(self, net):
        root = ET.parse(net.additional_file).getroot()
        known = set(net.parking_capacity)
        for rr in root.findall("rerouter"):
            for interval in rr.findall("interval"):
                for reroute in interval.findall("parkingAreaReroute"):
                    assert reroute.get("id") in known


@pytest.mark.slow
class TestSumoRun:
    def test_short_run_completes_and_reports(self, tmp_path):
        from src.sumo.backend import run_sumo_simulation

        summary = run_sumo_simulation(
            scenario="baseline", seed=1, horizon_min=6.0, work_dir=tmp_path
        )
        assert summary["backend"] == "sumo"
        assert summary["n_vehicles"] > 0
        assert 0.0 <= summary["illegal_parking_rate"] <= 1.0

    def test_taxi_reservations_are_dispatched(self, tmp_path):
        from src.sumo.backend import run_sumo_simulation

        summary = run_sumo_simulation(
            scenario="baseline", seed=2, horizon_min=8.0, work_dir=tmp_path
        )
        dispatch = summary.get("ridehail_dispatch", {})
        assert dispatch.get("dispatched", 0) > 0, "SUMO taxi device produced no dispatches"
        assert dispatch["completed_pickups"] <= dispatch["dispatched"]
