"""TraCI backend: SUMO moves the vehicles, Python makes the decisions.

The control loop is one second of simulated time per iteration:

    1. release any vehicles whose arrival time has come
    2. let SUMO advance the traffic
    3. read back what happened (who parked, who left, who is stuck)
    4. re-decide for the vehicles that need a decision (search again, give up,
       double-park) and push those decisions back through TraCI

Curb space is represented by SUMO ``parkingArea`` elements generated from the
same inventory the SimPy engine uses, so "is there a stall free on this block"
is answered by the microsimulation rather than by a counter, and a vehicle that
double-parks physically blocks the lane behind it.

Scope
-----
This backend exists to check that the curb-competition mechanism survives
contact with lane-level traffic dynamics, and to give the model a path to
realistic network effects. It is roughly two orders of magnitude slower than the
SimPy engine, so calibration and optimization stay on SimPy; see the README.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.agents.base_agent import curb_choice_score
from src.config import VEHICLE_CLASSES, RunConfig, load_scenario
from src.simulation.engine import profile_multiplier
from src.sumo import require_sumo, sumo_binary
from src.sumo.network import SumoNetworkFiles, build_sumo_network

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


@dataclass
class SumoTrip:
    """Per-vehicle bookkeeping, mirroring the SimPy ``TripRecord`` fields."""

    veh_id: str
    vehicle_class: str
    destination_node: str
    t_arrive_s: float
    dwell_s: float
    # When the search clock starts: the vehicle is released at the district
    # boundary, and the drive in is approach, not search. Counting the approach
    # against search patience makes vehicles "give up" before they have even
    # reached the block they were heading for - which inflates failed attempts
    # and double-parking, and is not comparable with the SimPy engine, where
    # search begins on arrival at the destination.
    t_search_start_s: float = 0.0
    candidates: list[tuple[str, float, str]] = field(default_factory=list)
    attempt: int = 0
    target_pa: str | None = None
    t_parked_s: float | None = None
    t_done_s: float | None = None
    outcome: str = "searching"
    walk_m: float = 0.0
    failed_attempts: int = 0

    @property
    def search_time_min(self) -> float:
        if self.t_parked_s is None:
            return 0.0
        return max(0.0, (self.t_parked_s - self.t_search_start_s)) / 60.0


class SumoBackend:
    """Runs one replication of the district in SUMO."""

    def __init__(
        self,
        cfg: RunConfig,
        work_dir: Path | str,
        gui: bool = False,
        step_length: float = 1.0,
        seed: int | None = None,
        use_taxi_device: bool = True,
    ) -> None:
        require_sumo()
        self.cfg = cfg
        self.work_dir = Path(work_dir)
        self.gui = gui
        self.step_length = float(step_length)
        self.seed = cfg.seed if seed is None else int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.use_taxi_device = use_taxi_device

        self.files: SumoNetworkFiles = build_sumo_network(cfg, self.work_dir, force=False)
        self.inventory = self.files.inventory
        self.trips: dict[str, SumoTrip] = {}
        self.finished: list[SumoTrip] = []
        self.illegal_events = 0
        self.dispatcher = None
        self._traci = None
        self._segment_by_pa: dict[str, str] = {
            pa: seg for seg, regs in self.files.parking_areas.items() for pa in regs.values()
        }

    # -- lifecycle --------------------------------------------------------------
    def start(self) -> None:
        import traci

        self._traci = traci
        binary = sumo_binary("sumo-gui" if self.gui else "sumo")
        cmd = [
            binary,
            "-c",
            str(self.files.sumocfg),
            "--step-length",
            str(self.step_length),
            "--seed",
            str(self.seed),
            "--no-step-log",
            "true",
            "--no-warnings",
            "true",
            "--time-to-teleport",
            "600",
            "--device.rerouting.probability",
            "1.0",
        ]
        if self.use_taxi_device:
            # Hand reservation matching to TraCI; without this SUMO uses its own
            # greedy dispatcher and `dispatchTaxi` is rejected.
            cmd += ["--device.taxi.dispatch-algorithm", "traci"]
        traci.start(cmd)

    def close(self) -> None:
        if self._traci is not None:
            with contextlib.suppress(Exception):  # SUMO may already be gone
                self._traci.close()
            self._traci = None

    # -- geometry helpers -------------------------------------------------------
    def _edges_into(self, node: str) -> list[str]:
        return [eid for (u, v), eid in self.files.edge_by_link.items() if v == node]

    def _edges_out_of(self, node: str) -> list[str]:
        return [eid for (u, v), eid in self.files.edge_by_link.items() if u == node]

    def _random_boundary_edge(self) -> str:
        edges = list(self.files.edge_by_link.values())
        return edges[int(self.rng.integers(len(edges)))]

    # -- demand -----------------------------------------------------------------
    def _arrival_schedule(self) -> list[tuple[float, str]]:
        """Pre-generate arrivals for the whole horizon (non-homogeneous Poisson)."""
        horizon_min = self.cfg.horizon_min
        profile = self.cfg.scenario_spec.get("demand_profile") or [1.0]
        peak = max(profile)
        out: list[tuple[float, str]] = []
        for cls, rate in self.cfg.demand.items():
            if rate <= 0:
                continue
            lam_max = rate * peak / 60.0
            t = 0.0
            while True:
                t += float(self.rng.exponential(1.0 / lam_max))
                if t >= horizon_min:
                    break
                lam = rate * profile_multiplier(profile, t, horizon_min) / 60.0
                if self.rng.random() <= lam / lam_max:
                    out.append((t * 60.0, cls))
        out.sort()
        return out

    # -- curb choice ------------------------------------------------------------
    def _candidates(
        self, node: str, vehicle_class: str, limit: int = 6
    ) -> list[tuple[str, float, str]]:
        """Rank ``(parkingArea, walk_m, regulation)`` around ``node``."""
        traci = self._traci
        params = self.cfg.agents[vehicle_class]
        regs = (
            ("delivery", "passenger")
            if vehicle_class == "delivery"
            else (("ridehail", "passenger") if vehicle_class == "ridehail" else ("passenger",))
        )
        max_walk = float(
            params.get("max_walk_distance_m", params.get("max_pickup_distance_m", 300.0))
        )
        radius = float(params["search_radius_m"])
        dwell_h = self._mean_dwell_min(vehicle_class) / 60.0

        node_xy = self.inventory.segments[0].xy if False else self._node_xy(node)
        scored: list[tuple[float, str, float, str]] = []
        for seg, dist in self._segments_near(node, radius):
            walk = float(np.hypot(seg.xy[0] - node_xy[0], seg.xy[1] - node_xy[1]))
            if walk > max_walk:
                continue
            for reg in regs:
                pa = self.files.parking_areas.get(seg.id, {}).get(reg)
                if pa is None:
                    continue
                capacity = self.files.parking_capacity[pa]
                occupied = traci.parkingarea.getVehicleCount(pa) if traci else 0
                occ = occupied / capacity if capacity else 1.0
                approach_min = dist / float(self.cfg.agents["common"]["search_speed_mps"]) / 60.0
                cost = curb_choice_score(
                    params,
                    walk_m=walk,
                    price=seg.price_per_hour * dwell_h,
                    approach_min=approach_min,
                    occupancy=occ,
                )
                scored.append((cost, pa, walk, reg))
                break
        scored.sort(key=lambda t: t[0])
        return [(pa, w, r) for _c, pa, w, r in scored[:limit]]

    def _node_xy(self, node: str) -> tuple[float, float]:
        from src.simulation.routing import build_grid_network

        if not hasattr(self, "_node_xy_cache"):
            graph, _ = build_grid_network(self.cfg.network)
            self._node_xy_cache = {
                v: (graph.nodes[v]["x"], graph.nodes[v]["y"]) for v in graph.nodes
            }
        return self._node_xy_cache[node]

    def _segments_near(self, node: str, radius: float):
        xy = self._node_xy(node)
        d = self.inventory.distances_from(xy)
        order = np.argsort(d)
        out = []
        for i in order:
            if d[int(i)] > radius:
                break
            out.append((self.inventory.segments[int(i)], float(d[int(i)])))
        return out

    def _mean_dwell_min(self, vehicle_class: str) -> float:
        a = self.cfg.agents[vehicle_class]
        if vehicle_class == "passenger":
            return float(a["mean_dwell_min"])
        if vehicle_class == "delivery":
            return float(a["mean_service_min"])
        return float(a["mean_dwell_s"]) / 60.0

    def _sample_dwell_s(self, vehicle_class: str) -> float:
        a = self.cfg.agents[vehicle_class]
        mean = self._mean_dwell_min(vehicle_class)
        cv = float(a.get("dwell_cv", a.get("service_cv", 0.5)))
        sigma = float(np.sqrt(np.log(1.0 + cv**2)))
        mu = float(np.log(max(mean, 1e-9)) - 0.5 * sigma**2)
        return float(self.rng.lognormal(mu, sigma)) * 60.0

    # -- vehicle release --------------------------------------------------------
    def _release(self, veh_class: str, now_s: float, index: int) -> None:
        traci = self._traci
        node = self._sample_destination(veh_class)
        dest_edges = self._edges_into(node)
        if not dest_edges:
            return
        dest_edge = dest_edges[int(self.rng.integers(len(dest_edges)))]
        origin_edge = self._random_boundary_edge()
        veh_id = f"{veh_class}_{index}"
        route_id = f"r_{veh_id}"
        try:
            route = traci.simulation.findRoute(origin_edge, dest_edge)
            if not route.edges:
                return
            traci.route.add(route_id, route.edges)
            traci.vehicle.add(veh_id, route_id, typeID=veh_class, departLane="best")
        except traci.TraCIException:
            return

        trip = SumoTrip(
            veh_id=veh_id,
            vehicle_class=veh_class,
            destination_node=node,
            t_arrive_s=now_s,
            t_search_start_s=now_s + float(getattr(route, "travelTime", 0.0)),
            dwell_s=self._sample_dwell_s(veh_class),
        )
        trip.candidates = self._candidates(node, veh_class)
        self.trips[veh_id] = trip
        self._try_stop(trip)

    def _sample_destination(self, veh_class: str) -> str:
        if not hasattr(self, "_weights"):
            from src.simulation.environment import CurbWorld

            world = CurbWorld(self.cfg)
            self._weights = world.node_weights
            self._nodes = world.nodes
        idx = self.rng.choice(len(self._nodes), p=self._weights[veh_class])
        return self._nodes[int(idx)]

    def _pa_edge(self, pa: str) -> str:
        """Edge a parking area sits on (cached)."""
        if not hasattr(self, "_pa_edge_cache"):
            self._pa_edge_cache: dict[str, str] = {}
        edge = self._pa_edge_cache.get(pa)
        if edge is None:
            lane = self._traci.parkingarea.getLaneID(pa)
            edge = lane.rsplit("_", 1)[0]
            self._pa_edge_cache[pa] = edge
        return edge

    def _try_stop(self, trip: SumoTrip) -> bool:
        """Point a searching vehicle at its next candidate parking area.

        The vehicle must be re-routed toward the candidate block first: SUMO
        rejects a stop that is not downstream of the current route, and a driver
        who decides to try the next block does in fact change where they are
        driving. This is the search itself, expressed in TraCI.
        """
        traci = self._traci
        while trip.attempt < len(trip.candidates):
            pa, walk, _reg = trip.candidates[trip.attempt]
            try:
                traci.vehicle.changeTarget(trip.veh_id, self._pa_edge(pa))
                traci.vehicle.setParkingAreaStop(trip.veh_id, pa, duration=trip.dwell_s)
                trip.target_pa = pa
                trip.walk_m = walk
                return True
            except traci.TraCIException:
                trip.attempt += 1
                trip.failed_attempts += 1
        return False

    def _district_wide_fallback(self, trip: SumoTrip) -> bool:
        """Compliant fallback: take the nearest free stall anywhere in the district.

        Mirrors ``BaseAgent.search_district_wide`` in the SimPy engine. Without
        it the SUMO backend converts every local search failure straight into a
        double-park, and the two backends stop being comparable on the illegal
        parking rate for a reason that has nothing to do with traffic physics.
        """
        traci = self._traci
        params = self.cfg.agents[trip.vehicle_class]
        compliance = float(
            params.get(
                "compliance_probability", 1.0 - params.get("illegal_parking_probability", 0.0)
            )
        )
        if self.rng.random() > compliance:
            return False
        node_xy = self._node_xy(trip.destination_node)
        regs = ("delivery", "passenger") if trip.vehicle_class == "delivery" else ("passenger",)
        best, best_walk = None, float("inf")
        for seg in self.inventory:
            for reg in regs:
                pa = self.files.parking_areas.get(seg.id, {}).get(reg)
                if pa is None:
                    continue
                if traci.parkingarea.getVehicleCount(pa) >= self.files.parking_capacity[pa]:
                    continue
                walk = float(np.hypot(seg.xy[0] - node_xy[0], seg.xy[1] - node_xy[1]))
                if walk < best_walk:
                    best, best_walk = (pa, reg), walk
                break
        if best is None:
            return False
        trip.candidates.append((best[0], best_walk, best[1]))
        trip.attempt = len(trip.candidates) - 1
        if self._try_stop(trip):
            trip.outcome = "searching"
            return True
        return False

    def _give_up(self, trip: SumoTrip, now_s: float) -> None:
        """No stall found: double-park in the lane, blocking traffic behind."""
        if self._district_wide_fallback(trip):
            return
        traci = self._traci
        try:
            edge = traci.vehicle.getRoadID(trip.veh_id)
            if edge.startswith(":"):  # inside a junction; try again next step
                return
            length = traci.lane.getLength(f"{edge}_0")
            traci.vehicle.setStop(
                trip.veh_id,
                edge,
                pos=max(5.0, length - 8.0),
                laneIndex=0,
                duration=min(trip.dwell_s, 600.0),
                flags=0,
            )
            trip.outcome = "illegal"
            trip.t_parked_s = now_s
            self.illegal_events += 1
        except traci.TraCIException:
            trip.outcome = "abandoned"
            trip.t_done_s = now_s
            self._retire(trip)

    def _retire(self, trip: SumoTrip) -> None:
        self.finished.append(trip)
        self.trips.pop(trip.veh_id, None)

    # -- main loop --------------------------------------------------------------
    def run(self, max_steps: int | None = None, verbose: bool = False) -> dict[str, Any]:
        traci = self._traci
        if traci is None:
            self.start()
            traci = self._traci

        schedule = self._arrival_schedule()
        if self.use_taxi_device:
            self._spawn_taxi_fleet()

        horizon_s = self.cfg.horizon_min * 60.0
        tail_s = 45 * 60.0
        next_arrival = 0
        t0 = time.perf_counter()
        step = 0
        while True:
            now_s = traci.simulation.getTime()
            if now_s > horizon_s + tail_s:
                break
            if max_steps is not None and step >= max_steps:
                break
            if traci.simulation.getMinExpectedNumber() <= 0 and next_arrival >= len(schedule):
                break

            while next_arrival < len(schedule) and schedule[next_arrival][0] <= now_s:
                t, cls = schedule[next_arrival]
                if cls == "ridehail" and self.use_taxi_device:
                    self._submit_person(next_arrival, now_s)
                else:
                    self._release(cls, now_s, next_arrival)
                next_arrival += 1

            traci.simulationStep()
            step += 1
            self._collect(traci.simulation.getTime())
            if self.dispatcher is not None:
                self.dispatcher.step(traci.simulation.getTime())
            if verbose and step % 600 == 0:
                print(
                    f"  t={traci.simulation.getTime() / 60:6.1f} min  "
                    f"running={traci.vehicle.getIDCount():4d}  finished={len(self.finished):5d}",
                    flush=True,
                )

        wall = time.perf_counter() - t0
        summary = self.summarise()
        summary["wall_time_s"] = wall
        summary["steps"] = step
        self.close()
        return summary

    def _collect(self, now_s: float) -> None:
        traci = self._traci
        for veh_id in traci.simulation.getParkingStartingVehiclesIDList():
            trip = self.trips.get(veh_id)
            if trip is not None and trip.t_parked_s is None:
                trip.t_parked_s = now_s
                trip.outcome = "parked"
        for veh_id in traci.simulation.getArrivedIDList():
            trip = self.trips.get(veh_id)
            if trip is not None:
                trip.t_done_s = now_s
                if trip.outcome == "searching":
                    trip.outcome = "abandoned"
                self._retire(trip)

        # Vehicles that have been searching too long take their fallback action.
        patience_s = {
            c: float(self.cfg.agents[c]["search_patience_min"]) * 60.0 for c in VEHICLE_CLASSES
        }
        for trip in list(self.trips.values()):
            if trip.outcome != "searching":
                continue
            if now_s - trip.t_search_start_s > patience_s[trip.vehicle_class]:
                trip.attempt += 1
                trip.failed_attempts += 1
                if not self._try_stop(trip):
                    self._give_up(trip, now_s)

    # -- ridehail ---------------------------------------------------------------
    def _spawn_taxi_fleet(self) -> None:
        from src.sumo.dispatch import TaxiDispatcher

        traci = self._traci
        cfg = self.cfg.agents["ridehail"]["fleet"]
        cycle_h = float(cfg["mean_cycle_min_estimate"]) / 60.0
        size = max(
            int(cfg["min_vehicles"]),
            int(np.ceil(self.cfg.demand["ridehail"] * cycle_h * float(cfg["slack_factor"]))),
        )
        fleet = []
        edges = list(self.files.edge_by_link.values())
        for i in range(size):
            veh = f"taxi_{i}"
            edge = edges[int(self.rng.integers(len(edges)))]
            rid = f"rt_{i}"
            try:
                traci.route.add(rid, [edge])
                traci.vehicle.add(veh, rid, typeID="ridehail", line="taxi")
                fleet.append(veh)
            except traci.TraCIException:  # pragma: no cover
                continue
        self.dispatcher = TaxiDispatcher(fleet)

    def _submit_person(self, index: int, now_s: float) -> None:
        traci = self._traci
        pickup_node = self._sample_destination("ridehail")
        dropoff_node = self._sample_destination("ridehail")
        p_edges, d_edges = self._edges_into(pickup_node), self._edges_into(dropoff_node)
        if not p_edges or not d_edges:
            return
        from_edge = p_edges[int(self.rng.integers(len(p_edges)))]
        to_edge = d_edges[int(self.rng.integers(len(d_edges)))]
        if from_edge == to_edge:
            return
        pid = f"person_{index}"
        try:
            traci.person.add(pid, from_edge, pos=20.0, depart=now_s)
            traci.person.appendDrivingStage(pid, to_edge, lines="taxi")
            traci.person.setColor(pid, (255, 200, 0, 255))
        except traci.TraCIException:  # pragma: no cover
            pass

    # -- reporting --------------------------------------------------------------
    def summarise(self) -> dict[str, Any]:
        rows = self.finished + list(self.trips.values())
        out: dict[str, Any] = {
            "scenario": self.cfg.scenario,
            "seed": self.seed,
            "backend": "sumo",
            "n_vehicles": len(rows),
            "illegal_parking_events": self.illegal_events,
        }
        for cls in VEHICLE_CLASSES:
            sub = [t for t in rows if t.vehicle_class == cls]
            parked = [t for t in sub if t.outcome == "parked"]
            out[f"{cls}_vehicles"] = len(sub)
            out[f"{cls}_parked_rate"] = len(parked) / len(sub) if sub else 0.0
            out[f"{cls}_illegal_rate"] = (
                sum(1 for t in sub if t.outcome == "illegal") / len(sub) if sub else 0.0
            )
            out[f"{cls}_search_time_min"] = (
                float(np.mean([t.search_time_min for t in parked])) if parked else 0.0
            )
            out[f"{cls}_failed_attempts"] = (
                float(np.mean([t.failed_attempts for t in sub])) if sub else 0.0
            )
        out["illegal_parking_rate"] = self.illegal_events / len(rows) if rows else 0.0
        if self.dispatcher is not None:
            d = self.dispatcher.summary()
            out["ridehail_dispatch"] = d
            # Ridehail is served through SUMO's taxi device, so its trips do not
            # appear in the per-vehicle table above; fold the dispatch record in
            # so the SUMO summary is comparable with the SimPy one.
            out["ridehail_vehicles"] = d["dispatched"]
            out["ridehail_wait_min"] = d["mean_wait_min"]
            out["ridehail_served_rate"] = (
                d["completed_pickups"] / d["dispatched"] if d["dispatched"] else 0.0
            )
        return out


def run_sumo_simulation(
    scenario: str = "baseline",
    seed: int = 1,
    horizon_min: float | None = None,
    work_dir: Path | str | None = None,
    gui: bool = False,
    use_taxi_device: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """Convenience wrapper: build the network, run one replication, return metrics."""
    overrides: dict[str, Any] = {}
    if horizon_min is not None:
        overrides["horizon_min"] = float(horizon_min)
        overrides["warmup_min"] = min(float(horizon_min) / 3.0, 30.0)
    cfg = load_scenario(scenario, seed=seed, overrides=overrides or None)
    work = Path(work_dir) if work_dir is not None else RESULTS_DIR / "sumo"
    backend = SumoBackend(cfg, work, gui=gui, use_taxi_device=use_taxi_device)
    backend.start()
    try:
        return backend.run(verbose=verbose)
    finally:
        backend.close()


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m src.sumo.backend",
        description="Run the district in SUMO with Python-side curb decisions.",
    )
    p.add_argument("--scenario", default="baseline")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--horizon", type=float, default=60.0, help="simulated minutes")
    p.add_argument("--gui", action="store_true")
    p.add_argument("--no-taxi-device", action="store_true")
    p.add_argument("--out", default=str(RESULTS_DIR / "sumo"))
    args = p.parse_args(argv)

    summary = run_sumo_simulation(
        scenario=args.scenario,
        seed=args.seed,
        horizon_min=args.horizon,
        work_dir=Path(args.out),
        gui=args.gui,
        use_taxi_device=not args.no_taxi_device,
        verbose=True,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"sumo_summary_seed{args.seed}.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    print("\nSUMO backend summary")
    print("-" * 52)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k:32s} {v:10.3f}")
        elif not isinstance(v, dict):
            print(f"{k:32s} {v!s:>10}")
    if "ridehail_dispatch" in summary:
        print("\nridehail dispatch:", json.dumps(summary["ridehail_dispatch"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
