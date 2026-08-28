"""Ridehail (TNC) vehicles: short dwell, no tolerance for walking, and a fleet.

Ridehail is modelled at the *fleet* level rather than as independent trips,
because the quantity a TNC actually competes for is a curb stop of 30-180 s at a
precise location, and the cost of failing to get one falls on a waiting
passenger, not on the driver's schedule. Modelling the fleet also makes the
supply side explicit: a demand shock scenario must not silently improve service
quality by conjuring extra vehicles, so fleet size is derived from demand once,
at construction time, and then held fixed.

Life cycle of one vehicle:

    idle -> assigned -> deadhead to pickup -> secure curb (or circle, or
    double-park) -> board -> carry passenger -> drop off -> idle

The fallback ladder at the pickup is the behaviourally interesting part: circle
the block (pure deadweight VMT), then double-park (externality imposed on
everyone else on the link).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import simpy

from src.agents.base_agent import BaseAgent, SearchOutcome


@dataclass
class RidehailRequest:
    """A passenger request waiting to be served."""

    request_id: int
    pickup: str
    dropoff: str
    t_request_min: float
    assigned_event: simpy.Event | None = None
    t_assigned_min: float | None = None


class RidehailFleet:
    """Dispatcher and vehicle pool.

    Dispatch rule is nearest-idle-vehicle by free-flow travel time, with
    first-come-first-served queueing when no vehicle is idle. This is the
    standard myopic baseline; swapping in batched matching would only require
    replacing :meth:`_try_dispatch`.
    """

    def __init__(self, world, arrivals_per_hour: float) -> None:
        self.world = world
        self.env = world.env
        cfg = world.agent_params["ridehail"]["fleet"]
        cycle_h = float(cfg["mean_cycle_min_estimate"]) / 60.0
        size = math.ceil(arrivals_per_hour * cycle_h * float(cfg["slack_factor"]))
        self.size = max(int(cfg["min_vehicles"]), size)

        self.idle: list[RidehailVehicle] = []
        self.pending: list[RidehailRequest] = []
        self.vehicles: list[RidehailVehicle] = []
        self.assignments: dict[int, RidehailVehicle] = {}
        self._next_request_id = 0

        for _ in range(self.size):
            v = RidehailVehicle(world, self, world.sample_node_uniform())
            self.vehicles.append(v)
            self.idle.append(v)
            self.env.process(v.run())

    # -- request intake ---------------------------------------------------------
    def submit(self, pickup: str, dropoff: str) -> RidehailRequest:
        self._next_request_id += 1
        req = RidehailRequest(
            request_id=self._next_request_id,
            pickup=pickup,
            dropoff=dropoff,
            t_request_min=self.world.now,
            assigned_event=self.env.event(),
        )
        self.pending.append(req)
        self._try_dispatch()
        return req

    def release(self, vehicle: RidehailVehicle) -> None:
        """Return a vehicle to the idle pool and immediately re-dispatch."""
        self.idle.append(vehicle)
        self._try_dispatch()

    def _try_dispatch(self) -> None:
        while self.pending and self.idle:
            req = self.pending[0]
            best = min(
                self.idle,
                key=lambda v: self.world.router.free_flow_time_s(v.node, req.pickup),
            )
            self.idle.remove(best)
            self.pending.pop(0)
            req.t_assigned_min = self.world.now
            self.assignments[req.request_id] = best
            best.assign(req)

    def assigned_vehicle(self, req: RidehailRequest) -> RidehailVehicle | None:
        """Which vehicle served ``req``, if it has been dispatched yet."""
        return self.assignments.get(req.request_id)

    # -- statistics -------------------------------------------------------------
    @property
    def utilisation(self) -> float:
        return 1.0 - len(self.idle) / self.size if self.size else 0.0

    @property
    def queue_length(self) -> int:
        return len(self.pending)


class RidehailVehicle(BaseAgent):
    """One TNC vehicle. Long-lived: it serves many requests over a run."""

    vehicle_class = "ridehail"

    def __init__(self, world, fleet: RidehailFleet, start_node: str) -> None:
        # A fleet vehicle has no single destination; the base record is replaced
        # per trip in `_serve`.
        super().__init__(world, start_node, world.now)
        self.fleet = fleet
        self.node = start_node
        self._assignment: simpy.Event = self.env.event()
        self._request: RidehailRequest | None = None
        self.trips_served = 0

    def assign(self, req: RidehailRequest) -> None:
        self._request = req
        if not self._assignment.triggered:
            self._assignment.succeed()

    def max_walk_m(self) -> float:
        return float(self.p["max_pickup_distance_m"])

    def eligible_classes(self) -> tuple[str, ...]:
        """Dedicated TNC space first, then any open metered stall.

        This is the mechanism that couples ridehail to passenger parking: when
        dedicated pickup zones are scarce, TNC vehicles compete for the same
        metered stalls as parked cars, and when *those* are full too they
        double-park. Restricting TNCs to dedicated zones only would make the
        two markets independent and hide the cross-modal effect the study is
        about.
        """
        return ("ridehail", "passenger")

    def expected_dwell_hours(self) -> float:
        return float(self.p["mean_dwell_s"]) / 3600.0

    def run(self):
        while True:
            yield self._assignment
            self._assignment = self.env.event()
            req = self._request
            self._request = None
            if req is None:  # pragma: no cover - defensive
                continue
            yield from self._serve(req)
            self.fleet.release(self)

    def _serve(self, req: RidehailRequest):
        world = self.world
        # Fresh trip record for this assignment.
        self.destination = req.pickup
        self.record = _new_record(world, req)
        self.segment = None

        # 1. Deadhead to the pickup point.
        pickup_dist = yield from world.drive(self.node, req.pickup, self.vehicle_class)
        self.node = req.pickup
        self.record.pickup_distance_m = float(pickup_dist)

        # 2. Secure a curb stop, or fall back.
        seg, reg, walk = yield from self.search_for_curb(req.pickup, self.eligible_classes())
        circled = 0.0
        if seg is None:
            seg, reg, walk, circled = yield from self._circle_or_double_park(req.pickup)
        self.record.circling_time_min = circled

        dwell_min = world.lognormal(float(self.p["mean_dwell_s"]) / 60.0, float(self.p["dwell_cv"]))
        if seg is not None:
            self.record.curb_id = seg.id
            self.record.walk_distance_m = walk
            self.record.outcome = SearchOutcome.PARKED
            self.pay_meter(seg, dwell_min, weight=float(self.p["beta_price"]))
            # The passenger walks to the vehicle while it waits at the curb.
            board_min = max(dwell_min, world.walk_time_min(walk))
            yield self.env.timeout(board_min)
            seg.release(reg, world.now)
        else:
            self.double_park(req.pickup, dwell_min)
            yield self.env.timeout(dwell_min)
            self.end_double_park()

        # 3. Passenger has boarded: the wait clock stops here.
        self.record.wait_time_min = world.now - req.t_request_min

        # 4. Carry the passenger and drop off.
        yield from world.drive(req.pickup, req.dropoff, self.vehicle_class)
        self.node = req.dropoff
        yield from self._dropoff(req.dropoff)
        self.trips_served += 1
        self.record.t_resolved_min = world.now
        self.record.warmup = self.record.t_arrive_min < world.warmup_min
        world.metrics.add_trip(self.record)

    def _circle_or_double_park(self, node: str):
        """Fallback ladder at the pickup: circle the block, then double-park."""
        world = self.world
        circled_min = 0.0
        if self.rng.random() < float(self.p["circling_probability"]):
            loops = int(self.p["max_circling_loops"])
            loop_min = float(self.p["circling_loop_s"]) / 60.0
            for _ in range(loops):
                yield self.env.timeout(loop_min)
                circled_min += loop_min
                world.metrics.count("ridehail_circling_loop")
                # Circling a block is roughly four block faces of deadweight VMT.
                if not world.in_warmup:
                    world.metrics.add_vmt(
                        self.vehicle_class,
                        4.0 * float(world.cfg.network["network"]["block_length_m"]),
                    )
                self.record.search_distance_m += 4.0 * float(
                    world.cfg.network["network"]["block_length_m"]
                )
                seg, reg, walk = yield from self.search_for_curb(
                    node, self.eligible_classes(), max_attempts=2, patience_min=0.6
                )
                if seg is not None:
                    self.record.search_time_min += circled_min
                    return seg, reg, walk, circled_min
        self.record.search_time_min += circled_min
        return None, None, 0.0, circled_min

    def _dropoff(self, node: str):
        """A drop-off is a shorter, less discretionary stop than a pickup."""
        world = self.world
        # The acceptable-walk filter in `rank_candidates` is measured from
        # `self.destination`, so it must be moved to the drop-off point before
        # searching - otherwise every candidate is rejected for being too far
        # from the (now irrelevant) pickup and the vehicle always double-parks.
        self.destination = node
        dwell = float(self.p["dropoff_dwell_s"]) / 60.0
        seg, reg, _walk = yield from self.search_for_curb(
            node,
            self.eligible_classes(),
            max_attempts=int(self.p["dropoff_search_attempts"]),
            patience_min=0.5,
        )
        if seg is not None:
            yield self.env.timeout(dwell)
            seg.release(reg, world.now)
        else:
            self.double_park(node, dwell, set_outcome=False, event="illegal_ridehail_dropoff")
            yield self.env.timeout(dwell)
            self.end_double_park()


def _new_record(world, req: RidehailRequest):
    from src.simulation.environment import TripRecord

    return TripRecord(
        agent_id=req.request_id,
        vehicle_class="ridehail",
        t_arrive_min=req.t_request_min,
        warmup=req.t_request_min < world.warmup_min,
    )
