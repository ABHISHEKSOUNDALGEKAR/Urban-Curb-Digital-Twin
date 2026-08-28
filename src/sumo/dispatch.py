"""Ridehail dispatch on top of SUMO's taxi device and reservation API.

SUMO models TNC service natively: a vehicle carrying the ``taxi`` device can be
handed *reservations* created by persons whose route contains a ``taxi`` stage,
and ``traci.vehicle.dispatchTaxi`` assigns one to a specific vehicle. That gives
realistic pickup/drop-off manoeuvres and occupancy handling for free, while
leaving the interesting decision - *which* vehicle serves *which* request - in
Python, where the policy belongs.

The matching rule here is nearest-idle-vehicle by network distance, the same
myopic baseline the SimPy backend uses, so the two are comparable. Anything
smarter (batched assignment, pooling) plugs in at :meth:`TaxiDispatcher.match`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import traci


@dataclass
class Reservation:
    """A pending request, as SUMO reports it."""

    id: str
    person_id: str
    from_edge: str
    to_edge: str
    reservation_time: float


@dataclass
class DispatchRecord:
    reservation_id: str
    vehicle_id: str
    t_request_s: float
    t_dispatch_s: float
    t_pickup_s: float | None = None
    t_dropoff_s: float | None = None

    @property
    def wait_time_s(self) -> float | None:
        return None if self.t_pickup_s is None else self.t_pickup_s - self.t_request_s


class TaxiDispatcher:
    """Assign SUMO taxi reservations to idle fleet vehicles."""

    #: Reservation state flag meaning "not yet assigned to a vehicle".
    STATE_NEW = 0

    def __init__(self, fleet: Sequence[str]) -> None:
        self.fleet = list(fleet)
        self.records: dict[str, DispatchRecord] = {}
        self.assigned: dict[str, str] = {}  # reservation id -> vehicle id
        self.unserved: set[str] = set()

    # -- fleet state ------------------------------------------------------------
    def idle_vehicles(self) -> list[str]:
        """Fleet vehicles that are in the network and carrying no passenger."""
        live = set(traci.vehicle.getIDList())
        out = []
        for v in self.fleet:
            if v not in live:
                continue
            state = traci.vehicle.getParameter(v, "device.taxi.state")
            # 0 = empty/idle. Anything else means en route to, or carrying, a fare.
            if state in ("", "0"):
                out.append(v)
        return out

    def pending_reservations(self) -> list[Reservation]:
        out = []
        for r in traci.person.getTaxiReservations(self.STATE_NEW):
            if r.id in self.assigned:
                continue
            out.append(
                Reservation(
                    id=r.id,
                    person_id=r.persons[0] if r.persons else "",
                    from_edge=r.fromEdge,
                    to_edge=r.toEdge,
                    reservation_time=float(r.reservationTime),
                )
            )
        return out

    # -- matching ---------------------------------------------------------------
    def match(self, reservation: Reservation, idle: Sequence[str]) -> str | None:
        """Nearest idle vehicle by driving distance to the pickup edge."""
        best, best_d = None, float("inf")
        for v in idle:
            try:
                d = traci.vehicle.getDrivingDistance(v, reservation.from_edge, 0.0)
            except traci.TraCIException:  # pragma: no cover - unreachable edge
                continue
            if d < 0:  # pickup is behind the vehicle; approximate with a re-route
                d = 1e6
            if d < best_d:
                best, best_d = v, d
        return best

    def step(self, now_s: float) -> int:
        """Dispatch as many pending reservations as there are idle vehicles."""
        self.poll_service_events(now_s)
        pending = self.pending_reservations()
        if not pending:
            return 0
        idle = self.idle_vehicles()
        dispatched = 0
        for res in sorted(pending, key=lambda r: r.reservation_time):
            if not idle:
                self.unserved.add(res.id)
                continue
            veh = self.match(res, idle)
            if veh is None:
                self.unserved.add(res.id)
                continue
            try:
                traci.vehicle.dispatchTaxi(veh, [res.id])
            except traci.TraCIException:  # pragma: no cover - race with SUMO state
                continue
            idle.remove(veh)
            self.assigned[res.id] = veh
            self.unserved.discard(res.id)
            self.records[res.id] = DispatchRecord(
                reservation_id=res.id,
                vehicle_id=veh,
                t_request_s=res.reservation_time,
                t_dispatch_s=now_s,
            )
            dispatched += 1
        return dispatched

    def poll_service_events(self, now_s: float) -> None:
        """Detect pickups and drop-offs from the fleet's passenger counts.

        SUMO does not push a "picked up" event, so the occupancy of each
        dispatched vehicle is polled instead: 0 -> >0 passengers is a pickup,
        and the return to 0 is the drop-off.
        """
        live = set(traci.vehicle.getIDList())
        for rec in self.records.values():
            if rec.vehicle_id not in live:
                if rec.t_pickup_s is not None and rec.t_dropoff_s is None:
                    rec.t_dropoff_s = now_s
                continue
            onboard = traci.vehicle.getPersonNumber(rec.vehicle_id)
            if rec.t_pickup_s is None and onboard > 0:
                rec.t_pickup_s = now_s
            elif rec.t_pickup_s is not None and rec.t_dropoff_s is None and onboard == 0:
                rec.t_dropoff_s = now_s

    def note_pickup(self, reservation_id: str, now_s: float) -> None:
        rec = self.records.get(reservation_id)
        if rec and rec.t_pickup_s is None:
            rec.t_pickup_s = now_s

    def note_dropoff(self, reservation_id: str, now_s: float) -> None:
        rec = self.records.get(reservation_id)
        if rec and rec.t_dropoff_s is None:
            rec.t_dropoff_s = now_s

    def summary(self) -> dict[str, Any]:
        waits = [r.wait_time_s for r in self.records.values() if r.wait_time_s is not None]
        rides = [
            r.t_dropoff_s - r.t_pickup_s
            for r in self.records.values()
            if r.t_dropoff_s is not None and r.t_pickup_s is not None
        ]
        return {
            "fleet_size": len(self.fleet),
            "dispatched": len(self.records),
            "completed_pickups": len(waits),
            "completed_rides": len(rides),
            "unserved": len(self.unserved),
            "mean_wait_s": float(sum(waits) / len(waits)) if waits else 0.0,
            "mean_wait_min": float(sum(waits) / len(waits) / 60.0) if waits else 0.0,
        }
