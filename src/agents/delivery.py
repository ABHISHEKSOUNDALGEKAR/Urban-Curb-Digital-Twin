"""Delivery vehicle: the proximity-constrained, time-pressured parker.

Freight differs from passenger traffic in three ways that matter for the curb:

* **Distance aversion.** A courier carrying goods will not walk 300 m; the
  acceptable walk is tens of metres, so a loading zone two blocks away is not a
  substitute for one outside the door.
* **Expensive time.** Driver wages and delivery-window penalties make cruising
  far costlier per minute than it is for a commuter.
* **Willingness to double-park.** When no loading zone is available, the
  operationally rational choice is usually to stop in the travel lane and accept
  the citation risk. That behaviour is the main link between freight curb
  scarcity and general traffic delay.

A commercial vehicle may legally use a metered general stall, so its choice set
is loading zones first, then metered stalls.
"""

from __future__ import annotations

from src.agents.base_agent import BaseAgent, SearchOutcome


class DeliveryVehicle(BaseAgent):
    vehicle_class = "delivery"

    def __init__(self, world, destination: str, t_arrive_min: float) -> None:
        super().__init__(world, destination, t_arrive_min)
        self.service_min = world.lognormal(
            float(self.p["mean_service_min"]), float(self.p["service_cv"])
        )
        self.record.dwell_min = self.service_min

    def expected_dwell_hours(self) -> float:
        return self.service_min / 60.0

    def eligible_classes(self) -> tuple[str, ...]:
        return ("delivery", "passenger")

    def run(self):
        world = self.world
        entry = world.sample_node_uniform()
        yield from world.drive(entry, self.destination, self.vehicle_class)

        # Loading zones first; a metered stall is an acceptable second choice.
        seg, reg, walk = yield from self.search_for_curb(
            self.destination, ("delivery", "passenger")
        )

        if seg is None:
            if self.rng.random() < float(self.p["illegal_parking_probability"]):
                yield from self._double_park_and_serve()
                return
            # Compliant courier: widen the search, accepting a long walk.
            seg, reg, walk = yield from self.search_district_wide(
                self.destination, ("delivery", "passenger")
            )
            if seg is None:
                yield from self._double_park_and_serve()
                return
            outcome = SearchOutcome.DIVERTED
        else:
            outcome = SearchOutcome.PARKED

        # Commercial vehicles pay the meter at the posted rate.
        self.pay_meter(seg, self.service_min, weight=float(self.p["beta_price"]))
        self.record.curb_id = seg.id
        self.record.walk_distance_m = 2.0 * walk
        # Delivery delay: everything on top of the service itself — cruising,
        # plus the round trip on foot from the stall to the door.
        walk_min = world.walk_time_min(walk)
        self.record.service_delay_min = self.record.search_time_min + 2.0 * walk_min
        yield self.env.timeout(self.service_min + 2.0 * walk_min)
        seg.release(reg, world.now)
        yield from world.drive(self.destination, entry, self.vehicle_class)
        self.finish(outcome)

    def _double_park_and_serve(self):
        world = self.world
        self.double_park(self.destination, self.service_min)
        self.record.service_delay_min = self.record.search_time_min
        self.record.walk_distance_m = 0.0
        yield self.env.timeout(self.service_min)
        self.end_double_park()
        yield from world.drive(self.destination, world.sample_node_uniform(), self.vehicle_class)
        self.finish(SearchOutcome.ILLEGAL)
