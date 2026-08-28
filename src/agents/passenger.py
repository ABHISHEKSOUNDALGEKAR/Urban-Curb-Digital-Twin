"""Passenger car: the discretionary parker.

A passenger driver arrives at a destination, cruises for a metered stall,
trades off walking distance against price and expected search time, and — if
the search fails — either accepts a much longer walk (compliant) or parks
illegally (non-compliant). Dwell is long (tens of minutes), so a passenger car
that succeeds removes a stall from the pool for a long time. That asymmetry
between long passenger dwell and short ridehail dwell is the central source of
curb competition in the model.
"""

from __future__ import annotations

from src.agents.base_agent import BaseAgent, SearchOutcome


class PassengerCar(BaseAgent):
    vehicle_class = "passenger"

    def __init__(self, world, destination: str, t_arrive_min: float) -> None:
        super().__init__(world, destination, t_arrive_min)
        self.dwell_min = world.lognormal(float(self.p["mean_dwell_min"]), float(self.p["dwell_cv"]))
        self.record.dwell_min = self.dwell_min

    def expected_dwell_hours(self) -> float:
        return self.dwell_min / 60.0

    def eligible_classes(self) -> tuple[str, ...]:
        return ("passenger",)

    def run(self):
        world = self.world
        # 1. Approach: enter the district and drive to the destination block.
        entry = world.sample_node_uniform()
        yield from world.drive(entry, self.destination, self.vehicle_class)

        # 2. Compete for a metered stall near the destination.
        seg, reg, walk = yield from self.search_for_curb(self.destination, ("passenger",))

        if seg is None:
            # 3. Fallback. Compliance is a behavioural parameter, not a rule.
            if self.rng.random() < float(self.p["compliance_probability"]):
                seg, reg, walk = yield from self.search_district_wide(
                    self.destination, ("passenger",)
                )
                if seg is None:
                    # The district is saturated: the trip is abandoned.
                    world.metrics.count("abandoned_passenger")
                    yield from world.drive(self.destination, entry, self.vehicle_class)
                    self.finish(SearchOutcome.ABANDONED)
                    return
                outcome = SearchOutcome.DIVERTED
            else:
                yield from self._park_illegally()
                return
        else:
            outcome = SearchOutcome.PARKED

        # 4. Park: pay, walk to the destination, dwell, walk back.
        self.pay_meter(seg, self.dwell_min)
        self.record.walk_distance_m = 2.0 * walk
        self.record.curb_id = seg.id
        walk_min = world.walk_time_min(walk)
        yield self.env.timeout(walk_min)
        yield self.env.timeout(max(0.0, self.dwell_min - 2 * walk_min))
        if self.dwell_min > seg.time_limit_min:
            world.metrics.count("overstay_passenger")
        yield self.env.timeout(walk_min)
        seg.release(reg, world.now)

        # 5. Leave the district.
        yield from world.drive(self.destination, entry, self.vehicle_class)
        self.finish(outcome)

    def _park_illegally(self):
        """Non-compliant fallback: park at a hydrant/red zone or double-park."""
        self.double_park(self.destination, self.dwell_min)
        self.record.walk_distance_m = 0.0
        yield self.env.timeout(self.dwell_min)
        self.end_double_park()
        self.finish(SearchOutcome.ILLEGAL)
