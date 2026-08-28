"""Shared agent scaffolding: curb evaluation, cruising search, fallback logic.

Every vehicle class implements the same three-stage life cycle —

    approach -> compete for curb -> resolve (park / park illegally / divert)

— but weights the terms of the curb choice differently and takes a different
fallback when legal curb space cannot be found. Those two differences are the
entire behavioural content of the model, so they are the only things subclasses
are expected to override.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from src.simulation.curb import CurbSegment
from src.simulation.environment import CurbWorld, TripRecord


def curb_choice_score(
    params: dict, walk_m: float, price: float, approach_min: float, occupancy: float
) -> float:
    """Generalised cost of a curb option; lower is better.

        cost = alpha*walk + beta*price + gamma*expected_search + delta*occupancy

    Pulled out as a free function so that both backends - the SimPy engine and
    the SUMO/TraCI backend - score curb options with literally the same code.
    A behavioural model that differs between backends is not one model.
    """
    return (
        float(params["alpha_walk_per_m"]) * float(walk_m)
        + float(params["beta_price"]) * float(price)
        + float(params["gamma_search_per_min"]) * float(approach_min)
        + float(params["delta_occupancy"]) * float(occupancy)
    )


class SearchOutcome:
    """Result codes for a curb search."""

    PARKED = "parked"
    ILLEGAL = "illegal"
    DIVERTED = "diverted"  # parked, but only after abandoning the target area
    ABANDONED = "abandoned"  # left the district without completing the activity


class BaseAgent:
    """Base class for a vehicle competing for curb space.

    Subclasses set :attr:`vehicle_class` and implement :meth:`run`, the SimPy
    process describing the trip.
    """

    vehicle_class: str = "base"

    def __init__(self, world: CurbWorld, destination: str, t_arrive_min: float) -> None:
        self.world = world
        self.env = world.env
        self.rng = world.rng
        self.id = world.next_agent_id()
        self.destination = destination
        self.p = world.agent_params[self.vehicle_class]
        self.common = world.common
        self.record = TripRecord(
            agent_id=self.id,
            vehicle_class=self.vehicle_class,
            t_arrive_min=t_arrive_min,
            warmup=world.in_warmup,
        )
        self.segment: CurbSegment | None = None
        self._double_parked_link: tuple[str, str] | None = None

    # -- curb choice ------------------------------------------------------------
    def eligible_classes(self) -> tuple[str, ...]:
        """Regulation types this vehicle may legally occupy, in preference order."""
        return (self.vehicle_class,)

    def expected_dwell_hours(self) -> float:
        """Planned dwell, used to convert a $/hour price into a trip cost."""
        return 1.0

    def score(
        self,
        seg: CurbSegment,
        walk_m: float,
        approach_min: float,
        regulation: str,
    ) -> float:
        """Generalised cost of choosing ``seg``; lower is better.

            cost = alpha*walk + beta*price*dwell + gamma*expected_search + delta*occupancy

        The occupancy term is an *expectation* formed from what the driver can
        see from the street (how full the block looks), not privileged knowledge
        of whether a specific stall is free.
        """
        return curb_choice_score(
            self.p,
            walk_m=walk_m,
            price=seg.price_per_hour * self.expected_dwell_hours(),
            approach_min=approach_min,
            occupancy=seg.occupancy_rate(regulation),
        )

    def rank_candidates(
        self,
        from_node: str,
        radius_m: float,
        regulations: Sequence[str],
        exclude: Iterable[str] = (),
        limit: int = 8,
    ) -> list[tuple[CurbSegment, float, str]]:
        """Rank nearby curb segments by generalised cost.

        Returns ``(segment, walking_distance_m, regulation)`` triples. A segment
        with zero capacity for every eligible regulation is not a candidate at
        all — an empty allocation removes it from the choice set rather than
        making it an always-full option.
        """
        excluded = set(exclude)
        dest_xy = self.world.router.xy(self.destination)
        scored: list[tuple[float, CurbSegment, float, str]] = []
        for seg, _dist_from_search_origin in self.world.candidate_segments(from_node, radius_m):
            if seg.id in excluded:
                continue
            walk = float(np.hypot(seg.xy[0] - dest_xy[0], seg.xy[1] - dest_xy[1]))
            if walk > float(self.max_walk_m()):
                continue
            access = self.world.segment_access_node(seg)
            approach_s = self.world.router.free_flow_time_s(from_node, access)
            approach_min = approach_s / 60.0 + float(self.common["scan_time_s"]) / 60.0
            for reg in regulations:
                if seg.capacity(reg) <= 0:
                    continue
                scored.append((self.score(seg, walk, approach_min, reg), seg, walk, reg))
                break  # take the highest-preference regulation this segment offers
        scored.sort(key=lambda t: t[0])
        return [(s, w, r) for _c, s, w, r in scored[:limit]]

    def max_walk_m(self) -> float:
        return float(self.p.get("max_walk_distance_m", self.p.get("max_pickup_distance_m", 300.0)))

    # -- search -----------------------------------------------------------------
    def search_for_curb(
        self,
        start_node: str,
        regulations: Sequence[str],
        radius_m: float | None = None,
        max_attempts: int | None = None,
        patience_min: float | None = None,
    ):
        """SimPy process: cruise from block to block looking for an open stall.

        Returns ``(segment, regulation, walking_distance_m)`` on success and
        ``(None, None, 0.0)`` if patience or attempts run out. Time spent here is
        charged to ``search_time_min`` and distance to ``search_distance_m``.
        """
        radius = float(radius_m if radius_m is not None else self.p["search_radius_m"])
        max_attempts = int(
            max_attempts if max_attempts is not None else self.common["max_search_attempts"]
        )
        patience = float(
            patience_min if patience_min is not None else self.p["search_patience_min"]
        )
        t0 = self.world.now
        node = start_node
        tried: set[str] = set()
        attempts = 0

        while attempts < max_attempts and (self.world.now - t0) < patience:
            candidates = self.rank_candidates(node, radius, regulations, exclude=tried)
            if not candidates:
                # Nothing left within the acceptable radius from here: widen once
                # by moving to the next block and re-scanning.
                if attempts == 0:
                    break
                break
            seg, walk, reg = candidates[0]
            access = self.world.segment_access_node(seg)
            dist = yield from self.world.cruise(node, access, self.vehicle_class)
            self.record.search_distance_m += float(dist)
            node = access
            # Slow-rolling past the face to read the signs and look for a gap.
            yield self.env.timeout(float(self.common["scan_time_s"]) / 60.0)
            attempts += 1
            tried.add(seg.id)
            if seg.occupy(reg, self.world.now):
                self.record.search_time_min += self.world.now - t0
                self.record.failed_attempts += attempts - 1
                self.segment = seg
                return seg, reg, walk
            self.world.metrics.count(f"failed_attempt_{self.vehicle_class}")

        self.record.search_time_min += self.world.now - t0
        self.record.failed_attempts += attempts
        return None, None, 0.0

    def search_district_wide(self, start_node: str, regulations: Sequence[str]):
        """Last-resort search: accept the nearest free stall anywhere in the district.

        This is the behavioural counterpart of "give up on this block and take
        whatever you can get", and is what a *compliant* driver does instead of
        parking illegally. It is deliberately unbounded in walking distance,
        and the resulting walk is recorded so the cost shows up in the results.
        """
        dest_xy = self.world.router.xy(self.destination)
        best: tuple[float, CurbSegment, str] | None = None
        for seg in self.world.inventory:
            for reg in regulations:
                if seg.capacity(reg) > 0 and seg.available(reg) > 0:
                    walk = float(np.hypot(seg.xy[0] - dest_xy[0], seg.xy[1] - dest_xy[1]))
                    if best is None or walk < best[0]:
                        best = (walk, seg, reg)
                    break
        if best is None:
            return None, None, 0.0
        walk, seg, reg = best
        access = self.world.segment_access_node(seg)
        dist = yield from self.world.cruise(start_node, access, self.vehicle_class)
        self.record.search_distance_m += float(dist)
        if seg.occupy(reg, self.world.now):
            self.segment = seg
            return seg, reg, walk
        return None, None, 0.0

    # -- illegal parking --------------------------------------------------------
    def double_park(
        self, node: str, duration_min: float, *, set_outcome: bool = True, event: str | None = None
    ) -> None:
        """Register a double-parking event on the link the vehicle is sitting on.

        The blockage is added to the router so that *other* vehicles' travel
        times rise: this is how a curb shortage propagates into a network cost.

        ``set_outcome=False`` is used for secondary stops (a ridehail drop-off
        after a successful pickup) that should be counted as illegal events but
        must not overwrite the outcome of the trip they belong to.
        """
        link = self._pick_link(node)
        self._double_parked_link = link
        self.world.router.add_double_park(link)
        if set_outcome:
            self.record.outcome = SearchOutcome.ILLEGAL
        self.world.metrics.count(event or f"illegal_{self.vehicle_class}")
        if self.rng.random() < float(self.p["enforcement_probability"]):
            self.record.cited = True
            self.record.fine_usd = float(self.p["fine_amount"])
            self.world.fine(self.record.fine_usd)
            self.world.metrics.count("citations")

    def end_double_park(self) -> None:
        if self._double_parked_link is not None:
            self.world.router.remove_double_park(self._double_parked_link)
            self._double_parked_link = None

    def _pick_link(self, node: str) -> tuple[str, str]:
        succ = list(self.world.graph.successors(node))
        if not succ:  # pragma: no cover - the grid is strongly connected
            return next(iter(self.world.graph.edges))
        return (node, succ[int(self.rng.integers(len(succ)))])

    # -- bookkeeping ------------------------------------------------------------
    def finish(self, outcome: str) -> None:
        self.record.outcome = outcome
        self.record.t_resolved_min = self.world.now
        self.record.warmup = self.record.t_arrive_min < self.world.warmup_min
        self.world.metrics.add_trip(self.record)

    def pay_meter(self, seg: CurbSegment, dwell_min: float, weight: float = 1.0) -> float:
        cost = seg.price_per_hour * (dwell_min / 60.0) * weight
        self.record.parking_cost_usd += cost
        self.world.charge(cost)
        return cost

    def run(self):  # pragma: no cover - abstract
        raise NotImplementedError
