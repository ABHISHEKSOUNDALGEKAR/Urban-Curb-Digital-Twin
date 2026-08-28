"""Curb inventory: segments, stalls, regulation types and pricing.

A *curb segment* is one face of one block: the atomic unit of curb regulation.
Each segment holds a small number of stalls, and every stall carries a
regulation type (``passenger``, ``delivery`` or ``ridehail``). Vehicles compete
for stalls of the type they are entitled to use.

Why not ``simpy.Resource``
--------------------------
SimPy's ``Resource`` is a queueing primitive: a request that cannot be served
waits. Curb space does not work that way — a driver who finds a block full
*balks* and drives to the next block, and that cruising is the phenomenon the
whole model exists to measure. On top of that, the time-of-day scenario changes
a segment's regulation mix mid-run, which ``Resource`` capacity is not designed
to support. So occupancy is tracked explicitly here and the SimPy environment is
used for what it is good at: process scheduling and the event clock.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from src.config import VEHICLE_CLASSES, ConfigError, validate_allocation

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass


@dataclass
class Occupancy:
    """Occupancy counters for one regulation type on one segment."""

    capacity: int = 0
    occupied: int = 0

    @property
    def available(self) -> int:
        return max(0, self.capacity - self.occupied)

    @property
    def utilisation(self) -> float:
        return self.occupied / self.capacity if self.capacity else 0.0


class CurbSegment:
    """One face of one block of curb.

    Parameters
    ----------
    curb_id:
        Stable identifier, e.g. ``"C0102_N"``.
    link:
        ``(u, v)`` node pair of the street link this segment sits on.
    xy:
        Metric coordinates of the segment midpoint; used for walking distance.
    total_stalls:
        Physical stall count. The split of these stalls between regulation types
        is set by :meth:`set_allocation` and may change during a run.
    price_per_hour:
        Posted meter price for passenger stalls. Delivery and ridehail stalls
        inherit it but their agents weight price differently.
    time_limit_min:
        Posted time limit; dwell beyond it is recorded as an overstay.
    face_length_m:
        Length of the block face, used to place stalls along the kerb in SUMO.
    """

    __slots__ = (
        "id",
        "link",
        "xy",
        "total_stalls",
        "face_length_m",
        "price_per_hour",
        "base_price_per_hour",
        "time_limit_min",
        "pools",
        "_occupied_time",
        "_last_change_t",
        "_arrivals",
        "_rejections",
    )

    def __init__(
        self,
        curb_id: str,
        link: tuple[str, str],
        xy: tuple[float, float],
        total_stalls: int,
        price_per_hour: float,
        time_limit_min: float,
        face_length_m: float = 130.0,
    ) -> None:
        if total_stalls < 0:
            raise ConfigError(f"segment {curb_id}: total_stalls must be >= 0")
        self.id = curb_id
        self.link = link
        self.xy = xy
        self.total_stalls = int(total_stalls)
        # Physical length of the block face, needed by the SUMO backend to lay
        # parking areas out along the kerb.
        self.face_length_m = float(face_length_m)
        self.price_per_hour = float(price_per_hour)
        self.base_price_per_hour = float(price_per_hour)
        self.time_limit_min = float(time_limit_min)
        self.pools: dict[str, Occupancy] = {c: Occupancy() for c in VEHICLE_CLASSES}
        # Time-weighted occupancy accumulators, for occupancy statistics.
        self._occupied_time: dict[str, float] = {c: 0.0 for c in VEHICLE_CLASSES}
        self._last_change_t: float = 0.0
        self._arrivals: dict[str, int] = {c: 0 for c in VEHICLE_CLASSES}
        self._rejections: dict[str, int] = {c: 0 for c in VEHICLE_CLASSES}

    # -- capacity ---------------------------------------------------------------
    def set_allocation(self, allocation: dict[str, float], now: float = 0.0) -> None:
        """(Re)assign stalls to regulation types using largest-remainder rounding.

        Stalls that are currently occupied are never taken away from a parked
        vehicle: a re-allocation can leave a pool temporarily over-subscribed
        (``occupied > capacity``), which drains as those vehicles depart. This is
        the honest representation of a mid-day regulation change.
        """
        allocation = validate_allocation(allocation)
        self._accrue(now)
        counts = largest_remainder([allocation[c] for c in VEHICLE_CLASSES], self.total_stalls)
        for cls, n in zip(VEHICLE_CLASSES, counts, strict=True):
            self.pools[cls].capacity = int(n)

    def capacity(self, vehicle_class: str) -> int:
        return self.pools[vehicle_class].capacity

    def available(self, vehicle_class: str) -> int:
        return self.pools[vehicle_class].available

    def has_space(self, vehicle_class: str) -> bool:
        return self.pools[vehicle_class].available > 0

    @property
    def occupied_total(self) -> int:
        return sum(p.occupied for p in self.pools.values())

    def occupancy_rate(self, vehicle_class: str | None = None) -> float:
        """Occupancy as a fraction of capacity, clipped to [0, 1]."""
        if vehicle_class is None:
            if self.total_stalls == 0:
                return 0.0
            return min(1.0, self.occupied_total / self.total_stalls)
        pool = self.pools[vehicle_class]
        if pool.capacity == 0:
            return 1.0 if pool.occupied else 0.0
        return min(1.0, pool.occupied / pool.capacity)

    # -- occupancy transitions --------------------------------------------------
    def occupy(self, vehicle_class: str, now: float) -> bool:
        """Take one stall of ``vehicle_class``. Returns False if none is free."""
        pool = self.pools[vehicle_class]
        self._arrivals[vehicle_class] += 1
        if pool.available <= 0:
            self._rejections[vehicle_class] += 1
            return False
        self._accrue(now)
        pool.occupied += 1
        return True

    def release(self, vehicle_class: str, now: float) -> None:
        pool = self.pools[vehicle_class]
        if pool.occupied <= 0:
            raise RuntimeError(f"segment {self.id}: release of {vehicle_class} with zero occupancy")
        self._accrue(now)
        pool.occupied -= 1

    def _accrue(self, now: float) -> None:
        dt = now - self._last_change_t
        if dt > 0:
            for cls, pool in self.pools.items():
                self._occupied_time[cls] += pool.occupied * dt
        self._last_change_t = now

    # -- statistics -------------------------------------------------------------
    def time_weighted_occupancy(self, now: float, *, since: float = 0.0) -> float:
        """Mean occupancy over ``[since, now]`` as a fraction of total stalls."""
        self._accrue(now)
        span = now - since
        if span <= 0 or self.total_stalls == 0:
            return 0.0
        occupied_time = sum(self._occupied_time.values())
        return min(1.0, occupied_time / (self.total_stalls * span))

    @property
    def arrivals(self) -> dict[str, int]:
        return dict(self._arrivals)

    @property
    def rejections(self) -> dict[str, int]:
        return dict(self._rejections)

    def reset_statistics(self, now: float) -> None:
        """Discard accumulated statistics; used to drop the warm-up period."""
        self._accrue(now)
        self._occupied_time = {c: 0.0 for c in VEHICLE_CLASSES}
        self._arrivals = {c: 0 for c in VEHICLE_CLASSES}
        self._rejections = {c: 0 for c in VEHICLE_CLASSES}
        self._last_change_t = now

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        pools = ", ".join(
            f"{c}={self.pools[c].occupied}/{self.pools[c].capacity}" for c in VEHICLE_CLASSES
        )
        return f"<CurbSegment {self.id} link={self.link} {pools} ${self.price_per_hour:.2f}/h>"


def largest_remainder(shares: Iterable[float], total: int) -> list[int]:
    """Apportion ``total`` integer units across ``shares`` (Hamilton's method).

    Guarantees the result sums to exactly ``total``, which matters because a
    naive ``round`` can silently create or destroy curb capacity and quietly
    invalidate a whole experiment.
    """
    shares = list(shares)
    if total <= 0:
        return [0] * len(shares)
    ssum = sum(shares)
    if ssum <= 0:
        return [total] + [0] * (len(shares) - 1)
    exact = [s / ssum * total for s in shares]
    floors = [int(np.floor(e)) for e in exact]
    remainder = total - sum(floors)
    if remainder:
        order = np.argsort([-(e - f) for e, f in zip(exact, floors, strict=True)])
        for i in order[:remainder]:
            floors[int(i)] += 1
    return floors


@dataclass
class CurbInventory:
    """All curb segments in the district, with spatial and link indexes."""

    segments: list[CurbSegment] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._by_id = {s.id: s for s in self.segments}
        self._by_link: dict[tuple[str, str], list[CurbSegment]] = {}
        for s in self.segments:
            self._by_link.setdefault(s.link, []).append(s)
        self._xy = (
            np.array([s.xy for s in self.segments], dtype=float)
            if self.segments
            else np.zeros((0, 2))
        )

    def __len__(self) -> int:
        return len(self.segments)

    def __iter__(self):
        return iter(self.segments)

    def __getitem__(self, curb_id: str) -> CurbSegment:
        return self._by_id[curb_id]

    def get(self, curb_id: str) -> CurbSegment | None:
        return self._by_id.get(curb_id)

    def on_link(self, link: tuple[str, str]) -> list[CurbSegment]:
        return self._by_link.get(link, [])

    @property
    def total_stalls(self) -> int:
        return sum(s.total_stalls for s in self.segments)

    def capacity_by_class(self) -> dict[str, int]:
        return {c: sum(s.capacity(c) for s in self.segments) for c in VEHICLE_CLASSES}

    def occupancy_by_class(self) -> dict[str, float]:
        out = {}
        for c in VEHICLE_CLASSES:
            cap = sum(s.capacity(c) for s in self.segments)
            occ = sum(s.pools[c].occupied for s in self.segments)
            out[c] = occ / cap if cap else 0.0
        return out

    def set_allocation(self, allocation: dict[str, float], now: float = 0.0) -> None:
        """Apportion stalls to regulation types so *district* shares are exact.

        Rounding each segment independently is the obvious implementation and it
        is wrong: with ~12 stalls per face, a segment can only express shares in
        steps of about 8%, so district-wide allocations that differ by less than
        that collapse onto the same physical inventory. That quantisation makes
        the optimizer's objective surface flat in places and jumpy in others, for
        reasons that have nothing to do with curb policy.

        So: apportion at the district level first (largest remainder over the
        total stall count), lay that down segment by segment, then repair the
        residual by moving single stalls between classes on segments that can
        give one up. Per-segment totals are preserved exactly, and district
        totals hit the requested shares to within one stall.
        """
        allocation = validate_allocation(allocation)
        total = self.total_stalls
        targets = dict(
            zip(
                VEHICLE_CLASSES,
                largest_remainder([allocation[c] for c in VEHICLE_CLASSES], total),
                strict=True,
            )
        )
        for seg in self.segments:
            seg.set_allocation(allocation, now)

        counts = {c: sum(s.pools[c].capacity for s in self.segments) for c in VEHICLE_CLASSES}
        # Repair: move one stall at a time from an over-supplied class to an
        # under-supplied one, preferring donors that are not currently occupied.
        for _ in range(4 * total):
            over = [c for c in VEHICLE_CLASSES if counts[c] > targets[c]]
            under = [c for c in VEHICLE_CLASSES if counts[c] < targets[c]]
            if not over or not under:
                break
            donor, receiver = over[0], under[0]
            seg = self._pick_donor_segment(donor)
            if seg is None:
                break
            seg.pools[donor].capacity -= 1
            seg.pools[receiver].capacity += 1
            counts[donor] -= 1
            counts[receiver] += 1

    def _pick_donor_segment(self, donor: str) -> CurbSegment | None:
        """A segment that can give up one stall of ``donor``, free stalls first."""
        free = [s for s in self.segments if s.pools[donor].available > 0]
        if free:
            return max(free, key=lambda s: s.pools[donor].capacity)
        occupied = [s for s in self.segments if s.pools[donor].capacity > 0]
        return max(occupied, key=lambda s: s.pools[donor].capacity) if occupied else None

    def reset_statistics(self, now: float) -> None:
        for s in self.segments:
            s.reset_statistics(now)

    def within(self, xy: tuple[float, float], radius_m: float) -> list[CurbSegment]:
        """Segments whose midpoint is within ``radius_m`` (straight line) of ``xy``."""
        if not self.segments:
            return []
        d = np.hypot(self._xy[:, 0] - xy[0], self._xy[:, 1] - xy[1])
        idx = np.nonzero(d <= radius_m)[0]
        return [self.segments[int(i)] for i in idx]

    def distances_from(self, xy: tuple[float, float]) -> np.ndarray:
        return np.hypot(self._xy[:, 0] - xy[0], self._xy[:, 1] - xy[1])


class PricingPolicy:
    """Meter pricing controller.

    ``static``  keeps posted prices fixed for the whole run.
    ``dynamic`` implements an SFpark-style occupancy-responsive rule: on a fixed
    review interval, prices on each segment step up if measured occupancy is
    above the target band and down if it is below, within posted bounds.
    """

    def __init__(self, policy: str, params: dict | None = None) -> None:
        self.policy = policy
        p = params or {}
        self.target_low = float(p.get("target_occupancy_low", 0.70))
        self.target_high = float(p.get("target_occupancy_high", 0.85))
        self.review_interval_min = float(p.get("review_interval_min", 30.0))
        self.step_up = float(p.get("step_up", 1.25))
        self.step_down = float(p.get("step_down", 0.85))
        self.min_price = float(p.get("min_price", 1.0))
        self.max_price = float(p.get("max_price", 12.0))
        self.history: list[dict] = []

    @property
    def is_dynamic(self) -> bool:
        return self.policy == "dynamic"

    def review(self, inventory: CurbInventory, now: float) -> None:
        """Adjust posted prices from occupancy observed since the last review."""
        if not self.is_dynamic:
            return
        for seg in inventory:
            occ = seg.time_weighted_occupancy(now, since=max(0.0, now - self.review_interval_min))
            old = seg.price_per_hour
            if occ > self.target_high:
                new = old * self.step_up
            elif occ < self.target_low:
                new = old * self.step_down
            else:
                new = old
            seg.price_per_hour = float(np.clip(new, self.min_price, self.max_price))
        self.history.append(
            {
                "t_min": now,
                "mean_price": float(np.mean([s.price_per_hour for s in inventory])),
                "mean_occupancy": float(np.mean([s.occupancy_rate() for s in inventory])),
            }
        )
