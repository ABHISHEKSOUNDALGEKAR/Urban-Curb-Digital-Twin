"""The discrete-event engine: demand generation, control processes, run entry point.

``run_simulation(cfg)`` is the single function the rest of the codebase calls.
It is deliberately pure: same :class:`~src.config.RunConfig` in, same raw output
out, no filesystem or global state touched. Everything downstream —
experiments, calibration, optimization — is built on repeated calls to it.

Demand
------
Arrivals are a non-homogeneous Poisson process. The hourly rate for each vehicle
class is scaled by a piecewise-linear time-of-day profile and realised by
thinning (accept/reject against the profile maximum), which keeps the process
exactly Poisson rather than approximately so.

Warm-up
-------
The first ``warmup_min`` minutes fill the curb from an empty state, which is not
a state any real district is ever in. Statistics are reset at the end of warm-up
and trips that *arrived* during warm-up are excluded from the reported metrics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import simpy

from src.agents.delivery import DeliveryVehicle
from src.agents.passenger import PassengerCar
from src.agents.ridehail import RidehailFleet
from src.config import RunConfig
from src.simulation.environment import CurbWorld

SAMPLE_INTERVAL_MIN = 5.0


@dataclass
class SimulationResult:
    """Raw output of one replication. Aggregation happens elsewhere."""

    cfg: RunConfig
    trips: list[dict[str, Any]]
    occupancy_samples: list[dict[str, Any]]
    network_samples: list[dict[str, Any]]
    events: dict[str, int]
    vmt_m: dict[str, float]
    revenue_usd: float
    fines_usd: float
    capacity_by_class: dict[str, int]
    total_stalls: int
    fleet_size: int
    price_history: list[dict[str, Any]] = field(default_factory=list)
    wall_time_s: float = 0.0

    @property
    def seed(self) -> int:
        return self.cfg.seed

    @property
    def scenario(self) -> str:
        return self.cfg.scenario


def profile_multiplier(profile: list[float], t_min: float, horizon_min: float) -> float:
    """Piecewise-linear interpolation of a per-hour demand profile."""
    if not profile:
        return 1.0
    if len(profile) == 1:
        return float(profile[0])
    frac = min(max(t_min / horizon_min, 0.0), 1.0)
    pos = frac * (len(profile) - 1)
    lo = int(np.floor(pos))
    hi = min(lo + 1, len(profile) - 1)
    w = pos - lo
    return float(profile[lo] * (1 - w) + profile[hi] * w)


def _arrival_process(world: CurbWorld, vehicle_class: str, rate_per_hour: float, spawn):
    """Non-homogeneous Poisson arrivals by thinning."""
    if rate_per_hour <= 0:
        return
    profile = world.cfg.scenario_spec.get("demand_profile") or [1.0]
    horizon = world.horizon_min
    peak = max(profile) if profile else 1.0
    lam_max = rate_per_hour * peak / 60.0  # per minute
    while True:
        gap = world.rng.exponential(1.0 / lam_max)
        yield world.env.timeout(gap)
        if world.env.now >= horizon:
            return
        lam_t = rate_per_hour * profile_multiplier(profile, world.env.now, horizon) / 60.0
        if world.rng.random() <= lam_t / lam_max:
            spawn()


def _pricing_process(world: CurbWorld):
    interval = world.pricing.review_interval_min
    while True:
        yield world.env.timeout(interval)
        world.pricing.review(world.inventory, world.now)


def _regulation_process(world: CurbWorld, schedule: list[dict]):
    """Apply a time-of-day regulation schedule to the whole inventory."""
    for window in sorted(schedule, key=lambda w: float(w["from_min"])):
        start = float(window["from_min"])
        if world.env.now < start:
            yield world.env.timeout(start - world.env.now)
        world.inventory.set_allocation(window["allocation"], world.now)
        world.metrics.count("regulation_change")


def _sampler_process(world: CurbWorld):
    """Snapshot the system on a fixed interval, between warm-up and the horizon.

    Sampling must stop at the horizon: demand generation stops there and the
    model then drains, so including the drain-down would bias every occupancy
    statistic toward zero.
    """
    while world.env.now < world.horizon_min:
        yield world.env.timeout(SAMPLE_INTERVAL_MIN)
        if world.in_warmup or world.env.now > world.horizon_min:
            continue
        world.metrics.sample_occupancy(world.now, world.inventory)
        world.metrics.sample_network(world.now, world.router)


def _warmup_process(world: CurbWorld):
    if world.warmup_min <= 0:
        return
    yield world.env.timeout(world.warmup_min)
    world.inventory.reset_statistics(world.now)
    world.metrics.events.clear()
    world.metrics.vmt_m.clear()
    world.metrics.revenue_usd = 0.0
    world.metrics.fines_usd = 0.0


def run_simulation(cfg: RunConfig) -> SimulationResult:
    """Run one replication and return its raw output."""
    t_wall = time.perf_counter()
    env = simpy.Environment()
    world = CurbWorld(cfg, env)
    demand = cfg.demand

    fleet = RidehailFleet(world, demand["ridehail"])

    def spawn_passenger() -> None:
        agent = PassengerCar(world, world.sample_destination("passenger"), world.now)
        env.process(agent.run())

    def spawn_delivery() -> None:
        agent = DeliveryVehicle(world, world.sample_destination("delivery"), world.now)
        env.process(agent.run())

    def spawn_ridehail_request() -> None:
        pickup = world.sample_destination("ridehail")
        dropoff = world.sample_destination("ridehail")
        if dropoff == pickup:
            dropoff = world.sample_node_uniform()
        fleet.submit(pickup, dropoff)

    env.process(_arrival_process(world, "passenger", demand["passenger"], spawn_passenger))
    env.process(_arrival_process(world, "delivery", demand["delivery"], spawn_delivery))
    env.process(_arrival_process(world, "ridehail", demand["ridehail"], spawn_ridehail_request))
    env.process(_warmup_process(world))
    env.process(_sampler_process(world))
    if world.pricing.is_dynamic:
        env.process(_pricing_process(world))
    schedule = cfg.scenario_spec.get("time_of_day_regulation")
    if schedule:
        env.process(_regulation_process(world, schedule))

    # Run past the horizon so in-flight trips can resolve, but stop generating
    # new demand at the horizon (handled inside `_arrival_process`). The tail is
    # capped so a pathological saturation case cannot run forever.
    env.run(until=cfg.horizon_min + _tail_minutes(cfg))

    m = world.metrics
    return SimulationResult(
        cfg=cfg,
        trips=[t.as_dict() for t in m.trips],
        occupancy_samples=m.occupancy_samples,
        network_samples=m.network_samples,
        events=dict(m.events),
        vmt_m=dict(m.vmt_m),
        revenue_usd=m.revenue_usd,
        fines_usd=m.fines_usd,
        capacity_by_class=world.inventory.capacity_by_class(),
        total_stalls=world.inventory.total_stalls,
        fleet_size=fleet.size,
        price_history=world.pricing.history,
        wall_time_s=time.perf_counter() - t_wall,
    )


def _tail_minutes(cfg: RunConfig) -> float:
    """Cool-down long enough for the longest plausible dwell to finish."""
    mean_dwell = float(cfg.agents["passenger"]["mean_dwell_min"])
    return min(180.0, 3.0 * mean_dwell)
