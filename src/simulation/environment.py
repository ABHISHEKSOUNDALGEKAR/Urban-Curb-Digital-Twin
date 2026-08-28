"""The simulated world: clock, network, curb inventory, demand geography, metrics.

:class:`CurbWorld` is the single object every agent process talks to. It owns
the SimPy environment (so "now" is always well defined), the road network and
router, the curb inventory, the pricing controller, the random number generator
and the metrics recorder.

Time is measured in **minutes** throughout the SimPy clock; distances in metres;
money in dollars. Travel-time helpers return seconds because that is how link
performance functions are conventionally written, and are converted at the call
site. The unit of every recorded metric is in its name.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import simpy

from src.config import VEHICLE_CLASSES, RunConfig
from src.simulation.curb import CurbInventory, CurbSegment, PricingPolicy
from src.simulation.routing import METRES_PER_MILE, Router, build_grid_network


@dataclass
class TripRecord:
    """One completed (or abandoned) vehicle trip."""

    agent_id: int
    vehicle_class: str
    t_arrive_min: float
    t_resolved_min: float = 0.0
    outcome: str = "unresolved"  # parked | illegal | diverted | abandoned
    search_time_min: float = 0.0
    search_distance_m: float = 0.0
    failed_attempts: int = 0
    walk_distance_m: float = 0.0
    parking_cost_usd: float = 0.0
    dwell_min: float = 0.0
    curb_id: str | None = None
    # class-specific
    wait_time_min: float = 0.0  # ridehail passenger wait (request -> boarding)
    pickup_distance_m: float = 0.0  # ridehail deadhead distance
    circling_time_min: float = 0.0
    service_delay_min: float = 0.0  # delivery: delay vs. an ideal loading zone
    cited: bool = False
    fine_usd: float = 0.0
    warmup: bool = False

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class MetricsRecorder:
    """Collects raw per-trip records and periodic system snapshots.

    Recording is deliberately dumb: it stores facts, never derived statistics.
    All aggregation lives in :mod:`src.experiments.metrics` so that the same raw
    output can be re-summarised later without re-running the simulation.
    """

    def __init__(self) -> None:
        self.trips: list[TripRecord] = []
        self.occupancy_samples: list[dict[str, Any]] = []
        self.network_samples: list[dict[str, Any]] = []
        self.events: dict[str, int] = defaultdict(int)
        self.vmt_m: dict[str, float] = defaultdict(float)
        self.revenue_usd: float = 0.0
        self.fines_usd: float = 0.0

    def add_trip(self, record: TripRecord) -> None:
        self.trips.append(record)

    def count(self, name: str, k: int = 1) -> None:
        self.events[name] += k

    def add_vmt(self, vehicle_class: str, metres: float) -> None:
        self.vmt_m[vehicle_class] += metres

    def sample_occupancy(self, t_min: float, inventory: CurbInventory) -> None:
        self.occupancy_samples.append(
            {
                "t_min": t_min,
                "overall": float(np.mean([s.occupancy_rate() for s in inventory])),
                **{f"occ_{c}": inventory.occupancy_by_class()[c] for c in VEHICLE_CLASSES},
                "mean_price": float(np.mean([s.price_per_hour for s in inventory])),
                "per_segment": {s.id: s.occupancy_rate() for s in inventory},
            }
        )

    def sample_network(self, t_min: float, router: Router) -> None:
        self.network_samples.append(
            {
                "t_min": t_min,
                "mean_speed_kph": router.mean_speed_kph(t_min),
                "congestion_index": router.congestion_index(t_min),
            }
        )

    @property
    def total_vmt_miles(self) -> float:
        return sum(self.vmt_m.values()) / METRES_PER_MILE


class CurbWorld:
    """Shared simulation state handed to every agent process."""

    def __init__(self, cfg: RunConfig, env: simpy.Environment | None = None) -> None:
        self.cfg = cfg
        self.env = env if env is not None else simpy.Environment()
        self.rng = np.random.default_rng(cfg.seed)

        self.graph, self.inventory = build_grid_network(cfg.network)
        self.router = Router(self.graph, cfg.network)
        self.inventory.set_allocation(cfg.allocation, now=0.0)

        self.pricing = PricingPolicy(
            cfg.scenario_spec.get("pricing_policy", "static"),
            cfg.scenario_spec.get("pricing_params"),
        )
        self.metrics = MetricsRecorder()
        self.agent_params = cfg.agents
        self.common = cfg.agents["common"]
        self._next_agent_id = 0
        self.warmup_min = cfg.warmup_min
        self.horizon_min = cfg.horizon_min

        self._build_demand_geography()

        # Curb segments indexed by the node they are nearest to, so a search can
        # start from the destination intersection without a global scan.
        self._segments_near_node: dict[str, list[tuple[CurbSegment, float]]] = {}
        self._precompute_segment_proximity()

    # -- identity ---------------------------------------------------------------
    def next_agent_id(self) -> int:
        self._next_agent_id += 1
        return self._next_agent_id

    @property
    def now(self) -> float:
        """Current simulation time in minutes."""
        return float(self.env.now)

    @property
    def in_warmup(self) -> bool:
        return self.env.now < self.warmup_min

    # -- demand geography -------------------------------------------------------
    def _build_demand_geography(self) -> None:
        geo = self.cfg.network.get("demand_geography", {})
        bias = geo.get("core_bias", {})
        nodes = list(self.graph.nodes)
        xy = np.array([[self.graph.nodes[v]["x"], self.graph.nodes[v]["y"]] for v in nodes])
        centre = xy.mean(axis=0)
        d = np.hypot(xy[:, 0] - centre[0], xy[:, 1] - centre[1])
        dn = d / (d.max() if d.max() > 0 else 1.0)

        self.nodes = nodes
        self.node_weights: dict[str, np.ndarray] = {}
        for cls in VEHICLE_CLASSES:
            b = float(bias.get(cls, 1.0))
            w = (1.0 - 0.65 * dn) ** b
            self.node_weights[cls] = w / w.sum()

        # Commercial frontages: the most central share of intersections.
        share = float(geo.get("commercial_node_share", 0.35))
        k = max(1, int(round(share * len(nodes))))
        order = np.argsort(d)
        self.commercial_nodes = [nodes[int(i)] for i in order[:k]]
        cw = np.array([1.0 if v in set(self.commercial_nodes) else 0.0 for v in nodes])
        self.node_weights["delivery"] = self.node_weights["delivery"] * cw
        total = self.node_weights["delivery"].sum()
        self.node_weights["delivery"] = (
            self.node_weights["delivery"] / total
            if total > 0
            else np.full(len(nodes), 1 / len(nodes))
        )

    def sample_destination(self, vehicle_class: str) -> str:
        idx = self.rng.choice(len(self.nodes), p=self.node_weights[vehicle_class])
        return self.nodes[int(idx)]

    def sample_node_uniform(self) -> str:
        return self.nodes[int(self.rng.integers(len(self.nodes)))]

    # -- curb search support ----------------------------------------------------
    def _precompute_segment_proximity(self) -> None:
        """For each node, the curb segments sorted by walking distance."""
        for v in self.graph.nodes:
            vx, vy = self.graph.nodes[v]["x"], self.graph.nodes[v]["y"]
            dists = self.inventory.distances_from((vx, vy))
            order = np.argsort(dists)
            self._segments_near_node[v] = [
                (self.inventory.segments[int(i)], float(dists[int(i)])) for i in order
            ]

    def candidate_segments(
        self, node: str, radius_m: float, limit: int | None = None
    ) -> list[tuple[CurbSegment, float]]:
        """Segments within ``radius_m`` of ``node``, nearest first."""
        out = []
        for seg, dist in self._segments_near_node[node]:
            if dist > radius_m:
                break
            out.append((seg, dist))
            if limit is not None and len(out) >= limit:
                break
        return out

    def segment_access_node(self, seg: CurbSegment) -> str:
        """The intersection a driver reaches when they arrive at ``seg``.

        A segment is entered from the downstream end of its link, which is the
        node a vehicle is at once it has driven past the whole face.
        """
        return seg.link[1]

    # -- money ------------------------------------------------------------------
    def charge(self, amount: float) -> None:
        if not self.in_warmup:
            self.metrics.revenue_usd += amount

    def fine(self, amount: float) -> None:
        if not self.in_warmup:
            self.metrics.fines_usd += amount

    def value_of_time_per_min(self, vehicle_class: str) -> float:
        return float(self.common["value_of_time_per_hour"][vehicle_class]) / 60.0

    # -- movement helpers -------------------------------------------------------
    def drive(self, origin: str, destination: str, vehicle_class: str):
        """SimPy process: move a vehicle along the shortest path, accruing VMT.

        Yields for the congested travel time and returns the distance covered.
        """
        if origin == destination:
            return 0.0
        tt_s = self.router.travel_time_s(origin, destination, self.now)
        self.router.record_traversal(origin, destination, self.now)
        dist = self.router.path_length_m(origin, destination)
        yield self.env.timeout(tt_s / 60.0)
        if not self.in_warmup:
            self.metrics.add_vmt(vehicle_class, dist)
        return dist

    def cruise(self, origin: str, destination: str, vehicle_class: str):
        """Drive at reduced *search* speed: cruising for parking is slower."""
        if origin == destination:
            return 0.0
        dist = self.router.path_length_m(origin, destination)
        speed = float(self.common["search_speed_mps"])
        # Search speed sets a floor on travel time; congestion can make it worse.
        tt_s = max(dist / speed, self.router.travel_time_s(origin, destination, self.now))
        self.router.record_traversal(origin, destination, self.now)
        yield self.env.timeout(tt_s / 60.0)
        if not self.in_warmup:
            self.metrics.add_vmt(vehicle_class, dist)
        return dist

    def walk_time_min(self, metres: float) -> float:
        return metres / float(self.common["walking_speed_mps"]) / 60.0

    # -- random draws -----------------------------------------------------------
    def lognormal(self, mean: float, cv: float) -> float:
        """Positive draw with the requested mean and coefficient of variation."""
        if cv <= 0:
            return mean
        sigma = float(np.sqrt(np.log(1.0 + cv**2)))
        mu = float(np.log(max(mean, 1e-9)) - 0.5 * sigma**2)
        return float(self.rng.lognormal(mu, sigma))
