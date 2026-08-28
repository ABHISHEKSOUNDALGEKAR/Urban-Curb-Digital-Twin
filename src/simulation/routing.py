"""Road network construction and travel-time modelling.

The district is a NetworkX ``DiGraph``: nodes are intersections carrying metric
coordinates, edges are one-block street links carrying length, free-flow travel
time and a curb inventory reference.

Travel time on a link is the free-flow time inflated by two effects:

1. **Recurrent congestion** — a BPR (Bureau of Public Roads) volume-delay
   function driven by a rolling count of traversals, so that heavy cruising
   traffic slows itself down.
2. **Curb friction** — each vehicle currently double-parked in the travel lane
   adds a fixed delay, which is the mechanism by which a curb-supply failure
   becomes a network-wide congestion cost.

When the SUMO backend is enabled these functions are bypassed and travel times
come from the microsimulation instead (see ``src/sumo``).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

import networkx as nx
import numpy as np

from src.simulation.curb import CurbInventory, CurbSegment

MPS_PER_KPH = 1000.0 / 3600.0
METRES_PER_MILE = 1609.344


def node_name(row: int, col: int) -> str:
    """Grid node identifier, e.g. ``N02_03``."""
    return f"N{row:02d}_{col:02d}"


def build_grid_network(net_cfg: dict) -> tuple[nx.DiGraph, CurbInventory]:
    """Build the synthetic grid road network and its curb inventory.

    Returns the directed graph and a :class:`CurbInventory` whose segments are
    attached to graph edges via the ``curb_ids`` edge attribute.
    """
    n = net_cfg["network"]
    rows, cols = int(n["grid_rows"]), int(n["grid_cols"])
    block = float(n["block_length_m"])
    ox, oy = n.get("origin_xy", [0.0, 0.0])
    free_flow_mps = float(n["free_flow_speed_kph"]) * MPS_PER_KPH

    graph = nx.DiGraph()
    for r in range(rows):
        for col in range(cols):
            graph.add_node(
                node_name(r, col),
                x=ox + col * block,
                y=oy + r * block,
                row=r,
                col=col,
            )

    # Undirected street grid, materialised as both directions.
    undirected: list[tuple[str, str]] = []
    for r in range(rows):
        for col in range(cols):
            if col + 1 < cols:
                undirected.append((node_name(r, col), node_name(r, col + 1)))
            if r + 1 < rows:
                undirected.append((node_name(r, col), node_name(r + 1, col)))

    for u, v in undirected:
        length = float(
            np.hypot(
                graph.nodes[v]["x"] - graph.nodes[u]["x"], graph.nodes[v]["y"] - graph.nodes[u]["y"]
            )
        )
        for a, b in ((u, v), (v, u)):
            graph.add_edge(
                a,
                b,
                length_m=length,
                free_flow_s=length / free_flow_mps,
                curb_ids=[],
            )

    inventory = _build_curb_inventory(graph, undirected, net_cfg)
    for seg in inventory:
        graph.edges[seg.link]["curb_ids"].append(seg.id)
    return graph, inventory


def _build_curb_inventory(
    graph: nx.DiGraph, undirected: Iterable[tuple[str, str]], net_cfg: dict
) -> CurbInventory:
    c = net_cfg["curb"]
    rng = np.random.default_rng(20240501)  # inventory is fixed infrastructure
    stall_len = float(c["stall_length_m"])
    usable = float(c["usable_fraction"])
    jitter = float(c.get("capacity_jitter", 0.0))
    per_link = int(c.get("segments_per_link", 2))
    price_cfg = c["pricing"]
    time_limit = float(c.get("time_limit_min", 120))

    xs = [graph.nodes[v]["x"] for v in graph.nodes]
    ys = [graph.nodes[v]["y"] for v in graph.nodes]
    centre = (float(np.mean(xs)), float(np.mean(ys)))

    segments: list[CurbSegment] = []
    for u, v in undirected:
        ux, uy = graph.nodes[u]["x"], graph.nodes[u]["y"]
        vx, vy = graph.nodes[v]["x"], graph.nodes[v]["y"]
        length = float(np.hypot(vx - ux, vy - uy))
        mid = ((ux + vx) / 2.0, (uy + vy) / 2.0)
        base_stalls = length * usable / stall_len
        for side in range(per_link):
            stalls = int(max(1, round(base_stalls * (1.0 + rng.uniform(-jitter, jitter)))))
            # The two faces of a block are served by opposite directions of
            # travel; a driver can only stop at the curb on their own side.
            link = (u, v) if side == 0 else (v, u)
            dist_to_centre = float(np.hypot(mid[0] - centre[0], mid[1] - centre[1]))
            price = (
                float(price_cfg["core_price_per_hour"])
                if dist_to_centre <= float(price_cfg["core_radius_m"])
                else float(price_cfg["edge_price_per_hour"])
            )
            seg_id = f"{u}_{v}_S{side}"
            segments.append(
                CurbSegment(
                    curb_id=seg_id,
                    link=link,
                    xy=mid,
                    total_stalls=stalls,
                    price_per_hour=price,
                    time_limit_min=time_limit,
                    face_length_m=length,
                )
            )
    return CurbInventory(segments)


@dataclass
class _LinkState:
    traversals: deque
    double_parked: int = 0


class Router:
    """Shortest-path routing plus a dynamic link performance function.

    Paths are computed on free-flow travel time and cached: re-planning every
    vehicle on live congestion would be both slower and less realistic than
    drivers following habitual routes.
    """

    def __init__(self, graph: nx.DiGraph, net_cfg: dict, flow_window_min: float = 15.0) -> None:
        self.graph = graph
        cong = net_cfg.get("congestion", {})
        self.alpha = float(cong.get("bpr_alpha", 0.15))
        self.beta = float(cong.get("bpr_beta", 4.0))
        self.capacity_vph = float(cong.get("link_capacity_vph", 600.0))
        self.double_park_delay_s = float(cong.get("double_park_delay_s", 20.0))
        self.flow_window_min = float(flow_window_min)
        self._state: dict[tuple[str, str], _LinkState] = {
            e: _LinkState(deque()) for e in graph.edges
        }
        self._path_cache: dict[tuple[str, str], list[str]] = {}
        self._free_flow_cache: dict[tuple[str, str], float] = {}
        self.nodes: list[str] = list(graph.nodes)
        self._node_xy = {v: (graph.nodes[v]["x"], graph.nodes[v]["y"]) for v in graph.nodes}

    # -- geometry ---------------------------------------------------------------
    def xy(self, node: str) -> tuple[float, float]:
        return self._node_xy[node]

    def nearest_node(self, xy: tuple[float, float]) -> str:
        return min(
            self.nodes,
            key=lambda v: (self._node_xy[v][0] - xy[0]) ** 2 + (self._node_xy[v][1] - xy[1]) ** 2,
        )

    # -- paths ------------------------------------------------------------------
    def path(self, origin: str, destination: str) -> list[str]:
        key = (origin, destination)
        cached = self._path_cache.get(key)
        if cached is None:
            cached = nx.shortest_path(self.graph, origin, destination, weight="free_flow_s")
            self._path_cache[key] = cached
        return cached

    def path_length_m(self, origin: str, destination: str) -> float:
        p = self.path(origin, destination)
        return sum(self.graph.edges[a, b]["length_m"] for a, b in zip(p[:-1], p[1:], strict=True))

    def free_flow_time_s(self, origin: str, destination: str) -> float:
        key = (origin, destination)
        cached = self._free_flow_cache.get(key)
        if cached is None:
            p = self.path(origin, destination)
            cached = sum(
                self.graph.edges[a, b]["free_flow_s"] for a, b in zip(p[:-1], p[1:], strict=True)
            )
            self._free_flow_cache[key] = cached
        return cached

    # -- link performance -------------------------------------------------------
    def link_travel_time_s(self, link: tuple[str, str], now_min: float) -> float:
        """Congested travel time on one link, in seconds."""
        edge = self.graph.edges[link]
        state = self._state[link]
        self._expire(state, now_min)
        flow_vph = len(state.traversals) * (60.0 / self.flow_window_min)
        ratio = flow_vph / self.capacity_vph
        bpr = 1.0 + self.alpha * ratio**self.beta
        return edge["free_flow_s"] * bpr + state.double_parked * self.double_park_delay_s

    def travel_time_s(self, origin: str, destination: str, now_min: float) -> float:
        if origin == destination:
            return 0.0
        p = self.path(origin, destination)
        return sum(
            self.link_travel_time_s((a, b), now_min) for a, b in zip(p[:-1], p[1:], strict=True)
        )

    def record_traversal(self, origin: str, destination: str, now_min: float) -> None:
        """Register a vehicle passing over every link of a path, for the BPR flow."""
        if origin == destination:
            return
        p = self.path(origin, destination)
        for a, b in zip(p[:-1], p[1:], strict=True):
            self._state[a, b].traversals.append(now_min)

    def _expire(self, state: _LinkState, now_min: float) -> None:
        cutoff = now_min - self.flow_window_min
        q = state.traversals
        while q and q[0] < cutoff:
            q.popleft()

    # -- curb friction ----------------------------------------------------------
    def add_double_park(self, link: tuple[str, str]) -> None:
        self._state[link].double_parked += 1

    def remove_double_park(self, link: tuple[str, str]) -> None:
        st = self._state[link]
        st.double_parked = max(0, st.double_parked - 1)

    def mean_speed_kph(self, now_min: float) -> float:
        """Flow-weighted mean link speed across the network."""
        num, den = 0.0, 0.0
        for link in self.graph.edges:
            edge = self.graph.edges[link]
            tt = self.link_travel_time_s(link, now_min)
            w = max(1.0, len(self._state[link].traversals))
            num += w * (edge["length_m"] / tt) if tt > 0 else 0.0
            den += w
        return (num / den) / MPS_PER_KPH if den else 0.0

    def congestion_index(self, now_min: float) -> float:
        """Mean ratio of congested to free-flow travel time across links."""
        vals = []
        for link in self.graph.edges:
            ff = self.graph.edges[link]["free_flow_s"]
            vals.append(self.link_travel_time_s(link, now_min) / ff if ff else 1.0)
        return float(np.mean(vals)) if vals else 1.0
