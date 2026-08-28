#!/usr/bin/env python3
"""Export the district's data layer: curb inventory, demand table, network geometry.

The engine builds its network from ``config/network.yaml`` at run time, so these
files are not an input it depends on - they are the *data layer made explicit*.
That separation is the point: the same schemas are what a real curb inventory,
meter-transaction extract and demand table would be loaded into, so pointing the
model at empirical data means replacing these files, not rewriting the engine.

Usage::

    python scripts/generate_data.py [--out data]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import VEHICLE_CLASSES, load_scenario
from src.simulation.engine import profile_multiplier
from src.simulation.routing import build_grid_network

ROOT = Path(__file__).resolve().parents[1]


def curb_segments_table(cfg) -> pd.DataFrame:
    _graph, inventory = build_grid_network(cfg.network)
    inventory.set_allocation(cfg.allocation)
    rows = []
    for seg in inventory:
        rows.append(
            {
                "curb_id": seg.id,
                "from_node": seg.link[0],
                "to_node": seg.link[1],
                "x_m": seg.xy[0],
                "y_m": seg.xy[1],
                "face_length_m": seg.face_length_m,
                "total_stalls": seg.total_stalls,
                **{f"stalls_{c}": seg.capacity(c) for c in VEHICLE_CLASSES},
                "price_usd_per_hour": seg.price_per_hour,
                "time_limit_min": seg.time_limit_min,
            }
        )
    return pd.DataFrame(rows)


def demand_table(cfg) -> pd.DataFrame:
    """Hourly arrival rates implied by the base rates and the time-of-day profile."""
    profile = cfg.scenario_spec.get("demand_profile") or [1.0]
    horizon = cfg.horizon_min
    rows = []
    hour = 0
    t = 0.0
    while t < horizon:
        mult = profile_multiplier(profile, t + 30.0, horizon)
        for cls, rate in cfg.demand.items():
            rows.append(
                {
                    "hour_index": hour,
                    "t_start_min": t,
                    "vehicle_class": cls,
                    "base_rate_veh_per_hour": rate,
                    "profile_multiplier": round(mult, 4),
                    "arrival_rate_veh_per_hour": round(rate * mult, 2),
                }
            )
        hour += 1
        t += 60.0
    return pd.DataFrame(rows)


def network_geojson(cfg) -> dict:
    """Street network as a GeoJSON FeatureCollection in local metric coordinates."""
    graph, inventory = build_grid_network(cfg.network)
    inventory.set_allocation(cfg.allocation)
    features = []
    for name, data in graph.nodes(data=True):
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [data["x"], data["y"]]},
                "properties": {"kind": "intersection", "node_id": name},
            }
        )
    for u, v, data in graph.edges(data=True):
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [graph.nodes[u]["x"], graph.nodes[u]["y"]],
                        [graph.nodes[v]["x"], graph.nodes[v]["y"]],
                    ],
                },
                "properties": {
                    "kind": "street_link",
                    "from": u,
                    "to": v,
                    "length_m": round(data["length_m"], 2),
                    "free_flow_s": round(data["free_flow_s"], 2),
                    "curb_ids": data["curb_ids"],
                },
            }
        )
    for seg in inventory:
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": list(seg.xy)},
                "properties": {
                    "kind": "curb_segment",
                    "curb_id": seg.id,
                    "total_stalls": seg.total_stalls,
                    **{f"stalls_{c}": seg.capacity(c) for c in VEHICLE_CLASSES},
                    "price_usd_per_hour": seg.price_per_hour,
                },
            }
        )
    return {
        "type": "FeatureCollection",
        "name": cfg.network["network"]["name"],
        "crs": {"type": "name", "properties": {"name": "local-metric-not-georeferenced"}},
        "features": features,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(ROOT / "data"))
    p.add_argument("--scenario", default="baseline")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_scenario(args.scenario, seed=0)

    segments = curb_segments_table(cfg)
    segments.to_csv(out / "curb_segments.csv", index=False)
    demand = demand_table(cfg)
    demand.to_csv(out / "demand.csv", index=False)
    (out / "network.geojson").write_text(json.dumps(network_geojson(cfg), indent=1))

    print(f"curb_segments.csv  {len(segments):4d} segments, {segments.total_stalls.sum()} stalls")
    print(f"demand.csv         {len(demand):4d} rows")
    print("network.geojson    written")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
