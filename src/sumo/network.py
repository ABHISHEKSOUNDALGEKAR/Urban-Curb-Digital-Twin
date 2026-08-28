"""Generate a SUMO network and curb infrastructure from the same configuration.

There is exactly one source of truth for the district: ``config/network.yaml``.
The SimPy backend turns it into a NetworkX graph; this module turns the *same*
configuration into SUMO node/edge XML, runs ``netconvert``, and writes an
additional-file describing every curb segment as a set of ``parkingArea``
elements - one per regulation type, sized by the current allocation.

Keeping both backends downstream of one config is what makes a comparison
between them meaningful: if the curb inventories differed, any difference in
results would be uninterpretable.
"""

from __future__ import annotations

import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from src.config import VEHICLE_CLASSES, RunConfig
from src.simulation.curb import CurbInventory
from src.simulation.routing import MPS_PER_KPH, build_grid_network
from src.sumo import require_sumo, sumo_binary

#: Metres reserved at each end of a block face for the junction and daylighting.
JUNCTION_CLEARANCE_M = 12.0
#: Longitudinal space one parked vehicle occupies inside a parkingArea.
STALL_PITCH_M = 7.0

VEHICLE_TYPES: dict[str, dict[str, str]] = {
    "passenger": {
        "vClass": "passenger",
        "length": "4.8",
        "color": "0.2,0.6,1.0",
        "accel": "2.6",
        "decel": "4.5",
    },
    "delivery": {
        "vClass": "delivery",
        "length": "7.5",
        "color": "1.0,0.6,0.1",
        "accel": "1.8",
        "decel": "4.0",
    },
    "ridehail": {
        "vClass": "taxi",
        "length": "4.8",
        "color": "0.1,0.8,0.4",
        "accel": "2.6",
        "decel": "4.5",
    },
}


def edge_id(u: str, v: str) -> str:
    return f"{u}__{v}"


def parking_area_id(segment_id: str, regulation: str) -> str:
    return f"pa_{segment_id}_{regulation}"


@dataclass
class SumoNetworkFiles:
    """Paths to everything the SUMO backend needs to start."""

    directory: Path
    net_file: Path
    additional_file: Path
    sumocfg: Path
    inventory: CurbInventory
    edge_by_link: dict[tuple[str, str], str]
    parking_areas: dict[str, dict[str, str]]  # segment_id -> {regulation: parkingArea id}
    parking_capacity: dict[str, int]  # parkingArea id -> capacity


def build_sumo_network(
    cfg: RunConfig, out_dir: Path | str, force: bool = False, with_rerouters: bool = True
) -> SumoNetworkFiles:
    """Write the SUMO network and curb infrastructure for ``cfg``."""
    require_sumo()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    graph, inventory = build_grid_network(cfg.network)
    inventory.set_allocation(cfg.allocation, now=0.0)

    nod_file = out / "district.nod.xml"
    edg_file = out / "district.edg.xml"
    net_file = out / "district.net.xml"
    add_file = out / "district.add.xml"
    sumocfg = out / "district.sumocfg"

    speed = float(cfg.network["network"]["free_flow_speed_kph"]) * MPS_PER_KPH
    _write_nodes(nod_file, graph)
    edge_by_link = _write_edges(edg_file, graph, speed)

    if force or not net_file.exists():
        _run_netconvert(nod_file, edg_file, net_file)

    parking_areas, capacity = _write_additional(add_file, inventory, edge_by_link, with_rerouters)
    _write_sumocfg(sumocfg, net_file, add_file, cfg)

    return SumoNetworkFiles(
        directory=out,
        net_file=net_file,
        additional_file=add_file,
        sumocfg=sumocfg,
        inventory=inventory,
        edge_by_link=edge_by_link,
        parking_areas=parking_areas,
        parking_capacity=capacity,
    )


def _write_nodes(path: Path, graph) -> None:
    root = ET.Element("nodes")
    for name, data in graph.nodes(data=True):
        ET.SubElement(
            root,
            "node",
            id=name,
            x=f"{data['x']:.2f}",
            y=f"{data['y']:.2f}",
            type="priority",
        )
    _write_xml(root, path)


def _write_edges(path: Path, graph, speed_mps: float) -> dict[tuple[str, str], str]:
    root = ET.Element("edges")
    mapping: dict[tuple[str, str], str] = {}
    for u, v, data in graph.edges(data=True):
        eid = edge_id(u, v)
        mapping[(u, v)] = eid
        ET.SubElement(
            root,
            "edge",
            id=eid,
            **{"from": u, "to": v},
            numLanes="1",
            speed=f"{speed_mps:.3f}",
            length=f"{data['length_m']:.2f}",
        )
    _write_xml(root, path)
    return mapping


def _run_netconvert(nod: Path, edg: Path, net: Path) -> None:
    binary = sumo_binary("netconvert")
    cmd = [
        str(binary),
        "--node-files",
        str(nod),
        "--edge-files",
        str(edg),
        "--output-file",
        str(net),
        "--no-turnarounds",
        "true",
        "--junctions.corner-detail",
        "0",
        "--offset.disable-normalization",
        "true",
        "--no-warnings",
        "true",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:  # pragma: no cover - depends on local SUMO build
        raise RuntimeError(f"netconvert failed:\n{proc.stdout}\n{proc.stderr}")


def _write_additional(
    path: Path,
    inventory: CurbInventory,
    edge_by_link: dict[tuple[str, str], str],
    with_rerouters: bool,
) -> tuple[dict[str, dict[str, str]], dict[str, int]]:
    """Write vehicle types and one parkingArea per (segment, regulation)."""
    root = ET.Element("additional")

    for name, attrs in VEHICLE_TYPES.items():
        vtype = ET.SubElement(root, "vType", id=name, **attrs)
        if name == "ridehail":
            # SUMO's taxi device is what exposes the reservation/dispatch API.
            ET.SubElement(vtype, "param", key="has.taxi.device", value="true")
            ET.SubElement(vtype, "param", key="device.taxi.pickUpDuration", value="30")
            ET.SubElement(vtype, "param", key="device.taxi.dropOffDuration", value="20")
            # Idle taxis must stay in the network to be dispatchable; without an
            # idle algorithm they reach the end of their route and are removed.
            ET.SubElement(vtype, "param", key="device.taxi.idleAlgorithm", value="randomCircling")

    areas: dict[str, dict[str, str]] = {}
    capacity: dict[str, int] = {}
    for seg in inventory:
        eid = edge_by_link.get(seg.link)
        if eid is None:  # pragma: no cover - defensive
            continue
        lane = f"{eid}_0"
        # A block face is shared between the regulation types: lay them out
        # end-to-end in proportion to their stall counts, leaving the junction
        # approaches clear.
        face_len = seg.face_length_m
        usable_start = JUNCTION_CLEARANCE_M
        usable_end = max(usable_start + STALL_PITCH_M, face_len - JUNCTION_CLEARANCE_M)
        usable = usable_end - usable_start
        stalls = max(1, seg.total_stalls)
        cursor = usable_start
        areas[seg.id] = {}
        for reg in VEHICLE_CLASSES:
            n = seg.capacity(reg)
            if n <= 0:
                continue
            span = usable * (n / stalls)
            pa_id = parking_area_id(seg.id, reg)
            ET.SubElement(
                root,
                "parkingArea",
                id=pa_id,
                lane=lane,
                startPos=f"{cursor:.2f}",
                endPos=f"{min(cursor + span, usable_end):.2f}",
                roadsideCapacity=str(n),
                friendlyPos="true",
                angle="0",
                name=f"{seg.id}:{reg}",
            )
            areas[seg.id][reg] = pa_id
            capacity[pa_id] = n
            cursor += span

    if with_rerouters:
        _write_rerouters(root, inventory, edge_by_link, areas)

    _write_xml(root, path)
    return areas, capacity


def _write_rerouters(root, inventory, edge_by_link, areas) -> None:
    """Install SUMO parking-area rerouters as a native-search fallback.

    The Python layer normally does the searching, but having rerouters in the
    network means a vehicle whose chosen stall fills up between the decision and
    the arrival is redirected by SUMO instead of stalling on the lane - the same
    failure mode a real driver handles by simply carrying on to the next block.
    """
    by_edge: dict[str, list[str]] = {}
    for seg in inventory:
        eid = edge_by_link.get(seg.link)
        if eid is None:
            continue
        for pa in areas.get(seg.id, {}).values():
            by_edge.setdefault(eid, []).append(pa)
    for eid, pas in by_edge.items():
        rr = ET.SubElement(root, "rerouter", id=f"rr_{eid}", edges=eid)
        interval = ET.SubElement(rr, "interval", begin="0", end="86400")
        for pa in pas:
            ET.SubElement(interval, "parkingAreaReroute", id=pa, visible="true")


def _write_sumocfg(path: Path, net: Path, add: Path, cfg: RunConfig) -> None:
    root = ET.Element("configuration")
    inp = ET.SubElement(root, "input")
    ET.SubElement(inp, "net-file", value=net.name)
    ET.SubElement(inp, "additional-files", value=add.name)
    t = ET.SubElement(root, "time")
    ET.SubElement(t, "begin", value="0")
    ET.SubElement(t, "end", value=str(int(cfg.horizon_min * 60 + 3600)))
    ET.SubElement(t, "step-length", value="1.0")
    proc = ET.SubElement(root, "processing")
    ET.SubElement(proc, "time-to-teleport", value="600")
    ET.SubElement(proc, "collision.action", value="warn")
    rep = ET.SubElement(root, "report")
    ET.SubElement(rep, "no-step-log", value="true")
    ET.SubElement(rep, "no-warnings", value="true")
    _write_xml(root, path)


def _write_xml(root: ET.Element, path: Path) -> None:
    ET.indent(root, space="  ")
    path.write_bytes(ET.tostring(root, encoding="utf-8", xml_declaration=True))
