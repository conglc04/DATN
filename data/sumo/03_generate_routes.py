"""Step 3 — Generate ambulance routes converging to Bạch Mai Hospital.

Strategy: query sumolib for road edges whose FROM node is OUTSIDE the cell
(> R_CELL_M from gNB) so that ambulances enter the cell from outside.
FROM dist > TO dist ensures the edge geometrically converges toward gNB.
K edges are chosen at maximally-spread compass bearings.

gNB anchor SSOT: utils/config.py → BACH_MAI_LAT / BACH_MAI_LON.

Reference: sumo-user.pdf §route definition; sumolib API §getNeighboringEdges.
"""

from __future__ import annotations

import math
import os
import sys
import xml.etree.ElementTree as ET

import sumolib

HERE    = os.path.dirname(os.path.abspath(__file__))
NET_FILE = os.path.join(HERE, "bachmaiHN.net.xml")

# SSOT import — config lives in baselines/utils/config.py
sys.path.insert(0, os.path.join(HERE, "..", "..", "baselines"))
from utils.config import BACH_MAI_LAT as GNB_LAT, BACH_MAI_LON as GNB_LON  # noqa: E402
from utils.config import R_CELL_M                                             # noqa: E402

SPEED_MS     = 60.0 / 3.6   # 60 km/h — ambulance vType maxSpeed
DEPART_SPEED = "max"         # use edge speed limit to avoid "slow lane" departure error

# Edge-search radii — ambulances start OUTSIDE the cell and drive in
_SEARCH_R_M  = R_CELL_M * 2.5  # outer search radius (m)
_MIN_DIST_M  = R_CELL_M        # FROM node must be at or beyond cell edge
_MAX_DIST_M  = R_CELL_M * 2.0  # FROM node at most 2× cell radius away


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gnb_xy(net: sumolib.net.Net) -> tuple[float, float]:
    """gNB anchor in SUMO projected (XY) metres."""
    return net.convertLonLat2XY(GNB_LON, GNB_LAT)


def _dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def _from_dist(edge: sumolib.net.edge.Edge,
               gnb_x: float, gnb_y: float) -> float:
    x, y = edge.getFromNode().getCoord()
    return _dist(x, y, gnb_x, gnb_y)


def _to_dist(edge: sumolib.net.edge.Edge,
             gnb_x: float, gnb_y: float) -> float:
    x, y = edge.getToNode().getCoord()
    return _dist(x, y, gnb_x, gnb_y)


def bearing_deg(ax: float, ay: float,
                bx: float, by: float) -> float:
    """Compass bearing (0=N, 90=E) from a to b in SUMO projected coords."""
    return math.degrees(math.atan2(bx - ax, by - ay)) % 360


def find_dest_edge(net: sumolib.net.Net,
                   gnb_x: float, gnb_y: float) -> str:
    """Edge id closest to the gNB anchor (destination for all routes)."""
    for r in (50.0, 100.0, 200.0):
        cands = net.getNeighboringEdges(gnb_x, gnb_y, r=r, includeJunctions=False)
        if cands:
            cands.sort(key=lambda x: x[1])
            return cands[0][0].getID()
    raise RuntimeError("No destination edge found near gNB.")


def find_start_edges_by_bearing(net: sumolib.net.Net,
                                 gnb_x: float, gnb_y: float,
                                 K: int) -> list[sumolib.net.edge.Edge]:
    """Return K edges that start OUTSIDE the cell and converge toward gNB.

    Selection:
      1. Candidate edges whose FROM node is in [R_CELL_M, 2×R_CELL_M] (outside cell).
      2. FROM dist > TO dist  (geometrically closer at TO → vehicle converges inward).
      3. K edges chosen at maximally-spread compass bearings.
    """
    raw_cands = net.getNeighboringEdges(gnb_x, gnb_y, r=_SEARCH_R_M,
                                         includeJunctions=False)

    pool: list[tuple[sumolib.net.edge.Edge, float]] = []
    for edge, _ in raw_cands:
        fd = _from_dist(edge, gnb_x, gnb_y)
        td = _to_dist(edge, gnb_x, gnb_y)
        if _MIN_DIST_M <= fd <= _MAX_DIST_M and fd > td:
            fx, fy = edge.getFromNode().getCoord()
            bear = bearing_deg(gnb_x, gnb_y, fx, fy)
            pool.append((edge, bear))

    if len(pool) < K:
        raise RuntimeError(
            f"Only {len(pool)} converging edges within {_MAX_DIST_M} m of gNB "
            f"— need {K}.  Check network or reduce K."
        )

    print(f"Candidate pool ({len(pool)} edges):")
    for e, b in sorted(pool, key=lambda x: x[1]):
        fd = _from_dist(e, gnb_x, gnb_y)
        td = _to_dist(e, gnb_x, gnb_y)
        print(f"  {e.getID():30s}  bear={b:.1f}°  FROM={fd:.0f}m  TO={td:.0f}m")

    target_bearings = [360.0 * i / K for i in range(K)]
    selected: list[sumolib.net.edge.Edge] = []
    used_ids: set[str] = set()
    for target in target_bearings:
        available = [(e, b) for e, b in pool if e.getID() not in used_ids]
        best = min(available, key=lambda x: abs((x[1] - target + 180) % 360 - 180))
        selected.append(best[0])
        used_ids.add(best[0].getID())

    return selected


# ---------------------------------------------------------------------------
# Route writer
# ---------------------------------------------------------------------------

def write_routes(
    start_edges: list[sumolib.net.edge.Edge],
    dest_edge_id: str,
    ambulance_ids: list[str],
    out_path: str,
) -> None:
    root = ET.Element("routes")
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root.set(
        "xsi:noNamespaceSchemaLocation",
        "http://sumo.dlr.de/xsd/routes_file.xsd",
    )

    vtype = ET.SubElement(root, "vType")
    vtype.set("id", "ambulance")
    vtype.set("vClass", "emergency")  # siren always active: passes red lights
    vtype.set("maxSpeed", f"{SPEED_MS:.2f}")
    vtype.set("accel", "3.0")
    vtype.set("decel", "4.5")
    vtype.set("length", "5.0")

    for aid, src_edge in zip(ambulance_ids, start_edges):
        trip = ET.SubElement(root, "trip")
        trip.set("id", aid)
        trip.set("type", "ambulance")
        trip.set("from", src_edge.getID())
        trip.set("to", dest_edge_id)
        trip.set("depart", "0.00")
        trip.set("departSpeed", str(DEPART_SPEED))

    ET.indent(root, space="    ")
    tree = ET.ElementTree(root)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        tree.write(f, encoding="unicode", xml_declaration=False)

    print(f"Written: {out_path}")
    for aid, edge in zip(ambulance_ids, start_edges):
        print(f"  {aid}: edge={edge.getID()!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not os.path.exists(NET_FILE):
        print(f"ERROR: {NET_FILE} not found. Run 02_build_network.sh first.")
        raise SystemExit(1)

    print(f"gNB anchor (SSOT): {GNB_LAT}°N, {GNB_LON}°E")
    print(f"R_CELL_M: {R_CELL_M} m")
    print()
    print("Loading network…")
    net = sumolib.net.readNet(NET_FILE)

    gx, gy = gnb_xy(net)
    print(f"gNB projected XY: ({gx:.1f}, {gy:.1f})")

    dest_edge_id = find_dest_edge(net, gx, gy)
    print(f"Destination edge: {dest_edge_id!r}")
    print()

    # K=3
    k3_starts = find_start_edges_by_bearing(net, gx, gy, K=3)
    print()
    write_routes(
        k3_starts,
        dest_edge_id,
        ["amb_0", "amb_1", "amb_2"],
        os.path.join(HERE, "ambulance_routes_k3.xml"),
    )

    # K=1: use the nearest-to-North edge
    k1_starts = [k3_starts[0]]
    write_routes(
        k1_starts,
        dest_edge_id,
        ["amb_0"],
        os.path.join(HERE, "ambulance_routes_k1.xml"),
    )

    print("\nDone.  Run 04_run_simulation.sh to generate FCD traces.")
