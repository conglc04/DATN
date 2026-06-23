"""Generate synthetic SUMO FCD traces for the Bạch Mai MCI scenario.

Produces SUMO-compatible FCD XML (geo=True: x=lon, y=lat) without requiring
SUMO to be installed. Traces mimic the output of a SUMO simulation where 3
ambulances converge toward Bạch Mai Hospital (gNB location) within a 300m cell.

Scenario (M10.2):
  - gNB at Bạch Mai Hospital: 21.002966°N, 105.840780°E  (SSOT: utils/config.py)
  - K=3 ambulances at different bearings, all within 300m cell
  - Speed: 60 km/h = 16.67 m/s (hard-mission default)
  - Episode duration: 1.0s → FCD timestep 0.1s (11 snapshots)

Usage:
    cd data/sumo
    python generate_traces.py

Outputs:
    bachmaiHN_mci_k1.fcd.xml
    bachmaiHN_mci_k3.fcd.xml

Reference:
    sumo-user.pdf §FCD (Floating Car Data output format)
    Guastella 2023 §netconvert + simulation workflow
"""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET

# ---------------------------------------------------------------------------
# Geographic constants
# ---------------------------------------------------------------------------

GNB_LAT: float = 21.002965894776974   # SSOT from utils/config.BACH_MAI_LAT
GNB_LON: float = 105.84078002433277   # SSOT from utils/config.BACH_MAI_LON
METERS_PER_DEG_LAT: float = 111_320.0
METERS_PER_DEG_LON: float = 111_320.0 * math.cos(math.radians(GNB_LAT))


def metric_to_gps(x_m: float, y_m: float) -> tuple[float, float]:
    """Convert (east_m, north_m) relative to gNB → (lat, lon)."""
    lat = GNB_LAT + y_m / METERS_PER_DEG_LAT
    lon = GNB_LON + x_m / METERS_PER_DEG_LON
    return lat, lon


def bearing_deg(x_m: float, y_m: float) -> float:
    """Compass bearing of velocity vector (north=0°, east=90°)."""
    angle_rad = math.atan2(x_m, y_m)   # atan2(east, north)
    return math.degrees(angle_rad) % 360.0


# ---------------------------------------------------------------------------
# Ambulance trajectories (converging toward gNB at origin)
# ---------------------------------------------------------------------------

SPEED_MS: float = 60.0 / 3.6   # 60 km/h = 16.667 m/s
FCD_DT: float = 0.1             # FCD timestep (s)
N_STEPS: int = 11               # 0.0 to 1.0 inclusive


def _convergence_trajectory(
    x0_m: float, y0_m: float, speed_ms: float = SPEED_MS
) -> list[tuple[float, float]]:
    """Generate (x_m, y_m) positions converging toward origin (gNB).

    Velocity = -normalize(start_pos) * speed (straight line, no jitter).
    """
    dist = math.sqrt(x0_m ** 2 + y0_m ** 2)
    if dist == 0:
        return [(0.0, 0.0)] * N_STEPS
    vx = -x0_m / dist * speed_ms
    vy = -y0_m / dist * speed_ms
    return [(x0_m + vx * i * FCD_DT, y0_m + vy * i * FCD_DT) for i in range(N_STEPS)]


# Ambulance start positions (metric, relative to gNB)
AMB_STARTS: dict[str, tuple[float, float]] = {
    "amb_0": (0.0,   200.0),    # 200m north → heading south
    "amb_1": (-90.0, -155.0),   # 180m SW (210°) → heading NE
    "amb_2": (129.0, -153.0),   # 200m SE (140°) → heading NW
}


# ---------------------------------------------------------------------------
# XML generation
# ---------------------------------------------------------------------------

def _vehicle_element(veh_id: str, lat: float, lon: float, speed_ms: float,
                     vx: float, vy: float) -> ET.Element:
    el = ET.Element("vehicle")
    el.set("id", veh_id)
    el.set("x", f"{lon:.6f}")    # SUMO geo: x=lon
    el.set("y", f"{lat:.6f}")    # SUMO geo: y=lat
    el.set("angle", f"{bearing_deg(vx, vy):.2f}")
    el.set("speed", f"{speed_ms:.4f}")
    return el


def build_fcd(ambulance_ids: list[str]) -> ET.Element:
    """Build full FCD XML for given ambulance IDs."""
    root = ET.Element("fcd-export")
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root.set("xsi:noNamespaceSchemaLocation",
             "http://sumo.dlr.de/xsd/fcd_file.xsd")

    # Precompute trajectories and velocities
    trajs: dict[str, list[tuple[float, float]]] = {}
    vels: dict[str, tuple[float, float]] = {}
    for aid in ambulance_ids:
        x0, y0 = AMB_STARTS[aid]
        trajs[aid] = _convergence_trajectory(x0, y0)
        dist = math.sqrt(x0 ** 2 + y0 ** 2)
        vels[aid] = (-x0 / dist * SPEED_MS, -y0 / dist * SPEED_MS) if dist > 0 else (0.0, 0.0)

    for step in range(N_STEPS):
        ts_el = ET.SubElement(root, "timestep")
        ts_el.set("time", f"{step * FCD_DT:.2f}")
        for aid in ambulance_ids:
            x_m, y_m = trajs[aid][step]
            lat, lon = metric_to_gps(x_m, y_m)
            vx, vy = vels[aid]
            ts_el.append(_vehicle_element(aid, lat, lon, SPEED_MS, vx, vy))
    return root


def _indent(elem: ET.Element, level: int = 0) -> None:
    """In-place pretty-print indentation for ET."""
    indent = "\n" + "    " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "    "
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent
        for child in elem:
            _indent(child, level + 1)
        if not child.tail or not child.tail.strip():  # noqa: F821
            child.tail = indent
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = indent
    if not level:
        elem.tail = "\n"


def write_fcd(ambulance_ids: list[str], out_path: str) -> None:
    root = build_fcd(ambulance_ids)
    _indent(root)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(
            f"<!-- SUMO FCD trace: Bạch Mai MCI scenario (synthetic)\n"
            f"     Ambulances: {ambulance_ids}\n"
            f"     Origin (gNB): {GNB_LAT}°N {GNB_LON}°E\n"
            f"     Speed: {SPEED_MS:.2f} m/s ({SPEED_MS * 3.6:.0f} km/h)\n"
            f"     Timestep: {FCD_DT}s | Duration: {(N_STEPS-1)*FCD_DT}s\n"
            f"-->\n"
        )
        tree.write(f, encoding="unicode", xml_declaration=False)
    print(f"Written: {out_path}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))

    write_fcd(["amb_0"], os.path.join(here, "bachmaiHN_mci_k1.fcd.xml"))
    write_fcd(["amb_0", "amb_1", "amb_2"], os.path.join(here, "bachmaiHN_mci_k3.fcd.xml"))
    print("Done.")
