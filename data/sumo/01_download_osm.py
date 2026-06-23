"""Step 1 — Download OSM data for Bạch Mai area via Overpass API.

Bounding box: centred on Bạch Mai Hospital gNB (21.002966°N, 105.840780°E),
±2.5km each side.  Saves raw OSM XML → bachmaiHN.osm

GNB_LAT / GNB_LON are the SSOT values from utils/config.py
(BACH_MAI_LAT / BACH_MAI_LON).  Do NOT use rounded approximations.

No SUMO binary required for this step.
"""

from __future__ import annotations

import math
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

# SSOT gNB anchor — must match utils/config.BACH_MAI_LAT / BACH_MAI_LON exactly
GNB_LAT = 21.002965894776974
GNB_LON = 105.84078002433277

# 2.5 km offset in degrees
LAT_OFF = 2500 / 111_320
LON_OFF = 2500 / (111_320 * math.cos(math.radians(GNB_LAT)))

BBOX = (
    GNB_LAT - LAT_OFF,  # south
    GNB_LON - LON_OFF,  # west
    GNB_LAT + LAT_OFF,  # north
    GNB_LON + LON_OFF,  # east
)

# Overpass query: roads + traffic signals only (no buildings, POI)
OVERPASS_QUERY = f"""
[out:xml][timeout:60];
(
  way["highway"]({BBOX[0]:.6f},{BBOX[1]:.6f},{BBOX[2]:.6f},{BBOX[3]:.6f});
  node["highway"="traffic_signals"]({BBOX[0]:.6f},{BBOX[1]:.6f},{BBOX[2]:.6f},{BBOX[3]:.6f});
);
out body;
>;
out skel qt;
""".strip()

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OUT_OSM = os.path.join(HERE, "bachmaiHN.osm")


def download() -> None:
    print(f"Downloading OSM bbox: {BBOX}")
    import urllib.parse
    url = OVERPASS_URL + "?data=" + urllib.parse.quote(OVERPASS_QUERY)
    req = urllib.request.Request(url, headers={"User-Agent": "thesis-sumo-pipeline/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        content = resp.read()
    with open(OUT_OSM, "wb") as f:
        f.write(content)
    size_kb = os.path.getsize(OUT_OSM) / 1024
    print(f"Saved: {OUT_OSM}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    download()
