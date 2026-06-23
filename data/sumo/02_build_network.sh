#!/usr/bin/env bash
# Step 2 — Convert OSM → SUMO network via netconvert.
# Requires: sudo apt-get install -y sumo sumo-tools
#
# Reference: sumo-user.pdf §netconvert; Guastella 2023 §netconvert workflow
#
# Fidelity choices (M10.1b):
#   - Traffic signals: --tls.guess (auto-detect from highway=traffic_signals tags)
#   - Speed limits:    kept from maxspeed OSM tag
#   - Background density sweep: done in step 03 via randomTrips

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OSM="$HERE/bachmaiHN.osm"
NET="$HERE/bachmaiHN.net.xml"

if [ ! -f "$OSM" ]; then
    echo "ERROR: $OSM not found. Run 01_download_osm.py first."
    exit 1
fi

netconvert \
    --osm-files "$OSM" \
    --output-file "$NET" \
    --tls.guess true \
    --tls.guess.threshold 0 \
    --geometry.remove true \
    --roundabouts.guess true \
    --junctions.join true \
    --keep-edges.by-vclass passenger \
    --proj "+proj=utm +zone=48 +ellps=WGS84 +datum=WGS84 +units=m +no_defs" \
    --proj.plain-geo true \
    2>&1 | tee "$HERE/netconvert.log"

echo "Network written: $NET"
