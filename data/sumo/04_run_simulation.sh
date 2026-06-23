#!/usr/bin/env bash
# Step 4 — Run SUMO simulation and export FCD traces.
# Requires: sumo binary (sudo apt-get install -y sumo sumo-tools)
#
# Output:
#   bachmaiHN_mci_k1.fcd.xml   — K=1 scenario (overwrites synthetic version)
#   bachmaiHN_mci_k3.fcd.xml   — K=3 scenario (overwrites synthetic version)
#
# FCD format: --fcd-output.geo true → x=lon, y=lat (same format sumo_mobility.py expects)
#
# Reference: sumo-user.pdf §FCD output; Guastella 2023 §simulation

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
NET="$HERE/bachmaiHN.net.xml"

if [ ! -f "$NET" ]; then
    echo "ERROR: $NET not found. Run 02_build_network.sh first."
    exit 1
fi

run_sumo() {
    local ROUTES="$1"
    local OUT_FCD="$2"
    local K="$3"

    echo "Running K=$K simulation → $OUT_FCD"
    sumo \
        --net-file "$NET" \
        --route-files "$ROUTES" \
        --fcd-output "$OUT_FCD" \
        --fcd-output.geo \
        --step-length 0.1 \
        --begin 0 \
        --end 400.0 \
        --no-step-log \
        --no-warnings \
        2>&1 | tee "$HERE/sumo_k${K}.log"

    echo "FCD written: $OUT_FCD"
}

# K=1
if [ -f "$HERE/ambulance_routes_k1.xml" ]; then
    run_sumo \
        "$HERE/ambulance_routes_k1.xml" \
        "$HERE/bachmaiHN_mci_k1.fcd.xml" \
        1
else
    echo "WARN: ambulance_routes_k1.xml not found; run 03_generate_routes.py first"
fi

# K=3
if [ -f "$HERE/ambulance_routes_k3.xml" ]; then
    run_sumo \
        "$HERE/ambulance_routes_k3.xml" \
        "$HERE/bachmaiHN_mci_k3.fcd.xml" \
        3
else
    echo "WARN: ambulance_routes_k3.xml not found; run 03_generate_routes.py first"
fi

echo "All done. Verify with: python3 -m pytest baselines/tests/test_sumo_mobility.py -v"
