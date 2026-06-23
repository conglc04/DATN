"""Tier-A density sweep — generate K×density FCD traces via SUMO.

For each (K, density_profile) combination:
  1. Run randomTrips.py to generate N_BG background vehicle trips
     Period = SIM_DURATION / N_BG  → exactly N_BG departures over [0, SIM_DURATION)
  2. Run SUMO with background trips + ambulance routes
     - Traffic lights: SUMO-generated via --tls.guess (from bachmaiHN.net.xml)
     - Car-following: Krauss model (SUMO default) with IDM fallback
     - --device.fcd.explicit <amb_ids> → FCD only for ambulances (small files)
  3. Verify output contains ambulance timesteps

Simulation specs (DECLARED — not Hà Nội ground truth):
  SIM_DURATION = 400 s   (matches RL episode max; ensures all ambulances reach 37370971#0)
  STEP_LENGTH  = 0.1 s   (same as existing single-density traces)
  Ambulances depart at t=0; background vehicles spread across [0, SIM_DURATION)

Density profiles (background vehicles, SIM_DURATION=400s):
  Vehicle counts scaled from 300s baseline to maintain the same spawning period
  (concurrent vehicle count ≈ constant across 300s and 400s runs).
  Tier-A:
    light  = 67  bg vehicles  →  period ≈ 5.97 s/vehicle  (≈300s baseline: 50 veh/6.0s)
    medium = 267 bg vehicles  →  period ≈ 1.50 s/vehicle  (≈300s baseline: 200 veh/1.5s)
    heavy  = 667 bg vehicles  →  period ≈ 0.60 s/vehicle  (≈300s baseline: 500 veh/0.6s)
  Smoke (pipeline test only — not for training):
    light_smoke  = 13  bg vehicles  →  period ≈ 30.8 s/vehicle
    medium_smoke = 67  bg vehicles  →  period ≈ 5.97 s/vehicle
    heavy_smoke  = 200 bg vehicles  →  period = 2.00 s/vehicle

CLAIM: Synthetic density sweep over a SUMO-modeled network.
  - NOT actual Hà Nội traffic density or timing.
  - TLS: SUMO-generated via netconvert --tls.guess (OSM highway=traffic_signals tags).
    Cycle times are SUMO heuristic defaults, not field-measured.
  - Car-following: Krauss stochastic (SUMO default), not calibrated to Hà Nội drivers.
  - Background vehicles: random passenger cars on all accessible edges.
    No motorbike, bus, or pedestrian models.

Output: data/sumo/density/bachmaiHN_mci_k{K}_{profile}.fcd.xml
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
DENSITY_DIR = os.path.join(HERE, "density")
LOG_DIR = os.path.join(DENSITY_DIR, "logs")
NET_FILE = os.path.join(HERE, "bachmaiHN.net.xml")
RANDOM_TRIPS = "/usr/share/sumo/tools/randomTrips.py"

os.makedirs(DENSITY_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Simulation constants (DECLARED — see module docstring)
# ---------------------------------------------------------------------------

SIM_DURATION = 400       # seconds — matches RL episode max; amb_2 arrives ~372s
STEP_LENGTH  = 0.1       # seconds — same as existing traces
BG_SEED      = 42        # reproducibility seed for randomTrips

# ---------------------------------------------------------------------------
# Density profiles
# ---------------------------------------------------------------------------

PROFILES: dict[str, int] = {
    # Tier-A (training) — scaled to maintain period ≈ same as 300s baseline
    "light":        67,   # ≈50×(400/300); period ≈5.97s (baseline 6.0s)
    "medium":       267,  # ≈200×(400/300); period ≈1.50s
    "heavy":        667,  # ≈500×(400/300); period ≈0.60s
    # Smoke (pipeline test only) — scaled proportionally
    "light_smoke":  13,   # ≈10×(400/300); period ≈30.8s
    "medium_smoke": 67,   # ≈50×(400/300); period ≈5.97s
    "heavy_smoke":  200,  # =150×(400/300); period =2.00s
}

# K values and their ambulance IDs
K_CONFIGS: dict[int, list[str]] = {
    1: ["amb_0"],
    3: ["amb_0", "amb_1", "amb_2"],
}

# ---------------------------------------------------------------------------
# Prerequisite checks
# ---------------------------------------------------------------------------

def check_prereqs() -> None:
    if not os.path.exists(NET_FILE):
        sys.exit(f"ERROR: {NET_FILE} not found. Run 02_build_network.sh first.")
    if not os.path.exists(RANDOM_TRIPS):
        sys.exit(f"ERROR: {RANDOM_TRIPS} not found. Install sumo-tools.")
    for K in K_CONFIGS:
        amb_routes = os.path.join(HERE, f"ambulance_routes_k{K}.xml")
        if not os.path.exists(amb_routes):
            sys.exit(f"ERROR: {amb_routes} not found. Run 03_generate_routes.py first.")
    print("[prereq] All prerequisites satisfied.")


def count_tls() -> int:
    """Count <tlLogic> entries in the network file."""
    count = 0
    for _, elem in ET.iterparse(NET_FILE, events=["start"]):
        if elem.tag == "tlLogic":
            count += 1
        elem.clear()
    return count


# ---------------------------------------------------------------------------
# Background trip generation
# ---------------------------------------------------------------------------

def generate_bg_trips(
    profile: str,
    n_bg: int,
    K: int,
    out_path: str,
) -> None:
    """Run randomTrips.py to generate background vehicle trips."""
    period = SIM_DURATION / n_bg

    cmd = [
        sys.executable, RANDOM_TRIPS,
        "--net-file",     NET_FILE,
        "--output-trip-file", out_path,
        "--begin",        "0",
        "--end",          str(SIM_DURATION),
        "--period",       f"{period:.4f}",
        "--vehicle-class","passenger",
        "--prefix",       "bg_",
        "--seed",         str(BG_SEED),
        "--fringe-factor","10",          # prefer fringe edges as trip sources
        "--min-distance", "100",         # avoid trivial trips
        "--remove-loops",                # no U-turns at origin
    ]
    log_path = os.path.join(LOG_DIR, f"randomtrips_{profile}_k{K}.log")
    print(f"  [randomTrips] {profile} K={K}: n_bg={n_bg}, period={period:.2f}s")
    with open(log_path, "w") as log_f:
        result = subprocess.run(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )
    if result.returncode != 0:
        sys.exit(
            f"ERROR: randomTrips.py failed for {profile} K={K}. "
            f"See {log_path}"
        )
    if not os.path.exists(out_path):
        sys.exit(
            f"ERROR: randomTrips.py did not produce {out_path}. "
            f"See {log_path}"
        )
    print(f"  [randomTrips] OK → {os.path.basename(out_path)}")


# ---------------------------------------------------------------------------
# SUMO simulation
# ---------------------------------------------------------------------------

def run_sumo(
    profile: str,
    K: int,
    ambulance_ids: list[str],
    bg_trips_path: str,
    out_fcd_path: str,
) -> None:
    """Run SUMO with background trips + ambulance routes."""
    amb_routes = os.path.join(HERE, f"ambulance_routes_k{K}.xml")
    log_path = os.path.join(LOG_DIR, f"sumo_{profile}_k{K}.log")

    # --device.fcd.explicit: comma-separated list of vehicle IDs to track.
    # Only ambulances get FCD output → files stay small regardless of bg count.
    amb_explicit = ",".join(ambulance_ids)

    # FCD filter strategy: probability=0 disables FCD for ALL vehicles by default;
    # explicit= re-enables only for the listed ambulance IDs.
    # This is more reliable than explicit alone (which adds the device but doesn't
    # suppress the default all-vehicles output in SUMO 1.12).
    temp_fcd_path = out_fcd_path + ".tmp"
    cmd = [
        "sumo",
        "--net-file",           NET_FILE,
        "--route-files",        f"{bg_trips_path},{amb_routes}",
        "--fcd-output",         temp_fcd_path,
        "--fcd-output.geo",                     # x=lon, y=lat (sumo_mobility.py expects this)
        "--device.fcd.probability", "0",        # disable FCD for all vehicles by default
        "--device.fcd.explicit",    amb_explicit, # re-enable for ambulances only
        "--device.fcd.deterministic",           # deterministic (not probabilistic) assignment
        # Emergency siren: ambulances run with bluelight (vClass=emergency, always active).
        # Background vehicles yield at reactiondist=25m; ambulances pass red lights.
        "--device.bluelight.explicit",    amb_explicit,
        "--device.bluelight.reactiondist", "25",
        "--step-length",        str(STEP_LENGTH),
        "--begin",              "0",
        "--end",                str(SIM_DURATION),
        "--no-step-log",
        "--no-warnings",
        "--ignore-route-errors",                # tolerate occasional unroutable bg trips
    ]
    print(f"  [sumo]        {profile} K={K}: sim {SIM_DURATION}s, step {STEP_LENGTH}s")
    with open(log_path, "w") as log_f:
        result = subprocess.run(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )
    if result.returncode != 0:
        sys.exit(
            f"ERROR: SUMO failed for {profile} K={K}. See {log_path}"
        )
    if not os.path.exists(temp_fcd_path):
        sys.exit(
            f"ERROR: SUMO did not produce FCD output for {profile} K={K}. "
            f"See {log_path}"
        )

    # Post-process: filter to ambulance-only as a safety net.
    # Even if device filtering worked, this guarantees clean output.
    ambulance_set = set(ambulance_ids)
    _filter_fcd_to_vehicles(temp_fcd_path, out_fcd_path, ambulance_set)
    os.remove(temp_fcd_path)
    print(f"  [sumo]        OK → {os.path.basename(out_fcd_path)}")


# ---------------------------------------------------------------------------
# FCD post-processing filter
# ---------------------------------------------------------------------------

def _filter_fcd_to_vehicles(src: str, dst: str, keep_ids: set[str]) -> None:
    """Stream-filter an FCD XML: keep only <vehicle id=...> in keep_ids.

    Reads src line-by-line (no full-DOM load) so it handles large files
    (500 bg vehicles × 3000 steps) without excessive memory use.
    Writes dst with only the ambulance vehicle elements retained.
    Empty <timestep> elements are preserved (SUMO reader expects them).
    """
    import re

    vehicle_pat = re.compile(r'<vehicle\s+id="([^"]+)"')
    timestep_open = re.compile(r'<timestep\b')
    timestep_close = re.compile(r'</timestep>')

    with open(src, "r", encoding="utf-8") as fin, \
         open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            # Check if this is a <vehicle ...> line
            m = vehicle_pat.search(line)
            if m:
                if m.group(1) in keep_ids:
                    fout.write(line)
                # else: skip background vehicle
            else:
                fout.write(line)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_fcd(fcd_path: str, ambulance_ids: list[str]) -> dict[str, int]:
    """Count timesteps per ambulance in the FCD file."""
    counts: dict[str, int] = {aid: 0 for aid in ambulance_ids}
    for _, elem in ET.iterparse(fcd_path, events=["start"]):
        if elem.tag == "vehicle":
            vid = elem.get("id", "")
            if vid in counts:
                counts[vid] += 1
        elem.clear()
    return counts


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_profile(profile: str, n_bg: int, K: int, ambulance_ids: list[str]) -> bool:
    """Run full pipeline for one (profile, K) combination.

    Returns True if succeeded, False if skipped (output already exists).
    """
    out_fcd = os.path.join(DENSITY_DIR, f"bachmaiHN_mci_k{K}_{profile}.fcd.xml")
    if os.path.exists(out_fcd):
        print(f"  [skip]        {out_fcd} already exists (--force to overwrite)")
        return False

    # Step 1: background trips
    bg_trips = os.path.join(
        DENSITY_DIR, "logs", f"bg_trips_{profile}_k{K}.xml"
    )
    generate_bg_trips(profile, n_bg, K, bg_trips)

    # Step 2: SUMO simulation
    run_sumo(profile, K, ambulance_ids, bg_trips, out_fcd)

    # Step 3: verify
    counts = verify_fcd(out_fcd, ambulance_ids)
    total_timesteps = sum(counts.values())
    if total_timesteps == 0:
        print(
            f"  [WARN]        No ambulance timesteps in {out_fcd}! "
            "Check ambulance routes reach destination."
        )
    else:
        step_summary = ", ".join(f"{k}={v}" for k, v in counts.items())
        print(f"  [verify]      timesteps: {step_summary} (total={total_timesteps})")

    return True


def main(force: bool = False) -> None:
    check_prereqs()

    # Report TLS count (logged once — network is shared across all runs)
    print("Counting tlLogic entries in network (this may take a moment)…")
    n_tls = count_tls()
    print(
        f"[tls]   {n_tls} tlLogic entries in bachmaiHN.net.xml\n"
        f"        Source: SUMO-generated via netconvert --tls.guess\n"
        f"        Claim:  SUMO heuristic TLS (NOT field-measured Hà Nội timing)\n"
    )

    print(
        f"[config] SIM_DURATION={SIM_DURATION}s  STEP_LENGTH={STEP_LENGTH}s  "
        f"BG_SEED={BG_SEED}\n"
        f"         Claim: synthetic density sweep — NOT Hà Nội ground-truth density\n"
    )

    if force:
        print("[force]  --force flag set: will overwrite existing FCD files\n")

    results: list[str] = []
    errors: list[str] = []

    for profile, n_bg in PROFILES.items():
        period = SIM_DURATION / n_bg
        tier = "smoke" if "smoke" in profile else "Tier-A"
        print(
            f"\n{'='*60}\n"
            f"Profile: {profile}  ({tier})\n"
            f"  n_bg={n_bg} background vehicles,  period={period:.2f}s/vehicle\n"
            f"  Concurrent on-road estimate (assume 90s avg trip): "
            f"~{int(n_bg * 90 / SIM_DURATION)} vehicles at steady state\n"
            f"{'='*60}"
        )

        for K, amb_ids in K_CONFIGS.items():
            print(f"\n-- K={K} --")
            if force:
                out_fcd = os.path.join(
                    DENSITY_DIR, f"bachmaiHN_mci_k{K}_{profile}.fcd.xml"
                )
                if os.path.exists(out_fcd):
                    os.remove(out_fcd)
            try:
                run_profile(profile, n_bg, K, amb_ids)
                tag = f"k{K}_{profile}"
                results.append(tag)
            except SystemExit as e:
                errors.append(f"k{K}_{profile}: {e}")
                print(f"  [ERROR] {e}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Done. {len(results)} FCD files generated in {DENSITY_DIR}/")
    if errors:
        print(f"\n{len(errors)} errors:")
        for e in errors:
            print(f"  FAILED: {e}")
        sys.exit(1)
    else:
        print(
            "\nNext step: update PooledSumoMobilityProvider route pool to include\n"
            "  density-variant paths via density_fcd_path(K, density).\n"
            "  Run: pytest baselines/tests/test_density_sweep_pipeline.py -v"
        )


if __name__ == "__main__":
    force_flag = "--force" in sys.argv
    main(force=force_flag)
