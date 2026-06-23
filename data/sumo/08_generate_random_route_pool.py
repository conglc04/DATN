"""Step 8 — Generate random-start route pool for training diversity.

For each (K, variant_index, density) combination:
  1. Sample N_VARIANTS random starting edges from the OSM network, each
     OUTSIDE the 1-km cell (FROM-node dist ∈ [R_CELL_M, MAX_DIST_M]).
  2. Write an ambulance route XML for that variant.
  3. Run SUMO with background traffic (light/medium/heavy) → FCD output
     containing only ambulance traces (same filter as 07_generate_density_sweep.py).

Outputs:
  data/sumo/variants/ambulance_routes_k{K}_v{i:02d}.xml   (route specs)
  data/sumo/density/variants/bachmaiHN_mci_k{K}_v{i:02d}_{density}.fcd.xml

The script is idempotent: existing FCD files are skipped unless --force is given.

Usage:
  python3 data/sumo/08_generate_random_route_pool.py [--variants 30] [--seed 0] [--force]
  python3 data/sumo/08_generate_random_route_pool.py --variants 10 --seed 42 --force

Runtime estimate (Intel i7, SUMO 1.12):
  ~15 s per SUMO run × N_VARIANTS × len(K_CONFIGS) × len(DENSITIES)
  Default (30 variants × 2 K × 3 densities = 180 runs) ≈ 45 min.

CLAIM: synthetic diversity — NOT Hà Nội ground-truth traffic or road conditions.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import subprocess
import sys
import xml.etree.ElementTree as ET

import sumolib

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
NET_FILE = os.path.join(HERE, "bachmaiHN.net.xml")
DENSITY_DIR = os.path.join(HERE, "density")
VARIANTS_ROUTE_DIR = os.path.join(HERE, "variants")
VARIANTS_FCD_DIR = os.path.join(DENSITY_DIR, "variants")
LOG_DIR = os.path.join(VARIANTS_FCD_DIR, "logs")
RANDOM_TRIPS = "/usr/share/sumo/tools/randomTrips.py"

# ---------------------------------------------------------------------------
# Config SSOT imports
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.join(HERE, "..", "..", "baselines"))
from utils.config import (  # noqa: E402
    BACH_MAI_LAT as GNB_LAT,
    BACH_MAI_LON as GNB_LON,
    R_CELL_M,
)

# ---------------------------------------------------------------------------
# Simulation constants (match 07_generate_density_sweep.py)
# ---------------------------------------------------------------------------

SIM_DURATION = 400       # seconds
STEP_LENGTH  = 0.1       # seconds
BG_SEED      = 42        # background trip seed (fixed for reproducibility)

# Starting-edge search band — ambulances must start OUTSIDE the cell.
# Upper bound 2.5 × R_CELL ensures travel time ≤ 2500 m / 8 m/s ≈ 313 s < 400 s.
MIN_DIST_M = R_CELL_M           # 1000 m
MAX_DIST_M = R_CELL_M * 2.5    # 2500 m

# Destination edge: centroid of BV Bạch Mai stopping cluster (same as existing routes)
DEST_EDGE_ID = "37370971#0"

# Density profiles (background vehicle counts) — Tier-A only (no smoke variants)
DENSITIES: dict[str, int] = {
    "light":  67,
    "medium": 267,
    "heavy":  667,
}

K_CONFIGS: dict[int, list[str]] = {
    1: ["amb_0"],
    3: ["amb_0", "amb_1", "amb_2"],
}

SPEED_MS = 60.0 / 3.6   # 16.67 m/s — matches ambulance vType in existing routes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def find_candidate_edges(net: sumolib.net.Net) -> list[sumolib.net.edge.Edge]:
    """Return all edges whose FROM-node is in [MIN_DIST_M, MAX_DIST_M] and
    FROM-dist > TO-dist (vehicle converges toward gNB when traversing the edge).

    Excludes internal/junction edges and edges with no lanes.
    """
    gx, gy = net.convertLonLat2XY(GNB_LON, GNB_LAT)

    raw = net.getNeighboringEdges(gx, gy, r=MAX_DIST_M, includeJunctions=False)

    candidates: list[sumolib.net.edge.Edge] = []
    for edge, _ in raw:
        if edge.getFunction() == "internal":
            continue
        if edge.getLaneNumber() == 0:
            continue
        fx, fy = edge.getFromNode().getCoord()
        tx, ty = edge.getToNode().getCoord()
        fd = _dist(fx, fy, gx, gy)
        td = _dist(tx, ty, gx, gy)
        if MIN_DIST_M <= fd <= MAX_DIST_M and fd > td:
            candidates.append(edge)

    return candidates


def sample_starts_k1(
    candidates: list[sumolib.net.edge.Edge],
    n: int,
    rng: random.Random,
) -> list[list[sumolib.net.edge.Edge]]:
    """Return n single-edge start lists for K=1."""
    pool = list(candidates)
    rng.shuffle(pool)
    return [[pool[i % len(pool)]] for i in range(n)]


def sample_starts_k3(
    candidates: list[sumolib.net.edge.Edge],
    n: int,
    rng: random.Random,
) -> list[list[sumolib.net.edge.Edge]]:
    """Return n 3-edge start lists for K=3 (distinct edges per variant)."""
    if len(candidates) < 3:
        raise RuntimeError(
            f"Need ≥3 candidate edges for K=3 (found {len(candidates)})"
        )
    result: list[list[sumolib.net.edge.Edge]] = []
    for _ in range(n):
        result.append(rng.sample(candidates, 3))
    return result


# ---------------------------------------------------------------------------
# Route file writer
# ---------------------------------------------------------------------------

def write_route_file(
    start_edges: list[sumolib.net.edge.Edge],
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

    for aid, edge in zip(ambulance_ids, start_edges):
        trip = ET.SubElement(root, "trip")
        trip.set("id", aid)
        trip.set("type", "ambulance")
        trip.set("from", edge.getID())
        trip.set("to", DEST_EDGE_ID)
        trip.set("depart", "0.00")
        trip.set("departSpeed", "max")

    ET.indent(root, space="    ")
    tree = ET.ElementTree(root)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        tree.write(f, encoding="unicode", xml_declaration=False)


# ---------------------------------------------------------------------------
# Background trip generation (reused from 07_generate_density_sweep.py)
# ---------------------------------------------------------------------------

def generate_bg_trips(profile: str, n_bg: int, K: int, out_path: str) -> None:
    period = SIM_DURATION / n_bg
    cmd = [
        sys.executable, RANDOM_TRIPS,
        "--net-file",         NET_FILE,
        "--output-trip-file", out_path,
        "--begin",            "0",
        "--end",              str(SIM_DURATION),
        "--period",           f"{period:.4f}",
        "--vehicle-class",    "passenger",
        "--prefix",           "bg_",
        "--seed",             str(BG_SEED),
        "--fringe-factor",    "10",
        "--min-distance",     "100",
        "--remove-loops",
    ]
    log_path = os.path.join(LOG_DIR, f"randomtrips_{profile}_k{K}.log")
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(log_path, "w") as lf:
        result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        sys.exit(f"ERROR: randomTrips.py failed for {profile} K={K}. See {log_path}")


def _filter_fcd(src: str, dst: str, keep_ids: set[str]) -> None:
    """Stream-filter FCD to keep only ambulance vehicles."""
    import re
    vehicle_pat = re.compile(r'<vehicle\s+id="([^"]+)"')
    with open(src, "r", encoding="utf-8") as fin, \
         open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            m = vehicle_pat.search(line)
            if m:
                if m.group(1) in keep_ids:
                    fout.write(line)
            else:
                fout.write(line)


def run_sumo_variant(
    K: int,
    variant_idx: int,
    ambulance_ids: list[str],
    density: str,
    n_bg: int,
    route_file: str,
) -> str:
    """Run SUMO for one (K, variant, density) → returns FCD output path."""
    tag = f"k{K}_v{variant_idx:02d}_{density}"
    out_fcd = os.path.join(VARIANTS_FCD_DIR, f"bachmaiHN_mci_k{K}_v{variant_idx:02d}_{density}.fcd.xml")

    # Background trips (shared across variants for same density+K)
    bg_trips = os.path.join(LOG_DIR, f"bg_trips_{density}_k{K}.xml")
    if not os.path.exists(bg_trips):
        print(f"    [randomTrips] generating bg trips for {density} K={K}…")
        generate_bg_trips(density, n_bg, K, bg_trips)

    log_path = os.path.join(LOG_DIR, f"sumo_{tag}.log")
    amb_explicit = ",".join(ambulance_ids)
    temp_fcd = out_fcd + ".tmp"

    cmd = [
        "sumo",
        "--net-file",               NET_FILE,
        "--route-files",            f"{bg_trips},{route_file}",
        "--fcd-output",             temp_fcd,
        "--fcd-output.geo",
        "--device.fcd.probability", "0",
        "--device.fcd.explicit",    amb_explicit,
        "--device.fcd.deterministic",
        # Emergency siren: ambulances pass red lights; background vehicles yield at 25m.
        "--device.bluelight.explicit",     amb_explicit,
        "--device.bluelight.reactiondist", "25",
        "--step-length",            str(STEP_LENGTH),
        "--begin",                  "0",
        "--end",                    str(SIM_DURATION),
        "--no-step-log",
        "--no-warnings",
        "--ignore-route-errors",
    ]
    with open(log_path, "w") as lf:
        result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)

    if result.returncode != 0:
        print(f"    [WARN] SUMO non-zero exit for {tag} — skipping. See {log_path}")
        if os.path.exists(temp_fcd):
            os.remove(temp_fcd)
        return ""

    if not os.path.exists(temp_fcd):
        # SUMO exited cleanly but produced no FCD — ambulance edge is unroutable.
        print(
            f"    [WARN] SUMO produced no FCD for {tag} (unroutable start edge?) "
            f"— skipping. See {log_path}"
        )
        return ""

    _filter_fcd(temp_fcd, out_fcd, set(ambulance_ids))
    os.remove(temp_fcd)
    return out_fcd


def verify_fcd(fcd_path: str, ambulance_ids: list[str]) -> int:
    total = 0
    ids = set(ambulance_ids)
    for _, elem in ET.iterparse(fcd_path, events=["start"]):
        if elem.tag == "vehicle" and elem.get("id", "") in ids:
            total += 1
        elem.clear()
    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(n_variants: int, seed: int, force: bool) -> None:
    for prereq in [NET_FILE, RANDOM_TRIPS]:
        if not os.path.exists(prereq):
            sys.exit(f"ERROR: {prereq} not found.")

    os.makedirs(VARIANTS_ROUTE_DIR, exist_ok=True)
    os.makedirs(VARIANTS_FCD_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"Loading network: {NET_FILE}")
    net = sumolib.net.readNet(NET_FILE, withInternal=False)

    print("Finding candidate edges outside 1-km cell…")
    candidates = find_candidate_edges(net)
    print(f"  {len(candidates)} candidate edges in [{MIN_DIST_M:.0f}m, {MAX_DIST_M:.0f}m]")

    rng = random.Random(seed)
    starts: dict[int, list[list[sumolib.net.edge.Edge]]] = {
        1: sample_starts_k1(candidates, n_variants, rng),
        3: sample_starts_k3(candidates, n_variants, rng),
    }

    total_runs = n_variants * len(K_CONFIGS) * len(DENSITIES)
    done = skipped = failed = 0

    print(
        f"\nGenerating {n_variants} variants × {len(K_CONFIGS)} K-values "
        f"× {len(DENSITIES)} densities = {total_runs} SUMO runs\n"
        f"Seed={seed}, SIM_DURATION={SIM_DURATION}s, STEP_LENGTH={STEP_LENGTH}s\n"
    )

    for K, amb_ids in K_CONFIGS.items():
        for i, start_edges in enumerate(starts[K]):
            route_file = os.path.join(
                VARIANTS_ROUTE_DIR, f"ambulance_routes_k{K}_v{i:02d}.xml"
            )
            # Write route file (idempotent — always rewrite; it's tiny)
            write_route_file(start_edges, amb_ids, route_file)
            edge_summary = ", ".join(
                f"{e.getID()}" for e in start_edges
            )
            print(f"  K={K} v{i:02d}: start edges [{edge_summary}]")

            for density, n_bg in DENSITIES.items():
                out_fcd = os.path.join(
                    VARIANTS_FCD_DIR,
                    f"bachmaiHN_mci_k{K}_v{i:02d}_{density}.fcd.xml",
                )
                if os.path.exists(out_fcd) and not force:
                    print(f"    [skip] {os.path.basename(out_fcd)}")
                    skipped += 1
                    continue

                print(f"    [sumo] K={K} v{i:02d} {density} ({n_bg} bg vehicles)…")
                result_path = run_sumo_variant(K, i, amb_ids, density, n_bg, route_file)

                if not result_path:
                    failed += 1
                    continue

                n_ts = verify_fcd(result_path, amb_ids)
                if n_ts == 0:
                    print(f"    [WARN] {os.path.basename(out_fcd)}: 0 ambulance timesteps!")
                    failed += 1
                else:
                    print(f"    [OK]   {os.path.basename(out_fcd)} ({n_ts} ambulance timesteps)")
                    done += 1

    print(
        f"\n{'='*60}\n"
        f"Done: {done} generated, {skipped} skipped, {failed} failed\n"
        f"FCD files in: {VARIANTS_FCD_DIR}\n"
        f"\nNext: run pytest baselines/tests/ -v to verify pool expansion.\n"
        f"The pool auto-discovers these files via default_route_pool(K).\n"
    )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants", type=int, default=30,
        help="Number of random starting-position variants per K (default: 30)",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for edge sampling (default: 0)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing FCD files",
    )
    args = parser.parse_args()
    main(n_variants=args.variants, seed=args.seed, force=args.force)
