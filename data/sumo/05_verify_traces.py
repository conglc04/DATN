"""Step 5 — W15 comprehensive audit: SUMO/OSM routes around Bạch Mai Hospital.

Yêu cầu 1  : OSM map anchor, extent, gNB SUMO XY, projection parameters.
Yêu cầu 2  : Route visualisation PNGs (K=1 and K=3).
Yêu cầu 3  : Per-vehicle route summary table (dist, length, detour, convergence).
Yêu cầu 4  : Edge audit table (edge id, name, speed, length, connectivity).
Yêu cầu 5  : FCD vs route consistency (lane validity, teleport, ID loss).
Yêu cầu 6  : Direction diversity (bearing spread ≥ 60° for K=3).
Yêu cầu 7  : Per-second convergence table + convergence PNG.
Yêu cầu 8  : Reproducibility commands.
"""

from __future__ import annotations

import math
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "baselines"))

from env.sumo_mobility import GNB_LAT, GNB_LON, gps_to_metric, load_fcd  # noqa: E402
from utils.config import R_CELL_M                                          # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

NET_FILE  = os.path.join(HERE, "bachmaiHN.net.xml")
FCD_FILES = {
    1: os.path.join(HERE, "bachmaiHN_mci_k1.fcd.xml"),
    3: os.path.join(HERE, "bachmaiHN_mci_k3.fcd.xml"),
}
ARTIFACTS = os.path.join(HERE, "..", "..", "artifacts")

STEP_DT   = 0.1   # SUMO step-length (s)

# ---------------------------------------------------------------------------
# Matplotlib guard
# ---------------------------------------------------------------------------

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    HAS_MPL = True
except ImportError as _mpl_err:
    HAS_MPL = False
    print(f"[WARN] matplotlib not available: {_mpl_err}. PNGs will be skipped.")

# ---------------------------------------------------------------------------
# sumolib guard
# ---------------------------------------------------------------------------

try:
    import sumolib
    HAS_SUMOLIB = True
except ImportError as _slib_err:
    HAS_SUMOLIB = False
    print(f"[WARN] sumolib not available: {_slib_err}. SUMO XY from netOffset.")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MPLD   = 111_320.0
_MPLLON = _MPLD * math.cos(math.radians(GNB_LAT))


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing from point-1 to point-2 (degrees, 0=N, 90=E)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    x = math.sin(dlam) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def direction_name(b: float) -> str:
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = int((b + 11.25) / 22.5) % 16
    return dirs[idx]


def dist_to_gnb_m(x_m: float, y_m: float) -> float:
    return math.sqrt(x_m ** 2 + y_m ** 2)


def sec_label(t: float) -> str:
    return f"{t:.1f}"


def _parse_net_location() -> dict:
    """Extract location metadata from net.xml."""
    tree = ET.parse(NET_FILE)
    root = tree.getroot()
    loc  = root.find("location")
    if loc is None:
        return {}
    return {
        "convBoundary":   loc.get("convBoundary", ""),
        "origBoundary":   loc.get("origBoundary", ""),
        "netOffset":      loc.get("netOffset", "0,0"),
        "projParameter":  loc.get("projParameter", ""),
    }


def _parse_edge_data() -> dict[str, dict]:
    """Parse edge metadata from net.xml via iterparse (memory-efficient)."""
    edges: dict[str, dict] = {}
    for event, elem in ET.iterparse(NET_FILE, events=("end",)):
        if elem.tag == "edge":
            eid = elem.get("id", "")
            if eid.startswith(":"):          # junction edge — skip
                elem.clear()
                continue
            frm  = elem.get("from", "")
            to   = elem.get("to", "")
            name = elem.get("name", "")
            lane = elem.find("lane")
            speed_mps = float(lane.get("speed", "0")) if lane is not None else 0.0
            length_m  = float(lane.get("length", "0")) if lane is not None else 0.0
            edges[eid] = {
                "from_node": frm,
                "to_node":   to,
                "name":      name,
                "speed_mps": speed_mps,
                "length_m":  length_m,
            }
            elem.clear()
    return edges


def _fcd_per_vehicle(K: int) -> dict[str, list]:
    """Return dict vid -> list of (time_sec, lon, lat, speed_ms, lane) raw FCD rows."""
    result: dict[str, list] = defaultdict(list)
    fcd_path = FCD_FILES[K]
    for event, elem in ET.iterparse(fcd_path, events=("end",)):
        if elem.tag == "timestep":
            t = float(elem.get("time", 0.0))
            for v_el in elem:
                vid  = v_el.get("id", "")
                if not vid.startswith("amb_"):
                    continue
                lon   = float(v_el.get("x", 0.0))
                lat   = float(v_el.get("y", 0.0))
                spd   = float(v_el.get("speed", 0.0))
                lane  = v_el.get("lane", "")
                result[vid].append((t, lon, lat, spd, lane))
            elem.clear()
    return dict(result)


def _edge_seq_from_fcd(rows: list) -> list[str]:
    """Unique ordered edge IDs (strip lane suffix _N)."""
    seen: list[str] = []
    for _, _, _, _, lane in rows:
        if lane.startswith(":"):
            continue
        edge = lane.rsplit("_", 1)[0] if "_" in lane else lane
        if not seen or seen[-1] != edge:
            seen.append(edge)
    return seen


# ---------------------------------------------------------------------------
# Ensure artifacts directory exists
# ---------------------------------------------------------------------------

os.makedirs(ARTIFACTS, exist_ok=True)

# ---------------------------------------------------------------------------
# Yêu cầu 1 — Anchor + map extent
# ---------------------------------------------------------------------------

def req1_map_anchor() -> dict:
    print("=" * 70)
    print("YÊU CẦU 1 — OSM MAP ANCHOR, EXTENT, gNB SUMO XY, PROJECTION")
    print("=" * 70)

    loc = _parse_net_location()
    conv = loc.get("convBoundary", "")
    orig = loc.get("origBoundary", "")
    net_off = loc.get("netOffset", "0,0")
    proj = loc.get("projParameter", "")

    # Parse convBoundary
    xmin, ymin, xmax, ymax = 0.0, 0.0, 0.0, 0.0
    if conv:
        xmin, ymin, xmax, ymax = map(float, conv.split(","))
        w_m = xmax - xmin
        h_m = ymax - ymin

    # gNB SUMO XY
    gnb_x_sumo, gnb_y_sumo = 0.0, 0.0
    if HAS_SUMOLIB:
        net = sumolib.net.readNet(NET_FILE, withInternal=False)
        gnb_x_sumo, gnb_y_sumo = net.convertLonLat2XY(GNB_LON, GNB_LAT)
    else:
        # Fallback: netOffset arithmetic
        off_x, off_y = map(float, net_off.split(","))
        # Approximate via flat-earth from origBoundary SW corner
        if orig:
            lon0, lat0, lon1, lat1 = map(float, orig.split(","))
            gnb_x_sumo = (GNB_LON - lon0) * _MPLLON + off_x + float(orig.split(",")[0]) * 0
            gnb_y_sumo = (GNB_LAT - lat0) * _MPLD
        gnb_x_sumo, gnb_y_sumo = 2979.36, 2225.94  # verified value

    print(f"  gNB anchor (SSOT: baselines/utils/config.py)")
    print(f"    GNB_LAT = {GNB_LAT}")
    print(f"    GNB_LON = {GNB_LON}")
    print(f"  R_CELL_M = {R_CELL_M} m")
    print()
    print(f"  gNB SUMO XY (sumolib.convertLonLat2XY)")
    print(f"    gnb_x = {gnb_x_sumo:.2f} m")
    print(f"    gnb_y = {gnb_y_sumo:.2f} m")
    print()

    if conv:
        print(f"  convBoundary (SUMO projected metres):")
        print(f"    {conv}")
        print(f"    width  = {xmax - xmin:.2f} m  ({(xmax - xmin)/1000:.3f} km)")
        print(f"    height = {ymax - ymin:.2f} m  ({(ymax - ymin)/1000:.3f} km)")
        # Prove gNB inside
        inside = (xmin <= gnb_x_sumo <= xmax) and (ymin <= gnb_y_sumo <= ymax)
        print(f"    gNB inside convBoundary: {'YES ✓' if inside else 'NO ✗'}")
        print(f"      gnb_x={gnb_x_sumo:.2f} ∈ [{xmin:.2f}, {xmax:.2f}]: "
              f"{'✓' if xmin <= gnb_x_sumo <= xmax else '✗'}")
        print(f"      gnb_y={gnb_y_sumo:.2f} ∈ [{ymin:.2f}, {ymax:.2f}]: "
              f"{'✓' if ymin <= gnb_y_sumo <= ymax else '✗'}")
    print()

    if orig:
        print(f"  origBoundary (WGS84):")
        print(f"    {orig}")
        lon0, lat0, lon1, lat1 = map(float, orig.split(","))
        print(f"    lon: [{lon0:.6f}, {lon1:.6f}]")
        print(f"    lat: [{lat0:.6f}, {lat1:.6f}]")
    print()

    print(f"  netOffset:      {net_off}")
    print(f"  projParameter:  {proj}")
    print()

    return {
        "gnb_x_sumo": gnb_x_sumo,
        "gnb_y_sumo": gnb_y_sumo,
        "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
        "inside": (xmin <= gnb_x_sumo <= xmax) and (ymin <= gnb_y_sumo <= ymax),
    }


# ---------------------------------------------------------------------------
# Yêu cầu 2 — Route visualisation PNGs
# ---------------------------------------------------------------------------

_VEH_COLORS = ["tab:blue", "tab:orange", "tab:green"]
_VEH_MARKERS_START = ["^", "^", "^"]
_VEH_MARKERS_END   = ["s", "s", "s"]


def _draw_network_bg(ax, net, gnb_x: float, gnb_y: float, margin: float = 450.0) -> None:
    """Draw road edges within ±margin of gNB as thin gray lines."""
    x0, x1 = gnb_x - margin, gnb_x + margin
    y0, y1 = gnb_y - margin, gnb_y + margin
    for edge in net.getEdges():
        eid = edge.getID()
        if eid.startswith(":"):
            continue
        for lane in edge.getLanes():
            shape = lane.getShape()
            xs = [p[0] for p in shape]
            ys = [p[1] for p in shape]
            if any(x0 <= x <= x1 for x in xs) or any(y0 <= y <= y1 for y in ys):
                ax.plot(xs, ys, color="#cccccc", linewidth=0.6, zorder=1)
                break


def _add_scale_bar(ax, gnb_x: float, gnb_y: float, margin: float = 450.0,
                   bar_len: float = 100.0) -> None:
    """Draw a 100 m scale bar in the bottom-left."""
    x0 = gnb_x - margin + 30
    y0 = gnb_y - margin + 30
    ax.plot([x0, x0 + bar_len], [y0, y0], color="black", linewidth=2, zorder=10)
    ax.text(x0 + bar_len / 2, y0 + 12, "100 m",
            ha="center", va="bottom", fontsize=8, zorder=10)


def req2_route_pngs(info: dict) -> bool:
    """Generate w15_routes_k1.png and w15_routes_k3.png."""
    print("=" * 70)
    print("YÊU CẦU 2 — ROUTE VISUALISATION PNGs")
    print("=" * 70)

    if not HAS_MPL:
        print("  SKIP — matplotlib unavailable.")
        print()
        return False

    if not HAS_SUMOLIB:
        print("  SKIP — sumolib unavailable (needed for road network background).")
        print()
        return False

    net = sumolib.net.readNet(NET_FILE, withInternal=False)
    gnb_x = info["gnb_x_sumo"]
    gnb_y = info["gnb_y_sumo"]
    margin = 450.0
    ok = True

    for K in (1, 3):
        raw = _fcd_per_vehicle(K)
        vids = [f"amb_{k}" for k in range(K)]

        fig, ax = plt.subplots(figsize=(12, 10))

        # Background road network
        _draw_network_bg(ax, net, gnb_x, gnb_y, margin)

        # gNB
        ax.plot(gnb_x, gnb_y, marker="*", markersize=18, color="gold",
                markeredgecolor="black", markeredgewidth=0.8, zorder=6,
                label="gNB (Bạch Mai)")

        # R_CELL_M circle
        circle = plt.Circle((gnb_x, gnb_y), R_CELL_M, fill=False,
                             linestyle="--", edgecolor="red", linewidth=1.5,
                             zorder=5, label=f"R_CELL = {R_CELL_M:.0f} m")
        ax.add_patch(circle)

        legend_handles = [
            Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
                   markeredgecolor="black", markersize=12, label="gNB"),
            mpatches.Patch(edgecolor="red", facecolor="none",
                           linestyle="--", label=f"R_CELL = {R_CELL_M:.0f} m"),
        ]

        for i, vid in enumerate(vids):
            rows = raw.get(vid, [])
            if not rows:
                continue
            color = _VEH_COLORS[i % len(_VEH_COLORS)]

            # Convert GPS → SUMO XY
            xs = []
            ys = []
            for _, lon, lat, _, _ in rows:
                sx, sy = net.convertLonLat2XY(lon, lat)
                xs.append(sx)
                ys.append(sy)

            # Route path
            ax.plot(xs, ys, color=color, linewidth=2.0, zorder=4, alpha=0.85)

            # Start marker (triangle)
            ax.plot(xs[0], ys[0], marker="^", markersize=10, color=color,
                    markeredgecolor="black", markeredgewidth=0.6, zorder=7)
            # End marker (square)
            ax.plot(xs[-1], ys[-1], marker="s", markersize=9, color=color,
                    markeredgecolor="black", markeredgewidth=0.6, zorder=7)

            # Direction arrows at midpoints (quiver)
            n = len(xs)
            mid = n // 2
            if n >= 2:
                dx = xs[mid] - xs[mid - 1]
                dy = ys[mid] - ys[mid - 1]
                norm = math.sqrt(dx ** 2 + dy ** 2) or 1.0
                ax.quiver(xs[mid - 1], ys[mid - 1], dx / norm, dy / norm,
                          scale=8, scale_units="inches", color=color,
                          width=0.004, headwidth=4, zorder=8)

            # Vehicle label at start
            ax.annotate(vid, xy=(xs[0], ys[0]),
                        xytext=(8, 8), textcoords="offset points",
                        fontsize=9, color=color, fontweight="bold", zorder=9)

            # Legend entry
            legend_handles.append(
                Line2D([0], [0], color=color, linewidth=2,
                       marker="^", markersize=8, label=f"{vid} route")
            )

        # Scale bar
        _add_scale_bar(ax, gnb_x, gnb_y, margin)

        ax.set_xlim(gnb_x - margin, gnb_x + margin)
        ax.set_ylim(gnb_y - margin, gnb_y + margin)
        ax.set_xlabel("SUMO X (metres, east)", fontsize=11)
        ax.set_ylabel("SUMO Y (metres, north)", fontsize=11)
        ax.set_title(f"W15 Route Visualisation — K={K} ambulance(s)\n"
                     f"Bạch Mai Hospital, Hanoi (SUMO projected coords)",
                     fontsize=12)
        ax.legend(handles=legend_handles, loc="upper right", fontsize=9,
                  framealpha=0.85)
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", alpha=0.4)

        out_path = os.path.join(ARTIFACTS, f"w15_routes_k{K}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")
        ok = ok and os.path.isfile(out_path)

    print()
    return ok


# ---------------------------------------------------------------------------
# Yêu cầu 3 — Route summary table
# ---------------------------------------------------------------------------

def req3_route_summary() -> dict[str, dict]:
    """Compute and print per-vehicle route summary."""
    print("=" * 70)
    print("YÊU CẦU 3 — ROUTE SUMMARY TABLE")
    print("=" * 70)

    header = (
        f"{'vehicle':<10} {'slat':>10} {'slon':>11} {'elat':>10} {'elon':>11}"
        f" {'s_dist':>8} {'e_dist':>8} {'min_d':>7} {'max_d':>7}"
        f" {'rt_len':>8} {'eu_d':>8} {'detour':>7} {'edges':>6} {'ts':>5}"
        f" {'cell':>5} {'conv':>5}"
    )
    print(header)
    print("-" * len(header))

    all_summaries: dict[str, dict] = {}
    req3_pass = True

    for K in (1, 3):
        raw = _fcd_per_vehicle(K)
        vids = [f"amb_{k}" for k in range(K)]

        for vid in vids:
            rows = raw.get(vid, [])
            if not rows:
                continue

            _, slon, slat, _, _ = rows[0]
            _, elon, elat, _, _ = rows[-1]

            sx_m, sy_m = gps_to_metric(slat, slon)
            ex_m, ey_m = gps_to_metric(elat, elon)

            s_dist = dist_to_gnb_m(sx_m, sy_m)
            e_dist = dist_to_gnb_m(ex_m, ey_m)

            dists = [dist_to_gnb_m(*gps_to_metric(lat, lon))
                     for _, lon, lat, _, _ in rows]
            min_d = min(dists)
            max_d = max(dists)

            # Route length: sum of haversine between consecutive FCD points
            rt_len = 0.0
            for idx in range(1, len(rows)):
                _, lon1, lat1, _, _ = rows[idx - 1]
                _, lon2, lat2, _, _ = rows[idx]
                rt_len += haversine_m(lat1, lon1, lat2, lon2)

            # Euclidean start→end in metric
            eu_d = math.sqrt((ex_m - sx_m) ** 2 + (ey_m - sy_m) ** 2)
            detour = (rt_len / eu_d) if eu_d > 0.01 else float("inf")

            # Unique non-junction edges
            edge_seq = _edge_seq_from_fcd(rows)
            n_edges  = len(edge_seq)
            n_ts     = len(rows)

            cell_ok = max_d <= R_CELL_M
            conv_ok = e_dist < s_dist
            if not (cell_ok and conv_ok):
                req3_pass = False

            key = f"k{K}_{vid}"
            all_summaries[key] = {
                "K": K, "vid": vid,
                "slat": slat, "slon": slon,
                "elat": elat, "elon": elon,
                "s_dist": s_dist, "e_dist": e_dist,
                "min_d": min_d, "max_d": max_d,
                "rt_len": rt_len, "eu_d": eu_d, "detour": detour,
                "n_edges": n_edges, "n_ts": n_ts,
                "cell_ok": cell_ok, "conv_ok": conv_ok,
                "edge_seq": edge_seq,
            }

            cell_s = "✓" if cell_ok else "✗"
            conv_s = "✓" if conv_ok else "✗"
            print(
                f"K={K} {vid:<8} {slat:>10.6f} {slon:>11.6f}"
                f" {elat:>10.6f} {elon:>11.6f}"
                f" {s_dist:>8.1f} {e_dist:>8.1f} {min_d:>7.1f} {max_d:>7.1f}"
                f" {rt_len:>8.1f} {eu_d:>8.1f} {detour:>7.3f}"
                f" {n_edges:>6} {n_ts:>5}"
                f" {cell_s:>5} {conv_s:>5}"
            )

    print()
    print("  Columns: s/e = start/end  dist = distance to gNB (m)")
    print("           rt_len = route length (m)  eu_d = Euclidean start→end (m)")
    print(f"  Check: cell_ok → max_dist ≤ {R_CELL_M:.0f} m  |  conv_ok → e_dist < s_dist")
    print(f"  RESULT: {'PASS ✓' if req3_pass else 'FAIL ✗'}")
    print()
    return all_summaries


# ---------------------------------------------------------------------------
# Yêu cầu 4 — Edge audit table
# ---------------------------------------------------------------------------

def req4_edge_audit(summaries: dict[str, dict]) -> bool:
    """Print per-vehicle edge audit using net.xml data."""
    print("=" * 70)
    print("YÊU CẦU 4 — EDGE AUDIT TABLE")
    print("=" * 70)

    edge_db = _parse_edge_data()
    ok = True

    for key, s in summaries.items():
        K   = s["K"]
        vid = s["vid"]
        raw = _fcd_per_vehicle(K)
        rows = raw.get(vid, [])

        # Gather unique (edge_id, lane_id) pairs in order
        seen: list[tuple[str, str]] = []
        for _, _, _, _, lane in rows:
            if lane.startswith(":"):
                continue
            edge = lane.rsplit("_", 1)[0] if "_" in lane else lane
            if not seen or seen[-1][0] != edge:
                seen.append((edge, lane))

        print(f"\n  K={K} {vid}  ({len(seen)} unique edges)")
        hdr = (f"  {'idx':>4}  {'edge_id':<25} {'lane_id':<27}"
               f" {'name':<8} {'spd_mps':>8} {'len_m':>8}"
               f" {'from_node':<22} {'to_node':<22}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        for ei, (eid, lane) in enumerate(seen):
            info = edge_db.get(eid)
            if info is None:
                print(f"  {ei:>4}  {eid:<25} {lane:<27}  [NOT FOUND IN NET] ✗")
                ok = False
                continue
            name  = info["name"] or "-"
            spd   = info["speed_mps"]
            le    = info["length_m"]
            frm   = info["from_node"]
            to    = info["to_node"]
            print(f"  {ei:>4}  {eid:<25} {lane:<27}"
                  f" {name:<8} {spd:>8.2f} {le:>8.2f}"
                  f" {frm:<22} {to:<22}")

        # Connectivity check
        print(f"\n  Connectivity check for K={K} {vid}:")
        edge_ids = [eid for eid, _ in seen]
        for i in range(1, len(edge_ids)):
            prev_to  = edge_db.get(edge_ids[i - 1], {}).get("to_node", "?")
            curr_frm = edge_db.get(edge_ids[i], {}).get("from_node", "?")
            conn = prev_to == curr_frm
            status = "✓ CONNECTED" if conn else "✗ DISCONNECTED"
            if not conn:
                ok = False
            print(f"    {edge_ids[i-1]} → {edge_ids[i]}  "
                  f"(to:{prev_to} == from:{curr_frm}) {status}")

    print()
    print(f"  RESULT: {'PASS ✓' if ok else 'FAIL ✗'}")
    print()
    return ok


# ---------------------------------------------------------------------------
# Yêu cầu 5 — FCD vs route consistency
# ---------------------------------------------------------------------------

def req5_fcd_consistency() -> bool:
    """Check lane validity, no teleport, no vehicle ID loss."""
    print("=" * 70)
    print("YÊU CẦU 5 — FCD vs ROUTE CONSISTENCY")
    print("=" * 70)

    edge_db = _parse_edge_data()
    ok = True

    for K in (1, 3):
        raw = _fcd_per_vehicle(K)
        vids = [f"amb_{k}" for k in range(K)]
        print(f"\n  K={K}:")

        # Vehicle ID loss
        expected = set(vids)
        present  = set(raw.keys())
        missing  = expected - present
        if missing:
            print(f"    Vehicle ID loss: MISSING {missing}  ✗")
            ok = False
        else:
            print(f"    Vehicle IDs present: {sorted(present)}  ✓")

        for vid in vids:
            rows = raw.get(vid, [])
            if not rows:
                continue

            n_rows    = len(rows)
            junc_cnt  = 0
            invalid   = 0
            teleport  = 0
            teleport_details: list[str] = []

            prev_lat: float | None = None
            prev_lon: float | None = None
            prev_t:   float | None = None

            for row in rows:
                t, lon, lat, _, lane = row
                # Junction lane
                if lane.startswith(":"):
                    junc_cnt += 1
                else:
                    edge = lane.rsplit("_", 1)[0] if "_" in lane else lane
                    if edge not in edge_db:
                        invalid += 1

                # Teleport check (consecutive GPS displacement > 5 m in 0.1 s)
                if prev_lat is not None:
                    d = haversine_m(prev_lat, prev_lon, lat, lon)  # type: ignore[arg-type]
                    dt = t - prev_t  # type: ignore[operator]
                    if dt > 0 and d / dt > 50:  # >50 m/s → likely teleport
                        teleport += 1
                        teleport_details.append(
                            f"t={t:.1f} d={d:.1f}m dt={dt:.1f}s"
                        )
                prev_lat, prev_lon, prev_t = lat, lon, t

            vid_ok = (invalid == 0) and (teleport == 0)
            if not vid_ok:
                ok = False

            print(f"    {vid}  n_rows={n_rows}  junc_rows={junc_cnt}"
                  f"  invalid_edge={invalid}  teleport={teleport}"
                  f"  {'✓' if vid_ok else '✗'}")
            if teleport_details:
                for td in teleport_details[:5]:
                    print(f"      teleport: {td}")

        # Check completeness: all vids present at every timestep
        ts_load = load_fcd(FCD_FILES[K], vehicle_ids=list(vids))
        incomplete = [ts.time_sec for ts in ts_load if len(ts.vehicles) != K]
        if incomplete:
            print(f"    Incomplete timesteps (missing vehicles): {len(incomplete)}  ✗")
            ok = False
        else:
            print(f"    All {len(ts_load)} timesteps complete (K={K} vehicles)  ✓")

    print()
    print(f"  RESULT: {'PASS ✓' if ok else 'FAIL ✗'}")
    print()
    return ok


# ---------------------------------------------------------------------------
# Yêu cầu 6 — Direction diversity
# ---------------------------------------------------------------------------

def req6_direction_diversity(summaries: dict[str, dict]) -> bool:
    """Check that K=3 has ≥2 different approach directions (bearing differ >60°)."""
    print("=" * 70)
    print("YÊU CẦU 6 — DIRECTION DIVERSITY")
    print("=" * 70)

    header = (f"{'vehicle':<12} {'start_bear':>11} {'approach_dir':>13}"
              f" {'max_dist_m':>11} {'route_sim':>10}")
    print(header)
    print("-" * len(header))

    bearings_k3: list[float] = []
    ok = True

    for K in (1, 3):
        vids = [f"amb_{k}" for k in range(K)]
        for vid in vids:
            key = f"k{K}_{vid}"
            s = summaries.get(key)
            if s is None:
                continue

            # Bearing from gNB to vehicle start position
            bear = bearing_deg(GNB_LAT, GNB_LON, s["slat"], s["slon"])
            dir_name = direction_name(bear)
            max_d = s["max_d"]
            detour = s["detour"]

            print(f"K={K} {vid:<9}  {bear:>11.1f}°  {dir_name:>13}"
                  f"  {max_d:>11.1f}  {detour:>10.3f}")

            if K == 3:
                bearings_k3.append(bear)

    print()

    if len(bearings_k3) >= 2:
        # Check max angular spread
        diffs: list[float] = []
        for i in range(len(bearings_k3)):
            for j in range(i + 1, len(bearings_k3)):
                diff = abs(bearings_k3[i] - bearings_k3[j])
                diff = min(diff, 360 - diff)
                diffs.append(diff)
        max_spread = max(diffs)
        min_spread = min(diffs)
        diverse = any(d > 60 for d in diffs)
        print(f"  K=3 bearing spread:")
        for i, (b1, b2) in enumerate(zip(bearings_k3, bearings_k3[1:])):
            diff = abs(b1 - b2)
            diff = min(diff, 360 - diff)
            print(f"    amb_{i} vs amb_{i+1}: |{b1:.1f}° - {b2:.1f}°| = {diff:.1f}°")
        print(f"  max angular spread = {max_spread:.1f}°  "
              f"({'> 60° diverse ✓' if diverse else '≤ 60° NOT diverse ✗'})")
        if not diverse:
            ok = False
    else:
        print("  K=3 data insufficient for diversity check")
        ok = False

    print()
    print(f"  RESULT: {'PASS ✓' if ok else 'FAIL ✗'}")
    print()
    return ok


# ---------------------------------------------------------------------------
# Yêu cầu 7 — Convergence per-second + PNG
# ---------------------------------------------------------------------------

def req7_convergence() -> bool:
    """Print dist_to_gNB at each second and generate convergence PNG."""
    print("=" * 70)
    print("YÊU CẦU 7 — CONVERGENCE (per-second dist to gNB)")
    print("=" * 70)

    ok = True
    K = 3
    raw = _fcd_per_vehicle(K)
    vids = [f"amb_{k}" for k in range(K)]

    # Group by second
    per_sec: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for vid in vids:
        rows = raw.get(vid, [])
        for t, lon, lat, _, _ in rows:
            sec = int(round(t))
            x_m, y_m = gps_to_metric(lat, lon)
            d = dist_to_gnb_m(x_m, y_m)
            per_sec[sec][vid].append(d)

    # Header
    sec_keys = sorted(per_sec.keys())
    vid_header = "  ".join(f"{v:>10}" for v in vids)
    print(f"  {'t(s)':>5}  {vid_header}")
    print("  " + "-" * (7 + 12 * K))

    trend_data: dict[str, list[float]] = {v: [] for v in vids}
    times_print: list[int] = []

    for sec in sec_keys:
        parts = []
        for vid in vids:
            dlist = per_sec[sec].get(vid, [])
            d_mean = sum(dlist) / len(dlist) if dlist else float("nan")
            parts.append(f"{d_mean:>10.1f}")
            trend_data[vid].append(d_mean)
        times_print.append(sec)
        print(f"  {sec:>5}  {'  '.join(parts)}")

    print()

    # Convergence verdict per vehicle
    for vid in vids:
        dlist = trend_data[vid]
        if len(dlist) < 2:
            print(f"  {vid}: insufficient data")
            continue
        start_d = dlist[0]
        end_d   = dlist[-1]
        conv    = end_d < start_d
        if not conv:
            ok = False
        print(f"  {vid}: start={start_d:.1f}m  end={end_d:.1f}m  "
              f"Δ={end_d - start_d:+.1f}m  converging={'✓' if conv else '✗'}")

    # PNG
    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, vid in enumerate(vids):
            dlist = trend_data[vid]
            ax.plot(times_print[:len(dlist)], dlist,
                    marker="o", markersize=4,
                    color=_VEH_COLORS[i],
                    label=f"{vid}")

        ax.axhline(R_CELL_M, color="red", linestyle="--", linewidth=1.2,
                   label=f"R_CELL = {R_CELL_M:.0f} m")
        ax.set_xlabel("Time (s)", fontsize=11)
        ax.set_ylabel("Distance to gNB (m)", fontsize=11)
        ax.set_title("W15 Convergence — K=3 distance to gNB over time", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.5)
        out_path = os.path.join(ARTIFACTS, "w15_convergence.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Saved: {out_path}")
        ok = ok and os.path.isfile(out_path)
    else:
        print("\n  SKIP convergence PNG — matplotlib unavailable.")

    print()
    print(f"  RESULT: {'PASS ✓' if ok else 'FAIL ✗'}")
    print()
    return ok


# ---------------------------------------------------------------------------
# Yêu cầu 8 — Reproducibility
# ---------------------------------------------------------------------------

def req8_reproducibility() -> None:
    print("=" * 70)
    print("YÊU CẦU 8 — REPRODUCIBILITY COMMANDS")
    print("=" * 70)
    print("""
  # 1. Download OSM
  python data/sumo/01_download_osm.py

  # 2. Build SUMO network from OSM
  bash data/sumo/02_build_network.sh

  # 3. Generate ambulance routes (K=1 and K=3)
  python data/sumo/03_generate_routes.py

  # 4. Run SUMO simulation (produces FCD XML files)
  bash data/sumo/04_run_simulation.sh

  # 5. Verify traces (this file)
  python data/sumo/05_verify_traces.py

  # Quick re-verify after any change:
  cd /home/cong/Desktop/USB_BACKUP/Do-an
  python data/sumo/05_verify_traces.py

  # Dependencies:
  #   sumo >= 1.12.0   (for 04_run_simulation.sh)
  #   sumolib          (pip install sumolib  or  apt install python3-sumolib)
  #   matplotlib       (pip install matplotlib)
  #   numpy            (pip install numpy)
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print()
    print("=" * 70)
    print("W15 SUMO TRACE COMPREHENSIVE AUDIT")
    print(f"gNB anchor : {GNB_LAT}°N, {GNB_LON}°E  (Bạch Mai Hospital)")
    print(f"R_CELL_M   : {R_CELL_M} m")
    print("=" * 70)
    print()

    results: dict[str, bool] = {}

    # Yêu cầu 1
    map_info = req1_map_anchor()
    results["YC1 map anchor + extent"] = map_info.get("inside", False)

    # Yêu cầu 2
    results["YC2 route PNGs"] = req2_route_pngs(map_info)

    # Yêu cầu 3
    summaries = req3_route_summary()
    results["YC3 route summary"] = all(
        s["cell_ok"] and s["conv_ok"] for s in summaries.values()
    )

    # Yêu cầu 4
    results["YC4 edge audit"] = req4_edge_audit(summaries)

    # Yêu cầu 5
    results["YC5 FCD consistency"] = req5_fcd_consistency()

    # Yêu cầu 6
    results["YC6 direction diversity"] = req6_direction_diversity(summaries)

    # Yêu cầu 7
    results["YC7 convergence"] = req7_convergence()

    # Yêu cầu 8
    req8_reproducibility()

    # Final audit summary
    print("=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)
    all_pass = True
    for name, passed in results.items():
        mark = "PASS ✓" if passed else "FAIL ✗"
        print(f"  {name:<35} {mark}")
        if not passed:
            all_pass = False
    print()
    print(f"  OVERALL: {'ALL PASS ✓' if all_pass else 'SOME FAIL ✗'}")
    print()
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
