"""Gate 3x audit — post-training convergence + constraint-satisfaction report.

Reads the per-episode metrics CSV (logs/<algo>_seed<S>/metrics.csv) and the
summary JSON, then prints a PASS/WARN/FAIL scorecard for one training week:

  * Convergence   — reward trend (first-window vs last-window), stability
  * C1 latency    — final mean_e2e_ms vs D_max^sev (tightest severity)
  * C2 delay-tail — viol_rate trend (rule-of-three caveat vs eps^sev)
  * C3 eMBB floor — mean_embb_mbps vs FIXED 10 Mbps; c3_viol_rate
  * C4/C5 AoI     — via lambda_C4/C5 duals (raw AoI not in CSV — see note)
  * lambda-sat    — every lambda_global < LAMBDA_MAX (no dual blow-up)
  * health        — NaN/Inf guard

Usage (from baselines/):
  python -m scripts.audit_gate3 --algo ppo --seeds 0
  python -m scripts.audit_gate3 --algo td3 --seeds 0 1 2 3 4
  python -m scripts.audit_gate3 --algo ppo            # auto-discover seeds

NOTE: AoI (C4/C5) raw satisfaction is not currently in metrics.csv (only the
duals). To audit C4/C5 directly, add mean_aoi_ms / aoi_viol_rate columns to the
logger. This script flags that gap rather than silently passing C4/C5.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re

from utils.config import SEVERITY_QOS, CMDP_D_J_SEVERITY, LAMBDA_MAX

OK, WARN, FAIL = "PASS", "WARN", "FAIL"
MARK = {OK: "✅", WARN: "⚠️ ", FAIL: "❌"}


def _read_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def _f(rows, col):
    out = []
    for r in rows:
        v = r.get(col, "")
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(math.nan)
    return out


def _parse_sev(rows):
    raw = rows[-1].get("severity_per_amb", rows[-1].get("lambda_severity_per_amb", "[5]"))
    return [int(x) for x in re.findall(r"\d+", raw)] or [5]


def _window_mean(xs, frac=0.1):
    xs = [x for x in xs if not math.isnan(x)]
    if not xs:
        return math.nan
    n = max(1, int(len(xs) * frac))
    return sum(xs[-n:]) / n


def audit_seed(algo, seed):
    csv_path = f"logs/{algo}_seed{seed}/metrics.csv"
    if not os.path.exists(csv_path):
        return None, [f"{FAIL}: no metrics at {csv_path}"]
    rows = _read_csv(csv_path)
    if not rows:
        return None, [f"{FAIL}: empty {csv_path}"]

    sev = _parse_sev(rows)
    sev_tight = max(sev)                       # tightest QoS = highest severity
    d_max_ms = SEVERITY_QOS[sev_tight]["D_max"] * 1e3
    eps = SEVERITY_QOS[sev_tight]["eps"]
    c3_floor = CMDP_D_J_SEVERITY[sev_tight]["d3_embb_mbps"]   # fixed 10 Mbps

    reward = _f(rows, "ep_reward")
    e2e = _f(rows, "mean_e2e_ms")
    viol = _f(rows, "viol_rate")
    embb = _f(rows, "mean_embb_mbps")
    c3v = _f(rows, "c3_viol_rate")
    aoi_ms = _f(rows, "mean_aoi_ms")
    aoi_v = _f(rows, "aoi_viol_rate")
    n_ep = len(rows)
    aoi_max_ms = SEVERITY_QOS[sev_tight]["AoI_max"] * 1e3
    eps_aoi = SEVERITY_QOS[sev_tight]["eps_aoi"]

    # lambda columns (generic over K)
    lam_cols = [c for c in rows[0].keys() if c.startswith("lambda_global_")]
    lam_final = {c: float(rows[-1].get(c, "nan") or "nan") for c in lam_cols}

    checks = []

    # --- Convergence ---
    r0, r1 = _window_mean(reward[:max(1, n_ep // 10) or 1] or reward, 1.0), _window_mean(reward)
    first = _window_mean(reward[:max(1, len(reward) // 10)], 1.0) if len(reward) >= 10 else reward[0]
    last = _window_mean(reward)
    stab = (max(reward[-max(1, n_ep // 10):]) - min(reward[-max(1, n_ep // 10):])) if n_ep >= 10 else float("nan")
    if n_ep < 50:
        conv = WARN; conv_msg = f"only {n_ep} episodes (need >=~500 to judge convergence)"
    elif last >= first:
        conv = OK; conv_msg = f"reward {first:.1f} -> {last:.1f} (improving/stable)"
    else:
        conv = WARN; conv_msg = f"reward {first:.1f} -> {last:.1f} (degraded — inspect)"
    checks.append(("Convergence", conv, conv_msg))

    # --- C1 latency mean ---
    e2e_f = _window_mean(e2e)
    c1 = OK if e2e_f <= d_max_ms else FAIL
    checks.append(("C1 latency", c1, f"mean_e2e={e2e_f:.3f}ms vs D_max^sev{sev_tight}={d_max_ms:.2f}ms"))

    # --- C2 delay tail ---
    v_f = _window_mean(viol)
    # eps is tiny (1e-5); finite-sample viol_rate can't certify it -> rule-of-three caveat
    c2 = OK if v_f <= 0.05 else (WARN if v_f <= 0.2 else FAIL)
    checks.append(("C2 delay-tail", c2, f"viol_rate={v_f:.4f} (eps^sev={eps:.0e}; rule-of-three caveat)"))

    # --- C3 eMBB floor ---
    e_f = _window_mean(embb)
    c3vf = _window_mean(c3v)
    c3 = OK if e_f >= c3_floor else FAIL
    checks.append(("C3 eMBB floor", c3, f"mean_embb={e_f:.1f}Mbps vs floor={c3_floor:.0f}Mbps; c3_viol={c3vf:.3f}"))

    # --- C4 AoI mean ---
    aoi_f = _window_mean(aoi_ms)
    if math.isnan(aoi_f):
        c4 = WARN; c4_msg = "mean_aoi_ms not in CSV (older run — re-train to log AoI)"
    else:
        c4 = OK if aoi_f <= aoi_max_ms else FAIL
        c4_msg = f"mean_aoi={aoi_f:.1f}ms vs AoI_max^sev{sev_tight}={aoi_max_ms:.0f}ms"
    checks.append(("C4 AoI mean", c4, c4_msg))

    # --- C5 AoI tail ---
    aoiv_f = _window_mean(aoi_v)
    if math.isnan(aoiv_f):
        c5 = WARN; c5_msg = "aoi_viol_rate not in CSV (older run — re-train to log AoI)"
    else:
        c5 = OK if aoiv_f <= 0.05 else (WARN if aoiv_f <= 0.2 else FAIL)
        c5_msg = f"aoi_viol_rate={aoiv_f:.4f} (eps_aoi^sev={eps_aoi:.0e}; rule-of-three caveat)"
    checks.append(("C5 AoI tail", c5, c5_msg))

    # --- lambda saturation ---
    sat = [c for c, v in lam_final.items() if not math.isnan(v) and v >= 0.95 * LAMBDA_MAX]
    lam = OK if not sat else FAIL
    lam_msg = "no dual near LAMBDA_MAX" if not sat else f"SATURATED: {sat}"
    checks.append(("lambda-sat", lam, f"{lam_msg} (max={max((v for v in lam_final.values() if not math.isnan(v)), default=0):.3f}/{LAMBDA_MAX})"))

    # --- health (NaN/Inf) ---
    bad = any(math.isnan(x) or math.isinf(x) for x in (e2e_f, e_f, last))
    h = FAIL if bad else OK
    checks.append(("health", h, "NaN/Inf in key metrics" if bad else "finite"))

    return {"seed": seed, "sev": sev, "n_ep": n_ep, "lam_final": lam_final}, checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    args = ap.parse_args()

    seeds = args.seeds
    if not seeds:
        seeds = sorted(int(m.group(1)) for d in glob.glob(f"logs/{args.algo}_seed*")
                       for m in [re.search(r"_seed(\d+)$", d)] if m)
    if not seeds:
        print(f"No logs found for algo={args.algo}. Run training first.")
        return 1

    print("=" * 84)
    print(f"GATE 3x AUDIT — algo={args.algo}  seeds={seeds}")
    print("=" * 84)
    worst = OK
    for s in seeds:
        meta, checks = audit_seed(args.algo, s)
        hdr = f"seed {s}" + (f"  sev={meta['sev']}  episodes={meta['n_ep']}" if meta else "")
        print(f"\n--- {hdr} ---")
        for name, status, msg in checks:
            print(f"  {MARK[status]} {name:14s} {msg}")
            if status == FAIL:
                worst = FAIL
            elif status == WARN and worst != FAIL:
                worst = WARN
    print("\n" + "=" * 84)
    verdict = {OK: "GATE 3x PASS — ready for next week (await user approval)",
               WARN: "GATE 3x WARN — inspect flagged items before proceeding",
               FAIL: "GATE 3x FAIL — do NOT proceed; fix and re-train"}[worst]
    print(verdict)
    print("=" * 84)
    return 0 if worst != FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
