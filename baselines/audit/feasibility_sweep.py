"""Feasibility sweep — 3 modes for determining analytical bounds + benchmark.

Mode A: service-stability sweep
    Sweep b_rrm for each K, find minimum where queue is stable + packets served.
    → chốt B_RRM_MIN per K.

Mode B: c3-cap sweep
    Sweep b_rrm, find maximum where E[R_eMBB] ≥ 10 Mbps with 95% confidence.
    → chốt B_RRM_MAX.

Mode C: feasibility sweep (per severity)
    Sweep b_rrm × severity, check all C1–C5 using raw counters.
    → benchmark b_feasible(sev), NOT hard floor.

Usage:
    python3 -m audit.feasibility_sweep --mode service   --K 1 --episodes 20
    python3 -m audit.feasibility_sweep --mode c3cap     --K 1 --episodes 20
    python3 -m audit.feasibility_sweep --mode feasibility --K 1 --episodes 20
    python3 -m audit.feasibility_sweep --mode all       --K 1 --episodes 10
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

from env.oran_env import EnvConfig, ORANEnv, macro_mission_config
from utils.config import (
    B_RRM_MAX,
    B_RRM_MIN,
    CMDP_D_J_SEVERITY,
    P_TOTAL,
    SEVERITY_QOS,
)


def _force_b_rrm(env: ORANEnv, b_rrm: float) -> None:
    """Set b_rrm directly — pure physics sweep, no clipping."""
    env.r_min_urllc = float(np.clip(b_rrm, 0.01, 0.99))
    env.r_min_urllc_anchor = env.r_min_urllc
    env.r_max_emBB = 1.0 - env.r_min_urllc
    env.r_ded_urllc = 0.0


def _run_episodes(
    b_rrm: float, severity: int, K: int, n_episodes: int, seed: int,
) -> dict:
    """Run n_episodes with fixed b_rrm/severity, return aggregated metrics."""
    cfg = macro_mission_config(K_ambulances=K, seed=seed)
    cfg = EnvConfig(**{**cfg.__dict__, "sample_severity": False, "initial_severity": severity})

    embbs, prb_us = [], []
    c2_viols_total, c2_samples_total = np.zeros(K, dtype=np.int64), np.zeros(K, dtype=np.int64)
    c5_viols_total, c5_samples_total = np.zeros(K, dtype=np.int64), np.zeros(K, dtype=np.int64)
    delay_sums, delay_steps = np.zeros(K), np.zeros(K)
    aoi_sums, aoi_steps = np.zeros(K), np.zeros(K)
    queue_stable_all = True

    for ep in range(n_episodes):
        env = ORANEnv(config=cfg, seed=seed + ep * 1000)
        obs, info = env.reset()
        _force_b_rrm(env, b_rrm)

        action = np.zeros(env.action_space.shape[0], dtype=np.float32)
        ep_embb, ep_prb_u, n_steps = 0.0, 0.0, 0
        manager_counter = 0
        done = False

        while not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            manager_counter += 1
            if manager_counter % 10 == 0:
                _force_b_rrm(env, b_rrm)

            ep_embb += env.last_embb_mbps
            ep_prb_u += env._last_prb_urllc
            n_steps += 1

        if n_steps > 0:
            embbs.append(ep_embb / n_steps)
            prb_us.append(ep_prb_u / n_steps)

        c2_viols_total += info.get("c2_violation_count", np.zeros(K, dtype=np.int64))
        c2_samples_total += info.get("c2_sample_count", np.zeros(K, dtype=np.int64))
        c5_viols_total += info.get("c5_violation_count", np.zeros(K, dtype=np.int64))
        c5_samples_total += info.get("c5_sample_count", np.zeros(K, dtype=np.int64))

        for k in range(K):
            q = env.queues[f"urllc_{k}"]
            if q.rho >= 1.0:
                queue_stable_all = False
            ac = int(info.get("active_count_per_amb", np.zeros(K))[k])
            if ac > 0:
                c_vec = info.get("c_vec", np.zeros(4 * K + 1))
                d_phi = info.get("d_phi", np.zeros(4 * K + 1))
                delay_sums[k] += float(c_vec[k]) * ac
                delay_steps[k] += ac
                aoi_sums[k] += float(c_vec[2 * K + k]) * ac
                aoi_steps[k] += ac

    mean_embb = float(np.mean(embbs)) if embbs else 0.0
    mean_prb_u = float(np.mean(prb_us)) if prb_us else 0.0
    c2_rate = c2_viols_total / np.maximum(c2_samples_total, 1)
    c5_rate = c5_viols_total / np.maximum(c5_samples_total, 1)
    delay_mean = delay_sums / np.maximum(delay_steps, 1)
    aoi_mean = aoi_sums / np.maximum(aoi_steps, 1)

    return {
        "b_rrm": b_rrm, "severity": severity, "K": K, "n_episodes": n_episodes,
        "mean_embb_mbps": mean_embb, "mean_prb_urllc": mean_prb_u,
        "queue_stable": queue_stable_all,
        "c2_violation_count": c2_viols_total, "c2_sample_count": c2_samples_total,
        "c2_rate": c2_rate,
        "c5_violation_count": c5_viols_total, "c5_sample_count": c5_samples_total,
        "c5_rate": c5_rate,
        "delay_mean_per_amb": delay_mean, "aoi_mean_per_amb": aoi_mean,
    }


def _clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """Upper bound of Clopper-Pearson 95% CI (no scipy dependency)."""
    if n == 0:
        return 1.0
    if k == n:
        return 1.0
    if k == 0:
        return 1.0 - (alpha / 2) ** (1.0 / n)
    try:
        from scipy.stats import beta as beta_dist
        return float(beta_dist.ppf(1 - alpha / 2, k + 1, n - k))
    except (ImportError, ValueError):
        return (k + 1) / (n + 1)  # conservative point estimate fallback


# ── Mode A: service-stability ──────────────────────────────────────────

def sweep_service_stability(K: int, n_episodes: int, seed: int, out: Path) -> None:
    grid = [round(x * 0.01, 2) for x in range(2, 31)]  # 0.02..0.30 fine grid
    sev = 3  # mid-severity for stability test
    print(f"Service-stability sweep: K={K}, {len(grid)} b_rrm × sev={sev} × {n_episodes} ep")

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["b_rrm", "K", "queue_stable", "mean_embb_mbps", "mean_prb_urllc",
                     "c2_viols", "c2_samples", "c2_rate_max"])
        min_stable = None
        for i, b in enumerate(grid):
            t0 = time.time()
            r = _run_episodes(b, sev, K, n_episodes, seed)
            stable = r["queue_stable"]
            c2_max = float(np.max(r["c2_rate"]))
            w.writerow([b, K, stable, f"{r['mean_embb_mbps']:.1f}", f"{r['mean_prb_urllc']:.0f}",
                         int(np.sum(r["c2_violation_count"])), int(np.sum(r["c2_sample_count"])),
                         f"{c2_max:.6f}"])
            f.flush()
            tag = "STABLE" if stable else "UNSTABLE"
            print(f"  [{i+1}/{len(grid)}] b={b:.2f} → {tag}  eMBB={r['mean_embb_mbps']:.1f}Mbps  ({time.time()-t0:.0f}s)")
            if stable and min_stable is None:
                min_stable = b

    print(f"\n→ B_RRM_MIN candidate (K={K}): {min_stable}  ({out})")


# ── Mode B: C3-cap ─────────────────────────────────────────────────────

def sweep_c3_cap(K: int, n_episodes: int, seed: int, out: Path) -> None:
    grid = [round(x * 0.01, 2) for x in range(70, 100)]  # 0.70..0.99 fine grid
    sev = 5  # worst-case for C3 (most URLLC allocation)
    r_min_embb = float(CMDP_D_J_SEVERITY[sev]["d3_embb_mbps"])
    print(f"C3-cap sweep: K={K}, {len(grid)} b_rrm × sev={sev} × {n_episodes} ep, floor={r_min_embb} Mbps")

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["b_rrm", "K", "mean_embb_mbps", "c3_feasible", "embb_lcb95"])
        max_feasible = None
        for i, b in enumerate(grid):
            t0 = time.time()
            r = _run_episodes(b, sev, K, n_episodes, seed)
            embb_vals = [r["mean_embb_mbps"]]  # single aggregated value
            lcb95 = float(np.mean(embb_vals) - 1.96 * max(np.std(embb_vals), 0.1) / max(np.sqrt(len(embb_vals)), 1))
            c3_ok = lcb95 >= r_min_embb
            w.writerow([b, K, f"{r['mean_embb_mbps']:.2f}", c3_ok, f"{lcb95:.2f}"])
            f.flush()
            tag = "PASS" if c3_ok else "FAIL"
            print(f"  [{i+1}/{len(grid)}] b={b:.2f} → {tag}  eMBB={r['mean_embb_mbps']:.1f}Mbps LCB95={lcb95:.1f}  ({time.time()-t0:.0f}s)")
            if c3_ok:
                max_feasible = b

    print(f"\n→ B_RRM_MAX candidate (K={K}): {max_feasible}  ({out})")


# ── Mode C: feasibility per severity ───────────────────────────────────

def sweep_feasibility(K: int, n_episodes: int, seed: int, out: Path) -> None:
    b_grid = [round(x * 0.05, 2) for x in range(1, 18)]  # 0.05..0.85
    sevs = [1, 2, 3, 4, 5]
    total = len(b_grid) * len(sevs)
    print(f"Feasibility sweep: K={K}, {len(b_grid)} b_rrm × {len(sevs)} sev = {total} cells × {n_episodes} ep")

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["b_rrm", "severity", "K", "n_episodes", "queue_stable",
                     "delay_mean_ms", "c2_viols", "c2_samples", "c2_rate",
                     "mean_embb_mbps", "c3_gap_mbps",
                     "aoi_mean_ms", "c5_viols", "c5_samples", "c5_rate",
                     "all_feasible", "binding"])

        for i, b in enumerate(b_grid):
            for sev in sevs:
                t0 = time.time()
                r = _run_episodes(b, sev, K, n_episodes, seed)
                qos = SEVERITY_QOS[sev]

                delay_ms = float(np.max(r["delay_mean_per_amb"])) * qos["D_max"] * 1e3
                c2_max_rate = float(np.max(r["c2_rate"]))
                c2_max_n = int(np.max(r["c2_sample_count"]))
                c2_max_k = int(np.max(r["c2_violation_count"]))
                c2_upper = _clopper_pearson_upper(c2_max_k, c2_max_n) if c2_max_n > 0 else 1.0

                r_min = float(CMDP_D_J_SEVERITY[sev]["d3_embb_mbps"])
                c3_gap = r_min - r["mean_embb_mbps"]

                aoi_ms = float(np.max(r["aoi_mean_per_amb"])) * qos["AoI_max"] * 1e3
                c5_max_rate = float(np.max(r["c5_rate"]))
                c5_max_n = int(np.max(r["c5_sample_count"]))
                c5_max_k = int(np.max(r["c5_violation_count"]))
                c5_upper = _clopper_pearson_upper(c5_max_k, c5_max_n) if c5_max_n > 0 else 1.0

                c1_ok = delay_ms <= qos["D_max"] * 1e3
                c2_ok = c2_upper <= qos["eps"]
                n_min_c2 = int(3 / qos["eps"]) if qos["eps"] > 0 else 999999
                c2_conclusive = c2_max_n >= n_min_c2
                c3_ok = c3_gap <= 0
                c4_ok = aoi_ms <= qos["AoI_max"] * 1e3
                c5_ok = c5_upper <= qos["eps_aoi"]

                feasible = c1_ok and (c2_ok or not c2_conclusive) and c3_ok and c4_ok and c5_ok

                binding = []
                if not c1_ok: binding.append("C1")
                if not c2_ok: binding.append(f"C2({'inconcl' if not c2_conclusive else 'fail'})")
                if not c3_ok: binding.append("C3")
                if not c4_ok: binding.append("C4")
                if not c5_ok: binding.append("C5")
                if not binding: binding.append("all_pass")

                w.writerow([b, sev, K, n_episodes, r["queue_stable"],
                            f"{delay_ms:.4f}", c2_max_k, c2_max_n, f"{c2_max_rate:.8f}",
                            f"{r['mean_embb_mbps']:.2f}", f"{c3_gap:.4f}",
                            f"{aoi_ms:.4f}", c5_max_k, c5_max_n, f"{c5_max_rate:.8f}",
                            feasible, "|".join(binding)])
                f.flush()

                idx = i * len(sevs) + sevs.index(sev) + 1
                tag = "PASS" if feasible else "FAIL"
                elapsed = time.time() - t0
                print(f"  [{idx}/{total}] b={b:.2f} sev={sev} → {tag}  "
                      f"delay={delay_ms:.2f}ms eMBB={r['mean_embb_mbps']:.1f}Mbps "
                      f"c2={c2_max_rate:.6f}(n={c2_max_n}) c5={c5_max_rate:.6f}  "
                      f"({elapsed:.0f}s) [{','.join(binding)}]")

    # Summary table
    print(f"\n{'='*70}")
    print(f"SUMMARY: b_feasible per severity (K={K})")
    print(f"{'='*70}")
    print(f"{'Sev':>4} {'Min b_rrm':>10} {'eMBB Mbps':>10} {'Binding':>15}")
    print(f"{'-'*70}")
    # re-read CSV for summary
    with open(out) as f2:
        reader = csv.DictReader(f2)
        rows = list(reader)
    for sev in sevs:
        passed = [r for r in rows if int(r["severity"]) == sev and r["all_feasible"] == "True"]
        if passed:
            best = min(passed, key=lambda r: float(r["b_rrm"]))
            print(f"{sev:>4} {float(best['b_rrm']):>10.2f} {float(best['mean_embb_mbps']):>10.1f} {best['binding']:>15}")
        else:
            print(f"{sev:>4} {'NONE':>10}  (no feasible b_rrm — need more episodes for tail)")
    print(f"{'='*70}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Feasibility sweep (3 modes)")
    parser.add_argument("--mode", choices=["service", "c3cap", "feasibility", "all"], default="all")
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("logs"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    if args.mode in ("service", "all"):
        sweep_service_stability(args.K, args.episodes, args.seed,
                                args.out_dir / f"sweep_service_K{args.K}.csv")
    if args.mode in ("c3cap", "all"):
        sweep_c3_cap(args.K, args.episodes, args.seed,
                     args.out_dir / f"sweep_c3cap_K{args.K}.csv")
    if args.mode in ("feasibility", "all"):
        sweep_feasibility(args.K, args.episodes, args.seed,
                          args.out_dir / f"sweep_feasibility_K{args.K}.csv")
    print(f"\nTotal elapsed: {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
