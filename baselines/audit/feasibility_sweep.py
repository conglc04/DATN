"""Feasibility baseline sweep: fixed b_rrm × severity → constraint table.

Answers: "What is the MINIMUM b_rrm that satisfies all CMDP constraints
for each severity level?" — no RL, no Manager, just env physics.

Usage:
    python3 -m audit.feasibility_sweep --K 1 --episodes-per-cell 20
    python3 -m audit.feasibility_sweep --K 3 --episodes-per-cell 10 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from env.oran_env import EnvConfig, ORANEnv, macro_mission_config
from utils.config import (
    B_RRM_FLOOR_BY_SEV,
    B_RRM_MAX,
    B_RRM_MIN,
    CMDP_D_J_SEVERITY,
    P_TOTAL,
    SEVERITY_QOS,
)

B_RRM_GRID = [round(x * 0.05, 2) for x in range(4, 18)]  # 0.20, 0.25, ..., 0.85


def _force_b_rrm(env: ORANEnv, b_rrm: float) -> None:
    """Set b_rrm bypassing severity floor — pure physics sweep."""
    env.r_min_urllc = float(np.clip(b_rrm, 0.01, 0.99))
    env.r_min_urllc_anchor = env.r_min_urllc
    env.r_max_emBB = 1.0 - env.r_min_urllc
    env.r_ded_urllc = 0.0


@dataclass(frozen=True)
class CellResult:
    b_rrm: float
    severity: int
    n_episodes: int
    mean_delay_ms: float
    delay_viol_rate: float
    mean_embb_mbps: float
    c3_shortfall_mbps: float
    mean_aoi_ms: float
    aoi_viol_rate: float
    delivery_rate: float
    mean_prb_urllc: float
    mean_prb_embb: float
    feasible: bool
    binding_constraint: str


def run_cell(
    b_rrm: float,
    severity: int,
    K: int,
    n_episodes: int,
    seed: int,
) -> CellResult:
    """Run n_episodes with fixed b_rrm and fixed severity (no RL)."""
    cfg = macro_mission_config(K_ambulances=K, seed=seed)
    cfg = EnvConfig(
        **{
            **cfg.__dict__,
            "sample_severity": False,
            "initial_severity": severity,
        }
    )

    delays, viols, embbs, aois, aoi_viols = [], [], [], [], []
    deliveries, prb_us, prb_es = [], [], []
    c3_gaps = []

    for ep in range(n_episodes):
        ep_seed = seed + ep * 1000
        env = ORANEnv(config=cfg, seed=ep_seed)
        obs, info = env.reset()
        _force_b_rrm(env, b_rrm)

        action_dim = env.action_space.shape[0]
        action = np.zeros(action_dim, dtype=np.float32)

        ep_delay, ep_viol, ep_embb = 0.0, 0.0, 0.0
        ep_aoi, ep_aoi_viol = 0.0, 0.0
        ep_prb_u, ep_prb_e = 0.0, 0.0
        n_steps = 0
        manager_counter = 0

        done = False
        while not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            manager_counter += 1
            if manager_counter % 10 == 0:
                _force_b_rrm(env, b_rrm)

            c_vec = info.get("c_vec", np.zeros(4 * K + 1))
            d_phi = info.get("d_phi", np.zeros(4 * K + 1))

            if env.active_mask.any():
                active_idx = np.where(env.active_mask)[0]
                d_maxs = np.array([
                    SEVERITY_QOS[int(env.severity_per_amb[k])]["D_max"]
                    for k in active_idx
                ])
                delays_k = c_vec[active_idx] * d_maxs * 1e3
                ep_delay += float(np.mean(delays_k))
                viols_k = c_vec[K + active_idx[0]:K + active_idx[-1] + 1] if len(active_idx) > 0 else np.array([0.0])
                ep_viol += float(np.mean(c_vec[K:2*K][active_idx]))

                aoi_maxs = np.array([
                    SEVERITY_QOS[int(env.severity_per_amb[k])]["AoI_max"]
                    for k in active_idx
                ])
                ep_aoi += float(np.mean(c_vec[2*K:3*K][active_idx] * aoi_maxs * 1e3))
                ep_aoi_viol += float(np.mean(c_vec[3*K:4*K][active_idx]))

            ep_embb += env.last_embb_mbps
            ep_prb_u += env._last_prb_urllc
            ep_prb_e += env._last_prb_embb
            n_steps += 1

        if n_steps > 0:
            delays.append(ep_delay / n_steps)
            viols.append(ep_viol / n_steps)
            embbs.append(ep_embb / n_steps)
            aois.append(ep_aoi / n_steps)
            aoi_viols.append(ep_aoi_viol / n_steps)
            prb_us.append(ep_prb_u / n_steps)
            prb_es.append(ep_prb_e / n_steps)

        pkt_del = int(info.get("aoi_pkt_delivered", np.zeros(K)).sum())
        pkt_fail = (
            int(info.get("aoi_pkt_failed_bler", np.zeros(K)).sum())
            + int(info.get("aoi_pkt_failed_no_prb", np.zeros(K)).sum())
            + int(info.get("aoi_pkt_failed_no_capacity", np.zeros(K)).sum())
        )
        deliveries.append(pkt_del / max(pkt_del + pkt_fail, 1))

        r_min_embb = float(CMDP_D_J_SEVERITY[severity]["d3_embb_mbps"])
        c3_gaps.append(r_min_embb - (ep_embb / max(n_steps, 1)))

    mean_delay = float(np.mean(delays)) if delays else 0.0
    mean_viol = float(np.mean(viols)) if viols else 0.0
    mean_embb = float(np.mean(embbs)) if embbs else 0.0
    mean_aoi = float(np.mean(aois)) if aois else 0.0
    mean_aoi_viol = float(np.mean(aoi_viols)) if aoi_viols else 0.0
    mean_delivery = float(np.mean(deliveries)) if deliveries else 0.0
    mean_c3_gap = float(np.mean(c3_gaps)) if c3_gaps else 0.0
    mean_prb_u = float(np.mean(prb_us)) if prb_us else 0.0
    mean_prb_e = float(np.mean(prb_es)) if prb_es else 0.0

    qos = SEVERITY_QOS[severity]
    d_max_ms = qos["D_max"] * 1e3
    eps_rel = qos["eps"]
    eps_aoi = qos["eps_aoi"]
    aoi_max_ms = qos["AoI_max"] * 1e3

    c1_ok = mean_delay <= d_max_ms
    c2_ok = mean_viol <= eps_rel
    c3_ok = mean_c3_gap <= 0.0
    c4_ok = mean_aoi <= aoi_max_ms
    c5_ok = mean_aoi_viol <= eps_aoi

    feasible = c1_ok and c2_ok and c3_ok and c4_ok and c5_ok

    binding = []
    if not c1_ok:
        binding.append(f"C1(delay>{d_max_ms:.1f}ms)")
    if not c2_ok:
        binding.append(f"C2(viol>{eps_rel:.0e})")
    if not c3_ok:
        binding.append(f"C3(eMBB<{CMDP_D_J_SEVERITY[severity]['d3_embb_mbps']}Mbps)")
    if not c4_ok:
        binding.append(f"C4(AoI>{aoi_max_ms:.0f}ms)")
    if not c5_ok:
        binding.append(f"C5(AoI_viol>{eps_aoi:.0e})")
    if not binding:
        margins = {
            "C1": d_max_ms - mean_delay,
            "C2": eps_rel - mean_viol,
            "C3": -mean_c3_gap,
            "C4": aoi_max_ms - mean_aoi,
        }
        tightest = min(margins, key=lambda k: margins[k] / max(abs(d_max_ms if k == "C1" else (aoi_max_ms if k == "C4" else 1.0)), 1e-9))
        binding.append(f"{tightest}(tightest)")

    return CellResult(
        b_rrm=b_rrm,
        severity=severity,
        n_episodes=n_episodes,
        mean_delay_ms=mean_delay,
        delay_viol_rate=mean_viol,
        mean_embb_mbps=mean_embb,
        c3_shortfall_mbps=max(0.0, mean_c3_gap),
        mean_aoi_ms=mean_aoi,
        aoi_viol_rate=mean_aoi_viol,
        delivery_rate=mean_delivery,
        mean_prb_urllc=mean_prb_u,
        mean_prb_embb=mean_prb_e,
        feasible=feasible,
        binding_constraint="|".join(binding),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Feasibility baseline sweep")
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--episodes-per-cell", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("logs/feasibility_sweep.csv"))
    parser.add_argument("--severities", type=str, default="1,2,3,4,5")
    parser.add_argument("--b-rrm-grid", type=str, default=None,
                        help="Comma-separated b_rrm values (default: 0.20..0.85 step 0.05)")
    args = parser.parse_args()

    sevs = [int(s) for s in args.severities.split(",")]
    grid = [float(x) for x in args.b_rrm_grid.split(",")] if args.b_rrm_grid else B_RRM_GRID

    args.out.parent.mkdir(parents=True, exist_ok=True)
    results: list[CellResult] = []

    total_cells = len(grid) * len(sevs)
    print(f"Feasibility sweep: K={args.K}, {len(grid)} b_rrm × {len(sevs)} sev = {total_cells} cells × {args.episodes_per_cell} ep")
    t0 = time.time()

    for i, b_rrm in enumerate(grid):
        for sev in sevs:
            cell_t = time.time()
            r = run_cell(b_rrm, sev, args.K, args.episodes_per_cell, args.seed)
            results.append(r)
            tag = "PASS" if r.feasible else "FAIL"
            elapsed = time.time() - cell_t
            idx = i * len(sevs) + sevs.index(sev) + 1
            print(f"  [{idx}/{total_cells}] b={b_rrm:.2f} sev={sev} → {tag}  "
                  f"delay={r.mean_delay_ms:.2f}ms eMBB={r.mean_embb_mbps:.1f}Mbps "
                  f"viol={r.delay_viol_rate:.4f} AoI={r.mean_aoi_ms:.1f}ms  "
                  f"({elapsed:.1f}s) [{r.binding_constraint}]")

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "b_rrm", "severity", "n_episodes",
            "mean_delay_ms", "delay_viol_rate",
            "mean_embb_mbps", "c3_shortfall_mbps",
            "mean_aoi_ms", "aoi_viol_rate",
            "delivery_rate", "mean_prb_urllc", "mean_prb_embb",
            "feasible", "binding_constraint",
        ])
        for r in results:
            writer.writerow([
                r.b_rrm, r.severity, r.n_episodes,
                f"{r.mean_delay_ms:.4f}", f"{r.delay_viol_rate:.6f}",
                f"{r.mean_embb_mbps:.2f}", f"{r.c3_shortfall_mbps:.4f}",
                f"{r.mean_aoi_ms:.4f}", f"{r.aoi_viol_rate:.6f}",
                f"{r.delivery_rate:.4f}", f"{r.mean_prb_urllc:.1f}", f"{r.mean_prb_embb:.1f}",
                r.feasible, r.binding_constraint,
            ])

    print(f"\nDone in {time.time()-t0:.0f}s → {args.out}")

    print("\n" + "=" * 80)
    print("SUMMARY: Minimum feasible b_rrm per severity")
    print("=" * 80)
    print(f"{'Sev':>4} {'Min b_rrm':>10} {'URLLC PRB':>10} {'eMBB Mbps':>10} {'Tightest C':>15} {'Floor(cfg)':>10}")
    print("-" * 80)
    for sev in sevs:
        sev_results = [r for r in results if r.severity == sev and r.feasible]
        if sev_results:
            best = min(sev_results, key=lambda r: r.b_rrm)
            floor = B_RRM_FLOOR_BY_SEV.get(sev, B_RRM_MIN)
            print(f"{sev:>4} {best.b_rrm:>10.2f} {best.mean_prb_urllc:>10.0f} "
                  f"{best.mean_embb_mbps:>10.1f} {best.binding_constraint:>15} {floor:>10.2f}")
        else:
            print(f"{sev:>4} {'NONE':>10}  (no feasible b_rrm in grid)")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
