#!/usr/bin/env python3
"""Q1+Q9 full root-cause audit: K=3 feasibility with long episodes + b_rrm sweep.

Findings from first pass:
  - Only amb_0 enters cell in first 2s; need 400s episodes
  - At SINR≈3dB with 9 PRBs/amb, capacity gate blocks AoI delivery
  - C3 fails at B_RRM_MAX because eMBB gets only 41 PRBs
  - Episode metrics (viol, AoI) include INACTIVE ambulances → inflated

This script:
  1. Runs a full macro episode (400s) with oracle Manager at various b_rrm
  2. Tracks per-ambulance metrics ONLY during active periods
  3. Identifies the feasible b_rrm range where ALL C1-C5 pass
  4. Tests the AoI capacity gate specifically
"""
import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from env.oran_env import ORANEnv, macro_mission_config, capacity_per_prb_bps
from utils.config import (
    P_TOTAL, B_RRM_MAX, B_RRM_MIN, SEVERITY_QOS, URLLC_PKT_BITS,
    SHANNON_ETA, B_PRB, TTI_SEC,
)

SEED = 0  # same seed as smoke train

def min_sinr_for_prb(prb: int) -> float:
    if prb <= 0:
        return float('inf')
    needed_cap = URLLC_PKT_BITS / (prb * TTI_SEC)
    ratio = needed_cap / (SHANNON_ETA * B_PRB)
    linear = 2**ratio - 1
    if linear <= 0:
        return -float('inf')
    return 10 * math.log10(linear)


def run_oracle(b_rrm: float, sev_tuple: tuple, max_steps=10000):
    cfg = macro_mission_config(K_ambulances=3, seed=SEED)
    cfg.sample_severity = False
    cfg.initial_severity = max(sev_tuple)
    env = ORANEnv(cfg, seed=SEED)
    obs, info = env.reset(seed=SEED, options={"severity_per_amb": list(sev_tuple)})
    env.set_rrm_budget(b_rrm)

    K = 3
    # Per-ambulance tracking (active periods only)
    active_started = np.full(K, np.nan)
    active_ended = np.full(K, np.nan)
    delay_accum = np.zeros(K)
    delay_viol_accum = np.zeros(K)
    aoi_accum = np.zeros(K)
    aoi_viol_accum = np.zeros(K)
    active_step_count = np.zeros(K)
    no_cap_count = np.zeros(K)
    delivery_count = np.zeros(K)
    bler_fail_count = np.zeros(K)
    no_prb_count = np.zeros(K)
    sinr_accum = np.zeros(K)
    prb_accum = np.zeros(K)
    embb_accum = []
    embb_viol_count = 0

    steps_done = 0
    for step_i in range(max_steps):
        action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # equal allocation
        obs, rew, terminated, truncated, info = env.step(action)
        env.set_rrm_budget(b_rrm)
        steps_done += 1

        for k in range(K):
            if env.active_mask[k]:
                if np.isnan(active_started[k]):
                    active_started[k] = step_i
                active_ended[k] = step_i
                active_step_count[k] += 1

                sinr_k = float(env.last_sinr_db[k])
                prb_k = int(env._last_prb_per_amb[k])
                delay_k = float(env._last_d_e2e_per_amb[k])
                aoi_k = float(env._last_aoi_per_amb[k])
                sev_k = int(env.severity_per_amb[k])
                d_max_k = SEVERITY_QOS[sev_k]["D_max"]
                aoi_max_k = SEVERITY_QOS[sev_k]["AoI_max"]

                sinr_accum[k] += sinr_k
                prb_accum[k] += prb_k
                delay_accum[k] += delay_k
                aoi_accum[k] += aoi_k
                if delay_k > d_max_k:
                    delay_viol_accum[k] += 1
                if aoi_k > aoi_max_k:
                    aoi_viol_accum[k] += 1

        embb_mbps = env.last_embb_mbps
        embb_accum.append(embb_mbps)
        if embb_mbps < 10.0:
            embb_viol_count += 1

        if terminated or truncated:
            break

    # Final packet counters
    for k in range(K):
        delivery_count[k] = env._aoi_pkt_delivered[k]
        bler_fail_count[k] = env._aoi_pkt_failed_bler[k]
        no_prb_count[k] = env._aoi_pkt_failed_no_prb[k]
        no_cap_count[k] = env._aoi_pkt_failed_no_capacity[k]

    prb_urllc = int(b_rrm * P_TOTAL)
    prb_embb = P_TOTAL - prb_urllc
    embb_mean = np.mean(embb_accum) if embb_accum else 0
    c3_rate = embb_viol_count / len(embb_accum) if embb_accum else 0

    print(f"\n--- b_rrm={b_rrm:.2f} PRB_U={prb_urllc} PRB_E={prb_embb} "
          f"sev={sev_tuple} steps={steps_done} ---")

    all_pass = True
    for k in range(K):
        n = int(active_step_count[k])
        sev_k = int(sev_tuple[k])
        if n == 0:
            d_s = float(active_started[k]) if not np.isnan(active_started[k]) else -1
            print(f"  amb_{k} (sev={sev_k}): NEVER ACTIVE")
            continue
        sinr_mean = sinr_accum[k] / n
        prb_mean = prb_accum[k] / n
        delay_mean = delay_accum[k] / n
        delay_viol_rate = delay_viol_accum[k] / n
        aoi_mean = aoi_accum[k] / n
        aoi_viol_rate = aoi_viol_accum[k] / n
        d_max = SEVERITY_QOS[sev_k]["D_max"]
        eps = SEVERITY_QOS[sev_k]["eps"]
        aoi_max = SEVERITY_QOS[sev_k]["AoI_max"]
        eps_aoi = SEVERITY_QOS[sev_k]["eps_aoi"]

        c1_pass = delay_mean <= d_max
        c2_pass = delay_viol_rate <= eps
        c5_pass = aoi_viol_rate <= eps_aoi
        min_sinr = min_sinr_for_prb(max(1, int(prb_mean)))
        cap_ok = sinr_mean >= min_sinr

        dur_s = (active_ended[k] - active_started[k] + 1) * 0.01  # steps × 10ms
        total_pkt = int(delivery_count[k] + bler_fail_count[k] + no_prb_count[k] + no_cap_count[k])
        del_rate = delivery_count[k] / total_pkt if total_pkt > 0 else 0

        print(f"  amb_{k} sev={sev_k} active={n}steps ({dur_s:.1f}s) "
              f"SINR={sinr_mean:.1f}dB PRB={prb_mean:.0f} cap_ok={cap_ok}")
        print(f"    delay={delay_mean*1e3:.3f}ms (D_max={d_max*1e3:.0f}ms) C1={'P' if c1_pass else 'F'} "
              f"C2_viol={delay_viol_rate:.6f}(eps={eps}) C2={'P' if c2_pass else 'F'}")
        print(f"    AoI={aoi_mean:.4f}s (max={aoi_max}s) C5_viol={aoi_viol_rate:.6f}(eps={eps_aoi}) "
              f"C5={'P' if c5_pass else 'F'}")
        print(f"    pkts: del={int(delivery_count[k])} bler={int(bler_fail_count[k])} "
              f"no_prb={int(no_prb_count[k])} no_cap={int(no_cap_count[k])} rate={del_rate:.3f}")
        if not (c1_pass and c2_pass and c5_pass):
            all_pass = False

    c3_pass = c3_rate <= 0.01
    print(f"  eMBB={embb_mean:.1f}Mbps C3_viol={c3_rate:.4f} C3={'P' if c3_pass else 'F'}")
    if not c3_pass:
        all_pass = False
    print(f"  ALL={'PASS' if all_pass else 'FAIL'}")
    return all_pass, embb_mean


def main():
    print("="*80)
    print("Q1+Q9 FULL AUDIT — K=3 macro 400s episodes, b_rrm sweep")
    print("="*80)

    # Q9: capacity gate table
    print("\nQ9: AoI capacity gate — min SINR for delivery")
    print(f"{'PRB':>6} {'min_SINR':>10} {'ok@3dB':>8} {'ok@-3dB':>8}")
    for prb in [1, 5, 9, 10, 15, 16, 20, 27, 50, 77, 100, 232, 273]:
        ms = min_sinr_for_prb(prb)
        print(f"  {prb:>4} {ms:>10.1f} {'Y' if ms<=3 else 'N':>8} {'Y' if ms<=-3 else 'N':>8}")

    # Q1: b_rrm sweep with sev=[5,5,5] (hardest) and sev=[2,1,1] (easiest)
    sev_tuples = [(1, 1, 1), (3, 3, 3), (5, 5, 5)]
    b_rrm_vals = [0.10, 0.15, 0.20, 0.30, 0.50, 0.85]

    print("\n" + "="*80)
    print("Q1: b_rrm SWEEP — oracle Manager, equal Worker")
    print("="*80)

    summary = []
    for sev in sev_tuples:
        for b in b_rrm_vals:
            passed, embb = run_oracle(b, sev, max_steps=15000)
            summary.append({"sev": sev, "b_rrm": b, "pass": passed, "embb": embb})

    print("\n" + "="*80)
    print("FEASIBILITY MATRIX")
    print("="*80)
    print(f"{'sev':>14} {'b_rrm':>8} {'PASS?':>8} {'eMBB':>8}")
    for r in summary:
        print(f"  {str(r['sev']):>12} {r['b_rrm']:>8.2f} "
              f"{'PASS' if r['pass'] else 'FAIL':>8} {r['embb']:>8.1f}")

    any_pass = any(r["pass"] for r in summary)
    print(f"\nVERDICT: {'FEASIBLE' if any_pass else 'INFEASIBLE'}")
    if any_pass:
        for r in summary:
            if r["pass"]:
                print(f"  PASS: sev={r['sev']} b_rrm={r['b_rrm']:.2f} eMBB={r['embb']:.1f}Mbps")


if __name__ == "__main__":
    main()
