#!/usr/bin/env python3
"""Q1+Q9 Root-cause audit: K=3 physical feasibility under oracle allocations.

Runs a controlled K=3 scenario with fixed seed/severity/route, then tests:
  A) Oracle Manager (B_RRM_MAX) + various Worker allocations (equal, favor, grid)
  B) AoI service-model capacity gate analysis (Q9)
  C) Per-ambulance SINR/PRB/service_bits/delivery tracking

Outputs PASS/FAIL with exact evidence.
"""
import math
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from env.oran_env import ORANEnv, EnvConfig, macro_mission_config, capacity_per_prb_bps
from utils.config import (
    P_TOTAL, B_RRM_MAX, B_RRM_MIN, SEVERITY_QOS, URLLC_PKT_BITS,
    URLLC_OFFERED_LOAD_BPS, SHANNON_ETA, B_PRB, TTI_SEC, MAC_TICKS_PER_WORKER,
)

SEED = 42
K = 3
N_STEPS = 200  # 200 Worker steps = 2s of sim time

def min_sinr_for_prb(prb: int) -> float:
    """Minimum SINR (dB) needed for service_bits >= URLLC_PKT_BITS with given PRBs."""
    if prb <= 0:
        return float('inf')
    needed_cap = URLLC_PKT_BITS / (prb * TTI_SEC)
    ratio = needed_cap / (SHANNON_ETA * B_PRB)
    linear = 2**ratio - 1
    if linear <= 0:
        return -float('inf')
    return 10 * math.log10(linear)

def run_oracle_scenario(alloc_name: str, alloc_fn, severity_tuple, verbose=False):
    """Run N_STEPS with oracle Manager (B_RRM_MAX) and a fixed Worker allocation."""
    cfg = macro_mission_config(K_ambulances=K, seed=SEED)
    cfg.episode_duration_sec = 10.0  # short controlled run
    cfg.sample_severity = False
    cfg.initial_severity = max(severity_tuple)
    env = ORANEnv(cfg, seed=SEED)
    obs, info = env.reset(seed=SEED, options={"severity_per_amb": list(severity_tuple)})

    env.set_rrm_budget(B_RRM_MAX)

    per_amb_stats = {k_: {
        "sinr_samples": [], "prb_samples": [], "service_bits_samples": [],
        "aoi_samples": [], "delay_samples": [], "delivery_success": 0,
        "delivery_fail_bler": 0, "delivery_fail_no_prb": 0,
        "delivery_fail_no_cap": 0, "active_ticks": 0,
    } for k_ in range(K)}

    total_viol_ticks = 0
    total_active_ticks = 0
    c1_viol_per_amb = np.zeros(K)
    c2_viol_per_amb = np.zeros(K)
    c5_viol_per_amb = np.zeros(K)
    c5_active_per_amb = np.zeros(K)
    c3_viol_count = 0
    embb_samples = []

    for step_i in range(N_STEPS):
        action = alloc_fn(env, obs)
        obs, rew, terminated, truncated, info = env.step(action)

        env.set_rrm_budget(B_RRM_MAX)

        for k_ in range(K):
            if env.active_mask[k_]:
                sinr_k = float(env.last_sinr_db[k_])
                prb_k = int(env._last_prb_per_amb[k_])
                cap_k = capacity_per_prb_bps(sinr_k, eta=SHANNON_ETA)
                svc_bits = prb_k * cap_k * TTI_SEC
                aoi_k = float(env._last_aoi_per_amb[k_])
                delay_k = float(env._last_d_e2e_per_amb[k_])
                sev_k = int(env.severity_per_amb[k_])
                d_max_k = SEVERITY_QOS[sev_k]["D_max"]
                aoi_max_k = SEVERITY_QOS[sev_k]["AoI_max"]

                per_amb_stats[k_]["sinr_samples"].append(sinr_k)
                per_amb_stats[k_]["prb_samples"].append(prb_k)
                per_amb_stats[k_]["service_bits_samples"].append(svc_bits)
                per_amb_stats[k_]["aoi_samples"].append(aoi_k)
                per_amb_stats[k_]["delay_samples"].append(delay_k)
                per_amb_stats[k_]["active_ticks"] += 1

                if delay_k > d_max_k:
                    c1_viol_per_amb[k_] += 1
                    c2_viol_per_amb[k_] += 1
                if aoi_k > aoi_max_k:
                    c5_viol_per_amb[k_] += 1
                c5_active_per_amb[k_] += 1

        embb_mbps = env.last_embb_mbps
        embb_samples.append(embb_mbps)
        d3_floor = 10.0
        if embb_mbps < d3_floor:
            c3_viol_count += 1

        if info.get("c_vec") is not None:
            cv = info["c_vec"]

        for k_ in range(K):
            prev_arr = per_amb_stats[k_]["delivery_success"]

        if terminated or truncated:
            break

    for k_ in range(K):
        s = per_amb_stats[k_]
        s["delivery_success"] = int(env._aoi_pkt_delivered[k_])
        s["delivery_fail_bler"] = int(env._aoi_pkt_failed_bler[k_])
        s["delivery_fail_no_prb"] = int(env._aoi_pkt_failed_no_prb[k_])
        s["delivery_fail_no_cap"] = int(env._aoi_pkt_failed_no_capacity[k_])

    result = {
        "alloc_name": alloc_name,
        "severity_tuple": severity_tuple,
        "steps_run": min(step_i + 1, N_STEPS),
    }

    print(f"\n{'='*80}")
    print(f"ALLOCATION: {alloc_name}  |  severity={severity_tuple}  |  steps={result['steps_run']}")
    print(f"{'='*80}")

    all_c_pass = True
    for k_ in range(K):
        s = per_amb_stats[k_]
        sev_k = int(severity_tuple[k_])
        n = s["active_ticks"]
        if n == 0:
            print(f"  amb_{k_}: NEVER ACTIVE (sev={sev_k})")
            continue
        sinr_mean = np.mean(s["sinr_samples"])
        sinr_min = np.min(s["sinr_samples"])
        prb_mean = np.mean(s["prb_samples"])
        svc_bits_mean = np.mean(s["service_bits_samples"])
        aoi_mean = np.mean(s["aoi_samples"])
        aoi_max_obs = np.max(s["aoi_samples"])
        delay_mean = np.mean(s["delay_samples"])
        d_max = SEVERITY_QOS[sev_k]["D_max"]
        aoi_max_thresh = SEVERITY_QOS[sev_k]["AoI_max"]
        eps = SEVERITY_QOS[sev_k]["eps"]
        eps_aoi = SEVERITY_QOS[sev_k]["eps_aoi"]

        c1_rate = c1_viol_per_amb[k_] / n if n > 0 else 0
        c5_rate = c5_viol_per_amb[k_] / c5_active_per_amb[k_] if c5_active_per_amb[k_] > 0 else 0

        c1_pass = delay_mean <= d_max
        c2_pass = c1_rate <= eps
        c5_pass_flag = c5_rate <= eps_aoi

        pkt_total = s["delivery_success"] + s["delivery_fail_bler"] + s["delivery_fail_no_prb"] + s["delivery_fail_no_cap"]
        delivery_rate = s["delivery_success"] / pkt_total if pkt_total > 0 else 0.0

        min_sinr_needed = min_sinr_for_prb(int(prb_mean))
        cap_feasible = sinr_min >= min_sinr_needed

        print(f"  amb_{k_} (sev={sev_k}, active={n} ticks):")
        print(f"    SINR: mean={sinr_mean:.1f} dB, min={sinr_min:.1f} dB")
        print(f"    PRB:  mean={prb_mean:.0f}, min_sinr_for_delivery={min_sinr_needed:.1f} dB")
        print(f"    service_bits/tick: mean={svc_bits_mean:.0f} (need≥{URLLC_PKT_BITS})")
        print(f"    capacity_feasible: {cap_feasible} (sinr_min={sinr_min:.1f} vs need={min_sinr_needed:.1f})")
        print(f"    Delay: mean={delay_mean*1e3:.3f} ms (D_max={d_max*1e3:.1f} ms) → C1={'PASS' if c1_pass else 'FAIL'}")
        print(f"    C2 viol_rate={c1_rate:.6f} (eps={eps}) → C2={'PASS' if c2_pass else 'FAIL'}")
        print(f"    AoI: mean={aoi_mean:.4f}s, max={aoi_max_obs:.4f}s (AoI_max={aoi_max_thresh}s)")
        print(f"    C5 viol_rate={c5_rate:.6f} (eps_aoi={eps_aoi}) → C5={'PASS' if c5_pass_flag else 'FAIL'}")
        print(f"    Packets: delivered={s['delivery_success']}, fail_bler={s['delivery_fail_bler']}, "
              f"fail_no_prb={s['delivery_fail_no_prb']}, fail_no_cap={s['delivery_fail_no_cap']}")
        print(f"    Delivery rate: {delivery_rate:.4f}")

        if not (c1_pass and c2_pass and c5_pass_flag):
            all_c_pass = False

    embb_mean = np.mean(embb_samples) if embb_samples else 0
    c3_rate = c3_viol_count / len(embb_samples) if embb_samples else 0
    c3_pass = c3_rate <= 0.01
    print(f"  eMBB: mean={embb_mean:.1f} Mbps, C3_viol={c3_rate:.4f} → C3={'PASS' if c3_pass else 'FAIL'}")
    if not c3_pass:
        all_c_pass = False

    print(f"  === ALL CONSTRAINTS: {'PASS' if all_c_pass else 'FAIL'} ===")
    return all_c_pass, per_amb_stats, embb_mean


def alloc_equal(env, obs):
    return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

def alloc_favor_0(env, obs):
    return np.array([0.0, 3.0, -3.0, -3.0], dtype=np.float32)

def alloc_favor_1(env, obs):
    return np.array([0.0, -3.0, 3.0, -3.0], dtype=np.float32)

def alloc_favor_2(env, obs):
    return np.array([0.0, -3.0, -3.0, 3.0], dtype=np.float32)

def alloc_severity_heuristic(env, obs):
    sev = env.severity_per_amb.astype(np.float32)
    logits = sev / 5.0 * 3.0
    return np.concatenate([[0.0], logits]).astype(np.float32)

def alloc_qos_slack_heuristic(env, obs):
    K_ = env.config.K_ambulances
    logits = np.zeros(K_, dtype=np.float32)
    for k_ in range(K_):
        if env.active_mask[k_]:
            aoi_k = env._last_aoi_per_amb[k_]
            aoi_max_k = SEVERITY_QOS[int(env.severity_per_amb[k_])]["AoI_max"]
            logits[k_] = max(0, aoi_k / aoi_max_k) * 3.0
    return np.concatenate([[2.0], logits]).astype(np.float32)


# =============== Q9: AoI capacity-gate analysis ===============
def q9_capacity_gate_analysis():
    """Q9: Test if AoI delivery works correctly with varying PRBs."""
    print("\n" + "="*80)
    print("Q9: AoI CAPACITY-GATE ANALYSIS")
    print("="*80)

    print("\nMinimum SINR (dB) needed for AoI packet delivery (service_bits >= 3200):")
    print(f"{'PRB':>6} {'min_SINR(dB)':>14} {'feasible_at_cell_edge(2.7dB)':>30}")
    for prb in [1, 5, 9, 10, 15, 20, 27, 50, 77, 100, 232, 273]:
        ms = min_sinr_for_prb(prb)
        feasible = "YES" if ms <= 2.7 else "NO"
        print(f"  {prb:>4} {ms:>14.1f} {feasible:>30}")

    print(f"\nAt SINR=-15dB (clamp min), max service_bits with all {P_TOTAL} PRBs:")
    cap_min = capacity_per_prb_bps(-15.0, eta=SHANNON_ETA)
    svc_all = P_TOTAL * cap_min * TTI_SEC
    print(f"  cap_per_prb = {cap_min:.0f} bps")
    print(f"  service_bits = {P_TOTAL} × {cap_min:.0f} × {TTI_SEC} = {svc_all:.0f} bits")
    print(f"  vs URLLC_PKT_BITS = {URLLC_PKT_BITS}")
    print(f"  → {'CAN' if svc_all >= URLLC_PKT_BITS else 'CANNOT'} deliver at SINR=-15dB")

    min_sinr_all = min_sinr_for_prb(P_TOTAL)
    print(f"  Minimum SINR for delivery with ALL PRBs: {min_sinr_all:.1f} dB")

    # Test: run env with fixed PRB counts and measure AoI response
    print("\nQ9 controlled test: same scenario, varying fixed URLLC budgets")
    for b_rrm in [0.05, 0.10, 0.30, 0.50, 0.85]:
        prb_urllc = int(b_rrm * P_TOTAL)
        prb_per_amb = prb_urllc // 3

        cfg = macro_mission_config(K_ambulances=3, seed=SEED)
        cfg.episode_duration_sec = 2.0
        cfg.sample_severity = False
        cfg.initial_severity = 3
        env = ORANEnv(cfg, seed=SEED)
        obs, info = env.reset(seed=SEED, options={"severity_per_amb": [3, 3, 3]})
        env.set_rrm_budget(b_rrm)

        for _ in range(100):
            action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            obs, rew, term, trunc, info = env.step(action)
            env.set_rrm_budget(b_rrm)
            if term or trunc:
                break

        aoi_ms = env.mean_aoi_ms()
        viol = env.aoi_violation_rate()
        delivered = env._aoi_pkt_delivered.sum()
        no_cap = env._aoi_pkt_failed_no_capacity.sum()
        no_prb = env._aoi_pkt_failed_no_prb.sum()
        bler_fail = env._aoi_pkt_failed_bler.sum()
        sinr_mean = float(np.mean(env.last_sinr_db))

        print(f"  b_rrm={b_rrm:.2f} PRB_URLLC={prb_urllc:>3} PRB/amb≈{prb_per_amb:>3} "
              f"SINR={sinr_mean:>6.1f}dB "
              f"AoI={aoi_ms:>8.0f}ms viol={viol:.4f} "
              f"del={int(delivered):>4} no_cap={int(no_cap):>4} no_prb={int(no_prb):>4} bler={int(bler_fail):>4}")


def main():
    print("="*80)
    print("Q1 ROOT-CAUSE AUDIT: K=3 PHYSICAL FEASIBILITY")
    print("="*80)

    # First run Q9 capacity analysis
    q9_capacity_gate_analysis()

    # Q1: Oracle feasibility tests
    severity_tuples = [
        (2, 1, 1),
        (5, 1, 5),
        (2, 5, 3),
        (5, 5, 5),
        (3, 3, 3),
    ]

    allocations = [
        ("equal", alloc_equal),
        ("favor_amb0", alloc_favor_0),
        ("favor_amb1", alloc_favor_1),
        ("favor_amb2", alloc_favor_2),
        ("severity_heuristic", alloc_severity_heuristic),
        ("qos_slack_heuristic", alloc_qos_slack_heuristic),
    ]

    print("\n" + "="*80)
    print("Q1: ORACLE MANAGER (B_RRM_MAX) + VARIOUS WORKER ALLOCATIONS")
    print(f"B_RRM_MAX={B_RRM_MAX} → PRB_URLLC={int(B_RRM_MAX*P_TOTAL)}/{P_TOTAL}")
    print("="*80)

    results_summary = []
    for sev_tuple in severity_tuples:
        for alloc_name, alloc_fn in allocations:
            passed, stats, embb = run_oracle_scenario(
                alloc_name, alloc_fn, sev_tuple
            )
            results_summary.append({
                "severity": sev_tuple,
                "alloc": alloc_name,
                "pass": passed,
                "embb": embb,
            })

    # Print summary table
    print("\n" + "="*80)
    print("Q1 SUMMARY TABLE")
    print("="*80)
    print(f"{'severity':>14} {'allocation':>22} {'ALL_C_PASS':>12} {'eMBB(Mbps)':>12}")
    for r in results_summary:
        print(f"  {str(r['severity']):>12} {r['alloc']:>22} "
              f"{'PASS' if r['pass'] else 'FAIL':>12} {r['embb']:>12.1f}")

    any_pass = any(r["pass"] for r in results_summary)
    print(f"\n{'='*80}")
    if any_pass:
        passing = [r for r in results_summary if r["pass"]]
        print(f"CONCLUSION: FEASIBLE — {len(passing)}/{len(results_summary)} allocations pass all constraints")
        print("Root cause is NOT infeasibility. Continue audit: Manager/Worker/PPO.")
    else:
        print("CONCLUSION: INFEASIBLE — NO allocation passes all C1-C5 simultaneously")
        print("Root cause is formulation/environment/feasibility, NOT PPO learning.")
    print("="*80)


if __name__ == "__main__":
    main()
