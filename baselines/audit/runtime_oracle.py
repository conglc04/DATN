"""Gate 9 — Runtime numerical oracle (dump + independent recompute).

Drives the REAL ORANEnv for one Manager window at K=1 and K=3, dumps every
intermediate, and recomputes each derived quantity with an INDEPENDENT
implementation, asserting agreement within tolerance:

  (a) per-vehicle D_e2e from raw queue (lambda, mu) via textbook M/G/1 PK
  (b) normalized deviation (c_vec - d_phi)/dual_scales
  (c) augmented reward r - lambda . max(0, deviation)  (hinge, fixed 2026-06-22
      bonus-masking audit — a slack constraint must contribute 0, not a bonus)
  (d) dual ascent lambda_new = clip(lambda + alpha * mean_window_deviation, 0, Lmax)
      — dual ascent itself stays on the RAW signed deviation (unchanged); only
      the reward-side penalty is hinge-clipped.

The env supplies the RAW signals (queue state, c_vec, d_phi, reward); the oracle
verifies the entire Lagrangian arithmetic and the delay model independently of
LambdaState / oran_env compute paths.

Run:  python -m audit.runtime_oracle
"""

from __future__ import annotations

import math

import numpy as np

from agents.lagrangian import LambdaState
from env.oran_env import EnvConfig, ORANEnv
from utils.config import (
    ALPHA_LAMBDA_DUAL,
    D_BH,
    D_DET,
    D_FH,
    D_STOCH,
    LAMBDA_MAX,
    R_REF_EMBB_MBPS,
    WORKER_STEPS_PER_MANAGER,
    build_d_phi_vector,
    build_dual_scales,
    get_severity_alpha,
)

TOL = 1e-6


def independent_pk_delay(lam: float, mu: float) -> float:
    """Textbook M/G/1 end-to-end delay, recomputed from scratch."""
    if mu <= 0:
        return math.inf
    rho = lam / mu
    if rho >= 0.9:
        return math.inf
    e_s_pure = 1.0 / mu
    var_total = e_s_pure ** 2 + D_STOCH ** 2
    e_s = e_s_pure + D_STOCH
    e_s2 = var_total + e_s ** 2
    e_dq = lam * e_s2 / (2.0 * (1.0 - rho))
    return D_DET + e_s_pure + e_dq + D_FH + D_BH


def run_window(K: int, severity_per_amb, seed: int) -> list:
    failures = []
    cfg = EnvConfig(K_ambulances=K, sample_severity=False, initial_severity=max(severity_per_amb))
    env = ORANEnv(config=cfg, seed=seed)
    _, info = env.reset(seed=seed, options={"severity_per_amb": severity_per_amb})

    ls = LambdaState(K=K)
    ls.reset_episode(severity_per_amb, max(severity_per_amb))
    scale = build_dual_scales(K)

    # Independent window accumulator for the dual
    win_dev = np.zeros(4 * K + 1)
    lam_pre = ls.get_lambda_local().copy()

    action = np.zeros(env.action_space.shape, dtype=np.float32)
    print(f"\n--- K={K}  severity={severity_per_amb}  seed={seed} ---")
    for step in range(WORKER_STEPS_PER_MANAGER):
        obs, reward, term, trunc, info = env.step(action)
        c_vec = np.asarray(info["c_vec"], dtype=np.float64)
        d_phi = np.asarray(info["d_phi"], dtype=np.float64)

        # (b) independent normalized deviation
        dev_indep = (c_vec - d_phi) / scale
        dev_code = ls._normalized_deviation(c_vec, d_phi)
        if not np.allclose(dev_indep, dev_code, atol=TOL):
            failures.append(f"step{step}: deviation mismatch {dev_indep} vs {dev_code}")

        # (c) independent augmented reward (using pre-update lambda, hinge penalty)
        r_aug_indep = float(reward) - float(
            np.dot(ls.get_lambda_local(), np.maximum(0.0, dev_indep))
        )
        r_aug_code = ls.augmented_reward(float(reward), c_vec, d_phi)
        if abs(r_aug_indep - r_aug_code) > TOL:
            failures.append(f"step{step}: r_aug {r_aug_indep} vs {r_aug_code}")

        # (a) independent per-vehicle delay from raw queue state
        for k in range(K):
            q = env.queues[f"urllc_{k}"]
            if q.is_stable and env.active_mask[k]:
                d_indep = independent_pk_delay(q.arrival_rate, q.service_rate)
                d_code = env._compute_e2e_delay_per_amb()[k]
                if abs(d_indep - d_code) > 1e-9:
                    failures.append(
                        f"step{step} veh{k}: delay {d_indep*1e3:.4f}ms vs {d_code*1e3:.4f}ms")

        ls.accumulate(c_vec, d_phi)
        win_dev += dev_indep

        if step == 0:
            print(f"  reward(base)={reward:+.5f}  r_aug={r_aug_code:+.5f}")
            print(f"  c_vec ={np.array2string(c_vec, precision=5, max_line_width=200)}")
            print(f"  d_phi ={np.array2string(d_phi, precision=5, max_line_width=200)}")
            print(f"  dev   ={np.array2string(dev_indep, precision=5, max_line_width=200)}")
        if term or trunc:
            break

    # (d) independent dual ascent
    n = ls.win_steps
    g_hat_indep = win_dev / n
    lam_new_indep = np.clip(lam_pre + ALPHA_LAMBDA_DUAL * g_hat_indep, 0.0, LAMBDA_MAX)
    ls.on_manager_step_end()
    lam_new_code = ls.get_lambda_global()
    if not np.allclose(lam_new_indep, lam_new_code, atol=TOL):
        failures.append(f"dual update mismatch:\n  indep={lam_new_indep}\n  code ={lam_new_code}")
    print(f"  lambda_pre [0:3]={lam_pre[:3]}")
    print(f"  lambda_post[0:3]={lam_new_code[:3]}  (indep={lam_new_indep[:3]})")

    # Reward base FORM check (last MAC tick): r_tick = alpha_e(sev_ref)*log(1+R_eMBB/R_REF).
    # Note: env step() reward is the SUM over 20 MAC ticks; this verifies the
    # single-term FORMULA against the last tick's R_eMBB snapshot.
    sev_ref = max(severity_per_amb)
    _, alpha_e = get_severity_alpha(sev_ref)
    r_embb = env._compute_embb_throughput_mbps()
    r_tick_indep = alpha_e * math.log(1.0 + r_embb / R_REF_EMBB_MBPS)
    print(f"  reward-form (last tick): alpha_e={alpha_e:.3f}  R_eMBB={r_embb:.2f}Mbps  "
          f"r_tick_indep={r_tick_indep:+.5f}  (window reward = sum of 20 ticks)")

    return failures


def main():
    print("=" * 100)
    print("GATE 9 — RUNTIME NUMERICAL ORACLE (real env dump + independent recompute)")
    print("=" * 100)
    all_fail = []
    all_fail += run_window(1, [5], seed=42)
    all_fail += run_window(3, [5, 3, 1], seed=7)
    all_fail += run_window(3, [4, 4, 4], seed=11)
    print("\n" + "=" * 100)
    if all_fail:
        print(f"GATE 9 FAIL — {len(all_fail)} mismatches:")
        for f in all_fail:
            print("  " + f)
        print("=" * 100)
        return 1
    print("GATE 9 PASS — every runtime quantity matches independent recompute within "
          f"tol={TOL}")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
