"""W05 Phase 2 compliance tests — Phase 2.1 reward + Phase 2.2 constraint tracking.

Per docs/13_methodology_walkthrough.md (post-critique restructure 2026-05-26):
    Phase 2.1: r = alpha_e(phi) * log1p(R_eMBB / R_REF_EMBB_MBPS)  (eMBB log-utility ONLY)
               URLLC enforced via Lagrangian C1, C2 (LambdaState) — NOT in reward.
    Phase 2.2: 5 hard constraints c_vec + per-step severity threshold d_phi in info dict
    Phase 1.1: observation s_t^L = 30-dim spec (24 fixed + 5K + F = 24+5+1=30 for K=1, F=1)

Gate G2.1 acceptance (W05; obs reshaped 2026-06-14 phase→severity 33→31, then
F=4→F=1 stream consolidation 31→28):
    - obs dim == 28 (K=1, F=1)
    - reward is single-term eMBB log-utility (no URLLC in reward, no enforce_c3/beta_c3 trace)
    - c_vec shape (5,) + d_phi shape (5,) per step
    - get_severity_thresholds helper returns Master Table values
    - info["l_urllc_mean"] available for diagnostics (NOT in reward)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from env.oran_env import EnvConfig, ORANEnv
from utils import config as cfg
from utils.config import (
    D_REF_URLLC,
    GAMMA_MANAGER,
    GAMMA_WORKER,
    R_REF_EMBB_MBPS,
    get_severity_alpha,
    get_severity_thresholds,
)


# ----------------------------------------------------------------------------
# Test helpers
# ----------------------------------------------------------------------------


def _force_severity(env: ORANEnv, sev: int) -> None:
    """Pin env severity state after reset (white-box unit test — does NOT use sample_severity=False).

    sample_severity=True is a project requirement; formula-testing unit tests must set
    severity directly on the live env object instead of relying on the config field.
    """
    env.severity = sev
    env.severity_per_amb = np.full(env.config.K_ambulances, sev, dtype=int)


# ----------------------------------------------------------------------------
# Config helpers (Phase 1.3 Master Table)
# ----------------------------------------------------------------------------


def test_phase2_constants_present():
    """Phase 2 reward normalization constants must be in config."""
    assert D_REF_URLLC == 1e-3, "D_REF_URLLC should be 1 ms (tightest D_max)"
    assert R_REF_EMBB_MBPS == 100.0, "R_REF_EMBB_MBPS should be 100 Mbps"


def test_gamma_hierarchy():
    """GAMMA_MANAGER = GAMMA_WORKER^WORKER_STEPS_PER_MANAGER (Phase 3.2.4)."""
    assert GAMMA_WORKER == 0.99
    expected_gamma_h = 0.99 ** cfg.WORKER_STEPS_PER_MANAGER
    assert abs(GAMMA_MANAGER - expected_gamma_h) < 1e-9
    assert abs(GAMMA_MANAGER - 0.9043820750088045) < 1e-9


def test_three_rate_hierarchy_locked():
    """alpha_pi_H < alpha_lambda < alpha_pi_L (Phase 2.3.4 three-rate hierarchy)."""
    assert cfg.LR_PI_H < cfg.ALPHA_LAMBDA_DUAL < cfg.LR_PI_L
    assert cfg.LR_PI_H == 3e-5
    assert cfg.ALPHA_LAMBDA_DUAL == 2e-4
    assert cfg.LR_PI_L == 3e-4


def test_get_severity_thresholds_immediate():
    """Severity 5 IMMEDIATE thresholds (tightest) match Master Table exactly."""
    th = get_severity_thresholds(5)
    assert th["d1"] == 1e-3       # D_max = 1 ms
    assert th["d2"] == 1e-5       # eps = 1e-5
    assert th["d3"] == 0.0        # C3 threshold; R_min_eMBB is embedded in signed gap
    assert th["d4"] == 0.1        # AoI_max = 100 ms
    assert th["d5"] == 1e-3       # eps_AoI = 1e-3


def test_get_severity_thresholds_nonurgent_relaxed():
    """Severity 1 NON_URGENT thresholds (relaxed) match Master Table."""
    th = get_severity_thresholds(1)
    assert th["d1"] == 20e-3
    assert th["d2"] == 1e-3
    assert th["d3"] == 0.0
    assert th["d4"] == 1.0
    assert th["d5"] == 1e-2


def test_get_severity_thresholds_invalid_phase():
    """Phase 0 or 6 must raise ValueError."""
    import pytest
    with pytest.raises(ValueError):
        get_severity_thresholds(0)
    with pytest.raises(ValueError):
        get_severity_thresholds(6)


def test_get_severity_alpha_sums_to_one():
    """alpha_U + alpha_e == 1.0 per phase (SEVERITY_ALPHA invariant)."""
    for phi in range(1, 6):
        au, ae = get_severity_alpha(phi)
        assert abs(au + ae - 1.0) < 1e-9, f"phase {phi}: alpha sum != 1"


def test_get_severity_alpha_immediate_urllc_priority():
    """Severity 5 IMMEDIATE: URLLC heavily prioritized (alpha_U = 0.95)."""
    au, ae = get_severity_alpha(5)
    assert au == 0.95
    assert ae == 0.05


# ----------------------------------------------------------------------------
# Env Phase 1.1 — 40-dim observation
# ----------------------------------------------------------------------------


def test_observation_dim_32_with_K1_F1():
    """Formal spec: |s_t^L| = 20 + 11K + F = 20 + 11 + 1 = 32 for K=1, F=1
    (per-ambulance severity_k epic 2026-06-15: fixed 20-dim block +
    11-dim interleaved per-ambulance block [SINR,d,v,delay_norm,AoI_norm,
    severity_norm,lambda_C1,lambda_C2,lambda_C4,lambda_C5,active_mask] + F-dim mean-AoI tail)."""
    env = ORANEnv(config=EnvConfig(K_ambulances=1, num_streams=1), seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (32,), f"Expected (32,), got {obs.shape}"
    assert env.observation_space.shape == (32,)


def test_observation_finite_no_nan():
    """All obs values must be finite (NaN/Inf would crash PPO)."""
    env = ORANEnv(seed=0)
    obs, _ = env.reset(seed=0)
    assert np.all(np.isfinite(obs)), "obs contains NaN/Inf"
    for _ in range(5):
        obs, _, _, _, _ = env.step(env.action_space.sample())
        assert np.all(np.isfinite(obs)), "obs contains NaN/Inf after step"


def test_observation_layout_severity_one_hot():
    """Severity one-hot is at indices [10:15] of the fixed block."""
    env = ORANEnv(config=EnvConfig(), seed=0)
    obs, info = env.reset(seed=0)
    sev_oh = obs[10:15]
    assert sev_oh.sum() == 1.0
    sev = info["severity"]          # sampled severity (1-indexed)
    assert sev_oh[sev - 1] == 1.0  # one-hot at position (severity - 1)


def test_observation_dim_54_with_K3_F1():
    """K=3, F=1: |s_t^L| = 20 + 11K + F = 20 + 33 + 1 = 54."""
    env = ORANEnv(config=EnvConfig(K_ambulances=3, num_streams=1), seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (54,), f"Expected (54,), got {obs.shape}"
    assert env.observation_space.shape == (54,)


def test_per_ambulance_delay_norm_differs_under_unequal_sinr():
    """K=3: ambulances with different SINR get different delay_norm_k/AoI_norm_k,
    even though they share the same severity — fixing the critical observability
    gap (2026-06-14) where the policy could not distinguish per-ambulance QoS
    proximity."""
    K = 3
    env = ORANEnv(config=EnvConfig(K_ambulances=K, num_streams=1), seed=0)
    obs, _ = env.reset(seed=0)

    # Give ambulances very different channel quality, then refresh service rates.
    env.last_sinr_db = np.array([-5.0, 15.0, 35.0], dtype=np.float64)
    for k in range(K):
        env.queues[f"urllc_{k}"].set_arrival_rate(env.config.urllc_arrival_rate)
    env._update_queue_service_rates()

    d_e2e_per_amb = env._compute_e2e_delay_per_amb()
    assert d_e2e_per_amb.shape == (K,)
    assert np.all(np.isfinite(d_e2e_per_amb))
    # Worse SINR (index 0) ⇒ lower service rate ⇒ higher (or equal) D_e2e.
    assert d_e2e_per_amb[0] >= d_e2e_per_amb[2]
    assert len(set(np.round(d_e2e_per_amb, 9))) > 1, "expected differing per-amb delays"

    for _ in range(5):
        obs, _, _, _, info = env.step(env.action_space.sample())

    assert obs.shape == (54,)
    assert np.all(np.isfinite(obs))
    c_vec = info["c_vec"]
    assert c_vec.shape == (4 * K + 1,)
    assert np.all(np.isfinite(c_vec))


def test_observation_no_duplicate_q_bug():
    """W05 fix: arrival rate appears once at [7:9] (NOT duplicated at [9:11] like old stub)."""
    env = ORANEnv(seed=0)
    obs, _ = env.reset(seed=0)
    # arr_urllc at index 7, arr_emBB at index 8
    # rho_urllc at index 0, rho_emBB at index 1 (different signal — utilization)
    # These should be different scales/values when arrival_rate > 0
    env.queues["urllc_0"].set_arrival_rate(500.0)
    env.queues["eMBB"].set_arrival_rate(1000.0)
    obs2 = env._observe()
    # arrival rates normalized: urllc/1e3 = 0.5, eMBB/1e4 = 0.1
    assert abs(obs2[7] - 0.5) < 1e-5, f"Expected arr_urllc/1e3 = 0.5, got {obs2[7]}"
    assert abs(obs2[8] - 0.1) < 1e-5, f"Expected arr_eMBB/1e4 = 0.1, got {obs2[8]}"


# ----------------------------------------------------------------------------
# Env Phase 2.1 — pure multi-objective reward
# ----------------------------------------------------------------------------


def test_reward_is_embb_log_utility_only():
    """Reward = log(1 + R_eMBB / R_REF) — pure eMBB utility, no α_e.

    Severity differentiation is entirely via constraints C1–C5 + λ.
    Reward must be non-negative (log(1+x) ≥ 0 for x ≥ 0).
    Summed over 20 MAC ticks per Worker step.
    """
    env = ORANEnv(config=EnvConfig(initial_severity=5), seed=0)
    env.reset(seed=0)
    rewards = []
    for _ in range(10):
        _, r, _, _, _ = env.step(env.action_space.sample())
        rewards.append(r)
    assert all(math.isfinite(r) for r in rewards), "non-finite reward"
    assert all(r >= -1e-9 for r in rewards), (
        f"reward must be >= 0 (pure eMBB log-utility); got {rewards}"
    )
    # 20 ticks × log(1 + R/100): at R=300 Mbps → log(4)≈1.39 × 20 ≈ 27.7
    assert all(r < 40.0 for r in rewards), f"reward unexpectedly large: {rewards}"


def test_info_exports_l_urllc_for_diagnostics():
    """L_URLLC remains available in info dict for diagnostics, but is NOT used in reward."""
    env = ORANEnv(config=EnvConfig(initial_severity=3), seed=0)
    env.reset(seed=0)
    _, _, _, _, info = env.step(env.action_space.sample())
    assert "l_urllc_mean" in info, "l_urllc_mean must be exported for diagnostics"
    assert info["l_urllc_mean"] >= 0.0, "l_urllc_mean must be non-negative"


def test_reward_invariant_to_severity_no_alpha_e():
    """Reward is INVARIANT to severity at identical dynamics (α_e removed 2026-06-23).

    The reward is pure eMBB log-utility r = mean_tick log(1 + R/R_REF) with NO
    severity weight. Severity does not change channel/traffic dynamics, so with
    the same seed the per-step reward at sev=1 must EQUAL that at sev=5 — severity
    differentiation lives entirely in the constraints C1–C5 + λ, not the reward.
    (Regression guard: if α_e leaks back in, sev=1 reward would be 14× sev=5.)
    """
    env1 = ORANEnv(config=EnvConfig(initial_severity=1, sample_severity=False), seed=0)
    env1.reset(seed=0)
    env1.set_rrm_budget(0.20)
    r1 = env1.step(np.zeros(env1.action_space.shape[0], dtype=np.float32))[1]

    env5 = ORANEnv(config=EnvConfig(initial_severity=5, sample_severity=False), seed=0)
    env5.reset(seed=0)
    env5.set_rrm_budget(0.20)
    r5 = env5.step(np.zeros(env5.action_space.shape[0], dtype=np.float32))[1]

    assert r1 == pytest.approx(r5, rel=1e-9), (
        f"reward differs by severity (sev1={r1}, sev5={r5}) — α_e leaked back in"
    )


# ----------------------------------------------------------------------------
# Env Phase 2.2 — 5-constraint tracking in info dict
# ----------------------------------------------------------------------------


def test_info_has_c_vec_shape_5():
    """env.step() info dict must include c_vec (5,) per Phase 2.2.1."""
    env = ORANEnv(seed=0)
    env.reset(seed=0)
    _, _, _, _, info = env.step(env.action_space.sample())
    assert "c_vec" in info, "Missing c_vec in info"
    assert info["c_vec"].shape == (5,), f"c_vec shape {info['c_vec'].shape} != (5,)"


def test_info_exports_queue_diag_for_pk_audit():
    """Reviewer M2 (internal review, W02, 2026-05-27): σ_S² + ρ + μ exposed via info dict.

    Allows reviewers to audit Pollaczek-Khinchine formula application:
       E[D_queue] = λ · E[S²] / (2(1 − ρ))
    with E[S²] = σ_S² + (E[S])² and σ_S² = 1/μ² + d_stoch² (Exponential + retx).
    See docs/13 §1.3 + docs/04 §Queue Model.
    """
    env = ORANEnv(config=EnvConfig(initial_severity=3), seed=0)
    env.reset(seed=0)
    _, _, _, _, info = env.step(env.action_space.sample())
    for key in ("queue_diag_urllc", "queue_diag_embb"):
        assert key in info, f"Missing {key} in info dict (reviewer M2)"
        diag = info[key]
        # PK formula audit fields
        for field in ("lambda", "mu", "rho", "E_S", "E_S2", "E_D_queue", "stable"):
            assert field in diag, f"{key} missing field '{field}'"
        # Sanity: E[S²] should be positive (computed analytically)
        assert diag["E_S2"] > 0.0, f"{key} E[S²] must be > 0 (got {diag['E_S2']})"
        # Stability invariant (queue must be stable for PK to apply)
        assert 0.0 <= diag["rho"], f"{key} ρ must be ≥ 0 (got {diag['rho']})"
        # E[S²] >= (E[S])² (variance non-negative)
        assert diag["E_S2"] >= diag["E_S"] ** 2 - 1e-9, (
            f"{key} E[S²]={diag['E_S2']} < (E[S])²={diag['E_S'] ** 2} violates Var(S) ≥ 0"
        )


def test_info_has_d_phi_shape_5():
    """env.step() info dict must include d_phi (5,) per Phase 1.3 Master Table lookup."""
    env = ORANEnv(seed=0)
    env.reset(seed=0)
    _, _, _, _, info = env.step(env.action_space.sample())
    assert "d_phi" in info, "Missing d_phi in info"
    assert info["d_phi"].shape == (5,), f"d_phi shape {info['d_phi'].shape} != (5,)"


def test_info_d_phi_matches_master_table_immediate():
    """d_phi at severity 5 IMMEDIATE must match Master Table values exactly.

    (4K+1)-dim layout [d1_C1, d2_C2, d4_C4, d5_C5, d3_C3_shared] (K=1) — the
    permutation [0,1,3,4,2] of the legacy [d1,d2,d3,d4,d5] order."""
    env = ORANEnv(config=EnvConfig(), seed=0)
    env.reset(seed=0)
    _force_severity(env, 5)       # pin severity for formula unit test
    _, _, _, _, info = env.step(env.action_space.sample())
    d_phi = info["d_phi"]
    expected = [1e-3, 1e-5, 0.1, 1e-3, 0.0]  # d1,d2,d4,d5,d3 at severity 5
    for j, exp in enumerate(expected):
        assert abs(float(d_phi[j]) - exp) < 1e-6, (
            f"d_phi[{j}] = {d_phi[j]} != expected {exp} at severity 5"
        )


def test_info_d_phi_changes_with_severity():
    """d_phi[0] (D_max) differs across severity (1ms@IMMEDIATE vs 20ms@NON_URGENT)."""
    env5 = ORANEnv(config=EnvConfig(), seed=0)
    env5.reset(seed=0)
    _force_severity(env5, 5)      # pin severity for formula unit test
    _, _, _, _, info5 = env5.step(env5.action_space.sample())

    env1 = ORANEnv(config=EnvConfig(), seed=0)
    env1.reset(seed=0)
    _force_severity(env1, 1)      # pin severity for formula unit test
    _, _, _, _, info1 = env1.step(env1.action_space.sample())

    assert info5["d_phi"][0] == 1e-3
    assert info1["d_phi"][0] == 20e-3
    assert info5["d_phi"][0] != info1["d_phi"][0]


def test_c_vec_aggregates_20_mac_ticks():
    """c_vec is mean across 20 MAC ticks per Worker step (Phase 1.4 ratio)."""
    env = ORANEnv(config=EnvConfig(initial_severity=3), seed=0)
    env.reset(seed=0)
    _, _, _, _, info = env.step(env.action_space.sample())
    # Worker step counter should reflect 20 ticks
    assert env._worker_tick_count == 20, f"Expected 20 MAC ticks/Worker step, got {env._worker_tick_count}"


def test_c_vec_c3_signed_gap_finite():
    """c_vec[4] (C3_shared, last slot at K=1) is signed R_min - R_eMBB, so surplus
    throughput may make it negative."""
    env = ORANEnv(config=EnvConfig(initial_severity=3), seed=0)
    env.reset(seed=0)
    for _ in range(5):
        _, _, _, _, info = env.step(env.action_space.sample())
        assert np.isfinite(info["c_vec"][4]), f"c_vec[4] = {info['c_vec'][4]} should be finite"


def test_c_vec_c2_c5_indicator_in_unit_interval():
    """c_vec[1] (C2, URLLC tail viol) and c_vec[3] (C5, AoI tail viol) are fractions ∈ [0, 1]."""
    env = ORANEnv(seed=0)
    env.reset(seed=0)
    for _ in range(5):
        _, _, _, _, info = env.step(env.action_space.sample())
        assert 0.0 <= info["c_vec"][1] <= 1.0, f"c_vec[1] = {info['c_vec'][1]} should be in [0, 1]"
        assert 0.0 <= info["c_vec"][3] <= 1.0, f"c_vec[3] = {info['c_vec'][3]} should be in [0, 1]"


def test_info_exports_severity():
    """info exports the exogenous severity (1..5) and its correct name."""
    from utils.config import SEVERITY_QOS
    env = ORANEnv(config=EnvConfig(), seed=0)
    _, reset_info = env.reset(seed=0)
    _, _, _, _, info = env.step(env.action_space.sample())
    assert info["severity"] in SEVERITY_QOS               # valid severity level
    assert info["severity"] == reset_info["severity"]     # consistent within episode
    assert info["severity_name"] == SEVERITY_QOS[info["severity"]]["name"]
