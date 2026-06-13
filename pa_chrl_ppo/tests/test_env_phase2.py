"""W05 Phase 2 compliance tests — Phase 2.1 reward + Phase 2.2 constraint tracking.

Per docs/13_methodology_walkthrough.md (post-critique restructure 2026-05-26):
    Phase 2.1: r = alpha_e(phi) * log1p(R_eMBB / R_REF_EMBB_MBPS)  (eMBB log-utility ONLY)
               URLLC enforced via Lagrangian C1, C2 (LambdaState) — NOT in reward.
    Phase 2.2: 5 hard constraints c_vec + per-step phase threshold d_phi in info dict
    Phase 1.1: observation s_t^L = 40-dim formal spec (33 fixed + 3K + F = 33+3+4=40 for K=1, F=4)

Gate G2.1 acceptance (W05):
    - obs dim == 40 (K=1, F=4)
    - reward is single-term eMBB log-utility (no URLLC in reward, no enforce_c3/beta_c3 trace)
    - c_vec shape (5,) + d_phi shape (5,) per step
    - get_phase_thresholds helper returns Master Table values
    - info["l_urllc_mean"] available for diagnostics (NOT in reward)
"""

from __future__ import annotations

import math

import numpy as np

from env.oran_env import EnvConfig, ORANEnv
from utils import config as cfg
from utils.config import (
    D_REF_URLLC,
    GAMMA_MANAGER,
    GAMMA_WORKER,
    R_REF_EMBB_MBPS,
    get_phase_alpha,
    get_phase_thresholds,
)


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
    assert cfg.LR_PI_H == 1e-5
    assert cfg.ALPHA_LAMBDA_DUAL == 1e-4
    assert cfg.LR_PI_L == 1e-3


def test_get_phase_thresholds_phi3():
    """Phase 3 SCENE thresholds match Master Table exactly."""
    th = get_phase_thresholds(3)
    assert th["d1"] == 1e-3       # D_max = 1 ms
    assert th["d2"] == 1e-5       # eps = 1e-5
    assert th["d3"] == 0.0        # C3 threshold; R_min_eMBB is embedded in signed gap
    assert th["d4"] == 0.1        # AoI_max_HR = 100 ms
    assert th["d5"] == 1e-3       # eps_AoI = 1e-3


def test_get_phase_thresholds_phi1_relaxed():
    """Phase 1 STANDBY thresholds (relaxed) match Master Table."""
    th = get_phase_thresholds(1)
    assert th["d1"] == 20e-3
    assert th["d2"] == 1e-3
    assert th["d3"] == 0.0
    assert th["d4"] == 1.0
    assert th["d5"] == 1e-2


def test_get_phase_thresholds_invalid_phase():
    """Phase 0 or 6 must raise ValueError."""
    import pytest
    with pytest.raises(ValueError):
        get_phase_thresholds(0)
    with pytest.raises(ValueError):
        get_phase_thresholds(6)


def test_get_phase_alpha_sums_to_one():
    """alpha_U + alpha_e == 1.0 per phase (PHASE_ALPHA invariant)."""
    for phi in range(1, 6):
        au, ae = get_phase_alpha(phi)
        assert abs(au + ae - 1.0) < 1e-9, f"phase {phi}: alpha sum != 1"


def test_get_phase_alpha_phi3_urllc_priority():
    """Phase 3 SCENE: URLLC heavily prioritized (alpha_U = 0.95)."""
    au, ae = get_phase_alpha(3)
    assert au == 0.95
    assert ae == 0.05


# ----------------------------------------------------------------------------
# Env Phase 1.1 — 40-dim observation
# ----------------------------------------------------------------------------


def test_observation_dim_40_with_K1_F4():
    """Phase 1.1 formal spec: |s_t^L| = 33 + 3K + F = 33 + 3 + 4 = 40 for K=1, F=4."""
    env = ORANEnv(config=EnvConfig(K_ambulances=1, num_streams=4), seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (40,), f"Expected (40,), got {obs.shape}"
    assert env.observation_space.shape == (40,)


def test_observation_finite_no_nan():
    """All obs values must be finite (NaN/Inf would crash PPO)."""
    env = ORANEnv(seed=0)
    obs, _ = env.reset(seed=0)
    assert np.all(np.isfinite(obs)), "obs contains NaN/Inf"
    for _ in range(5):
        obs, _, _, _, _ = env.step(env.action_space.sample())
        assert np.all(np.isfinite(obs)), "obs contains NaN/Inf after step"


def test_observation_layout_phase_one_hot():
    """Phase one-hot is at indices [10:15] of fixed 33-dim block."""
    env = ORANEnv(config=EnvConfig(initial_phase=3), seed=0)
    obs, _ = env.reset(seed=0)
    phase_oh = obs[10:15]
    assert phase_oh.sum() == 1.0
    assert phase_oh[2] == 1.0  # phase 3 (1-indexed) → position 2


def test_observation_no_duplicate_q_bug():
    """W05 fix: arrival rate appears once at [7:9] (NOT duplicated at [9:11] like old stub)."""
    env = ORANEnv(seed=0)
    obs, _ = env.reset(seed=0)
    # arr_urllc at index 7, arr_emBB at index 8
    # rho_urllc at index 0, rho_emBB at index 1 (different signal — utilization)
    # These should be different scales/values when arrival_rate > 0
    env.queues["urllc"].set_arrival_rate(500.0)
    env.queues["eMBB"].set_arrival_rate(1000.0)
    obs2 = env._observe()
    # arrival rates normalized: urllc/1e3 = 0.5, eMBB/1e4 = 0.1
    assert abs(obs2[7] - 0.5) < 1e-5, f"Expected arr_urllc/1e3 = 0.5, got {obs2[7]}"
    assert abs(obs2[8] - 0.1) < 1e-5, f"Expected arr_eMBB/1e4 = 0.1, got {obs2[8]}"


# ----------------------------------------------------------------------------
# Env Phase 2.1 — pure multi-objective reward
# ----------------------------------------------------------------------------


def test_reward_is_embb_log_utility_only():
    """Reward = alpha_e(phi) * log1p(R_eMBB / R_REF_EMBB_MBPS) ONLY (post-critique restructure).

    URLLC enforced via Lagrangian C1, C2 in LambdaState — NOT in reward.
    Reward must be non-negative single-term.
    """
    env = ORANEnv(config=EnvConfig(initial_phase=3), seed=0)
    env.reset(seed=0)
    rewards = []
    for _ in range(10):
        _, r, _, _, _ = env.step(env.action_space.sample())
        rewards.append(r)
    assert all(math.isfinite(r) for r in rewards), "non-finite reward"
    # Single-term alpha_e * log1p(R/R_REF) is non-negative (log1p >= 0 for R >= 0; alpha_e > 0)
    assert all(r >= -1e-9 for r in rewards), (
        f"reward must be >= 0 (single-term eMBB log-utility); got {rewards}"
    )
    # Bounded above: alpha_e_max=0.7 * log(1 + R_max/100) — even at R=2 Gbps → log(21) ~ 3.04 → r ~ 2.1
    assert all(r < 10.0 for r in rewards), f"reward unexpectedly large: {rewards}"


def test_info_exports_l_urllc_for_diagnostics():
    """L_URLLC remains available in info dict for diagnostics, but is NOT used in reward."""
    env = ORANEnv(config=EnvConfig(initial_phase=3), seed=0)
    env.reset(seed=0)
    _, _, _, _, info = env.step(env.action_space.sample())
    assert "l_urllc_mean" in info, "l_urllc_mean must be exported for diagnostics"
    assert info["l_urllc_mean"] >= 0.0, "l_urllc_mean must be non-negative"


def test_reward_phi1_higher_than_phi3_due_to_alpha_embb():
    """Phase 1 has alpha_e=0.7 → reward ~14x larger than phase 3 (alpha_e=0.05) for same R_eMBB.

    Single-term reward r = alpha_e(phi) * log(1 + R/R_REF), so ratio = alpha_e(1)/alpha_e(3) = 14.
    """
    env1 = ORANEnv(config=EnvConfig(initial_phase=1), seed=0)
    env1.reset(seed=0)
    r1_total = sum(env1.step(env1.action_space.sample())[1] for _ in range(20))

    env3 = ORANEnv(config=EnvConfig(initial_phase=3), seed=0)
    env3.reset(seed=0)
    r3_total = sum(env3.step(env3.action_space.sample())[1] for _ in range(20))

    # Phase 1 (alpha_e=0.7) should yield strictly larger cumulative reward than phase 3 (alpha_e=0.05)
    # under similar eMBB throughput conditions. Soft check (random actions may vary throughput).
    assert r1_total >= r3_total, (
        f"Phase 1 (alpha_e=0.7) cumulative reward {r1_total} should be >= "
        f"phase 3 (alpha_e=0.05) cumulative reward {r3_total}"
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
    """Reviewer M2 (Gemini W02, 2026-05-27): σ_S² + ρ + μ exposed via info dict.

    Allows reviewers to audit Pollaczek-Khinchine formula application:
       E[D_queue] = λ · E[S²] / (2(1 − ρ))
    with E[S²] = σ_S² + (E[S])² and σ_S² = 1/μ² + d_stoch² (Exponential + retx).
    See docs/13 §1.3 + docs/04 §Queue Model.
    """
    env = ORANEnv(config=EnvConfig(initial_phase=3), seed=0)
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


def test_info_d_phi_matches_master_table_phi3():
    """d_phi at phase 3 must match Master Table values exactly."""
    env = ORANEnv(config=EnvConfig(initial_phase=3), seed=0)
    env.reset(seed=0)
    _, _, _, _, info = env.step(env.action_space.sample())
    d_phi = info["d_phi"]
    expected = [1e-3, 1e-5, 0.0, 0.1, 1e-3]  # d1..d5 at phase 3
    for j, exp in enumerate(expected):
        assert abs(float(d_phi[j]) - exp) < 1e-6, (
            f"d_phi[{j}] = {d_phi[j]} != expected {exp} at phase 3"
        )


def test_info_d_phi_changes_with_phase():
    """d_phi[0] (D_max) must change when phase changes (1ms@phi_3 vs 20ms@phi_1)."""
    env3 = ORANEnv(config=EnvConfig(initial_phase=3), seed=0)
    env3.reset(seed=0)
    _, _, _, _, info3 = env3.step(env3.action_space.sample())

    env1 = ORANEnv(config=EnvConfig(initial_phase=1), seed=0)
    env1.reset(seed=0)
    _, _, _, _, info1 = env1.step(env1.action_space.sample())

    assert info3["d_phi"][0] == 1e-3
    assert info1["d_phi"][0] == 20e-3
    assert info3["d_phi"][0] != info1["d_phi"][0]


def test_c_vec_aggregates_20_mac_ticks():
    """c_vec is mean across 20 MAC ticks per Worker step (Phase 1.4 ratio)."""
    env = ORANEnv(config=EnvConfig(initial_phase=3), seed=0)
    env.reset(seed=0)
    _, _, _, _, info = env.step(env.action_space.sample())
    # Worker step counter should reflect 20 ticks
    assert env._worker_tick_count == 20, f"Expected 20 MAC ticks/Worker step, got {env._worker_tick_count}"


def test_c_vec_c3_signed_gap_finite():
    """c_vec[2] is signed R_min - R_eMBB, so surplus throughput may make it negative."""
    env = ORANEnv(config=EnvConfig(initial_phase=3), seed=0)
    env.reset(seed=0)
    for _ in range(5):
        _, _, _, _, info = env.step(env.action_space.sample())
        assert np.isfinite(info["c_vec"][2]), f"c_vec[2] = {info['c_vec'][2]} should be finite"


def test_c_vec_c2_c5_indicator_in_unit_interval():
    """c_vec[1] (URLLC tail viol) and c_vec[4] (AoI tail viol) are fractions ∈ [0, 1]."""
    env = ORANEnv(seed=0)
    env.reset(seed=0)
    for _ in range(5):
        _, _, _, _, info = env.step(env.action_space.sample())
        assert 0.0 <= info["c_vec"][1] <= 1.0, f"c_vec[1] = {info['c_vec'][1]} should be in [0, 1]"
        assert 0.0 <= info["c_vec"][4] <= 1.0, f"c_vec[4] = {info['c_vec'][4]} should be in [0, 1]"


def test_info_phase_now_alias():
    """info['phase_now'] is alias for info['phase'] (docs/13 naming convention)."""
    env = ORANEnv(seed=0)
    env.reset(seed=0)
    _, _, _, _, info = env.step(env.action_space.sample())
    assert info["phase_now"] == info["phase"]
