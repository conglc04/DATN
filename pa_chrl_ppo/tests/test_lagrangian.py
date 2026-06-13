"""W06 Lagrangian infrastructure tests — LambdaState class standalone.

Per docs/13_methodology_walkthrough.md:
    Phase 2.3.3 — dual update rule (projected gradient ascent)
    Phase 2.3.5 — Option b interval-window subgradient
    Phase 3.2.6 — λ_warm[φ] EMA refresh table
    Phase 3.4.4 N9 — phase transition handling (Fix Error 1: sync both global + local)

Tests verify each property independently of RL algorithm.
"""

from __future__ import annotations

import numpy as np

from agents.lagrangian import CONSTRAINT_DUAL_SCALES, LambdaState, N_HARD_CONSTRAINTS
from utils.config import ALPHA_LAMBDA_DUAL, LAMBDA_WARM, WORKER_STEPS_PER_MANAGER


# ----------------------------------------------------------------------------
# Construction / config defaults
# ----------------------------------------------------------------------------


def test_default_construction():
    """Default LambdaState has 5-dim λ + locked α_λ + W=10."""
    lam = LambdaState()
    assert lam.n_constraints == 5
    assert lam.alpha_lambda == ALPHA_LAMBDA_DUAL
    assert lam.alpha_lambda == 1e-4
    assert lam.worker_steps_per_manager == WORKER_STEPS_PER_MANAGER
    assert lam.lambda_global.shape == (5,)
    assert lam.lambda_local.shape == (5,)
    assert set(lam.lambda_warm.keys()) == {1, 2, 3, 4, 5}


def test_default_lambda_warm_matches_config():
    """λ_warm table loaded from utils.config.LAMBDA_WARM at construction."""
    lam = LambdaState()
    for phi in range(1, 6):
        expected = np.asarray(LAMBDA_WARM[phi], dtype=np.float64)
        np.testing.assert_array_equal(lam.lambda_warm[phi], expected)


# ----------------------------------------------------------------------------
# Episode reset — Fix Error 1: sync BOTH λ_global + λ_local
# ----------------------------------------------------------------------------


def test_reset_episode_syncs_both_lambdas():
    """reset_episode must set λ_global AND λ_local from λ_warm[phase]."""
    lam = LambdaState()
    lam.reset_episode(initial_phase=3)
    # Phase 3 SCENE warm: [1.80, 2.20, 0.10, 1.50, 2.00]
    expected = np.array([1.80, 2.20, 0.10, 1.50, 2.00])
    np.testing.assert_allclose(lam.lambda_global, expected)
    np.testing.assert_allclose(lam.lambda_local, expected)
    # They must be COPIES (mutating one doesn't affect the other)
    lam.lambda_global[0] = 99.0
    assert lam.lambda_local[0] == 1.80, "λ_local should be independent copy of λ_global"


def test_reset_episode_resets_window():
    """reset_episode must zero win_c and win_steps."""
    lam = LambdaState()
    lam.win_c = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    lam.win_steps = 7
    lam.reset_episode(initial_phase=1)
    np.testing.assert_array_equal(lam.win_c, np.zeros(5))
    assert lam.win_steps == 0


def test_reset_episode_invalid_phase():
    """reset_episode(phi=0) or phi=6 must raise ValueError."""
    lam = LambdaState()
    import pytest
    with pytest.raises(ValueError):
        lam.reset_episode(initial_phase=0)
    with pytest.raises(ValueError):
        lam.reset_episode(initial_phase=99)


# ----------------------------------------------------------------------------
# Phase transition sync — Fix Error 1 (CRITICAL)
# ----------------------------------------------------------------------------


def test_phase_transition_syncs_both_lambdas():
    """on_manager_step_start(phi_now ≠ phi_prev) syncs BOTH global + local."""
    lam = LambdaState()
    lam.reset_episode(initial_phase=1)  # λ ← LAMBDA_WARM[1]
    # Simulate some learning at φ=1
    lam.lambda_global = np.array([0.5, 0.3, 1.2, 0.4, 0.2])
    lam.lambda_local = lam.lambda_global.copy()
    # Phase transition φ_1 → φ_3
    lam.on_manager_step_start(phi_now=3)
    # Expected: λ_global AND λ_local both reload from λ_warm[3]
    warm_3 = np.asarray(LAMBDA_WARM[3], dtype=np.float64)
    np.testing.assert_allclose(lam.lambda_global, warm_3)
    np.testing.assert_allclose(lam.lambda_local, warm_3)
    assert lam.phi_prev == 3


def test_phase_transition_no_op_when_same_phase():
    """on_manager_step_start(phi_now == phi_prev) is a no-op."""
    lam = LambdaState()
    lam.reset_episode(initial_phase=3)
    before_global = lam.lambda_global.copy()
    lam.lambda_global[0] = 5.0  # mutate
    lam.on_manager_step_start(phi_now=3)  # same phase, should not reset
    assert lam.lambda_global[0] == 5.0, "Same-phase call should not overwrite λ"


def test_phase_transition_ema_saves_old_lambda():
    """On transition φ_1 → φ_3, λ_warm[1] gets EMA update from current λ_global."""
    lam = LambdaState(beta_ema=0.5)  # large β for visible effect
    lam.reset_episode(initial_phase=1)
    initial_warm_1 = lam.lambda_warm[1].copy()  # [0.10, 0.20, 0.00, 0.10, 0.00]
    # Simulate λ_global drifted during φ_1
    new_lambda = np.array([2.0, 3.0, 0.5, 1.5, 1.0])
    lam.lambda_global = new_lambda.copy()
    # Transition φ_1 → φ_3
    lam.on_manager_step_start(phi_now=3)
    # Expected: λ_warm[1] = 0.5 * old + 0.5 * new
    expected_warm_1 = 0.5 * initial_warm_1 + 0.5 * new_lambda
    np.testing.assert_allclose(lam.lambda_warm[1], expected_warm_1, rtol=1e-9)


def test_phase_transition_resets_window():
    """On phase transition, win_c + win_steps reset (window context changed)."""
    lam = LambdaState()
    lam.reset_episode(initial_phase=1)
    lam.win_c = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    lam.win_steps = 7
    lam.on_manager_step_start(phi_now=3)
    np.testing.assert_array_equal(lam.win_c, np.zeros(5))
    assert lam.win_steps == 0


def test_episode_end_ema_saves_last_active_phase():
    """Episode boundary must persist the last active phase before reset_episode reloads."""
    lam = LambdaState(beta_ema=0.5)
    lam.reset_episode(initial_phase=4)
    initial_warm_4 = lam.lambda_warm[4].copy()
    lam.lambda_global = np.array([2.0, 2.5, 0.2, 1.8, 1.9])
    lam.on_episode_end(final_phase=5)
    expected_warm_4 = 0.5 * initial_warm_4 + 0.5 * np.array([2.0, 2.5, 0.2, 1.8, 1.9])
    np.testing.assert_allclose(lam.lambda_warm[4], expected_warm_4)
    np.testing.assert_allclose(lam.lambda_global, lam.lambda_warm[5])
    assert lam.phi_prev == 5


# ----------------------------------------------------------------------------
# Interval-window accumulation — Option b (Phase 2.3.5)
# ----------------------------------------------------------------------------


def test_accumulate_adds_deviation():
    """accumulate(c, d) adds (c - d) to win_c and increments win_steps."""
    lam = LambdaState()
    lam.reset_episode(initial_phase=1)
    c_vec = np.array([2.0, 0.1, 5.0, 0.5, 0.01])
    d_phi = np.array([1.0, 0.05, 10.0, 1.0, 0.1])
    lam.accumulate(c_vec, d_phi)
    expected_deviation = (c_vec - d_phi) / CONSTRAINT_DUAL_SCALES
    np.testing.assert_allclose(lam.win_c, expected_deviation)
    assert lam.win_steps == 1
    # Add again
    lam.accumulate(c_vec, d_phi)
    np.testing.assert_allclose(lam.win_c, 2 * expected_deviation)
    assert lam.win_steps == 2


def test_accumulate_validates_shape():
    """accumulate() raises on mismatched shapes."""
    lam = LambdaState()
    import pytest
    with pytest.raises(ValueError):
        lam.accumulate(np.zeros(3), np.zeros(5))
    with pytest.raises(ValueError):
        lam.accumulate(np.zeros(5), np.zeros(4))


# ----------------------------------------------------------------------------
# Dual ascent — Phase 2.3.3 sign convention + Fix Error 2 reset
# ----------------------------------------------------------------------------


def test_dual_ascent_sign_positive_deviation_increases_lambda():
    """c_j > d_j (constraint violated) → g_hat > 0 → λ_j increases (penalty stronger)."""
    lam = LambdaState()
    lam.reset_episode(initial_phase=3)
    initial_lambda = lam.lambda_global.copy()
    # All constraints violated by +1.0
    c_vec = np.array([2.0, 1e-3, 100.0, 1.0, 0.1])
    d_phi = np.array([1.0, 1e-5, 0.0, 0.1, 1e-3])  # c > d for all j
    # Accumulate W=10 worker steps (1 Manager step)
    for _ in range(10):
        lam.accumulate(c_vec, d_phi)
    lam.on_manager_step_end()
    # All λ_j should increase (penalty stronger when violated)
    assert np.all(lam.lambda_global > initial_lambda), (
        f"λ should increase when c > d. before: {initial_lambda}, after: {lam.lambda_global}"
    )


def test_dual_ascent_sign_negative_deviation_decreases_lambda():
    """c_j < d_j (constraint satisfied) → g_hat < 0 → λ_j decreases (relax)."""
    lam = LambdaState()
    lam.reset_episode(initial_phase=3)
    initial_lambda = lam.lambda_global.copy()
    # All constraints satisfied (c < d)
    c_vec = np.array([0.5, 1e-6, 0.0, 0.05, 1e-4])
    d_phi = np.array([1.0, 1e-5, 0.0, 0.1, 1e-3])
    # Note: c3 < d3 → c-d=20 (positive) wait that's not satisfied
    # For eMBB throughput c3 is the signed gap R_min - R_eMBB and d3 = 0.
    # So c3 needs to be < 0 which is impossible (deficit ≥ 0)
    # Use simpler artificial test: all c < d
    c_test = np.array([0.5, 1e-6, 0.0, 0.05, 1e-4])
    d_test = np.array([1.0, 1e-5, 0.0, 0.1, 1e-3])  # d3=0 (defined floor) so c-d=0
    for _ in range(10):
        lam.accumulate(c_test, d_test)
    lam.on_manager_step_end()
    # λ should decrease (or stay 0) where c-d < 0
    # λ_1, λ_2, λ_4, λ_5 should decrease; λ_3 stays
    for j in [0, 1, 3, 4]:
        assert lam.lambda_global[j] <= initial_lambda[j], (
            f"λ_{j} should decrease when c-d<0. before: {initial_lambda[j]}, after: {lam.lambda_global[j]}"
        )


def test_dual_ascent_lambda_nonneg_projection():
    """λ projection: max(0, λ + α·g_hat) — never negative."""
    lam = LambdaState()
    lam.reset_episode(initial_phase=3)
    # Force λ to want to go very negative
    lam.lambda_global = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
    # Massively negative deviation
    c_vec = np.zeros(5)
    d_phi = np.full(5, 1e6)  # c-d = -1e6
    for _ in range(10):
        lam.accumulate(c_vec, d_phi)
    lam.on_manager_step_end()
    assert np.all(lam.lambda_global >= 0.0), "λ must be ≥ 0 (projection)"


def test_dual_ascent_lambda_max_projection():
    """Reviewer M4 (Gemini W06, 2026-05-27): λ projection clips at LAMBDA_MAX.

    Bounded projection Π_Λ(λ) = clip(λ, 0, LAMBDA_MAX) prevents dual blow-up
    under sustained constraint violations. Soft safety net (empirical λ ≤ 2.5).
    """
    from utils.config import LAMBDA_MAX

    lam = LambdaState()
    lam.reset_episode(initial_phase=3)
    # Force massive positive subgradient → dual ascent wants to blow up
    lam.lambda_global = np.full(5, LAMBDA_MAX - 0.1)  # near ceiling
    # Massively positive deviation (huge constraint violation)
    c_vec = np.full(5, 1e10)
    d_phi = np.zeros(5)
    for _ in range(10):
        lam.accumulate(c_vec, d_phi)
    lam.on_manager_step_end()
    assert np.all(lam.lambda_global <= LAMBDA_MAX + 1e-6), (
        f"λ must be ≤ LAMBDA_MAX={LAMBDA_MAX}. got: {lam.lambda_global}"
    )
    # And local matches global (push to Worker happens at end of step)
    assert np.allclose(lam.lambda_local, lam.lambda_global), (
        "lambda_local must mirror lambda_global after on_manager_step_end"
    )


def test_dual_ascent_resets_window_after_update():
    """After on_manager_step_end(), win_c + win_steps reset to 0 (Fix Error 2)."""
    lam = LambdaState()
    lam.reset_episode(initial_phase=3)
    lam.accumulate(np.ones(5), np.zeros(5))
    lam.accumulate(np.ones(5), np.zeros(5))
    assert lam.win_steps == 2
    lam.on_manager_step_end()
    np.testing.assert_array_equal(lam.win_c, np.zeros(5))
    assert lam.win_steps == 0


def test_dual_ascent_pushes_lambda_local():
    """After dual update, λ_local = λ_global.copy() (push to Worker)."""
    lam = LambdaState()
    lam.reset_episode(initial_phase=3)
    # Modify local artificially to be different
    lam.lambda_local = np.zeros(5)
    # Accumulate + step
    for _ in range(10):
        lam.accumulate(np.array([2.0, 1e-3, 50.0, 0.5, 0.05]),
                       np.array([1.0, 1e-5, 0.0, 0.1, 1e-3]))
    lam.on_manager_step_end()
    # λ_local should equal λ_global after push
    np.testing.assert_array_equal(lam.lambda_local, lam.lambda_global)


def test_empty_window_skips_dual_update():
    """on_manager_step_end() with win_steps=0 is no-op (no division by zero)."""
    lam = LambdaState()
    lam.reset_episode(initial_phase=3)
    initial_lambda = lam.lambda_global.copy()
    out = lam.on_manager_step_end()
    # λ unchanged
    np.testing.assert_array_equal(lam.lambda_global, initial_lambda)
    # Returns sensible diagnostic
    assert out["subgradient_mean"] == 0.0


# ----------------------------------------------------------------------------
# Augmented reward (Phase 3.2.1)
# ----------------------------------------------------------------------------


def test_augmented_reward_zero_deviation():
    """When c == d (constraints exactly met), r_aug == r."""
    lam = LambdaState()
    lam.reset_episode(initial_phase=3)
    c = np.ones(5)
    d = np.ones(5)  # c - d = 0 → penalty = 0
    r_aug = lam.augmented_reward(reward=-2.5, c_vec=c, d_phi=d)
    assert abs(r_aug - (-2.5)) < 1e-9


def test_augmented_reward_positive_deviation_reduces_reward():
    """When c > d, penalty positive → r_aug < r."""
    lam = LambdaState()
    lam.reset_episode(initial_phase=3)
    # λ_local = LAMBDA_WARM[3] = [1.80, 2.20, 0.10, 1.50, 2.00]
    # c - d = [1, 1, 1, 1, 1] → penalty = sum(λ) = 7.6
    c = np.array([2.0, 1.0+1e-5, 31.0, 1.1, 1.0+1e-3])
    d = np.array([1.0, 1e-5, 30.0, 0.1, 1e-3])  # c-d = [1, 1, 1, 1, 1]
    r_aug = lam.augmented_reward(reward=0.0, c_vec=c, d_phi=d)
    expected_penalty = float(
        np.dot(np.array([1.80, 2.20, 0.10, 1.50, 2.00]), 1.0 / CONSTRAINT_DUAL_SCALES)
    )
    assert abs(r_aug - (-expected_penalty)) < 1e-6


# ----------------------------------------------------------------------------
# Integration scenario — multiple Manager steps with phase transitions
# ----------------------------------------------------------------------------


def test_integration_multi_manager_step_with_transition():
    """End-to-end: 3 Manager steps in φ_3, then transition to φ_1, then 2 more steps."""
    lam = LambdaState()
    lam.reset_episode(initial_phase=3)

    # 3 Manager steps at φ_3 with mild URLLC violation
    for k in range(3):
        for _ in range(10):  # 10 worker steps per Manager
            lam.accumulate(
                c_vec=np.array([1.5e-3, 5e-5, 0.0, 0.05, 1e-4]),
                d_phi=np.array([1e-3, 1e-5, 0.0, 0.1, 1e-3]),
            )
        lam.on_manager_step_end()
    lambda_at_end_phi3 = lam.lambda_global.copy()

    # Phase transition φ_3 → φ_1
    lam.on_manager_step_start(phi_now=1)
    # λ_global should now be λ_warm[1] (after EMA update of λ_warm[3])
    np.testing.assert_allclose(lam.lambda_global, lam.lambda_warm[1])
    # λ_warm[3] should have been EMA-updated (no longer raw initial)
    initial_warm_3 = np.asarray(LAMBDA_WARM[3], dtype=np.float64)
    assert np.max(np.abs(lam.lambda_warm[3] - initial_warm_3)) > 1e-8, (
        "λ_warm[3] should have been EMA-updated from λ_global at φ_3 end"
    )

    # 2 Manager steps at φ_1 with c < d (constraints satisfied)
    for k in range(2):
        for _ in range(10):
            lam.accumulate(
                c_vec=np.array([1e-3, 0.0, -5.0, 0.5, 0.0]),
                d_phi=np.array([20e-3, 1e-3, 0.0, 1.0, 1e-2]),
            )
        lam.on_manager_step_end()

    # No exception, all λ non-negative
    assert np.all(lam.lambda_global >= 0.0)
    assert lam.phi_prev == 1
