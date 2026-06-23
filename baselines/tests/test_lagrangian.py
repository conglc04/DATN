"""W06 Lagrangian infrastructure tests — LambdaState class standalone.

Per docs/13_methodology_walkthrough.md:
    Phase 2.3.3 — dual update rule (projected gradient ascent)
    Phase 2.3.5 — Option b interval-window subgradient
    Phase 3.2.6 — λ_warm[severity] EMA refresh table
    Phase 3.4.4 N9 — phase transition handling (Fix Error 1: sync both global + local)

Per-ambulance severity_k epic (2026-06-15): LambdaState is now K-aware with
(4K+1)-dim vectors laid out as
    [C1_0..C1_{K-1}, C2_0..C2_{K-1}, C4_0..C4_{K-1}, C5_0..C5_{K-1}, C3_shared]
keyed by ``severity_per_amb`` (tuple, len K) + ``severity_ref`` (= max).
At K=1 this is the permutation [0,1,3,4,2] of the legacy 5-dim
[C1,C2,C3,C4,C5] order — exact numeric preservation.

Tests verify each property independently of RL algorithm.
"""

from __future__ import annotations

import numpy as np
import pytest

from agents.lagrangian import LambdaState, N_HARD_CONSTRAINTS
from utils.config import (
    ALPHA_LAMBDA_DUAL,
    LAMBDA_MAX,
    LAMBDA_WARM,
    WORKER_STEPS_PER_MANAGER,
    build_lambda_warm_vector,
)


# ----------------------------------------------------------------------------
# Construction / config defaults (K=1)
# ----------------------------------------------------------------------------


def test_default_construction():
    """Default LambdaState is K=1 → (4*1+1)=5-dim λ + locked α_λ + W=10."""
    lam = LambdaState()
    assert lam.K == 1
    assert lam.n_constraints == 5 == N_HARD_CONSTRAINTS
    assert lam.alpha_lambda == ALPHA_LAMBDA_DUAL
    assert lam.alpha_lambda == 2e-4  # A/B 5e-4 reverted to 2e-4 on 2026-06-22 (α_λ wrong lever)
    assert lam.worker_steps_per_manager == WORKER_STEPS_PER_MANAGER
    assert lam.lambda_global.shape == (5,)
    assert lam.lambda_local.shape == (5,)
    # Warm table starts empty; entries are populated lazily via EMA on
    # severity transitions / episode end.
    assert lam.lambda_warm == {}


def test_lambda_warm_vector_matches_permutation_of_legacy_table():
    """build_lambda_warm_vector(K=1) == permutation [0,1,3,4,2] of LAMBDA_WARM[sev]."""
    for sev in range(1, 6):
        legacy = np.asarray(LAMBDA_WARM[sev], dtype=np.float64)
        expected = legacy[[0, 1, 3, 4, 2]]
        got = build_lambda_warm_vector((sev,), sev)
        np.testing.assert_array_equal(got, expected)


# ----------------------------------------------------------------------------
# Episode reset — Fix Error 1: sync BOTH λ_global + λ_local
# ----------------------------------------------------------------------------


def test_reset_episode_syncs_both_lambdas():
    """reset_episode must set λ_global AND λ_local from λ_warm[severity_per_amb]."""
    lam = LambdaState()
    lam.reset_episode((5,), 5)
    expected = build_lambda_warm_vector((5,), 5)
    np.testing.assert_allclose(lam.lambda_global, expected)
    np.testing.assert_allclose(lam.lambda_local, expected)
    # They must be COPIES (mutating one doesn't affect the other)
    lam.lambda_global[0] = 99.0
    assert lam.lambda_local[0] == expected[0], "λ_local should be independent copy of λ_global"


def test_reset_episode_resets_window():
    """reset_episode must zero win_c and win_steps."""
    lam = LambdaState()
    lam.win_c = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    lam.win_steps = 7
    lam.reset_episode((1,), 1)
    np.testing.assert_array_equal(lam.win_c, np.zeros(5))
    assert lam.win_steps == 0


def test_reset_episode_invalid_severity():
    """reset_episode with an unknown severity must raise ValueError."""
    lam = LambdaState()
    with pytest.raises(ValueError):
        lam.reset_episode((0,), 0)
    with pytest.raises(ValueError):
        lam.reset_episode((99,), 99)


# ----------------------------------------------------------------------------
# Phase transition sync — Fix Error 1 (CRITICAL)
# ----------------------------------------------------------------------------


def test_phase_transition_syncs_both_lambdas():
    """on_manager_step_start(severity != sev_prev) syncs BOTH global + local."""
    lam = LambdaState()
    lam.reset_episode((1,), 1)  # λ ← λ_warm[(1,)]
    # Simulate some learning at severity 1
    lam.lambda_global = np.array([0.5, 0.3, 1.2, 0.4, 0.2])
    lam.lambda_local = lam.lambda_global.copy()
    # Severity transition (1,) -> (3,)
    lam.on_manager_step_start((3,), 3)
    # Expected: λ_global AND λ_local both reload from λ_warm[(3,)]
    warm_3 = build_lambda_warm_vector((3,), 3)
    np.testing.assert_allclose(lam.lambda_global, warm_3)
    np.testing.assert_allclose(lam.lambda_local, warm_3)
    assert lam.sev_prev == (3,)
    assert lam.sev_ref_prev == 3


def test_phase_transition_no_op_when_same_severity():
    """on_manager_step_start(same severity_per_amb/ref) is a no-op."""
    lam = LambdaState()
    lam.reset_episode((3,), 3)
    lam.lambda_global[0] = 5.0  # mutate
    lam.on_manager_step_start((3,), 3)  # same severity, should not reset
    assert lam.lambda_global[0] == 5.0, "Same-severity call should not overwrite λ"


def test_phase_transition_ema_saves_old_lambda():
    """On transition (1,)->(3,), λ_warm[(1,)] gets EMA update from current λ_global."""
    lam = LambdaState(beta_ema=0.5)  # large β for visible effect
    lam.reset_episode((1,), 1)
    initial_warm_1 = build_lambda_warm_vector((1,), 1)
    # Simulate λ_global drifted during severity 1
    new_lambda = np.array([2.0, 3.0, 0.5, 1.5, 1.0])
    lam.lambda_global = new_lambda.copy()
    # Transition (1,) -> (3,)
    lam.on_manager_step_start((3,), 3)
    # Expected: λ_warm[(1,)] = 0.5 * old + 0.5 * new
    expected_warm_1 = 0.5 * initial_warm_1 + 0.5 * new_lambda
    np.testing.assert_allclose(lam.lambda_warm[(1,)], expected_warm_1, rtol=1e-9)


def test_phase_transition_resets_window():
    """On severity transition, win_c + win_steps reset (window context changed)."""
    lam = LambdaState()
    lam.reset_episode((1,), 1)
    lam.win_c = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    lam.win_steps = 7
    lam.on_manager_step_start((3,), 3)
    np.testing.assert_array_equal(lam.win_c, np.zeros(5))
    assert lam.win_steps == 0


def test_episode_end_ema_saves_last_active_severity():
    """Episode boundary must persist the last active severity before reset_episode reloads."""
    lam = LambdaState(beta_ema=0.5)
    lam.reset_episode((4,), 4)
    initial_warm_4 = build_lambda_warm_vector((4,), 4)
    lam.lambda_global = np.array([2.0, 2.5, 0.2, 1.8, 1.9])
    lam.on_episode_end((4,), 4)
    expected_warm_4 = 0.5 * initial_warm_4 + 0.5 * np.array([2.0, 2.5, 0.2, 1.8, 1.9])
    np.testing.assert_allclose(lam.lambda_warm[(4,)], expected_warm_4)


# ----------------------------------------------------------------------------
# Interval-window accumulation — Option b (Phase 2.3.5)
# ----------------------------------------------------------------------------


def test_accumulate_adds_deviation():
    """accumulate(c, d) adds (c - d) / dual_scales to win_c and increments win_steps."""
    lam = LambdaState()
    lam.reset_episode((1,), 1)
    c_vec = np.array([2.0, 0.1, 5.0, 0.5, 0.01])
    d_phi = np.array([1.0, 0.05, 10.0, 1.0, 0.1])
    lam.accumulate(c_vec, d_phi)
    expected_deviation = (c_vec - d_phi) / lam.dual_scales
    np.testing.assert_allclose(lam.win_c, expected_deviation)
    assert lam.win_steps == 1
    # Add again
    lam.accumulate(c_vec, d_phi)
    np.testing.assert_allclose(lam.win_c, 2 * expected_deviation)
    assert lam.win_steps == 2


def test_accumulate_validates_shape():
    """accumulate() raises on mismatched shapes."""
    lam = LambdaState()
    with pytest.raises(ValueError):
        lam.accumulate(np.zeros(3), np.zeros(5))
    with pytest.raises(ValueError):
        lam.accumulate(np.zeros(5), np.zeros(4))


# ----------------------------------------------------------------------------
# Dual ascent — Phase 2.3.3 sign convention + Fix Error 2 reset
# ----------------------------------------------------------------------------


def test_dual_ascent_sign_positive_deviation_increases_lambda():
    """c_j > d_j (constraint violated) -> g_hat > 0 -> λ_j increases (penalty stronger)."""
    lam = LambdaState()
    lam.reset_episode((3,), 3)
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
    """c_j < d_j (constraint satisfied) -> g_hat < 0 -> λ_j decreases (relax)."""
    lam = LambdaState()
    lam.reset_episode((3,), 3)
    initial_lambda = lam.lambda_global.copy()
    # c - d < 0 for all j
    c_test = np.array([0.5, 1e-6, 0.0, 0.05, 1e-4])
    d_test = np.array([1.0, 1e-5, 0.0, 0.1, 1e-3])
    for _ in range(10):
        lam.accumulate(c_test, d_test)
    lam.on_manager_step_end()
    # λ should decrease (or stay at 0 / unchanged for the zero-deviation index)
    for j in [0, 1, 3, 4]:
        assert lam.lambda_global[j] <= initial_lambda[j], (
            f"λ_{j} should decrease when c-d<0. before: {initial_lambda[j]}, after: {lam.lambda_global[j]}"
        )


def test_dual_ascent_lambda_nonneg_projection():
    """λ projection: max(0, λ + α·g_hat) — never negative."""
    lam = LambdaState()
    lam.reset_episode((3,), 3)
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
    """Reviewer M4 (internal review, W06, 2026-05-27): λ projection clips at LAMBDA_MAX.

    Bounded projection Π_Λ(λ) = clip(λ, 0, LAMBDA_MAX) prevents dual blow-up
    under sustained constraint violations. Soft safety net (empirical λ ≤ 2.5).
    """
    lam = LambdaState()
    lam.reset_episode((3,), 3)
    # Force massive positive subgradient -> dual ascent wants to blow up
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
    lam.reset_episode((3,), 3)
    lam.accumulate(np.ones(5), np.zeros(5))
    lam.accumulate(np.ones(5), np.zeros(5))
    assert lam.win_steps == 2
    lam.on_manager_step_end()
    np.testing.assert_array_equal(lam.win_c, np.zeros(5))
    assert lam.win_steps == 0


def test_dual_ascent_pushes_lambda_local():
    """After dual update, λ_local = λ_global.copy() (push to Worker)."""
    lam = LambdaState()
    lam.reset_episode((3,), 3)
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
    lam.reset_episode((3,), 3)
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
    lam.reset_episode((3,), 3)
    c = np.ones(5)
    d = np.ones(5)  # c - d = 0 -> penalty = 0
    r_aug = lam.augmented_reward(reward=-2.5, c_vec=c, d_phi=d)
    assert abs(r_aug - (-2.5)) < 1e-9


def test_augmented_reward_positive_deviation_reduces_reward():
    """When c > d, penalty positive -> r_aug < r."""
    lam = LambdaState()
    lam.reset_episode((5,), 5)
    c = np.array([2.0, 1.0 + 1e-5, 31.0, 1.1, 1.0 + 1e-3])
    d = np.array([1.0, 1e-5, 30.0, 0.1, 1e-3])  # c-d = [1, 1, 1, 1, 1]
    r_aug = lam.augmented_reward(reward=0.0, c_vec=c, d_phi=d)
    expected_penalty = float(np.dot(lam.lambda_local, 1.0 / lam.dual_scales))
    assert abs(r_aug - (-expected_penalty)) < 1e-6


def test_augmented_reward_slack_constraint_creates_no_bonus():
    """Bonus-masking regression (2026-06-22 audit): a deeply slack C1 (mean
    delay far under threshold) must contribute ZERO, not a reward bonus that
    masks a smaller C2 (tail) violation. Pre-fix (signed deviation), this
    exact shape produced a net reward INCREASE despite a real violation."""
    lam = LambdaState()
    lam.reset_episode((5,), 5)
    # C1 deeply slack (c << d) — would be a huge negative term pre-fix.
    # C2 mildly violated (c > d) — the real safety-relevant signal.
    c = np.array([0.001, 1e-5 + 1e-5, 30.0, 0.1, 1e-3])
    d = np.array([1.0, 1e-5, 30.0, 0.1, 1e-3])
    r_aug = lam.augmented_reward(reward=0.0, c_vec=c, d_phi=d)
    c2_only_penalty = lam.lambda_local[lam.K] * (1e-5 / lam.dual_scales[lam.K])
    assert abs(r_aug - (-c2_only_penalty)) < 1e-9, (
        "Slack C1 must contribute exactly 0 (hinge), not a bonus that "
        "offsets/masks the C2 violation penalty"
    )
    assert r_aug < 0.0, "A violated tail constraint must never net out to a reward bonus"


def test_penalty_breakdown_sums_to_scalar_penalty():
    """Σ_j penalty_breakdown[j] must equal the scalar penalty (r_base − r_aug)."""
    lam = LambdaState()
    lam.reset_episode((5,), 5)
    c = np.array([2.0, 1.0 + 1e-5, 31.0, 1.1, 1.0 + 1e-3])
    d = np.array([1.0, 1e-5, 30.0, 0.1, 1e-3])
    breakdown = lam.penalty_breakdown(c_vec=c, d_phi=d)
    assert breakdown.shape == (lam.n_constraints,)
    scalar_penalty = 0.0 - lam.augmented_reward(reward=0.0, c_vec=c, d_phi=d)
    assert abs(float(breakdown.sum()) - scalar_penalty) < 1e-9


def test_penalty_breakdown_sign_separates_slack_from_violated():
    """Slack constraint (c<d) -> exactly zero (no bonus, hinge); violated (c>d) -> positive."""
    lam = LambdaState()
    lam.reset_episode((5,), 5)  # all warm λ > 0
    # C1 slack (c<d), C2 violated (c>d); rest exactly met.
    c = np.array([0.5, 1.0 + 1e-5, 30.0, 0.1, 1e-3])
    d = np.array([1.0, 1e-5, 30.0, 0.1, 1e-3])
    b = lam.penalty_breakdown(c_vec=c, d_phi=d)
    assert b[0] == 0.0         # C1 slack -> hinge clips to zero (no bonus)
    assert b[lam.K] > 0.0      # C2 violated -> penalty
    assert b[2 * lam.K] == 0.0  # C4 exactly met -> zero


def test_penalty_breakdown_does_not_mutate_state():
    """penalty_breakdown is pure: λ_local/λ_global/win_c unchanged."""
    lam = LambdaState()
    lam.reset_episode((4,), 4)
    before_g = lam.lambda_global.copy()
    before_l = lam.lambda_local.copy()
    before_win = lam.win_c.copy()
    lam.penalty_breakdown(c_vec=np.ones(5) * 2, d_phi=np.ones(5))
    np.testing.assert_array_equal(lam.lambda_global, before_g)
    np.testing.assert_array_equal(lam.lambda_local, before_l)
    np.testing.assert_array_equal(lam.win_c, before_win)


# ----------------------------------------------------------------------------
# Integration scenario — multiple Manager steps with severity transitions (K=1)
# ----------------------------------------------------------------------------


def test_integration_multi_manager_step_with_transition():
    """End-to-end: 3 Manager steps at severity 3, then transition to 1, then 2 more steps."""
    lam = LambdaState()
    lam.reset_episode((3,), 3)

    # 3 Manager steps at severity 3 with mild URLLC violation
    for _ in range(3):
        for _ in range(10):  # 10 worker steps per Manager
            lam.accumulate(
                c_vec=np.array([1.5e-3, 5e-5, 0.0, 0.05, 1e-4]),
                d_phi=np.array([1e-3, 1e-5, 0.0, 0.1, 1e-3]),
            )
        lam.on_manager_step_end()

    # Severity transition (3,) -> (1,)
    lam.on_manager_step_start((1,), 1)
    # λ_global should now be λ_warm[(1,)] — (1,) was never EMA-saved before,
    # so _warm_for() falls back to the raw build_lambda_warm_vector((1,), 1).
    np.testing.assert_allclose(lam.lambda_global, build_lambda_warm_vector((1,), 1))
    # λ_warm[(3,)] should have been EMA-updated (no longer raw initial)
    initial_warm_3 = build_lambda_warm_vector((3,), 3)
    assert np.max(np.abs(lam.lambda_warm[(3,)] - initial_warm_3)) > 1e-8, (
        "λ_warm[(3,)] should have been EMA-updated from λ_global at severity 3"
    )

    # 2 Manager steps at severity 1 with c < d (constraints satisfied)
    for _ in range(2):
        for _ in range(10):
            lam.accumulate(
                c_vec=np.array([1e-3, 0.0, -5.0, 0.5, 0.0]),
                d_phi=np.array([20e-3, 1e-3, 0.0, 1.0, 1e-2]),
            )
        lam.on_manager_step_end()

    # No exception, all λ non-negative
    assert np.all(lam.lambda_global >= 0.0)
    assert lam.sev_prev == (1,)


# ----------------------------------------------------------------------------
# K=3 — per-ambulance severity (4K+1)-dim
# ----------------------------------------------------------------------------


def test_k3_construction_and_n_constraints():
    lam = LambdaState(K=3)
    assert lam.n_constraints == 4 * 3 + 1 == 13
    assert lam.lambda_global.shape == (13,)
    assert lam.lambda_local.shape == (13,)
    assert lam.dual_scales.shape == (13,)


def test_k3_reset_episode_matches_warm_vector():
    lam = LambdaState(K=3)
    severity_per_amb = (1, 3, 5)
    severity_ref = 5
    lam.reset_episode(severity_per_amb, severity_ref)
    expected = build_lambda_warm_vector(severity_per_amb, severity_ref)
    np.testing.assert_allclose(lam.lambda_global, expected)
    np.testing.assert_allclose(lam.lambda_local, expected)


def test_k3_accumulate_and_dual_ascent_shapes():
    lam = LambdaState(K=3)
    lam.reset_episode((1, 1, 1), 1)
    c_vec = np.full(13, 2.0)
    d_phi = np.full(13, 1.0)  # violated everywhere
    initial = lam.lambda_global.copy()
    for _ in range(10):
        lam.accumulate(c_vec, d_phi)
    lam.on_manager_step_end()
    assert lam.lambda_global.shape == (13,)
    assert np.all(lam.lambda_global >= initial)


def test_k3_on_manager_step_start_no_op_same_severity():
    lam = LambdaState(K=3)
    lam.reset_episode((1, 2, 3), 3)
    lam.lambda_global[0] = 5.0
    lam.on_manager_step_start((1, 2, 3), 3)
    assert lam.lambda_global[0] == 5.0
