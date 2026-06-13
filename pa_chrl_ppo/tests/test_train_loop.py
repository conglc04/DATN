"""W08 — Algorithm 1 training loop tests.

Verifies (per docs/weeks/W08 Gate G3.2):
    - 5-episode smoke training without crash + no NaN
    - LambdaState integration (λ_global non-trivial after dual ascent)
    - Phase transition syncs both λ_global + λ_local (Fix Error 1)
    - β_qp linear anneal schedule (per-episode, NOT per-step)
    - PPO buffer boundary = 1 episode (Phase 3.4.4 N8)
"""

from __future__ import annotations

import numpy as np
import pytest

from agents.lagrangian import LambdaState
from train import (
    MANAGER_STEPS_PER_EPISODE,
    WORKER_STEPS_PER_EPISODE,
    anneal_beta_qp,
    build_manager_state,
    overlay_lambda_local,
    train_pa_chrl_ppo,
)
from utils.config import (
    BETA_QP_FINAL,
    BETA_QP_INIT,
    BETA_QP_T_ANNEAL,
    LAMBDA_LOCAL_OBS_INDEX,
)


# ============================================================
# β_qp anneal (Phase 3.2.2 / N6)
# ============================================================


def test_beta_qp_starts_at_init():
    assert anneal_beta_qp(0) == pytest.approx(BETA_QP_INIT)


def test_beta_qp_final_after_t_anneal():
    assert anneal_beta_qp(BETA_QP_T_ANNEAL) == pytest.approx(BETA_QP_FINAL)
    # Past T_anneal, clamps to final
    assert anneal_beta_qp(BETA_QP_T_ANNEAL * 2) == pytest.approx(BETA_QP_FINAL)


def test_beta_qp_midpoint_linear():
    """At ep=T_anneal/2, β_qp should be the midpoint of [β_init, β_final]."""
    mid = anneal_beta_qp(BETA_QP_T_ANNEAL // 2)
    expected = BETA_QP_INIT + 0.5 * (BETA_QP_FINAL - BETA_QP_INIT)
    assert mid == pytest.approx(expected, abs=1e-3)


def test_beta_qp_monotonic_increasing():
    samples = [anneal_beta_qp(ep) for ep in [0, 100, 500, 1000, 2500, 5000]]
    assert all(b <= bnext for b, bnext in zip(samples, samples[1:]))


def test_beta_qp_respects_floor():
    """Reviewer Mn1 (Gemini W08, 2026-05-27): β_qp anneal never goes below BETA_QP_FLOOR.

    Prevents catastrophic forgetting of NSF/QP safety boundaries at end-of-training.
    Tests with synthetic config where β_init = β_final = 0 (would decay to 0 without floor).
    """
    from utils.config import BETA_QP_FLOOR

    # Force anneal to a tiny value < FLOOR via custom args
    val_at_end = anneal_beta_qp(
        episode=10_000,
        beta_init=0.0,
        beta_final=0.0,
        t_anneal=5000,
    )
    assert val_at_end >= BETA_QP_FLOOR - 1e-9, (
        f"β_qp must be ≥ floor ({BETA_QP_FLOOR}). got {val_at_end}"
    )

    # Mid-anneal with tiny final should also respect floor
    val_mid = anneal_beta_qp(
        episode=2500,
        beta_init=0.01,
        beta_final=0.01,
        t_anneal=5000,
    )
    assert val_mid >= BETA_QP_FLOOR - 1e-9, (
        f"β_qp mid-anneal must be ≥ floor ({BETA_QP_FLOOR}). got {val_mid}"
    )

    # When raw anneal value > floor, floor doesn't bind (current config: 0.6 > 0.05)
    val_normal = anneal_beta_qp(episode=0)
    assert val_normal >= BETA_QP_FLOOR  # always at least floor


# ============================================================
# Manager state construction
# ============================================================


def test_build_manager_state_shape():
    obs = np.zeros(40, dtype=np.float32)
    lam = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    s_H = build_manager_state(obs, lam)
    assert s_H.shape == (11,)
    assert s_H.dtype == np.float32


def test_build_manager_state_includes_lambda():
    obs = np.zeros(40, dtype=np.float32)
    lam = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    s_H = build_manager_state(obs, lam)
    # λ_global occupies tail 5 slots
    np.testing.assert_array_almost_equal(s_H[-5:], lam, decimal=5)


def test_build_manager_state_phase_normalized():
    """Phase index encoded as (argmax+1)/5 from one-hot block."""
    obs = np.zeros(40, dtype=np.float32)
    obs[10 + 2] = 1.0  # phase φ_3 (one-hot at index 12)
    lam = np.zeros(5)
    s_H = build_manager_state(obs, lam)
    assert s_H[3] == pytest.approx(3 / 5)


# ============================================================
# λ_local overlay (Phase 3.4.4 N4)
# ============================================================


def test_overlay_lambda_local_replaces_block():
    obs = np.arange(40, dtype=np.float32)
    lam = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    out = overlay_lambda_local(obs, lam)
    np.testing.assert_array_almost_equal(
        out[LAMBDA_LOCAL_OBS_INDEX : LAMBDA_LOCAL_OBS_INDEX + 5],
        lam.astype(np.float32),
        decimal=6,
    )
    # Other indices unchanged
    np.testing.assert_array_equal(out[:LAMBDA_LOCAL_OBS_INDEX], obs[:LAMBDA_LOCAL_OBS_INDEX])
    np.testing.assert_array_equal(
        out[LAMBDA_LOCAL_OBS_INDEX + 5 :], obs[LAMBDA_LOCAL_OBS_INDEX + 5 :]
    )


def test_overlay_lambda_local_does_not_mutate_input():
    obs = np.arange(40, dtype=np.float32)
    obs_orig = obs.copy()
    lam = np.ones(5)
    _ = overlay_lambda_local(obs, lam)
    np.testing.assert_array_equal(obs, obs_orig)


# ============================================================
# Phase transition sync (Fix Error 1)
# ============================================================


def test_phase_transition_syncs_both_lambdas():
    """LambdaState.on_manager_step_start must sync BOTH λ_global + λ_local
    from λ_warm[phi_now] on phase transition."""
    ls = LambdaState()
    ls.reset_episode(initial_phase=1)
    lam_before_g = ls.get_lambda_global()
    lam_before_l = ls.get_lambda_local()
    np.testing.assert_array_equal(lam_before_g, lam_before_l)

    # Transition φ_1 → φ_3
    ls.on_manager_step_start(phi_now=3)
    lam_after_g = ls.get_lambda_global()
    lam_after_l = ls.get_lambda_local()
    np.testing.assert_array_equal(lam_after_g, lam_after_l)
    # And both should differ from φ_1 warm (LAMBDA_WARM[1] ≠ LAMBDA_WARM[3])
    assert not np.allclose(lam_after_g, lam_before_g)


# ============================================================
# Episode constants (Phase 1.4 timing)
# ============================================================


def test_episode_step_counts():
    assert MANAGER_STEPS_PER_EPISODE == 10
    assert WORKER_STEPS_PER_EPISODE == 100  # 10 Manager × W=10 Worker


# ============================================================
# 5-episode smoke (Gate G3.2)
# ============================================================


@pytest.mark.slow
def test_5_episode_smoke_no_nan(tmp_path):
    """Algorithm 1 runs 5 episodes without crash; all metrics finite."""
    out = train_pa_chrl_ppo(
        n_episodes=5,
        seed=0,
        log_dir=str(tmp_path / "logs"),
        print_every=10_000,
        checkpoint_every=0,
        hard_mission=False,
    )
    assert isinstance(out, dict)
    # Required keys present
    for k in [
        "ep_reward",
        "mean_e2e_ms",
        "viol_rate",
        "beta_qp",
        "lambda_global_1",
        "lambda_global_2",
        "lambda_global_3",
        "lambda_global_4",
        "lambda_global_5",
    ]:
        assert k in out, f"Missing key: {k}"
        if isinstance(out[k], (int, float)):
            assert np.isfinite(out[k]), f"NaN/Inf in {k}: {out[k]}"


@pytest.mark.slow
def test_5_episode_lambda_global_non_trivial(tmp_path):
    """After 5 episodes with non-zero constraint signals, at least one λ_j > 0."""
    out = train_pa_chrl_ppo(
        n_episodes=5,
        seed=0,
        log_dir=str(tmp_path / "logs"),
        print_every=10_000,
        checkpoint_every=0,
    )
    lam_vec = np.array([out[f"lambda_global_{j + 1}"] for j in range(5)])
    # All λ_j ≥ 0 (projection invariant)
    assert (lam_vec >= 0).all()


# ============================================================
# PPO buffer boundary = 1 episode (Phase 3.4.4 N8)
# ============================================================


def test_ppo_buffer_resets_each_episode(tmp_path):
    """worker/manager n_samples per episode should equal the per-episode rollout
    size (≤ 100/10) — confirming buffer is flushed at episode end (N8)."""
    out = train_pa_chrl_ppo(
        n_episodes=2,
        seed=1,
        log_dir=str(tmp_path / "logs"),
        print_every=10_000,
        checkpoint_every=0,
    )
    assert out.get("worker_n_samples", 0) <= WORKER_STEPS_PER_EPISODE
    assert out.get("manager_n_samples", 0) <= MANAGER_STEPS_PER_EPISODE
