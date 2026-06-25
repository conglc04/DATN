"""W08 — Algorithm 1 training loop tests.

Verifies (per docs/weeks/W08 Gate G3.2):
    - 5-episode smoke training without crash + no NaN
    - LambdaState integration (λ_global non-trivial after dual ascent)
    - Phase transition syncs both λ_global + λ_local (Fix Error 1)
    - PPO buffer boundary = 1 episode (Phase 3.4.4 N8)

Per-ambulance severity_k epic (2026-06-15): obs is now 20+10K+F-dim (31 at
K=1, F=1), λ vectors are (4K+1)-dim in NEW order
[C1_0..C1_{K-1}, C2_0..C2_{K-1}, C4_0..C4_{K-1}, C5_0..C5_{K-1}, C3_shared],
and overlay_lambda_local scatters into non-contiguous per-amb obs slots.
"""

from __future__ import annotations

import numpy as np
import pytest

from agents.lagrangian import LambdaState
from train import (
    MANAGER_STEPS_PER_ROLLOUT,
    WORKER_STEPS_PER_ROLLOUT,
    build_manager_state,
    overlay_lambda_local,
    train_ppo,
)
from utils.config import (
    AMB_LAMBDA_C1_OFFSET,
    AMB_LAMBDA_C2_OFFSET,
    AMB_LAMBDA_C4_OFFSET,
    AMB_LAMBDA_C5_OFFSET,
    LAMBDA_C3_SHARED_OBS_INDEX,
    OBS_FIXED_BLOCK_LEN,
    OBS_PER_AMB_BLOCK_LEN,
    SEVERITY_OH_OBS_INDEX,
)

OBS_DIM_K1 = 32  # 20 + 11*1 + 1 (F=1, incl. active_mask_k)


# ============================================================
# Manager state construction
# ============================================================


def test_build_manager_state_shape():
    obs = np.zeros(OBS_DIM_K1, dtype=np.float32)
    lam = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)  # (4*1+1,)
    g_hat = np.zeros(5, dtype=np.float64)
    s_H = build_manager_state(obs, lam, g_hat)
    assert s_H.shape == (18,)  # 8 + 2*(4K+1) = 8+10 at K=1
    assert s_H.dtype == np.float32


def test_build_manager_state_includes_lambda():
    obs = np.zeros(OBS_DIM_K1, dtype=np.float32)
    lam = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    g_hat = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    s_H = build_manager_state(obs, lam, g_hat)
    # λ_global/LAMBDA_MAX occupies [8:13], g_hat (raw) occupies the trailing [13:18] slots
    from utils.config import LAMBDA_MAX
    np.testing.assert_array_almost_equal(s_H[8:13], lam / LAMBDA_MAX, decimal=5)
    np.testing.assert_array_almost_equal(s_H[13:18], g_hat, decimal=5)


def test_build_manager_state_phase_normalized():
    """severity_ref index encoded as (argmax+1)/5 from one-hot block at SEVERITY_OH_OBS_INDEX."""
    obs = np.zeros(OBS_DIM_K1, dtype=np.float32)
    obs[SEVERITY_OH_OBS_INDEX + 2] = 1.0  # severity_ref = 3 (one-hot index 2)
    lam = np.zeros(5)
    g_hat = np.zeros(5)
    s_H = build_manager_state(obs, lam, g_hat)
    assert s_H[3] == pytest.approx(3 / 5)


# ============================================================
# λ_local overlay (Phase 3.4.4 N4) — non-contiguous per-amb scatter
# ============================================================


def test_overlay_lambda_local_replaces_slots_k1():
    obs = np.arange(OBS_DIM_K1, dtype=np.float32)
    lam = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)  # [C1_0,C2_0,C4_0,C5_0,C3_shared]
    out = overlay_lambda_local(obs, lam, K=1)

    base = OBS_FIXED_BLOCK_LEN  # + 10*0
    assert out[base + AMB_LAMBDA_C1_OFFSET] == pytest.approx(lam[0])
    assert out[base + AMB_LAMBDA_C2_OFFSET] == pytest.approx(lam[1])
    assert out[base + AMB_LAMBDA_C4_OFFSET] == pytest.approx(lam[2])
    assert out[base + AMB_LAMBDA_C5_OFFSET] == pytest.approx(lam[3])
    assert out[LAMBDA_C3_SHARED_OBS_INDEX] == pytest.approx(lam[4])

    # Untouched indices unchanged
    touched = {
        base + AMB_LAMBDA_C1_OFFSET,
        base + AMB_LAMBDA_C2_OFFSET,
        base + AMB_LAMBDA_C4_OFFSET,
        base + AMB_LAMBDA_C5_OFFSET,
        LAMBDA_C3_SHARED_OBS_INDEX,
    }
    for i in range(OBS_DIM_K1):
        if i not in touched:
            assert out[i] == obs[i]


def test_overlay_lambda_local_does_not_mutate_input():
    obs = np.arange(OBS_DIM_K1, dtype=np.float32)
    obs_orig = obs.copy()
    lam = np.ones(5)
    _ = overlay_lambda_local(obs, lam, K=1)
    np.testing.assert_array_equal(obs, obs_orig)


def test_overlay_lambda_local_k3_per_amb_slots():
    K = 3
    obs_dim = OBS_FIXED_BLOCK_LEN + OBS_PER_AMB_BLOCK_LEN * K + 1
    obs = np.arange(obs_dim, dtype=np.float32)
    lam = np.arange(4 * K + 1, dtype=np.float64) / 10.0  # distinct values
    out = overlay_lambda_local(obs, lam, K=K)

    for k in range(K):
        base = OBS_FIXED_BLOCK_LEN + OBS_PER_AMB_BLOCK_LEN * k
        assert out[base + AMB_LAMBDA_C1_OFFSET] == pytest.approx(lam[k])
        assert out[base + AMB_LAMBDA_C2_OFFSET] == pytest.approx(lam[K + k])
        assert out[base + AMB_LAMBDA_C4_OFFSET] == pytest.approx(lam[2 * K + k])
        assert out[base + AMB_LAMBDA_C5_OFFSET] == pytest.approx(lam[3 * K + k])
    assert out[LAMBDA_C3_SHARED_OBS_INDEX] == pytest.approx(lam[4 * K])


# ============================================================
# Phase transition sync (Fix Error 1)
# ============================================================


def test_phase_transition_syncs_both_lambdas():
    """LambdaState.on_manager_step_start must sync BOTH λ_global + λ_local
    from λ_warm[severity] on severity transition."""
    ls = LambdaState()
    ls.reset_episode((1,), 1)
    lam_before_g = ls.get_lambda_global()
    lam_before_l = ls.get_lambda_local()
    np.testing.assert_array_equal(lam_before_g, lam_before_l)

    # Transition severity (1,) -> (3,)
    ls.on_manager_step_start((3,), 3)
    lam_after_g = ls.get_lambda_global()
    lam_after_l = ls.get_lambda_local()
    np.testing.assert_array_equal(lam_after_g, lam_after_l)
    # And both should differ from severity-1 warm (LAMBDA_WARM[1] ≠ LAMBDA_WARM[3])
    assert not np.allclose(lam_after_g, lam_before_g)


# ============================================================
# Episode constants (Phase 1.4 timing)
# ============================================================


def test_episode_step_counts():
    assert MANAGER_STEPS_PER_ROLLOUT == 10
    assert WORKER_STEPS_PER_ROLLOUT == 100  # 10 Manager × W=10 Worker


# ============================================================
# 5-episode smoke (Gate G3.2)
# ============================================================


@pytest.mark.slow
def test_5_episode_smoke_no_nan(tmp_path):
    """Algorithm 1 runs 5 episodes without crash; all metrics finite."""
    out = train_ppo(
        n_episodes=5,
        seed=0,
        log_dir=str(tmp_path / "logs"),
        checkpoint_dir=str(tmp_path / "logs" / "checkpoints"),
        print_every=10_000,
        checkpoint_every=0,
        hard_mission=False,
    )
    assert isinstance(out, dict)
    # Required keys present (K=1: C1_0/C2_0/C4_0/C5_0 + shared C3)
    for k in [
        "ep_reward",
        "mean_e2e_ms",
        "viol_rate",
        "lambda_global_C1_0",
        "lambda_global_C2_0",
        "lambda_global_C4_0",
        "lambda_global_C5_0",
        "lambda_global_C3_shared",
    ]:
        assert k in out, f"Missing key: {k}"
        if isinstance(out[k], (int, float)):
            assert np.isfinite(out[k]), f"NaN/Inf in {k}: {out[k]}"


def test_k1_skips_worker_actor_update(tmp_path):
    """K=1: Worker action is a true no-op (softmax([ℓ_0])=[1.0] always).
    WorkerAgent.update() guards this internally (P1 fix: skip_actor =
    action_dim==1, agents/worker_agent.py:217) — the ACTOR gradient is
    skipped (worker_actor_loss stays exactly 0.0, hardcoded in that branch's
    return dict), but the CRITIC still trains every rollout (worker_n_updates
    stays >0, since update() is still called and still does useful work)."""
    out = train_ppo(
        n_episodes=2,
        seed=0,
        log_dir=str(tmp_path / "logs_k1"),
        checkpoint_dir=str(tmp_path / "logs_k1" / "checkpoints"),
        print_every=10_000,
        checkpoint_every=0,
        hard_mission=False,
        K_ambulances=1,
    )
    assert out.get("worker_actor_skipped_k1") == 1, "K=1 must set worker_actor_skipped_k1"
    assert out.get("worker_actor_loss", -1.0) == 0.0, (
        f"K=1 actor loss must be exactly 0.0 (skipped), got {out.get('worker_actor_loss')}"
    )
    # Critic must still update (update() is still called every rollout).
    assert out.get("worker_n_updates", 0) > 0, "Worker critic update must NOT be skipped at K=1"
    # Manager must still update normally
    assert out.get("manager_n_updates", 0) > 0, "Manager update must NOT be skipped at K=1"


def test_k3_does_not_skip_worker_actor_update(tmp_path):
    """K=3: Worker action is meaningful (softmax over 3 logits) — actor must
    update normally (worker_actor_skipped_k1=0, nonzero actor loss)."""
    out = train_ppo(
        n_episodes=2,
        seed=0,
        log_dir=str(tmp_path / "logs_k3"),
        checkpoint_dir=str(tmp_path / "logs_k3" / "checkpoints"),
        print_every=10_000,
        checkpoint_every=0,
        hard_mission=False,
        K_ambulances=3,
    )
    assert out.get("worker_actor_skipped_k1") == 0, "K=3 must NOT set worker_actor_skipped_k1"
    assert out.get("worker_n_updates", 0) > 0, (
        f"K=3 must NOT skip Worker PPO update, got worker_n_updates={out.get('worker_n_updates')}"
    )
    assert out.get("worker_actor_loss", 0.0) != 0.0, (
        "K=3 actor loss should be nonzero (real gradient updates happened)"
    )


@pytest.mark.slow
def test_5_episode_lambda_global_non_trivial(tmp_path):
    """After 5 episodes with non-zero constraint signals, λ_global is finite and ≥ 0."""
    out = train_ppo(
        n_episodes=5,
        seed=0,
        log_dir=str(tmp_path / "logs"),
        checkpoint_dir=str(tmp_path / "logs" / "checkpoints"),
        print_every=10_000,
        checkpoint_every=0,
    )
    lam_vec = np.array(
        [
            out["lambda_global_C1_0"],
            out["lambda_global_C2_0"],
            out["lambda_global_C4_0"],
            out["lambda_global_C5_0"],
            out["lambda_global_C3_shared"],
        ]
    )
    # All λ_j ≥ 0 (projection invariant)
    assert (lam_vec >= 0).all()


# ============================================================
# PPO buffer boundary = 1 episode (Phase 3.4.4 N8)
# ============================================================


def test_ppo_buffer_resets_each_episode(tmp_path):
    """worker/manager n_samples per episode should equal the per-episode rollout
    size (≤ 100/10) — confirming buffer is flushed at episode end (N8)."""
    out = train_ppo(
        n_episodes=2,
        seed=1,
        log_dir=str(tmp_path / "logs"),
        checkpoint_dir=str(tmp_path / "logs" / "checkpoints"),
        print_every=10_000,
        checkpoint_every=0,
    )
    assert out.get("worker_n_samples", 0) <= WORKER_STEPS_PER_ROLLOUT
    assert out.get("manager_n_samples", 0) <= MANAGER_STEPS_PER_ROLLOUT
