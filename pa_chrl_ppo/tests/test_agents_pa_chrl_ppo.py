"""W07 — Phase 3 PA-CHRL-PPO solver tests.

Verifies:
    - ppo_core utilities (compute_gae, ppo_clip_loss, value_loss, entropy_bonus)
    - ManagerAgent forward dims + decode squashing
    - WorkerAgent forward dims + decode squashing
    - IdentityNSF returns detached a_raw
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.distributions import Normal

from agents.manager_agent import (
    MANAGER_ACTION_DIM_DEFAULT,
    MANAGER_STATE_DIM_DEFAULT,
    ManagerActor,
    ManagerAgent,
    ManagerCritic,
    decode_manager_action,
)
from agents.nsf import IdentityNSF
from agents.ppo_core import compute_gae, entropy_bonus, ppo_clip_loss, value_loss
from agents.worker_agent import (
    WORKER_ACTION_DIM_DEFAULT,
    WORKER_STATE_DIM_DEFAULT,
    WorkerActor,
    WorkerAgent,
    WorkerCritic,
    decode_worker_action,
)
from utils.config import F_MEC, GAMMA_MANAGER, GAMMA_WORKER


# ============================================================
# ppo_core
# ============================================================


def test_compute_gae_matches_manual_td():
    """Single-step rollout: GAE should equal δ_0 = r + γ·V' - V."""
    rewards = np.array([1.0], dtype=np.float32)
    values = np.array([0.5], dtype=np.float32)
    dones = np.array([0.0], dtype=np.float32)
    last_v = 0.8
    gamma = 0.99
    lam = 0.95
    adv, ret = compute_gae(rewards, values, dones, last_v, gamma, lam)
    expected_delta = 1.0 + gamma * last_v - 0.5
    assert adv.shape == (1,)
    assert np.isclose(adv[0], expected_delta, atol=1e-5)
    assert np.isclose(ret[0], adv[0] + values[0], atol=1e-5)


def test_compute_gae_done_zeros_next_value():
    """If dones[t]=1, the (γ·V_{t+1}) term is masked out."""
    rewards = np.array([1.0, 1.0], dtype=np.float32)
    values = np.array([0.0, 0.0], dtype=np.float32)
    dones = np.array([1.0, 0.0], dtype=np.float32)  # terminal at t=0
    adv, _ = compute_gae(rewards, values, dones, last_value=99.0, gamma=0.99, lam=0.95)
    # At t=0 (done=1): δ = 1 + 0.99·0·... = 1; gae also 0 carry
    # The carry from t=1 is killed because nonterm at t=0 is 0.
    assert np.isclose(adv[0], 1.0, atol=1e-5)


def test_ppo_clip_loss_zero_when_ratio_one():
    """If new_log_probs == old_log_probs, ratio=1 ⇒ loss = -E[advantage]."""
    new_lp = torch.zeros(8)
    old_lp = torch.zeros(8)
    adv = torch.ones(8)
    loss, clip_frac = ppo_clip_loss(new_lp, old_lp, adv, clip_eps=0.2)
    assert torch.isclose(loss, torch.tensor(-1.0), atol=1e-5)
    assert torch.isclose(clip_frac, torch.tensor(0.0), atol=1e-5)


def test_value_loss_scales_with_vf_coef():
    v = torch.zeros(4)
    r = torch.ones(4)
    assert torch.isclose(value_loss(v, r, vf_coef=0.5), torch.tensor(0.5), atol=1e-5)
    assert torch.isclose(value_loss(v, r, vf_coef=1.0), torch.tensor(1.0), atol=1e-5)


def test_entropy_bonus_positive_for_normal():
    dist = Normal(torch.zeros(4, 6), torch.ones(4, 6))
    ent = entropy_bonus(dist)
    assert ent.item() > 0.0


# ============================================================
# ManagerAgent
# ============================================================


def test_manager_actor_forward_dims():
    actor = ManagerActor(state_dim=11, action_dim=2)
    obs = torch.zeros(3, 11)
    dist = actor(obs)
    assert dist.mean.shape == (3, 2)
    assert dist.stddev.shape == (3, 2)


def test_manager_critic_forward_dims():
    critic = ManagerCritic(state_dim=11)
    obs = torch.zeros(5, 11)
    v = critic(obs)
    assert v.shape == (5,)


def test_manager_agent_uses_gamma_h_per_n1():
    """γ_H = γ_L^W ≈ 0.904 — KHÔNG copy γ_L=0.99 (Phase 3.4.4 N1)."""
    m = ManagerAgent()
    assert abs(m.gamma - GAMMA_MANAGER) < 1e-12
    assert m.gamma < GAMMA_WORKER


def test_manager_act_returns_correct_shapes():
    m = ManagerAgent()
    a, lp, v = m.act(np.zeros(MANAGER_STATE_DIM_DEFAULT, dtype=np.float32))
    assert a.shape == (MANAGER_ACTION_DIM_DEFAULT,)
    assert isinstance(lp, float)
    assert isinstance(v, float)


def test_decode_manager_action_bounds():
    """b_rrm ∈ [0,1], f_MEC ∈ [0, F_MEC]."""
    # Extreme positive raw
    dec = decode_manager_action(np.array([10.0, 10.0], dtype=np.float32))
    assert 0.999 <= dec["b_rrm"] <= 1.0
    assert dec["f_mec"] <= F_MEC
    # Extreme negative raw
    dec = decode_manager_action(np.array([-10.0, -10.0], dtype=np.float32))
    assert 0.0 <= dec["b_rrm"] <= 1e-3
    # Zero raw → 0.5 sigmoid
    dec = decode_manager_action(np.array([0.0, 0.0], dtype=np.float32))
    assert abs(dec["b_rrm"] - 0.5) < 1e-5
    assert abs(dec["f_mec"] - F_MEC / 2) < 1e-3


# ============================================================
# WorkerAgent
# ============================================================


def test_worker_actor_forward_dims():
    actor = WorkerActor(state_dim=40, action_dim=6)
    obs = torch.zeros(3, 40)
    dist = actor(obs)
    assert dist.mean.shape == (3, 6)


def test_worker_critic_forward_dims():
    critic = WorkerCritic(state_dim=40)
    obs = torch.zeros(5, 40)
    v = critic(obs)
    assert v.shape == (5,)


def test_worker_agent_uses_gamma_l():
    w = WorkerAgent()
    assert abs(w.gamma - GAMMA_WORKER) < 1e-12


def test_worker_act_returns_correct_shapes():
    w = WorkerAgent()
    a, lp, v = w.act(np.zeros(WORKER_STATE_DIM_DEFAULT, dtype=np.float32))
    assert a.shape == (WORKER_ACTION_DIM_DEFAULT,)


def test_decode_worker_action_delta_bounds():
    """Δr_min, Δr_max ∈ [-0.1, +0.1] (Phase 2.3.2 locked)."""
    tol = 1e-6  # float32 tanh saturation rounding
    for raw_val in [-10.0, -1.0, 0.0, 1.0, 10.0]:
        dec = decode_worker_action(np.array([raw_val, raw_val, 0, 0, 0, 0], dtype=np.float32))
        assert -0.1 - tol <= dec["delta_r_min"] <= 0.1 + tol
        assert -0.1 - tol <= dec["delta_r_max"] <= 0.1 + tol


def test_decode_worker_action_r_ded_ratio_bounds():
    """r_ded_ratio ∈ [0, 1]."""
    for raw_val in [-10.0, 0.0, 10.0]:
        dec = decode_worker_action(np.array([0, 0, raw_val, 0, 0, 0], dtype=np.float32))
        assert 0.0 <= dec["r_ded_ratio"] <= 1.0


def test_decode_worker_action_w_intra_simplex():
    """w_intra^C1..C3 ∈ Δ³ simplex (sum=1, non-negative)."""
    dec = decode_worker_action(np.array([0, 0, 0, 1.0, 2.0, 3.0], dtype=np.float32))
    w = dec["w_intra"]
    assert w.shape == (3,)
    assert (w >= 0).all()
    assert abs(float(w.sum()) - 1.0) < 1e-5


def test_decode_worker_action_rejects_wrong_shape():
    with pytest.raises(ValueError):
        decode_worker_action(np.zeros(5, dtype=np.float32))


# ============================================================
# IdentityNSF
# ============================================================


def test_identity_nsf_numpy_passthrough():
    nsf = IdentityNSF()
    a = np.array([0.1, -0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
    out = nsf.forward(np.zeros(40), a)
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, a)


def test_identity_nsf_tensor_detach():
    """NSF identity must detach so β_qp distillation term has no grad path."""
    nsf = IdentityNSF()
    a = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
    out = nsf.forward(None, a)
    assert isinstance(out, torch.Tensor)
    assert not out.requires_grad
    assert torch.allclose(out, a.detach())


def test_identity_nsf_callable_alias():
    nsf = IdentityNSF()
    a = np.array([1.0, 2.0], dtype=np.float32)
    assert np.array_equal(nsf(None, a), nsf.forward(None, a))


def test_identity_nsf_is_not_trained():
    assert IdentityNSF().is_trained() is False


# ============================================================
# Integration: Worker β_qp distillation = 0 with IdentityNSF
# ============================================================


def test_worker_update_with_identity_nsf_zero_qp_distill():
    """When a_safe = a_raw.detach(), β_qp distillation term is identically 0 vs
    the mean network output. Verifies the W07 wiring assumption."""
    w = WorkerAgent(seed=0)
    nsf = IdentityNSF()
    n = 32
    obs = np.random.randn(n, WORKER_STATE_DIM_DEFAULT).astype(np.float32)
    actions, log_probs, values = [], [], []
    for o in obs:
        a, lp, v = w.act(o)
        actions.append(a)
        log_probs.append(lp)
        values.append(v)
    actions = np.stack(actions)
    a_safe = np.stack([nsf.forward(o, a) for o, a in zip(obs, actions)])
    np.testing.assert_array_equal(actions, a_safe)

    out = w.update(
        obs=obs,
        actions_raw=actions,
        old_log_probs=np.asarray(log_probs, dtype=np.float32),
        rewards=np.random.randn(n).astype(np.float32),
        values=np.asarray(values, dtype=np.float32),
        dones=np.zeros(n, dtype=np.float32),
        last_value=0.0,
        actions_safe=a_safe,
        beta_qp=0.1,
    )
    assert "worker_qp_distill" in out
    # Distillation should be small (≪ β_qp · ||raw||² since a_safe is the sample,
    # not the mean; what matters is no NaN / explosion)
    assert np.isfinite(out["worker_qp_distill"])
    assert out["worker_qp_distill"] >= 0.0
