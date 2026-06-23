"""Commit 2 verification: P1-P6 PPO/training bug fixes.

P1: Worker K=1 actor update skipped (no-op action).
P2: PPO metrics are mean across all minibatches/epochs, not last-only.
P3: Manager PPO epochs capped when n is small (anti-overfitting).
P4: Partial-batch guards already exist (classification: expected behavior).
P5: LambdaState serialized in checkpoint for resume.
P6: CSV provenance includes git commit + config hash.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from agents.lagrangian import LambdaState
from agents.manager_agent import ManagerAgent
from agents.worker_agent import WorkerAgent


# ────────────────────────────────────────────────────────────────────
# P1: Worker K=1 skips actor update
# ────────────────────────────────────────────────────────────────────


class TestP1WorkerK1NoOp:
    def test_k1_actor_weights_unchanged(self):
        """K=1 (action_dim=1): actor weights should NOT change after update."""
        w = WorkerAgent(state_dim=30, action_dim=1, seed=0)
        params_before = {k: v.clone() for k, v in w.actor.state_dict().items()}

        obs = np.random.randn(64, 30).astype(np.float32)
        actions = np.random.randn(64, 1).astype(np.float32)
        log_probs = np.random.randn(64).astype(np.float32)
        rewards = np.random.randn(64).astype(np.float32)
        values = np.random.randn(64).astype(np.float32)
        dones = np.zeros(64, dtype=np.float32)

        result = w.update(obs, actions, log_probs, rewards, values, dones, last_value=0.0)

        for k, v in w.actor.state_dict().items():
            torch.testing.assert_close(v, params_before[k], msg=f"Actor param {k} changed for K=1")

        assert result.get("worker_actor_skipped_k1") == 1

    def test_k1_critic_weights_DO_change(self):
        """K=1: critic should still be trained (value function needed for GAE)."""
        w = WorkerAgent(state_dim=30, action_dim=1, seed=0)
        params_before = {k: v.clone() for k, v in w.critic.state_dict().items()}

        obs = np.random.randn(64, 30).astype(np.float32)
        actions = np.random.randn(64, 1).astype(np.float32)
        log_probs = np.random.randn(64).astype(np.float32)
        rewards = np.random.randn(64).astype(np.float32)
        values = np.random.randn(64).astype(np.float32)
        dones = np.zeros(64, dtype=np.float32)

        w.update(obs, actions, log_probs, rewards, values, dones, last_value=0.0)

        changed = False
        for k, v in w.critic.state_dict().items():
            if not torch.equal(v, params_before[k]):
                changed = True
                break
        assert changed, "Critic weights should change even for K=1"

    def test_k3_actor_weights_change(self):
        """K=3 (action_dim=4): actor should be updated normally."""
        w = WorkerAgent(state_dim=30, action_dim=4, seed=0)
        params_before = {k: v.clone() for k, v in w.actor.state_dict().items()}

        obs = np.random.randn(64, 30).astype(np.float32)
        actions = np.random.randn(64, 4).astype(np.float32)
        log_probs = np.random.randn(64).astype(np.float32)
        rewards = np.random.randn(64).astype(np.float32)
        values = np.random.randn(64).astype(np.float32)
        dones = np.zeros(64, dtype=np.float32)

        result = w.update(obs, actions, log_probs, rewards, values, dones, last_value=0.0)

        changed = False
        for k, v in w.actor.state_dict().items():
            if not torch.equal(v, params_before[k]):
                changed = True
                break
        assert changed, "Actor weights should change for K=3"
        assert result.get("worker_actor_skipped_k1", 0) == 0


# ────────────────────────────────────────────────────────────────────
# P2: PPO metrics aggregated across minibatches
# ────────────────────────────────────────────────────────────────────


class TestP2MetricsAggregation:
    def test_worker_metrics_not_zero_with_data(self):
        """Worker update with enough data should return non-zero metrics."""
        w = WorkerAgent(state_dim=30, action_dim=4, seed=0)
        obs = np.random.randn(128, 30).astype(np.float32)
        actions = np.random.randn(128, 4).astype(np.float32)
        log_probs = np.random.randn(128).astype(np.float32)
        rewards = np.random.randn(128).astype(np.float32)
        values = np.random.randn(128).astype(np.float32)
        dones = np.zeros(128, dtype=np.float32)

        result = w.update(obs, actions, log_probs, rewards, values, dones, last_value=0.0)
        assert result["worker_critic_loss"] != 0.0
        assert result["worker_entropy"] != 0.0

    def test_manager_metrics_not_zero_with_data(self):
        """Manager update should return non-zero metrics."""
        m = ManagerAgent(state_dim=11, action_dim=1, seed=0)
        obs = np.random.randn(20, 11).astype(np.float32)
        actions = np.random.randn(20, 1).astype(np.float32)
        log_probs = np.random.randn(20).astype(np.float32)
        rewards = np.random.randn(20).astype(np.float32)
        values = np.random.randn(20).astype(np.float32)
        dones = np.zeros(20, dtype=np.float32)

        result = m.update(obs, actions, log_probs, rewards, values, dones, last_value=0.0)
        assert result["manager_critic_loss"] != 0.0


# ────────────────────────────────────────────────────────────────────
# P3: Manager epoch cap
# ────────────────────────────────────────────────────────────────────


class TestP3ManagerEpochCap:
    def test_small_n_caps_epochs(self):
        """n=4 transitions → k_epochs_eff = min(10, max(1, 4//4)) = 1."""
        m = ManagerAgent(state_dim=11, action_dim=1, seed=0, k_epochs=10)
        obs = np.random.randn(4, 11).astype(np.float32)
        actions = np.random.randn(4, 1).astype(np.float32)
        log_probs = np.random.randn(4).astype(np.float32)
        rewards = np.random.randn(4).astype(np.float32)
        values = np.random.randn(4).astype(np.float32)
        dones = np.zeros(4, dtype=np.float32)

        result = m.update(obs, actions, log_probs, rewards, values, dones, last_value=0.0)
        assert result["manager_k_epochs_eff"] == 1

    def test_large_n_uses_full_epochs(self):
        """n=100 transitions → k_epochs_eff = min(10, 25) = 10."""
        m = ManagerAgent(state_dim=11, action_dim=1, seed=0, k_epochs=10)
        obs = np.random.randn(100, 11).astype(np.float32)
        actions = np.random.randn(100, 1).astype(np.float32)
        log_probs = np.random.randn(100).astype(np.float32)
        rewards = np.random.randn(100).astype(np.float32)
        values = np.random.randn(100).astype(np.float32)
        dones = np.zeros(100, dtype=np.float32)

        result = m.update(obs, actions, log_probs, rewards, values, dones, last_value=0.0)
        assert result["manager_k_epochs_eff"] == 10

    def test_n8_caps_to_2_epochs(self):
        """n=8 → k_epochs_eff = min(10, max(1, 8//4)) = 2."""
        m = ManagerAgent(state_dim=11, action_dim=1, seed=0, k_epochs=10)
        obs = np.random.randn(8, 11).astype(np.float32)
        actions = np.random.randn(8, 1).astype(np.float32)
        log_probs = np.random.randn(8).astype(np.float32)
        rewards = np.random.randn(8).astype(np.float32)
        values = np.random.randn(8).astype(np.float32)
        dones = np.zeros(8, dtype=np.float32)

        result = m.update(obs, actions, log_probs, rewards, values, dones, last_value=0.0)
        assert result["manager_k_epochs_eff"] == 2


# ────────────────────────────────────────────────────────────────────
# P5: LambdaState checkpoint roundtrip
# ────────────────────────────────────────────────────────────────────


class TestP5LambdaStateCheckpoint:
    def test_roundtrip_serialization(self):
        """state_dict → load_state_dict should preserve all fields."""
        ls = LambdaState(K=3)
        ls.lambda_global = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0])
        ls.lambda_local = ls.lambda_global * 0.5
        ls.win_c = np.ones(13) * 0.1
        ls.win_steps = 42
        ls.sev_prev = (3, 2, 1)
        ls.sev_ref_prev = 3
        ls.lambda_warm[(3, 2, 1)] = np.ones(13) * 0.5
        ls.lambda_warm[(1, 1, 1)] = np.zeros(13)

        sd = ls.state_dict()
        sd_json = json.dumps(sd, default=float)

        ls2 = LambdaState(K=3)
        ls2.load_state_dict(json.loads(sd_json))

        np.testing.assert_array_almost_equal(ls2.lambda_global, ls.lambda_global)
        np.testing.assert_array_almost_equal(ls2.lambda_local, ls.lambda_local)
        np.testing.assert_array_almost_equal(ls2.win_c, ls.win_c)
        assert ls2.win_steps == 42
        assert ls2.sev_prev == (3, 2, 1)
        assert ls2.sev_ref_prev == 3
        assert len(ls2.lambda_warm) == 2
        np.testing.assert_array_almost_equal(ls2.lambda_warm[(3, 2, 1)], np.ones(13) * 0.5)

    def test_empty_state_roundtrip(self):
        """Fresh LambdaState (no warm table) should roundtrip cleanly."""
        ls = LambdaState(K=1)
        sd = ls.state_dict()
        ls2 = LambdaState(K=1)
        ls2.load_state_dict(sd)
        np.testing.assert_array_equal(ls2.lambda_global, np.zeros(5))
        assert ls2.sev_prev is None
        assert len(ls2.lambda_warm) == 0


# ────────────────────────────────────────────────────────────────────
# P6: CSV provenance helpers
# ────────────────────────────────────────────────────────────────────


class TestP6Provenance:
    def test_git_commit_returns_string(self):
        from train import _git_commit_short
        result = _git_commit_short()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_config_hash_deterministic(self):
        from train import _config_hash
        h1 = _config_hash()
        h2 = _config_hash()
        assert h1 == h2
        assert len(h1) == 8
