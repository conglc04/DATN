"""TD3 agent + TD3 baseline smoke tests."""

from __future__ import annotations

import numpy as np
import pytest


class TestTD3Agent:
    def test_select_action_in_range(self):
        from agents.td3_agent import TD3Agent
        low = np.array([-1, -1, 0, 0, 0, 0], dtype=np.float32)
        high = np.array([1, 1, 1, 1, 1, 1], dtype=np.float32)
        agent = TD3Agent(state_dim=10, action_dim=6, action_low=low, action_high=high,
                         warmup_steps=0, seed=0)
        obs = np.zeros(10, dtype=np.float32)
        for _ in range(20):
            a = agent.select_action(obs, deterministic=True)
            assert a.shape == (6,)
            assert np.all(a >= low - 1e-6)
            assert np.all(a <= high + 1e-6)

    def test_warmup_uses_random_action(self):
        from agents.td3_agent import TD3Agent
        low = np.array([-1, -1, 0, 0, 0, 0], dtype=np.float32)
        high = np.array([1, 1, 1, 1, 1, 1], dtype=np.float32)
        agent = TD3Agent(state_dim=4, action_dim=6, action_low=low, action_high=high,
                         warmup_steps=10, seed=42)
        obs = np.zeros(4, dtype=np.float32)
        # During warmup, actions should be random uniform → high variance
        actions = np.array([agent.select_action(obs, deterministic=False) for _ in range(5)])
        # Increment step_count to simulate warmup
        for _ in range(5):
            agent.step_count += 1
            actions = np.vstack([actions, agent.select_action(obs, deterministic=False)])
        # At least some variation across calls
        assert float(actions.std()) > 0.0

    def test_replay_buffer_add_sample(self):
        from agents.td3_agent import ReplayBuffer
        buf = ReplayBuffer(capacity=100, state_dim=4, action_dim=6)
        for i in range(50):
            buf.add(np.ones(4) * i, np.zeros(6), 0.5, np.ones(4) * (i + 1), False)
        assert buf.size == 50
        s, a, r, ns, d = buf.sample(8, rng=np.random.default_rng(0))
        assert s.shape == (8, 4)
        assert a.shape == (8, 6)
        assert r.shape == (8,)
        assert ns.shape == (8, 4)
        assert d.shape == (8,)

    def test_update_returns_critic_loss(self):
        from agents.td3_agent import TD3Agent
        low = np.array([-1, -1, 0, 0, 0, 0], dtype=np.float32)
        high = np.array([1, 1, 1, 1, 1, 1], dtype=np.float32)
        agent = TD3Agent(state_dim=8, action_dim=6, action_low=low, action_high=high,
                         warmup_steps=0, batch_size=32, seed=0)
        # Fill buffer with random transitions
        rng = np.random.default_rng(0)
        for _ in range(40):
            obs = rng.normal(size=8).astype(np.float32)
            action = rng.uniform(low, high).astype(np.float32)
            next_obs = rng.normal(size=8).astype(np.float32)
            agent.store(obs, action, 0.1, next_obs, False)
        # update should now return a critic_loss
        out = agent.update()
        assert "critic_loss" in out
        assert np.isfinite(out["critic_loss"])

    def test_save_load_roundtrip(self, tmp_path):
        from agents.td3_agent import TD3Agent
        low = np.array([-1, -1, 0, 0, 0, 0], dtype=np.float32)
        high = np.array([1, 1, 1, 1, 1, 1], dtype=np.float32)
        agent = TD3Agent(state_dim=4, action_dim=6, action_low=low, action_high=high,
                         warmup_steps=0, seed=42)
        # Make some training noise
        rng = np.random.default_rng(0)
        for _ in range(40):
            obs = rng.normal(size=4).astype(np.float32)
            action = rng.uniform(low, high).astype(np.float32)
            next_obs = rng.normal(size=4).astype(np.float32)
            agent.store(obs, action, 0.1, next_obs, False)
        for _ in range(5):
            agent.update()

        path = tmp_path / "td3.pt"
        agent.save(str(path))

        agent2 = TD3Agent(state_dim=4, action_dim=6, action_low=low, action_high=high,
                          warmup_steps=0, seed=99)
        agent2.load(str(path))
        obs = np.ones(4, dtype=np.float32)
        a1 = agent.select_action(obs, deterministic=True)
        a2 = agent2.select_action(obs, deterministic=True)
        np.testing.assert_allclose(a1, a2, atol=1e-5)


class TestTD3Baseline:
    """W07: TD3 uses 5-dim LambdaState (sibling solver to PPO + SAC)."""

    def test_instantiate(self):
        from solvers.td3 import TD3Baseline
        agent = TD3Baseline(state_dim=40, action_dim=6, seed=0)
        assert agent is not None
        assert agent.lambda_state.n_constraints == 5
        assert hasattr(agent, "store_transition")
        # Old 2-dim API removed
        assert not hasattr(agent, "lagrangian")

    def test_select_action_returns_tuple(self):
        from solvers.td3 import TD3Baseline
        agent = TD3Baseline(state_dim=40, action_dim=6, seed=0)
        obs = np.zeros(40, dtype=np.float32)
        action, log_prob, value = agent.select_action(obs)
        assert action.shape == (6,)
        assert log_prob == 0.0
        assert value == 0.0

    def test_lambda_state_lifecycle(self):
        """on_episode_start → on_manager_step_start → accumulate → on_manager_step_end."""
        from solvers.td3 import TD3Baseline
        agent = TD3Baseline(state_dim=40, action_dim=6, seed=0, alpha_lambda=0.5)
        agent.on_episode_start(3)
        agent.on_manager_step_start(3)
        # Heavy URLLC violation: positive c_vec[0..1] - d_phi[0..1]
        c_vec = np.array([10e-3, 0.5, 0.0, 0.0, 0.0])  # c1 = 10ms; c2 = 50% viol
        d_phi = np.array([1e-3, 1e-5, 0.0, 0.0, 0.0])
        for _ in range(10):
            agent.accumulate_constraint(c_vec, d_phi)
        agent.on_manager_step_end()
        # λ_1, λ_2 should have grown (binding URLLC)
        lam = agent.lambda_state.get_lambda_global()
        assert lam[0] > 0
        assert lam[1] > 0

    def test_mask_phase_in_obs(self):
        from solvers._common import PHASE_OH_START_INDEX, PHASE_OH_LEN
        from solvers.td3 import TD3Baseline
        agent = TD3Baseline(state_dim=40, action_dim=6, seed=0)
        obs = np.arange(40, dtype=np.float32)
        masked = agent.maybe_mask(obs)
        assert np.all(masked[PHASE_OH_START_INDEX:PHASE_OH_START_INDEX + PHASE_OH_LEN] == 0.0)


class TestTD3SmokeTrainOneEpisode:
    """1-episode smoke through smoke_train.train()."""

    def test_runs_without_crash(self):
        from solvers.smoke_train import train
        stats = train(
            baseline_name="td3",
            n_episodes=1,
            seed=0,
            log_dir="logs/_smoke_unittest_td3",
            initial_phase=3,
            print_every=10_000,
            checkpoint_every=0,
        )
        assert "ep_reward" in stats
        assert np.isfinite(stats["ep_reward"])
