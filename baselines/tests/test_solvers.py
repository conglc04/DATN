"""Week 5 tests — PPO core + 6 solvers (smoke-level).

These tests verify the solvers don't crash and produce valid output, but do
NOT verify convergence. Convergence checks come in Week 6 with full training.
"""

from __future__ import annotations

import numpy as np
import pytest


# ============================================================
# PPO agent core
# ============================================================


class TestPPOAgent:
    def test_select_action_shape(self):
        from agents.ppo_agent import PPOAgent
        agent = PPOAgent(state_dim=10, action_dim=6, seed=0)
        obs = np.random.randn(10).astype(np.float32)
        action, log_prob, value = agent.select_action(obs)
        assert action.shape == (6,)
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_deterministic_mode_returns_mean(self):
        from agents.ppo_agent import PPOAgent
        agent = PPOAgent(state_dim=10, action_dim=6, seed=0)
        obs = np.zeros(10, dtype=np.float32)
        a1, _, _ = agent.select_action(obs, deterministic=True)
        a2, _, _ = agent.select_action(obs, deterministic=True)
        np.testing.assert_allclose(a1, a2)

    def test_save_load_roundtrip(self, tmp_path):
        from agents.ppo_agent import PPOAgent
        agent = PPOAgent(state_dim=10, action_dim=6, seed=42)
        path = tmp_path / "ppo.pt"
        agent.save(str(path))
        # Build a new agent, load, check that deterministic actions agree
        agent2 = PPOAgent(state_dim=10, action_dim=6, seed=999)
        agent2.load(str(path))
        obs = np.ones(10, dtype=np.float32)
        a1, _, _ = agent.select_action(obs, deterministic=True)
        a2, _, _ = agent2.select_action(obs, deterministic=True)
        np.testing.assert_allclose(a1, a2, atol=1e-5)


class TestRolloutBuffer:
    def test_add_and_gae(self):
        from agents.ppo_agent import RolloutBuffer
        buf = RolloutBuffer(capacity=10, state_dim=5, action_dim=3)
        for t in range(10):
            buf.add(
                obs=np.ones(5, dtype=np.float32) * t,
                action=np.zeros(3, dtype=np.float32),
                log_prob=0.0,
                reward=1.0,
                value=0.5,
                done=(t == 9),
            )
        assert buf.full
        buf.compute_gae(last_value=0.0)
        assert buf.returns[:10].sum() != 0.0
        # Advantage should be non-trivial
        assert not np.allclose(buf.advantages[:10], 0.0)

    def test_reset_clears_pointer(self):
        from agents.ppo_agent import RolloutBuffer
        buf = RolloutBuffer(capacity=4, state_dim=3, action_dim=2)
        buf.add(np.zeros(3), np.zeros(2), 0.0, 0.0, 0.0, False)
        buf.reset()
        assert buf.ptr == 0


# ============================================================
# Common infrastructure
# ============================================================


class TestCMDPLagrangian:
    def test_starts_at_zero(self):
        from solvers._common import CMDPLagrangian
        L = CMDPLagrangian(n=5)
        assert np.all(L.lambdas == 0.0)

    def test_step_increases_when_constraint_violated(self):
        from solvers._common import CMDPLagrangian
        L = CMDPLagrangian(n=2, alpha=0.1)
        L.step([0.5, 0.2])  # both positive ⇒ both λ go up
        assert L.lambdas[0] > 0
        assert L.lambdas[1] > 0

    def test_step_stays_nonnegative(self):
        from solvers._common import CMDPLagrangian
        L = CMDPLagrangian(n=2, alpha=1.0)
        L.lambdas[:] = [0.1, 0.0]
        L.step([-5.0, -5.0])           # very negative deviations
        assert (L.lambdas >= 0).all()

    def test_penalty_zero_when_constraints_negative(self):
        from solvers._common import CMDPLagrangian
        L = CMDPLagrangian(n=2)
        L.lambdas[:] = [1.0, 2.0]
        # All constraints negative → no penalty (max(0, c_j)=0)
        assert L.penalty([-1.0, -1.0]) == 0.0

    def test_penalty_sums_when_violated(self):
        from solvers._common import CMDPLagrangian
        L = CMDPLagrangian(n=2)
        L.lambdas[:] = [1.0, 2.0]
        assert L.penalty([0.5, 0.5]) == pytest.approx(1.0 * 0.5 + 2.0 * 0.5)


class TestPhaseMask:
    def test_mask_zeros_out_phase_block(self):
        from solvers._common import mask_severity, SEVERITY_OH_START_INDEX, SEVERITY_OH_LEN
        obs = np.arange(30).astype(np.float32)
        out = mask_severity(obs)
        assert np.all(out[SEVERITY_OH_START_INDEX : SEVERITY_OH_START_INDEX + SEVERITY_OH_LEN] == 0.0)
        # Other entries preserved
        assert out[0] == 0.0
        assert out[SEVERITY_OH_START_INDEX + SEVERITY_OH_LEN] != 0.0

    def test_mask_does_not_mutate_input(self):
        from solvers._common import mask_severity
        obs = np.ones(30, dtype=np.float32)
        _ = mask_severity(obs)
        assert np.all(obs == 1.0)


# ============================================================
# Ablation variants — smoke instantiation
# ============================================================


@pytest.mark.parametrize(
    "module_path,cls_name",
    [
        ("solvers.sac",           "SACSolver"),
        ("solvers.pa_ppo_soft",       "PAPPOSoftBaseline"),
        ("solvers.ppo_cmdp_flat",     "PPOCMDPFlatBaseline"),
    ],
)
class TestBaselineSmokeAPI:
    def test_instantiate(self, module_path, cls_name):
        import importlib
        cls = getattr(importlib.import_module(module_path), cls_name)
        agent = cls(state_dim=28, action_dim=6, seed=0)
        assert agent is not None

    def test_select_action_shape(self, module_path, cls_name):
        import importlib
        cls = getattr(importlib.import_module(module_path), cls_name)
        agent = cls(state_dim=28, action_dim=6, seed=0)
        obs = np.zeros(28, dtype=np.float32)
        action, log_prob, value = agent.select_action(obs)
        assert action.shape == (6,)


# ============================================================
# Phase flag semantics
# ============================================================


class TestPhaseFlagSemantics:
    def test_pa_ppo_soft_keeps_phase(self):
        from solvers.pa_ppo_soft import PAPPOSoftBaseline
        agent = PAPPOSoftBaseline(state_dim=28, action_dim=6, seed=0)
        obs = np.arange(28, dtype=np.float32)
        out = agent.maybe_mask(obs)
        np.testing.assert_array_equal(out, obs)

    def test_sac_has_5dim_lambda(self):
        """SAC (B7): 5-dim λ via LambdaState (Phase 3 sibling solver to
        PPO + TD3, applied AFTER Phase 2 statement complete)."""
        from solvers.sac import SACSolver
        agent = SACSolver(state_dim=40, action_dim=6, seed=0)
        assert agent.lambda_state.n_constraints == 5
        # Old 2-dim API removed
        assert not hasattr(agent, "lagrangian")

    def test_sac_keeps_phase(self):
        # SAC is a phase-AWARE equal sibling: maybe_mask must keep the phase
        # one-hot intact (same as PPO / TD3). Regression guard for fairness.
        from solvers.sac import SACSolver
        agent = SACSolver(state_dim=40, action_dim=6, seed=0)
        assert agent.flags.use_phase is True
        obs = np.arange(40, dtype=np.float32)
        np.testing.assert_array_equal(agent.maybe_mask(obs), obs)



# ============================================================
# Smoke train — 3 main solvers for Gate P3 (1 episode only — keep fast)
# ============================================================


class TestSmokeTrainOneEpisode:
    """Gate P3 prep: each main baseline runs 1 episode without crash."""

    def _run_one(self, baseline_name, tmp_path):
        from solvers.train_offpolicy import train
        stats = train(
            baseline_name=baseline_name,
            n_episodes=1,
            seed=0,
            log_dir=str(tmp_path / "smoke_unittest"),
            checkpoint_dir=str(tmp_path / "smoke_unittest" / "checkpoints"),
            initial_severity=3,
            print_every=10_000,           # silence
        )
        assert "ep_reward" in stats
        assert np.isfinite(stats["ep_reward"])
        assert np.isfinite(stats["mean_e2e_ms"])

    def test_sac(self, tmp_path):
        """SAC — Phase 3 sibling solver (5-dim λ via LambdaState)."""
        self._run_one("sac", tmp_path)

    def test_td3(self, tmp_path):
        """TD3 — Phase 3 sibling solver (5-dim λ via LambdaState)."""
        self._run_one("td3", tmp_path)
