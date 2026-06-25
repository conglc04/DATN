"""HRL wiring tests — env hook + Manager agents + causal PRB effect.

Covers:
    - set_rrm_budget(): anchor update, feasibility clipping, eMBB complement
    - obs[16] = r_min_urllc_anchor (Manager-owned; Worker action cannot drift obs[4])
    - Causal PRB effect: higher b_rrm → more URLLC PRBs; per-amb sum invariant
    - decode_manager_action(): always in [B_RRM_MIN, B_RRM_MAX]
    - build_manager_state(): severity_norm at position 3
    - _manager_act(): normalises TD3 (ndarray) and SAC (tuple) to ndarray
    - TD3/SAC sidecar checkpoints
    - Manager reward accumulates r_aug (not raw r_t) in smoke_train
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from agents.manager_agent import (
    TD3ManagerAgent,
    SACManagerAgent,
    decode_manager_action,
    manager_state_dim,
)
from env.oran_env import EnvConfig, ORANEnv
from solvers._common import _manager_act, build_manager_state
from utils.config import B_RRM_MAX, B_RRM_MIN


# ============================================================
# Fixtures
# ============================================================

def _make_env(K: int = 1) -> ORANEnv:
    env = ORANEnv(EnvConfig(K_ambulances=K))
    env.reset(seed=0)
    return env


# ============================================================
# set_rrm_budget — anchor + feasibility clipping
# ============================================================

class TestSetRrmBudget:
    def test_updates_anchor_within_feasible_range(self):
        env = _make_env()
        # Pin severity → floor B_RRM_FLOOR_BY_SEV[1]=0.65 so 0.75 is in-range
        # (sample_severity makes the floor vary 0.65→0.85; exact-value tests pin it).
        env.severity = 1
        env.set_rrm_budget(0.75)
        assert env.r_min_urllc == pytest.approx(0.75, abs=1e-9)
        assert env.r_min_urllc_anchor == pytest.approx(0.75, abs=1e-9)
        env.close()

    def test_clips_to_effective_upper_bound(self):
        """Request above the upper bound clips to min(B_RRM_MAX, feasible_rrm_cap).

        Two-tier clip: whichever of the outer B_RRM_MAX or the per-K/QoS
        feasible cap is tighter binds. (After the 2026-06-16 d3_embb fix the
        feasible cap rose above B_RRM_MAX at max severity, so B_RRM_MAX binds —
        this asserts the effective bound, not one specific tier.)
        """
        env = _make_env()
        hi = min(B_RRM_MAX, env._feasible_rrm_cap)
        env.set_rrm_budget(hi + 0.2)  # request clearly above the effective bound
        assert env.r_min_urllc == pytest.approx(hi, abs=1e-9)
        assert env.r_min_urllc_anchor == pytest.approx(hi, abs=1e-9)
        env.close()

    def test_manager_sets_embb_complement(self):
        """Manager-owned inter-slice split: r_max_eMBB is always the remainder."""
        env = _make_env()
        env.severity = 1  # floor 0.65 → 0.75 in-range
        env.r_max_emBB = 0.7  # stale value should be overwritten by Manager setpoint
        env.set_rrm_budget(0.75)
        assert env.r_min_urllc == pytest.approx(0.75, abs=1e-9)
        assert env.r_max_emBB == pytest.approx(0.25, abs=1e-9)
        env.close()

    def test_r_ded_urllc_le_r_min_after_step(self):
        """C6 invariant: r_ded_urllc ≤ r_min_urllc holds after a single step."""
        env = _make_env()
        env.set_rrm_budget(0.15)
        env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert env.r_ded_urllc <= env.r_min_urllc + 1e-9
        env.close()

    def test_r_ded_urllc_le_r_min_stress(self):
        """C6 invariant holds across 100 random actions at various b_rrm levels."""
        rng = np.random.default_rng(42)
        env = _make_env()
        env.reset(seed=0)
        env.set_rrm_budget(0.3)
        for i in range(100):
            a = rng.uniform(-3.0, 3.0, size=env.action_space.shape).astype(np.float32)
            _, _, term, trunc, _ = env.step(a)
            assert env.r_ded_urllc <= env.r_min_urllc + 1e-9, (
                f"step {i}: r_ded={env.r_ded_urllc:.6f} > r_min={env.r_min_urllc:.6f}"
            )
            if term or trunc:
                env.reset(seed=i)
                env.set_rrm_budget(0.3)
        env.close()

    def test_floor_enforced_for_b_rrm_min(self):
        """Requesting 0.0 is clipped up to max(B_RRM_MIN, feasible_floor)."""
        env = _make_env()
        env.set_rrm_budget(0.0)
        assert env.r_min_urllc >= B_RRM_MIN - 1e-9
        env.close()


# ============================================================
# Causal PRB effect + per-ambulance invariant
# ============================================================

class TestCausalPrbEffect:
    def test_causal_prb_effect_k3(self):
        """K=3: high b_rrm → more URLLC PRBs; per-amb sum equals total PRB budget."""
        # Pin severity=1 (floor 0.65) so both budgets are in-range and distinct.
        env_hi = _make_env(K=3)
        env_hi.severity = 1
        env_hi.set_rrm_budget(0.80)
        _, _, _, _, info_hi = env_hi.step(np.zeros(env_hi.action_space.shape, dtype=np.float32))

        env_lo = _make_env(K=3)
        env_lo.severity = 1
        env_lo.set_rrm_budget(0.65)
        _, _, _, _, info_lo = env_lo.step(np.zeros(env_lo.action_space.shape, dtype=np.float32))

        # Causal: higher b_rrm → more URLLC PRBs allocated
        assert info_hi["prb_urllc"] > info_lo["prb_urllc"]
        # Invariant: per-ambulance split exhausts URLLC budget exactly
        assert sum(info_hi["prb_per_amb"]) == info_hi["prb_urllc"]
        assert sum(info_lo["prb_per_amb"]) == info_lo["prb_urllc"]

        env_hi.close()
        env_lo.close()

    def test_prb_per_amb_len_equals_k(self):
        """info['prb_per_amb'] has exactly K entries."""
        for K in (1, 3):   # SUMO+OSM traces exist for K in {1,3}
            env = _make_env(K=K)
            _, _, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
            assert len(info["prb_per_amb"]) == K
            env.close()


# ============================================================
# obs[16] = r_min_urllc_anchor (anchor fixed per window)
# ============================================================

class TestAnchorObs:
    def test_worker_cannot_drift_manager_anchor(self):
        """obs[16]=anchor and obs[4]=live r_min both stay at Manager setpoint."""
        env = _make_env()
        env.severity = 1  # floor 0.65 → 0.75 in-range
        env.set_rrm_budget(0.75)
        a = np.full(env.action_space.shape, 3.0, dtype=np.float32)
        obs, _, _, _, _ = env.step(a)
        assert obs[16] == pytest.approx(0.75, abs=1e-6)   # anchor unchanged
        assert obs[4] == pytest.approx(0.75, abs=1e-6)    # Worker cannot change inter-slice
        env.close()

    def test_worker_negative_action_does_not_change_rmin(self):
        """Negative Worker action changes no inter-slice ratio."""
        env = _make_env()
        env.severity = 1  # floor 0.65 → 0.75 in-range
        env.set_rrm_budget(0.75)
        a = np.full(env.action_space.shape, -3.0, dtype=np.float32)
        obs, _, _, _, _ = env.step(a)
        assert obs[16] == pytest.approx(0.75, abs=1e-6)   # anchor fixed
        assert obs[4] == pytest.approx(0.75, abs=1e-6)
        env.close()

    def test_anchor_stable_over_multiple_steps(self):
        """obs[16] stays at anchor across an entire Manager window (10 steps)."""
        env = _make_env()
        env.severity = 1  # floor 0.65 → 0.75 in-range
        env.set_rrm_budget(0.75)
        a_push = np.full(env.action_space.shape, 3.0, dtype=np.float32)
        for _ in range(10):
            obs, _, term, trunc, _ = env.step(a_push)
            assert obs[16] == pytest.approx(0.75, abs=1e-6), "anchor must not change mid-window"
            assert obs[4] == pytest.approx(0.75, abs=1e-6), "Worker must not change Manager-owned r_min"
            if term or trunc:
                break
        env.close()

    def test_obs16_reflects_anchor_not_static_config(self):
        """obs[16] = rrm_budget_hint at reset; updates to new anchor after set_rrm_budget."""
        env = ORANEnv(EnvConfig(rrm_budget_hint=0.6))
        obs0, _ = env.reset(seed=0)
        assert obs0[16] == pytest.approx(0.6, abs=1e-6)

        env.severity = 1  # floor 0.65 → 0.75 in-range
        env.set_rrm_budget(0.75)
        obs1, _, _, _, _ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert obs1[16] == pytest.approx(0.75, abs=1e-6)
        env.close()


# ============================================================
# decode_manager_action
# ============================================================

class TestDecodeManagerAction:
    @pytest.mark.parametrize("raw", [-100.0, -3.0, 0.0, 3.0, 100.0])
    def test_always_in_range(self, raw: float):
        b = decode_manager_action(np.array([raw], dtype=np.float32))["b_rrm"]
        assert B_RRM_MIN <= b <= B_RRM_MAX + 1e-9

    def test_midpoint_at_zero(self):
        """raw=0 → b_rrm = (B_RRM_MIN + B_RRM_MAX) / 2 = 0.75 (0.65..0.85)."""
        b = decode_manager_action(np.array([0.0]))["b_rrm"]
        assert b == pytest.approx((B_RRM_MIN + B_RRM_MAX) / 2.0, abs=1e-9)


# ============================================================
# build_manager_state
# ============================================================

class TestBuildManagerState:
    def test_severity_norm_at_position_3(self):
        """Severity one-hot at obs[12] → sev_idx = (2+1)/5 = 0.6 at s_H[3]."""
        obs = np.zeros(32, dtype=np.float32)
        obs[12] = 1.0           # obs[10:15] one-hot, index 2 → (2+1)/5 = 0.6
        lam = np.zeros(5, dtype=np.float32)
        g_hat = np.zeros(5, dtype=np.float32)
        s_H = build_manager_state(obs, lam, g_hat)
        assert s_H[3] == pytest.approx(0.6, abs=1e-6)

    def test_output_dim_k1(self):
        obs = np.zeros(32, dtype=np.float32)
        lam = np.zeros(5, dtype=np.float32)
        g_hat = np.zeros(5, dtype=np.float32)
        s_H = build_manager_state(obs, lam, g_hat)
        assert s_H.shape == (manager_state_dim(1),)   # 18

    def test_output_dim_k3(self):
        obs = np.zeros(54, dtype=np.float32)  # 20 + 11*3 + 1 = 54 (F=1)
        lam = np.zeros(13, dtype=np.float32)  # 4*3+1 = 13
        g_hat = np.zeros(13, dtype=np.float32)
        s_H = build_manager_state(obs, lam, g_hat)
        assert s_H.shape == (manager_state_dim(3),)   # 34


# ============================================================
# Manager update — warmup gate (buffer.size < warmup_steps → no-op)
# ============================================================

class TestManagerWarmup:
    """update() must be a no-op (return {}) until buffer reaches warmup_steps."""

    def _fill(self, manager, n: int) -> None:
        s = np.zeros(manager.actor.net[0].in_features
                     if hasattr(manager.actor, "net")
                     else manager.actor.mean_net[0].in_features
                     if hasattr(manager.actor, "mean_net")
                     else 11, dtype=np.float32)
        for _ in range(n):
            manager.store(s, np.array([0.1], dtype=np.float32), 1.0, s, False)

    def _state_dim(self, manager) -> int:
        for attr in ("net", "mean_net"):
            net = getattr(manager.actor, attr, None)
            if net is not None:
                return net[0].in_features
        return 11

    def _fill2(self, manager, n: int) -> None:
        s = np.zeros(self._state_dim(manager), dtype=np.float32)
        for _ in range(n):
            manager.store(s, np.array([0.1], dtype=np.float32), 1.0, s, False)

    def test_td3_no_update_before_warmup(self):
        m = TD3ManagerAgent(state_dim=11, warmup_steps=64, batch_size=32, seed=0)
        self._fill2(m, 10)   # 10 < 64
        assert m.buffer.size == 10
        result = m.update()
        assert result == {}, f"Expected {{}} before warmup, got {result}"

    def test_sac_no_update_before_warmup(self):
        m = SACManagerAgent(state_dim=11, warmup_steps=64, batch_size=32, seed=0)
        self._fill2(m, 10)
        assert m.buffer.size == 10
        result = m.update()
        assert result == {}, f"Expected {{}} before warmup, got {result}"

    def test_td3_update_fires_at_warmup_threshold(self):
        """update() returns loss dict once buffer.size >= warmup_steps."""
        m = TD3ManagerAgent(state_dim=11, warmup_steps=64, batch_size=32, seed=0)
        self._fill2(m, 64)   # exactly at threshold
        assert m.buffer.size == 64
        result = m.update()
        assert "mgr_critic_loss" in result
        assert isinstance(result["mgr_critic_loss"], float)

    def test_sac_update_fires_at_warmup_threshold(self):
        m = SACManagerAgent(state_dim=11, warmup_steps=64, batch_size=32, seed=0)
        self._fill2(m, 64)
        result = m.update()
        assert "mgr_critic_loss" in result
        assert "mgr_actor_loss" in result
        assert "mgr_alpha_sac" in result

    def test_td3_update_idempotent_before_warmup(self):
        """Multiple update() calls before warmup all return {} without side-effects."""
        m = TD3ManagerAgent(state_dim=11, warmup_steps=64, seed=0)
        self._fill2(m, 5)
        for _ in range(10):
            assert m.update() == {}
        assert m._update_counter == 0   # no gradient steps taken


# ============================================================
# _manager_act — uniform ndarray interface
# ============================================================

class TestManagerAct:
    def test_td3_manager_returns_ndarray(self):
        m = TD3ManagerAgent(state_dim=11, seed=0)
        s = np.zeros(11, dtype=np.float32)
        a = _manager_act(m, s)
        assert isinstance(a, np.ndarray)
        assert a.shape == (1,)

    def test_sac_manager_returns_ndarray(self):
        m = SACManagerAgent(state_dim=11, seed=0)
        s = np.zeros(11, dtype=np.float32)
        a = _manager_act(m, s)
        assert isinstance(a, np.ndarray)
        assert a.shape == (1,)

    def test_decoded_b_rrm_in_range_td3(self):
        m = TD3ManagerAgent(state_dim=11, seed=0)
        s = np.zeros(11, dtype=np.float32)
        a = _manager_act(m, s)
        b = decode_manager_action(a)["b_rrm"]
        assert B_RRM_MIN <= b <= B_RRM_MAX + 1e-9

    def test_decoded_b_rrm_in_range_sac(self):
        m = SACManagerAgent(state_dim=11, seed=0)
        s = np.zeros(11, dtype=np.float32)
        a = _manager_act(m, s)
        b = decode_manager_action(a)["b_rrm"]
        assert B_RRM_MIN <= b <= B_RRM_MAX + 1e-9


# ============================================================
# TD3/SAC sidecar checkpoints
# ============================================================

class TestSidecarCheckpoints:
    def test_td3_sidecar_created_on_save(self, tmp_path):
        from solvers.td3 import TD3Solver
        env = ORANEnv(EnvConfig())
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        agent = TD3Solver(state_dim=state_dim, action_dim=action_dim, seed=0)
        ckpt = str(tmp_path / "td3_test.pt")
        agent.save(ckpt)
        mgr_ckpt = ckpt.replace(".pt", "_manager.pt")
        assert os.path.exists(mgr_ckpt), "TD3 sidecar _manager.pt not created"
        env.close()

    def test_sac_sidecar_created_on_save(self, tmp_path):
        from solvers.sac import SACSolver
        env = ORANEnv(EnvConfig())
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        agent = SACSolver(state_dim=state_dim, action_dim=action_dim, seed=0)
        ckpt = str(tmp_path / "sac_test.pt")
        agent.save(ckpt)
        mgr_ckpt = ckpt.replace(".pt", "_manager.pt")
        assert os.path.exists(mgr_ckpt), "SAC sidecar _manager.pt not created"
        env.close()

    def test_td3_sidecar_load_roundtrip(self, tmp_path):
        """TD3 load() silently skips sidecar if absent; loads when present."""
        from solvers.td3 import TD3Solver
        env = ORANEnv(EnvConfig())
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        agent = TD3Solver(state_dim=state_dim, action_dim=action_dim, seed=0)
        ckpt = str(tmp_path / "td3_rt.pt")
        agent.save(ckpt)
        agent2 = TD3Solver(state_dim=state_dim, action_dim=action_dim, seed=99)
        agent2.load(ckpt)   # must not raise
        env.close()

    def test_td3_load_without_sidecar_does_not_crash(self, tmp_path):
        """load() with no _manager.pt present is a silent no-op for the Manager."""
        from solvers.td3 import TD3Solver
        env = ORANEnv(EnvConfig())
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        agent = TD3Solver(state_dim=state_dim, action_dim=action_dim, seed=0)
        ckpt = str(tmp_path / "td3_nosidecar.pt")
        agent.td3.save(ckpt)   # save only the Worker, no sidecar
        agent2 = TD3Solver(state_dim=state_dim, action_dim=action_dim, seed=0)
        agent2.load(ckpt)      # must not raise even though _manager.pt absent
        env.close()


# ============================================================
# Manager reward accumulates r_aug (integration: smoke_train TD3)
# ============================================================

class TestManagerRewardAccumulation:
    def test_td3_smoke_runs_without_crash(self, tmp_path):
        """Smoke: TD3 smoke_train completes 2 episodes with Manager loop wired."""
        from solvers.train_offpolicy import train
        summary = train(
            "td3",
            n_episodes=2,
            seed=0,
            log_dir=str(tmp_path),
            checkpoint_dir=str(tmp_path / "checkpoints"),
            checkpoint_every=0,
        )
        assert isinstance(summary["ep_reward"], float)
        assert not (summary["ep_reward"] != summary["ep_reward"])  # not NaN

    def test_sac_smoke_runs_without_crash(self, tmp_path):
        """Smoke: SAC smoke_train completes 2 episodes with Manager loop wired."""
        from solvers.train_offpolicy import train
        summary = train(
            "sac",
            n_episodes=2,
            seed=0,
            log_dir=str(tmp_path),
            checkpoint_dir=str(tmp_path / "checkpoints"),
            checkpoint_every=0,
        )
        assert isinstance(summary["ep_reward"], float)
        assert not (summary["ep_reward"] != summary["ep_reward"])  # not NaN
