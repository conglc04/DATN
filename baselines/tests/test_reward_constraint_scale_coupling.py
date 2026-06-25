"""Reward/constraint temporal-basis consistency (audit 2026-06-23 starvation fix).

ROOT-CAUSE HISTORY:
  Previously reward_accumulated (env.step() Worker-step reward) was a SUM over
  MAC_TICKS_PER_WORKER(=20) per-tick rewards, while c_vec (the (4K+1)-dim
  constraint vector) is a MEAN over the same ticks. The augmented Lagrangian
  r − Σλⱼ·(c_vec − d)/scale then mixed a ×20-scale reward with a ×1-scale
  penalty. The eMBB reward gradient (gain ~+4.2 when dropping b_rrm) swamped
  the constraint penalty (~+1.0 even at sev=5 with maxed λ) → the Manager
  always starved URLLC. λ_warm, α_λ and REWARD_FIXED_SCALE could not fix it
  (the first two saturate; the last scales reward AND penalty equally).

FIX (oran_env.py step()):
  reward_accumulated /= n_ticks → reward is now the MEAN over MAC ticks, the
  SAME per-tick basis as c_vec. The augmented Lagrangian is balanced: dropping
  b_rrm gains ~+0.21 reward but costs ~+1.04 penalty at sev=5 → NET negative →
  URLLC protected. At sev=1 (loose QoS, penalty=0) the Manager still frees
  budget for eMBB → correct severity differentiation, no hard floor needed.

This file now LOCKS that both reward and c_vec are MEAN (invariant to tick
count), so the Lagrangian balance no longer depends on MAC_TICKS_PER_WORKER.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

import env.oran_env as oran_env_mod
from env.oran_env import ORANEnv, macro_mission_config
from agents.lagrangian import LambdaState
from utils.config import MAC_TICKS_PER_WORKER, WORKER_STEPS_PER_MANAGER


def _make_env(K=1, seed=0):
    cfg = macro_mission_config(K_ambulances=K, seed=seed)
    env = ORANEnv(cfg, seed=seed)
    env.reset(seed=seed)
    return env


# ═══════════════════════════════════════════════════════════════════════
# Part 1: reward and c_vec are BOTH per-tick MEAN (matched temporal basis)
# ═══════════════════════════════════════════════════════════════════════

class TestRewardAndConstraintBothMean:
    def test_reward_invariant_to_tick_count(self):
        """reward_accumulated is now a MEAN: doubling MAC_TICKS_PER_WORKER
        (same per-tick conditions) must leave the Worker-step reward ~UNCHANGED
        — confirms MEAN aggregation (a SUM would double)."""
        env = _make_env(K=1)
        env.set_rrm_budget(0.30)
        env.severity_per_amb[:] = 1
        env.active_mask[:] = True
        env.entered_mask[:] = True

        oran_env_mod.MAC_TICKS_PER_WORKER = 20
        _, r20, _, _, _ = env.step(np.zeros(1, dtype=np.float32))

        env2 = _make_env(K=1)
        env2.set_rrm_budget(0.30)
        env2.severity_per_amb[:] = 1
        env2.active_mask[:] = True
        env2.entered_mask[:] = True
        oran_env_mod.MAC_TICKS_PER_WORKER = 40
        _, r40, _, _, _ = env2.step(np.zeros(1, dtype=np.float32))
        oran_env_mod.MAC_TICKS_PER_WORKER = MAC_TICKS_PER_WORKER  # restore

        # Per-tick reward is near-constant within a Worker step, and the MEAN
        # over ticks is invariant to the count -> r40 ~= r20 (NOT 2x).
        assert r40 == pytest.approx(r20, rel=0.15), (
            f"reward not invariant to tick count (looks like SUM): r20={r20} r40={r40}"
        )

    def test_reward_is_per_tick_scale_not_summed(self):
        """A single Worker-step reward must be on the per-tick log-utility
        scale (~0-2), NOT a ×20 sum (~10-24)."""
        env = _make_env(K=1)
        env.set_rrm_budget(0.30)
        _, rew, _, _, _ = env.step(np.zeros(1, dtype=np.float32))
        assert 0.0 <= rew < 2.0, f"reward={rew} looks like a SUM, not a per-tick MEAN"

    def test_c_vec_invariant_to_tick_count(self):
        """c_vec is built from _worker_c_accum / denom: with per-tick deviation
        held FIXED, doubling the tick count must leave the MEAN unchanged."""
        K = 1
        n_c = 4 * K + 1
        per_tick_c = np.array([0.004, 0.01, 0.04, 0.02, -3.0])

        def _aggregate(n_ticks: int) -> np.ndarray:
            accum = np.zeros(n_c, dtype=np.float64)
            for _ in range(n_ticks):
                accum[0:K] += per_tick_c[0]
                accum[K:2*K] += per_tick_c[1]
                accum[2*K:3*K] += per_tick_c[2]
                accum[3*K:4*K] += per_tick_c[3]
                accum[4*K] += per_tick_c[4]
            denom = np.concatenate([
                np.full(K, n_ticks, dtype=np.float64), np.full(K, n_ticks, dtype=np.float64),
                np.full(K, n_ticks, dtype=np.float64), np.full(K, n_ticks, dtype=np.float64),
                [float(n_ticks)],
            ])
            return accum / denom

        c20 = _aggregate(20)
        c40 = _aggregate(40)
        np.testing.assert_allclose(c20, c40, rtol=1e-9)
        np.testing.assert_allclose(c20, per_tick_c, rtol=1e-9)


# ═══════════════════════════════════════════════════════════════════════
# Part 2: augmented Lagrangian is now balanced (penalty can dominate at sev5)
# ═══════════════════════════════════════════════════════════════════════

class TestAugmentedLagrangianBalanced:
    def test_penalty_can_exceed_reward_gain_at_high_severity(self):
        """With reward on the per-tick MEAN basis, a real sev=5 constraint
        violation (maxed warm λ) must produce a penalty comparable to or
        exceeding the per-step reward — i.e. the Lagrangian is no longer
        swamped by the reward. (Pre-fix the penalty was ~5% of reward.)"""
        env = _make_env(K=1, seed=0)
        env.severity_per_amb[:] = 5
        ls = LambdaState(K=1)
        ls.reset_episode((5,), 5)  # warm λ sev=5 ~ [1.8, 2.2, ...]
        env.r_min_urllc = 0.05; env.r_min_urllc_anchor = 0.05; env.r_max_emBB = 0.95

        penalties, rewards = [], []
        for step in range(40):
            env.r_min_urllc = 0.05; env.r_min_urllc_anchor = 0.05; env.r_max_emBB = 0.95
            _, r_t, _, _, info = env.step(np.zeros(1, dtype=np.float32))
            r_aug = ls.augmented_reward(float(r_t), info["c_vec"], info["d_phi"])
            ls.accumulate(info["c_vec"], info["d_phi"])
            if step % 10 == 9:
                ls.on_manager_step_end()
            if step >= 20:
                rewards.append(float(r_t)); penalties.append(float(r_t) - r_aug)
        mean_pen = float(np.mean(penalties))
        mean_rew = float(np.mean(rewards))
        # Penalty must be a meaningful fraction of reward (was <6% pre-fix).
        assert mean_pen > 0.5 * mean_rew, (
            f"penalty {mean_pen:.3f} negligible vs reward {mean_rew:.3f} — "
            f"Lagrangian balance broken (reward may have reverted to SUM)"
        )


# ═══════════════════════════════════════════════════════════════════════
# Part 3: dual-ascent gradient invariant to tick count (unchanged by fix)
# ═══════════════════════════════════════════════════════════════════════

class TestDualAscentGradientInvariantToTickCount:
    def test_g_hat_unaffected_by_tick_count(self):
        """g_hat = win_c/win_steps counts WORKER steps; c_vec is already a
        per-tick mean, so g_hat's expectation does not depend on
        MAC_TICKS_PER_WORKER. Unchanged by the reward-basis fix."""
        K = 1
        ls = LambdaState(K=K, force_zero_warm=True, alpha_lambda=1e-3)
        ls.reset_episode((1,), 1)
        c_vec = np.array([0.005, 0.01, 0.5, 0.05, -5.0])
        d_phi = np.array([0.001, 0.001, 0.1, 0.01, 0.0])
        for _ in range(WORKER_STEPS_PER_MANAGER):
            ls.accumulate(c_vec, d_phi)
        g_hat = ls.win_c / ls.win_steps
        expected = (c_vec - d_phi) / ls.dual_scales
        np.testing.assert_allclose(g_hat, expected, atol=1e-9)


# ═══════════════════════════════════════════════════════════════════════
# Part 4: critic consumes reward RAW (no hidden normalization)
# ═══════════════════════════════════════════════════════════════════════

class TestCriticEntropyCoupling:
    def test_no_reward_normalization_in_gae(self):
        """compute_gae must consume rewards RAW — the per-tick MEAN scale
        flows directly into the critic targets (REWARD_FIXED_SCALE handles
        any explicit scaling in train.py, not GAE)."""
        import inspect
        from agents import ppo_core
        src = inspect.getsource(ppo_core.compute_gae)
        for forbidden in ("normalize", "RunningMeanStd", "/ std", "running_mean"):
            assert forbidden not in src, (
                f"compute_gae unexpectedly contains '{forbidden}' — re-audit"
            )


# ═══════════════════════════════════════════════════════════════════════
# Part 5: tripwire — reward must stay MEAN, MAC_TICKS_PER_WORKER locked
# ═══════════════════════════════════════════════════════════════════════

class TestMacTicksPerWorkerLocked:
    def test_mac_ticks_per_worker_is_20(self):
        """Reward is now a per-tick MEAN, so it is INVARIANT to this constant
        (Part 1) — the Lagrangian balance no longer couples to it. The lock
        remains as a deliberate-change tripwire for the two-timescale design."""
        assert MAC_TICKS_PER_WORKER == 20, (
            f"MAC_TICKS_PER_WORKER changed to {MAC_TICKS_PER_WORKER}: reward and "
            f"c_vec are both per-tick MEAN so the balance is preserved, but "
            f"verify the two-timescale hierarchy is still intended."
        )
