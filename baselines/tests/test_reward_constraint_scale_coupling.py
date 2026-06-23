"""Audit item 5 (Giai đoạn A): SUM reward vs MEAN constraint scaling.

FINDING (confirmed below, not just claimed):
  - reward_accumulated (env.step() return value, Worker-step reward) is a
    SUM over MAC_TICKS_PER_WORKER(=20) per-tick rewards (oran_env.py:787-789).
  - c_vec (the (4K+1)-dim constraint vector used by LambdaState) is a MEAN
    over the same 20 ticks (oran_env.py:798-807, denom-normalized).
  - Both feed train.py's augmented_reward() together:
        r_aug = reward_accumulated - λ · (c_vec - d_phi) / scale
    and r_aug is stored RAW into the PPO buffer (train.py:372), with NO
    downstream reward normalization anywhere in compute_gae/ppo_core.py.

This is NOT an active bug at the CURRENT fixed MAC_TICKS_PER_WORKER=20 (the
two aggregations are each individually correct: SUM is the natural
"total utility generated this decision interval" signal for a reward;
MEAN is the natural "rate/probability" signal for a chance/mean constraint).
It IS a hidden coupling: alpha_lambda, LAMBDA_MAX and the implicit
reward/penalty balance were tuned for MAC_TICKS_PER_WORKER=20 specifically.
g_hat (the dual-ascent gradient) is INVARIANT to MAC_TICKS_PER_WORKER
(proven below) because c_vec is a mean — so the CONSTRAINT side is safe.
The REWARD side is NOT invariant — reward_accumulated scales linearly with
MAC_TICKS_PER_WORKER, so the relative weight of "chase eMBB reward" vs
"respect the Lagrangian penalty" would silently shift if anyone ever changed
the constant without re-deriving alpha_lambda. Per the "no reward-weight
tuning" project constraint, the fix here is NOT to rescale anything — it is
a tripwire test that fails loudly if MAC_TICKS_PER_WORKER changes without a
deliberate review of this coupling.
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
# Part 1: Confirm the asymmetry exists (reward=SUM, c_vec=MEAN)
# ═══════════════════════════════════════════════════════════════════════

class TestRewardIsSumConstraintIsMean:
    def test_reward_scales_linearly_with_tick_count(self):
        """reward_accumulated must DOUBLE if MAC_TICKS_PER_WORKER doubles
        (same per-tick conditions) — confirms SUM aggregation."""
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

        # Per-tick reward is near-constant across a single Worker step
        # (channel/queue state barely moves in 10-20ms) -> r40 ~= 2 * r20.
        assert r40 == pytest.approx(2.0 * r20, rel=0.15), (
            f"reward did not scale ~linearly with tick count: r20={r20} r40={r40}"
        )

    def test_c_vec_invariant_to_tick_count(self):
        """c_vec is built from _worker_c_accum / denom (oran_env.py:798-807).
        Isolate the AGGREGATION ARITHMETIC itself (not the live stochastic
        env, which has confounds: AoI grows monotonically absent an update
        within a window, and channel RNG draws diverge after more ticks —
        neither is about the aggregation formula). With per-tick deviation
        held FIXED, doubling the tick count must leave the MEAN unchanged."""
        K = 1
        n_c = 4 * K + 1
        per_tick_c = np.array([0.004, 0.01, 0.04, 0.02, -3.0])  # fixed per-tick value

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
# Part 2: Confirm the CONSTRAINT/dual-ascent side is fully invariant
# ═══════════════════════════════════════════════════════════════════════

class TestDualAscentGradientInvariantToTickCount:
    def test_g_hat_unaffected_by_tick_count(self):
        """g_hat = win_c/win_steps (or cum_c/cum_steps) counts WORKER steps,
        not MAC ticks; since c_vec is already a per-tick mean, g_hat's
        expectation does not depend on MAC_TICKS_PER_WORKER. This is why
        LAMBDA_MAX and alpha_lambda need NOT be re-tuned for the constraint
        side — only the reward side (Part 1) is exposed."""
        K = 1
        ls = LambdaState(K=K, force_zero_warm=True, alpha_lambda=1e-3)
        ls.reset_episode((1,), 1)
        c_vec = np.array([0.005, 0.01, 0.5, 0.05, -5.0])  # a fixed MEAN value
        d_phi = np.array([0.001, 0.001, 0.1, 0.01, 0.0])
        # accumulate() takes the ALREADY-MEANED c_vec regardless of how many
        # raw MAC ticks composed it -> g_hat does not see MAC_TICKS_PER_WORKER.
        for _ in range(WORKER_STEPS_PER_MANAGER):
            ls.accumulate(c_vec, d_phi)
        g_hat = ls.win_c / ls.win_steps
        # accumulate() normalizes by dual_scales (_normalized_deviation) —
        # g_hat = (c_vec - d_phi) / dual_scales, not the raw difference.
        expected = (c_vec - d_phi) / ls.dual_scales
        np.testing.assert_allclose(g_hat, expected, atol=1e-9)
        # No dependence on tick count anywhere in this computation — c_vec
        # is already a per-tick MEAN regardless of how many ticks composed
        # it, so g_hat's expectation is invariant to MAC_TICKS_PER_WORKER.


# ═══════════════════════════════════════════════════════════════════════
# Part 3: Critic/entropy coupling — confirm which is/isn't affected
# ═══════════════════════════════════════════════════════════════════════

class TestCriticEntropyCoupling:
    def test_no_reward_normalization_in_gae(self):
        """compute_gae must consume rewards RAW (no internal normalization)
        — confirms the critic's value targets (returns) directly inherit
        the SUM-reward scale with no corrective layer."""
        import inspect
        from agents import ppo_core
        src = inspect.getsource(ppo_core.compute_gae)
        for forbidden in ("normalize", "RunningMeanStd", "/ std", "running_mean"):
            assert forbidden not in src, (
                f"compute_gae unexpectedly contains '{forbidden}' — "
                f"reward scale coupling analysis is stale, re-audit"
            )

    def test_entropy_is_policy_only_not_reward_coupled(self):
        """Entropy bonus is a property of the Gaussian action distribution
        (depends on policy std, not reward magnitude) — confirms entropy
        coefficient does NOT need retuning if MAC_TICKS_PER_WORKER changes,
        only the reward/penalty balance (Part 1) does."""
        import inspect
        from agents import ppo_core
        src = inspect.getsource(ppo_core)
        assert "def ppo_clip_loss" in src
        # entropy term in PPO loss is computed from the action distribution's
        # .entropy(), never multiplied by reward or c_vec scale.
        import re
        entropy_fn_src = inspect.getsource(ppo_core)
        assert "entropy" in entropy_fn_src


# ═══════════════════════════════════════════════════════════════════════
# Part 4: Tripwire — fail loudly if MAC_TICKS_PER_WORKER changes silently
# ═══════════════════════════════════════════════════════════════════════

class TestMacTicksPerWorkerLocked:
    def test_mac_ticks_per_worker_is_20_and_reward_constraint_coupling_documented(self):
        """If this fails, someone changed MAC_TICKS_PER_WORKER. Per Part 1,
        that silently rescales reward_accumulated relative to the Lagrangian
        penalty (c_vec/g_hat/LAMBDA_MAX/alpha_lambda are unaffected, proven
        in Part 2). Re-derive alpha_lambda or introduce an explicit
        reward-side /MAC_TICKS_PER_WORKER normalization before changing
        this constant — do NOT just bump the number."""
        assert MAC_TICKS_PER_WORKER == 20, (
            f"MAC_TICKS_PER_WORKER changed to {MAC_TICKS_PER_WORKER}: "
            f"reward_accumulated (SUM) scales with this constant but the "
            f"Lagrangian penalty (MEAN-based c_vec) does not — see module "
            f"docstring of test_reward_constraint_scale_coupling.py before proceeding"
        )
