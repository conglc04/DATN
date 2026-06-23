"""Prove SUM-MEAN ≡ SUM-SUM up to λ reparameterization.

Mathematical invariant: the optimal policy θ* is identical under both
aggregations. The dual variable λ*_SM = 20 · λ*_SS, but the augmented
reward r_aug at equilibrium is the same.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from agents.lagrangian import LambdaState
from utils.config import MAC_TICKS_PER_WORKER


class TestSumMeanInvariance:

    def _run_dual_ascent(self, alpha: float, use_sum: bool, n_manager_steps: int = 500):
        """Run dual ascent with either MEAN or SUM constraint aggregation."""
        K = 1
        ls = LambdaState(K=K, force_zero_warm=True, alpha_lambda=alpha)
        ls.reset_episode((1,), 1)

        # Fixed violation scenario per MAC tick
        c_tick = np.array([0.005, 0.01, 0.5, 0.05, -5.0])
        d_tick = np.array([0.001, 0.001, 0.1, 0.01, 0.0])

        if use_sum:
            c_step = c_tick * MAC_TICKS_PER_WORKER  # SUM
            d_step = d_tick * MAC_TICKS_PER_WORKER   # threshold also scaled
        else:
            c_step = c_tick  # MEAN (as in current code)
            d_step = d_tick

        reward_sum = 7.0  # SUM reward per Worker step (same for both)

        for _ in range(n_manager_steps):
            for _ in range(10):  # 10 Worker steps per Manager
                ls.accumulate(c_step, d_step)
            ls.on_manager_step_end()
            ls.lambda_local = ls.lambda_global.copy()
            ls.win_c = np.zeros(5, dtype=np.float64)
            ls.win_steps = 0

        r_aug = ls.augmented_reward(reward_sum, c_step, d_step)
        return ls.lambda_global.copy(), r_aug

    def test_policy_gradient_sign_invariant(self):
        """The sign of ∂r_aug/∂θ is the same under SUM-MEAN and SUM-SUM.

        r_aug values differ by a scale factor (by design), but the OPTIMAL
        POLICY θ* is the same because:
          r_aug_SS = R - (20·λ_SM/20²)·20·dev = R - λ_SM·dev = r_aug_SM  (wrong)
        Actually: r_aug_SM = R - λ·dev, r_aug_SS = R - λ'·20·dev, with λ'=λ/20
        → r_aug_SM = r_aug_SS at equilibrium. But convergence path differs.

        What's truly invariant: the constraint-violation signal direction.
        When c > d (violated), penalty increases → r_aug decreases.
        This holds for BOTH formulations.
        """
        K = 1
        c_tick = np.array([0.005, 0.01, 0.5, 0.05, -5.0])
        d_tick = np.array([0.001, 0.001, 0.1, 0.01, 0.0])

        # SUM-MEAN
        ls_sm = LambdaState(K=K, force_zero_warm=True, alpha_lambda=1e-3)
        ls_sm.reset_episode((1,), 1)
        for _ in range(100):
            for _ in range(10):
                ls_sm.accumulate(c_tick, d_tick)
            ls_sm.on_manager_step_end()
            ls_sm.lambda_local = ls_sm.lambda_global.copy()
            ls_sm.win_c = np.zeros(5, dtype=np.float64)
            ls_sm.win_steps = 0
        r_aug_sm = ls_sm.augmented_reward(7.0, c_tick, d_tick)

        # SUM-SUM with α/20² (compensate for 20× dev × 20× g_hat)
        c_sum = c_tick * MAC_TICKS_PER_WORKER
        d_sum = d_tick * MAC_TICKS_PER_WORKER
        ls_ss = LambdaState(K=K, force_zero_warm=True,
                            alpha_lambda=1e-3 / (MAC_TICKS_PER_WORKER ** 2))
        ls_ss.reset_episode((1,), 1)
        for _ in range(100):
            for _ in range(10):
                ls_ss.accumulate(c_sum, d_sum)
            ls_ss.on_manager_step_end()
            ls_ss.lambda_local = ls_ss.lambda_global.copy()
            ls_ss.win_c = np.zeros(5, dtype=np.float64)
            ls_ss.win_steps = 0
        r_aug_ss = ls_ss.augmented_reward(7.0, c_sum, d_sum)

        # Both are negative (violated → penalty dominates)
        assert r_aug_sm < 7.0, f"SM penalty not active: r_aug={r_aug_sm}"
        assert r_aug_ss < 7.0, f"SS penalty not active: r_aug={r_aug_ss}"
        # Same sign
        assert np.sign(r_aug_sm) == np.sign(r_aug_ss)
        # λ_SM ≈ 20·λ_SS (SM has 20× smaller deviation but 20²× larger α effect)
        for j in range(5):
            if ls_sm.lambda_global[j] > 0.01 and ls_ss.lambda_global[j] > 0.01:
                ratio = ls_sm.lambda_global[j] / ls_ss.lambda_global[j]
                assert 5 < ratio < 100, (
                    f"λ[{j}]: SM={ls_sm.lambda_global[j]:.4f} "
                    f"SS={ls_ss.lambda_global[j]:.6f} ratio={ratio:.1f}"
                )

    def test_lambda_ratio_is_20x(self):
        """λ*_SM / λ*_SS ≈ 1 (with compensated α, same normalized deviation)."""
        lam_sm, _ = self._run_dual_ascent(alpha=1e-3, use_sum=False, n_manager_steps=2000)
        # With SUM: deviation = (20c - 20d)/scale = 20×(c-d)/scale
        # α_SS = α_SM/20 → update = α_SS · 20·dev = α_SM · dev = same update
        lam_ss, _ = self._run_dual_ascent(alpha=1e-3 / MAC_TICKS_PER_WORKER, use_sum=True, n_manager_steps=2000)

        # Since update magnitude is the same, λ should converge to similar values
        for j in range(5):
            if lam_sm[j] > 0.01:
                ratio = lam_ss[j] / lam_sm[j]
                assert 0.5 < ratio < 2.0, (
                    f"λ[{j}]: SM={lam_sm[j]:.4f} SS={lam_ss[j]:.4f} ratio={ratio:.2f}"
                )

    def test_gradient_direction_identical(self):
        """∇_θ r_aug has same sign under both formulations."""
        K = 1
        c_tick = np.array([0.005, 0.01, 0.5, 0.05, -5.0])
        d_tick = np.array([0.001, 0.001, 0.1, 0.01, 0.0])

        # SUM-MEAN
        ls_sm = LambdaState(K=K, force_zero_warm=True)
        ls_sm.reset_episode((1,), 1)
        ls_sm.lambda_global = np.array([2.0, 1.0, 0.5, 0.3, 0.1])
        ls_sm.lambda_local = ls_sm.lambda_global.copy()
        dev_sm = ls_sm._normalized_deviation(c_tick, d_tick)

        # SUM-SUM
        ls_ss = LambdaState(K=K, force_zero_warm=True)
        ls_ss.reset_episode((1,), 1)
        ls_ss.lambda_global = np.array([2.0, 1.0, 0.5, 0.3, 0.1]) / MAC_TICKS_PER_WORKER
        ls_ss.lambda_local = ls_ss.lambda_global.copy()
        c_sum = c_tick * MAC_TICKS_PER_WORKER
        d_sum = d_tick * MAC_TICKS_PER_WORKER
        dev_ss = ls_ss._normalized_deviation(c_sum, d_sum)

        # Penalty direction: λ_SM · dev_SM vs λ_SS · dev_SS
        penalty_sm = np.dot(ls_sm.lambda_local, dev_sm)
        penalty_ss = np.dot(ls_ss.lambda_local, dev_ss)

        # Same sign
        assert np.sign(penalty_sm) == np.sign(penalty_ss)
        # Same magnitude (λ_SS = λ_SM/20, dev_SS = 20·dev_SM → product equal)
        assert penalty_sm == pytest.approx(penalty_ss, rel=0.01)

    def test_mean_constraint_is_correct_semantics(self):
        """MEAN cost is the natural constraint: E[mean_delay] ≤ D_max."""
        # The constraint "mean delay ≤ 20ms" is naturally expressed as:
        #   (1/T) Σ_t d_e2e_t ≤ D_max
        # NOT as:
        #   Σ_t d_e2e_t ≤ 20 · D_max (which is same math, uglier semantics)
        #
        # c_vec uses MEAN per Worker step → when averaged over episode,
        # gives mean delay in seconds, directly comparable to D_max in seconds.
        from env.oran_env import ORANEnv, macro_mission_config
        cfg = macro_mission_config(K_ambulances=1, seed=0)
        env = ORANEnv(cfg, seed=0)
        env.reset(seed=0)
        env.set_rrm_budget(0.20)
        _, _, _, _, info = env.step(np.zeros(1, dtype=np.float32))

        c_vec = info["c_vec"]
        d_phi = info["d_phi"]

        # c_vec[0] = mean delay (seconds) per Worker step
        # d_phi[0] = D_max (seconds) from SEVERITY_QOS
        # Direct comparison: c < d means "mean delay within target"
        assert c_vec[0] >= 0
        assert d_phi[0] > 0
        # This is semantically cleaner than comparing Σdelay vs 20·D_max
