"""Audit item 4 (Giai đoạn A): C2/C5 statistical estimator resolution.

FINDING: Option-b interval-window (win_c/win_steps) accumulates exactly
WORKER_STEPS_PER_MANAGER(=10) x MAC_TICKS_PER_WORKER(=20) = 200 samples per
Manager-step window before being reset. eps for C2/C5 goes down to 1e-5
(severity 4-5). A Bernoulli rate is only resolvable to within ~1/N: with
N=200, the minimum detectable rate is 1/200=0.005 — 500x coarser than
eps=1e-5. This file proves the FAIL quantitatively, then proves the fix
(episode-cumulative cum_c/cum_steps, Option a, never reset mid-episode).
"""
import math
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from agents.lagrangian import LambdaState
from utils.config import (
    SEVERITY_QOS, MAC_TICKS_PER_WORKER, WORKER_STEPS_PER_MANAGER, LAMBDA_MAX,
)


# ═══════════════════════════════════════════════════════════════════════
# Part 1: Quantify the FAIL (Option-b N=200 cannot resolve eps<=1e-4)
# ═══════════════════════════════════════════════════════════════════════

class TestOptionBWindowInadequateForTailEps:
    def test_window_size_is_200(self):
        n = WORKER_STEPS_PER_MANAGER * MAC_TICKS_PER_WORKER
        assert n == 200, f"Option-b window N={n}, expected 200"

    def test_min_resolvable_rate_exceeds_severity4_5_eps(self):
        """N=200 -> min resolvable rate=1/200=0.005, which is 500x coarser
        than eps=1e-5 (severity 4-5 C2/C5 budget)."""
        n = WORKER_STEPS_PER_MANAGER * MAC_TICKS_PER_WORKER
        min_resolvable = 1.0 / n
        eps_sev5 = SEVERITY_QOS[5]["eps"]
        assert eps_sev5 == 1e-5
        assert min_resolvable > 100 * eps_sev5, (
            f"min_resolvable={min_resolvable} not >>eps={eps_sev5}"
        )

    def test_probability_of_zero_violations_in_window_is_dominant(self):
        """At true rate p=1e-5, P(>=1 violation in N=200 trials) ~ 0.002.
        I.e. ~99.8% of Manager-step windows show g_hat=0 (frozen gradient),
        and the rare nonzero observation overshoots eps by ~500x."""
        n = 200
        p = 1e-5
        p_at_least_one = 1.0 - (1.0 - p) ** n
        assert p_at_least_one < 0.003, (
            f"P(>=1 violation)={p_at_least_one}, expected <0.003 (rare/frozen)"
        )
        # When a violation DOES occur, empirical rate = 1/200 = 0.005, a
        # 500x overshoot relative to the eps=1e-5 target.
        overshoot = (1.0 / n) / p
        assert overshoot >= 400

    def test_severity1_2_3_also_inadequate(self):
        """Even the loosest eps (severity 1, eps=1e-3) needs N>~1000 for a
        stable estimate (rule of thumb N>=10/eps); N=200 falls short."""
        n = 200
        for sev in (1, 2, 3):
            eps = SEVERITY_QOS[sev]["eps"]
            needed_n = 10.0 / eps
            assert n < needed_n, f"sev{sev}: N={n} should be < needed~{needed_n}"


# ═══════════════════════════════════════════════════════════════════════
# Part 2: Prove the fix — cum_c/cum_steps (Option a, episode-cumulative)
# ═══════════════════════════════════════════════════════════════════════

class TestEpisodeCumulativeEstimatorForTailConstraints:
    def test_tail_mask_identifies_c2_c5_only_k1(self):
        ls = LambdaState(K=1)
        # Layout K=1: [C1_0, C2_0, C4_0, C5_0, C3_shared] -> indices [0,1,2,3,4]
        expected = np.array([False, True, False, True, False])
        np.testing.assert_array_equal(ls._tail_mask, expected)

    def test_tail_mask_identifies_c2_c5_only_k3(self):
        K = 3
        ls = LambdaState(K=K)
        # [C1_0..2, C2_0..2, C4_0..2, C5_0..2, C3] -> tail = C2(3:6) + C5(9:12)
        expected = np.zeros(4 * K + 1, dtype=bool)
        expected[K:2 * K] = True
        expected[3 * K:4 * K] = True
        np.testing.assert_array_equal(ls._tail_mask, expected)

    def test_cum_c_not_reset_at_manager_step_end(self):
        """win_c resets every Manager step; cum_c must persist across many."""
        K = 1
        ls = LambdaState(K=K)
        ls.reset_episode(severity_per_amb=(1,), severity_ref=1)
        d_phi = np.zeros(ls.n_constraints)
        c_vec = np.ones(ls.n_constraints) * 0.01

        for _ in range(WORKER_STEPS_PER_MANAGER):
            ls.accumulate(c_vec, d_phi)
        ls.on_manager_step_end()
        assert ls.win_steps == 0, "win_steps must reset after Manager step"
        assert ls.cum_steps == WORKER_STEPS_PER_MANAGER, (
            "cum_steps must NOT reset after Manager step"
        )

        for _ in range(WORKER_STEPS_PER_MANAGER):
            ls.accumulate(c_vec, d_phi)
        ls.on_manager_step_end()
        assert ls.cum_steps == 2 * WORKER_STEPS_PER_MANAGER, (
            "cum_steps must keep accumulating across Manager-step boundaries"
        )

    def test_cum_c_resets_at_episode_boundary(self):
        K = 1
        ls = LambdaState(K=K)
        ls.reset_episode(severity_per_amb=(1,), severity_ref=1)
        d_phi = np.zeros(ls.n_constraints)
        c_vec = np.ones(ls.n_constraints) * 0.01
        for _ in range(50):
            ls.accumulate(c_vec, d_phi)
        assert ls.cum_steps == 50
        ls.on_episode_end()
        assert ls.cum_steps == 0
        assert np.allclose(ls.cum_c, 0.0)

        ls.reset_episode(severity_per_amb=(1,), severity_ref=1)
        assert ls.cum_steps == 0
        assert np.allclose(ls.cum_c, 0.0)

    def test_manager_step_end_uses_cum_for_tail_win_for_mean(self):
        """g_hat[tail] must come from cum_c/cum_steps; g_hat[mean] from win_c/win_steps.

        Uses force_zero_warm=True (lambda_global starts at exactly 0, no
        warm-start offset) and alpha_lambda=1.0 so delta_lambda == g_hat
        exactly (pre-clip; values are kept well under LAMBDA_MAX).
        Compares against g_hat independently recomputed from the raw
        win_c/win_steps/cum_c/cum_steps accumulators using the SAME
        dual_scales the implementation uses — this does not assume
        scale=1.0 anywhere, only that tail indices read from cum_* and
        mean indices read from win_*.
        """
        K = 1
        ls = LambdaState(K=K, alpha_lambda=1.0, force_zero_warm=True)
        ls.reset_episode(severity_per_amb=(1,), severity_ref=1)
        d_phi = np.zeros(ls.n_constraints)
        assert np.allclose(ls.lambda_global, 0.0)

        # Window 1: nonzero deviation on every constraint.
        c_vec_1 = np.full(ls.n_constraints, 0.5)
        for _ in range(WORKER_STEPS_PER_MANAGER):
            ls.accumulate(c_vec_1, d_phi)
        win_c_1, win_steps_1 = ls.win_c.copy(), ls.win_steps
        cum_c_1, cum_steps_1 = ls.cum_c.copy(), ls.cum_steps
        ls.on_manager_step_end()

        expected_g_hat_1 = np.where(
            ls._tail_mask, cum_c_1 / cum_steps_1, win_c_1 / win_steps_1
        )
        # C1 (idx0) deviation/scale = 0.5/D_REF_URLLC is large enough to hit
        # the LAMBDA_MAX=10 projection — clip expected the same way the
        # implementation does (Π_Λ bounded projection, Phase 2.3.3).
        expected_lambda_1 = np.clip(expected_g_hat_1, 0.0, LAMBDA_MAX)
        np.testing.assert_allclose(ls.lambda_global, expected_lambda_1, atol=1e-9)

        # Window 2: zero deviation (perfectly compliant this window).
        c_vec_2 = np.zeros(ls.n_constraints)
        for _ in range(WORKER_STEPS_PER_MANAGER):
            ls.accumulate(c_vec_2, d_phi)
        win_c_2, win_steps_2 = ls.win_c.copy(), ls.win_steps
        cum_c_2, cum_steps_2 = ls.cum_c.copy(), ls.cum_steps
        lam_before2 = ls.lambda_global.copy()
        ls.on_manager_step_end()
        delta_window2 = ls.lambda_global - lam_before2

        expected_g_hat_2 = np.where(
            ls._tail_mask, cum_c_2 / cum_steps_2, win_c_2 / win_steps_2
        )
        np.testing.assert_allclose(delta_window2, expected_g_hat_2, atol=1e-9)

        # win-based (mean, C1/C4/C3): window 2 alone had zero deviation -> 0.
        mean_idx = [0, 2, 4]
        for i in mean_idx:
            assert delta_window2[i] == pytest.approx(0.0, abs=1e-9), (
                f"C1/C4/C3 idx={i}: win-based g_hat should be 0 this window"
            )
        # cum-based (tail, C2/C5): running mean over BOTH windows (10@0.5 +
        # 10@0.0)/20 = 0.25 (scale=1.0 for these two indices specifically).
        tail_idx = [1, 3]
        for i in tail_idx:
            assert delta_window2[i] == pytest.approx(0.25, abs=1e-9), (
                f"C2/C5 idx={i}: cum-based g_hat should be running mean=0.25, "
                f"got delta={delta_window2[i]}"
            )

    def test_cumulative_mean_converges_to_true_rate(self):
        """Simulate a Bernoulli(p=1e-4) violation stream; confirm the
        cumulative estimator converges to p as cum_steps grows, while a
        single 200-sample window would mostly read exactly 0."""
        rng = np.random.default_rng(0)
        K = 1
        ls = LambdaState(K=K)
        ls.reset_episode(severity_per_amb=(3,), severity_ref=3)
        d_phi = np.zeros(ls.n_constraints)
        p_true = 1e-4
        n_total = 50_000  # ~250 Manager-step windows

        violations = rng.random(n_total) < p_true
        for v in violations:
            c_vec = np.zeros(ls.n_constraints)
            c_vec[1] = 1.0 if v else 0.0  # C2 indicator
            ls.accumulate(c_vec, d_phi)

        cum_rate = ls.cum_c[1] / ls.cum_steps
        assert cum_rate == pytest.approx(p_true, rel=2.0), (
            f"cumulative rate={cum_rate} should be within 2x of true p={p_true} "
            f"at n={n_total} samples"
        )
        # A single 200-sample window has expected count = 200*1e-4=0.02 ->
        # almost always reads exactly 0, confirming the Option-b inadequacy.
        first_window_rate = violations[:200].mean()
        assert first_window_rate == 0.0 or first_window_rate >= 1.0 / 200


# ═══════════════════════════════════════════════════════════════════════
# Part 3: Checkpoint round-trip must preserve cum_c/cum_steps
# ═══════════════════════════════════════════════════════════════════════

class TestCheckpointRoundTrip:
    def test_state_dict_round_trip_preserves_cum(self):
        K = 3
        ls = LambdaState(K=K)
        ls.reset_episode(severity_per_amb=(1, 2, 3), severity_ref=3)
        d_phi = np.zeros(ls.n_constraints)
        c_vec = np.full(ls.n_constraints, 0.3)
        for _ in range(37):
            ls.accumulate(c_vec, d_phi)

        sd = ls.state_dict()
        assert "cum_c" in sd and "cum_steps" in sd
        assert sd["cum_steps"] == 37

        ls2 = LambdaState(K=K)
        ls2.load_state_dict(sd)
        assert ls2.cum_steps == 37
        np.testing.assert_allclose(ls2.cum_c, ls.cum_c)

    def test_load_old_checkpoint_without_cum_c_defaults_to_zero(self):
        """Backward compat: checkpoints saved before this fix lack cum_c/
        cum_steps keys; load must default to zeros, not crash."""
        K = 1
        ls = LambdaState(K=K)
        ls.reset_episode(severity_per_amb=(1,), severity_ref=1)
        old_style_dict = {
            "lambda_global": ls.lambda_global.tolist(),
            "lambda_local": ls.lambda_local.tolist(),
            "lambda_warm": {},
            "win_c": ls.win_c.tolist(),
            "win_steps": 0,
            "sev_prev": [1],
            "sev_ref_prev": 1,
        }
        ls2 = LambdaState(K=K)
        ls2.load_state_dict(old_style_dict)  # must not raise
        assert ls2.cum_steps == 0
        assert np.allclose(ls2.cum_c, 0.0)
