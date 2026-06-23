"""Rigorous formula tests — W09/W10 GAE, SMDP return, PPO clip loss.

Every expected value is hand-computed from first principles. No function is
called as its own oracle.

W09 GAE (Schulman 2016 eq.16):
  δ_t   = r_t + γ·V_{t+1}·(1-done) − V_t
  A_t   = δ_t + (γλ)·(1-done)·A_{t+1}     (backward pass)
  R_t   = A_t + V_t                         (discounted returns)

W10 SMDP return:
  r_H = Σ_{i=0}^{W-1} γ_L^i · r_aug_i     (SMDP-discounted Manager reward)

W10 γ_H = γ_L^W:
  GAMMA_MANAGER = GAMMA^WORKER_STEPS_PER_MANAGER = 0.99^10

W09 PPO clip (Schulman 2017 eq.7):
  ratio  = exp(new_log_prob − old_log_prob)
  surr1  = ratio · A
  surr2  = clip(ratio, 1−ε, 1+ε) · A
  L^CLIP = −mean(min(surr1, surr2))
  clip_frac = fraction of |ratio − 1| > ε

W09 value_loss:
  L_V = 0.5 · MSE(V(s), R)
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from agents.ppo_core import compute_gae, ppo_clip_loss, value_loss
from utils.config import GAMMA, GAMMA_MANAGER, GAMMA_WORKER, WORKER_STEPS_PER_MANAGER


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lp_for_ratio(ratio: float, old_lp: float = 0.0) -> float:
    """new_log_prob that gives exactly `ratio` when old_lp is given."""
    return old_lp + math.log(ratio)


def _tensor(*values: float) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


# ---------------------------------------------------------------------------
# W09 — compute_gae
# ---------------------------------------------------------------------------
# Reference computation for a 3-step episode (T=3, no terminal):
#   rewards = [1.0, 0.5, 0.2], values = [2.0, 1.8, 1.5], dones=[0,0,0]
#   last_value = 1.0, gamma = 0.99, lam = 0.95
#
# Backward pass:
#   t=2: delta = 0.2 + 0.99*1.0 - 1.5 = -0.31
#        gae   = -0.31
#   t=1: delta = 0.5 + 0.99*1.5 - 1.8 = 0.185
#        gae   = 0.185 + 0.9405*(-0.31) = -0.106555
#   t=0: delta = 1.0 + 0.99*1.8 - 2.0 = 0.782
#        gae   = 0.782 + 0.9405*(-0.106555) ≈ 0.68178...
# returns = advantages + values


class TestComputeGAE:
    def _setup_3step(self):
        r = np.array([1.0, 0.5, 0.2], dtype=np.float32)
        v = np.array([2.0, 1.8, 1.5], dtype=np.float32)
        d = np.array([0, 0, 0], dtype=np.float32)
        return r, v, d

    def test_three_step_adv_t2(self):
        r, v, d = self._setup_3step()
        adv, _ = compute_gae(r, v, d, last_value=1.0, gamma=0.99, lam=0.95)
        # t=2: delta = 0.2 + 0.99*1.0 - 1.5 = -0.31 (exact)
        assert adv[2] == pytest.approx(-0.31, abs=1e-5)

    def test_three_step_adv_t1(self):
        r, v, d = self._setup_3step()
        adv, _ = compute_gae(r, v, d, last_value=1.0, gamma=0.99, lam=0.95)
        # t=1: delta=0.185, gae = 0.185 + 0.9405*(-0.31) = -0.106555
        assert adv[1] == pytest.approx(-0.106555, abs=1e-5)

    def test_three_step_adv_t0(self):
        r, v, d = self._setup_3step()
        adv, _ = compute_gae(r, v, d, last_value=1.0, gamma=0.99, lam=0.95)
        # t=0: delta=0.782, gae = 0.782 + 0.9405*(-0.106555) ≈ 0.68178
        assert adv[0] == pytest.approx(0.68178, abs=1e-4)

    def test_three_step_returns(self):
        r, v, d = self._setup_3step()
        adv, ret = compute_gae(r, v, d, last_value=1.0, gamma=0.99, lam=0.95)
        # returns = advantages + values
        np.testing.assert_allclose(ret, adv + np.array([2.0, 1.8, 1.5], dtype=np.float32), atol=1e-5)

    def test_three_step_returns_t2(self):
        r, v, d = self._setup_3step()
        _, ret = compute_gae(r, v, d, last_value=1.0, gamma=0.99, lam=0.95)
        # ret[2] = adv[2] + v[2] = -0.31 + 1.5 = 1.19
        assert ret[2] == pytest.approx(1.19, abs=1e-5)

    def test_three_step_output_shape(self):
        r, v, d = self._setup_3step()
        adv, ret = compute_gae(r, v, d, last_value=1.0, gamma=0.99, lam=0.95)
        assert adv.shape == (3,)
        assert ret.shape == (3,)

    def test_terminal_step_ignores_bootstrap(self):
        # dones=[0,1]: step t=1 is terminal → last_value should NOT be used in t=1's delta
        # t=1: nonterm=0 → delta = r1 + 0 - v1, gae = delta (no future GAE propagated)
        r = np.array([1.0, 2.0], dtype=np.float32)
        v = np.array([0.5, 0.5], dtype=np.float32)
        d = np.array([0, 1], dtype=np.float32)
        adv, ret = compute_gae(r, v, d, last_value=100.0, gamma=0.99, lam=0.95)
        # t=1: delta = 2.0 + 0 - 0.5 = 1.5, gae = 1.5
        assert adv[1] == pytest.approx(1.5, abs=1e-5)
        # t=0: nonterm=1, delta = 1.0 + 0.99*0.5 - 0.5 = 0.995
        #      gae = 0.995 + 0.9405*1.5 = 0.995 + 1.41075 = 2.40575
        assert adv[0] == pytest.approx(2.40575, abs=1e-4)

    def test_terminal_returns_equal_reward_at_terminal(self):
        # At terminal step: return = reward (since V_{T+1}=0 for done, and return = adv + v)
        # adv[1] = r1 - v1 → ret[1] = (r1 - v1) + v1 = r1
        r = np.array([1.0, 2.0], dtype=np.float32)
        v = np.array([0.5, 0.5], dtype=np.float32)
        d = np.array([0, 1], dtype=np.float32)
        _, ret = compute_gae(r, v, d, last_value=100.0, gamma=0.99, lam=0.95)
        assert ret[1] == pytest.approx(2.0, abs=1e-5)

    def test_single_step_no_bootstrap_when_done(self):
        # T=1, done=1: adv = r - v, NOT r + gamma*last_value - v
        r = np.array([3.0], dtype=np.float32)
        v = np.array([1.0], dtype=np.float32)
        d = np.array([1], dtype=np.float32)
        adv, ret = compute_gae(r, v, d, last_value=999.0, gamma=0.99, lam=0.95)
        assert adv[0] == pytest.approx(3.0 - 1.0, abs=1e-5)
        assert ret[0] == pytest.approx(3.0, abs=1e-5)

    def test_gamma_zero_reduces_to_td0(self):
        # With gamma=0: delta_t = r_t - V_t (no bootstrap), gae propagation=0
        r = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        v = np.array([0.5, 1.0, 1.5], dtype=np.float32)
        d = np.array([0, 0, 0], dtype=np.float32)
        adv, _ = compute_gae(r, v, d, last_value=0.0, gamma=0.0, lam=0.95)
        expected = r - v
        np.testing.assert_allclose(adv, expected, atol=1e-6)

    def test_lam_zero_reduces_to_td_residual_only(self):
        # lam=0: gae = delta_t only (no advantage propagation from future)
        r = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        v = np.array([1.5, 2.5, 3.5], dtype=np.float32)
        d = np.array([0, 0, 0], dtype=np.float32)
        adv, _ = compute_gae(r, v, d, last_value=0.0, gamma=0.99, lam=0.0)
        # t=2: delta = 3.0 + 0.99*0 - 3.5 = -0.5; gae = -0.5
        assert adv[2] == pytest.approx(-0.5, abs=1e-5)
        # t=1: delta = 2.0 + 0.99*3.5 - 2.5 = 2.0 + 3.465 - 2.5 = 2.965; gae = 2.965
        assert adv[1] == pytest.approx(2.965, abs=1e-5)
        # t=0: delta = 1.0 + 0.99*2.5 - 1.5 = 1.0 + 2.475 - 1.5 = 1.975; gae = 1.975
        assert adv[0] == pytest.approx(1.975, abs=1e-5)


# ---------------------------------------------------------------------------
# W10 — SMDP return
# ---------------------------------------------------------------------------
# r_H = Σ_{i=0}^{W-1} γ_L^i · r_aug_i
# (SMDP-discounted Manager reward over a window of W Worker steps)


def smdp_return(r_aug_list: list[float], gamma_l: float = GAMMA) -> float:
    """Reference implementation of the SMDP-discounted Manager return."""
    return sum(gamma_l ** i * r for i, r in enumerate(r_aug_list))


class TestSMDPReturn:
    def test_single_step_window(self):
        # W=1: r_H = r0 (no discounting)
        assert smdp_return([1.0]) == pytest.approx(1.0)
        assert smdp_return([2.5]) == pytest.approx(2.5)

    def test_three_step_window_exact(self):
        # W=3: r_H = 1.0 + 0.5*γ + 0.2*γ²
        gamma = 0.99
        r_aug = [1.0, 0.5, 0.2]
        expected = 1.0 + 0.5 * 0.99 + 0.2 * (0.99 ** 2)
        # = 1.0 + 0.495 + 0.2*0.9801 = 1.0 + 0.495 + 0.19602 = 1.69102
        assert expected == pytest.approx(1.69102, abs=1e-8)
        assert smdp_return(r_aug, gamma) == pytest.approx(1.69102, abs=1e-8)

    def test_five_step_uniform_window(self):
        # W=5, all r_aug=0.5: r_H = 0.5 * Σ_{i=0}^{4} γ^i = 0.5 * (1-γ^5)/(1-γ)
        gamma = 0.99
        r_aug = [0.5] * 5
        geo_sum = (1 - gamma ** 5) / (1 - gamma)
        expected = 0.5 * geo_sum
        assert smdp_return(r_aug, gamma) == pytest.approx(expected, rel=1e-9)

    def test_ten_step_window_gamma_manager_equivalent(self):
        # Geometric sum of 10 discounted steps with uniform r=1:
        # Σ_{i=0}^{9} 0.99^i = (1 - 0.99^10) / (1 - 0.99) = (1 - γ_H) / (1 - γ_L)
        gamma = 0.99
        W = WORKER_STEPS_PER_MANAGER
        r_aug = [1.0] * W
        expected = (1.0 - gamma ** W) / (1.0 - gamma)
        assert smdp_return(r_aug, gamma) == pytest.approx(expected, rel=1e-9)

    def test_zero_rewards_returns_zero(self):
        assert smdp_return([0.0] * 10) == pytest.approx(0.0)

    def test_negative_rewards(self):
        # Penalty-heavy episode
        r_aug = [-1.0, -0.5]
        expected = -1.0 + (-0.5) * 0.99
        assert smdp_return(r_aug) == pytest.approx(expected, rel=1e-9)

    def test_discounting_reduces_value(self):
        # r_H with discounting < sum of raw rewards (for positive rewards)
        r_aug = [1.0] * 10
        raw_sum = sum(r_aug)
        discounted = smdp_return(r_aug)
        assert discounted < raw_sum

    def test_gamma_manager_is_gamma_power_w(self):
        # The γ_H used for the Manager IS exactly γ_L^W (config invariant)
        expected = GAMMA_WORKER ** WORKER_STEPS_PER_MANAGER
        assert GAMMA_MANAGER == pytest.approx(expected, rel=1e-12)
        assert GAMMA_MANAGER == pytest.approx(0.99 ** 10, rel=1e-12)


# ---------------------------------------------------------------------------
# W09 — ppo_clip_loss
# ---------------------------------------------------------------------------
# For each element:
#   ratio  = exp(new_lp - old_lp)
#   surr1  = ratio * A
#   surr2  = clip(ratio, 1-ε, 1+ε) * A
#   element loss = -min(surr1, surr2)
# Final loss = mean over batch; clip_frac = fraction with |ratio-1| > ε


class TestPPOClipLoss:
    def _make(self, ratio: float, adv: float, eps: float = 0.2):
        old_lp = _tensor(0.0)
        new_lp = _tensor(_lp_for_ratio(ratio))
        a = _tensor(adv)
        return ppo_clip_loss(new_lp, old_lp, a, eps)

    # --- No-clip cases (ratio in [1-ε, 1+ε]) ---

    def test_ratio_one_no_clip_positive_adv(self):
        # ratio=1: surr1=surr2=A → loss = -A, clip_frac=0
        loss, frac = self._make(1.0, 1.0)
        assert float(loss) == pytest.approx(-1.0, abs=1e-5)
        assert float(frac) == pytest.approx(0.0)

    def test_ratio_one_no_clip_negative_adv(self):
        # ratio=1, A=-1: loss = -(-1) = 1, clip_frac=0
        loss, frac = self._make(1.0, -1.0)
        assert float(loss) == pytest.approx(1.0, abs=1e-5)
        assert float(frac) == pytest.approx(0.0)

    def test_ratio_slightly_above_one_no_clip(self):
        # ratio=1.1, eps=0.2: in [0.8,1.2] → no clip
        loss, frac = self._make(1.1, 1.0)
        assert float(loss) == pytest.approx(-1.1, abs=1e-5)
        assert float(frac) == pytest.approx(0.0)

    def test_ratio_slightly_below_one_no_clip(self):
        # ratio=0.9, eps=0.2: in [0.8,1.2] → no clip
        loss, frac = self._make(0.9, 1.0)
        assert float(loss) == pytest.approx(-0.9, abs=1e-5)
        assert float(frac) == pytest.approx(0.0)

    # --- Clipped above (ratio > 1+ε, positive advantage) ---

    def test_ratio_above_eps_positive_adv_clipped(self):
        # ratio=1.5, eps=0.2, A=1: clip to 1.2, surr2=1.2 < surr1=1.5
        # min(1.5, 1.2) = 1.2 → loss = -1.2
        loss, frac = self._make(1.5, 1.0)
        assert float(loss) == pytest.approx(-1.2, abs=1e-5)
        assert float(frac) == pytest.approx(1.0)

    def test_ratio_above_eps_negative_adv_uses_surr1(self):
        # ratio=1.5, A=-1: surr1=-1.5, surr2=-1.2 → min(-1.5,-1.2)=-1.5
        # loss = 1.5 (pessimistic bound)
        loss, frac = self._make(1.5, -1.0)
        assert float(loss) == pytest.approx(1.5, abs=1e-5)
        assert float(frac) == pytest.approx(1.0)

    # --- Clipped below (ratio < 1-ε, negative advantage) ---

    def test_ratio_below_eps_negative_adv_clipped(self):
        # ratio=0.5, eps=0.2, A=-1: clip to 0.8, surr2=-0.8
        # surr1=-0.5, min(-0.5,-0.8)=-0.8 → loss=0.8
        loss, frac = self._make(0.5, -1.0)
        assert float(loss) == pytest.approx(0.8, abs=1e-5)
        assert float(frac) == pytest.approx(1.0)

    def test_ratio_below_eps_positive_adv_uses_surr1(self):
        # ratio=0.5, A=1: surr1=0.5, surr2=0.8 → min(0.5,0.8)=0.5 (surr1 wins)
        # loss = -0.5
        loss, frac = self._make(0.5, 1.0)
        assert float(loss) == pytest.approx(-0.5, abs=1e-5)
        assert float(frac) == pytest.approx(1.0)

    # --- Batch tests ---

    def test_batch_clip_fraction_half(self):
        # 4 elements: ratios=[1.0, 1.5, 0.5, 1.1], eps=0.2
        # Clipped: [F, T, T, F] → clip_frac = 0.5
        old_lps = torch.zeros(4)
        new_lps = torch.tensor(
            [_lp_for_ratio(1.0), _lp_for_ratio(1.5), _lp_for_ratio(0.5), _lp_for_ratio(1.1)]
        )
        advs = torch.ones(4)
        _, frac = ppo_clip_loss(new_lps, old_lps, advs, 0.2)
        assert float(frac) == pytest.approx(0.5, abs=1e-5)

    def test_batch_loss_averages_element_losses(self):
        # 2 elements: ratio=1.0 A=1 (loss=-1.0), ratio=1.0 A=-1 (loss=1.0)
        # mean = 0.0
        old_lps = torch.zeros(2)
        new_lps = torch.tensor([0.0, 0.0])
        advs = torch.tensor([1.0, -1.0])
        loss, _ = ppo_clip_loss(new_lps, old_lps, advs, 0.2)
        assert float(loss) == pytest.approx(0.0, abs=1e-5)

    def test_clip_fraction_all_clipped(self):
        # All ratios outside clip range → clip_frac = 1.0
        old_lps = torch.zeros(3)
        new_lps = torch.tensor(
            [_lp_for_ratio(2.0), _lp_for_ratio(0.3), _lp_for_ratio(3.0)]
        )
        advs = torch.ones(3)
        _, frac = ppo_clip_loss(new_lps, old_lps, advs, 0.2)
        assert float(frac) == pytest.approx(1.0, abs=1e-5)

    def test_clip_fraction_none_clipped(self):
        # All ratios inside clip range → clip_frac = 0.0
        old_lps = torch.zeros(3)
        new_lps = torch.tensor(
            [_lp_for_ratio(1.0), _lp_for_ratio(1.05), _lp_for_ratio(0.95)]
        )
        advs = torch.ones(3)
        _, frac = ppo_clip_loss(new_lps, old_lps, advs, 0.2)
        assert float(frac) == pytest.approx(0.0, abs=1e-5)

    def test_custom_eps_changes_clip_boundary(self):
        # ratio=1.15 with eps=0.1 → clipped; with eps=0.2 → not clipped
        old_lps = torch.zeros(1)
        new_lps = torch.tensor([_lp_for_ratio(1.15)])
        advs = torch.ones(1)
        _, frac_tight = ppo_clip_loss(new_lps, old_lps, advs, 0.1)
        _, frac_loose = ppo_clip_loss(new_lps, old_lps, advs, 0.2)
        assert float(frac_tight) == pytest.approx(1.0)
        assert float(frac_loose) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# W09 — value_loss
# ---------------------------------------------------------------------------
# L_V = vf_coef * MSE(V(s), R) = vf_coef * mean((V-R)^2)


class TestValueLoss:
    def test_perfect_prediction_zero_loss(self):
        v = _tensor(1.0, 2.0, 3.0)
        r = _tensor(1.0, 2.0, 3.0)
        loss = value_loss(v, r, vf_coef=0.5)
        assert float(loss) == pytest.approx(0.0, abs=1e-7)

    def test_single_element_exact(self):
        # MSE([3.0], [1.0]) = (3-1)^2 = 4; 0.5 * 4 = 2.0
        v = _tensor(3.0)
        r = _tensor(1.0)
        loss = value_loss(v, r, vf_coef=0.5)
        assert float(loss) == pytest.approx(2.0, abs=1e-6)

    def test_two_elements_exact(self):
        # MSE([1.0, 3.0], [2.0, 2.0]) = ((1-2)^2 + (3-2)^2) / 2 = 1.0
        # 0.5 * 1.0 = 0.5
        v = _tensor(1.0, 3.0)
        r = _tensor(2.0, 2.0)
        loss = value_loss(v, r, vf_coef=0.5)
        assert float(loss) == pytest.approx(0.5, abs=1e-6)

    def test_vf_coef_scales_loss(self):
        v = _tensor(2.0)
        r = _tensor(1.0)
        loss_half = value_loss(v, r, vf_coef=0.5)    # 0.5 * 1 = 0.5
        loss_full = value_loss(v, r, vf_coef=1.0)    # 1.0 * 1 = 1.0
        assert float(loss_full) == pytest.approx(2 * float(loss_half), rel=1e-5)

    def test_symmetric_errors(self):
        # Errors symmetric in sign → same loss
        v_pos = _tensor(2.0)
        v_neg = _tensor(0.0)
        r = _tensor(1.0)
        loss_pos = value_loss(v_pos, r)
        loss_neg = value_loss(v_neg, r)
        assert float(loss_pos) == pytest.approx(float(loss_neg), rel=1e-5)

    def test_three_elements_exact(self):
        # v=[0,0,0], r=[1,2,3]: MSE=(1+4+9)/3=14/3; 0.5 * 14/3 = 7/3
        v = _tensor(0.0, 0.0, 0.0)
        r = _tensor(1.0, 2.0, 3.0)
        loss = value_loss(v, r, vf_coef=0.5)
        assert float(loss) == pytest.approx(7 / 3, rel=1e-5)
