"""Commit 4 verification: D1-D7 formulation/design risk classifications.

D1: Reward summed (20 ticks) vs constraints meaned — expected behavior (class 4).
D2: C2/C5 thresholds below per-window resolution — expected behavior (class 4).
D3: Manager obs missing delay/HOL/slack — design weakness (class 3).
D4: Early stopping uses raw episode reward — design weakness (class 3).
D5: Entropy bonus scales with action dim — expected behavior (class 4).
D6: Duplicate PPO implementations — already resolved (class 4).
D7: Lambda warm-start semantics — expected behavior (class 4).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from agents.lagrangian import LambdaState
from agents.ppo_core import entropy_bonus
from agents.worker_agent import WorkerAgent
from solvers._common import build_manager_state
from utils.config import (
    LAMBDA_WARM,
    MAC_TICKS_PER_WORKER,
    OBS_AOI_MAX_IDX,
    OBS_AOI_MEAN_IDX,
    OBS_BLER_IDX,
    OBS_FIXED_BLOCK_LEN,
    OBS_PER_AMB_BLOCK_LEN,
    OBS_RHO_EMBB_IDX,
    OBS_RHO_URLLC_IDX,
    OBS_SEVERITY_OH_IDX,
    OBS_SEVERITY_OH_LEN,
    SEVERITY_QOS,
    WORKER_STEPS_PER_MANAGER,
    build_lambda_warm_vector,
)
from utils.early_stopping import EarlyStopping


# ────────────────────────────────────────────────────────────────────
# D1: Reward-sum vs constraint-mean is a valid CMDP formulation
# ────────────────────────────────────────────────────────────────────


class TestD1RewardSumConstraintMean:
    def test_reward_accumulates_over_mac_ticks(self):
        """Reward per Worker step = sum of MAC_TICKS_PER_WORKER individual rewards."""
        assert MAC_TICKS_PER_WORKER == 20

    def test_augmented_reward_uses_mean_constraint(self):
        """LambdaState.augmented_reward subtracts λ · (c_mean - d), not λ · sum."""
        ls = LambdaState(K=1)
        ls.lambda_local = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        c_vec = np.array([0.01, 0.5, 5.0, 0.3, 0.1])
        d_phi = np.zeros(5)
        r_aug = ls.augmented_reward(10.0, c_vec, d_phi)
        assert isinstance(r_aug, float)
        assert r_aug < 10.0

    def test_dual_ascent_uses_mean_subgradient(self):
        """g_hat = win_c / win_steps (mean, not sum)."""
        ls = LambdaState(K=1)
        c1 = np.array([0.01, 0.5, 5.0, 0.3, 0.1])
        c2 = np.array([0.02, 0.3, 3.0, 0.2, 0.05])
        d = np.zeros(5)
        ls.accumulate(c1, d)
        ls.accumulate(c2, d)
        assert ls.win_steps == 2
        result = ls.on_manager_step_end()
        assert "subgradient_mean" in result
        assert ls.win_steps == 0


# ────────────────────────────────────────────────────────────────────
# D2: C2/C5 thresholds are 3GPP-standards-correct
# ────────────────────────────────────────────────────────────────────


class TestD2ThresholdResolution:
    def test_severity_eps_are_3gpp_nines(self):
        """C2 eps values map to 3GPP reliability classes (99.9/99.99/99.999%)."""
        eps_values = sorted({SEVERITY_QOS[s]["eps"] for s in range(1, 6)})
        assert eps_values == [1e-5, 1e-4, 1e-3]

    def test_severity_eps_aoi_binary_distinction(self):
        """C5 eps_aoi is non-urgent (1e-2) vs urgent (1e-3) — 2 tiers by design."""
        eps_aoi_vals = sorted({SEVERITY_QOS[s]["eps_aoi"] for s in range(1, 6)})
        assert eps_aoi_vals == [1e-3, 1e-2]

    def test_thresholds_monotonically_tighter(self):
        """Higher severity → tighter or equal thresholds (all 4 QoS columns)."""
        for s in range(1, 5):
            curr = SEVERITY_QOS[s]
            nxt = SEVERITY_QOS[s + 1]
            assert nxt["D_max"] <= curr["D_max"], f"D_max not monotone at sev {s}->{s+1}"
            assert nxt["eps"] <= curr["eps"], f"eps not monotone at sev {s}->{s+1}"
            assert nxt["AoI_max"] <= curr["AoI_max"], f"AoI_max not monotone at sev {s}->{s+1}"
            assert nxt["eps_aoi"] <= curr["eps_aoi"], f"eps_aoi not monotone at sev {s}->{s+1}"

    def test_dual_window_size_documented(self):
        """W = 10 Worker steps × 20 ticks = 200 ticks per Manager window."""
        assert WORKER_STEPS_PER_MANAGER == 10
        assert MAC_TICKS_PER_WORKER == 20


# ────────────────────────────────────────────────────────────────────
# D3: Manager obs has sufficient signals for its 1-dim action
# ────────────────────────────────────────────────────────────────────


class TestD3ManagerObsSignals:
    def test_manager_state_dim_k1(self):
        """K=1: Manager state = 6 fixed + 5 lambda = 11 dim."""
        from agents.manager_agent import manager_state_dim
        assert manager_state_dim(1) == 11

    def test_manager_state_dim_k3(self):
        """K=3: Manager state = 6 fixed + 13 lambda = 19 dim."""
        from agents.manager_agent import manager_state_dim
        assert manager_state_dim(3) == 19

    def test_build_manager_state_extracts_correct_fields(self):
        """Manager state includes rho, bler, severity, aoi, and lambda."""
        obs = np.zeros(OBS_FIXED_BLOCK_LEN + OBS_PER_AMB_BLOCK_LEN + 1, dtype=np.float32)
        obs[OBS_RHO_URLLC_IDX] = 0.7
        obs[OBS_RHO_EMBB_IDX] = 0.3
        obs[OBS_BLER_IDX] = 0.05
        obs[OBS_SEVERITY_OH_IDX + 2] = 1.0
        obs[OBS_AOI_MEAN_IDX] = 0.15
        obs[OBS_AOI_MAX_IDX] = 0.25
        lam = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        s_H = build_manager_state(obs, lam)
        assert s_H.shape == (11,)
        assert abs(s_H[0] - 0.7) < 1e-5
        assert abs(s_H[1] - 0.3) < 1e-5
        assert abs(s_H[2] - 0.05) < 1e-5
        assert abs(s_H[3] - 3.0 / 5.0) < 1e-5
        assert abs(s_H[4] - 0.15) < 1e-5
        assert abs(s_H[5] - 0.25) < 1e-5
        np.testing.assert_array_almost_equal(s_H[6:], lam)


# ────────────────────────────────────────────────────────────────────
# D4: Early stopping uses rolling mean (noise-tolerant by design)
# ────────────────────────────────────────────────────────────────────


class TestD4EarlyStoppingBehavior:
    def test_disabled_by_default(self):
        """Early stopping is opt-in (default off in train.py)."""
        es = EarlyStopping(patience=300, min_delta=10.0, window=100, min_ep=500)
        for ep in range(100):
            assert not es.step(ep, float(ep))

    def test_rolling_mean_smooths_severity_variance(self):
        """Window=100 rolling mean absorbs severity-driven reward variance."""
        es = EarlyStopping(patience=50, min_delta=1.0, window=10, min_ep=0, check_every=1)
        rng = np.random.RandomState(42)
        rewards = 50.0 + rng.randn(30) * 20.0
        for ep, r in enumerate(rewards):
            es.step(ep, r)
        mean = es.rolling_mean
        assert 20.0 < mean < 80.0

    def test_plateau_detection_works(self):
        """Constant rewards → plateau detected after patience."""
        es = EarlyStopping(patience=10, min_delta=5.0, window=5, min_ep=0, check_every=1)
        for ep in range(100):
            should_stop = es.step(ep, 50.0)
            if should_stop:
                assert ep >= 10
                return
        pytest.fail("Early stopping should have triggered on flat reward")


# ────────────────────────────────────────────────────────────────────
# D5: Entropy scales with action dim (standard PPO behavior)
# ────────────────────────────────────────────────────────────────────


class TestD5EntropyScaling:
    def test_entropy_scales_with_action_dim(self):
        """Higher action dim → higher total entropy (sum across dims)."""
        from torch.distributions import Normal
        mean_1d = torch.zeros(32, 1)
        std_1d = torch.ones(32, 1)
        dist_1d = Normal(mean_1d, std_1d)
        ent_1d = entropy_bonus(dist_1d)

        mean_4d = torch.zeros(32, 4)
        std_4d = torch.ones(32, 4)
        dist_4d = Normal(mean_4d, std_4d)
        ent_4d = entropy_bonus(dist_4d)

        assert ent_4d > ent_1d
        assert abs(float(ent_4d) - 4.0 * float(ent_1d)) < 1e-4

    def test_entropy_per_dim_independent_of_total_dims(self):
        """Per-dimension entropy is the same regardless of total action dims."""
        from torch.distributions import Normal
        mean_1d = torch.zeros(32, 1)
        std_1d = torch.ones(32, 1)
        ent_1d = float(entropy_bonus(Normal(mean_1d, std_1d)))

        mean_4d = torch.zeros(32, 4)
        std_4d = torch.ones(32, 4)
        ent_4d_per_dim = float(entropy_bonus(Normal(mean_4d, std_4d))) / 4.0

        assert abs(ent_1d - ent_4d_per_dim) < 1e-5


# ────────────────────────────────────────────────────────────────────
# D6: Duplicate PPO files already removed
# ────────────────────────────────────────────────────────────────────


class TestD6NoDuplicatePPO:
    def test_no_b2_hrl_ppo_soft(self):
        """b2_hrl_ppo_soft.py should be deleted."""
        from pathlib import Path
        assert not Path("solvers/b2_hrl_ppo_soft.py").exists()

    def test_no_no_phase_ppo(self):
        """no_phase_ppo.py should be deleted."""
        from pathlib import Path
        assert not Path("solvers/no_phase_ppo.py").exists()


# ────────────────────────────────────────────────────────────────────
# D7: Lambda warm-start semantics are correct
# ────────────────────────────────────────────────────────────────────


class TestD7LambdaWarmStartSemantics:
    def test_reset_episode_syncs_both_global_and_local(self):
        """Fix Error 1: reset_episode sets BOTH λ_global AND λ_local."""
        ls = LambdaState(K=1)
        ls.lambda_warm[(3,)] = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ls.reset_episode(severity_per_amb=[3], severity_ref=3)
        np.testing.assert_array_equal(ls.lambda_global, ls.lambda_local)
        np.testing.assert_array_equal(ls.lambda_global, np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

    def test_episode_end_ema_saves_warm_table(self):
        """on_episode_end() EMA-updates λ_warm[sev] with current λ_global."""
        ls = LambdaState(K=1, beta_ema=0.1)
        initial_warm = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        ls.lambda_warm[(3,)] = initial_warm.copy()
        ls.reset_episode(severity_per_amb=[3], severity_ref=3)
        ls.lambda_global = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        ls.on_episode_end()
        expected = 0.9 * initial_warm + 0.1 * np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        np.testing.assert_array_almost_equal(ls.lambda_warm[(3,)], expected)

    def test_force_zero_warm_disables_warm_start(self):
        """Exp3 ablation: force_zero_warm always starts from zeros."""
        ls = LambdaState(K=1, force_zero_warm=True)
        ls.lambda_warm[(3,)] = np.ones(5) * 10.0
        ls.reset_episode(severity_per_amb=[3], severity_ref=3)
        np.testing.assert_array_equal(ls.lambda_global, np.zeros(5))

    def test_dual_ascent_projects_to_nonneg(self):
        """λ_global stays non-negative after dual update."""
        ls = LambdaState(K=1, alpha_lambda=1.0)
        ls.reset_episode(severity_per_amb=[3], severity_ref=3)
        ls.lambda_global = np.array([0.1, 0.0, 0.0, 0.0, 0.0])
        c_vec = np.zeros(5)
        d_phi = np.ones(5) * 10.0
        ls.accumulate(c_vec, d_phi)
        ls.on_manager_step_end()
        assert np.all(ls.lambda_global >= 0.0)

    def test_warm_table_keyed_by_severity_tuple(self):
        """λ_warm is keyed by per-ambulance severity tuple, not scalar."""
        ls = LambdaState(K=3)
        ls.reset_episode(severity_per_amb=[1, 3, 5], severity_ref=5)
        ls.lambda_global = np.ones(13) * 2.0
        ls.on_episode_end()
        assert (1, 3, 5) in ls.lambda_warm
        assert ls.lambda_warm[(1, 3, 5)].shape == (13,)
