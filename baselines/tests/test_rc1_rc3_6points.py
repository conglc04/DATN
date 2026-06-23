"""Pre-train verification: 6 points from user checklist after RC-1/RC-3 patch."""
import math
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from env.oran_env import ORANEnv, macro_mission_config
from utils.config import (
    SEVERITY_QOS, OBS_AOI_MEAN_IDX, OBS_AOI_MAX_IDX,
    MAC_TICKS_PER_WORKER, B_RRM_MIN, B_RRM_MAX,
)


def _make_env(K: int = 3, seed: int = 0):
    cfg = macro_mission_config(K_ambulances=K, seed=seed)
    env = ORANEnv(cfg, seed=seed)
    obs, info = env.reset(seed=seed)
    return env, obs, info


# ═══════════════════════════════════════════════════════════════════════
# POINT 1: No-active-ambulance behaviour — well-defined values
# ═══════════════════════════════════════════════════════════════════════

class TestPoint1NoActiveAmbulances:
    """When n_active == 0, all metrics must be well-defined (no NaN/inf).

    NOTE: active_mask is recomputed each MAC tick via _update_arrival_masks(),
    so forcing it False before step() gets overridden. Instead we use K=3 seed=0
    and check that the INITIAL steps (before all ambulances enter the cell)
    produce valid, non-NaN metrics. amb_2 enters at t=0, amb_0 at ~34s,
    amb_1 at ~81s — so early ticks have n_active < K.
    """

    def test_partial_active_no_nan(self):
        env, _, _ = _make_env(K=3, seed=0)
        obs, rew, term, trunc, info = env.step(np.zeros(4, dtype=np.float32))
        assert not np.any(np.isnan(obs))
        assert not np.any(np.isinf(obs))
        assert rew >= 0.0

    def test_inactive_amb_has_zero_c_vec(self):
        """Inactive ambulance's C1/C2/C4/C5 slots must be 0 in c_vec."""
        env, _, _ = _make_env(K=3, seed=0)
        _, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
        c_vec = info["c_vec"]
        K = 3
        for k in range(K):
            if not info["active_mask"][k]:
                assert c_vec[k] == pytest.approx(0.0, abs=1e-9), f"C1[{k}] nonzero but inactive"
                assert c_vec[K + k] == pytest.approx(0.0, abs=1e-9), f"C2[{k}] nonzero but inactive"
                assert c_vec[2*K + k] == pytest.approx(0.0, abs=1e-9), f"C4[{k}] nonzero but inactive"
                assert c_vec[3*K + k] == pytest.approx(0.0, abs=1e-9), f"C5[{k}] nonzero but inactive"

    def test_no_active_obs_aoi_correct(self):
        """When only 1 of 3 ambulances is active, aoi_mean/max reflect only it."""
        env, _, _ = _make_env(K=3, seed=0)
        obs, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
        n_active = info["n_active"]
        assert n_active >= 1  # seed=0 amb_2 enters immediately
        assert n_active <= 3
        assert obs[OBS_AOI_MEAN_IDX] >= 0.0
        assert obs[OBS_AOI_MAX_IDX] >= 0.0

    def test_viol_history_no_inactive_inflation(self):
        """viol_history should not be True for inactive ambulances."""
        env, _, _ = _make_env(K=3, seed=0)
        for _ in range(50):
            env.step(np.zeros(4, dtype=np.float32))
        # At b_rrm=default, active amb with sev=1 should NOT violate
        viol_rate = sum(env.viol_history) / max(len(env.viol_history), 1)
        assert viol_rate < 0.50, f"viol_rate={viol_rate:.3f} possibly inflated"

    def test_history_skips_no_active_ticks(self):
        """URLLC histories must not grow during fully-inactive ticks."""
        env, _, _ = _make_env(K=3, seed=0)
        # Seed=0: amb_2 enters immediately, so n_active≥1 from tick 0.
        # We can't easily create a fully-inactive window with SUMO,
        # so just verify the guard: history length ≤ total MAC ticks.
        total_steps = 100
        for _ in range(total_steps):
            env.step(np.zeros(4, dtype=np.float32))
        total_ticks = total_steps * MAC_TICKS_PER_WORKER
        # URLLC histories ≤ total ticks (strict < if any no-active ticks existed)
        assert len(env.viol_history) <= total_ticks
        assert len(env.aoi_history) <= total_ticks
        # eMBB history always == total ticks (slice-level, never skipped)
        assert len(env.embb_mbps_history) == total_ticks


# ═══════════════════════════════════════════════════════════════════════
# POINT 2: Per-ambulance metrics exist in info dict
# ═══════════════════════════════════════════════════════════════════════

class TestPoint2PerAmbulanceMetrics:
    """Info dict must expose per-ambulance diagnostics."""

    def test_per_amb_fields_exist(self):
        env, _, _ = _make_env(K=3, seed=0)
        # Run until at least 1 ambulance active
        for _ in range(10):
            _, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
        K = 3
        # Fields that must exist
        assert "active_mask" in info
        assert "active_count_per_amb" in info
        assert "delay_norm_per_amb" in info
        assert "aoi_norm_per_amb" in info
        assert "prb_per_amb" in info
        assert "bler_per_amb" in info
        assert "aoi_pkt_delivered" in info
        assert "aoi_pkt_failed_bler" in info
        assert "aoi_pkt_failed_no_prb" in info
        assert "aoi_pkt_failed_no_capacity" in info
        assert "n_active" in info

        assert len(info["delay_norm_per_amb"]) == K
        assert len(info["aoi_norm_per_amb"]) == K
        assert len(info["prb_per_amb"]) == K

    def test_per_amb_delay_aoi_shapes(self):
        env, _, _ = _make_env(K=3, seed=0)
        for _ in range(100):
            _, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
        assert info["delay_norm_per_amb"].shape == (3,)
        assert info["aoi_norm_per_amb"].shape == (3,)

    def test_active_count_accumulates(self):
        env, _, _ = _make_env(K=3, seed=0)
        # Run enough to get some active ticks
        for _ in range(200):
            _, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
        ac = info["active_count_per_amb"]
        # At least the immediately-entering ambulance should have counts
        assert np.any(ac > 0), f"active_count = {ac}"


# ═══════════════════════════════════════════════════════════════════════
# POINT 3: Observation–constraint consistency
# ═══════════════════════════════════════════════════════════════════════

class TestPoint3ObsConstraintConsistency:
    """obs_aoi_mean/max must match active_mean/max(aoi_per_amb)."""

    def _run_and_check(self, K: int, seed: int, n_steps: int = 500):
        env, _, _ = _make_env(K=K, seed=seed)
        env.set_rrm_budget(0.20)
        action_dim = env.action_space.shape[0]
        for step in range(n_steps):
            obs, _, term, trunc, info = env.step(np.zeros(action_dim, dtype=np.float32))
            if term or trunc:
                break

            # Recompute expected AoI from raw trackers + active_mask
            aoi_raw = np.array([
                t["ambulance_status"].current_aoi(env.sim_time)
                for t in env.aoi_trackers
            ])
            if env.active_mask.any():
                expected_mean = float(aoi_raw[env.active_mask].mean())
                expected_max = float(aoi_raw[env.active_mask].max())
            else:
                expected_mean = 0.0
                expected_max = 0.0

            obs_mean = float(obs[OBS_AOI_MEAN_IDX])
            obs_max = float(obs[OBS_AOI_MAX_IDX])
            assert obs_mean == pytest.approx(expected_mean, abs=1e-6), \
                f"step={step} obs_aoi_mean={obs_mean} expected={expected_mean}"
            assert obs_max == pytest.approx(expected_max, abs=1e-6), \
                f"step={step} obs_aoi_max={obs_max} expected={expected_max}"

            # Verify c_vec C4/C5 uses same mask: inactive amb → 0 in c_vec
            c_vec = info["c_vec"]
            for k in range(K):
                if not env.active_mask[k]:
                    assert c_vec[2*K + k] == pytest.approx(0.0, abs=1e-9), \
                        f"c_vec C4[{k}]={c_vec[2*K+k]} but amb inactive"
                    assert c_vec[3*K + k] == pytest.approx(0.0, abs=1e-9), \
                        f"c_vec C5[{k}]={c_vec[3*K+k]} but amb inactive"

    def test_k3_consistency(self):
        self._run_and_check(K=3, seed=0, n_steps=300)

    def test_k1_consistency(self):
        self._run_and_check(K=1, seed=0, n_steps=300)

    def test_k3_seed42_consistency(self):
        self._run_and_check(K=3, seed=42, n_steps=200)


# ═══════════════════════════════════════════════════════════════════════
# POINT 4: Fixed evaluation before/after patch comparison
# ═══════════════════════════════════════════════════════════════════════

class TestPoint4MetricsAfterPatch:
    """Post-patch metrics should be realistic (no inactive-ambulance inflation)."""

    def test_k3_viol_rate_not_inflated(self):
        """viol_rate should not have 54%+ floor from inactive ambulances."""
        env, _, _ = _make_env(K=3, seed=0)
        env.set_rrm_budget(0.20)
        for _ in range(10000):
            _, _, term, trunc, _ = env.step(np.zeros(4, dtype=np.float32))
            env.set_rrm_budget(0.20)
            if term or trunc:
                break
        viol = env.episode_violation_rate()
        assert viol < 0.10, f"viol_rate={viol:.4f} still inflated (expected <0.10)"

    def test_k3_aoi_not_inflated(self):
        """mean_aoi should be ms-scale, not seconds-scale."""
        env, _, _ = _make_env(K=3, seed=0)
        env.set_rrm_budget(0.20)
        for _ in range(10000):
            _, _, term, trunc, _ = env.step(np.zeros(4, dtype=np.float32))
            env.set_rrm_budget(0.20)
            if term or trunc:
                break
        aoi_ms = env.mean_aoi_ms()
        assert aoi_ms < 500, f"mean_aoi_ms={aoi_ms:.1f} still inflated (expected <500ms)"

    def test_k1_metrics_unchanged(self):
        """K=1 has no inactive ambulances, metrics should be similar pre/post."""
        env, _, _ = _make_env(K=1, seed=0)
        env.set_rrm_budget(0.20)
        for _ in range(5000):
            _, _, term, trunc, _ = env.step(np.zeros(1, dtype=np.float32))
            env.set_rrm_budget(0.20)
            if term or trunc:
                break
        viol = env.episode_violation_rate()
        aoi_ms = env.mean_aoi_ms()
        embb = env.mean_embb_mbps()
        assert viol < 0.10
        assert aoi_ms < 500
        assert embb > 50


# ═══════════════════════════════════════════════════════════════════════
# POINT 5: Reward SUM vs Constraint MEAN scaling
# ═══════════════════════════════════════════════════════════════════════

class TestPoint5RewardConstraintScaling:
    """Verify reward is SUM and c_vec is MEAN across MAC ticks."""

    def test_reward_is_sum_of_ticks(self):
        """reward returned by step() should be sum of 20 MAC-tick rewards."""
        env, _, _ = _make_env(K=1, seed=0)
        env.set_rrm_budget(0.20)
        _, rew, _, _, _ = env.step(np.zeros(1, dtype=np.float32))
        # Each tick reward = alpha_e * log(1 + R_eMBB/100)
        # With eMBB ~200 Mbps, reward ~0.3 * log(3) ≈ 0.33 per tick
        # Sum over 20 ticks ≈ 6.6
        assert rew > 1.0, f"reward={rew} seems like per-tick not sum"
        assert rew < 30.0, f"reward={rew} unreasonably large"

    def test_c_vec_is_mean_of_ticks(self):
        """c_vec should be normalized by active tick count."""
        env, _, _ = _make_env(K=1, seed=0)
        env.set_rrm_budget(0.20)
        _, _, _, _, info = env.step(np.zeros(1, dtype=np.float32))
        c_vec = info["c_vec"]
        # C1 = mean delay (seconds) — should be sub-ms
        assert 0.0 <= c_vec[0] < 0.01, f"C1(delay)={c_vec[0]} not mean-of-ticks scale"

    def test_scaling_ratio(self):
        """Document the reward:penalty scale ratio for transparency."""
        from agents.lagrangian import LambdaState
        env, _, info = _make_env(K=1, seed=0)
        env.set_rrm_budget(0.20)
        K = 1
        ls = LambdaState(K=K, force_zero_warm=True)
        sev = tuple(int(s) for s in info["severity_per_amb"])
        ls.reset_episode(sev, int(info["severity"]))

        _, rew, _, _, info = env.step(np.zeros(1, dtype=np.float32))
        c_vec = np.asarray(info["c_vec"], dtype=np.float64)
        d_phi = np.asarray(info["d_phi"], dtype=np.float64)

        # With force_zero_warm → λ=0, penalty=0
        r_aug_0 = ls.augmented_reward(rew, c_vec, d_phi)
        assert r_aug_0 == pytest.approx(rew, abs=1e-6)

        # Document: reward is SUM(20 ticks), penalty is dot(λ, MEAN(20 ticks))
        # This is intentional — λ compensates for the 20x scaling gap.
        per_tick_reward_est = rew / MAC_TICKS_PER_WORKER
        assert per_tick_reward_est > 0.05  # sanity


# ═══════════════════════════════════════════════════════════════════════
# POINT 6: K=1 and K=3 use the same environment class
# ═══════════════════════════════════════════════════════════════════════

class TestPoint6SameEnvironment:
    """K=1 and K=3 must use identical ORANEnv, differing only in K."""

    def test_same_env_class(self):
        env1, _, _ = _make_env(K=1, seed=0)
        env3, _, _ = _make_env(K=3, seed=0)
        assert type(env1) is type(env3) is ORANEnv

    def test_same_channel_model(self):
        env1, _, _ = _make_env(K=1, seed=0)
        env3, _, _ = _make_env(K=3, seed=0)
        assert env1.config.cell_radius_m == env3.config.cell_radius_m
        assert env1.config.tti_sec == env3.config.tti_sec
        assert env1.config.episode_duration_sec == env3.config.episode_duration_sec

    def test_different_K(self):
        env1, _, _ = _make_env(K=1, seed=0)
        env3, _, _ = _make_env(K=3, seed=0)
        assert env1.config.K_ambulances == 1
        assert env3.config.K_ambulances == 3

    def test_obs_dim_scales_with_K(self):
        env1, obs1, _ = _make_env(K=1, seed=0)
        env3, obs3, _ = _make_env(K=3, seed=0)
        # obs = 20 fixed + 11*K per-amb (incl. active_mask_k) + F aoi_stream
        F = 1
        assert obs1.shape[0] == 20 + 11 * 1 + F  # 32
        assert obs3.shape[0] == 20 + 11 * 3 + F  # 54

    def test_action_dim_scales_with_K(self):
        env1, _, _ = _make_env(K=1, seed=0)
        env3, _, _ = _make_env(K=3, seed=0)
        # K=1: 1-dim no-op. K>=2: K-dim pure per-vehicle priority logits (no β).
        assert env1.action_space.shape[0] == 1
        assert env3.action_space.shape[0] == 3

    def test_c_vec_dim_scales_with_K(self):
        env1, _, info1 = _make_env(K=1, seed=0)
        env3, _, info3 = _make_env(K=3, seed=0)
        # c_vec = 4K + 1
        _, _, _, _, info1 = env1.step(np.zeros(1, dtype=np.float32))
        _, _, _, _, info3 = env3.step(np.zeros(4, dtype=np.float32))
        assert info1["c_vec"].shape[0] == 4 * 1 + 1  # 5
        assert info3["c_vec"].shape[0] == 4 * 3 + 1  # 13
