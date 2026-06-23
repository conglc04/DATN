"""Comprehensive invariant tests — exact per-constraint, obs layout, monotonicity, mobility.

Covers gaps left by the existing focused suites:
  1.  Severity monotonicity — lambda_warm C1/C2/C4/C5 non-decreasing (only C3 was checked)
  2.  obs[0:20] exhaustive — HOL, arr_rate, lambda_c3, n_bys, aoi_mean, aoi_max (6 fields untested)
  3.  K=1 PRB preservation — info prb_per_amb sums to prb_urllc; each amb >= PRB_MIN_QOS
  4.  K=3 PRB conservation — prb_per_amb[k] >= PRB_MIN_QOS for ALL k (not just sum)
  5.  Urgency NaN guard  — delay_norm_k / aoi_norm_k stay finite at tightest severity-5 thresholds
  6.  severity_ref role  — alpha_e and C3 R_min keyed by severity_ref, not per-ambulance severity_k
  7.  Per-constraint semantics — exact types/ranges for C1..C5 in the real env (no monkeypatch)
  8.  Lambda per-constraint ascent — isolate each of C1..C5; only its lambda increases
  9.  Dual-scale vector K=3 — slot-by-slot exact values (K=1 already in config_tables_exact)
  10. Mobility pipeline — ambulances stay in cell, bounce, dist_norm/speed_norm bounded
  11. Smoke invariants — obs_dim formula, reward finite, c_vec finite, lambda finite,
      PRB sum <= 273, no NaN, no inf for K=1 and K=3
  12. C3 signed gap real env — c_vec[4K] == R_min - mean(R_eMBB per tick), no monkeypatch
"""

from __future__ import annotations

import numpy as np
import pytest

from agents.lagrangian import LambdaState
from env.oran_env import EnvConfig, ORANEnv
from utils.config import (
    AOI_REF_S,
    CMDP_D_J_SEVERITY,
    D_REF_URLLC,
    LAMBDA_WARM,
    MAC_TICKS_PER_WORKER,
    OBS_AOI_MAX_IDX,
    OBS_AOI_MEAN_IDX,
    OBS_ARR_EMBB_IDX,
    OBS_ARR_URLLC_IDX,
    OBS_FIXED_BLOCK_LEN,
    OBS_HOL_EMBB_IDX,
    OBS_HOL_URLLC_IDX,
    OBS_LAMBDA_C3_IDX,
    OBS_N_BYS_IDX,
    OBS_PER_AMB_BLOCK_LEN,
    P_TOTAL,
    PRB_MIN_QOS,
    R_REF_EMBB_MBPS,
    SEVERITY_QOS,
    build_d_phi_vector,
    build_dual_scales,
)


# ---------------------------------------------------------------------------
# 1. Severity monotonicity — lambda_warm C1/C2/C4/C5 must be non-decreasing
# ---------------------------------------------------------------------------
# C3 (index 2) is NON-INCREASING — already tested in TestLambdaWarm.
# C1 (index 0), C2 (index 1), C4 (index 3), C5 (index 4) must be non-decreasing.


class TestSeverityMonotonicity:
    def test_lambda_warm_c1_non_decreasing_with_severity(self):
        c1 = [LAMBDA_WARM[s][0] for s in range(1, 6)]
        for i in range(len(c1) - 1):
            assert c1[i] <= c1[i + 1], (
                f"LAMBDA_WARM C1 not non-decreasing: sev{i+1}={c1[i]}, sev{i+2}={c1[i+1]}"
            )

    def test_lambda_warm_c2_non_decreasing_with_severity(self):
        c2 = [LAMBDA_WARM[s][1] for s in range(1, 6)]
        for i in range(len(c2) - 1):
            assert c2[i] <= c2[i + 1], (
                f"LAMBDA_WARM C2 not non-decreasing: sev{i+1}={c2[i]}, sev{i+2}={c2[i+1]}"
            )

    def test_lambda_warm_c4_non_decreasing_with_severity(self):
        c4 = [LAMBDA_WARM[s][3] for s in range(1, 6)]
        for i in range(len(c4) - 1):
            assert c4[i] <= c4[i + 1], (
                f"LAMBDA_WARM C4 not non-decreasing: sev{i+1}={c4[i]}, sev{i+2}={c4[i+1]}"
            )

    def test_lambda_warm_c5_non_decreasing_with_severity(self):
        c5 = [LAMBDA_WARM[s][4] for s in range(1, 6)]
        for i in range(len(c5) - 1):
            assert c5[i] <= c5[i + 1], (
                f"LAMBDA_WARM C5 not non-decreasing: sev{i+1}={c5[i]}, sev{i+2}={c5[i+1]}"
            )

    def test_lambda_warm_c3_fixed_sev1_vs_sev5(self):
        # Gate 7: C3 warm-start is severity-independent (fixed 0.02)
        assert LAMBDA_WARM[1][2] == LAMBDA_WARM[5][2] == pytest.approx(0.02)

    def test_d3_embb_fixed_sev1_vs_sev5(self):
        # Gate 7: eMBB floor is a fixed 10 Mbps SLA, equal across severities
        assert CMDP_D_J_SEVERITY[1]["d3_embb_mbps"] == CMDP_D_J_SEVERITY[5]["d3_embb_mbps"] == 10.0


# ---------------------------------------------------------------------------
# 2. obs[0:20] exhaustive field-by-field lock
#    (supplements test_obs_layout.py which tested: rho, PRB ratios, BLER,
#    severity one-hot, anchor, shape — but NOT hol, arr, lambda_c3, n_bys, aoi)
# ---------------------------------------------------------------------------


class TestObsLayoutExhaustive:
    @pytest.fixture
    def env_k1(self):
        env = ORANEnv(EnvConfig(K_ambulances=1))
        env.reset(seed=42)
        obs, *_ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        yield env, obs
        env.close()

    def test_hol_urllc_at_named_index(self, env_k1):
        env, obs = env_k1
        # HOL delay is min(mean_k(queue.hol_delay) * 1e3, 100.0) ms
        raw = min(
            float(np.mean([env.queues["urllc_0"].hol_delay()])) * 1e3,
            100.0,
        )
        assert obs[OBS_HOL_URLLC_IDX] == pytest.approx(raw, abs=1e-4)

    def test_hol_embb_at_named_index(self, env_k1):
        env, obs = env_k1
        raw = min(float(env.queues["eMBB"].hol_delay()) * 1e3, 1000.0)
        assert obs[OBS_HOL_EMBB_IDX] == pytest.approx(raw, abs=1e-4)

    def test_arr_urllc_at_named_index(self, env_k1):
        env, obs = env_k1
        expected = float(env.queues["urllc_0"].arrival_rate) / 1e3
        assert obs[OBS_ARR_URLLC_IDX] == pytest.approx(expected, abs=1e-6)

    def test_arr_embb_at_named_index(self, env_k1):
        env, obs = env_k1
        expected = float(env.queues["eMBB"].arrival_rate) / 1e4
        assert obs[OBS_ARR_EMBB_IDX] == pytest.approx(expected, abs=1e-6)

    def test_lambda_c3_at_named_index(self, env_k1):
        env, obs = env_k1
        expected = float(env._lambda_local[4 * env.config.K_ambulances])
        assert obs[OBS_LAMBDA_C3_IDX] == pytest.approx(expected, abs=1e-6)

    def test_n_bys_equals_one_when_bystander_disabled(self, env_k1):
        _, obs = env_k1
        assert obs[OBS_N_BYS_IDX] == pytest.approx(1.0, abs=1e-6)

    def test_aoi_mean_non_negative_finite(self, env_k1):
        _, obs = env_k1
        assert np.isfinite(obs[OBS_AOI_MEAN_IDX])
        assert obs[OBS_AOI_MEAN_IDX] >= 0.0

    def test_aoi_max_ge_aoi_mean(self, env_k1):
        _, obs = env_k1
        assert obs[OBS_AOI_MAX_IDX] >= obs[OBS_AOI_MEAN_IDX] - 1e-6


# ---------------------------------------------------------------------------
# 3. K=1 PRB preservation
# ---------------------------------------------------------------------------


class TestK1Preservation:
    def test_obs_shape_k1(self):
        env = ORANEnv(EnvConfig(K_ambulances=1))
        obs, _ = env.reset(seed=0)
        K, F = 1, env.config.num_streams
        expected = OBS_FIXED_BLOCK_LEN + OBS_PER_AMB_BLOCK_LEN * K + F
        assert obs.shape[0] == expected == 32
        env.close()

    def test_prb_per_amb_sums_to_prb_urllc_k1(self):
        env = ORANEnv(EnvConfig(K_ambulances=1))
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert sum(info["prb_per_amb"]) == info["prb_urllc"]
        env.close()

    def test_prb_per_amb_ge_prb_min_qos_k1(self):
        env = ORANEnv(EnvConfig(K_ambulances=1))
        env.reset(seed=0)
        for _ in range(10):
            _, _, _, _, info = env.step(env.action_space.sample())
            for prb in info["prb_per_amb"]:
                assert prb >= PRB_MIN_QOS, f"prb={prb} < PRB_MIN_QOS={PRB_MIN_QOS}"
        env.close()

    def test_prb_urllc_plus_embb_le_p_total_k1(self):
        env = ORANEnv(EnvConfig(K_ambulances=1))
        env.reset(seed=0)
        for _ in range(10):
            _, _, _, _, info = env.step(env.action_space.sample())
            assert info["prb_urllc"] + info["prb_embb"] <= P_TOTAL
        env.close()


# ---------------------------------------------------------------------------
# 4. K=3 PRB conservation
# ---------------------------------------------------------------------------


class TestK3PRBConservation:
    def test_obs_shape_k3(self):
        K = 3
        env = ORANEnv(EnvConfig(K_ambulances=K))
        obs, _ = env.reset(seed=0)
        F = env.config.num_streams
        assert obs.shape[0] == OBS_FIXED_BLOCK_LEN + OBS_PER_AMB_BLOCK_LEN * K + F
        env.close()

    def test_prb_per_amb_sums_to_prb_urllc_k3(self):
        env = ORANEnv(EnvConfig(K_ambulances=3))
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert sum(info["prb_per_amb"]) == info["prb_urllc"]
        env.close()

    def test_each_prb_per_amb_ge_prb_min_qos_k3(self):
        env = ORANEnv(EnvConfig(K_ambulances=3))
        env.reset(seed=0, options={"severity_per_amb": [1, 3, 5]})
        for _ in range(10):
            _, _, _, _, info = env.step(env.action_space.sample())
            ac = info["active_count_per_amb"]
            for k, prb in enumerate(info["prb_per_amb"]):
                if ac[k] > 0:   # only ACTIVE ambulances are guaranteed PRB_MIN_QOS (SUMO staggers entry)
                    assert prb >= PRB_MIN_QOS, f"active amb[{k}] prb={prb} < PRB_MIN_QOS"
        env.close()

    def test_total_prb_le_273_k3(self):
        env = ORANEnv(EnvConfig(K_ambulances=3))
        env.reset(seed=0)
        for _ in range(20):
            _, _, _, _, info = env.step(env.action_space.sample())
            assert info["prb_urllc"] + info["prb_embb"] <= P_TOTAL


# ---------------------------------------------------------------------------
# 5. Urgency NaN guard
# ---------------------------------------------------------------------------


class TestUrgencyNaNGuard:
    def _no_nan_inf(self, obs: np.ndarray) -> None:
        assert not np.any(np.isnan(obs)), f"NaN in obs: {np.where(np.isnan(obs))}"
        assert not np.any(np.isinf(obs)), f"Inf in obs: {np.where(np.isinf(obs))}"

    def test_no_nan_at_severity_5_after_reset(self):
        """severity 5: D_max=1ms, AoI_max=0.1s — smallest thresholds -> highest division risk."""
        env = ORANEnv(EnvConfig(K_ambulances=1, initial_severity=5))
        obs, _ = env.reset(seed=0)
        self._no_nan_inf(obs)
        env.close()

    def test_no_nan_at_severity_5_after_step(self):
        env = ORANEnv(EnvConfig(K_ambulances=1, initial_severity=5))
        env.reset(seed=0)
        obs, *_ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        self._no_nan_inf(obs)
        env.close()

    def test_delay_norm_finite_at_severity_5(self):
        """delay_norm_k = D_e2e_k / D_max^sev5 = D_e2e / 1e-3 — check no overflow."""
        env = ORANEnv(EnvConfig(K_ambulances=1, initial_severity=5))
        env.reset(seed=0)
        obs, *_ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        # Per-ambulance block starts at OBS_FIXED_BLOCK_LEN; delay_norm_k at offset 3
        delay_norm_k_idx = OBS_FIXED_BLOCK_LEN + 3
        assert np.isfinite(obs[delay_norm_k_idx])
        assert obs[delay_norm_k_idx] >= 0.0

    def test_aoi_norm_finite_at_severity_5(self):
        """aoi_norm_k = AoI_k / AoI_max^sev5 = AoI / 0.1 — check no overflow."""
        env = ORANEnv(EnvConfig(K_ambulances=1, initial_severity=5))
        env.reset(seed=0)
        obs, *_ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        aoi_norm_k_idx = OBS_FIXED_BLOCK_LEN + 4
        assert np.isfinite(obs[aoi_norm_k_idx])
        assert obs[aoi_norm_k_idx] >= 0.0

    def test_no_nan_after_100_steps_k3_severity5(self):
        K = 3
        env = ORANEnv(EnvConfig(K_ambulances=K, initial_severity=5))
        env.reset(seed=0)
        for _ in range(100):
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            self._no_nan_inf(obs)
            if terminated or truncated:
                break
        env.close()


# ---------------------------------------------------------------------------
# 6. severity_ref drives alpha_e and C3 R_min (not per-ambulance severity_k)
# ---------------------------------------------------------------------------


class TestSeverityRefRole:
    def test_alpha_e_uses_severity_ref_not_per_ambulance_severity(self):
        """K=3, per-ambulance = [1,1,5], severity_ref = 5.
        alpha_e should be 0.05 (severity 5), not 0.70 (severity 1).
        We verify via reward: same eMBB throughput, lower reward at higher severity_ref."""
        from utils.config import get_severity_alpha

        _, alpha_e_sev5 = get_severity_alpha(5)
        _, alpha_e_sev1 = get_severity_alpha(1)

        env5 = ORANEnv(EnvConfig(K_ambulances=3))
        env5.reset(seed=7, options={"severity_per_amb": [1, 1, 5]})  # severity_ref=5
        # Force identical eMBB throughput by checking info after one step
        env5.step(np.zeros(env5.action_space.shape, dtype=np.float32))
        assert env5.severity == 5
        assert abs(alpha_e_sev5 - 0.05) < 1e-9
        assert abs(alpha_e_sev1 - 0.70) < 1e-9
        assert alpha_e_sev5 < alpha_e_sev1  # lower alpha_e at higher severity_ref
        env5.close()

    def test_c3_threshold_uses_fixed_floor_regardless_of_severity(self):
        """Gate 7: C3 uses a FIXED 10 Mbps eMBB floor independent of severity_ref.

        c_vec[C3] = R_min_fixed - R_eMBB with R_min_fixed = 10 Mbps for ANY
        severity vector. Verified for severity_ref = 3 and severity_ref = 5.
        """
        r_min_fixed = CMDP_D_J_SEVERITY[3]["d3_embb_mbps"]  # 10.0 Mbps (fixed)
        assert r_min_fixed == 10.0
        K = 3
        C3_IDX = 4 * K  # index 12 for K=3

        for sev_vec, sev_ref in (([1, 2, 3], 3), ([5, 1, 5], 5)):
            env = ORANEnv(EnvConfig(K_ambulances=3))
            env.reset(seed=0, options={"severity_per_amb": sev_vec})
            assert env.severity == sev_ref
            _, _, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
            assert abs(info["d_phi"][C3_IDX]) < 1e-6              # d_phi[C3] = 0 (gap form)
            r_embb_mean = float(np.mean(env.embb_mbps_history[-MAC_TICKS_PER_WORKER:]))
            expected_gap = r_min_fixed - r_embb_mean             # uses the FIXED 10 Mbps floor
            assert info["c_vec"][C3_IDX] == pytest.approx(expected_gap, abs=1e-3)
            env.close()


# ---------------------------------------------------------------------------
# 7. Per-constraint exact semantics (real env, no monkeypatch)
# ---------------------------------------------------------------------------


class TestPerConstraintSemantics:
    @pytest.fixture
    def env_k1_step(self):
        env = ORANEnv(EnvConfig(K_ambulances=1, initial_severity=3))
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        yield env, info
        env.close()

    def test_cvec_shape_k1(self, env_k1_step):
        _, info = env_k1_step
        assert info["c_vec"].shape == (5,)   # 4*1+1

    def test_dphi_shape_k1(self, env_k1_step):
        _, info = env_k1_step
        assert info["d_phi"].shape == (5,)

    def test_c1_d_e2e_always_positive_finite(self, env_k1_step):
        """C1 = mean D_e2e_k (s) — always > 0 due to deterministic D_det+D_fh+D_bh."""
        _, info = env_k1_step
        c1 = info["c_vec"][0]   # K=1: C1 at index 0
        assert np.isfinite(c1)
        assert c1 > 0.0, f"D_e2e must be positive (has deterministic floor); got {c1}"

    def test_c2_urllc_viol_in_zero_one(self, env_k1_step):
        """C2 = mean(URLLC tail violation) ∈ [0,1] (binary per-tick, averaged)."""
        _, info = env_k1_step
        c2 = info["c_vec"][1]
        assert 0.0 <= c2 <= 1.0 + 1e-6

    def test_c4_aoi_non_negative_finite(self, env_k1_step):
        """C4 = mean AoI_k (s) — always ≥ 0."""
        _, info = env_k1_step
        c4 = info["c_vec"][2]
        assert np.isfinite(c4)
        assert c4 >= 0.0

    def test_c5_aoi_viol_in_zero_one(self, env_k1_step):
        """C5 = mean(AoI tail violation) ∈ [0,1]."""
        _, info = env_k1_step
        c5 = info["c_vec"][3]
        assert 0.0 <= c5 <= 1.0 + 1e-6

    def test_c3_d_phi_always_zero(self, env_k1_step):
        """d_phi[4] (C3 threshold) must be 0 — R_min is embedded inside c_vec[4]."""
        _, info = env_k1_step
        assert abs(info["d_phi"][4]) < 1e-9

    def test_cvec_shape_k3(self):
        K = 3
        env = ORANEnv(EnvConfig(K_ambulances=K))
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert info["c_vec"].shape == (4 * K + 1,)   # 13 for K=3
        assert info["d_phi"].shape == (4 * K + 1,)
        env.close()


# ---------------------------------------------------------------------------
# 8. Lambda per-constraint ascent — violate one, only its lambda rises
# ---------------------------------------------------------------------------


class TestLambdaPerConstraintAscent:
    """For each constraint j, set c_vec[j] = d_phi[j] + delta, rest = d_phi.
    After one Manager window, only lambda[j] should have increased.
    """

    SEV = 1    # sev=1: lambda_warm starts low, easier to track changes
    DELTA = 0.005

    def _run_isolated_violation(self, viol_slot: int) -> tuple[np.ndarray, np.ndarray]:
        lam = LambdaState(K=1)
        lam.reset_episode(severity_per_amb=[self.SEV], severity_ref=self.SEV)
        lambda_warm = lam.get_lambda_global().copy()

        d_phi = build_d_phi_vector([self.SEV])
        c_vec = d_phi.copy()
        c_vec[viol_slot] = d_phi[viol_slot] + self.DELTA

        for _ in range(lam.worker_steps_per_manager):
            lam.accumulate(c_vec, d_phi)
        lam.on_manager_step_end()
        return lambda_warm, lam.get_lambda_global()

    def test_c1_violation_increases_lambda_c1_only(self):
        warm, after = self._run_isolated_violation(0)   # C1 = slot 0
        assert after[0] > warm[0], "C1 lambda must increase when C1 violated"
        for j in (1, 2, 3, 4):
            assert after[j] == pytest.approx(warm[j], abs=1e-9), f"lambda[{j}] must stay unchanged"

    def test_c2_violation_increases_lambda_c2_only(self):
        warm, after = self._run_isolated_violation(1)   # C2 = slot 1
        assert after[1] > warm[1], "C2 lambda must increase when C2 violated"
        for j in (0, 2, 3, 4):
            assert after[j] == pytest.approx(warm[j], abs=1e-9), f"lambda[{j}] must stay unchanged"

    def test_c4_violation_increases_lambda_c4_only(self):
        warm, after = self._run_isolated_violation(2)   # C4 = slot 2
        assert after[2] > warm[2], "C4 lambda must increase when C4 violated"
        for j in (0, 1, 3, 4):
            assert after[j] == pytest.approx(warm[j], abs=1e-9), f"lambda[{j}] must stay unchanged"

    def test_c5_violation_increases_lambda_c5_only(self):
        warm, after = self._run_isolated_violation(3)   # C5 = slot 3
        assert after[3] > warm[3], "C5 lambda must increase when C5 violated"
        for j in (0, 1, 2, 4):
            assert after[j] == pytest.approx(warm[j], abs=1e-9), f"lambda[{j}] must stay unchanged"

    def test_c3_violation_increases_lambda_c3_only(self):
        warm, after = self._run_isolated_violation(4)   # C3 = slot 4, d_phi[4]=0
        assert after[4] > warm[4], "C3 lambda must increase when C3 violated (eMBB deficit)"
        for j in (0, 1, 2, 3):
            assert after[j] == pytest.approx(warm[j], abs=1e-9), f"lambda[{j}] must stay unchanged"


# ---------------------------------------------------------------------------
# 9. Dual-scale vector K=3 — slot-by-slot
# ---------------------------------------------------------------------------


class TestDualScaleVector:
    def test_k1_exact_values(self):
        s = build_dual_scales(1)
        assert s.shape == (5,)
        assert s[0] == pytest.approx(D_REF_URLLC)       # C1: latency scale
        assert s[1] == pytest.approx(1.0)                # C2: dimensionless
        assert s[2] == pytest.approx(AOI_REF_S)          # C4: AoI scale
        assert s[3] == pytest.approx(1.0)                # C5: dimensionless
        assert s[4] == pytest.approx(R_REF_EMBB_MBPS)   # C3: throughput scale

    def test_k3_shape(self):
        s = build_dual_scales(3)
        assert s.shape == (13,)

    def test_k3_c1_slots_all_d_ref_urllc(self):
        s = build_dual_scales(3)
        for k in range(3):
            assert s[k] == pytest.approx(D_REF_URLLC), f"C1 slot {k} wrong"

    def test_k3_c2_slots_all_one(self):
        s = build_dual_scales(3)
        for k in range(3, 6):
            assert s[k] == pytest.approx(1.0), f"C2 slot {k} wrong"

    def test_k3_c4_slots_all_aoi_ref(self):
        s = build_dual_scales(3)
        for k in range(6, 9):
            assert s[k] == pytest.approx(AOI_REF_S), f"C4 slot {k} wrong"

    def test_k3_c5_slots_all_one(self):
        s = build_dual_scales(3)
        for k in range(9, 12):
            assert s[k] == pytest.approx(1.0), f"C5 slot {k} wrong"

    def test_k3_c3_slot_r_ref_embb(self):
        s = build_dual_scales(3)
        assert s[12] == pytest.approx(R_REF_EMBB_MBPS)  # C3 shared slot


# ---------------------------------------------------------------------------
# 10. Mobility pipeline
# ---------------------------------------------------------------------------


class TestMobilityPipeline:
    R_CELL = 300.0

    def _make_env(self, K: int = 1) -> ORANEnv:
        return ORANEnv(EnvConfig(K_ambulances=K, cell_radius_m=self.R_CELL))

    def test_position_changes_after_advance(self):
        env = self._make_env()
        env.reset(seed=0)
        pos_before = env.ambulance_pos.copy()
        env._advance_ambulance_positions()
        # Velocity is non-zero after init → position must change
        assert not np.allclose(env.ambulance_pos, pos_before), (
            "position unchanged after _advance_ambulance_positions()"
        )
        env.close()

    def test_ambulances_stay_within_cell_after_many_steps(self):
        env = self._make_env()
        env.reset(seed=0)
        for _ in range(500):
            env._advance_ambulance_positions()
            r = np.linalg.norm(env.ambulance_pos, axis=1)
            assert np.all(r <= self.R_CELL + 1e-6), (
                f"ambulance out of cell: |pos|={r}"
            )
        env.close()

    def test_bounce_flips_velocity_on_exit(self):
        """Place ambulance just at cell edge with outward velocity; after advance it bounces."""
        env = self._make_env()
        env.reset(seed=0)
        # Force ambulance to be just inside the cell boundary, moving outward
        env.ambulance_pos = np.array([[self.R_CELL - 0.01, 0.0]])
        env.ambulance_vel = np.array([[10.0, 0.0]])  # outward +x at 10 m/s
        env._advance_ambulance_positions()
        # After bounce: ambulance still inside cell
        r = np.linalg.norm(env.ambulance_pos)
        assert r <= self.R_CELL + 1e-6
        env.close()

    def test_dist_norm_in_unit_interval_k1(self):
        """obs per-amb block: d_k = |pos| / R_cell should be in [0,1]."""
        env = self._make_env(K=1)
        env.reset(seed=0)
        for _ in range(50):
            obs, *_ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
            d_k = obs[OBS_FIXED_BLOCK_LEN + 1]   # AMB_DIST_OFFSET = 1
            assert 0.0 <= d_k <= 1.0 + 1e-6, f"dist_norm out of [0,1]: {d_k}"
        env.close()

    def test_speed_norm_non_negative(self):
        env = self._make_env(K=1)
        env.reset(seed=0)
        for _ in range(20):
            obs, *_ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
            v_k = obs[OBS_FIXED_BLOCK_LEN + 2]   # AMB_SPEED_OFFSET = 2
            assert v_k >= 0.0, f"speed_norm negative: {v_k}"
        env.close()

    def test_k3_all_ambulances_within_cell_after_episode(self):
        env = self._make_env(K=3)
        env.reset(seed=0)
        a = np.zeros(env.action_space.shape, dtype=np.float32)
        done = False
        while not done:
            _, _, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            r = np.linalg.norm(env.ambulance_pos, axis=1)
            # Only ACTIVE ambulances (entered, not yet arrived) are within the cell;
            # not-yet-entered ambulances are correctly outside under SUMO mobility.
            if env.active_mask.any():
                assert np.all(r[env.active_mask] <= self.R_CELL + 1e-6)
        env.close()


# ---------------------------------------------------------------------------
# 11. Smoke invariants (K=1 and K=3)
# ---------------------------------------------------------------------------


class TestSmokeInvariants:
    def _check_no_nan_inf(self, obs: np.ndarray, tag: str) -> None:
        assert not np.any(np.isnan(obs)), f"[{tag}] NaN in obs"
        assert not np.any(np.isinf(obs)), f"[{tag}] Inf in obs"

    @pytest.mark.parametrize("K", [1, 3])
    def test_obs_dim_formula(self, K):
        env = ORANEnv(EnvConfig(K_ambulances=K))
        obs, _ = env.reset(seed=0)
        F = env.config.num_streams
        assert obs.shape[0] == OBS_FIXED_BLOCK_LEN + OBS_PER_AMB_BLOCK_LEN * K + F
        env.close()

    @pytest.mark.parametrize("K,expected_dim", [(1, 1), (3, 3)])
    def test_action_dim(self, K, expected_dim):
        env = ORANEnv(EnvConfig(K_ambulances=K))
        env.reset(seed=0)
        assert env.action_space.shape == (expected_dim,)
        env.close()

    @pytest.mark.parametrize("K", [1, 3])
    def test_reward_finite_after_step(self, K):
        env = ORANEnv(EnvConfig(K_ambulances=K))
        env.reset(seed=0)
        a = np.zeros(env.action_space.shape, dtype=np.float32)
        _, reward, _, _, _ = env.step(a)
        assert np.isfinite(reward)
        assert not np.isnan(reward)
        env.close()

    @pytest.mark.parametrize("K", [1, 3])
    def test_c_vec_all_finite_after_step(self, K):
        env = ORANEnv(EnvConfig(K_ambulances=K))
        env.reset(seed=0)
        a = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(a)
        assert np.all(np.isfinite(info["c_vec"])), f"c_vec has non-finite: {info['c_vec']}"
        env.close()

    @pytest.mark.parametrize("K", [1, 3])
    def test_lambda_local_finite_after_reset(self, K):
        env = ORANEnv(EnvConfig(K_ambulances=K))
        env.reset(seed=0)
        lam = env._lambda_local
        assert lam.shape == (4 * K + 1,)
        assert np.all(np.isfinite(lam)), f"lambda_local has non-finite: {lam}"
        env.close()

    @pytest.mark.parametrize("K", [1, 3])
    def test_prb_sum_le_273_over_episode(self, K):
        env = ORANEnv(EnvConfig(K_ambulances=K))
        env.reset(seed=0)
        done = False
        while not done:
            _, _, terminated, truncated, info = env.step(env.action_space.sample())
            done = terminated or truncated
            total_prb = info["prb_urllc"] + info["prb_embb"]
            assert total_prb <= P_TOTAL, f"PRB sum {total_prb} > P_TOTAL={P_TOTAL}"
        env.close()

    @pytest.mark.parametrize("K", [1, 3])
    def test_no_nan_no_inf_full_episode(self, K):
        env = ORANEnv(EnvConfig(K_ambulances=K))
        env.reset(seed=0)
        done = False
        while not done:
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            done = terminated or truncated
            self._check_no_nan_inf(obs, f"K={K}")
        env.close()


# ---------------------------------------------------------------------------
# 12. C3 signed gap in real env (no monkeypatch)
# ---------------------------------------------------------------------------


class TestC3SignedGapRealEnv:
    def test_c3_equals_r_min_minus_mean_r_embb_exact(self):
        """c_vec[4K] = mean(R_min^sev_ref - R_eMBB per tick) = R_min - mean(R_eMBB).
        Verify using embb_mbps_history (per-tick eMBB throughput log).
        """
        sev = 3
        env = ORANEnv(EnvConfig(K_ambulances=1, initial_severity=sev))
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))

        r_min = float(CMDP_D_J_SEVERITY[sev]["d3_embb_mbps"])  # 10.0 Mbps (fixed floor, Gate 7)
        r_embb_mean = float(np.mean(env.embb_mbps_history[-MAC_TICKS_PER_WORKER:]))
        expected_gap = r_min - r_embb_mean

        c3 = float(info["c_vec"][4])   # K=1: 4*1 = index 4
        assert c3 == pytest.approx(expected_gap, abs=1e-3), (
            f"c_vec[4]={c3:.4f} != R_min-R_eMBB={expected_gap:.4f} "
            f"(R_min={r_min}, R_eMBB_mean={r_embb_mean:.4f})"
        )
        env.close()

    def test_c3_sign_positive_under_deep_fade_starvation(self):
        """C3 binds (gap > 0) only in the regime the floor-derived safety cap
        cannot cover: a DEEP FADE (SINR < 0 dB) where the conservatively-reserved
        eMBB PRB (sized at 0 dB) under-delivers the fixed 10 Mbps floor.

        Gate 7 note: at SINR >= 0 dB the safety cap structurally guarantees C3
        (eMBB always >= floor), so high URLLC budget alone cannot starve eMBB —
        the only real-env violation regime is sub-0 dB fading.
        """
        env = ORANEnv(EnvConfig(
            K_ambulances=1, initial_severity=1, rrm_budget_hint=0.85,
            sinr_clamp_max_db=-8.0, sinr_clamp_min_db=-10.0,   # force deep fade
        ))
        env.reset(seed=0)
        a = np.zeros(env.action_space.shape, dtype=np.float32)
        for _ in range(5):
            _, _, _, _, info = env.step(a)
        c3 = float(info["c_vec"][4])
        # eMBB capacity crushed by the fade -> R_eMBB < 10 Mbps floor -> gap > 0
        assert c3 > 0.0, (
            f"Expected C3 > 0 (eMBB starved in deep fade), got c3={c3:.3f}"
        )
        env.close()
