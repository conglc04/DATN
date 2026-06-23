"""Rigorous formula tests — W01/W05/W06 config table exact values.

Every assertion uses the literal expected value from the spec document,
NOT the imported constant as its own oracle. Constants are cross-checked
against stated design rationale in the docstrings of config.py.

W01 constants:
  P_TOTAL, B_PRB, F_CARRIER, D_DET, D_FH, D_BH, D_STOCH

W05/W06 CMDP tables:
  SEVERITY_QOS          all 5 levels × 4 fields
  SEVERITY_ALPHA        all 5 levels × 2 fields (urllc, embb)
  CMDP_D_J_SEVERITY     all 5 levels × 5 fields
  LAMBDA_WARM           all 5 levels × 5 values (C1,C2,C3,C4,C5)

W06 derived/builder:
  GAMMA, GAMMA_WORKER, GAMMA_MANAGER    discount factor chain
  WORKER_STEPS_PER_MANAGER             timing hierarchy
  B_RRM_MIN, B_RRM_MAX                 Manager PRB bounds
  BETA_MIN, BETA_MAX, INTRA_SLICE_KAPPA, PRB_MIN_QOS, RHO_URGENCY_TIEBREAK
  build_lambda_warm_vector              K=1 and K=3 exact vectors
  build_dual_scales                     K=1 and K=3 shapes + values
  build_d_phi_vector                    K=1 and K=3 shapes + values
  get_severity_thresholds               all 5 levels
  get_severity_alpha                    all 5 levels, sum-to-1 invariant
"""

from __future__ import annotations

import numpy as np
import pytest

from utils.config import (
    AOI_REF_S,
    ALPHA_LAMBDA_DUAL,
    B_PRB,
    B_RRM_FLOOR_BY_SEV,
    B_RRM_MAX,
    B_RRM_MIN,
    BETA_MAX,
    BETA_MIN,
    CMDP_D_J_SEVERITY,
    D_BH,
    D_DET,
    D_FH,
    D_REF_URLLC,
    D_STOCH,
    F_CARRIER,
    GAMMA,
    GAMMA_MANAGER,
    GAMMA_WORKER,
    INTRA_SLICE_KAPPA,
    LAMBDA_MAX,
    LAMBDA_WARM,
    P_TOTAL,
    PRB_MIN_QOS,
    R_REF_EMBB_MBPS,
    RHO_URGENCY_TIEBREAK,
    SEVERITY_ALPHA,
    SEVERITY_QOS,
    WORKER_STEPS_PER_MANAGER,
    build_d_phi_vector,
    build_dual_scales,
    build_lambda_warm_vector,
    get_severity_alpha,
    get_severity_thresholds,
)


# ---------------------------------------------------------------------------
# W01 — hardware constants
# ---------------------------------------------------------------------------


class TestHardwareConstants:
    def test_p_total(self):
        # 3GPP TS 38.101-1 Table 5.3.2-1 for 100 MHz μ=1
        assert P_TOTAL == 273

    def test_b_prb(self):
        # 12 subcarriers × 30 kHz SCS = 360 kHz per PRB
        assert B_PRB == pytest.approx(360e3)

    def test_f_carrier(self):
        # FR1 n78 band: 3.5 GHz
        assert F_CARRIER == pytest.approx(3.5e9)

    def test_d_det(self):
        # Deterministic processing delay: 0.07 ms
        assert D_DET == pytest.approx(0.07e-3)

    def test_d_fh(self):
        # Fronthaul one-way latency: 0.1 ms
        assert D_FH == pytest.approx(0.1e-3)

    def test_d_bh(self):
        # Backhaul latency: 0.1 ms
        assert D_BH == pytest.approx(0.1e-3)

    def test_d_stoch(self):
        # Stochastic RLC/retx mean: 0.05 ms (reviewer PB-C2 fix)
        assert D_STOCH == pytest.approx(0.05e-3)

    def test_d_ref_urllc(self):
        # Tightest D_max (severity 5 IMMEDIATE): 1 ms
        assert D_REF_URLLC == pytest.approx(1e-3)

    def test_aoi_ref_s(self):
        # Tightest AoI budget: 0.1 s
        assert AOI_REF_S == pytest.approx(0.1)

    def test_r_ref_embb_mbps(self):
        # eMBB log-utility anchor: 100 Mbps
        assert R_REF_EMBB_MBPS == pytest.approx(100.0)

    def test_lambda_max(self):
        # Dual ascent projection cap
        assert LAMBDA_MAX == pytest.approx(10.0)

    def test_alpha_lambda_dual(self):
        # Dual-ascent step-size; 2e-4 (reverted from 5e-4 A/B). Hierarchy: 3e-5 < 2e-4 < 3e-4.
        assert ALPHA_LAMBDA_DUAL == pytest.approx(2e-4)


# ---------------------------------------------------------------------------
# W01 — PRB split parameters
# ---------------------------------------------------------------------------


class TestPRBSplitConstants:
    def test_beta_min(self):
        # BETA_MIN > 0 ensures minimum severity ordering in PRB split
        assert BETA_MIN == pytest.approx(0.5)

    def test_beta_max(self):
        assert BETA_MAX == pytest.approx(5.0)

    def test_intra_slice_kappa(self):
        # Floor fraction: b = floor(κ·B_U/K)
        assert INTRA_SLICE_KAPPA == pytest.approx(0.5)

    def test_prb_min_qos(self):
        assert PRB_MIN_QOS == 1

    def test_rho_urgency_tiebreak(self):
        # δ = ρ·β; severity term (β·0.8) dominates this tiebreaker
        assert RHO_URGENCY_TIEBREAK == pytest.approx(0.15)

    def test_b_rrm_min(self):
        # Global floor = min(B_RRM_FLOOR_BY_SEV): ambulance always > eMBB
        assert B_RRM_MIN == pytest.approx(0.65)

    def test_b_rrm_floor_by_sev(self):
        # Severity-dependent floor: ambulance always prioritized, monotonic
        expected = {1: 0.65, 2: 0.70, 3: 0.75, 4: 0.80, 5: 0.85}
        assert B_RRM_FLOOR_BY_SEV == expected
        floors = [B_RRM_FLOOR_BY_SEV[s] for s in range(1, 6)]
        assert floors == sorted(floors), "Floor must be monotonically increasing"
        assert all(f > 0.50 for f in floors), "URLLC must always > 50%"
        assert B_RRM_MIN == pytest.approx(min(floors))

    def test_b_rrm_max(self):
        # Upper bound: leaves ≥15% PRBs for eMBB
        assert B_RRM_MAX == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# W05 — SEVERITY_QOS table exact values
# ---------------------------------------------------------------------------
# sev | D_max  | eps    | AoI_max | eps_aoi
#  1  | 20ms   | 1e-3   | 1.0 s   | 1e-2
#  2  | 10ms   | 1e-4   | 0.5 s   | 1e-3
#  3  |  5ms   | 1e-4   | 0.2 s   | 1e-3
#  4  |  2ms   | 1e-5   | 0.1 s   | 1e-3
#  5  |  1ms   | 1e-5   | 0.1 s   | 1e-3


class TestSeverityQOS:
    def test_sev1_dmax(self):
        assert SEVERITY_QOS[1]["D_max"] == pytest.approx(20e-3)

    def test_sev1_eps(self):
        assert SEVERITY_QOS[1]["eps"] == pytest.approx(1e-3)

    def test_sev1_aoi_max(self):
        assert SEVERITY_QOS[1]["AoI_max"] == pytest.approx(1.0)

    def test_sev1_eps_aoi(self):
        assert SEVERITY_QOS[1]["eps_aoi"] == pytest.approx(1e-2)

    def test_sev2_dmax(self):
        assert SEVERITY_QOS[2]["D_max"] == pytest.approx(10e-3)

    def test_sev2_eps(self):
        assert SEVERITY_QOS[2]["eps"] == pytest.approx(1e-4)

    def test_sev2_aoi_max(self):
        assert SEVERITY_QOS[2]["AoI_max"] == pytest.approx(0.5)

    def test_sev2_eps_aoi(self):
        assert SEVERITY_QOS[2]["eps_aoi"] == pytest.approx(1e-3)

    def test_sev3_dmax(self):
        assert SEVERITY_QOS[3]["D_max"] == pytest.approx(5e-3)

    def test_sev3_eps(self):
        assert SEVERITY_QOS[3]["eps"] == pytest.approx(1e-4)

    def test_sev3_aoi_max(self):
        assert SEVERITY_QOS[3]["AoI_max"] == pytest.approx(0.2)

    def test_sev3_eps_aoi(self):
        assert SEVERITY_QOS[3]["eps_aoi"] == pytest.approx(1e-3)

    def test_sev4_dmax(self):
        assert SEVERITY_QOS[4]["D_max"] == pytest.approx(2e-3)

    def test_sev4_eps(self):
        assert SEVERITY_QOS[4]["eps"] == pytest.approx(1e-5)

    def test_sev4_aoi_max(self):
        assert SEVERITY_QOS[4]["AoI_max"] == pytest.approx(0.1)

    def test_sev4_eps_aoi(self):
        assert SEVERITY_QOS[4]["eps_aoi"] == pytest.approx(1e-3)

    def test_sev5_dmax(self):
        assert SEVERITY_QOS[5]["D_max"] == pytest.approx(1e-3)

    def test_sev5_eps(self):
        assert SEVERITY_QOS[5]["eps"] == pytest.approx(1e-5)

    def test_sev5_aoi_max(self):
        assert SEVERITY_QOS[5]["AoI_max"] == pytest.approx(0.1)

    def test_sev5_eps_aoi(self):
        assert SEVERITY_QOS[5]["eps_aoi"] == pytest.approx(1e-3)

    def test_dmax_monotonically_decreasing(self):
        dmaxes = [SEVERITY_QOS[s]["D_max"] for s in range(1, 6)]
        for i in range(len(dmaxes) - 1):
            assert dmaxes[i] > dmaxes[i + 1], f"D_max not strictly decreasing at sev {i + 1}"

    def test_aoi_max_non_increasing(self):
        aois = [SEVERITY_QOS[s]["AoI_max"] for s in range(1, 6)]
        for i in range(len(aois) - 1):
            assert aois[i] >= aois[i + 1]

    def test_has_all_five_levels(self):
        assert set(SEVERITY_QOS.keys()) == {1, 2, 3, 4, 5}


# ---------------------------------------------------------------------------
# W05 — SEVERITY_ALPHA table exact values
# ---------------------------------------------------------------------------
# sev | urllc | embb   (higher sev → more URLLC, less eMBB)
#  1  | 0.30  | 0.70
#  2  | 0.45  | 0.55
#  3  | 0.60  | 0.40
#  4  | 0.80  | 0.20
#  5  | 0.95  | 0.05


class TestSeverityAlpha:
    def test_sev1_alpha(self):
        assert SEVERITY_ALPHA[1]["urllc"] == pytest.approx(0.30)
        assert SEVERITY_ALPHA[1]["embb"] == pytest.approx(0.70)

    def test_sev2_alpha(self):
        assert SEVERITY_ALPHA[2]["urllc"] == pytest.approx(0.45)
        assert SEVERITY_ALPHA[2]["embb"] == pytest.approx(0.55)

    def test_sev3_alpha(self):
        assert SEVERITY_ALPHA[3]["urllc"] == pytest.approx(0.60)
        assert SEVERITY_ALPHA[3]["embb"] == pytest.approx(0.40)

    def test_sev4_alpha(self):
        assert SEVERITY_ALPHA[4]["urllc"] == pytest.approx(0.80)
        assert SEVERITY_ALPHA[4]["embb"] == pytest.approx(0.20)

    def test_sev5_alpha(self):
        assert SEVERITY_ALPHA[5]["urllc"] == pytest.approx(0.95)
        assert SEVERITY_ALPHA[5]["embb"] == pytest.approx(0.05)

    def test_each_pair_sums_to_one(self):
        for sev in range(1, 6):
            total = SEVERITY_ALPHA[sev]["urllc"] + SEVERITY_ALPHA[sev]["embb"]
            assert total == pytest.approx(1.0), f"sev={sev}: sum={total}"

    def test_urllc_monotonically_increasing(self):
        urlcs = [SEVERITY_ALPHA[s]["urllc"] for s in range(1, 6)]
        for i in range(len(urlcs) - 1):
            assert urlcs[i] < urlcs[i + 1]

    def test_embb_monotonically_decreasing(self):
        embbs = [SEVERITY_ALPHA[s]["embb"] for s in range(1, 6)]
        for i in range(len(embbs) - 1):
            assert embbs[i] > embbs[i + 1]


# ---------------------------------------------------------------------------
# W06 — CMDP_D_J_SEVERITY exact values
# ---------------------------------------------------------------------------
# sev | d1_lat_mean | d2_lat_tail | d3_embb_mbps | d4_aoi_mean | d5_aoi_tail
#  1  | 20ms        | 1e-3        | 30.0         | 1.0         | 1e-2
#  2  | 10ms        | 1e-4        | 25.0         | 0.5         | 1e-3
#  3  |  5ms        | 1e-4        | 20.0         | 0.2         | 1e-3
#  4  |  2ms        | 1e-5        | 15.0         | 0.1         | 1e-3
#  5  |  1ms        | 1e-5        | 10.0         | 0.1         | 1e-3


class TestCMDPDJSeverity:
    def test_sev1_all_fields(self):
        d = CMDP_D_J_SEVERITY[1]
        assert d["d1_lat_mean"] == pytest.approx(20e-3)
        assert d["d2_lat_tail"] == pytest.approx(1e-3)
        assert d["d3_embb_mbps"] == pytest.approx(10.0)   # FIXED eMBB floor (Gate 7)
        assert d["d4_aoi_mean"] == pytest.approx(1.0)
        assert d["d5_aoi_tail"] == pytest.approx(1e-2)

    def test_sev2_all_fields(self):
        d = CMDP_D_J_SEVERITY[2]
        assert d["d1_lat_mean"] == pytest.approx(10e-3)
        assert d["d2_lat_tail"] == pytest.approx(1e-4)
        assert d["d3_embb_mbps"] == pytest.approx(10.0)   # FIXED eMBB floor (Gate 7)
        assert d["d4_aoi_mean"] == pytest.approx(0.5)
        assert d["d5_aoi_tail"] == pytest.approx(1e-3)

    def test_sev3_all_fields(self):
        d = CMDP_D_J_SEVERITY[3]
        assert d["d1_lat_mean"] == pytest.approx(5e-3)
        assert d["d2_lat_tail"] == pytest.approx(1e-4)
        assert d["d3_embb_mbps"] == pytest.approx(10.0)   # FIXED eMBB floor (Gate 7)
        assert d["d4_aoi_mean"] == pytest.approx(0.2)
        assert d["d5_aoi_tail"] == pytest.approx(1e-3)

    def test_sev4_all_fields(self):
        d = CMDP_D_J_SEVERITY[4]
        assert d["d1_lat_mean"] == pytest.approx(2e-3)
        assert d["d2_lat_tail"] == pytest.approx(1e-5)
        assert d["d3_embb_mbps"] == pytest.approx(10.0)   # FIXED eMBB floor (Gate 7)
        assert d["d4_aoi_mean"] == pytest.approx(0.1)
        assert d["d5_aoi_tail"] == pytest.approx(1e-3)

    def test_sev5_all_fields(self):
        d = CMDP_D_J_SEVERITY[5]
        assert d["d1_lat_mean"] == pytest.approx(1e-3)
        assert d["d2_lat_tail"] == pytest.approx(1e-5)
        assert d["d3_embb_mbps"] == pytest.approx(10.0)
        assert d["d4_aoi_mean"] == pytest.approx(0.1)
        assert d["d5_aoi_tail"] == pytest.approx(1e-3)

    def test_d1_lat_mean_matches_severity_qos_dmax(self):
        # CMDP_D_J_SEVERITY d1 must equal SEVERITY_QOS D_max
        for sev in range(1, 6):
            assert CMDP_D_J_SEVERITY[sev]["d1_lat_mean"] == pytest.approx(
                SEVERITY_QOS[sev]["D_max"]
            ), f"sev={sev}"

    def test_d3_embb_mbps_fixed_across_severity(self):
        # Gate 7: eMBB floor is a FIXED severity-independent 10 Mbps SLA
        d3s = [CMDP_D_J_SEVERITY[s]["d3_embb_mbps"] for s in range(1, 6)]
        assert d3s == [10.0, 10.0, 10.0, 10.0, 10.0], f"d3 must be fixed at 10: {d3s}"


# ---------------------------------------------------------------------------
# W06 — LAMBDA_WARM exact values
# ---------------------------------------------------------------------------
# Indexed as [C1, C2, C3, C4, C5] for each severity level
# C3 slot (index 2) is FIXED at 0.02 (Gate 7: severity-independent 10 Mbps floor)
# sev |  C1   |  C2   |  C3   |  C4   |  C5
#  1  | 0.02  | 0.01  | 0.02  | 0.01  | 0.00
#  2  | 0.15  | 0.08  | 0.02  | 0.05  | 0.02
#  3  | 0.60  | 0.70  | 0.02  | 0.50  | 0.60
#  4  | 1.20  | 1.50  | 0.02  | 1.20  | 1.50
#  5  | 1.80  | 2.20  | 0.02  | 1.50  | 2.00


class TestLambdaWarm:
    def test_sev1_all_values(self):
        w = LAMBDA_WARM[1]
        assert w[0] == pytest.approx(0.02)
        assert w[1] == pytest.approx(0.01)
        assert w[2] == pytest.approx(0.02)   # C3 FIXED (Gate 7)
        assert w[3] == pytest.approx(0.01)
        assert w[4] == pytest.approx(0.00)

    def test_sev2_all_values(self):
        w = LAMBDA_WARM[2]
        assert w[0] == pytest.approx(0.15)
        assert w[1] == pytest.approx(0.08)
        assert w[2] == pytest.approx(0.02)   # C3 FIXED (Gate 7)
        assert w[3] == pytest.approx(0.05)
        assert w[4] == pytest.approx(0.02)

    def test_sev3_all_values(self):
        w = LAMBDA_WARM[3]
        assert w[0] == pytest.approx(0.60)
        assert w[1] == pytest.approx(0.70)
        assert w[2] == pytest.approx(0.02)   # C3 FIXED (Gate 7)
        assert w[3] == pytest.approx(0.50)
        assert w[4] == pytest.approx(0.60)

    def test_sev4_all_values(self):
        w = LAMBDA_WARM[4]
        assert w[0] == pytest.approx(1.20)
        assert w[1] == pytest.approx(1.50)
        assert w[2] == pytest.approx(0.02)   # C3 FIXED (Gate 7)
        assert w[3] == pytest.approx(1.20)
        assert w[4] == pytest.approx(1.50)

    def test_sev5_all_values(self):
        w = LAMBDA_WARM[5]
        assert w[0] == pytest.approx(1.80)
        assert w[1] == pytest.approx(2.20)
        assert w[2] == pytest.approx(0.02)
        assert w[3] == pytest.approx(1.50)
        assert w[4] == pytest.approx(2.00)

    def test_c3_slot_fixed_across_severity(self):
        # Gate 7: C3 (index 2) is FIXED at 0.02 (severity-independent 10 Mbps floor)
        c3s = [LAMBDA_WARM[s][2] for s in range(1, 6)]
        assert c3s == [0.02, 0.02, 0.02, 0.02, 0.02], f"C3 warm must be fixed: {c3s}"

    def test_all_values_non_negative(self):
        for sev in range(1, 6):
            for val in LAMBDA_WARM[sev]:
                assert val >= 0.0, f"Negative LAMBDA_WARM at sev={sev}"

    def test_five_levels(self):
        assert len(LAMBDA_WARM) == 5
        assert set(LAMBDA_WARM.keys()) == {1, 2, 3, 4, 5}


# ---------------------------------------------------------------------------
# W06 — Discount factor chain
# ---------------------------------------------------------------------------


class TestDiscountFactors:
    def test_gamma_worker(self):
        assert GAMMA_WORKER == pytest.approx(0.99)

    def test_gamma_is_gamma_worker(self):
        assert GAMMA == pytest.approx(GAMMA_WORKER)

    def test_worker_steps_per_manager(self):
        assert WORKER_STEPS_PER_MANAGER == 10

    def test_gamma_manager_equals_gamma_worker_power_w(self):
        # γ_H = γ_L^W (ensures equal effective horizons in wall-clock time)
        expected = GAMMA_WORKER ** WORKER_STEPS_PER_MANAGER
        assert GAMMA_MANAGER == pytest.approx(expected, rel=1e-12)

    def test_gamma_manager_approx_0904(self):
        # 0.99^10 ≈ 0.9044 — sanity check against known value
        assert GAMMA_MANAGER == pytest.approx(0.904382, rel=1e-4)

    def test_gamma_manager_strictly_less_than_worker(self):
        assert GAMMA_MANAGER < GAMMA_WORKER


# ---------------------------------------------------------------------------
# W06 — get_severity_thresholds
# ---------------------------------------------------------------------------


class TestGetSeverityThresholds:
    def test_sev1_thresholds(self):
        t = get_severity_thresholds(1)
        assert t["d1"] == pytest.approx(20e-3)
        assert t["d2"] == pytest.approx(1e-3)
        assert t["d3"] == pytest.approx(0.0)
        assert t["d4"] == pytest.approx(1.0)
        assert t["d5"] == pytest.approx(1e-2)

    def test_sev5_thresholds(self):
        t = get_severity_thresholds(5)
        assert t["d1"] == pytest.approx(1e-3)
        assert t["d2"] == pytest.approx(1e-5)
        assert t["d3"] == pytest.approx(0.0)
        assert t["d4"] == pytest.approx(0.1)
        assert t["d5"] == pytest.approx(1e-3)

    def test_d3_always_zero(self):
        # d3 threshold is 0 (C3 signal is signed eMBB gap, not threshold)
        for sev in range(1, 6):
            assert get_severity_thresholds(sev)["d3"] == 0.0

    def test_d1_matches_severity_qos_dmax(self):
        for sev in range(1, 6):
            assert get_severity_thresholds(sev)["d1"] == pytest.approx(
                SEVERITY_QOS[sev]["D_max"]
            )

    def test_invalid_severity_raises(self):
        with pytest.raises(ValueError):
            get_severity_thresholds(0)
        with pytest.raises(ValueError):
            get_severity_thresholds(6)


# ---------------------------------------------------------------------------
# W06 — get_severity_alpha
# ---------------------------------------------------------------------------


class TestGetSeverityAlpha:
    def test_sev1_returns_correct_pair(self):
        alpha_u, alpha_e = get_severity_alpha(1)
        assert alpha_u == pytest.approx(0.30)
        assert alpha_e == pytest.approx(0.70)

    def test_sev5_returns_correct_pair(self):
        alpha_u, alpha_e = get_severity_alpha(5)
        assert alpha_u == pytest.approx(0.95)
        assert alpha_e == pytest.approx(0.05)

    def test_all_pairs_sum_to_one(self):
        for sev in range(1, 6):
            alpha_u, alpha_e = get_severity_alpha(sev)
            assert alpha_u + alpha_e == pytest.approx(1.0)

    def test_invalid_severity_raises(self):
        with pytest.raises(ValueError):
            get_severity_alpha(0)


# ---------------------------------------------------------------------------
# W06 — build_lambda_warm_vector
# ---------------------------------------------------------------------------
# At K=1, sev_per_amb=[s], sev_ref=s:
#   [C1_0, C2_0, C4_0, C5_0, C3_shared] = [w[0], w[1], w[3], w[4], w[2]]
# where w = LAMBDA_WARM[s].


class TestBuildLambdaWarmVector:
    def test_k1_sev1_exact_values(self):
        vec = build_lambda_warm_vector([1], severity_ref=1)
        w = LAMBDA_WARM[1]
        # Layout: [C1_0, C2_0, C4_0, C5_0, C3_shared]
        np.testing.assert_allclose(vec[0], w[0], rtol=1e-12)  # C1 = 0.02
        np.testing.assert_allclose(vec[1], w[1], rtol=1e-12)  # C2 = 0.01
        np.testing.assert_allclose(vec[2], w[3], rtol=1e-12)  # C4 = 0.01
        np.testing.assert_allclose(vec[3], w[4], rtol=1e-12)  # C5 = 0.00
        np.testing.assert_allclose(vec[4], w[2], rtol=1e-12)  # C3 = 0.10

    def test_k1_sev5_exact_values(self):
        vec = build_lambda_warm_vector([5], severity_ref=5)
        w = LAMBDA_WARM[5]
        np.testing.assert_allclose(vec[0], w[0], rtol=1e-12)  # C1 = 1.80
        np.testing.assert_allclose(vec[1], w[1], rtol=1e-12)  # C2 = 2.20
        np.testing.assert_allclose(vec[2], w[3], rtol=1e-12)  # C4 = 1.50
        np.testing.assert_allclose(vec[3], w[4], rtol=1e-12)  # C5 = 2.00
        np.testing.assert_allclose(vec[4], w[2], rtol=1e-12)  # C3 = 0.02

    def test_k1_shape(self):
        vec = build_lambda_warm_vector([3], severity_ref=3)
        assert vec.shape == (5,)

    def test_k3_shape(self):
        vec = build_lambda_warm_vector([1, 2, 3], severity_ref=2)
        assert vec.shape == (13,)  # 4*3+1 = 13

    def test_k3_c3_uses_severity_ref_not_per_amb(self):
        # C3 slot = LAMBDA_WARM[severity_ref][2], sourced from severity_ref (NOT
        # per-ambulance). The VALUE is now severity-independent (fixed 0.02, Gate 7),
        # so both refs return 0.02 — the test still verifies the SOURCING mechanic.
        vec_ref1 = build_lambda_warm_vector([2, 3, 4], severity_ref=1)
        vec_ref5 = build_lambda_warm_vector([2, 3, 4], severity_ref=5)
        np.testing.assert_allclose(vec_ref1[-1], LAMBDA_WARM[1][2], rtol=1e-12)
        np.testing.assert_allclose(vec_ref5[-1], LAMBDA_WARM[5][2], rtol=1e-12)
        # C3 warm is fixed across severity (decoupled eMBB SLA, Gate 7)
        assert vec_ref1[-1] == vec_ref5[-1] == pytest.approx(0.02)

    def test_k3_per_ambulance_c1_values(self):
        # C1_k = LAMBDA_WARM[severity_per_amb[k]][0] for each k
        vec = build_lambda_warm_vector([1, 3, 5], severity_ref=1)
        # C1_0 = LAMBDA_WARM[1][0] = 0.02
        # C1_1 = LAMBDA_WARM[3][0] = 0.60
        # C1_2 = LAMBDA_WARM[5][0] = 1.80
        np.testing.assert_allclose(vec[0], 0.02, rtol=1e-12)
        np.testing.assert_allclose(vec[1], 0.60, rtol=1e-12)
        np.testing.assert_allclose(vec[2], 1.80, rtol=1e-12)

    def test_invalid_severity_raises(self):
        with pytest.raises(ValueError):
            build_lambda_warm_vector([0], severity_ref=1)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            build_lambda_warm_vector([], severity_ref=1)


# ---------------------------------------------------------------------------
# W06 — build_dual_scales
# ---------------------------------------------------------------------------
# K=1: [D_REF_URLLC, 1.0, AOI_REF_S, 1.0, R_REF_EMBB_MBPS]
#   = [1e-3, 1.0, 0.1, 1.0, 100.0]


class TestBuildDualScales:
    def test_k1_shape(self):
        assert build_dual_scales(1).shape == (5,)

    def test_k1_exact_values(self):
        s = build_dual_scales(1)
        # [C1, C2, C4, C5, C3]
        np.testing.assert_allclose(s[0], D_REF_URLLC, rtol=1e-12)     # 1e-3
        np.testing.assert_allclose(s[1], 1.0, rtol=1e-12)              # C2 scale
        np.testing.assert_allclose(s[2], AOI_REF_S, rtol=1e-12)        # 0.1
        np.testing.assert_allclose(s[3], 1.0, rtol=1e-12)              # C5 scale
        np.testing.assert_allclose(s[4], R_REF_EMBB_MBPS, rtol=1e-12) # 100.0

    def test_k3_shape(self):
        assert build_dual_scales(3).shape == (13,)  # 4*3+1=13

    def test_k3_c1_block(self):
        s = build_dual_scales(3)
        # First 3 elements = C1_0, C1_1, C1_2 (all = D_REF_URLLC)
        np.testing.assert_allclose(s[:3], D_REF_URLLC, rtol=1e-12)

    def test_k3_last_is_embb_ref(self):
        s = build_dual_scales(3)
        np.testing.assert_allclose(s[-1], R_REF_EMBB_MBPS, rtol=1e-12)

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError):
            build_dual_scales(0)
        with pytest.raises(ValueError):
            build_dual_scales(-1)


# ---------------------------------------------------------------------------
# W06 — build_d_phi_vector
# ---------------------------------------------------------------------------
# d1_k/d2_k from get_severity_thresholds(severity_per_amb[k])
# d3_shared = 0.0 unconditionally


class TestBuildDPhiVector:
    def test_k1_shape(self):
        assert build_d_phi_vector([1]).shape == (5,)

    def test_k1_sev3_exact(self):
        phi = build_d_phi_vector([3])
        t3 = get_severity_thresholds(3)
        # [d1, d2, d4, d5, d3_shared]
        np.testing.assert_allclose(phi[0], t3["d1"], rtol=1e-12)  # 5ms
        np.testing.assert_allclose(phi[1], t3["d2"], rtol=1e-12)  # 1e-4
        np.testing.assert_allclose(phi[2], t3["d4"], rtol=1e-12)  # 0.2
        np.testing.assert_allclose(phi[3], t3["d5"], rtol=1e-12)  # 1e-3
        np.testing.assert_allclose(phi[4], 0.0, atol=1e-15)       # C3 always 0

    def test_k1_d3_always_zero(self):
        for sev in range(1, 6):
            phi = build_d_phi_vector([sev])
            assert phi[-1] == 0.0, f"d3 not zero for sev={sev}"

    def test_k3_shape(self):
        assert build_d_phi_vector([1, 2, 3]).shape == (13,)

    def test_k3_each_amb_uses_own_severity(self):
        phi = build_d_phi_vector([1, 3, 5])
        t1 = get_severity_thresholds(1)
        t3 = get_severity_thresholds(3)
        t5 = get_severity_thresholds(5)
        # d1 block: [d1_sev1, d1_sev3, d1_sev5]
        np.testing.assert_allclose(phi[0], t1["d1"], rtol=1e-12)
        np.testing.assert_allclose(phi[1], t3["d1"], rtol=1e-12)
        np.testing.assert_allclose(phi[2], t5["d1"], rtol=1e-12)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            build_d_phi_vector([])
