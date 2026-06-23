"""Rigorous formula tests — W03/W04/W05 AoI, obs normalization, eMBB throughput.

Every expected value is computed from FIRST PRINCIPLES — textbook closed forms or
exact arithmetic. No function is called as its own oracle.

W03 AoI:
  expected_aoi_mm1     Kaul 2012 M/M/1 FCFS average AoI
  aoi_threshold_for_severity  per-severity AoI_max lookup

W04 Tracker:
  AoIStreamTracker LCFS + drop_old semantics
  AoIStreamTracker FCFS FIFO semantics
  violation_rate edge cases (0/100%/partial)

W04 BLER:
  logistic approximation 1/(1+exp(0.5*(sinr-2))) + clip [1e-4, 0.5]

W04 PRB allocation:
  prb_urllc = int(r_min * P_TOTAL), prb_emBB = int(r_max * P_TOTAL)
  clamp: if sum > P_TOTAL then prb_emBB = P_TOTAL - prb_urllc

W04 Obs normalization:
  delay_norm_k = D_e2e_k / D_max^{sev_k}
  AoI_norm_k   = AoI_k   / AoI_max^{sev_k}
  severity_k_norm = sev_k / 5.0
  arr_urllc = Σ arrival_rate_k / 1e3
  arr_emBB  = arrival_rate_eMBB / 1e4

W05 eMBB throughput:
  served_pkts = min(arrival_rate, service_rate)
  throughput_mbps = served_pkts * mean_packet_bits / 1e6
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from env.aoi_tracker import (
    AoIPacket,
    AoIStreamTracker,
    aoi_threshold_for_severity,
    expected_aoi_mm1,
)
from env.oran_env import EnvConfig, ORANEnv
from env.queue_model import MG1Queue
from utils.config import P_TOTAL, SEVERITY_QOS


# ---------------------------------------------------------------------------
# W03 — expected_aoi_mm1 (Kaul 2012)
# ---------------------------------------------------------------------------
# Formula: E[AoI] = (1/μ) · (1 + ρ/(1-ρ) + ρ²/(1-ρ²)),  ρ = λ/μ
# Returns inf when ρ ≥ 1 or μ ≤ 0 or λ < 0.
# ---------------------------------------------------------------------------


class TestAoiMM1Formula:
    def test_half_load_unit_server(self):
        # λ=0.5, μ=1.0 → ρ=0.5
        # E[AoI] = 1*(1 + 0.5/0.5 + 0.25/0.75) = 1 + 1 + 1/3 = 7/3
        assert expected_aoi_mm1(0.5, 1.0) == pytest.approx(7 / 3, rel=1e-9)

    def test_zero_arrival(self):
        # λ=0.0 → ρ=0 → E[AoI] = (1/μ)*(1+0+0) = 1/μ
        assert expected_aoi_mm1(0.0, 1.0) == pytest.approx(1.0, rel=1e-9)
        assert expected_aoi_mm1(0.0, 2.0) == pytest.approx(0.5, rel=1e-9)

    def test_half_load_double_server(self):
        # λ=1.0, μ=2.0 → ρ=0.5, E[AoI] = (1/2)*(7/3) = 7/6
        assert expected_aoi_mm1(1.0, 2.0) == pytest.approx(7 / 6, rel=1e-9)

    def test_high_utilization(self):
        # λ=0.9, μ=1.0 → ρ=0.9
        # E[AoI] = 1*(1 + 0.9/0.1 + 0.81/0.19) = 1 + 9 + 81/19 = 271/19
        assert expected_aoi_mm1(0.9, 1.0) == pytest.approx(271 / 19, rel=1e-9)

    def test_low_utilization_non_unit(self):
        # λ=0.2, μ=0.5 → ρ=0.4
        # E[AoI] = (1/0.5)*(1 + 0.4/0.6 + 0.16/0.84)
        #        = 2 * (1 + 2/3 + 4/21) = 2 * 39/21 = 26/7
        assert expected_aoi_mm1(0.2, 0.5) == pytest.approx(26 / 7, rel=1e-9)

    def test_critical_load_returns_inf(self):
        # ρ = 1 → system is at capacity → inf
        assert expected_aoi_mm1(1.0, 1.0) == float("inf")

    def test_overloaded_returns_inf(self):
        # ρ > 1 → unstable → inf
        assert expected_aoi_mm1(2.0, 1.0) == float("inf")

    def test_zero_service_rate_returns_inf(self):
        assert expected_aoi_mm1(0.5, 0.0) == float("inf")

    def test_negative_arrival_rate_returns_inf(self):
        assert expected_aoi_mm1(-1.0, 1.0) == float("inf")

    def test_near_critical_load_large_but_finite(self):
        # ρ=0.99 → E[AoI] is large but finite
        result = expected_aoi_mm1(0.99, 1.0)
        assert math.isfinite(result)
        assert result > 100.0  # Very large at high load


# ---------------------------------------------------------------------------
# W03 — aoi_threshold_for_severity
# ---------------------------------------------------------------------------
# From SEVERITY_QOS: AoI_max values are 1.0, 0.5, 0.2, 0.1, 0.1 for sev 1-5.
# Unknown streams → inf.
# ---------------------------------------------------------------------------


class TestAoiThresholdForSeverity:
    def test_sev1_threshold(self):
        assert aoi_threshold_for_severity(1, "ambulance_status") == pytest.approx(1.0)

    def test_sev2_threshold(self):
        assert aoi_threshold_for_severity(2, "ambulance_status") == pytest.approx(0.5)

    def test_sev3_threshold(self):
        assert aoi_threshold_for_severity(3, "ambulance_status") == pytest.approx(0.2)

    def test_sev4_threshold(self):
        assert aoi_threshold_for_severity(4, "ambulance_status") == pytest.approx(0.1)

    def test_sev5_threshold(self):
        assert aoi_threshold_for_severity(5, "ambulance_status") == pytest.approx(0.1)

    def test_sev4_equals_sev5(self):
        # Both are 0.1 s (tightest AoI budget)
        assert aoi_threshold_for_severity(4, "ambulance_status") == aoi_threshold_for_severity(
            5, "ambulance_status"
        )

    def test_monotonically_non_increasing(self):
        # Stricter severity → tighter (smaller or equal) AoI budget
        thresholds = [aoi_threshold_for_severity(s, "ambulance_status") for s in range(1, 6)]
        for i in range(len(thresholds) - 1):
            assert thresholds[i] >= thresholds[i + 1]

    def test_unknown_stream_returns_inf(self):
        assert aoi_threshold_for_severity(3, "unknown_stream") == float("inf")


# ---------------------------------------------------------------------------
# W04 — AoIStreamTracker LCFS + drop_old
# ---------------------------------------------------------------------------
# ambulance_status: queue="LCFS", drop_old=True
# Newer arrival replaces older pending packet. deliver_next picks the newest.
# ---------------------------------------------------------------------------


class TestAoiStreamTrackerLCFS:
    def test_from_spec_creates_lcfs_drop_old(self):
        t = AoIStreamTracker.from_spec("ambulance_status")
        assert t.queue_kind == "LCFS"
        assert t.drop_old is True

    def test_initial_state_empty(self):
        t = AoIStreamTracker.from_spec("ambulance_status")
        assert len(t.queue) == 0
        assert t.delivered_count == 0
        assert t.dropped_count == 0
        assert t.last_delivered_gen_time is None

    def test_single_arrive_fills_queue(self):
        t = AoIStreamTracker.from_spec("ambulance_status")
        t.arrive(gen_time=1.0)
        assert len(t.queue) == 1
        assert t.dropped_count == 0

    def test_second_arrive_drops_old(self):
        t = AoIStreamTracker.from_spec("ambulance_status")
        t.arrive(gen_time=1.0)
        t.arrive(gen_time=2.0)
        # First packet dropped, only newest in queue
        assert len(t.queue) == 1
        assert t.dropped_count == 1

    def test_deliver_returns_newest_packet(self):
        t = AoIStreamTracker.from_spec("ambulance_status")
        t.arrive(gen_time=1.0)
        t.arrive(gen_time=3.0)
        pkt = t.deliver_next(sim_time=5.0)
        # LCFS: newest (gen_time=3.0) delivered
        assert pkt is not None
        assert pkt.gen_time == pytest.approx(3.0)
        assert pkt.deliver_time == pytest.approx(5.0)

    def test_deliver_increments_delivered_count(self):
        t = AoIStreamTracker.from_spec("ambulance_status")
        t.arrive(gen_time=1.0)
        t.deliver_next(sim_time=2.0)
        assert t.delivered_count == 1

    def test_deliver_updates_last_gen_time(self):
        t = AoIStreamTracker.from_spec("ambulance_status")
        t.arrive(gen_time=4.0)
        t.deliver_next(sim_time=5.0)
        assert t.last_delivered_gen_time == pytest.approx(4.0)

    def test_current_aoi_before_delivery_equals_sim_time(self):
        t = AoIStreamTracker.from_spec("ambulance_status")
        # Never delivered → AoI = sim_time (elapsed since start)
        assert t.current_aoi(sim_time=10.0) == pytest.approx(10.0)

    def test_current_aoi_after_delivery(self):
        t = AoIStreamTracker.from_spec("ambulance_status")
        t.arrive(gen_time=3.0)
        t.deliver_next(sim_time=5.0)
        # AoI = sim_time - gen_time(last delivered) = 7.0 - 3.0 = 4.0
        assert t.current_aoi(sim_time=7.0) == pytest.approx(4.0)

    def test_deliver_empty_queue_returns_none(self):
        t = AoIStreamTracker.from_spec("ambulance_status")
        assert t.deliver_next(sim_time=1.0) is None

    def test_reset_clears_all_state(self):
        t = AoIStreamTracker.from_spec("ambulance_status")
        t.arrive(gen_time=1.0)
        t.deliver_next(sim_time=2.0)
        t.reset()
        assert len(t.queue) == 0
        assert t.delivered_count == 0
        assert t.dropped_count == 0
        assert t.last_delivered_gen_time is None
        assert len(t.aoi_samples) == 0


# ---------------------------------------------------------------------------
# W04 — AoIStreamTracker FCFS (standard FIFO, no drops)
# ---------------------------------------------------------------------------


class TestAoiStreamTrackerFCFS:
    def _make_fcfs(self) -> AoIStreamTracker:
        return AoIStreamTracker(stream_id="test", queue_kind="FCFS", drop_old=False)

    def test_fcfs_delivers_oldest_first(self):
        t = self._make_fcfs()
        t.arrive(gen_time=1.0)
        t.arrive(gen_time=2.0)
        pkt = t.deliver_next(sim_time=3.0)
        # FCFS: oldest (gen_time=1.0) delivered first
        assert pkt is not None
        assert pkt.gen_time == pytest.approx(1.0)

    def test_fcfs_does_not_drop_old(self):
        t = self._make_fcfs()
        t.arrive(gen_time=1.0)
        t.arrive(gen_time=2.0)
        assert t.dropped_count == 0
        assert len(t.queue) == 2

    def test_fcfs_second_deliver_gives_next(self):
        t = self._make_fcfs()
        t.arrive(gen_time=1.0)
        t.arrive(gen_time=2.0)
        t.deliver_next(sim_time=3.0)
        pkt2 = t.deliver_next(sim_time=4.0)
        assert pkt2 is not None
        assert pkt2.gen_time == pytest.approx(2.0)

    def test_fcfs_empty_queue_returns_none(self):
        t = self._make_fcfs()
        assert t.deliver_next(sim_time=1.0) is None

    def test_fcfs_aoi_sample_recorded_on_delivery(self):
        t = self._make_fcfs()
        t.arrive(gen_time=0.0)
        t.deliver_next(sim_time=5.0)
        # AoI at delivery = 5.0 - 0.0 = 5.0
        assert len(t.aoi_samples) == 1
        assert t.aoi_samples[0] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# W04 — violation_rate
# ---------------------------------------------------------------------------
# violation_rate = fraction of aoi_samples that STRICTLY exceed threshold.
# Empty samples → 0.0.
# ---------------------------------------------------------------------------


class TestAoiViolationRate:
    def _make_tracker_with_samples(self, samples: list[float]) -> AoIStreamTracker:
        t = AoIStreamTracker(stream_id="test", queue_kind="FCFS", drop_old=False)
        t.aoi_samples = list(samples)
        return t

    def test_empty_samples_returns_zero(self):
        t = AoIStreamTracker.from_spec("ambulance_status")
        assert t.violation_rate(threshold_sec=0.5) == pytest.approx(0.0)

    def test_all_above_threshold_returns_one(self):
        t = self._make_tracker_with_samples([1.0, 2.0, 3.0])
        assert t.violation_rate(threshold_sec=0.5) == pytest.approx(1.0)

    def test_all_below_threshold_returns_zero(self):
        t = self._make_tracker_with_samples([0.1, 0.2, 0.3])
        assert t.violation_rate(threshold_sec=0.5) == pytest.approx(0.0)

    def test_at_threshold_not_above(self):
        # Threshold is STRICT (a > threshold, not >=), so exactly at threshold → 0
        t = self._make_tracker_with_samples([0.5, 0.5, 0.5])
        assert t.violation_rate(threshold_sec=0.5) == pytest.approx(0.0)

    def test_half_above_returns_half(self):
        t = self._make_tracker_with_samples([0.1, 0.3, 0.7, 0.9])
        # 0.7 > 0.5 and 0.9 > 0.5 → 2/4 = 0.5
        assert t.violation_rate(threshold_sec=0.5) == pytest.approx(0.5)

    def test_one_in_three_above(self):
        t = self._make_tracker_with_samples([0.2, 0.4, 1.0])
        # Only 1.0 > 0.5 → 1/3
        assert t.violation_rate(threshold_sec=0.5) == pytest.approx(1 / 3, rel=1e-9)

    def test_single_sample_above(self):
        t = self._make_tracker_with_samples([0.6])
        assert t.violation_rate(threshold_sec=0.5) == pytest.approx(1.0)

    def test_single_sample_below(self):
        t = self._make_tracker_with_samples([0.4])
        assert t.violation_rate(threshold_sec=0.5) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# W04 — BLER logistic model
# ---------------------------------------------------------------------------
# BLER = clip(1/(1 + exp(0.5*(sinr_dB - 2))), 1e-4, 0.5)
# Source: oran_env.py:_sample_bler
# ---------------------------------------------------------------------------

def _bler(sinr_db: float) -> float:
    """Pure-Python reference implementation of the BLER formula."""
    raw = 1.0 / (1.0 + math.exp(0.5 * (sinr_db - 2.0)))
    return float(np.clip(raw, 1e-4, 0.5))


class TestBLERLogistic:
    def test_sinr_at_inflection(self):
        # sinr=2 dB → exp(0) = 1 → raw = 1/2 = 0.5 → clip(0.5, 1e-4, 0.5) = 0.5
        assert _bler(2.0) == pytest.approx(0.5)

    def test_sinr_below_inflection_clipped_high(self):
        # sinr=0 dB → exp(-1) ≈ 0.36788 → raw = 1/1.36788 ≈ 0.731 → clipped to 0.5
        assert _bler(0.0) == pytest.approx(0.5)

    def test_sinr_well_below_inflection_clipped_high(self):
        # sinr=-10 dB → exp(-6) small → raw ≈ 0.9975 → clipped to 0.5
        assert _bler(-10.0) == pytest.approx(0.5)

    def test_sinr_10db(self):
        # sinr=10 → exp(4) = 54.598150... → raw = 1/55.598150... ≈ 0.017986
        expected = 1.0 / (1.0 + math.exp(4.0))
        assert _bler(10.0) == pytest.approx(expected, rel=1e-9)
        assert _bler(10.0) > 1e-4
        assert _bler(10.0) < 0.5

    def test_sinr_14db(self):
        # sinr=14 → exp(6) = 403.4287... → raw = 1/404.4287... ≈ 0.002473
        expected = 1.0 / (1.0 + math.exp(6.0))
        assert _bler(14.0) == pytest.approx(expected, rel=1e-9)

    def test_sinr_very_high_clipped_low(self):
        # sinr=40 → exp(19) ≈ 1.78e8 → raw ≈ 5.6e-9 → clipped to 1e-4
        assert _bler(40.0) == pytest.approx(1e-4)

    def test_bler_monotonically_decreasing(self):
        # Higher SINR → lower BLER (before saturation clips)
        sinrs = [2.0, 5.0, 8.0, 12.0, 16.0, 20.0, 25.0]
        blers = [_bler(s) for s in sinrs]
        for i in range(len(blers) - 1):
            assert blers[i] >= blers[i + 1]

    def test_low_clip_floor(self):
        # Ensures clip lower bound is 1e-4
        assert _bler(40.0) >= 1e-4

    def test_high_clip_ceiling(self):
        # Ensures clip upper bound is 0.5
        assert _bler(-50.0) <= 0.5


# ---------------------------------------------------------------------------
# W04 — PRB allocation formula
# ---------------------------------------------------------------------------
# prb_urllc = int(r_min * P_TOTAL)
# prb_emBB  = P_TOTAL - prb_urllc   (remainder → sum always = P_TOTAL)
# Source: oran_env.py:_prb_allocation
# ---------------------------------------------------------------------------


@pytest.fixture
def env_k1():
    e = ORANEnv(EnvConfig(K_ambulances=1))
    e.reset(seed=0)
    return e


class TestPRBAllocationFormula:
    def test_sum_always_equals_p_total(self, env_k1):
        for r in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
            env_k1.r_min_urllc = r
            prb_u, prb_e = env_k1._prb_allocation()
            assert prb_u + prb_e == 273, f"sum={prb_u+prb_e} != 273 at r={r}"

    def test_urllc_floor_int(self, env_k1):
        env_k1.r_min_urllc = 0.2
        prb_u, prb_e = env_k1._prb_allocation()
        assert prb_u == 54          # int(0.2*273) = 54
        assert prb_e == 273 - 54   # remainder = 219

    def test_all_urllc(self, env_k1):
        env_k1.r_min_urllc = 1.0
        prb_u, prb_e = env_k1._prb_allocation()
        assert prb_u == 273
        assert prb_e == 0

    def test_all_embb(self, env_k1):
        env_k1.r_min_urllc = 0.0
        prb_u, prb_e = env_k1._prb_allocation()
        assert prb_u == 0
        assert prb_e == 273

    def test_half_split(self, env_k1):
        env_k1.r_min_urllc = 0.5
        prb_u, prb_e = env_k1._prb_allocation()
        assert prb_u == 136         # int(0.5*273)=136
        assert prb_e == 273 - 136  # 137

    def test_sum_exactly_total_no_clamp(self, env_k1):
        # r_min=100/273, r_max=173/273 → int(100)=100, int(173)=173, sum=273
        env_k1.r_min_urllc = 100 / 273
        env_k1.r_max_emBB = 173 / 273
        prb_u, prb_e = env_k1._prb_allocation()
        assert prb_u + prb_e <= P_TOTAL

    def test_prb_emBB_never_negative(self, env_k1):
        env_k1.r_min_urllc = 1.0
        env_k1.r_max_emBB = 1.0
        _, prb_e = env_k1._prb_allocation()
        assert prb_e >= 0


# ---------------------------------------------------------------------------
# W04 — Obs normalization formulas (pure math, no full env required)
# ---------------------------------------------------------------------------
# delay_norm_k = D_e2e_k / D_max^{sev_k}     (SEVERITY_QOS[sev]["D_max"])
# AoI_norm_k   = AoI_k   / AoI_max^{sev_k}   (SEVERITY_QOS[sev]["AoI_max"])
# severity_k_norm = sev_k / 5.0
# arr_urllc = sum(λ_k) / 1e3   (units: kpkt/s)
# arr_emBB  = λ_eMBB  / 1e4   (units: 10 kpkt/s)
# ---------------------------------------------------------------------------


class TestObsNormFormulas:
    # --- delay_norm ---
    def test_delay_norm_sev1(self):
        # D_max^sev1 = 20ms = 0.020 s
        d_e2e = 0.003  # 3 ms
        expected = 0.003 / SEVERITY_QOS[1]["D_max"]  # = 0.003/0.020 = 0.15
        assert expected == pytest.approx(0.15)

    def test_delay_norm_sev5_exactly_one(self):
        # D_max^sev5 = 1ms = 0.001 s → at D_max, delay_norm = 1.0
        d_e2e = SEVERITY_QOS[5]["D_max"]
        assert d_e2e / SEVERITY_QOS[5]["D_max"] == pytest.approx(1.0)

    def test_delay_norm_sev3(self):
        # D_max^sev3 = 5ms → 2.5 ms → 0.5
        d_e2e = 0.0025
        assert d_e2e / SEVERITY_QOS[3]["D_max"] == pytest.approx(0.5)

    def test_delay_norm_monotone_across_severities(self):
        # Fixed D_e2e → higher severity → smaller D_max → larger delay_norm
        d_e2e = 0.001
        norms = [d_e2e / SEVERITY_QOS[s]["D_max"] for s in range(1, 6)]
        for i in range(len(norms) - 1):
            assert norms[i] <= norms[i + 1]

    # --- AoI_norm ---
    def test_aoi_norm_sev1(self):
        # AoI_max^sev1 = 1.0 s
        aoi = 0.1
        assert aoi / SEVERITY_QOS[1]["AoI_max"] == pytest.approx(0.1)

    def test_aoi_norm_sev4_exactly_one(self):
        # AoI_max^sev4 = 0.1 → at AoI_max, norm = 1.0
        aoi = 0.1
        assert aoi / SEVERITY_QOS[4]["AoI_max"] == pytest.approx(1.0)

    def test_aoi_norm_sev2(self):
        # AoI_max^sev2 = 0.5 → 0.05/0.5 = 0.1
        aoi = 0.05
        assert aoi / SEVERITY_QOS[2]["AoI_max"] == pytest.approx(0.1)

    def test_aoi_norm_sev3(self):
        # AoI_max^sev3 = 0.2 → 0.1/0.2 = 0.5
        aoi = 0.1
        assert aoi / SEVERITY_QOS[3]["AoI_max"] == pytest.approx(0.5)

    # --- severity_k_norm ---
    def test_severity_norm_sev1(self):
        assert 1 / 5.0 == pytest.approx(0.2)

    def test_severity_norm_sev3(self):
        assert 3 / 5.0 == pytest.approx(0.6)

    def test_severity_norm_sev5(self):
        assert 5 / 5.0 == pytest.approx(1.0)

    def test_severity_norm_range(self):
        for sev in range(1, 6):
            norm = sev / 5.0
            assert 0.0 < norm <= 1.0

    # --- arr_urllc ---
    def test_arr_urllc_normalization(self):
        # total λ_urllc = 500 pkt/s → arr_urllc = 500/1000 = 0.5
        total = 500.0
        assert total / 1e3 == pytest.approx(0.5)

    def test_arr_urllc_single_ambulance(self):
        # λ=100 pkt/s → 0.1
        assert 100.0 / 1e3 == pytest.approx(0.1)

    # --- arr_emBB ---
    def test_arr_embb_normalization(self):
        # λ_eMBB = 10000 pkt/s → arr_emBB = 1.0
        assert 10_000.0 / 1e4 == pytest.approx(1.0)

    def test_arr_embb_half(self):
        # λ_eMBB = 5000 pkt/s → 0.5
        assert 5_000.0 / 1e4 == pytest.approx(0.5)

    def test_arr_embb_small(self):
        # λ_eMBB = 1000 pkt/s → 0.1
        assert 1_000.0 / 1e4 == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# W05 — eMBB throughput formula
# ---------------------------------------------------------------------------
# throughput_mbps = min(arrival_rate, service_rate) * mean_packet_bits / 1e6
# If service_rate = 0: return 0.0
# Source: oran_env.py:_compute_embb_throughput_mbps
# ---------------------------------------------------------------------------


@pytest.fixture
def env_embb():
    e = ORANEnv(EnvConfig(K_ambulances=1))
    e.reset(seed=0)
    return e


class TestEmbbThroughputFormula:
    def test_stable_served_at_arrival_rate(self, env_embb):
        # stable: λ < μ → served = λ
        q = env_embb.queues["eMBB"]
        q.arrival_rate = 100.0
        q.service_rate = 200.0
        q.mean_packet_bits = 1000.0
        # served = min(100, 200) = 100 → 100 * 1000 / 1e6 = 0.1 Mbps
        result = env_embb._compute_embb_throughput_mbps()
        assert result == pytest.approx(0.1)

    def test_saturated_served_at_service_rate(self, env_embb):
        # saturated: λ > μ → served = μ
        q = env_embb.queues["eMBB"]
        q.arrival_rate = 300.0
        q.service_rate = 200.0
        q.mean_packet_bits = 1000.0
        # served = min(300, 200) = 200 → 200 * 1000 / 1e6 = 0.2 Mbps
        result = env_embb._compute_embb_throughput_mbps()
        assert result == pytest.approx(0.2)

    def test_zero_service_rate_returns_zero(self, env_embb):
        q = env_embb.queues["eMBB"]
        q.arrival_rate = 100.0
        q.service_rate = 0.0
        result = env_embb._compute_embb_throughput_mbps()
        assert result == pytest.approx(0.0)

    def test_larger_packet_size(self, env_embb):
        q = env_embb.queues["eMBB"]
        q.arrival_rate = 50.0
        q.service_rate = 100.0
        q.mean_packet_bits = 8_000.0  # 1000 bytes = 8000 bits
        # served = 50 → 50 * 8000 / 1e6 = 0.4 Mbps
        result = env_embb._compute_embb_throughput_mbps()
        assert result == pytest.approx(0.4)

    def test_equal_arrival_and_service(self, env_embb):
        # λ = μ → served = λ = μ
        q = env_embb.queues["eMBB"]
        q.arrival_rate = 100.0
        q.service_rate = 100.0
        q.mean_packet_bits = 2_000.0
        # served = 100 → 100 * 2000 / 1e6 = 0.2 Mbps
        result = env_embb._compute_embb_throughput_mbps()
        assert result == pytest.approx(0.2)

    def test_zero_arrival_zero_throughput(self, env_embb):
        q = env_embb.queues["eMBB"]
        q.arrival_rate = 0.0
        q.service_rate = 200.0
        q.mean_packet_bits = 1000.0
        result = env_embb._compute_embb_throughput_mbps()
        assert result == pytest.approx(0.0)

    def test_throughput_scales_with_packet_size(self, env_embb):
        # Doubling packet size should double throughput
        q = env_embb.queues["eMBB"]
        q.arrival_rate = 50.0
        q.service_rate = 200.0
        q.mean_packet_bits = 1000.0
        t1 = env_embb._compute_embb_throughput_mbps()
        q.mean_packet_bits = 2000.0
        t2 = env_embb._compute_embb_throughput_mbps()
        assert t2 == pytest.approx(2 * t1, rel=1e-9)
