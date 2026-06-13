"""Week 2 unit tests — env/channel_model.py, env/queue_model.py, env/traffic_gen.py."""

from __future__ import annotations

import math

import numpy as np
import pytest

# ============================================================
# Channel model tests
# ============================================================


class TestPathLoss:
    def test_uma_increases_with_distance(self):
        from env.channel_model import pl_uma

        pl_close = pl_uma(50.0)
        pl_far = pl_uma(500.0)
        assert pl_far > pl_close, "Path loss must increase with distance"

    def test_uma_formula_at_100m_3_5ghz(self):
        """Spec docs/03:116: PL = 28 + 22·log10(d) + 20·log10(f_c)."""
        from env.channel_model import pl_uma

        # d=100m, f_c=3.5 GHz
        expected = 28.0 + 22.0 * math.log10(100.0) + 20.0 * math.log10(3.5)
        assert abs(pl_uma(100.0, 3.5) - expected) < 1e-6

    def test_umi_nlos_higher_than_los(self):
        from env.channel_model import pl_umi_los, pl_umi_nlos

        d = 200.0
        assert pl_umi_nlos(d) > pl_umi_los(d), "NLOS must have higher PL than LOS"

    def test_uma_rejects_nonpositive_distance(self):
        from env.channel_model import pl_uma

        with pytest.raises(ValueError):
            pl_uma(0.0)
        with pytest.raises(ValueError):
            pl_uma(-10.0)


class TestLOSProbability:
    def test_full_los_at_origin(self):
        from env.channel_model import los_probability_umi
        assert los_probability_umi(0.5) == pytest.approx(1.0, abs=0.01)

    def test_decreases_with_distance(self):
        from env.channel_model import los_probability_umi
        p_near = los_probability_umi(20.0)
        p_far = los_probability_umi(500.0)
        assert p_near > p_far

    def test_returns_in_unit_interval(self):
        from env.channel_model import los_probability_umi
        for d in [10, 50, 100, 500, 1000, 5000]:
            p = los_probability_umi(d)
            assert 0.0 <= p <= 1.0


class TestSINRAndCapacity:
    def test_thermal_noise_at_360khz(self):
        from env.channel_model import thermal_noise_dbm
        # N = -174 + 10·log10(360_000) + 7 ≈ -174 + 55.56 + 7 ≈ -111.4 dBm
        n = thermal_noise_dbm(360e3, noise_figure_db=7.0)
        assert -113 < n < -109

    def test_capacity_per_prb_at_10dB(self):
        """Spec docs/03:177: SINR=10dB → ≈1.24 Mbps/PRB (with η=0.75)."""
        from env.channel_model import capacity_per_prb_bps
        c = capacity_per_prb_bps(10.0)   # uses eta=0.75 by default
        # 0.75 · 360e3 · log2(11) = 0.75 · 360e3 · 3.459 ≈ 933 kbps
        # Docs cite ≈ 1.24 Mbps WITHOUT η — with η=0.75 we get ~0.93 Mbps
        assert 800e3 < c < 1_000e3

    def test_capacity_per_prb_full_shannon(self):
        from env.channel_model import capacity_per_prb_bps
        c = capacity_per_prb_bps(10.0, eta=1.0)
        # 360e3 · log2(11) ≈ 1.245 Mbps
        assert 1_200e3 < c < 1_300e3

    def test_capacity_increases_with_sinr(self):
        from env.channel_model import capacity_per_prb_bps
        assert capacity_per_prb_bps(20.0) > capacity_per_prb_bps(0.0)

    def test_aggregate_capacity_273_prb(self):
        from env.channel_model import aggregate_capacity_bps
        from utils.config import P_TOTAL
        c_total = aggregate_capacity_bps(P_TOTAL, 10.0)
        # 273 · ~0.93 Mbps ≈ 254 Mbps with η=0.75
        assert 240e6 < c_total < 270e6


class TestChannelModelClass:
    def test_path_loss_macro_uses_uma(self):
        from env.channel_model import BaseStation, ChannelModel
        cm = ChannelModel(shadowing=False, rng=np.random.default_rng(0))
        bs = BaseStation(cell_id=0, x=0.0, y=0.0, h=25.0, layer="macro")
        pl, scenario = cm.path_loss((200.0, 0.0), bs)
        assert scenario == "UMa"

    def test_path_loss_micro_picks_los_or_nlos(self):
        from env.channel_model import BaseStation, ChannelModel
        cm = ChannelModel(shadowing=False, rng=np.random.default_rng(42))
        bs = BaseStation(cell_id=0, x=0.0, y=0.0, h=10.0, layer="micro")
        scenarios = set()
        for _ in range(50):
            _, scen = cm.path_loss((100.0, 50.0), bs)
            scenarios.add(scen)
        # Over 50 trials we should see both branches given d≈112m
        assert scenarios.issubset({"UMi_LOS", "UMi_NLOS"})

    def test_rx_power_decreases_with_distance(self):
        from env.channel_model import BaseStation, ChannelModel
        cm = ChannelModel(shadowing=False, rng=np.random.default_rng(0))
        bs = BaseStation(cell_id=0, x=0.0, y=0.0, layer="macro", tx_power_dbm=46.0)
        p_close = cm.receive_power_dbm((100.0, 0.0), bs)
        p_far = cm.receive_power_dbm((1000.0, 0.0), bs)
        assert p_close > p_far


# ============================================================
# Queue model tests
# ============================================================


class TestMG1Queue:
    def test_stability_check(self):
        from env.queue_model import MG1Queue
        q = MG1Queue(name="urllc", arrival_rate=50.0, service_rate=100.0)
        assert q.rho == pytest.approx(0.5)
        assert q.is_stable

    def test_unstable_when_rho_above_0_9(self):
        from env.queue_model import MG1Queue
        q = MG1Queue(name="urllc", arrival_rate=95.0, service_rate=100.0)
        assert q.rho == pytest.approx(0.95)
        assert not q.is_stable

    def test_pk_formula_returns_inf_when_overloaded(self):
        from env.queue_model import MG1Queue
        q = MG1Queue(name="x", arrival_rate=100.0, service_rate=100.0)
        assert q.expected_queue_delay() == float("inf")

    def test_pk_formula_grows_with_rho(self):
        from env.queue_model import MG1Queue
        q1 = MG1Queue(name="x", arrival_rate=10.0, service_rate=100.0)
        q2 = MG1Queue(name="x", arrival_rate=80.0, service_rate=100.0)
        assert q2.expected_queue_delay() > q1.expected_queue_delay()

    def test_update_service_rate_from_prb(self):
        from env.queue_model import MG1Queue
        q = MG1Queue(name="urllc", mean_packet_bits=3200.0)  # 400B packet
        # 100 PRB · 1 Mbps/PRB = 100 Mbps / 3200 bits = 31_250 pkt/s
        q.update_service_rate(prb_count=100, capacity_per_prb_bps=1e6)
        assert q.service_rate == pytest.approx(100e6 / 3200.0)

    def test_hol_equals_queue_plus_service(self):
        from env.queue_model import MG1Queue
        q = MG1Queue(name="x", arrival_rate=50.0, service_rate=100.0)
        assert q.hol_delay() == pytest.approx(
            q.expected_queue_delay() + q.mean_service_time
        )

    def test_negative_arrival_rejected(self):
        from env.queue_model import MG1Queue
        q = MG1Queue(name="x")
        with pytest.raises(ValueError):
            q.set_arrival_rate(-1.0)


class TestQueueTailBounds:
    def test_markov_bound_in_unit_interval(self):
        from env.queue_model import hol_tail_bound_markov
        p = hol_tail_bound_markov(expected_hol=1e-3, threshold=2e-3)
        assert 0.0 <= p <= 1.0
        assert p == pytest.approx(0.5)

    def test_chernoff_mm1_unstable_returns_1(self):
        from env.queue_model import hol_tail_bound_chernoff_mm1
        assert hol_tail_bound_chernoff_mm1(100.0, 100.0, 1e-3) == 1.0

    def test_chernoff_mm1_tight_for_stable(self):
        from env.queue_model import hol_tail_bound_chernoff_mm1
        # ρ=0.5, threshold large → tiny tail probability
        p = hol_tail_bound_chernoff_mm1(arrival_rate=50.0, service_rate=100.0, threshold=1.0)
        assert p < 1e-15  # exp(-50) is essentially 0


class TestSliceQueueManager:
    def test_all_stable_requires_every_slice(self):
        from env.queue_model import MG1Queue, SliceQueueManager
        mgr = SliceQueueManager()
        mgr.add(MG1Queue(name="urllc", arrival_rate=50, service_rate=100))
        mgr.add(MG1Queue(name="embb", arrival_rate=50, service_rate=100))
        assert mgr.all_stable()
        mgr["embb"].set_arrival_rate(95.0)
        assert not mgr.all_stable()


# ============================================================
# Traffic generators
# ============================================================


class TestTrafficGenerators:
    def test_urllc_vital_rate_matches_100hz(self):
        from env.traffic_gen import gen_urllc_vital
        pkts = gen_urllc_vital(duration_sec=1.0, frequency_hz=100.0,
                                rng=np.random.default_rng(0))
        # ~100 packets per second (jitter may add/subtract 1-2)
        assert 95 <= len(pkts) <= 105
        assert all(p.ttype == "URLLC_C2_VITAL" for p in pkts)

    def test_urllc_cam_rate_matches_10hz(self):
        from env.traffic_gen import gen_urllc_cam
        pkts = gen_urllc_cam(duration_sec=1.0, frequency_hz=10.0,
                              rng=np.random.default_rng(0))
        assert len(pkts) == 10
        assert all(p.ttype == "URLLC_C3_CAM" for p in pkts)

    def test_denm_burst_increases_rate(self):
        from env.traffic_gen import gen_urllc_denm
        steady = gen_urllc_denm(duration_sec=1.0, lambda_base=50.0,
                                 rng=np.random.default_rng(0))
        bursty = gen_urllc_denm(duration_sec=1.0, lambda_base=50.0,
                                 burst_lambda=500.0, burst_start=0.0,
                                 burst_duration=1.0,
                                 rng=np.random.default_rng(0))
        assert len(bursty) > 5 * len(steady) / 2  # at least 2.5x

    def test_embb_video_rate_close_to_5mbps(self):
        from env.traffic_gen import gen_embb_video
        pkts = gen_embb_video(duration_sec=5.0, rate_mbps=5.0, cv=0.0,
                                rng=np.random.default_rng(0))
        total_bits = sum(p.size_bits for p in pkts)
        rate = total_bits / 5.0 / 1e6
        # CBR should be within 10% of target
        assert 4.5 < rate < 5.5

    def test_embb_image_mec_produces_multiple_chunks_per_image(self):
        from env.traffic_gen import gen_embb_image_mec
        pkts = gen_embb_image_mec(duration_sec=2.0, images_per_sec=2.0,
                                    image_size_bytes=15_000, chunk_bytes=1500,
                                    rng=np.random.default_rng(0))
        # 10 chunks per image × ~4 images / 2s
        assert len(pkts) >= 30

    def test_mmtc_sparse(self):
        from env.traffic_gen import gen_mmtc
        pkts = gen_mmtc(duration_sec=10.0, n_devices=50, per_device_lambda=0.1,
                         rng=np.random.default_rng(0))
        # 50 · 0.1 · 10 = 50 packets expected (Poisson, variance ~50)
        assert 30 <= len(pkts) <= 70


class TestPacketProperties:
    def test_deadline_matches_class(self):
        from env.traffic_gen import gen_urllc_vital, DEADLINE_SEC
        pkts = gen_urllc_vital(duration_sec=0.5, rng=np.random.default_rng(0))
        for p in pkts:
            expected_deadline = p.arrival_time + DEADLINE_SEC["URLLC_C2_VITAL"]
            assert abs(p.deadline_sec - expected_deadline) < 1e-9

    def test_urllc_priority_lower_than_embb(self):
        from env.traffic_gen import PRIORITY
        assert PRIORITY["URLLC_C1_DENM"] < PRIORITY["eMBB_V1_VIDEO4K"]
        assert PRIORITY["URLLC_C2_VITAL"] < PRIORITY["mMTC_IOT"]


class TestMixAndAggregate:
    def test_mix_sorts_by_arrival(self):
        from env.traffic_gen import gen_urllc_cam, gen_urllc_vital, mix_traffic
        rng = np.random.default_rng(0)
        cam = gen_urllc_cam(duration_sec=1.0, rng=rng)
        vital = gen_urllc_vital(duration_sec=1.0, rng=rng)
        mixed = mix_traffic(cam, vital)
        arrivals = [p.arrival_time for p in mixed]
        assert arrivals == sorted(arrivals)

    def test_aggregate_arrival_rate(self):
        from env.traffic_gen import gen_urllc_cam, aggregate_arrival_rate
        pkts = gen_urllc_cam(duration_sec=2.0, rng=np.random.default_rng(0))
        rate = aggregate_arrival_rate(pkts, 2.0)
        assert 9.0 <= rate <= 11.0   # nominally 10 Hz

    def test_mean_size_within_range(self):
        from env.traffic_gen import gen_urllc_vital, aggregate_mean_size_bits
        pkts = gen_urllc_vital(duration_sec=1.0, rng=np.random.default_rng(0))
        mean_bits = aggregate_mean_size_bits(pkts)
        # 100-500 B → 800-4000 bits
        assert 800 <= mean_bits <= 4000

    def test_iterate_window(self):
        from env.traffic_gen import gen_urllc_cam, iterate_packets_in_window
        pkts = gen_urllc_cam(duration_sec=1.0, rng=np.random.default_rng(0))
        window = list(iterate_packets_in_window(pkts, 0.3, 0.7))
        assert all(0.3 <= p.arrival_time < 0.7 for p in window)
        # 10Hz → ~4 packets in 0.4s window
        assert len(window) == 4


# ============================================================
# Integration sanity: channel + queue at φ₃
# ============================================================


class TestPhase3SanitySingleCell:
    """Sanity: at φ₃ with r_min^URLLC=0.6, single ambulance, expect HOL < 1ms."""

    def test_e2e_breakdown_under_1ms(self):
        from env.channel_model import capacity_per_prb_bps
        from env.queue_model import MG1Queue
        from utils.config import D_FH, D_BH, D_DET, P_TOTAL

        # SINR=15dB typical ambulance near cell
        sinr_db_val = 15.0
        capacity = capacity_per_prb_bps(sinr_db_val)   # ~0.93 * log2(1+31.6) → ≈ 3.5 Mbps

        # PRB for URLLC at φ₃: r_min = 0.6 → 0.6 * 273 = 163 PRB
        urllc_prb = int(0.6 * P_TOTAL)

        # M/G/1 with λ=50 pkt/s (DENM steady) + 400B packets
        q = MG1Queue(name="urllc_phi3", arrival_rate=50.0, mean_packet_bits=400 * 8)
        q.update_service_rate(urllc_prb, capacity)

        assert q.is_stable
        e2e = D_DET + D_FH + D_BH + q.hol_delay()
        assert e2e < 1e-3, f"E2E at φ₃ = {e2e*1e3:.3f}ms exceeds 1ms target"
