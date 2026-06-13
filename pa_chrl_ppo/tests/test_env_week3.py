"""Week 3 unit tests — phase_detector, aoi_tracker, mec_model, vital_simulator."""

from __future__ import annotations

import numpy as np
import pytest


# ============================================================
# Phase FSM
# ============================================================


class TestPhaseFSM:
    def test_initial_state_is_standby(self):
        from env.phase_detector import Phase, PhaseDetector

        det = PhaseDetector()
        assert det.current_phase == Phase.STANDBY

    def test_normal_forward_chain(self):
        from env.phase_detector import Phase, PhaseDetector

        det = PhaseDetector(rng=np.random.default_rng(0))
        events = [
            ("dispatch_call_received", Phase.DISPATCH),
            ("arrived_at_scene",       Phase.SCENE),
            ("patient_loaded",         Phase.TRANSPORT),
            ("arrived_at_hospital",    Phase.RETURN),
            ("return_to_station",      Phase.STANDBY),
        ]
        t = 0.0
        for evt, expected in events:
            assert det.trigger(evt, t)
            assert det.current_phase == expected
            t += 1.0

    def test_invalid_event_no_transition(self):
        from env.phase_detector import Phase, PhaseDetector

        det = PhaseDetector(rng=np.random.default_rng(0))
        # STANDBY does not accept "arrived_at_hospital"
        assert not det.trigger("arrived_at_hospital", 0.0)
        assert det.current_phase == Phase.STANDBY

    def test_sudden_event_jumps_to_scene_from_any_state(self):
        from env.phase_detector import Phase, PhaseDetector

        for start in (Phase.STANDBY, Phase.DISPATCH, Phase.TRANSPORT, Phase.RETURN):
            det = PhaseDetector(current_phase=start, rng=np.random.default_rng(0))
            assert det.trigger("collision_shock", 0.5)
            assert det.current_phase == Phase.SCENE

    def test_sudden_event_idempotent_when_already_in_scene(self):
        from env.phase_detector import Phase, PhaseDetector

        det = PhaseDetector(current_phase=Phase.SCENE, rng=np.random.default_rng(0))
        assert not det.trigger("collision_shock", 0.5)

    def test_history_recorded(self):
        from env.phase_detector import Phase, PhaseDetector

        det = PhaseDetector(rng=np.random.default_rng(0))
        det.trigger("dispatch_call_received", 1.0)
        det.trigger("arrived_at_scene", 5.0)
        assert len(det.transition_history) == 2
        assert det.transition_history[0] == (1.0, Phase.STANDBY, Phase.DISPATCH, "dispatch_call_received")


class TestPhaseSignalingDelay:
    def test_normal_transition_delay_in_10_50ms(self):
        from env.phase_detector import PhaseDetector

        det = PhaseDetector(rng=np.random.default_rng(0), drop_probability=0.0)
        det.trigger("dispatch_call_received", sim_time=0.0)
        # Report time should be at 10-50ms
        assert 10e-3 <= det.last_report_time <= 50e-3

    def test_sudden_event_uses_fast_path(self):
        from env.phase_detector import PhaseDetector

        det = PhaseDetector(rng=np.random.default_rng(0))
        det.trigger("collision_shock", sim_time=0.0)
        # MAC CE single TTI
        assert det.last_report_time == pytest.approx(0.5e-3)

    def test_observed_phase_delayed(self):
        from env.phase_detector import Phase, PhaseDetector

        det = PhaseDetector(rng=np.random.default_rng(123), drop_probability=0.0)
        det.trigger("dispatch_call_received", sim_time=0.0)
        # Immediately, xApp still sees STANDBY (delay 10-50ms)
        assert det.observed_phase(sim_time=0.001) == Phase.STANDBY
        # After 60ms it sees DISPATCH
        assert det.observed_phase(sim_time=0.060) == Phase.DISPATCH

    def test_drop_keeps_stale_phase(self):
        from env.phase_detector import Phase, PhaseDetector

        det = PhaseDetector(rng=np.random.default_rng(0), drop_probability=1.0)
        det.trigger("dispatch_call_received", sim_time=0.0)
        # Drop=1 → xApp never sees the transition (last_report_time stays old)
        assert det.observed_phase(sim_time=1.0) == Phase.STANDBY


class TestPreTightening:
    def test_no_eta_no_pretighten(self):
        from env.phase_detector import Phase, PhaseDetector

        det = PhaseDetector(current_phase=Phase.DISPATCH)
        assert not det.should_pretighten()

    def test_far_eta_no_pretighten(self):
        from env.phase_detector import Phase, PhaseDetector

        det = PhaseDetector(current_phase=Phase.DISPATCH)
        det.set_eta_to_next_phase(60.0)  # 60s away
        assert not det.should_pretighten()

    def test_close_eta_pretighten_if_next_is_scene(self):
        from env.phase_detector import Phase, PhaseDetector

        det = PhaseDetector(current_phase=Phase.DISPATCH)
        det.set_eta_to_next_phase(15.0)  # < 30s
        assert det.should_pretighten()
        # effective_qos picks SCENE thresholds
        qos = det.effective_qos()
        assert qos["D_max"] == 1e-3       # φ₃ SCENE D_max

    def test_pretighten_not_for_return_phase(self):
        from env.phase_detector import Phase, PhaseDetector

        det = PhaseDetector(current_phase=Phase.TRANSPORT)
        det.set_eta_to_next_phase(10.0)
        # next is RETURN (φ₅) which is relaxed — no pre-tighten
        assert not det.should_pretighten()

    def test_negative_eta_rejected(self):
        from env.phase_detector import PhaseDetector

        det = PhaseDetector()
        with pytest.raises(ValueError):
            det.set_eta_to_next_phase(-1.0)


# ============================================================
# AoI tracker
# ============================================================


class TestAoIStreamClassification:
    def test_hr_lcfs_with_drop_old(self):
        from env.aoi_tracker import AoIStreamTracker

        t = AoIStreamTracker.from_spec("HR_aggregated")
        assert t.queue_kind == "LCFS"
        assert t.drop_old

    def test_ecg_fcfs_no_drop(self):
        from env.aoi_tracker import AoIStreamTracker

        t = AoIStreamTracker.from_spec("ECG_waveform")
        assert t.queue_kind == "FCFS"
        assert not t.drop_old

    def test_unknown_stream_raises(self):
        from env.aoi_tracker import AoIStreamTracker

        with pytest.raises(KeyError):
            AoIStreamTracker.from_spec("UNKNOWN")


class TestLCFSBehavior:
    def test_drop_old_keeps_only_newest(self):
        from env.aoi_tracker import AoIStreamTracker

        t = AoIStreamTracker.from_spec("HR_aggregated")
        t.arrive(gen_time=0.0, payload_id=0)
        t.arrive(gen_time=0.1, payload_id=1)
        t.arrive(gen_time=0.2, payload_id=2)
        # 2 older were dropped, queue has only newest
        assert t.dropped_count == 2
        assert len(t.queue) == 1

    def test_deliver_picks_freshest(self):
        from env.aoi_tracker import AoIStreamTracker

        t = AoIStreamTracker.from_spec("HR_aggregated")
        t.arrive(gen_time=0.0, payload_id=0)
        t.arrive(gen_time=0.1, payload_id=1)
        pkt = t.deliver_next(sim_time=0.15)
        assert pkt is not None
        assert pkt.payload_id == 1   # newest
        assert pkt.deliver_time == 0.15


class TestFCFSBehavior:
    def test_fcfs_preserves_order(self):
        from env.aoi_tracker import AoIStreamTracker

        t = AoIStreamTracker.from_spec("ECG_waveform")
        for i in range(5):
            t.arrive(gen_time=0.01 * i, payload_id=i)
        delivered = []
        for k in range(5):
            pkt = t.deliver_next(sim_time=0.1 + 0.01 * k)
            assert pkt is not None
            delivered.append(pkt.payload_id)
        assert delivered == [0, 1, 2, 3, 4]


class TestAoIComputation:
    def test_no_delivery_aoi_equals_elapsed(self):
        from env.aoi_tracker import AoIStreamTracker

        t = AoIStreamTracker.from_spec("HR_aggregated")
        assert t.current_aoi(sim_time=5.0) == 5.0

    def test_aoi_after_delivery(self):
        from env.aoi_tracker import AoIStreamTracker

        t = AoIStreamTracker.from_spec("HR_aggregated")
        t.arrive(gen_time=10.0)
        t.deliver_next(sim_time=10.05)
        # at t=10.1, AoI = 10.1 - 10.0 = 0.1
        assert t.current_aoi(sim_time=10.1) == pytest.approx(0.1)

    def test_violation_rate(self):
        from env.aoi_tracker import AoIStreamTracker

        t = AoIStreamTracker.from_spec("HR_aggregated")
        # Inject AoI samples manually
        t.aoi_samples = [0.05, 0.15, 0.08, 0.20, 0.12]
        # threshold 0.1: 3 of 5 exceed
        assert t.violation_rate(0.1) == pytest.approx(3 / 5)

    def test_mm1_aoi_kaul_2012(self):
        from env.aoi_tracker import expected_aoi_mm1

        # ρ=0.5, μ=100 → E[AoI] = (1/100)·(1 + 0.5/0.5 + 0.25/0.75)
        e = expected_aoi_mm1(50.0, 100.0)
        expected = (1.0 / 100.0) * (1.0 + 1.0 + 1.0 / 3.0)
        assert e == pytest.approx(expected, rel=1e-6)

    def test_mm1_aoi_unstable_inf(self):
        from env.aoi_tracker import expected_aoi_mm1

        assert expected_aoi_mm1(100.0, 100.0) == float("inf")


class TestAoIThresholdsByPhase:
    def test_hr_threshold_at_phi3(self):
        from env.aoi_tracker import aoi_threshold_for_phase
        # phase 3 HR_aggregated → AoI_max_HR = 0.1s
        assert aoi_threshold_for_phase(3, "HR_aggregated") == pytest.approx(0.1)

    def test_waveform_no_aoi_threshold(self):
        from env.aoi_tracker import aoi_threshold_for_phase
        assert aoi_threshold_for_phase(3, "ECG_waveform") == float("inf")


# ============================================================
# MEC model
# ============================================================


class TestMECDelay:
    def test_offloaded_delay_formula(self):
        from env.mec_model import MECTask

        # L=8000 bits, R_UL=10 Mbps, W=100 cycles/bit, f_MEC=1 GHz
        # D = 8000/1e7 + 100*8000/1e9 = 0.0008 + 0.0008 = 0.0016s
        t = MECTask(
            task_id=0, task_type="vital_sign_process", size_bits=8000,
            W_cycles_per_bit=100.0, f_mec_allocation_hz=1e9,
            uplink_rate_bps=10e6,
        )
        assert t.delay_offloaded() == pytest.approx(0.0016)

    def test_local_delay_larger(self):
        from env.mec_model import MECTask
        t = MECTask(
            task_id=0, task_type="video_analytics", size_bits=4_000_000,
            W_cycles_per_bit=1200.0, f_mec_allocation_hz=2e9,
            f_local_hz=0.5e9,
        )
        assert t.delay_local() > t.delay_offloaded()


class TestMECServer:
    def test_admission_within_budget(self):
        from env.mec_model import MECServer, MECTask
        s = MECServer(f_total_hz=10e9)
        t = MECTask(task_id=0, task_type="video_analytics", size_bits=1000,
                    W_cycles_per_bit=1000, f_mec_allocation_hz=2e9)
        assert s.can_admit(t)
        assert s.admit(t)
        assert s.utilization == pytest.approx(0.2)

    def test_admission_rejected_when_full(self):
        from env.mec_model import MECServer, MECTask
        s = MECServer(f_total_hz=10e9)
        for i in range(5):
            t = MECTask(task_id=i, task_type="video_analytics", size_bits=1000,
                        W_cycles_per_bit=1000, f_mec_allocation_hz=2e9)
            assert s.admit(t)
        # 6th task should fail
        t6 = MECTask(task_id=6, task_type="video_analytics", size_bits=1000,
                     W_cycles_per_bit=1000, f_mec_allocation_hz=1e8)
        assert s.can_admit(t6) is False
        assert s.admit(t6) is False

    def test_release_frees_budget(self):
        from env.mec_model import MECServer, MECTask
        s = MECServer(f_total_hz=10e9)
        t = MECTask(task_id=42, task_type="image_recognition", size_bits=1000,
                    W_cycles_per_bit=800, f_mec_allocation_hz=3e9)
        s.admit(t)
        assert s.utilization == pytest.approx(0.3)
        assert s.release(42)
        assert s.utilization == 0.0


class TestOffloadDecision:
    def test_video_offload_when_sinr_high_and_mec_light(self):
        from env.mec_model import offload_decision
        assert offload_decision("video_analytics", sinr_db=15.0, mec_load=0.3) is True

    def test_video_local_when_sinr_low(self):
        from env.mec_model import offload_decision
        assert offload_decision("video_analytics", sinr_db=2.0, mec_load=0.3) is False

    def test_video_local_when_mec_full(self):
        from env.mec_model import offload_decision
        assert offload_decision("video_analytics", sinr_db=15.0, mec_load=0.95) is False

    def test_vital_signs_always_local(self):
        from env.mec_model import offload_decision
        assert offload_decision("vital_sign_process", sinr_db=20.0, mec_load=0.1) is False


class TestTotalMECDelay:
    def test_offloaded_path(self):
        from env.mec_model import MECServer, MECTask, total_mec_delay
        s = MECServer(f_total_hz=10e9)
        t = MECTask(task_id=0, task_type="video_analytics", size_bits=10_000,
                    W_cycles_per_bit=1000, f_mec_allocation_hz=1e9,
                    uplink_rate_bps=50e6, f_local_hz=0.5e9)
        delay, off, reason = total_mec_delay(t, s, sinr_db=15.0)
        assert off
        assert reason == "offloaded"
        assert delay == pytest.approx(t.delay_offloaded())

    def test_local_when_mec_overloaded(self):
        from env.mec_model import MECServer, MECTask, total_mec_delay
        # Pre-fill server
        s = MECServer(f_total_hz=10e9)
        for i in range(5):
            s.admit(MECTask(task_id=i, task_type="video_analytics", size_bits=1,
                            W_cycles_per_bit=1, f_mec_allocation_hz=2e9))
        # Now utilization=1.0; offload rule says no anyway (mec_load≥0.8)
        t = MECTask(task_id=99, task_type="video_analytics", size_bits=10_000,
                    W_cycles_per_bit=1000, f_mec_allocation_hz=1e9,
                    f_local_hz=0.5e9)
        delay, off, reason = total_mec_delay(t, s, sinr_db=15.0)
        assert not off
        assert delay == pytest.approx(t.delay_local())


# ============================================================
# Vital simulator
# ============================================================


class TestVitalSimulator:
    def test_ecg_shape_and_amplitude(self):
        from env.vital_simulator import synth_ecg

        sig = synth_ecg(duration_sec=2.0, sampling_hz=250.0,
                        heart_rate_bpm=80.0, rng=np.random.default_rng(0))
        assert sig.shape == (500,)
        # R peak amplitude ~1 mV, with noise → max should be > 0.5
        assert sig.max() > 0.5

    def test_spo2_in_valid_range(self):
        from env.vital_simulator import synth_spo2

        sig = synth_spo2(duration_sec=60.0, rng=np.random.default_rng(0))
        assert sig.shape == (60,)
        assert sig.min() >= 70.0
        assert sig.max() <= 100.0

    def test_heart_rate_in_physiological_range(self):
        from env.vital_simulator import synth_heart_rate

        hr = synth_heart_rate(duration_sec=120.0, baseline_bpm=80.0,
                              rng=np.random.default_rng(0))
        assert hr.shape == (120,)
        assert hr.min() >= 40.0
        assert hr.max() <= 220.0

    def test_speed_zero_in_standby(self):
        from env.vital_simulator import synth_speed_kmh

        sp = synth_speed_kmh(duration_sec=30.0, phase=1,
                              rng=np.random.default_rng(0))
        assert np.all(sp == 0.0)

    def test_speed_nonzero_in_dispatch(self):
        from env.vital_simulator import synth_speed_kmh

        sp = synth_speed_kmh(duration_sec=30.0, phase=2,
                              rng=np.random.default_rng(0))
        assert sp.mean() > 0.0

    def test_trace_bundle_consistent_shapes(self):
        from env.vital_simulator import generate_trace_bundle

        bundle = generate_trace_bundle(
            duration_sec=10.0,
            phase=4,
            rng=np.random.default_rng(0),
            ecg_sampling_hz=250.0,
            slow_sampling_hz=1.0,
        )
        assert bundle.ecg.shape == (2500,)        # 10s * 250 Hz
        assert bundle.spo2.shape == (10,)         # 10s * 1 Hz
        assert bundle.heart_rate.shape == (10,)
        assert bundle.speed_kmh.shape == (10,)
