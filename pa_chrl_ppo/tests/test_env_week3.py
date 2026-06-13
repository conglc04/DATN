"""Week 3 unit tests — phase_detector, aoi_tracker."""

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


# MEC model removed (B0b): compute-offload (MECServer/MECTask/offload_decision)
# deleted. Ambulances are URLLC-only transmit; D_e2e has no D_MEC term
# (D_FH/D_BH delays retained) — see docs/04_data_flow.md.


# Vital simulator removed (B0b): fake physiological value generation deleted.
# URLLC telemetry is modeled as F=4 traffic streams (timestamp + size only),
# not synthesized vital values — see env/traffic_gen.py + env/aoi_tracker.py.
