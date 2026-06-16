"""Week 3 unit tests — aoi_tracker (phase_detector removed 2026-06-14)."""

from __future__ import annotations

import numpy as np
import pytest

# ============================================================
# AoI tracker
# ============================================================


class TestAoIStreamClassification:
    def test_ambulance_status_lcfs_with_drop_old(self):
        from env.aoi_tracker import AoIStreamTracker

        t = AoIStreamTracker.from_spec("ambulance_status")
        assert t.queue_kind == "LCFS"
        assert t.drop_old

    def test_unknown_stream_raises(self):
        from env.aoi_tracker import AoIStreamTracker

        with pytest.raises(KeyError):
            AoIStreamTracker.from_spec("UNKNOWN")


class TestLCFSBehavior:
    def test_drop_old_keeps_only_newest(self):
        from env.aoi_tracker import AoIStreamTracker

        t = AoIStreamTracker.from_spec("ambulance_status")
        t.arrive(gen_time=0.0, payload_id=0)
        t.arrive(gen_time=0.1, payload_id=1)
        t.arrive(gen_time=0.2, payload_id=2)
        # 2 older were dropped, queue has only newest
        assert t.dropped_count == 2
        assert len(t.queue) == 1

    def test_deliver_picks_freshest(self):
        from env.aoi_tracker import AoIStreamTracker

        t = AoIStreamTracker.from_spec("ambulance_status")
        t.arrive(gen_time=0.0, payload_id=0)
        t.arrive(gen_time=0.1, payload_id=1)
        pkt = t.deliver_next(sim_time=0.15)
        assert pkt is not None
        assert pkt.payload_id == 1   # newest
        assert pkt.deliver_time == 0.15


class TestAoIComputation:
    def test_no_delivery_aoi_equals_elapsed(self):
        from env.aoi_tracker import AoIStreamTracker

        t = AoIStreamTracker.from_spec("ambulance_status")
        assert t.current_aoi(sim_time=5.0) == 5.0

    def test_aoi_after_delivery(self):
        from env.aoi_tracker import AoIStreamTracker

        t = AoIStreamTracker.from_spec("ambulance_status")
        t.arrive(gen_time=10.0)
        t.deliver_next(sim_time=10.05)
        # at t=10.1, AoI = 10.1 - 10.0 = 0.1
        assert t.current_aoi(sim_time=10.1) == pytest.approx(0.1)

    def test_violation_rate(self):
        from env.aoi_tracker import AoIStreamTracker

        t = AoIStreamTracker.from_spec("ambulance_status")
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


class TestAoIThresholdsBySeverity:
    def test_ambulance_status_threshold_at_immediate(self):
        from env.aoi_tracker import aoi_threshold_for_severity
        # severity 5 IMMEDIATE ambulance_status → AoI_max = 0.1s
        assert aoi_threshold_for_severity(5, "ambulance_status") == pytest.approx(0.1)

    def test_unknown_stream_no_aoi_threshold(self):
        from env.aoi_tracker import aoi_threshold_for_severity
        assert aoi_threshold_for_severity(5, "unknown_stream") == float("inf")


# MEC model removed (B0b): compute-offload (MECServer/MECTask/offload_decision)
# deleted. Ambulances are URLLC-only transmit; D_e2e has no D_MEC term
# (D_FH/D_BH delays retained) — see docs/04_data_flow.md.


# Vital simulator removed (B0b): fake physiological value generation deleted.
# URLLC telemetry is modeled as a single consolidated "ambulance_status" AoI
# stream per ambulance (timestamp + size only, F=1, 2026-06-14 consolidation),
# not synthesized vital values — see env/traffic_gen.py + env/aoi_tracker.py.
