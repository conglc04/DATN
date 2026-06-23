"""P0-a verification: AoI is receiver-side, gated by PRB + BLER.

Tests confirm that:
  1. Packet counters (arrived/delivered/failed_bler/failed_no_prb) are consistent
  2. BLER gate: some packets fail the Bernoulli draw → delivery rate < 100%
  3. PRB gate: zero PRB → no delivery (unit-level test of _update_aoi_trackers)
  4. AoI is non-trivial (grows between deliveries, resets on success)
  5. K=3 per-ambulance AoI divergence
  6. K=3 PRB split: zero-PRB ambulance has worse AoI than positive-PRB ambulance
"""

from __future__ import annotations

import numpy as np
import pytest

from env.aoi_tracker import AoIStreamTracker
from env.oran_env import EnvConfig, ORANEnv
from utils.config import SEVERITY_QOS


def _make_env(K: int = 1, seed: int = 42) -> ORANEnv:
    cfg = EnvConfig(K_ambulances=K, episode_duration_sec=2.0)
    env = ORANEnv(cfg)
    env.reset(seed=seed)
    return env


def _force_severity(env: ORANEnv, sev: int) -> None:
    env.severity = sev
    env.severity_per_amb = np.full(env.config.K_ambulances, sev, dtype=int)


# ────────────────────────────────────────────────────────────────────
# 1. Packet counter consistency
# ────────────────────────────────────────────────────────────────────

class TestPacketCounters:
    def test_delivered_le_arrived(self):
        """Can't deliver more packets than were generated."""
        env = _make_env(K=1, seed=0)
        for _ in range(100):
            action = env.action_space.sample()
            _obs, _r, term, trunc, info = env.step(action)
            if term or trunc:
                break
        assert (info["aoi_pkt_delivered"] <= info["aoi_pkt_arrived"]).all()

    def test_service_outcomes_nonnegative(self):
        """All counters must be non-negative."""
        env = _make_env(K=1, seed=0)
        for _ in range(100):
            action = env.action_space.sample()
            _obs, _r, term, trunc, info = env.step(action)
            if term or trunc:
                break
        for key in ("aoi_pkt_arrived", "aoi_pkt_delivered",
                    "aoi_pkt_failed_bler", "aoi_pkt_failed_no_prb",
                    "aoi_pkt_failed_no_capacity"):
            assert (info[key] >= 0).all(), f"{key} has negative values"

    def test_counters_nonzero_after_steps(self):
        """After enough steps, at least some packets should have arrived."""
        env = _make_env(K=1, seed=7)
        for _ in range(200):
            action = env.action_space.sample()
            _obs, _r, term, trunc, info = env.step(action)
            if term or trunc:
                break
        assert info["aoi_pkt_arrived"].sum() > 0, "No URLLC arrivals in 200 steps"

    def test_counters_k3_independent(self):
        """Each ambulance in K=3 has its own counters."""
        env = _make_env(K=3, seed=11)
        for _ in range(100):
            action = env.action_space.sample()
            _obs, _r, term, trunc, info = env.step(action)
            if term or trunc:
                break
        for key in ("aoi_pkt_arrived", "aoi_pkt_delivered",
                    "aoi_pkt_failed_bler", "aoi_pkt_failed_no_prb",
                    "aoi_pkt_failed_no_capacity"):
            assert info[key].shape == (3,), f"{key} shape mismatch"

    @pytest.mark.parametrize("K", [1, 3])
    def test_delivered_le_arrived_k(self, K: int):
        """delivered <= arrived for all K."""
        env = _make_env(K=K, seed=55)
        for _ in range(150):
            action = env.action_space.sample()
            _obs, _r, term, trunc, info = env.step(action)
            if term or trunc:
                break
        assert (info["aoi_pkt_delivered"] <= info["aoi_pkt_arrived"]).all()


# ────────────────────────────────────────────────────────────────────
# 2. BLER gate: delivery rate < 100%
# ────────────────────────────────────────────────────────────────────

class TestBLERGate:
    def test_some_packets_fail_bler(self):
        """With typical SINR, some packets should fail the Bernoulli BLER draw."""
        env = _make_env(K=1, seed=99)
        for _ in range(300):
            action = env.action_space.sample()
            _obs, _r, term, trunc, info = env.step(action)
            if term or trunc:
                break
        arrived = int(info["aoi_pkt_arrived"].sum())
        failed_bler = int(info["aoi_pkt_failed_bler"].sum())
        if arrived > 10:
            assert failed_bler > 0, (
                f"No BLER failures in {arrived} arrivals — BLER gate not working"
            )

    def test_delivery_rate_bounded_by_bler(self):
        """Delivery rate should be less than 100% given non-zero BLER."""
        env = _make_env(K=1, seed=42)
        for _ in range(200):
            action = env.action_space.sample()
            _obs, _r, term, trunc, info = env.step(action)
            if term or trunc:
                break
        arrived = int(info["aoi_pkt_arrived"].sum())
        delivered = int(info["aoi_pkt_delivered"].sum())
        if arrived > 20:
            rate = delivered / arrived
            assert 0.0 < rate < 1.0, (
                f"Delivery rate {rate:.3f} out of expected range — "
                f"should be 0 < rate < 1 with non-zero BLER"
            )

    def test_bler_per_amb_exposed_in_info(self):
        """info['bler_per_amb'] should be a list of K floats in [0, 0.5]."""
        env = _make_env(K=3, seed=7)
        _obs, _r, term, trunc, info = env.step(env.action_space.sample())
        assert len(info["bler_per_amb"]) == 3
        for b in info["bler_per_amb"]:
            assert 0.0 <= b <= 0.5


# ────────────────────────────────────────────────────────────────────
# 3. PRB gate (unit-level): zero PRB → no delivery
# ────────────────────────────────────────────────────────────────────

class TestPRBGateUnit:
    def test_zero_prb_blocks_delivery(self):
        """Directly call _update_aoi_trackers with PRB=0 → no deliver_next."""
        env = _make_env(K=1, seed=42)
        env._last_prb_per_amb = np.zeros(1, dtype=np.int64)
        env.last_bler_per_amb = np.array([0.01])
        env.last_sinr_db = np.array([15.0])

        n_urllc = np.array([1], dtype=np.int64)
        env._update_aoi_trackers(n_urllc)

        assert int(env._aoi_pkt_arrived[0]) == 1
        assert int(env._aoi_pkt_failed_no_prb[0]) == 1
        assert int(env._aoi_pkt_delivered[0]) == 0

    def test_positive_prb_allows_delivery(self):
        """With PRB>0, sufficient capacity, and low BLER, delivery should succeed."""
        env = _make_env(K=1, seed=42)
        env._last_prb_per_amb = np.array([100], dtype=np.int64)
        env.last_bler_per_amb = np.array([0.001])
        env.last_sinr_db = np.array([15.0])

        delivered = 0
        for _ in range(100):
            n_urllc = np.array([1], dtype=np.int64)
            env._update_aoi_trackers(n_urllc)
            delivered = int(env._aoi_pkt_delivered[0])
        assert delivered > 90, f"Only {delivered}/100 delivered with BLER=0.001"

    def test_prb_gate_per_amb_k3(self):
        """K=3: amb-0 has PRBs, amb-1 and amb-2 have 0 → only amb-0 delivers."""
        env = _make_env(K=3, seed=42)
        env._last_prb_per_amb = np.array([50, 0, 0], dtype=np.int64)
        env.last_bler_per_amb = np.array([0.01, 0.01, 0.01])
        env.last_sinr_db = np.array([15.0, 15.0, 15.0])

        for _ in range(50):
            n_urllc = np.array([1, 1, 1], dtype=np.int64)
            env._update_aoi_trackers(n_urllc)

        assert int(env._aoi_pkt_delivered[0]) > 0, "Amb-0 should deliver"
        assert int(env._aoi_pkt_delivered[1]) == 0, "Amb-1 should NOT deliver (0 PRBs)"
        assert int(env._aoi_pkt_delivered[2]) == 0, "Amb-2 should NOT deliver (0 PRBs)"
        assert int(env._aoi_pkt_failed_no_prb[1]) == 50
        assert int(env._aoi_pkt_failed_no_prb[2]) == 50


# ────────────────────────────────────────────────────────────────────
# 4. AoI is non-trivial (not instant delivery)
# ────────────────────────────────────────────────────────────────────

class TestAoINonTrivial:
    def test_aoi_positive_after_steps(self):
        """Mean AoI should be positive (not 0 like instant delivery)."""
        env = _make_env(K=1, seed=42)
        aoi_accum = []
        for _ in range(100):
            action = env.action_space.sample()
            _obs, _r, term, trunc, info = env.step(action)
            aoi_accum.append(float(np.mean(info["aoi_norm_per_amb"])))
            if term or trunc:
                break
        mean_aoi_norm = float(np.mean(aoi_accum))
        assert mean_aoi_norm > 0.01, (
            f"Mean AoI norm {mean_aoi_norm:.6f} is too close to 0 — "
            f"receiver-side gate may not be working"
        )

    def test_aoi_tracker_has_failed_and_delivered(self):
        """Over 200 steps, we should see both delivered and failed packets."""
        env = _make_env(K=1, seed=77)
        for _ in range(200):
            action = env.action_space.sample()
            _obs, _r, term, trunc, info = env.step(action)
            if term or trunc:
                break
        arrived = int(info["aoi_pkt_arrived"].sum())
        delivered = int(info["aoi_pkt_delivered"].sum())
        failed = int(info["aoi_pkt_failed_bler"].sum() + info["aoi_pkt_failed_no_prb"].sum())
        if arrived > 10:
            assert delivered > 0, "No deliveries in 200 steps"
            assert failed > 0, "No failures in 200 steps — BLER/PRB gate not working"


# ────────────────────────────────────────────────────────────────────
# 5. K=3 per-ambulance AoI divergence
# ────────────────────────────────────────────────────────────────────

class TestK3AoIDivergence:
    def test_per_amb_aoi_not_all_identical(self):
        """With K=3 at different positions, per-amb AoI should diverge."""
        env = _make_env(K=3, seed=55)
        _force_severity(env, 3)
        for _ in range(200):
            action = env.action_space.sample()
            _obs, _r, term, trunc, info = env.step(action)
            if term or trunc:
                break
        aoi_per = info["aoi_norm_per_amb"]
        assert not np.allclose(aoi_per, aoi_per[0], atol=1e-4), (
            f"All K=3 ambulances have identical AoI — no per-amb differentiation: {aoi_per}"
        )


# ────────────────────────────────────────────────────────────────────
# 6. K=3 PRB split: zero-PRB amb has worse AoI (integration level)
# ────────────────────────────────────────────────────────────────────

class TestK3PRBSplitAoI:
    def test_zero_prb_amb_worse_aoi_unit(self):
        """Unit test: amb with 0 PRBs has AoI = sim_time (never delivered)."""
        env = _make_env(K=3, seed=42)
        env._last_prb_per_amb = np.array([100, 0, 0], dtype=np.int64)
        env.last_bler_per_amb = np.array([0.01, 0.01, 0.01])
        env.last_sinr_db = np.array([15.0, 15.0, 15.0])

        for _ in range(30):
            n_urllc = np.array([1, 1, 1], dtype=np.int64)
            env.sim_time += 0.01
            env._update_aoi_trackers(n_urllc)

        aoi_0 = env.aoi_trackers[0]["ambulance_status"].current_aoi(env.sim_time)
        aoi_1 = env.aoi_trackers[1]["ambulance_status"].current_aoi(env.sim_time)
        aoi_2 = env.aoi_trackers[2]["ambulance_status"].current_aoi(env.sim_time)
        assert aoi_1 > aoi_0, f"Amb-1 (0 PRBs) AoI={aoi_1:.4f} should be > amb-0 AoI={aoi_0:.4f}"
        assert aoi_2 > aoi_0, f"Amb-2 (0 PRBs) AoI={aoi_2:.4f} should be > amb-0 AoI={aoi_0:.4f}"


# ────────────────────────────────────────────────────────────────────
# 7. AoI tracker API consistency
# ────────────────────────────────────────────────────────────────────

class TestAoITrackerAPIConsistency:
    def test_aoi_monotonic_without_delivery(self):
        """Between deliveries, current_aoi() should increase monotonically."""
        t = AoIStreamTracker.from_spec("ambulance_status")
        t.arrive(gen_time=0.0)
        t.deliver_next(sim_time=0.001)
        aoi_prev = t.current_aoi(0.002)
        for dt in [0.005, 0.010, 0.020, 0.050]:
            aoi_now = t.current_aoi(dt)
            assert aoi_now >= aoi_prev, f"AoI decreased: {aoi_prev} → {aoi_now}"
            aoi_prev = aoi_now

    def test_delivery_resets_aoi(self):
        """deliver_next() should drop AoI back to near zero."""
        t = AoIStreamTracker.from_spec("ambulance_status")
        t.arrive(gen_time=0.0)
        t.deliver_next(sim_time=0.001)
        aoi_old = t.current_aoi(0.050)
        assert aoi_old > 0.04

        t.arrive(gen_time=0.050)
        t.deliver_next(sim_time=0.051)
        aoi_new = t.current_aoi(0.052)
        assert aoi_new < 0.01, f"AoI after fresh delivery should be near 0, got {aoi_new}"
