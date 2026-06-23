"""Audit 2026-06-22 — bound delay & AoI; fix delivery stall (fixes A+B+C).

Two symptom groups found in K=3 smoke train metrics.csv:
  - delay-explosion (ep0/6/17): PK queueing delay uncapped as rho→1 → 1e8 ms
  - AoI-explosion (ep9/10/13):   current_aoi uncapped + all-or-nothing capacity
                                  gate → delivery stalls → AoI to 28-53 s
Both blew up the Lagrangian penalty (1e7) and worker_critic_loss (1e18-1e20).

A: cap PK delay at OVERLOAD_DELAY_SEC for rho<1 too (not just rho>=1).
B: cap AoI at OVERLOAD_AOI_SEC wherever it feeds c_vec / obs.
C: RLC-style cross-TTI service-bit accumulation so low-SINR/low-PRB vehicles
   make partial progress and eventually deliver instead of stalling forever.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from env.oran_env import (
    ORANEnv, macro_mission_config, OVERLOAD_DELAY_SEC, OVERLOAD_AOI_SEC,
)
from env.channel_model import capacity_per_prb_bps
from utils.config import URLLC_PKT_BITS, SHANNON_ETA, SEVERITY_QOS


def _make_env(K=3, seed=0):
    env = ORANEnv(macro_mission_config(K_ambulances=K, seed=seed), seed=seed)
    env.reset(seed=seed)
    return env


# ══════════════════════════════════════════════════════════════════
# A: PK delay capped at rho→1
# ══════════════════════════════════════════════════════════════════

class TestADelayCap:
    def test_delay_capped_at_overload_for_rho_near_one(self):
        """Force a near-saturated URLLC queue and confirm d_e2e <= cap."""
        env = _make_env(K=3)
        # Drive every URLLC queue to rho just below 1 with a tiny service rate.
        for k in range(3):
            q = env.queues[f"urllc_{k}"]
            q.set_arrival_rate(50.0)
            q.update_service_rate(1, 1.0)   # near-zero capacity → rho→huge→clamped near 1
        d = env._compute_e2e_delay_per_amb()
        assert np.all(d <= OVERLOAD_DELAY_SEC + 1e-12), f"delay not capped: {d}"

    def test_delay_finite_full_episode(self):
        """No per-amb delay should exceed the cap over a long rollout."""
        env = _make_env(K=3)
        max_d = 0.0
        for _ in range(2000):
            _, _, t, tr, _ = env.step(np.zeros(3, dtype=np.float32))
            max_d = max(max_d, float(env._last_d_e2e_per_amb.max()))
            if t or tr:
                break
        assert max_d <= OVERLOAD_DELAY_SEC + 1e-12, f"max delay {max_d} > cap"


# ══════════════════════════════════════════════════════════════════
# B: AoI capped
# ══════════════════════════════════════════════════════════════════

class TestBAoICap:
    def test_overload_aoi_exceeds_all_thresholds(self):
        """Cap must exceed the loosest AoI_max so violations register for all sev."""
        max_aoi_max = max(SEVERITY_QOS[s]["AoI_max"] for s in range(1, 6))
        assert OVERLOAD_AOI_SEC > max_aoi_max, "cap must register violation for sev1 too"

    def test_aoi_in_cvec_capped_when_delivery_stalls(self):
        """Stall delivery (0 PRB to all URLLC) → AoI would grow unbounded, but
        the c_vec AoI term (C4) must be capped at OVERLOAD_AOI_SEC."""
        env = _make_env(K=3)
        # Force never-deliver: zero PRB to URLLC queues every step.
        for _ in range(3000):
            # set rrm budget very low so URLLC gets ~0 PRB
            env.set_rrm_budget(0.05)
            obs, _, t, tr, info = env.step(np.zeros(3, dtype=np.float32))
            if t or tr:
                break
        c_vec = np.asarray(info["c_vec"])
        K = 3
        c4 = c_vec[2 * K:3 * K]   # AoI mean term
        # Every C4 entry must be <= OVERLOAD_AOI_SEC (capped), not tens of seconds.
        assert np.all(c4 <= OVERLOAD_AOI_SEC + 1e-9), f"C4 not capped: {c4}"

    def test_obs_aoi_capped(self):
        """obs AoI summary (fixed block) must also be bounded by the cap."""
        env = _make_env(K=3)
        last_obs = None
        for _ in range(2000):
            env.set_rrm_budget(0.05)
            last_obs, _, t, tr, _ = env.step(np.zeros(3, dtype=np.float32))
            if t or tr:
                break
        # AoI mean/max are at OBS_AOI_MEAN_IDX/OBS_AOI_MAX_IDX (18, 19).
        assert last_obs[18] <= OVERLOAD_AOI_SEC + 1e-9
        assert last_obs[19] <= OVERLOAD_AOI_SEC + 1e-9


# ══════════════════════════════════════════════════════════════════
# C: cross-TTI accumulation prevents permanent delivery stall
# ══════════════════════════════════════════════════════════════════

class TestCCapacityAccumulation:
    def test_partial_bits_accumulate_then_deliver(self):
        """A vehicle with sub-packet per-TTI capacity eventually delivers via
        cross-TTI accumulation (was: never delivered, all-or-nothing gate)."""
        env = _make_env(K=1)
        sinr = 5.0
        cap = capacity_per_prb_bps(sinr, eta=SHANNON_ETA)
        # Choose PRB so one TTI delivers < a full packet but several TTIs do.
        one_tti_bits = 1 * cap * env.config.tti_sec
        assert one_tti_bits < URLLC_PKT_BITS, "precondition: 1 TTI insufficient"
        n_tti_needed = int(np.ceil(URLLC_PKT_BITS / one_tti_bits))

        env.last_bler_per_amb = np.array([0.0])
        env.last_sinr_db = np.array([sinr])
        env._last_prb_per_amb = np.array([1], dtype=np.int64)
        # Inject one packet
        env.aoi_trackers[0]["ambulance_status"].arrive(gen_time=0.0)

        delivered_at = None
        for tti in range(n_tti_needed + 2):
            env._update_aoi_trackers(np.array([0], dtype=np.int64))  # no new arrivals
            if int(env._aoi_pkt_delivered[0]) >= 1:
                delivered_at = tti + 1
                break
        assert delivered_at is not None, "packet never delivered despite accumulation"
        # Should deliver around n_tti_needed TTIs (not 1, not never)
        assert delivered_at >= n_tti_needed, f"delivered too early at {delivered_at}"
        assert delivered_at <= n_tti_needed + 1, f"delivered too late at {delivered_at}"

    def test_single_tti_insufficient_still_no_delivery(self):
        """Backward-compat with E3: ONE TTI of insufficient capacity → no delivery
        (partial accumulates but stays below threshold)."""
        env = _make_env(K=1)
        sinr = -10.0
        env.last_bler_per_amb = np.array([0.0])
        env.last_sinr_db = np.array([sinr])
        env._last_prb_per_amb = np.array([1], dtype=np.int64)
        env._update_aoi_trackers(np.array([1], dtype=np.int64))
        assert int(env._aoi_pkt_delivered[0]) == 0
        assert int(env._aoi_pkt_failed_no_capacity[0]) == 1

    def test_partial_bits_reset_when_queue_empty(self):
        """Partial accumulator drops when no packet is pending (no carry-over
        of stale bits into a future unrelated packet)."""
        env = _make_env(K=1)
        env._partial_service_bits[0] = 1234.0
        # Empty queue → step should reset partial
        env.aoi_trackers[0]["ambulance_status"].queue.clear()
        env._last_prb_per_amb = np.array([10], dtype=np.int64)
        env._update_aoi_trackers(np.array([0], dtype=np.int64))
        assert env._partial_service_bits[0] == 0.0


# ══════════════════════════════════════════════════════════════════
# Integration: penalty/critic bounded after all 3 fixes
# ══════════════════════════════════════════════════════════════════

class TestIntegrationBoundedPenalty:
    def test_cvec_bounded_under_starvation(self):
        """Under sustained near-starvation, the full c_vec stays bounded (no
        million-scale entries that previously blew up the critic)."""
        env = _make_env(K=3)
        max_c = 0.0
        for _ in range(3000):
            env.set_rrm_budget(0.05)
            _, _, t, tr, info = env.step(np.zeros(3, dtype=np.float32))
            max_c = max(max_c, float(np.abs(np.asarray(info["c_vec"])).max()))
            if t or tr:
                break
        # All c_vec entries (delay sec, AoI sec, indicators, eMBB gap Mbps) must
        # be small — the old explosion gave 1e4-1e5 (seconds) here.
        assert max_c <= max(OVERLOAD_AOI_SEC, OVERLOAD_DELAY_SEC, 300.0) + 1e-6, (
            f"c_vec still has large entries: max={max_c}"
        )
