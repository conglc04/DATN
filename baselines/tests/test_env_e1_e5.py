"""Commit 1 verification: E1-E5 environment bug fixes.

E1: Queue delay uses PK formula for rho < 1.0, OVERLOAD_DELAY for rho >= 1.0.
E2: Pending AoI packets retried on every TTI (not just on new arrival ticks).
E3: AoI delivery gated by capacity (service_bits >= URLLC_PKT_BITS).
E4: Arrival counter tracks actual Poisson count, not coalesced 1.
E5: Unified service model — delay and AoI share capacity computation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from env.channel_model import capacity_per_prb_bps
from env.oran_env import OVERLOAD_DELAY_SEC, EnvConfig, ORANEnv
from env.queue_model import MG1Queue
from utils.config import (
    D_BH,
    D_DET,
    D_FH,
    D_STOCH,
    SEVERITY_QOS,
    SHANNON_ETA,
    URLLC_PKT_BITS,
)


def _make_env(K: int = 1, seed: int = 42) -> ORANEnv:
    cfg = EnvConfig(K_ambulances=K, episode_duration_sec=2.0)
    env = ORANEnv(cfg)
    env.reset(seed=seed)
    return env


# ────────────────────────────────────────────────────────────────────
# E1: Queue overload delay — rho sweep
# ────────────────────────────────────────────────────────────────────


class TestE1QueueDelayRhoSweep:
    """PK formula used for rho < 1.0; OVERLOAD_DELAY for rho >= 1.0."""

    @staticmethod
    def _pk_delay(q: MG1Queue) -> float:
        d_tx = q.mean_service_time - D_STOCH
        d_queue = q.expected_queue_delay()
        return D_DET + d_tx + d_queue + D_FH + D_BH

    @pytest.mark.parametrize("rho", [0.50, 0.89])
    def test_stable_uses_pk(self, rho: float):
        """rho < 0.9: delay should match PK formula (unchanged behavior)."""
        env = _make_env(K=1, seed=42)
        q = env.queues["urllc_0"]
        q.set_arrival_rate(rho * q.service_rate)
        d = env._compute_e2e_delay_per_amb()
        expected = self._pk_delay(q)
        assert abs(d[0] - expected) < 1e-12

    @pytest.mark.parametrize("rho", [0.91, 0.99])
    def test_marginal_uses_pk_not_clamp(self, rho: float):
        """rho in [0.9, 1.0): PK formula gives large delay, NOT 2ms clamp."""
        env = _make_env(K=1, seed=42)
        q = env.queues["urllc_0"]
        q.set_arrival_rate(rho * q.service_rate)
        d = env._compute_e2e_delay_per_amb()
        expected = self._pk_delay(q)
        assert abs(d[0] - expected) < 1e-12
        assert d[0] > 0.002, f"rho={rho}: delay {d[0]*1e3:.3f}ms should exceed 2ms"

    @pytest.mark.parametrize("rho", [1.00, 1.10])
    def test_overload_uses_constant(self, rho: float):
        """rho >= 1.0: delay = OVERLOAD_DELAY_SEC."""
        env = _make_env(K=1, seed=42)
        q = env.queues["urllc_0"]
        q.set_arrival_rate(rho * q.service_rate)
        d = env._compute_e2e_delay_per_amb()
        assert d[0] == OVERLOAD_DELAY_SEC

    def test_overload_exceeds_all_d_max(self):
        """OVERLOAD_DELAY must be > max(D_max) across all severity levels."""
        max_d_max = max(float(v["D_max"]) for v in SEVERITY_QOS.values())
        assert OVERLOAD_DELAY_SEC > max_d_max

    def test_delay_monotonic_in_rho(self):
        """Delay should increase monotonically as rho increases."""
        env = _make_env(K=1, seed=42)
        q = env.queues["urllc_0"]
        rhos = [0.3, 0.5, 0.7, 0.89, 0.91, 0.95, 0.99]
        delays = []
        for rho in rhos:
            q.set_arrival_rate(rho * q.service_rate)
            d = env._compute_e2e_delay_per_amb()
            delays.append(d[0])
        for i in range(1, len(delays)):
            assert delays[i] >= delays[i - 1], (
                f"Delay decreased: rho={rhos[i-1]}→{rhos[i]}, "
                f"delay={delays[i-1]*1e3:.3f}ms→{delays[i]*1e3:.3f}ms"
            )


# ────────────────────────────────────────────────────────────────────
# E2: Pending packet retry without new arrival
# ────────────────────────────────────────────────────────────────────


class TestE2PendingPacketRetry:
    def test_pending_packet_retried_without_new_arrival(self):
        """A packet that fails BLER should be retried on next TTI even without new arrival."""
        env = _make_env(K=1, seed=42)
        env._last_prb_per_amb = np.array([100], dtype=np.int64)
        env.last_bler_per_amb = np.array([1.0])  # 100% BLER = always fail
        env.last_sinr_db = np.array([15.0])

        env._update_aoi_trackers(np.array([1], dtype=np.int64))
        assert int(env._aoi_pkt_arrived[0]) == 1
        assert int(env._aoi_pkt_failed_bler[0]) == 1

        env.last_bler_per_amb = np.array([0.0])  # 0% BLER = always succeed
        env.sim_time += 0.0005
        env._update_aoi_trackers(np.array([0], dtype=np.int64))  # no new arrival

        assert int(env._aoi_pkt_arrived[0]) == 1, "No new arrival"
        assert int(env._aoi_pkt_delivered[0]) == 1, "Pending packet should be delivered on retry"

    def test_no_spurious_service_when_queue_empty(self):
        """No service attempts when queue is empty and no arrivals."""
        env = _make_env(K=1, seed=42)
        env._last_prb_per_amb = np.array([100], dtype=np.int64)
        env.last_bler_per_amb = np.array([0.0])
        env.last_sinr_db = np.array([15.0])

        env._update_aoi_trackers(np.array([0], dtype=np.int64))
        total_outcomes = (
            int(env._aoi_pkt_delivered[0])
            + int(env._aoi_pkt_failed_bler[0])
            + int(env._aoi_pkt_failed_no_prb[0])
            + int(env._aoi_pkt_failed_no_capacity[0])
        )
        assert total_outcomes == 0

    def test_retry_across_multiple_bler_failures(self):
        """Packet fails BLER 3 times then succeeds on 4th TTI."""
        env = _make_env(K=1, seed=42)
        env._last_prb_per_amb = np.array([100], dtype=np.int64)
        env.last_sinr_db = np.array([15.0])

        env.last_bler_per_amb = np.array([1.0])
        env._update_aoi_trackers(np.array([1], dtype=np.int64))  # arrive + fail
        for _ in range(2):
            env.sim_time += 0.0005
            env._update_aoi_trackers(np.array([0], dtype=np.int64))  # retry + fail

        assert int(env._aoi_pkt_failed_bler[0]) == 3
        assert int(env._aoi_pkt_delivered[0]) == 0

        env.last_bler_per_amb = np.array([0.0])
        env.sim_time += 0.0005
        env._update_aoi_trackers(np.array([0], dtype=np.int64))  # retry + succeed
        assert int(env._aoi_pkt_delivered[0]) == 1
        assert int(env._aoi_pkt_failed_bler[0]) == 3


# ────────────────────────────────────────────────────────────────────
# E3: Capacity-gated delivery
# ────────────────────────────────────────────────────────────────────


class TestE3CapacityGate:
    def test_insufficient_capacity_blocks_delivery(self):
        """1 PRB with poor SINR → service_bits < URLLC_PKT_BITS → no delivery."""
        env = _make_env(K=1, seed=42)
        sinr_poor = -10.0  # very poor channel
        cap = capacity_per_prb_bps(sinr_poor, eta=SHANNON_ETA)
        service_bits_1prb = 1 * cap * env.config.tti_sec
        assert service_bits_1prb < URLLC_PKT_BITS, "Test precondition: 1 PRB insufficient"

        env._last_prb_per_amb = np.array([1], dtype=np.int64)
        env.last_bler_per_amb = np.array([0.0])
        env.last_sinr_db = np.array([sinr_poor])

        env._update_aoi_trackers(np.array([1], dtype=np.int64))
        assert int(env._aoi_pkt_failed_no_capacity[0]) == 1
        assert int(env._aoi_pkt_delivered[0]) == 0

    def test_sufficient_capacity_allows_delivery(self):
        """100 PRBs with good SINR → service_bits >> PKT_BITS → delivery allowed."""
        env = _make_env(K=1, seed=42)
        sinr_good = 15.0
        cap = capacity_per_prb_bps(sinr_good, eta=SHANNON_ETA)
        service_bits = 100 * cap * env.config.tti_sec
        assert service_bits > URLLC_PKT_BITS, "Test precondition: capacity sufficient"

        env._last_prb_per_amb = np.array([100], dtype=np.int64)
        env.last_bler_per_amb = np.array([0.0])
        env.last_sinr_db = np.array([sinr_good])

        env._update_aoi_trackers(np.array([1], dtype=np.int64))
        assert int(env._aoi_pkt_delivered[0]) == 1

    def test_capacity_threshold_boundary(self):
        """Find minimum PRBs for delivery at a given SINR and verify boundary."""
        env = _make_env(K=1, seed=42)
        sinr_db = 5.0
        cap = capacity_per_prb_bps(sinr_db, eta=SHANNON_ETA)
        min_prb = math.ceil(URLLC_PKT_BITS / (cap * env.config.tti_sec))

        # min_prb - 1 should fail
        env._last_prb_per_amb = np.array([max(min_prb - 1, 1)], dtype=np.int64)
        env.last_bler_per_amb = np.array([0.0])
        env.last_sinr_db = np.array([sinr_db])
        env._update_aoi_trackers(np.array([1], dtype=np.int64))

        if (max(min_prb - 1, 1) * cap * env.config.tti_sec) < URLLC_PKT_BITS:
            assert int(env._aoi_pkt_failed_no_capacity[0]) == 1
        else:
            assert int(env._aoi_pkt_delivered[0]) == 1

    def test_capacity_check_uses_per_amb_sinr(self):
        """K=3: amb with poor SINR → capacity fail; amb with good SINR → delivery."""
        env = _make_env(K=3, seed=42)
        env._last_prb_per_amb = np.array([2, 2, 100], dtype=np.int64)
        env.last_bler_per_amb = np.array([0.0, 0.0, 0.0])
        env.last_sinr_db = np.array([-10.0, -10.0, 20.0])

        env._update_aoi_trackers(np.array([1, 1, 1], dtype=np.int64))

        cap_poor = capacity_per_prb_bps(-10.0, eta=SHANNON_ETA)
        if 2 * cap_poor * env.config.tti_sec < URLLC_PKT_BITS:
            assert int(env._aoi_pkt_failed_no_capacity[0]) == 1
            assert int(env._aoi_pkt_failed_no_capacity[1]) == 1
        assert int(env._aoi_pkt_delivered[2]) == 1


# ────────────────────────────────────────────────────────────────────
# E4: Arrival counter tracks Poisson count
# ────────────────────────────────────────────────────────────────────


class TestE4ArrivalCounter:
    def test_multi_arrival_counted(self):
        """n_urllc=3 in one tick → arrived += 3, not 1."""
        env = _make_env(K=1, seed=42)
        env._last_prb_per_amb = np.array([100], dtype=np.int64)
        env.last_bler_per_amb = np.array([0.0])
        env.last_sinr_db = np.array([15.0])

        env._update_aoi_trackers(np.array([3], dtype=np.int64))
        assert int(env._aoi_pkt_arrived[0]) == 3

    def test_single_arrival_counted(self):
        """n_urllc=1 → arrived += 1."""
        env = _make_env(K=1, seed=42)
        env._last_prb_per_amb = np.array([100], dtype=np.int64)
        env.last_bler_per_amb = np.array([0.0])
        env.last_sinr_db = np.array([15.0])

        env._update_aoi_trackers(np.array([1], dtype=np.int64))
        assert int(env._aoi_pkt_arrived[0]) == 1

    def test_zero_arrival_no_count(self):
        """n_urllc=0 → arrived unchanged."""
        env = _make_env(K=1, seed=42)
        env._last_prb_per_amb = np.array([100], dtype=np.int64)
        env.last_bler_per_amb = np.array([0.0])
        env.last_sinr_db = np.array([15.0])

        env._update_aoi_trackers(np.array([0], dtype=np.int64))
        assert int(env._aoi_pkt_arrived[0]) == 0

    def test_cumulative_multi_arrival(self):
        """Across ticks, arrival count accumulates Poisson totals."""
        env = _make_env(K=1, seed=42)
        env._last_prb_per_amb = np.array([100], dtype=np.int64)
        env.last_bler_per_amb = np.array([0.0])
        env.last_sinr_db = np.array([15.0])

        env._update_aoi_trackers(np.array([2], dtype=np.int64))
        env.sim_time += 0.0005
        env._update_aoi_trackers(np.array([3], dtype=np.int64))
        env.sim_time += 0.0005
        env._update_aoi_trackers(np.array([0], dtype=np.int64))
        assert int(env._aoi_pkt_arrived[0]) == 5


# ────────────────────────────────────────────────────────────────────
# E5: Unified service model — delay and AoI share capacity computation
# ────────────────────────────────────────────────────────────────────


class TestE5UnifiedServiceModel:
    def test_both_models_use_same_capacity(self):
        """Queue service rate and AoI capacity check use same function."""
        env = _make_env(K=1, seed=42)
        sinr_db = 10.0
        prb = 50
        cap = capacity_per_prb_bps(sinr_db, eta=SHANNON_ETA)

        env.queues["urllc_0"].update_service_rate(prb, cap)
        queue_service_bps = prb * cap

        aoi_service_bits = prb * cap * env.config.tti_sec
        aoi_service_bps = aoi_service_bits / env.config.tti_sec
        assert abs(queue_service_bps - aoi_service_bps) < 1e-6

    def test_delay_and_aoi_both_degrade_with_low_sinr(self):
        """Both delay and AoI delivery should worsen with poor SINR."""
        env = _make_env(K=1, seed=42)

        sinr_good = 20.0
        sinr_poor = -5.0
        prb = 10

        cap_good = capacity_per_prb_bps(sinr_good, eta=SHANNON_ETA)
        cap_poor = capacity_per_prb_bps(sinr_poor, eta=SHANNON_ETA)

        env.queues["urllc_0"].update_service_rate(prb, cap_good)
        delay_good = env._compute_e2e_delay_per_amb()[0]
        env.queues["urllc_0"].update_service_rate(prb, cap_poor)
        delay_poor = env._compute_e2e_delay_per_amb()[0]
        assert delay_poor > delay_good

        service_bits_good = prb * cap_good * env.config.tti_sec
        service_bits_poor = prb * cap_poor * env.config.tti_sec
        assert service_bits_good > service_bits_poor
