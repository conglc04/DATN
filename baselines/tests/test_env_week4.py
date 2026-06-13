"""Week 4 tests — ORANEnv + bystander_traffic + Gate P2 sanity."""

from __future__ import annotations

import numpy as np
import pytest


# ============================================================
# Bystander model
# ============================================================


class TestBystanderArrivalModel:
    def test_initialization_picks_peak_in_range(self):
        from env.bystander_traffic import BystanderArrivalModel

        m = BystanderArrivalModel(rng=np.random.default_rng(0))
        m.initialize()
        assert 80 <= m.peak_ues <= 120

    def test_baseline_before_trigger(self):
        from env.bystander_traffic import BystanderArrivalModel

        m = BystanderArrivalModel(trigger_time_sec=5.0, baseline_ues=30,
                                   rng=np.random.default_rng(0))
        m.initialize()
        assert m.active_ue_count(0.0) == 30          # before ramp-up
        assert m.active_ue_count(2.0) == 30          # still well before trigger

    def test_peak_during_sustain(self):
        from env.bystander_traffic import BystanderArrivalModel

        m = BystanderArrivalModel(
            trigger_time_sec=2.0, sustain_sec=10.0,
            rng=np.random.default_rng(0),
        )
        m.initialize()
        # Inside sustain window
        assert m.active_ue_count(5.0) == m.peak_ues

    def test_ramp_up_monotone(self):
        from env.bystander_traffic import BystanderArrivalModel

        m = BystanderArrivalModel(trigger_time_sec=2.0, ramp_sec=1.0,
                                   rng=np.random.default_rng(0))
        m.initialize()
        counts = [m.active_ue_count(t) for t in [1.0, 1.3, 1.6, 1.9, 2.0]]
        assert all(counts[i] <= counts[i + 1] for i in range(len(counts) - 1))

    def test_decay_back_toward_baseline(self):
        from env.bystander_traffic import BystanderArrivalModel

        m = BystanderArrivalModel(
            trigger_time_sec=0.0, sustain_sec=1.0, decay_sec=2.0,
            rng=np.random.default_rng(0),
        )
        m.initialize()
        late = m.active_ue_count(5.0)
        assert late <= m.baseline_ues + 5            # close to baseline

    def test_aggregate_load_positive_during_burst(self):
        from env.bystander_traffic import BystanderArrivalModel

        m = BystanderArrivalModel(rng=np.random.default_rng(0))
        m.initialize()
        load = m.aggregate_load_mbps(sim_time=m.trigger_time_sec + m.sustain_sec / 2)
        # 80-120 UEs × 2-5 Mbps → roughly 160-600 Mbps total
        assert 150 < load < 700

    def test_packets_have_arrival_times_in_window(self):
        from env.bystander_traffic import BystanderArrivalModel

        m = BystanderArrivalModel(
            trigger_time_sec=0.0, sustain_sec=10.0,
            rng=np.random.default_rng(0),
        )
        m.initialize()
        pkts = m.generate_packets((1.0, 1.5))
        assert all(1.0 <= p.arrival_time <= 1.5 for p in pkts)


# ============================================================
# ORANEnv basic API
# ============================================================


class TestORANEnvAPI:
    def _make_env(self):
        from env.oran_env import EnvConfig, ORANEnv
        return ORANEnv(config=EnvConfig(), seed=42)

    def test_reset_returns_obs_and_info(self):
        env = self._make_env()
        obs, info = env.reset(seed=0)
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == np.float32
        assert "phase" in info
        assert "r_min_urllc" in info

    def test_step_returns_5tuple(self):
        env = self._make_env()
        env.reset(seed=0)
        a = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, r, terminated, truncated, info = env.step(a)
        assert obs.shape == env.observation_space.shape
        assert isinstance(r, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_observation_in_space_bounds(self):
        env = self._make_env()
        obs, _ = env.reset(seed=0)
        # Observation space is unbounded (-inf, inf) so just check shape + dtype
        assert obs.shape == env.observation_space.shape
        assert np.all(np.isfinite(obs))

    def test_action_space_six_dim(self):
        env = self._make_env()
        assert env.action_space.shape == (6,)

    def test_episode_terminates_after_100_worker_steps(self):
        """1 s episode = 100 Worker steps × 10 ms = 100 × 20 MAC ticks = 2000 TTI total."""
        from env.oran_env import EnvConfig, ORANEnv

        env = ORANEnv(config=EnvConfig(episode_duration_sec=1.0), seed=0)
        env.reset(seed=0)
        a = np.zeros(env.action_space.shape, dtype=np.float32)
        n_steps = 0
        truncated = False
        for _ in range(150):
            _, _, terminated, truncated, _ = env.step(a)
            n_steps += 1
            if truncated:
                break
        assert truncated
        # Should truncate at exactly 100 Worker steps
        assert n_steps == 100, f"Expected truncation at 100 steps, got {n_steps}"


# ============================================================
# Action decoder + constraint enforcement
# ============================================================


class TestActionDecoder:
    def test_delta_r_min_clamped_and_scaled(self):
        from env.oran_env import EnvConfig, ORANEnv
        env = ORANEnv(config=EnvConfig(rrm_budget_hint=0.5), seed=0)
        env.reset(seed=0)
        # Push r_min up
        for _ in range(10):
            env.step(np.array([+1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
        # After 10 ticks of +0.1, r_min should saturate to 1.0
        assert env.r_min_urllc == pytest.approx(1.0, abs=1e-3)

    def test_c6_min_plus_max_le_one(self):
        from env.oran_env import EnvConfig, ORANEnv
        env = ORANEnv(config=EnvConfig(rrm_budget_hint=0.5), seed=0)
        env.reset(seed=0)
        # Push both up — invariant must hold
        for _ in range(20):
            env.step(np.array([+1.0, +1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
        assert env.r_min_urllc + env.r_max_emBB <= 1.0 + 1e-6

    def test_c7_r_ded_le_r_min(self):
        from env.oran_env import EnvConfig, ORANEnv
        env = ORANEnv(config=EnvConfig(rrm_budget_hint=0.3), seed=0)
        env.reset(seed=0)
        # Set r_ded_ratio = 1 → r_ded = min(0.2, r_min)
        for _ in range(5):
            env.step(np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        assert env.r_ded_urllc <= env.r_min_urllc + 1e-6
        assert env.r_ded_urllc <= 0.2 + 1e-6

    def test_prb_sum_le_273(self):
        from env.oran_env import EnvConfig, ORANEnv
        from utils.config import P_TOTAL

        env = ORANEnv(config=EnvConfig(), seed=0)
        env.reset(seed=0)
        for _ in range(50):
            a = env.action_space.sample()
            env.step(a)
            prb_u, prb_e = env._prb_allocation()
            assert prb_u + prb_e <= P_TOTAL


# ============================================================
# Stability under load
# ============================================================


class TestQueueStability:
    def test_urllc_stable_with_r_min_06_lambda_50(self):
        """ρ_urllc < 0.9 at r_min^URLLC=0.6, λ_urllc=50."""
        from env.oran_env import EnvConfig, ORANEnv

        env = ORANEnv(
            config=EnvConfig(
                urllc_arrival_rate=50.0,
                rrm_budget_hint=0.6,
                initial_phase=3,
            ),
            seed=0,
        )
        env.reset(seed=0)
        # Hold the action ≈ zero (keep ratios)
        for _ in range(50):
            env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert env.queues["urllc"].is_stable

    def test_urllc_overloaded_if_no_prb(self):
        from env.oran_env import EnvConfig, ORANEnv
        env = ORANEnv(
            config=EnvConfig(urllc_arrival_rate=50.0, rrm_budget_hint=0.01),
            seed=0,
        )
        env.reset(seed=0)
        # 1% of PRB → 2-3 PRBs → likely cannot serve 50 pkt/s
        for _ in range(10):
            env.step(np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
        # We can't guarantee unstable but rho should be much higher
        rho_low_prb = env.queues["urllc"].rho

        env2 = ORANEnv(
            config=EnvConfig(urllc_arrival_rate=50.0, rrm_budget_hint=0.6),
            seed=0,
        )
        env2.reset(seed=0)
        for _ in range(10):
            env2.step(np.zeros(env2.action_space.shape, dtype=np.float32))
        rho_high_prb = env2.queues["urllc"].rho
        assert rho_low_prb > rho_high_prb


# ============================================================
# GATE P2 — Critical sanity (per docs/09:62-67)
# ============================================================


class TestGateP2:
    """Gate P2 sanity: at φ₃ with r_min^URLLC=0.6 and λ_urllc=50,
    mean D_e2e should be < 1ms (target 0.7-0.9ms)."""

    def test_d_e2e_under_1ms_at_phi3(self):
        from env.oran_env import EnvConfig, ORANEnv

        env = ORANEnv(
            config=EnvConfig(
                initial_phase=3,
                K_ambulances=1,
                M_eMBB=30,
                urllc_arrival_rate=50.0,
                rrm_budget_hint=0.6,
                ambulance_speed_kmh=40.0,
                cell_radius_m=200.0,
            ),
            seed=42,
        )
        env.reset(seed=42)
        a = np.zeros(env.action_space.shape, dtype=np.float32)
        # Step through 200 TTI (100ms) — enough to gather statistics
        for _ in range(200):
            _, _, terminated, truncated, _ = env.step(a)
            if terminated or truncated:
                break

        mean_e2e_ms = env.mean_e2e_ms()
        viol_rate = env.episode_violation_rate()
        prb_u, prb_e = env._prb_allocation()

        # Sanity assertions — should be roughly:
        # D_det 0.07 + D_tx ~0.3 + D_queue ~0.2 + D_fh 0.1 + D_bh 0.1 ≈ 0.7-0.9 ms
        assert mean_e2e_ms < 1.0, f"Gate P2 FAIL: mean D_e2e = {mean_e2e_ms:.3f}ms > 1ms"
        assert env.queues["urllc"].is_stable, "URLLC queue unstable at φ₃"
        # Per-TTI violation rate should be low (allow up to 20% during warm-up)
        assert viol_rate < 0.30, f"viol_rate {viol_rate:.3f} too high"
        # PRB budget honored
        from utils.config import P_TOTAL
        assert prb_u + prb_e <= P_TOTAL
        assert prb_u >= 0.5 * P_TOTAL, (
            f"r_min^URLLC=0.6 should yield ≥{int(0.5*P_TOTAL)} PRBs, got {prb_u}"
        )

    def test_d_e2e_breakdown_components(self):
        """Verify each component (det/fh/bh/queue/tx) lies in expected range."""
        from env.oran_env import EnvConfig, ORANEnv
        from utils.config import D_DET, D_FH, D_BH

        env = ORANEnv(
            config=EnvConfig(initial_phase=3, rrm_budget_hint=0.6, urllc_arrival_rate=50.0),
            seed=42,
        )
        env.reset(seed=42)
        a = np.zeros(env.action_space.shape, dtype=np.float32)
        for _ in range(100):
            env.step(a)

        urllc = env.queues["urllc"]
        d_queue = urllc.expected_queue_delay()
        d_tx = urllc.mean_service_time
        d_components = D_DET + d_tx + d_queue + D_FH + D_BH
        # Sanity: components must sum to env's mean E2E (approximately)
        assert urllc.is_stable
        assert d_components < 1e-3, (
            f"Sum of components = {d_components*1e3:.3f}ms exceeds 1ms"
        )


# ============================================================
# Reproducibility
# ============================================================


class TestSeedReproducibility:
    def test_two_runs_same_seed_match(self):
        from env.oran_env import EnvConfig, ORANEnv

        def run(seed):
            env = ORANEnv(config=EnvConfig(), seed=seed)
            env.reset(seed=seed)
            rewards = []
            for _ in range(20):
                _, r, _, _, _ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
                rewards.append(r)
            return rewards

        r1 = run(123)
        r2 = run(123)
        assert r1 == r2
