"""Tests for the hard-mission env (Option 3 — Week 5 follow-up).

Verifies:
    - urllc_burst window multiplies the arrival rate
    - bystander spike raises eMBB UE count during the burst window
    - configurable SINR clamp + BS TX power
    - hard_mission_config preset builds without error (fixed IMMEDIATE severity)
    - Backward compatibility: default EnvConfig still satisfies Gate P2
"""

from __future__ import annotations

import numpy as np
import pytest


def _zero_action():
    return np.zeros(1, dtype=np.float32)


# ============================================================
# Severity (sampled once per episode at reset, constant within episode)
# ============================================================

_VALID_SEVERITIES = {1, 2, 3, 4, 5}
_SEVERITY_NAMES = {"MINOR", "SEMI_URGENT", "URGENT", "CRITICAL", "IMMEDIATE"}


class TestSeverityFixed:
    def test_severity_is_valid_after_reset(self):
        from env.oran_env import EnvConfig, ORANEnv

        env = ORANEnv(config=EnvConfig(), seed=0)
        _, info = env.reset(seed=0)
        assert info["severity"] in _VALID_SEVERITIES
        assert info["severity_name"] in _SEVERITY_NAMES

    def test_severity_constant_within_episode(self):
        """Severity is sampled once at reset and MUST NOT change mid-episode."""
        from env.oran_env import EnvConfig, ORANEnv

        env = ORANEnv(config=EnvConfig(), seed=0)
        _, reset_info = env.reset(seed=0)
        severity_at_reset = reset_info["severity"]
        a = _zero_action()
        for _ in range(50):
            _, _, _, _, info = env.step(a)
            assert info["severity"] == severity_at_reset   # never changes mid-episode


# ============================================================
# URLLC burst
# ============================================================


class TestUrllcBurst:
    def test_burst_window_active_flag(self):
        from env.oran_env import EnvConfig, ORANEnv

        env = ORANEnv(config=EnvConfig(urllc_burst_at_sec=0.5,
                                        urllc_burst_duration_sec=0.10), seed=0)
        env.reset(seed=0)
        env.sim_time = 0.4
        assert not env._urllc_burst_active()
        env.sim_time = 0.5
        assert env._urllc_burst_active()
        env.sim_time = 0.55
        assert env._urllc_burst_active()
        env.sim_time = 0.60
        assert not env._urllc_burst_active()

    def test_burst_multiplies_arrival_rate(self):
        from env.oran_env import EnvConfig, ORANEnv

        env = ORANEnv(
            config=EnvConfig(
                urllc_arrival_rate=50.0,
                urllc_burst_at_sec=0.0,
                urllc_burst_duration_sec=1.0,        # active for whole episode
                urllc_burst_factor=10.0,
            ),
            seed=0,
        )
        env.reset(seed=0)
        a = _zero_action()
        for _ in range(50):
            env.step(a)
        # During burst, queue arrival rate ≈ 500 (instead of 50)
        assert env.queues["urllc_0"].arrival_rate == pytest.approx(500.0, rel=1e-6)

    def test_no_burst_keeps_base_rate(self):
        from env.oran_env import EnvConfig, ORANEnv

        env = ORANEnv(config=EnvConfig(urllc_arrival_rate=50.0, urllc_burst_at_sec=None), seed=0)
        env.reset(seed=0)
        for _ in range(10):
            env.step(_zero_action())
        assert env.queues["urllc_0"].arrival_rate == pytest.approx(50.0, rel=1e-6)


# ============================================================
# Bystander S2B
# ============================================================


class TestBystanderS2B:
    def test_bystander_disabled_by_default(self):
        from env.oran_env import EnvConfig, ORANEnv

        env = ORANEnv(config=EnvConfig(), seed=0)
        env.reset(seed=0)
        assert env.bystander is None

    def test_bystander_built_when_enabled(self):
        from env.oran_env import EnvConfig, ORANEnv

        env = ORANEnv(
            config=EnvConfig(enable_bystander=True, bystander_trigger_sec=0.4),
            seed=0,
        )
        env.reset(seed=0)
        assert env.bystander is not None
        assert 80 <= env.bystander.peak_ues <= 120

    def test_embb_load_jumps_during_spike(self):
        from env.oran_env import EnvConfig, ORANEnv

        env = ORANEnv(
            config=EnvConfig(
                enable_bystander=True,
                bystander_trigger_sec=0.0,         # spike from t=0
                M_eMBB=30,
                embb_arrival_rate=1000.0,
            ),
            seed=0,
        )
        env.reset(seed=0)
        a = _zero_action()
        # Sample inside the sustain window (which starts at trigger=0)
        for _ in range(200):
            env.step(a)
        # rate ≈ embb_arrival_rate × n_active (>= 80 UEs)
        assert env.queues["eMBB"].arrival_rate >= 1000.0 * 80


# ============================================================
# Channel knobs
# ============================================================


class TestChannelKnobs:
    def test_sinr_clamp_max_db_respected(self):
        from env.oran_env import EnvConfig, ORANEnv

        env = ORANEnv(config=EnvConfig(sinr_clamp_max_db=30.0,
                                        ambulance_start_distance_m=80.0), seed=0)
        env.reset(seed=0)
        for _ in range(20):
            env.step(_zero_action())
        assert float(np.max(env.last_sinr_db)) <= 30.0 + 1e-6

    def test_lower_tx_power_lowers_sinr(self):
        from env.oran_env import EnvConfig, ORANEnv

        e_hi = ORANEnv(config=EnvConfig(bs_tx_power_total_dbm=46.0,
                                         ambulance_start_distance_m=150.0), seed=0)
        e_lo = ORANEnv(config=EnvConfig(bs_tx_power_total_dbm=30.0,
                                         ambulance_start_distance_m=150.0), seed=0)
        e_hi.reset(seed=0)
        e_lo.reset(seed=0)
        for _ in range(20):
            e_hi.step(_zero_action())
            e_lo.step(_zero_action())
        assert float(np.mean(e_hi.last_sinr_db)) > float(np.mean(e_lo.last_sinr_db))

    # test_ambulance_start_distance_respected removed: RWP-only (ambulance_start_distance_m
    # placed vehicles at a fixed radius). The env now always uses SUMO+OSM mobility, so
    # start positions come from the FCD trace — this RWP knob no longer applies.


# ============================================================
# hard_mission_config integration
# ============================================================


class TestHardMissionConfig:
    def test_preset_builds(self):
        from env.oran_env import hard_mission_config, ORANEnv
        env = ORANEnv(config=hard_mission_config(), seed=0)
        _, info = env.reset(seed=0)
        assert env.bystander is not None
        assert info["severity"] in _VALID_SEVERITIES

    def test_full_episode_no_crash(self):
        from env.oran_env import hard_mission_config, ORANEnv

        env = ORANEnv(config=hard_mission_config(), seed=42)
        _, reset_info = env.reset(seed=42)
        severity_at_reset = reset_info["severity"]
        a = _zero_action()
        # Step through entire 1s episode
        steps = 0
        truncated = False
        info = {}
        while not truncated and steps < 2100:
            _, _, _, truncated, info = env.step(a)
            steps += 1
        assert truncated
        # Severity sampled once at reset — must stay constant for whole episode
        assert info["severity"] == severity_at_reset

    def test_burst_creates_violations_for_static_policy(self):
        """Static (zero-action) policy under hard mission should violate sometimes."""
        from env.oran_env import hard_mission_config, ORANEnv

        env = ORANEnv(config=hard_mission_config(), seed=42)
        env.reset(seed=42)
        a = _zero_action()
        truncated = False
        while not truncated:
            _, _, _, truncated, _ = env.step(a)
        # Hard mission is hard enough that a zero-action Static policy hits
        # > 1ms (severity-5 target) on SOME TTI. We don't pin a specific rate to keep
        # the test resilient to sampling variance — just check the regime
        # changed from the easy env (where viol was strictly 0).
        viol = env.episode_violation_rate()
        assert viol > 0.0


# ============================================================
# Backward compat — Gate P2 still passes with default config
# ============================================================


class TestBackwardCompat:
    def test_default_env_still_gate_p2(self):
        from env.oran_env import EnvConfig, ORANEnv

        env = ORANEnv(
            config=EnvConfig(initial_severity=3, rrm_budget_hint=0.6,
                              urllc_arrival_rate=50.0, K_ambulances=1, M_eMBB=30),
            seed=42,
        )
        env.reset(seed=42)
        a = _zero_action()
        for _ in range(200):
            env.step(a)
        assert env.mean_e2e_ms() < 1.0
        assert env.queues["urllc_0"].is_stable
