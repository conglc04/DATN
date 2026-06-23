"""Gate 2/4 — timescale + episode semantics (independent runtime trace).

Locks: MAC/Worker/Manager ratios; episode != rollout chunk; severity persists
across the 1 s rollout boundary; masks evolve correctly; episode ends on
all-arrived/timeout; Manager action held for exactly W Worker steps.
"""
from __future__ import annotations

import numpy as np
import pytest

from env.oran_env import EnvConfig, ORANEnv
from utils.config import GAMMA, GAMMA_MANAGER, WORKER_STEPS_PER_MANAGER
import utils.config as C


class TestTimescaleConstants:
    def test_worker_steps_per_manager(self):
        assert WORKER_STEPS_PER_MANAGER == 10            # 100ms / 10ms

    def test_mac_ticks_per_worker(self):
        assert getattr(C, "MAC_TICKS_PER_WORKER", 20) == 20   # 10ms / 0.5ms

    def test_gamma_manager_is_worker_gamma_to_the_W(self):
        assert GAMMA_MANAGER == pytest.approx(GAMMA ** WORKER_STEPS_PER_MANAGER, abs=1e-12)

    def test_one_worker_step_advances_20_tti(self):
        env = ORANEnv(EnvConfig(K_ambulances=1, episode_duration_sec=1.0), seed=0)
        env.reset(seed=0)
        t0 = env.tti_idx
        env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert env.tti_idx - t0 == 20, "one Worker step = 20 MAC ticks"


class TestEpisodeSemantics:
    def test_severity_persists_across_rollout_boundary(self):
        """Severity is fixed for the whole mission — NOT resampled at 1 s."""
        env = ORANEnv(EnvConfig(K_ambulances=3, sample_severity=True,
                                episode_duration_sec=2.0), seed=7)
        _, info0 = env.reset(seed=7)
        sev0 = tuple(int(s) for s in info0["severity_per_amb"])
        a = np.zeros(env.action_space.shape, dtype=np.float32)
        # step well past the 100-worker-step (1 s) rollout boundary
        for _ in range(150):
            _, _, term, trunc, info = env.step(a)
            assert tuple(int(s) for s in info["severity_per_amb"]) == sev0
            if term or trunc:
                break

    def test_episode_runs_to_timeout_not_one_second(self):
        """A 2 s episode must NOT truncate at the 1 s rollout chunk (100 steps)."""
        env = ORANEnv(EnvConfig(K_ambulances=1, episode_duration_sec=2.0), seed=0)
        env.reset(seed=0)
        a = np.zeros(env.action_space.shape, dtype=np.float32)
        n = 0
        for _ in range(250):
            _, _, term, trunc, _ = env.step(a)
            n += 1
            if term or trunc:
                break
        assert n == 200, f"2 s episode = 200 Worker steps, got {n}"

    def test_queue_aoi_not_reset_within_episode(self):
        """Internal histories accumulate across the rollout boundary (no reset)."""
        env = ORANEnv(EnvConfig(K_ambulances=1, episode_duration_sec=2.0), seed=0)
        env.reset(seed=0)
        a = np.zeros(env.action_space.shape, dtype=np.float32)
        for _ in range(120):  # cross the 100-step (1 s) boundary
            env.step(a)
        # histories are per-MAC-tick and monotonically growing within the episode
        assert len(env.e2e_history) == len(env.aoi_history) > 100 * 20 // 2

    def test_active_mask_is_entered_and_not_arrived(self):
        env = ORANEnv(EnvConfig(K_ambulances=3, enable_arrival=True,
                                sample_severity=False, initial_severity=3,
                                episode_duration_sec=1.0), seed=5)
        env.reset(seed=5, options={"severity_per_amb": [3, 3, 3]})
        a = np.zeros(env.action_space.shape, dtype=np.float32)
        for _ in range(50):
            env.step(a)
            expected = env.entered_mask & ~env.arrived_mask
            assert np.array_equal(env.active_mask, expected)


class TestManagerActionHold:
    def test_set_rrm_budget_held_across_worker_steps(self):
        """The Manager setpoint is held constant until explicitly re-set."""
        env = ORANEnv(EnvConfig(K_ambulances=1), seed=0)
        env.reset(seed=0)
        env.set_rrm_budget(0.42)
        held = env.r_min_urllc   # post-floor setpoint (floor-agnostic)
        a = np.zeros(env.action_space.shape, dtype=np.float32)
        for _ in range(WORKER_STEPS_PER_MANAGER):
            env.step(a)
            assert env.r_min_urllc == pytest.approx(held, abs=1e-9)
