"""F4: Early-episode termination when ALL K ambulances arrive tests.

Verifies:
- terminated=True only when enable_arrival=True and all ambulances arrived
- truncated still works when enable_arrival=False (legacy)
- terminated and truncated are mutually exclusive
- terminated propagates correctly in step() return
"""

from __future__ import annotations

import numpy as np
import pytest

from env.oran_env import EnvConfig, ORANEnv
from solvers._common import value_bootstrap_is_terminal


def _make_env(K: int = 3, enable_arrival: bool = True, threshold: float = 25.0) -> ORANEnv:
    cfg = EnvConfig(
        K_ambulances=K,
        enable_arrival=enable_arrival,
        arrival_radius_m=threshold,
    )
    return ORANEnv(cfg, seed=0)


class TestTermination:
    def test_terminated_true_when_all_arrived(self):
        env = _make_env(K=3, threshold=50.0)
        env.reset(seed=0)
        # Force both ambulances into threshold
        getattr(env._mobility, '_active', env._mobility)._reached_dest_mask = np.ones(env.config.K_ambulances, dtype=bool)
        env._update_arrival_masks()
        assert env.arrived_mask.all()
        # Next step should return terminated=True
        _, _, terminated, truncated, _ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert terminated
        assert not truncated

    def test_not_terminated_when_partial_arrival(self):
        env = _make_env(K=3, threshold=50.0)
        env.reset(seed=0)
        # Only ambulance 0 arrived
        getattr(env._mobility, '_active', env._mobility)._reached_dest_mask = np.array([True, False, False])
        env._update_arrival_masks()
        _, _, terminated, truncated, _ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert not terminated

    def test_not_terminated_when_enable_arrival_false(self):
        env = _make_env(K=3, enable_arrival=False)
        env.reset(seed=0)
        # Even if ambulances are at origin
        _, _, terminated, _, _ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert not terminated

    def test_terminated_and_truncated_mutually_exclusive(self):
        env = _make_env(K=3, threshold=50.0)
        env.reset(seed=0)
        getattr(env._mobility, '_active', env._mobility)._reached_dest_mask = np.ones(env.config.K_ambulances, dtype=bool)
        env._update_arrival_masks()
        _, _, terminated, truncated, _ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        # Cannot both be True simultaneously
        assert not (terminated and truncated)

    def test_episode_terminates_before_max_tti_when_all_arrive(self):
        env = _make_env(K=1, threshold=50.0)
        env.reset(seed=0)
        # Run a few steps normally, then force arrival
        a = np.zeros(env.action_space.shape, dtype=np.float32)
        for _ in range(5):
            env.step(a)
        getattr(env._mobility, '_active', env._mobility)._reached_dest_mask = np.ones(env.config.K_ambulances, dtype=bool)
        env._update_arrival_masks()
        # Step should terminate immediately, not run to max TTI
        _, _, terminated, truncated, info = env.step(a)
        assert terminated
        assert info["episode_end_reason"] == "all_arrived"
        assert info["tti"] < env._max_tti_for_episode()


class TestLegacyTruncation:
    def test_truncation_still_works_without_arrival(self):
        """enable_arrival=False: episode must end via truncation (legacy)."""
        cfg = EnvConfig(K_ambulances=1, enable_arrival=False)
        env = ORANEnv(cfg, seed=0)
        env.reset(seed=0)
        a = np.zeros(env.action_space.shape, dtype=np.float32)
        truncated = False
        terminated = False
        steps = 0
        while not (truncated or terminated) and steps < 5000:
            _, _, terminated, truncated, _ = env.step(a)
            steps += 1
        assert truncated
        assert not terminated


class TestValueBootstrapTruncationVsTermination:
    """The value bootstrap must zero ONLY on a true terminal, not on a timeout
    truncation (Pardo 2018). Regression guard for the truncation-vs-termination
    bug fixed 2026-06-21 in train.py + train_offpolicy.py."""

    def test_termination_zeroes_bootstrap(self):
        # all ambulances arrived → true MDP terminal → bootstrap value = 0
        assert value_bootstrap_is_terminal(True) is True

    def test_truncation_does_not_zero_bootstrap(self):
        # 400s timeout truncation is NOT terminal → must bootstrap V(s'), not 0
        assert value_bootstrap_is_terminal(False) is False

    def test_timeout_is_truncation_not_terminal_so_bootstraps(self):
        """End-to-end: drive the env to its max-TTI timeout and confirm the episode
        is truncated (NOT terminated), so `done` is True yet the bootstrap rule
        keeps it non-terminal — i.e. the critic must bootstrap, not inject V=0."""
        cfg = EnvConfig(K_ambulances=1, enable_arrival=False)   # never arrives ⇒ must time out
        env = ORANEnv(cfg, seed=0)
        env.reset(seed=0)
        a = np.zeros(env.action_space.shape, dtype=np.float32)
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated) and steps < 5000:
            _, _, terminated, truncated, _ = env.step(a)
            steps += 1
        assert truncated and not terminated
        done = terminated or truncated
        assert done is True                                  # episode ended ...
        assert value_bootstrap_is_terminal(terminated) is False  # ... but NOT as a terminal
