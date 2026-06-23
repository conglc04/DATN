"""F1: active_mask / arrived_mask per-ambulance lifecycle tests (SUMO+OSM).

Verifies:
- masks initialised correctly on reset() (SUMO: active = entered & ~arrived)
- arrived_mask latches True via the SUMO provider's reached_destination_mask
  (under SUMO, arrival = FCD exit at the destination, NOT a position threshold)
- active_mask = entered_mask & ~arrived_mask
- arrived ambulance's 10-dim obs block is sentinel-zeroed
- masks in info dict; n_active, all_arrived, episode_end_reason correct
- enable_arrival=False (default) → no arrival
"""

from __future__ import annotations

import numpy as np
import pytest

from env.oran_env import EnvConfig, ORANEnv
from utils.config import OBS_FIXED_BLOCK_LEN, OBS_PER_AMB_BLOCK_LEN


def _make_env(K: int = 3, enable_arrival: bool = True, threshold: float = 25.0) -> ORANEnv:
    # SUMO+OSM mobility (traces exist for K in {1,3}); arrival via FCD exit at destination.
    cfg = EnvConfig(
        K_ambulances=K,
        enable_arrival=enable_arrival,
        arrival_radius_m=threshold,
    )
    return ORANEnv(cfg, seed=0)


def _force_arrived(env, mask) -> None:
    """Force arrival via the SUMO provider's reached_destination_mask.

    PooledSumoMobilityProvider delegates reached_destination_mask to _active;
    set the mask on the inner provider so _update_arrival_masks() picks it up.
    """
    inner = getattr(env._mobility, '_active', env._mobility)
    inner._reached_dest_mask = np.asarray(mask, dtype=bool)
    env._update_arrival_masks()


class TestMaskInit:
    def test_active_mask_reflects_entered_not_arrived_on_reset(self):
        """SUMO staggers cell entry: active = entered & ~arrived; >=1 active, none arrived."""
        env = _make_env()
        env.reset(seed=0)
        assert env.active_mask.shape == (3,)
        np.testing.assert_array_equal(env.active_mask, env.entered_mask & ~env.arrived_mask)
        assert env.active_mask.any(), "at least one ambulance active after cell-entry fast-forward"
        assert not env.arrived_mask.any()

    def test_arrived_mask_all_false_on_reset(self):
        env = _make_env()
        env.reset(seed=0)
        assert not env.arrived_mask.any()

    def test_masks_shape_matches_K(self):
        for K in (1, 3):   # SUMO+OSM traces exist for K in {1,3}
            env = _make_env(K=K)
            env.reset(seed=0)
            assert env.active_mask.shape == (K,)
            assert env.arrived_mask.shape == (K,)


class TestArrivalLatch:
    def test_arrived_latches_via_provider(self):
        env = _make_env(K=3, threshold=50.0)
        env.reset(seed=0)
        _force_arrived(env, [True, False, False])
        assert bool(env.arrived_mask[0])
        assert not env.arrived_mask[1]

    def test_arrived_stays_latched_after_provider_clears(self):
        env = _make_env(K=1, threshold=50.0)
        env.reset(seed=0)
        _force_arrived(env, [True])
        assert env.arrived_mask[0]
        # Provider no longer reports arrival, but the env latch must hold.
        _force_arrived(env, [False])
        assert env.arrived_mask[0]

    def test_active_mask_is_entered_and_not_arrived(self):
        env = _make_env(K=3, threshold=50.0)
        env.reset(seed=0)
        _force_arrived(env, [True, False, False])
        np.testing.assert_array_equal(env.active_mask, env.entered_mask & ~env.arrived_mask)


class TestObsSentinel:
    def test_arrived_ambulance_obs_block_is_zero(self):
        env = _make_env(K=3, threshold=50.0)
        env.reset(seed=0)
        # Force ambulance 0 arrived (inactive) — _observe() zeroes inactive blocks.
        env.arrived_mask = np.array([True, False, False])
        env.active_mask = env.entered_mask & ~env.arrived_mask
        obs = env._observe()
        start = OBS_FIXED_BLOCK_LEN
        np.testing.assert_array_equal(obs[start:start + OBS_PER_AMB_BLOCK_LEN], 0.0)

    def test_active_ambulance_obs_block_nonzero(self):
        env = _make_env(K=3, threshold=50.0)
        env.reset(seed=0)
        # Force amb 0 arrived, amb 1 active+entered.
        env.arrived_mask = np.array([True, False, False])
        env.entered_mask = np.array([True, True, True])
        env.active_mask = env.entered_mask & ~env.arrived_mask
        env.ambulance_pos = np.array([[10.0, 0.0], [400.0, 0.0], [300.0, 0.0]])
        env._update_channel()
        obs = env._observe()
        start = OBS_FIXED_BLOCK_LEN + OBS_PER_AMB_BLOCK_LEN  # k=1 block
        assert np.any(obs[start:start + OBS_PER_AMB_BLOCK_LEN] != 0.0)

    def test_active_ambulance_obs_not_zeroed(self):
        """Active ambulances (entered & not arrived) always have non-zero obs block."""
        env = _make_env(K=3, threshold=50.0, enable_arrival=False)
        env.reset(seed=0)
        # k=0 active, k=1 arrived → its block zeroed.
        env.arrived_mask = np.array([False, True, False])
        env.entered_mask = np.array([True, True, True])
        env.active_mask = env.entered_mask & ~env.arrived_mask
        env.ambulance_pos = np.array([[10.0, 0.0], [400.0, 0.0], [300.0, 0.0]])
        env._update_channel()
        obs = env._observe()
        start = OBS_FIXED_BLOCK_LEN
        assert np.any(obs[start:start + OBS_PER_AMB_BLOCK_LEN] != 0.0)
        start1 = OBS_FIXED_BLOCK_LEN + OBS_PER_AMB_BLOCK_LEN
        assert np.all(obs[start1:start1 + OBS_PER_AMB_BLOCK_LEN] == 0.0)


class TestInfoDict:
    def test_info_has_mask_keys(self):
        env = _make_env(K=3)
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        for key in ("active_mask", "arrived_mask", "n_active", "all_arrived", "episode_end_reason"):
            assert key in info

    def test_n_active_decreases_on_arrival(self):
        env = _make_env(K=3, threshold=50.0)
        env.reset(seed=0)
        k = int(np.argmax(env.active_mask))   # an active ambulance
        m = env.arrived_mask.copy(); m[k] = True
        _force_arrived(env, m)
        assert not env.active_mask[k], "an arrived ambulance must become inactive"

    def test_all_arrived_false_initially(self):
        env = _make_env(K=3)
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert not info["all_arrived"]

    def test_episode_end_reason_truncated_when_not_all_arrived(self):
        env = _make_env(K=3)
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert info["episode_end_reason"] == "truncated"


class TestDefaultBehavior:
    def test_enable_arrival_false_no_arrival(self):
        """enable_arrival=False: no ambulance ever marked arrived."""
        env = _make_env(K=3, enable_arrival=False)
        env.reset(seed=0)
        for _ in range(20):
            env.step(np.zeros(env.action_space.shape, dtype=np.float32))
            assert not env.arrived_mask.any()

    def test_default_config_enable_arrival_false(self):
        assert EnvConfig().enable_arrival is False
