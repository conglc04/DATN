"""F2: sumo_route_pool EnvConfig field + env integration tests.

Verifies:
- sumo_route_pool and sumo_fcd_path are mutually exclusive
- empty sumo_route_pool raises ValueError
- EnvConfig with sumo_route_pool wires PooledSumoMobilityProvider in env
- reset() with different seeds can use the pool
"""

from __future__ import annotations

import numpy as np
import pytest

from env.oran_env import EnvConfig, ORANEnv
from env.sumo_mobility import density_fcd_path


class TestEnvConfigValidation:
    def test_mutual_exclusion_raises(self):
        fcd = density_fcd_path(1, "medium")
        with pytest.raises(ValueError, match="Cannot set both"):
            EnvConfig(
                sumo_fcd_path=fcd,
                sumo_route_pool=[fcd],
            )

    def test_empty_route_pool_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            EnvConfig(sumo_route_pool=[])

    def test_invalid_density_raises(self):
        with pytest.raises(ValueError, match="traffic_density"):
            EnvConfig(traffic_density="ultra")

    def test_valid_densities_accepted(self):
        for d in ["light", "medium", "heavy"]:
            cfg = EnvConfig(traffic_density=d)
            assert cfg.traffic_density == d

    def test_default_config_auto_discovers_route_pool(self):
        """Default EnvConfig auto-discovers the density route pool (no legacy fallback)."""
        cfg = EnvConfig()
        assert cfg.sumo_route_pool is not None
        assert len(cfg.sumo_route_pool) > 0


class TestEnvRoutePoolIntegration:
    def test_env_resets_with_route_pool(self):
        pool = [density_fcd_path(1, "medium")]
        cfg = EnvConfig(K_ambulances=1, sumo_route_pool=pool)
        env = ORANEnv(cfg, seed=0)
        obs, info = env.reset(seed=0)
        assert obs.shape[0] > 0

    def test_env_steps_after_route_pool_reset(self):
        pool = [density_fcd_path(1, "medium")]
        cfg = EnvConfig(K_ambulances=1, sumo_route_pool=pool)
        env = ORANEnv(cfg, seed=0)
        env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32)
        )
        assert obs.shape[0] > 0

    def test_different_seeds_still_run(self):
        fcd = density_fcd_path(1, "medium")
        pool = [fcd, fcd]
        cfg = EnvConfig(K_ambulances=1, sumo_route_pool=pool)
        env = ORANEnv(cfg, seed=0)
        for seed in [0, 1, 5]:
            env.reset(seed=seed)
            env.step(np.zeros(env.action_space.shape, dtype=np.float32))
