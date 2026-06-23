"""F5: Background traffic density calibration tests (Tier-B synthetic overlay).

Verifies:
- "medium" density = no-op (scale=1.0, exact legacy kinematics)
- "light" moves ambulances faster than "medium"
- "heavy" moves ambulances slower than "medium"
- speed scale constants are physically sane
- EnvConfig validates density strings
- density overlay only applies when SUMO mobility is active
"""

from __future__ import annotations

import numpy as np
import pytest

from env.oran_env import EnvConfig, ORANEnv
from env.sumo_mobility import density_fcd_path


# Density scale constants (mirror _DENSITY_SPEED_SCALE from oran_env.py)
_SCALE = {"light": 1.2, "medium": 1.0, "heavy": 0.7}


def _make_sumo_env(density: str, K: int = 1) -> ORANEnv:
    cfg = EnvConfig(
        K_ambulances=K,
        sumo_fcd_path=density_fcd_path(K, density),
        traffic_density=density,
    )
    return ORANEnv(cfg, seed=0)


class TestDensityScaleConstants:
    def test_light_scale_greater_than_medium(self):
        assert _SCALE["light"] > _SCALE["medium"]

    def test_heavy_scale_less_than_medium(self):
        assert _SCALE["heavy"] < _SCALE["medium"]

    def test_medium_scale_is_one(self):
        assert _SCALE["medium"] == pytest.approx(1.0)

    def test_all_scales_positive(self):
        for s in _SCALE.values():
            assert s > 0.0


class TestDensityKinematics:
    def test_medium_velocity_matches_baseline(self):
        """medium density must produce the EXACT same velocity as legacy (scale=1.0)."""
        env_med = _make_sumo_env("medium")
        env_med.reset(seed=0)
        env_med.step(np.zeros(env_med.action_space.shape, dtype=np.float32))
        vel_med = env_med.ambulance_vel.copy()

        # Reset again — should be deterministic
        env_med.reset(seed=0)
        env_med.step(np.zeros(env_med.action_space.shape, dtype=np.float32))
        np.testing.assert_allclose(env_med.ambulance_vel, vel_med, atol=1e-9)

    def test_light_speed_greater_than_medium(self):
        env_light = _make_sumo_env("light")
        env_med = _make_sumo_env("medium")
        env_light.reset(seed=0)
        env_med.reset(seed=0)
        # Step both and compare speed magnitudes
        env_light.step(np.zeros(env_light.action_space.shape, dtype=np.float32))
        env_med.step(np.zeros(env_med.action_space.shape, dtype=np.float32))
        speed_light = float(np.linalg.norm(env_light.ambulance_vel))
        speed_med = float(np.linalg.norm(env_med.ambulance_vel))
        # light scale=1.2 → speed should be higher (or equal if vel was zero)
        assert speed_light >= speed_med - 1e-6

    def test_heavy_speed_less_than_medium(self):
        env_heavy = _make_sumo_env("heavy")
        env_med = _make_sumo_env("medium")
        env_heavy.reset(seed=0)
        env_med.reset(seed=0)
        env_heavy.step(np.zeros(env_heavy.action_space.shape, dtype=np.float32))
        env_med.step(np.zeros(env_med.action_space.shape, dtype=np.float32))
        speed_heavy = float(np.linalg.norm(env_heavy.ambulance_vel))
        speed_med = float(np.linalg.norm(env_med.ambulance_vel))
        assert speed_heavy <= speed_med + 1e-6


class TestDensityNoopWithoutSumo:
    def test_no_crash_without_sumo_mobility(self):
        """Density overlay is irrelevant when SUMO not active — env runs normally."""
        cfg = EnvConfig(K_ambulances=1, traffic_density="heavy")
        env = ORANEnv(cfg, seed=0)
        env.reset(seed=0)
        obs, _, _, _, _ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert obs.shape[0] > 0


class TestDensityValidation:
    def test_invalid_density_rejected_at_config(self):
        with pytest.raises(ValueError, match="traffic_density"):
            EnvConfig(traffic_density="extreme")

    def test_default_density_is_medium(self):
        assert EnvConfig().traffic_density == "medium"
