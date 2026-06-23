"""W15-B2 Increment 1 — interference-margin channel term.

Verifies the additive, opt-in inter-cell interference margin:
  - None (default) => exact noise-limited SNR preserved (W12/micro unchanged)
  - a value        => SINR lowered by the noise-rise, gradient across cell kept

The margin lets a MACRO 1 km cell have spatially-varying SINR instead of
saturating at the clamp (Phase-0 finding 2026-06-18). Env stays single-cell;
interference is a constant noise-rise floor, not explicit multi-cell.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from env.channel_model import (
    db_to_linear,
    noise_plus_interference_dbm,
    pl_uma,
    thermal_noise_dbm,
)
from env.oran_env import EnvConfig, ORANEnv
from utils.config import B_PRB, F_CARRIER


def test_margin_none_returns_thermal_noise_unchanged():
    n = thermal_noise_dbm(B_PRB)
    assert noise_plus_interference_dbm(n, None) == n


def test_margin_combines_in_linear_domain():
    n = thermal_noise_dbm(B_PRB)              # ~ -111.4 dBm
    i = -72.0
    expected = 10.0 * math.log10(db_to_linear(n) + db_to_linear(i))
    assert noise_plus_interference_dbm(n, i) == pytest.approx(expected)


def test_margin_dominates_when_far_above_noise():
    # I 40 dB above noise => effective floor ~= I (noise negligible)
    n = thermal_noise_dbm(B_PRB)
    i = n + 40.0
    eff = noise_plus_interference_dbm(n, i)
    assert eff == pytest.approx(i, abs=0.05)   # within 0.05 dB of I
    assert eff > n                              # floor raised


def test_env_default_config_has_interference_margin():
    """Default EnvConfig uses macro UMa model with -86 dBm/PRB interference margin (W15-B2)."""
    assert EnvConfig().interference_margin_dbm_per_prb == -86.0


def test_env_sinr_unchanged_when_margin_none():
    """With margin None, last_sinr_db equals rx - thermal_noise (legacy)."""
    import math
    cfg = EnvConfig(K_ambulances=1, interference_margin_dbm_per_prb=None)
    env = ORANEnv(cfg, seed=0)
    env.reset(seed=0)
    env.ambulance_pos = np.array([[200.0, 0.0]])
    env.channel.shadowing = False              # deterministic for exact compare
    env._update_channel()
    # _update_channel computes tx_per_prb = total - 10*log10(N_prb)
    tx_per_prb = cfg.bs_tx_power_total_dbm - 10.0 * math.log10(cfg.bs_n_prb)
    rx = env.channel.receive_power_dbm(
        (200.0, 0.0), env.base_station,
        tx_power_dbm=tx_per_prb,
    )
    expected = np.clip(
        rx - thermal_noise_dbm(B_PRB),
        cfg.sinr_clamp_min_db, cfg.sinr_clamp_max_db,
    )
    assert env.last_sinr_db[0] == pytest.approx(expected)


def test_env_sinr_lower_with_margin():
    """Adding an interference margin strictly lowers SINR at the same position."""
    pos = np.array([[200.0, 0.0]])
    base = ORANEnv(EnvConfig(K_ambulances=1, interference_margin_dbm_per_prb=None), seed=0)
    base.reset(seed=0); base.ambulance_pos = pos.copy(); base.channel.shadowing = False
    base._update_channel()

    intf = ORANEnv(EnvConfig(K_ambulances=1, interference_margin_dbm_per_prb=-72.0), seed=0)
    intf.reset(seed=0); intf.ambulance_pos = pos.copy(); intf.channel.shadowing = False
    intf._update_channel()

    assert intf.last_sinr_db[0] < base.last_sinr_db[0]


def test_macro_interference_produces_spatial_gradient():
    """With margin + macro UMa + wide clamp, SINR falls with distance (not flat)."""
    cfg = EnvConfig(
        K_ambulances=1,
        interference_margin_dbm_per_prb=-72.0,
        bs_tx_power_total_dbm=30.0,             # macro calibration TX (Phase-0); avoids clamp at 50 m
        sinr_clamp_max_db=40.0,            # wide so it does not flatten
        sinr_clamp_min_db=-15.0,
    )
    env = ORANEnv(cfg, seed=0)
    env.reset(seed=0)
    env.base_station.layer = "macro"          # UMa pathloss
    env.channel.shadowing = False

    def sinr_at(d: float) -> float:
        env.ambulance_pos = np.array([[float(d), 0.0]])
        env._update_channel()
        return float(env.last_sinr_db[0])

    s50, s1000 = sinr_at(50.0), sinr_at(1000.0)
    # Gradient must be clearly non-flat (clamp may limit the tail at 1000m,
    # so we only assert > 10 dB, not the exact PL gap, to stay clamp-agnostic).
    assert s50 > s1000, "SINR must decrease with distance"
    assert s50 - s1000 > 10.0                  # clearly non-flat
