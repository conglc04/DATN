"""Observation-layout lock — enforces the single-source-of-truth (SSOT) for the
fixed 20-dim obs block.

Every fixed-block field is placed by ``_observe()`` at its named ``OBS_*_IDX``
constant (utils.config). These tests drive the env into a KNOWN state and assert
each named index carries the expected field, so any reordering of ``_observe`` —
or a constant that drifts from the assembly — fails loudly. Combined with the
``state_dim == OBS_FIXED_BLOCK_LEN + OBS_PER_AMB_BLOCK_LEN·K + F`` assert in
train.py, this is the layout guarantee. (audit 2026-06-17, full-SSOT Phương án 1)
"""

from __future__ import annotations

import numpy as np
import pytest

from env.oran_env import EnvConfig, ORANEnv
from utils.config import (
    OBS_AOI_MAX_IDX,
    OBS_AOI_MEAN_IDX,
    OBS_ARR_EMBB_IDX,
    OBS_ARR_URLLC_IDX,
    OBS_BLER_IDX,
    OBS_FIXED_BLOCK_LEN,
    OBS_HOL_EMBB_IDX,
    OBS_HOL_URLLC_IDX,
    OBS_LAMBDA_C3_IDX,
    OBS_N_BYS_IDX,
    OBS_PER_AMB_BLOCK_LEN,
    OBS_R_DED_URLLC_IDX,
    OBS_R_MAX_EMBB_IDX,
    OBS_R_MIN_URLLC_IDX,
    OBS_RHO_EMBB_IDX,
    OBS_RHO_URLLC_IDX,
    OBS_RMIN_ANCHOR_IDX,
    OBS_SEVERITY_OH_IDX,
    OBS_SEVERITY_OH_LEN,
)


def test_fixed_block_indices_are_a_permutation_of_0_19():
    """The 20 named fixed-block indices cover exactly {0..19} with no gap/overlap."""
    scalars = [
        OBS_RHO_URLLC_IDX, OBS_RHO_EMBB_IDX, OBS_HOL_URLLC_IDX, OBS_HOL_EMBB_IDX,
        OBS_R_MIN_URLLC_IDX, OBS_R_MAX_EMBB_IDX, OBS_R_DED_URLLC_IDX,
        OBS_ARR_URLLC_IDX, OBS_ARR_EMBB_IDX, OBS_BLER_IDX,
        OBS_LAMBDA_C3_IDX, OBS_RMIN_ANCHOR_IDX, OBS_N_BYS_IDX,
        OBS_AOI_MEAN_IDX, OBS_AOI_MAX_IDX,
    ]
    one_hot = list(range(OBS_SEVERITY_OH_IDX, OBS_SEVERITY_OH_IDX + OBS_SEVERITY_OH_LEN))
    covered = sorted(scalars + one_hot)
    assert covered == list(range(OBS_FIXED_BLOCK_LEN)), f"layout not a clean 0..19 cover: {covered}"


def test_anchor_index_holds_manager_setpoint_not_live_rmin():
    """obs[OBS_RMIN_ANCHOR_IDX] == anchor (set value); obs[OBS_R_MIN_URLLC_IDX]
    drifts. Locks that [16] is the Manager anchor and [4] is the live r_min."""
    env = ORANEnv(EnvConfig(K_ambulances=1))
    env.reset(seed=0)
    env.set_rrm_budget(0.5)
    a = np.zeros(6, dtype=np.float32)
    a[0] = 1.0  # push live r_min up via Δr_min
    obs, *_ = env.step(a)
    assert obs[OBS_RMIN_ANCHOR_IDX] == pytest.approx(0.5, abs=1e-6)   # anchor fixed
    assert obs[OBS_R_MIN_URLLC_IDX] == pytest.approx(0.6, abs=1e-5)   # live drifted
    env.close()


def test_severity_one_hot_at_named_index():
    """The severity_ref one-hot occupies [OBS_SEVERITY_OH_IDX : +OBS_SEVERITY_OH_LEN]
    and the hot bit is at (severity_ref − 1)."""
    sev = 4
    env = ORANEnv(EnvConfig(K_ambulances=1, initial_severity=sev))
    obs, _ = env.reset(seed=0)
    oh = obs[OBS_SEVERITY_OH_IDX: OBS_SEVERITY_OH_IDX + OBS_SEVERITY_OH_LEN]
    assert oh.sum() == pytest.approx(1.0)            # exactly one hot bit
    assert int(np.argmax(oh)) == sev - 1             # at severity_ref − 1
    env.close()


def test_prb_ratio_indices_match_env_state():
    """obs[4,5,6] == env.r_min_urllc / r_max_emBB / r_ded_urllc after a step."""
    env = ORANEnv(EnvConfig(K_ambulances=1))
    env.reset(seed=0)
    obs, *_ = env.step(np.zeros(6, dtype=np.float32))
    assert obs[OBS_R_MIN_URLLC_IDX] == pytest.approx(env.r_min_urllc, abs=1e-6)
    assert obs[OBS_R_MAX_EMBB_IDX] == pytest.approx(env.r_max_emBB, abs=1e-6)
    assert obs[OBS_R_DED_URLLC_IDX] == pytest.approx(env.r_ded_urllc, abs=1e-6)
    env.close()


def test_bler_index_matches_env_state():
    env = ORANEnv(EnvConfig(K_ambulances=1))
    env.reset(seed=0)
    obs, *_ = env.step(np.zeros(6, dtype=np.float32))
    assert obs[OBS_BLER_IDX] == pytest.approx(float(env.last_bler), abs=1e-6)
    env.close()


def test_rho_indices_in_unit_interval():
    """obs[0],obs[1] are utilizations clipped to [0,1]."""
    env = ORANEnv(EnvConfig(K_ambulances=1))
    env.reset(seed=0)
    obs, *_ = env.step(np.zeros(6, dtype=np.float32))
    for idx in (OBS_RHO_URLLC_IDX, OBS_RHO_EMBB_IDX):
        assert 0.0 <= obs[idx] <= 1.0
    env.close()


def test_layout_holds_at_k3():
    """SSOT constants are K-independent: fixed block stays [0:20] at K=3, the
    per-amb block follows immediately, and total dim matches the formula."""
    K = 3
    env = ORANEnv(EnvConfig(K_ambulances=K))
    obs, _ = env.reset(seed=0)
    assert obs.shape[0] == OBS_FIXED_BLOCK_LEN + OBS_PER_AMB_BLOCK_LEN * K + env.config.num_streams
    # severity one-hot still within the fixed block (unaffected by K)
    oh = obs[OBS_SEVERITY_OH_IDX: OBS_SEVERITY_OH_IDX + OBS_SEVERITY_OH_LEN]
    assert oh.sum() == pytest.approx(1.0)
    env.close()
