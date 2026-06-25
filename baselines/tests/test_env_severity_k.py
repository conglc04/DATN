"""Per-ambulance severity_k epic (2026-06-15) — structural tests.

Covers:
    - severity_per_amb sampled independently per ambulance, fixed for the episode
    - severity_ref = max(severity_per_amb) drives shared quantities
    - K-dependent action space (1-dim K=1 no-op, K-dim K>=2 pure-RL logits — NO β
      slot, gỡ 2026-06-21)
    - Intra-slice pure-RL softmax PRB split (NO Π_feasible tier-rule): K=1 numeric
      preservation
    - Severity-ordering ⟹ PRB-allocation-ordering ⟹ delay-ordering is now an
      EMPIRICAL/learned tendency (gradient-driven via λ_C1..C5 + r_aug), NOT a
      structural/algebraic property — the severity-tier protection phase that used
      to guarantee this was removed 2026-06-21 (see agents/worker_agent.py
      "pure-RL intra-slice" docstring)
"""

from __future__ import annotations

import numpy as np

from env.oran_env import EnvConfig, ORANEnv
from utils.config import BETA_MAX, BETA_MIN


# ----------------------------------------------------------------------------
# severity_per_amb sampling + severity_ref
# ----------------------------------------------------------------------------


def test_severity_per_amb_independent_and_fixed_for_episode():
    """K=3: sampled severity_per_amb is fixed within an episode."""
    K = 3
    env = ORANEnv(config=EnvConfig(K_ambulances=K, sample_severity=True), seed=0)
    _, info = env.reset(seed=0)
    sev0 = env.severity_per_amb.copy()
    assert sev0.shape == (K,)
    assert np.all((sev0 >= 1) & (sev0 <= 5))
    assert info["severity"] == int(sev0.max())
    assert list(info["severity_per_amb"]) == list(sev0)

    for _ in range(5):
        _, _, _, _, info = env.step(env.action_space.sample())
        np.testing.assert_array_equal(env.severity_per_amb, sev0)
        assert info["severity"] == int(sev0.max())


def test_sampled_severity_changes_only_on_scenario_reset():
    """Random severity is sampled at reset(), not during an active episode."""
    K = 3
    env = ORANEnv(config=EnvConfig(K_ambulances=K, sample_severity=True), seed=0)
    env.reset(seed=7)
    sev0 = env.severity_per_amb.copy()

    env_same = ORANEnv(config=EnvConfig(K_ambulances=K, sample_severity=True), seed=0)
    env_same.reset(seed=7)
    np.testing.assert_array_equal(env_same.severity_per_amb, sev0)

    for _ in range(10):
        env.step(env.action_space.sample())
        np.testing.assert_array_equal(env.severity_per_amb, sev0)

    changed = False
    for new_seed in range(8, 32):
        env.reset(seed=new_seed)
        if not np.array_equal(env.severity_per_amb, sev0):
            changed = True
            break
    assert changed, "sample_severity=True should draw a fresh tuple on new scenario reset"


def test_severity_ref_is_max_of_severity_per_amb():
    """severity_ref (info['severity'], severity one-hot) := max(severity_per_amb)."""
    K = 3
    env = ORANEnv(config=EnvConfig(K_ambulances=K), seed=0)
    _, info = env.reset(seed=0, options={"severity_per_amb": [1, 3, 5]})
    assert env.severity == 5
    assert info["severity"] == 5
    sev_oh = env._observe()[10:15]
    assert sev_oh[4] == 1.0  # severity 5 -> one-hot index 4


def test_severity_per_amb_manual_override_via_options():
    K = 3
    env = ORANEnv(config=EnvConfig(K_ambulances=K, sample_severity=True), seed=0)
    _, info = env.reset(seed=0, options={"severity_per_amb": [2, 4, 1]})
    np.testing.assert_array_equal(env.severity_per_amb, np.array([2, 4, 1]))
    assert env.severity == 4


def test_macro_mission_samples_severity_per_reset():
    from env.oran_env import macro_mission_config

    cfg = macro_mission_config(K_ambulances=3)
    assert cfg.sample_severity is True


# ----------------------------------------------------------------------------
# K-dependent action space (pure-RL, no β priority temperature)
# ----------------------------------------------------------------------------


def test_action_space_dim_1_at_k1():
    env = ORANEnv(config=EnvConfig(K_ambulances=1), seed=0)
    assert env.action_space.shape == (1,)


def test_action_space_dim_k_at_k_geq_2():
    """K>=2: action space is K-dim — pure per-vehicle priority logits ℓ_k.
    No β slot (pure-RL allocation has no urgency-temperature term)."""
    env = ORANEnv(config=EnvConfig(K_ambulances=3), seed=0)
    assert env.action_space.shape == (3,)  # K=3, no +1 for β


def test_beta_always_beta_min_pure_rl():
    """Pure-RL: β stays at BETA_MIN always (reserved, unused in allocation)."""
    env = ORANEnv(config=EnvConfig(K_ambulances=1), seed=0)
    env.reset(seed=0)
    env.step(env.action_space.sample())
    assert env._beta == BETA_MIN

    K = 3
    env3 = ORANEnv(config=EnvConfig(K_ambulances=K), seed=0)
    env3.reset(seed=0)
    env3.step(env3.action_space.sample())
    assert env3._beta == BETA_MIN


def test_action_is_logits_directly_no_beta_slot():
    """action[0:K] = ℓ_k directly (no shift for β)."""
    K = 3
    env = ORANEnv(config=EnvConfig(K_ambulances=K), seed=0)
    env.reset(seed=0)
    a = np.array([2.0, -1.0, 0.5], dtype=np.float32)
    env.step(a)
    np.testing.assert_array_almost_equal(env._prb_weights, [2.0, -1.0, 0.5])


# ----------------------------------------------------------------------------
# Intra-slice pure-RL softmax PRB split — K=1 numeric preservation
# ----------------------------------------------------------------------------


def test_prb_split_k1_returns_full_allocation_regardless_of_beta():
    """At K=1, softmax([x]) = [1.0] always -> PRB_0 = B_U regardless of β/severity."""
    env = ORANEnv(config=EnvConfig(K_ambulances=1), seed=0)
    env.reset(seed=0)
    for beta in (BETA_MIN, 1.0, BETA_MAX):
        env._beta = beta
        prb = env._prb_split_intra_slice(100)
        assert prb.shape == (1,)
        assert prb[0] == 100


# ----------------------------------------------------------------------------
# Severity-ordering ⟹ PRB-allocation-ordering ⟹ delay-ordering (structural)
# ----------------------------------------------------------------------------


def test_prb_split_k3_pure_rl_uniform_without_lambda():
    """Pure-RL allocation: zero λ + zero logits → uniform PRB split.
    No hard-coded severity ordering — RL must learn differentiation."""
    K = 3
    env = ORANEnv(config=EnvConfig(K_ambulances=K), seed=0)
    env.reset(seed=0, options={"severity_per_amb": [1, 3, 5]})
    env._beta = BETA_MIN
    env._lambda_local = np.zeros(4 * K + 1, dtype=np.float64)
    env._prb_weights = np.zeros(K, dtype=np.float64)
    env.last_sinr_db = np.full(K, -15.0, dtype=np.float64)
    env.active_mask = np.ones(K, dtype=bool)

    prb = env._prb_split_intra_slice(20)
    assert prb.shape == (K,)
    assert prb.sum() == 20
    spread = prb.max() - prb.min()
    assert spread <= 2, f"Zero λ/logits should be ~uniform, got {prb}"


def test_prb_split_k3_logits_drive_severity_priority():
    """Pure-RL: Worker logits directly drive severity ordering.
    Policy learns to output higher logits for high-severity ambulances."""
    K = 3
    env = ORANEnv(config=EnvConfig(K_ambulances=K), seed=0)
    env.reset(seed=0, options={"severity_per_amb": [1, 3, 5]})
    # Simulate trained policy: logit proportional to severity
    env._prb_weights = np.array([0.0, 2.0, 5.0], dtype=np.float64)
    env.last_sinr_db = np.full(K, -15.0, dtype=np.float64)
    env.active_mask = np.ones(K, dtype=bool)

    prb = env._prb_split_intra_slice(20)
    assert prb.sum() == 20
    assert prb[2] >= prb[1] >= prb[0], (
        f"logit-driven ordering failed: sev5={prb[2]} sev3={prb[1]} sev1={prb[0]}"
    )


# ----------------------------------------------------------------------------
# set_lambda_local shape validation (B5 epic)
# ----------------------------------------------------------------------------


def test_set_lambda_local_validates_shape():
    K = 3
    env = ORANEnv(config=EnvConfig(K_ambulances=K), seed=0)
    env.reset(seed=0)
    env.set_lambda_local(np.zeros(4 * K + 1))  # (13,) OK
    try:
        env.set_lambda_local(np.zeros(5))  # wrong shape for K=3
        assert False, "expected ValueError"
    except ValueError:
        pass
