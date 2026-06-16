"""Per-ambulance severity_k epic (2026-06-15) — structural tests.

Covers:
    - severity_per_amb sampled independently per ambulance, fixed for the episode
    - severity_ref = max(severity_per_amb) drives shared quantities
    - K-dependent action space (6-dim K=1, 7-dim K>=2 with β priority temperature)
    - Intra-slice Π_feasible PRB split: K=1 numeric preservation
    - Severity-ordering ⟹ PRB-allocation-ordering ⟹ delay-ordering (structural,
      replaces the C6 Lagrangian-constraint formulation — now an algebraic
      property of _prb_split_intra_slice + per-ambulance queues)
"""

from __future__ import annotations

import numpy as np

from env.oran_env import EnvConfig, ORANEnv
from utils.config import BETA_MAX, BETA_MIN


# ----------------------------------------------------------------------------
# severity_per_amb sampling + severity_ref
# ----------------------------------------------------------------------------


def test_severity_per_amb_independent_and_fixed_for_episode():
    """K=3: severity_per_amb is (K,), each in 1..5, fixed across steps."""
    K = 3
    env = ORANEnv(config=EnvConfig(K_ambulances=K), seed=0)
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
    env = ORANEnv(config=EnvConfig(K_ambulances=K), seed=0)
    _, info = env.reset(seed=0, options={"severity_per_amb": [2, 4, 1]})
    np.testing.assert_array_equal(env.severity_per_amb, np.array([2, 4, 1]))
    assert env.severity == 4


# ----------------------------------------------------------------------------
# K-dependent action space (β priority temperature)
# ----------------------------------------------------------------------------


def test_action_space_dim_6_at_k1():
    env = ORANEnv(config=EnvConfig(K_ambulances=1), seed=0)
    assert env.action_space.shape == (6,)


def test_action_space_dim_7_at_k_geq_2():
    env = ORANEnv(config=EnvConfig(K_ambulances=3), seed=0)
    assert env.action_space.shape == (7,)


def test_beta_fixed_at_beta_min_for_k1():
    """At K=1, β has no numeric effect (softmax([x])=[1.0]); kept at BETA_MIN."""
    env = ORANEnv(config=EnvConfig(K_ambulances=1), seed=0)
    env.reset(seed=0)
    env.step(env.action_space.sample())
    assert env._beta == BETA_MIN


def test_beta_driven_by_action_a6_at_k_geq_2():
    K = 3
    env = ORANEnv(config=EnvConfig(K_ambulances=K), seed=0)
    env.reset(seed=0)
    a = env.action_space.sample()
    a[6] = 3.0  # sigmoid(3) ~ 0.953 -> beta close to BETA_MAX
    env.step(a)
    assert BETA_MIN < env._beta <= BETA_MAX
    assert env._beta > (BETA_MIN + BETA_MAX) / 2


# ----------------------------------------------------------------------------
# Intra-slice Π_feasible PRB split — K=1 numeric preservation
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


def test_prb_split_k3_monotonic_in_severity_with_beta_max():
    """K=3, severity_per_amb = (1,3,5), β=BETA_MAX, zero urgency tiebreaker:
    PRB shares are non-decreasing in severity (softmax over β·severity_per_amb)."""
    K = 3
    env = ORANEnv(config=EnvConfig(K_ambulances=K), seed=0)
    env.reset(seed=0, options={"severity_per_amb": [1, 3, 5]})
    env._beta = BETA_MAX
    env._lambda_local = np.zeros(4 * K + 1, dtype=np.float64)

    prb = env._prb_split_intra_slice(P_URLLC := 200)
    assert prb.shape == (K,)
    assert prb.sum() == P_URLLC
    # Higher severity ambulance receives >= PRB than lower severity ambulance.
    assert prb[2] >= prb[1] >= prb[0]
    assert prb[2] > prb[0], "most severe ambulance should get strictly more PRB"


def test_severity_ordering_implies_delay_ordering_under_equal_sinr():
    """K=3, equal SINR across ambulances, severity_per_amb = (1,3,5), β=BETA_MAX:
    the more severe ambulance is allocated more PRB and therefore achieves
    a lower (or equal) end-to-end URLLC delay — the structural replacement
    for the demoted C6 Lagrangian constraint."""
    K = 3
    env = ORANEnv(config=EnvConfig(K_ambulances=K), seed=0)
    env.reset(seed=0, options={"severity_per_amb": [1, 3, 5]})
    env._beta = BETA_MAX
    env._lambda_local = np.zeros(4 * K + 1, dtype=np.float64)

    # Equal SINR across ambulances -> only PRB share differs.
    env.last_sinr_db = np.full(K, 10.0, dtype=np.float64)
    for k in range(K):
        env.queues[f"urllc_{k}"].set_arrival_rate(env.config.urllc_arrival_rate)
    env._update_queue_service_rates()

    d_e2e_per_amb = env._compute_e2e_delay_per_amb()
    assert d_e2e_per_amb.shape == (K,)
    assert np.all(np.isfinite(d_e2e_per_amb))
    # Higher severity (more PRB) -> lower or equal delay.
    assert d_e2e_per_amb[2] <= d_e2e_per_amb[1] <= d_e2e_per_amb[0]
    assert d_e2e_per_amb[2] < d_e2e_per_amb[0], "most severe ambulance should have strictly lower delay"


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
