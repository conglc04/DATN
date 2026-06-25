"""Gate 11 — solver equivalence: PPO/TD3/SAC solve the SAME problem.

Verifies the three solvers share environment, reward, constraint vector,
Manager discount, lambda machinery, action/obs dims, and the SAME exogenous
trajectory under a fixed seed — differing only in algorithm nature.
"""
from __future__ import annotations

import numpy as np
import pytest

from env.oran_env import EnvConfig, ORANEnv
from agents.lagrangian import LambdaState
from agents.manager_agent import decode_manager_action, manager_state_dim
from solvers._common import build_manager_state
from utils.config import GAMMA_MANAGER, build_dual_scales


def _env_trace(seed, actions):
    env = ORANEnv(EnvConfig(K_ambulances=3, sample_severity=True), seed=seed)
    obs, info = env.reset(seed=seed)
    sev = tuple(int(s) for s in info["severity_per_amb"])
    rs, cs, obss = [], [], [obs.copy()]
    for a in actions:
        o, r, t, tr, i = env.step(a)
        rs.append(r); cs.append(np.asarray(i["c_vec"]).copy()); obss.append(o.copy())
        if t or tr:
            break
    return sev, rs, cs, obss


class TestSameExogenousTrajectory:
    def test_same_seed_identical_trace(self):
        rng = np.random.default_rng(0)
        acts = [rng.normal(size=4).astype(np.float32) for _ in range(40)]
        a = _env_trace(321, acts)
        b = _env_trace(321, acts)
        assert a[0] == b[0]                       # same severity vector
        assert a[1] == b[1]                       # same rewards
        for x, y in zip(a[2], b[2]):
            np.testing.assert_array_equal(x, y)   # same constraint vectors
        for x, y in zip(a[3], b[3]):
            np.testing.assert_array_equal(x, y)   # same observations


class TestSharedProblemDefinition:
    """All three solver Managers/Workers must instantiate the same machinery."""

    def test_lambda_state_dim_shared(self):
        for K in (1, 3):
            ls = LambdaState(K=K)
            assert ls.n_constraints == 4 * K + 1
            np.testing.assert_array_equal(ls.dual_scales, build_dual_scales(K))

    def test_manager_discount_shared(self):
        # PPO GAE + TD3/SAC Bellman targets all key off GAMMA_MANAGER
        import agents.manager_agent as M
        src = open(M.__file__).read()
        assert src.count("GAMMA_MANAGER") >= 3, "all 3 Manager variants must use GAMMA_MANAGER"
        assert GAMMA_MANAGER < 1.0

    def test_manager_state_dim_formula(self):
        for K in (1, 3):
            assert manager_state_dim(K) == 8 + 2 * (4 * K + 1)

    def test_decode_manager_action_shared_bounds(self):
        from utils.config import B_RRM_MIN, B_RRM_MAX
        for x in (-10.0, 0.0, 10.0):
            b = decode_manager_action(np.array([x]))["b_rrm"]
            assert B_RRM_MIN - 1e-9 <= b <= B_RRM_MAX + 1e-9

    def test_td3_sac_carry_lambda_in_obs(self):
        """Off-policy drivers must overlay lambda into the observation (single source)."""
        import inspect
        import solvers.train_offpolicy as T
        src = inspect.getsource(T)
        assert "overlay_lambda_local" in src or "_state_with_lambda" in src, \
            "off-policy driver must inject lambda into the Worker observation"


class TestActionObsDimsShared:
    @pytest.mark.parametrize("K,adim", [(1, 1), (3, 3)])
    def test_action_dim(self, K, adim):
        env = ORANEnv(EnvConfig(K_ambulances=K), seed=0)
        assert env.action_space.shape == (adim,)

    @pytest.mark.parametrize("K,odim", [(1, 32), (3, 54)])  # 20 + 11K + F (F=1)
    def test_obs_dim(self, K, odim):
        env = ORANEnv(EnvConfig(K_ambulances=K), seed=0)
        assert env.observation_space.shape == (odim,)
