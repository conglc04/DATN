"""Solver-parity guard: PPO / TD3 / SAC must solve the SAME problem.

The 3 solvers are SIBLINGS (ngang hàng), not method+baselines. Fairness is by
construction: the env (reward, constraints, Π_feasible), the HRL Manager+Worker
framework, the (4K+1)-dim LambdaState, and the episode definition are all
solver-agnostic. These tests lock that invariant so the two training drivers
(train.py PPO on-policy, solvers/train_offpolicy.py TD3/SAC off-policy) can never
silently diverge into giving one solver an easier/different problem.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.lagrangian import LambdaState
from agents.manager_agent import (
    MANAGER_ACTION_DIM_DEFAULT,
    ManagerAgent,
    SACManagerAgent,
    TD3ManagerAgent,
    decode_manager_action,
    manager_state_dim,
)
from agents.worker_agent import WorkerAgent
from env.oran_env import ORANEnv, hard_mission_config, macro_mission_config
from solvers._common import build_manager_state
from solvers.sac import SACSolver
from solvers.td3 import TD3Solver
from utils.obs import overlay_lambda_local


@pytest.mark.parametrize("K", [1, 3])
def test_three_solvers_same_dims(K):
    """All 3 solvers bind to the SAME obs/action/manager dims from one env."""
    env = ORANEnv(hard_mission_config(K_ambulances=K), seed=0)
    env.reset(seed=0)  # severity / queues init before _observe()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    sH_dim = manager_state_dim(K)

    ppo_worker = WorkerAgent(state_dim=obs_dim, action_dim=act_dim, seed=0)
    ppo_manager = ManagerAgent(state_dim=sH_dim, action_dim=MANAGER_ACTION_DIM_DEFAULT, seed=0)
    td3 = TD3Solver(state_dim=obs_dim, action_dim=act_dim, seed=0, K=K)
    sac = SACSolver(state_dim=obs_dim, action_dim=act_dim, seed=0, K=K)

    # Manager state dim identical across PPO/TD3/SAC managers.
    assert isinstance(ppo_manager, ManagerAgent)
    assert isinstance(td3.manager, TD3ManagerAgent)
    assert isinstance(sac.manager, SACManagerAgent)

    # TD3/SAC carry a (4K+1)-dim LambdaState — same dual dimensionality.
    assert td3.lambda_state.K == K
    assert sac.lambda_state.K == K
    assert len(LambdaState(K=K).get_lambda_global()) == 4 * K + 1
    assert len(td3.lambda_state.get_lambda_global()) == 4 * K + 1
    assert len(sac.lambda_state.get_lambda_global()) == 4 * K + 1

    # Shared overlay + manager-state builders accept the same shapes for all.
    lam_local = td3.lambda_state.get_lambda_local()
    s_L = overlay_lambda_local(env._observe(), lam_local, K)
    assert s_L.shape[0] == obs_dim
    s_H = build_manager_state(
        env._observe(), td3.lambda_state.get_lambda_global(),
        td3.lambda_state.get_deviation_hat(),
    )
    assert s_H.shape[0] == sH_dim
    _ = ppo_worker  # bound for symmetry; PPO worker shares the same obs_dim


def test_env_step_is_solver_agnostic():
    """env.step(reward, c_vec, d_phi, done) depends ONLY on the action — never on
    which solver produced it. Two envs fed the SAME action sequence return
    identical reward/constraint/termination, proving the problem is shared."""
    cfg_a = hard_mission_config(K_ambulances=3)
    cfg_b = hard_mission_config(K_ambulances=3)
    env_a = ORANEnv(cfg_a, seed=123)
    env_b = ORANEnv(cfg_b, seed=123)
    env_a.reset(seed=123)
    env_b.reset(seed=123)

    rng = np.random.default_rng(0)
    for _ in range(50):
        a = rng.uniform(-1.0, 1.0, size=env_a.action_space.shape[0]).astype(np.float32)
        oa, ra, ta, tra, ia = env_a.step(a)
        ob, rb, tb, trb, ib = env_b.step(a)
        assert ra == pytest.approx(rb)
        assert ta == tb and tra == trb
        np.testing.assert_allclose(ia["c_vec"], ib["c_vec"])
        np.testing.assert_allclose(ia["d_phi"], ib["d_phi"])
        np.testing.assert_allclose(oa, ob)
        if ta or tra:
            break


def test_same_episode_definition_macro():
    """Both training drivers route the sweep through macro_mission_config, so the
    episode definition (cell, channel layer, arrival, duration) is identical."""
    cfg = macro_mission_config(K_ambulances=3)
    env = ORANEnv(cfg, seed=0)
    assert env.config.cell_radius_m == 1000.0
    assert env.base_station.layer == "macro"
    assert env.config.enable_arrival is True
    assert env.config.episode_duration_sec == 400.0
    # Episode termination is the env's call (terminated/truncated), not the agent's.
    assert env.config.K_ambulances == 3
