"""Ablation variant — Flat PPO with full CMDP ("w/o HRL" in Table II).

Compared to PPO full:
    Phase     ✓ (one-hot visible)
    HRL       ✗ (flat single-level, no Manager)
    CMDP      ✓ (full 5-dim Lagrangian)
    Safety QP ✓ (placeholder)

Used to isolate the value of the 2-level hierarchy.
Reference: docs/06_validation.md ablation Table.
"""

from __future__ import annotations

import numpy as np

from agents.ppo_agent import PPOAgent
from solvers._common import BaselineFlags, CMDPLagrangian


class PPOCMDPFlatBaseline:
    name = "ppo_cmdp_flat"
    FLAGS = BaselineFlags(use_phase=True, use_cmdp=True, use_hrl=False, n_constraints=5)

    def __init__(self, state_dim: int, action_dim: int = 6, seed: int = 0,
                 alpha_lambda: float = 0.01, device: str = "cpu"):
        self.flags = self.FLAGS
        self.ppo = PPOAgent(state_dim, action_dim,
                            actor_hidden=(256, 128, 64),
                            critic_hidden=(256, 128, 64),
                            seed=seed, device=device)
        self.lagrangian = CMDPLagrangian(n=5, alpha=alpha_lambda)

    def maybe_mask(self, obs):
        return obs

    def select_action(self, obs, deterministic: bool = False):
        return self.ppo.select_action(obs, deterministic=deterministic)

    def augment_reward(self, reward: float, constraints=None) -> float:
        if constraints is None:
            return reward
        return float(reward - self.lagrangian.penalty(constraints))

    def update_lambdas(self, mean_constraints) -> None:
        self.lagrangian.step(mean_constraints)

    def update(self, buffer) -> dict:
        out = self.ppo.update(buffer)
        for j in range(self.flags.n_constraints):
            out[f"lambda_{j+1}"] = float(self.lagrangian.lambdas[j])
        return out

    def save(self, path: str) -> None:
        self.ppo.save(path)

    def load(self, path: str) -> None:
        self.ppo.load(path)
