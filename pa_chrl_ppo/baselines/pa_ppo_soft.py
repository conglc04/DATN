"""Ablation variant — Phase-aware flat PPO, NO CMDP ("w/o CMDP" in Table II).

Compared to PA-CHRL-PPO full:
    Phase     ✓ (one-hot visible in obs)
    HRL       ✗ (flat single-level)
    CMDP      ✗ (no Lagrangian, soft penalty only)
    Safety QP ✗

Used in Exp6 ablation to isolate the value of CMDP hard constraints.
Reference: docs/06_validation.md ablation Table.
"""

from __future__ import annotations

import numpy as np

from agents.ppo_agent import PPOAgent
from baselines._common import BaselineFlags


BETA_SOFT_DEFAULT: float = 20.0


class PAPPOSoftBaseline:
    name = "pa_ppo_soft"
    FLAGS = BaselineFlags(use_phase=True, use_cmdp=False, use_hrl=False, n_constraints=0)

    def __init__(self, state_dim: int, action_dim: int = 6, seed: int = 0,
                 beta_soft: float = BETA_SOFT_DEFAULT, device: str = "cpu"):
        self.flags = self.FLAGS
        self.beta_soft = beta_soft
        self.ppo = PPOAgent(state_dim, action_dim,
                            actor_hidden=(256, 128, 64),
                            critic_hidden=(256, 128, 64),
                            seed=seed, device=device)

    def maybe_mask(self, obs):
        return obs

    def select_action(self, obs, deterministic: bool = False):
        return self.ppo.select_action(obs, deterministic=deterministic)

    def augment_reward(self, reward: float, d_e2e_samples=None, d_max: float = 1e-3) -> float:
        if d_e2e_samples is None or len(d_e2e_samples) == 0:
            return reward
        arr = np.asarray(d_e2e_samples, dtype=float)
        excess = np.maximum(0.0, arr - d_max).mean()
        return float(reward - self.beta_soft * excess)

    def update_lambdas(self, *_args, **_kwargs) -> None:
        pass

    def update(self, buffer) -> dict:
        return self.ppo.update(buffer)

    def save(self, path: str) -> None:
        self.ppo.save(path)

    def load(self, path: str) -> None:
        self.ppo.load(path)
