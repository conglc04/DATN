"""B2 — HRL-PPO soft baseline.

Same 2-level conceptual hierarchy as PPO, but with:
    USE_PHASE_FSM    = False   (phase masked out of obs)
    USE_CMDP         = False   (no Lagrangian, soft penalty only)

Reference: docs/06_validation.md:16-28
    The "HRL" part for Week 5 is conceptual — actual Manager/Worker net split
    comes in Week 8 when agents/manager_agent.py + worker_agent.py land.
    For smoke training we use a single flat PPO with soft-penalty reward.

    reward_aug = -mean(D_e2e) - BETA_SOFT * mean(max(0, D_e2e - 1ms))
"""

from __future__ import annotations

import numpy as np

from agents.ppo_agent import PPOAgent
from solvers._common import BaselineFlags, mask_phase


BETA_SOFT_DEFAULT: float = 20.0


class B2HRLPPOSoftBaseline:
    name = "b2_hrl_ppo_soft"
    FLAGS = BaselineFlags(use_phase=False, use_cmdp=False, use_hrl=True, n_constraints=0)

    def __init__(self, state_dim: int, action_dim: int = 6, seed: int = 0,
                 beta_soft: float = BETA_SOFT_DEFAULT, device: str = "cpu"):
        self.flags = self.FLAGS
        self.beta_soft = beta_soft
        self.ppo = PPOAgent(state_dim, action_dim,
                            actor_hidden=(256, 128, 64),
                            critic_hidden=(256, 128, 64),
                            seed=seed, device=device)

    def maybe_mask(self, obs):
        return mask_phase(obs)        # B2 hides phase

    def select_action(self, obs, deterministic: bool = False):
        return self.ppo.select_action(self.maybe_mask(obs), deterministic=deterministic)

    def augment_reward(self, reward: float, d_e2e_samples=None, d_max: float = 1e-3) -> float:
        """Soft penalty: r_aug = r - β · mean(max(0, D_e2e - D_max))."""
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
