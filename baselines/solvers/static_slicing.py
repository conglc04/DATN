"""Static 50/50 slicing baseline (lower bound for Exp1 Table I).

NO learning. Outputs a fixed action every step:
    Δr_min = 0,  Δr_max = 0,  r_ded_ratio = 0   (preserves initial r_min/r_max
    coming from EnvConfig.rrm_budget_hint = 0.5)
    w_intra = uniform (1/3, 1/3, 1/3)

Used to demonstrate that learning is necessary (claim #1 of paper).
"""

from __future__ import annotations

import numpy as np


class StaticSlicingBaseline:
    name = "static_slicing"

    def __init__(self, state_dim: int, action_dim: int = 6, seed: int = 0, **_kwargs):
        # Accept (and ignore) device / extra kwargs for parity with PPO-based solvers.
        self.state_dim = state_dim
        self.action_dim = action_dim
        self._seed = seed
        # Action: hold the ratios constant (zero deltas), uniform intra-weights.
        self._fixed_action = np.array([0.0, 0.0, 0.0, 1 / 3, 1 / 3, 1 / 3], dtype=np.float32)

    def select_action(self, obs, deterministic: bool = True):
        return self._fixed_action.copy(), 0.0, 0.0

    def update_lambdas(self, *_args, **_kwargs) -> None:
        pass

    def augment_reward(self, reward: float, _constraints) -> float:
        return reward

    def maybe_mask(self, obs):
        return obs

    def update(self, *_args, **_kwargs) -> dict:
        return {}

    def save(self, path: str) -> None:
        # Nothing learnable; touch file for parity.
        from pathlib import Path
        Path(path).write_text("static_slicing baseline: no learnable params\n", encoding="utf-8")

    def load(self, path: str) -> None:
        pass
