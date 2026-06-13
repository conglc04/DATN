"""SAC-Lag sibling solver — off-policy SAC + 5-dim λ via LambdaState (W07).

Phase 3 sibling solver — applied AFTER Phase 2 problem statement complete
(end of W06). SAC-Lag is **flat off-policy SAC + Lagrangian** with the same
5-dim λ machinery as PA-CHRL-PPO and TD3-Lag.

Differences from TD3-Lag:
    - Stochastic max-entropy actor (automatic temperature α) instead of
      deterministic actor + exploration noise
    - Otherwise identical sibling-solver wiring: λ update frequency tied to
      Worker step count via LambdaState (1 dual ascent per Manager step
      boundary, same as on-policy)

Used in Table I alongside TD3-Lag to show PA-CHRL-PPO generalizes across
on-policy and off-policy (deterministic + stochastic) constrained-RL families.

Reference:
    - Haarnoja et al. 2018 "Soft Actor-Critic" (ICML) +
      "SAC Algorithms and Applications" (arXiv:1812.05905)
    - docs/13_methodology_walkthrough.md Phase 2.3 (Lagrangian dual)
"""

from __future__ import annotations

import numpy as np

from agents.lagrangian import LambdaState
from agents.sac_agent import SACAgent
from baselines._common import BaselineFlags, mask_phase


class SACLagBaseline:
    name = "sac_lag"
    FLAGS = BaselineFlags(use_phase=False, use_cmdp=True, use_hrl=False, n_constraints=5)

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 6,
        seed: int = 0,
        alpha_lambda: float | None = None,
        device: str = "cpu",
        action_low: tuple[float, ...] = (-1.0, -1.0, 0.0, 0.0, 0.0, 0.0),
        action_high: tuple[float, ...] = (+1.0, +1.0, 1.0, 1.0, 1.0, 1.0),
    ) -> None:
        self.flags = self.FLAGS
        self.sac = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            action_low=np.asarray(action_low, dtype=np.float32),
            action_high=np.asarray(action_high, dtype=np.float32),
            hidden=(256, 256),
            seed=seed,
            device=device,
            warmup_steps=500,
            buffer_capacity=50_000,
            batch_size=256,
        )
        kwargs = {} if alpha_lambda is None else {"alpha_lambda": alpha_lambda}
        self.lambda_state = LambdaState(**kwargs)
        self.action_low = np.asarray(action_low, dtype=np.float32)
        self.action_high = np.asarray(action_high, dtype=np.float32)

    # ------------------------------------------------------------------
    # Phase masking
    # ------------------------------------------------------------------

    def maybe_mask(self, obs):
        return mask_phase(obs)

    def select_action(self, obs, deterministic: bool = False):
        masked = self.maybe_mask(obs)
        action = self.sac.select_action(masked, deterministic=deterministic)
        # Match PPO API tuple (action, log_prob, value) — SAC log_prob/value
        # are not surfaced through this sibling-solver interface.
        return action, 0.0, 0.0

    # ------------------------------------------------------------------
    # LambdaState lifecycle (sibling API)
    # ------------------------------------------------------------------

    def on_episode_start(self, initial_phase: int) -> None:
        self.lambda_state.reset_episode(initial_phase)

    def on_manager_step_start(self, phi_now: int) -> None:
        self.lambda_state.on_manager_step_start(phi_now)

    def accumulate_constraint(self, c_vec: np.ndarray, d_phi: np.ndarray) -> None:
        self.lambda_state.accumulate(c_vec, d_phi)

    def on_manager_step_end(self) -> dict[str, float]:
        return self.lambda_state.on_manager_step_end()

    def augment_reward(self, reward: float, c_vec: np.ndarray, d_phi: np.ndarray) -> float:
        return self.lambda_state.augmented_reward(reward, c_vec, d_phi)

    # ------------------------------------------------------------------
    # Off-policy transition storage + gradient update
    # ------------------------------------------------------------------

    def store_transition(self, obs, action, reward, next_obs, done) -> None:
        self.sac.store(
            self.maybe_mask(obs),
            action,
            float(reward),
            self.maybe_mask(next_obs),
            bool(done),
        )

    def update(self, buffer=None) -> dict:
        out = self.sac.update()
        lam = self.lambda_state.get_lambda_global()
        for j in range(5):
            out[f"lambda_global_{j + 1}"] = float(lam[j])
        return out

    def save(self, path: str) -> None:
        self.sac.save(path)

    def load(self, path: str) -> None:
        self.sac.load(path)
