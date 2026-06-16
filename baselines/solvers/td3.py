"""TD3 sibling solver — off-policy TD3 + 5-dim λ via LambdaState (W07).

Phase 3 sibling solver — applied AFTER Phase 2 problem statement complete
(end of W06). TD3 is **flat off-policy TD3 + Lagrangian** with the same
5-dim λ machinery as PPO and SAC.

Differences from on-policy PPO:
    - Off-policy backbone (TD3) with replay buffer
    - λ update frequency tied to Worker step count via LambdaState
      (still 1 dual ascent per Manager step boundary, same as on-policy)

Used in Table I alongside SAC to show PPO generalizes across
on-policy + off-policy (deterministic + stochastic) constrained-RL families.

Reference:
    - Fujimoto et al. 2018 "TD3" (ICML)
    - docs/13_methodology_walkthrough.md Phase 2.3 (Lagrangian dual)
"""

from __future__ import annotations

import os

import numpy as np

from agents.lagrangian import LambdaState
from agents.manager_agent import TD3ManagerAgent, manager_state_dim
from agents.td3_agent import TD3Agent
from solvers._common import BaselineFlags, mask_severity


class TD3Baseline:
    name = "td3"
    # Equal sibling solver: severity-aware (same observation as PPO). Severity
    # one-hot MUST stay visible — QoS targets d_phi are severity-dependent, so a
    # severity-blind policy cannot meet them and the PPO/TD3/SAC comparison is unfair.
    FLAGS = BaselineFlags(use_phase=True, use_cmdp=True, use_hrl=True, n_constraints=5)

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 6,
        seed: int = 0,
        alpha_lambda: float | None = None,
        device: str = "cpu",
        action_low: tuple[float, ...] | None = None,
        action_high: tuple[float, ...] | None = None,
        K: int = 1,
    ) -> None:
        self.flags = self.FLAGS
        if action_low is None:
            action_low = (-1.0, -1.0, 0.0, 0.0, 0.0, 0.0) + ((0.0,) if action_dim >= 7 else ())
        if action_high is None:
            action_high = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0) + ((1.0,) if action_dim >= 7 else ())
        self.td3 = TD3Agent(
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
        self.lambda_state = LambdaState(K=K, **kwargs)
        self.action_low = np.asarray(action_low, dtype=np.float32)
        self.action_high = np.asarray(action_high, dtype=np.float32)
        self.manager = TD3ManagerAgent(state_dim=manager_state_dim(K), seed=seed)

    # ------------------------------------------------------------------
    # Severity masking
    # ------------------------------------------------------------------

    def maybe_mask(self, obs):
        # Severity-aware sibling (use_phase=True): keep the severity one-hot intact.
        return mask_severity(obs) if not self.flags.use_phase else obs

    def select_action(self, obs, deterministic: bool = False):
        masked = self.maybe_mask(obs)
        action = self.td3.select_action(masked, deterministic=deterministic)
        # Match PPO API tuple (action, log_prob, value) — TD3 is deterministic
        return action, 0.0, 0.0

    # ------------------------------------------------------------------
    # LambdaState lifecycle (sibling API)
    # ------------------------------------------------------------------

    def on_episode_start(self, severity_per_amb, severity_ref: int) -> None:
        self.lambda_state.reset_episode(severity_per_amb, severity_ref)

    def on_manager_step_start(self, severity_per_amb, severity_ref: int) -> None:
        self.lambda_state.on_manager_step_start(severity_per_amb, severity_ref)

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
        self.td3.store(
            self.maybe_mask(obs),
            action,
            float(reward),
            self.maybe_mask(next_obs),
            bool(done),
        )

    def update(self, buffer=None) -> dict:
        out = self.td3.update()
        lam = self.lambda_state.get_lambda_global()
        K = self.lambda_state.K
        for k in range(K):
            out[f"lambda_global_C1_{k}"] = float(lam[k])
            out[f"lambda_global_C2_{k}"] = float(lam[K + k])
            out[f"lambda_global_C4_{k}"] = float(lam[2 * K + k])
            out[f"lambda_global_C5_{k}"] = float(lam[3 * K + k])
        out["lambda_global_C3_shared"] = float(lam[4 * K])
        return out

    def save(self, path: str) -> None:
        self.td3.save(path)
        self.manager.save(path.replace(".pt", "_manager.pt"))

    def load(self, path: str) -> None:
        self.td3.load(path)
        mgr_path = path.replace(".pt", "_manager.pt")
        if os.path.exists(mgr_path):
            self.manager.load(mgr_path)
