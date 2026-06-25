"""SAC sibling solver — off-policy SAC + HRL Manager+Worker + (4K+1)-dim λ (W07).

Phase 3 sibling solver — applied AFTER Phase 2 problem statement complete
(end of W06). SAC uses the SAME HRL two-timescale architecture as PPO and TD3:
    - SACManagerAgent (100ms, inter-slice b_rrm) — stochastic SAC core
    - SACWorker (10ms, intra-URLLC per-vehicle logits, pure-RL, no β) — max-entropy SAC core
    - (4K+1)-dim LambdaState dual ascent (1 per Manager step boundary)

Differences from TD3:
    - Stochastic max-entropy actor (automatic temperature α) instead of
      deterministic actor + exploration noise

Used in Table I alongside TD3: same CMDP problem, same HRL, only the RL core
differs — fair comparison across on-policy/off-policy families.

Reference:
    - Haarnoja et al. 2018 "Soft Actor-Critic" (ICML) +
      "SAC Algorithms and Applications" (arXiv:1812.05905)
    - docs/13_methodology_walkthrough.md Phase 2.3 (Lagrangian dual)
"""

from __future__ import annotations

import os

import numpy as np

from agents.lagrangian import LambdaState
from agents.manager_agent import SACManagerAgent, manager_state_dim
from agents.sac_agent import SACAgent
from solvers._common import BaselineFlags, mask_severity


class SACSolver:
    name = "sac"
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
            # Clean Worker layout: K=1 no-op or K>=2 pure per-vehicle logits (no β).
            action_low = (-3.0,) * action_dim
        if action_high is None:
            action_high = (3.0,) * action_dim
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
            zero_init_output=True,  # ĐX1 (audit 2026-06-24): this IS the Worker
        )
        kwargs = {} if alpha_lambda is None else {"alpha_lambda": alpha_lambda}
        self.lambda_state = LambdaState(K=K, **kwargs)
        self.action_low = np.asarray(action_low, dtype=np.float32)
        self.action_high = np.asarray(action_high, dtype=np.float32)
        self.manager = SACManagerAgent(state_dim=manager_state_dim(K), seed=seed)

    # ------------------------------------------------------------------
    # Severity masking
    # ------------------------------------------------------------------

    def maybe_mask(self, obs):
        # Severity-aware sibling (use_phase=True is a legacy flag name): keep the
        # severity one-hot intact. (phase FSM removed — phase->severity swap.)
        return mask_severity(obs) if not self.flags.use_phase else obs

    def select_action(self, obs, deterministic: bool = False):
        masked = self.maybe_mask(obs)
        action = self.sac.select_action(masked, deterministic=deterministic)
        # Match PPO API tuple (action, log_prob, value) — SAC log_prob/value
        # are not surfaced through this sibling-solver interface.
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
        K = self.lambda_state.K
        for k in range(K):
            out[f"lambda_global_C1_{k}"] = float(lam[k])
            out[f"lambda_global_C2_{k}"] = float(lam[K + k])
            out[f"lambda_global_C4_{k}"] = float(lam[2 * K + k])
            out[f"lambda_global_C5_{k}"] = float(lam[3 * K + k])
        out["lambda_global_C3_shared"] = float(lam[4 * K])
        return out

    def save(self, path: str) -> None:
        self.sac.save(path)
        self.manager.save(path.replace(".pt", "_manager.pt"))

    def load(self, path: str) -> None:
        self.sac.load(path)
        mgr_path = path.replace(".pt", "_manager.pt")
        if os.path.exists(mgr_path):
            self.manager.load(mgr_path)
