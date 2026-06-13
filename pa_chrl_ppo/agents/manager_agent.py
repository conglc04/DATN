"""Manager agent (rApp π_H) for PPO.

Phase 3.3.1 / 3.3.2: high-level policy operating on coarse multi-cell aggregate
state every Manager step (T_H = 100 ms sim = 10 Worker steps).

Output action dimensions (Phase 2.3.2; MEC offload removed — B0b):
    a_H = (b_rrm,)             # 1-dim continuous
        b_rrm ∈ [0, 1]          RRM budget hint to Worker (post-decoded via sigmoid)

Reference:
    - docs/13_methodology_walkthrough.md Phase 3.3.1 (Manager arch)
    - docs/13_methodology_walkthrough.md Phase 3.2.4 (γ_MANAGER = γ_WORKER^W ≈ 0.904)
    - Three-rate hierarchy locked: α_πH=1e-5 < α_λ=1e-4 < α_πL=1e-3
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from agents.ppo_core import compute_gae, entropy_bonus, ppo_clip_loss, value_loss
from utils.config import (
    GAE_LAMBDA,
    GAMMA_MANAGER,
    LR_PI_H,
    LR_V_H,
    MINIBATCH_SIZE,
    PPO_CLIP_EPS,
    PPO_K_EPOCHS,
)

MANAGER_STATE_DIM_DEFAULT: int = 11
MANAGER_ACTION_DIM_DEFAULT: int = 1


def _make_mlp(in_dim: int, hidden: Sequence[int], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class ManagerActor(nn.Module):
    """π_H — Gaussian policy over (b_rrm,).

    Outputs (mean, log_std) for 1-dim action. Post-processing (sigmoid scaling)
    happens in the decoder, NOT inside the actor, so log_prob math stays clean.
    """

    def __init__(
        self,
        state_dim: int = MANAGER_STATE_DIM_DEFAULT,
        action_dim: int = MANAGER_ACTION_DIM_DEFAULT,
        hidden: Sequence[int] = (256, 128, 64),
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.mean_net = _make_mlp(state_dim, hidden, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)  # init std ≈ 0.6

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.mean_net(obs)
        std = torch.exp(self.log_std.clamp(-5.0, 2.0))
        return Normal(mean, std)

    def forward(self, obs: torch.Tensor) -> Normal:
        return self.distribution(obs)


class ManagerCritic(nn.Module):
    """V_H — state-value of high-level state s_H."""

    def __init__(
        self,
        state_dim: int = MANAGER_STATE_DIM_DEFAULT,
        hidden: Sequence[int] = (256, 128, 64),
    ) -> None:
        super().__init__()
        self.net = _make_mlp(state_dim, hidden, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


def decode_manager_action(action_raw: np.ndarray) -> dict[str, float]:
    """Map raw Gaussian sample → (b_rrm,) on its proper support.

    a_raw is unbounded Gaussian; we squash via sigmoid.
    """
    b_rrm = float(1.0 / (1.0 + np.exp(-float(action_raw[0]))))                 # [0, 1]
    return {"b_rrm": b_rrm}


class ManagerAgent:
    """rApp Manager — Gaussian PPO actor-critic with γ_H ≈ 0.904 (N1).

    State_dim default = 11 (W07 placeholder; finalize in W08 with explicit
    multi-cell aggregator).
    """

    def __init__(
        self,
        state_dim: int = MANAGER_STATE_DIM_DEFAULT,
        action_dim: int = MANAGER_ACTION_DIM_DEFAULT,
        actor_hidden: Sequence[int] = (256, 128, 64),
        critic_hidden: Sequence[int] = (256, 128, 64),
        lr_actor: float = LR_PI_H,
        lr_critic: float = LR_V_H,
        gamma: float = GAMMA_MANAGER,
        gae_lambda: float = GAE_LAMBDA,
        clip_eps: float = PPO_CLIP_EPS,
        k_epochs: int = PPO_K_EPOCHS,
        minibatch_size: int = MINIBATCH_SIZE,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> None:
        if seed is not None:
            torch.manual_seed(seed)
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor = ManagerActor(state_dim, action_dim, actor_hidden).to(self.device)
        self.critic = ManagerCritic(state_dim, critic_hidden).to(self.device)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.k_epochs = k_epochs
        self.minibatch_size = minibatch_size
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(
        self,
        s_H: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, float, float]:
        """Sample a_H ~ π_H(·|s_H). Returns (action_raw, log_prob, value)."""
        obs_t = torch.as_tensor(s_H, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = self.actor(obs_t)
        action = dist.mean if deterministic else dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        value = self.critic(obs_t)
        return (
            action.squeeze(0).cpu().numpy().astype(np.float32),
            float(log_prob.item()),
            float(value.item()),
        )

    # ------------------------------------------------------------------
    # PPO update (mirrors WorkerAgent pattern for consistency)
    # ------------------------------------------------------------------

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_value: float,
    ) -> dict[str, float]:
        """One PPO epoch sweep over a rollout buffer with γ_H = γ_MANAGER."""
        n = len(rewards)
        if n == 0:
            return {}
        advantages, returns = compute_gae(
            rewards, values, dones, last_value, self.gamma, self.gae_lambda
        )
        # Normalize advantages
        if advantages.size > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        oldlp_t = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        last_actor_loss = last_critic_loss = last_entropy = last_clip_frac = 0.0
        idx_all = np.arange(n)
        for _ in range(self.k_epochs):
            np.random.shuffle(idx_all)
            for start in range(0, n, self.minibatch_size):
                idx = idx_all[start : start + self.minibatch_size]
                if len(idx) == 0:
                    continue
                b_obs = obs_t[idx]
                b_act = act_t[idx]
                b_oldlp = oldlp_t[idx]
                b_adv = adv_t[idx]
                b_ret = ret_t[idx]

                dist = self.actor(b_obs)
                new_log_probs = dist.log_prob(b_act).sum(-1)
                ent = entropy_bonus(dist)

                policy_loss, clip_frac = ppo_clip_loss(
                    new_log_probs, b_oldlp, b_adv, self.clip_eps
                )
                actor_loss = policy_loss - self.ent_coef * ent
                if not torch.isfinite(actor_loss):
                    continue

                self.opt_actor.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.opt_actor.step()

                v = self.critic(b_obs)
                critic_loss = value_loss(v, b_ret, self.vf_coef)
                if not torch.isfinite(critic_loss):
                    continue

                self.opt_critic.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.opt_critic.step()

                last_actor_loss = float(actor_loss.item())
                last_critic_loss = float(critic_loss.item())
                last_entropy = float(ent.item())
                last_clip_frac = float(clip_frac.item())

        return {
            "manager_actor_loss": last_actor_loss,
            "manager_critic_loss": last_critic_loss,
            "manager_entropy": last_entropy,
            "manager_clip_fraction": last_clip_frac,
            "manager_n_samples": n,
        }

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "opt_actor": self.opt_actor.state_dict(),
                "opt_critic": self.opt_critic.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        if "opt_actor" in ckpt:
            self.opt_actor.load_state_dict(ckpt["opt_actor"])
        if "opt_critic" in ckpt:
            self.opt_critic.load_state_dict(ckpt["opt_critic"])
