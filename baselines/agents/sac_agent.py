"""Soft Actor-Critic (SAC) — off-policy max-entropy continuous-action actor-critic.

Reference:
    Haarnoja et al. 2018 "Soft Actor-Critic: Off-Policy Maximum Entropy Deep
    Reinforcement Learning with a Stochastic Actor", ICML.
    Haarnoja et al. 2018 "Soft Actor-Critic Algorithms and Applications"
    (automatic temperature/entropy tuning, arXiv:1812.05905).

Design choices (mirrors agents/td3_agent.py for sibling parity):
    - Twin critics Q1, Q2 → take min for target → reduce overestimation
    - Target critic networks, Polyak update τ=0.005
    - Stochastic tanh-Gaussian actor (reparameterization trick) with
      rescale to action range — replaces TD3's deterministic + noise actor
    - Automatic entropy coefficient α tuning via target entropy
      = -action_dim (standard heuristic, Haarnoja 2018 §5)
    - Replay buffer (uniform sampling) — reused from agents.td3_agent
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.td3_agent import QCritic, ReplayBuffer, _make_mlp
from utils.config import GAMMA

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


# ============================================================
# Stochastic actor
# ============================================================


class GaussianTanhActor(nn.Module):
    """Tanh-squashed Gaussian policy with reparameterized sampling.

    Outputs (mean, log_std) of a diagonal Gaussian over a [-1, 1]^d latent
    action, squashed by tanh, then rescaled to [action_low, action_high].
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        hidden: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        prev = state_dim
        layers: list[nn.Module] = []
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev, action_dim)
        self.log_std_head = nn.Linear(prev, action_dim)
        self.register_buffer("action_low", torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(action_high, dtype=torch.float32))

    def _dist_params(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def _to_action_range(self, squashed: torch.Tensor) -> torch.Tensor:
        return self.action_low + (self.action_high - self.action_low) * 0.5 * (squashed + 1.0)

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reparameterized sample. Returns (action, log_prob) with log_prob
        summed over action dims, corrected for the tanh squashing
        (Haarnoja 2018 Appendix C)."""
        mean, log_std = self._dist_params(obs)
        std = log_std.exp()
        eps = torch.randn_like(mean)
        pre_tanh = mean + eps * std
        squashed = torch.tanh(pre_tanh)
        action = self._to_action_range(squashed)

        # log N(pre_tanh; mean, std)
        log_prob = (-0.5 * ((pre_tanh - mean) / std) ** 2 - log_std - 0.5 * math.log(2.0 * math.pi))
        log_prob = log_prob.sum(dim=-1)
        # Jacobian correction for tanh + affine rescale: -log(1 - tanh^2(x)) - log(scale/2)
        scale = (self.action_high - self.action_low) * 0.5
        log_prob -= torch.log(1.0 - squashed.pow(2) + 1e-6).sum(dim=-1)
        log_prob -= torch.log(scale).sum(dim=-1)
        return action, log_prob

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self._dist_params(obs)
        return self._to_action_range(torch.tanh(mean))


# ============================================================
# SAC agent
# ============================================================


class SACAgent:
    """SAC actor-critic with twin Q, target critics, automatic entropy tuning."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        hidden: Sequence[int] = (256, 256),
        gamma: float = GAMMA,
        tau: float = 0.005,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        init_alpha: float = 0.2,
        target_entropy: float | None = None,
        buffer_capacity: int = 100_000,
        batch_size: int = 256,
        warmup_steps: int = 1000,
        max_grad_norm: float = 1.0,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> None:
        if seed is not None:
            torch.manual_seed(seed)
        self.device = torch.device(device)
        self.action_dim = action_dim
        self.action_low_np = np.asarray(action_low, dtype=np.float32)
        self.action_high_np = np.asarray(action_high, dtype=np.float32)

        # Actor (stochastic, no target network needed)
        self.actor = GaussianTanhActor(
            state_dim, action_dim, self.action_low_np, self.action_high_np, hidden=hidden
        ).to(self.device)

        # Twin critics + targets
        self.critic_1 = QCritic(state_dim, action_dim, hidden=hidden).to(self.device)
        self.critic_2 = QCritic(state_dim, action_dim, hidden=hidden).to(self.device)
        self.critic_1_target = QCritic(state_dim, action_dim, hidden=hidden).to(self.device)
        self.critic_2_target = QCritic(state_dim, action_dim, hidden=hidden).to(self.device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        for p in self.critic_1_target.parameters():
            p.requires_grad_(False)
        for p in self.critic_2_target.parameters():
            p.requires_grad_(False)

        # Optimizers
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.opt_critic = torch.optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            lr=critic_lr,
        )

        # Automatic entropy (temperature) tuning — target entropy heuristic
        # = -action_dim (Haarnoja 2018 "SAC Algorithms and Applications" §5).
        self.target_entropy = (
            float(target_entropy) if target_entropy is not None else -float(action_dim)
        )
        self.log_alpha = torch.tensor(
            math.log(init_alpha), dtype=torch.float32, device=self.device, requires_grad=True
        )
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        # Hyperparams
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm

        self.buffer = ReplayBuffer(buffer_capacity, state_dim, action_dim)
        self.step_count = 0
        self._rng = np.random.default_rng(seed)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        # Random exploration during warmup (mirrors TD3Agent)
        if not deterministic and self.step_count < self.warmup_steps:
            return self._rng.uniform(self.action_low_np, self.action_high_np).astype(np.float32)

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if deterministic:
            action = self.actor.deterministic(obs_t)
        else:
            action, _ = self.actor.sample(obs_t)
        action = action.squeeze(0).cpu().numpy()
        return np.clip(action, self.action_low_np, self.action_high_np).astype(np.float32)

    def store(self, obs, action, reward: float, next_obs, done: bool) -> None:
        self.buffer.add(obs, action, reward, next_obs, done)
        self.step_count += 1

    def update(self) -> dict[str, float]:
        """One gradient step on critics, actor, and entropy temperature α."""
        if self.buffer.size < max(self.batch_size, self.warmup_steps):
            return {}

        obs, action, reward, next_obs, done = (
            t.to(self.device) for t in self.buffer.sample(self.batch_size, self._rng)
        )

        # --- Critic update: soft Bellman target with entropy bonus ---
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_obs)
            q1_t = self.critic_1_target(next_obs, next_action)
            q2_t = self.critic_2_target(next_obs, next_action)
            min_q_t = torch.min(q1_t, q2_t) - self.alpha * next_log_prob
            target_q = reward + (1.0 - done) * self.gamma * min_q_t

        q1 = self.critic_1(obs, action)
        q2 = self.critic_2(obs, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        if not torch.isfinite(critic_loss):
            return {"critic_loss_nan": 1.0}

        self.opt_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            self.max_grad_norm,
        )
        self.opt_critic.step()

        out = {"critic_loss": float(critic_loss.item())}

        # --- Actor update: maximize Q - α·log_prob ---
        new_action, log_prob = self.actor.sample(obs)
        q1_pi = self.critic_1(obs, new_action)
        q2_pi = self.critic_2(obs, new_action)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * log_prob - min_q_pi).mean()

        if not torch.isfinite(actor_loss):
            out["actor_loss_nan"] = 1.0
        else:
            self.opt_actor.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.opt_actor.step()
            out["actor_loss"] = float(actor_loss.item())

        # --- Entropy temperature update ---
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        if torch.isfinite(alpha_loss):
            self.opt_alpha.zero_grad()
            alpha_loss.backward()
            self.opt_alpha.step()
            out["alpha_loss"] = float(alpha_loss.item())
        out["alpha"] = float(self.alpha.item())

        # --- Polyak update target critics ---
        self._polyak_update(self.critic_1_target, self.critic_1)
        self._polyak_update(self.critic_2_target, self.critic_2)

        return out

    def _polyak_update(self, target: nn.Module, source: nn.Module) -> None:
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * sp.data)

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic_1": self.critic_1.state_dict(),
                "critic_2": self.critic_2.state_dict(),
                "critic_1_target": self.critic_1_target.state_dict(),
                "critic_2_target": self.critic_2_target.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "opt_actor": self.opt_actor.state_dict(),
                "opt_critic": self.opt_critic.state_dict(),
                "opt_alpha": self.opt_alpha.state_dict(),
                "step_count": self.step_count,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic_1.load_state_dict(ckpt["critic_1"])
        self.critic_2.load_state_dict(ckpt["critic_2"])
        self.critic_1_target.load_state_dict(ckpt["critic_1_target"])
        self.critic_2_target.load_state_dict(ckpt["critic_2_target"])
        with torch.no_grad():
            self.log_alpha.copy_(ckpt["log_alpha"].to(self.device))
        if "opt_actor" in ckpt:
            self.opt_actor.load_state_dict(ckpt["opt_actor"])
        if "opt_critic" in ckpt:
            self.opt_critic.load_state_dict(ckpt["opt_critic"])
        if "opt_alpha" in ckpt:
            self.opt_alpha.load_state_dict(ckpt["opt_alpha"])
        self.step_count = int(ckpt.get("step_count", 0))


__all__ = ["SACAgent", "GaussianTanhActor", "_make_mlp"]
