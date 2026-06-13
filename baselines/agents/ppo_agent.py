"""Generic PPO actor-critic.

Week 5 scope: minimal but functional implementation for solvers smoke-train.
Week 8 will extend with Manager/Worker hierarchical specifics.

Reference:
    - Schulman et al. 2017 "Proximal Policy Optimization Algorithms"
    - docs/08_implementation_notes.md Bảng 4.2B (DNN architecture)
    - docs/09_execution_plan.md hyperparameters (γ=0.99, λ_GAE=0.95, ε_clip=0.2)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from utils.config import (
    GAE_LAMBDA,
    GAMMA,
    LR_PI_L,
    LR_V_L,
    MINIBATCH_SIZE,
    PPO_CLIP_EPS,
    PPO_K_EPOCHS,
)


# ============================================================
# Networks
# ============================================================


def _make_mlp(in_dim: int, hidden: Sequence[int], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class ContinuousActor(nn.Module):
    """Gaussian policy with state-independent log_std (standard for PPO)."""

    def __init__(self, state_dim: int, action_dim: int, hidden: Sequence[int] = (256, 128, 64)):
        super().__init__()
        self.mean_net = _make_mlp(state_dim, hidden, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)   # init std≈0.6

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.mean_net(obs)
        std = torch.exp(self.log_std.clamp(-5.0, 2.0))
        return Normal(mean, std)

    def forward(self, obs: torch.Tensor) -> Normal:
        return self.distribution(obs)


class Critic(nn.Module):
    """State-value function."""

    def __init__(self, state_dim: int, hidden: Sequence[int] = (256, 128, 64)):
        super().__init__()
        self.net = _make_mlp(state_dim, hidden, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


# ============================================================
# Rollout buffer
# ============================================================


@dataclass
class RolloutBuffer:
    """Fixed-capacity on-policy buffer storing transitions for one batch update."""

    capacity: int
    state_dim: int
    action_dim: int

    obs: np.ndarray = field(init=False)
    actions: np.ndarray = field(init=False)
    log_probs: np.ndarray = field(init=False)
    rewards: np.ndarray = field(init=False)
    values: np.ndarray = field(init=False)
    dones: np.ndarray = field(init=False)
    advantages: np.ndarray = field(init=False)
    returns: np.ndarray = field(init=False)
    ptr: int = field(init=False, default=0)

    def __post_init__(self):
        self.obs = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.log_probs = np.zeros(self.capacity, dtype=np.float32)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.values = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.advantages = np.zeros(self.capacity, dtype=np.float32)
        self.returns = np.zeros(self.capacity, dtype=np.float32)
        self.ptr = 0

    @property
    def full(self) -> bool:
        return self.ptr >= self.capacity

    def add(self, obs, action, log_prob, reward, value, done) -> None:
        if self.ptr >= self.capacity:
            return
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.rewards[i] = reward
        self.values[i] = value
        self.dones[i] = float(done)
        self.ptr += 1

    def compute_gae(self, last_value: float, gamma: float = GAMMA, lam: float = GAE_LAMBDA) -> None:
        """Standard Generalized Advantage Estimation."""
        n = self.ptr
        adv = np.zeros(n, dtype=np.float32)
        gae = 0.0
        next_v = last_value
        for t in reversed(range(n)):
            nonterm = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_v * nonterm - self.values[t]
            gae = delta + gamma * lam * nonterm * gae
            adv[t] = gae
            next_v = self.values[t]
        self.advantages[:n] = adv
        self.returns[:n] = adv + self.values[:n]

    def reset(self) -> None:
        self.ptr = 0


# ============================================================
# PPO agent
# ============================================================


class PPOAgent:
    """PPO actor-critic for continuous control.

    Single environment, on-policy, clip-objective.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        actor_hidden: Sequence[int] = (256, 128, 64),
        critic_hidden: Sequence[int] = (256, 128, 64),
        lr_actor: float = LR_PI_L,
        lr_critic: float = LR_V_L,
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
        self.actor = ContinuousActor(state_dim, action_dim, actor_hidden).to(self.device)
        self.critic = Critic(state_dim, critic_hidden).to(self.device)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.clip_eps = clip_eps
        self.k_epochs = k_epochs
        self.minibatch_size = minibatch_size
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.action_dim = action_dim

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> tuple[np.ndarray, float, float]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = self.actor(obs_t)
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        value = self.critic(obs_t)
        return (
            action.squeeze(0).cpu().numpy(),
            float(log_prob.item()),
            float(value.item()),
        )

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        """Run k_epochs of clipped PPO updates over the buffered rollout."""
        n = buffer.ptr
        if n == 0:
            return {}
        obs = torch.as_tensor(buffer.obs[:n], device=self.device)
        actions = torch.as_tensor(buffer.actions[:n], device=self.device)
        old_log_probs = torch.as_tensor(buffer.log_probs[:n], device=self.device)
        advantages = torch.as_tensor(buffer.advantages[:n], device=self.device)
        returns = torch.as_tensor(buffer.returns[:n], device=self.device)

        # Normalise advantages (per-batch)
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        last_actor_loss = last_critic_loss = last_entropy = 0.0
        idx_all = np.arange(n)
        for _ in range(self.k_epochs):
            np.random.shuffle(idx_all)
            for start in range(0, n, self.minibatch_size):
                idx = idx_all[start : start + self.minibatch_size]
                if len(idx) == 0:
                    continue
                b_obs = obs[idx]
                b_act = actions[idx]
                b_oldlp = old_log_probs[idx]
                b_adv = advantages[idx]
                b_ret = returns[idx]

                dist = self.actor(b_obs)
                new_log_probs = dist.log_prob(b_act).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                ratio = torch.exp(new_log_probs - b_oldlp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_adv
                actor_loss = -torch.min(surr1, surr2).mean() - self.ent_coef * entropy

                # NaN guard: skip the step if the loss exploded
                if not torch.isfinite(actor_loss):
                    continue

                self.opt_actor.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.opt_actor.step()

                values = self.critic(b_obs)
                critic_loss = F.mse_loss(values, b_ret) * self.vf_coef

                if not torch.isfinite(critic_loss):
                    continue

                self.opt_critic.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.opt_critic.step()

                last_actor_loss = float(actor_loss.item())
                last_critic_loss = float(critic_loss.item())
                last_entropy = float(entropy.item())

        return {
            "actor_loss": last_actor_loss,
            "critic_loss": last_critic_loss,
            "entropy": last_entropy,
            "n_samples": n,
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
