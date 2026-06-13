"""Twin Delayed DDPG (TD3) — off-policy continuous-action actor-critic.

Reference:
    Fujimoto, van Hoof, Meger 2018 "Addressing Function Approximation Error
    in Actor-Critic Methods", ICML.

Design choices:
    - Twin critics Q1, Q2 → take min for target → reduce overestimation
    - Target networks for actor + both critics → Polyak update τ=0.005
    - Clipped double Q-learning: y = r + γ·min(Q1_target, Q2_target)(s', a')
    - Delayed policy update every `policy_delay`=2 critic steps
    - Target policy smoothing: noise on next-action when computing target
    - Tanh-squashed actor with rescale to action range
    - Replay buffer (uniform sampling)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import GAMMA


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


class DeterministicActor(nn.Module):
    """Tanh-squashed MLP actor producing a deterministic action in [low, high]."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        hidden: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        self.net = _make_mlp(state_dim, hidden, action_dim)
        self.register_buffer("action_low", torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(action_high, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        raw = self.net(obs)
        squashed = torch.tanh(raw)                                   # [-1, 1]
        return self.action_low + (self.action_high - self.action_low) * 0.5 * (squashed + 1.0)


class QCritic(nn.Module):
    """Q(s, a): MLP scalar value of state-action pair."""

    def __init__(self, state_dim: int, action_dim: int, hidden: Sequence[int] = (256, 256)) -> None:
        super().__init__()
        self.net = _make_mlp(state_dim + action_dim, hidden, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.net(x).squeeze(-1)


# ============================================================
# Replay buffer
# ============================================================


@dataclass
class ReplayBuffer:
    """Uniform-sampling replay buffer for off-policy training."""

    capacity: int
    state_dim: int
    action_dim: int

    obs: np.ndarray = field(init=False)
    actions: np.ndarray = field(init=False)
    rewards: np.ndarray = field(init=False)
    next_obs: np.ndarray = field(init=False)
    dones: np.ndarray = field(init=False)
    ptr: int = field(init=False, default=0)
    size: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.obs = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)

    def add(self, obs, action, reward: float, next_obs, done: bool) -> None:
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator | None = None
               ) -> tuple[torch.Tensor, ...]:
        if rng is None:
            idx = np.random.randint(0, self.size, size=batch_size)
        else:
            idx = rng.integers(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.obs[idx]),
            torch.as_tensor(self.actions[idx]),
            torch.as_tensor(self.rewards[idx]),
            torch.as_tensor(self.next_obs[idx]),
            torch.as_tensor(self.dones[idx]),
        )


# ============================================================
# TD3 agent
# ============================================================


class TD3Agent:
    """TD3 actor-critic with 2 critics, target networks, delayed policy update."""

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
        policy_delay: int = 2,
        target_noise_std: float = 0.2,
        target_noise_clip: float = 0.5,
        exploration_noise: float = 0.1,
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

        # Actor + target
        self.actor = DeterministicActor(state_dim, action_dim,
                                         self.action_low_np, self.action_high_np,
                                         hidden=hidden).to(self.device)
        self.actor_target = DeterministicActor(state_dim, action_dim,
                                                self.action_low_np, self.action_high_np,
                                                hidden=hidden).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        for p in self.actor_target.parameters():
            p.requires_grad_(False)

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

        # Hyperparams
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.target_noise_std = target_noise_std
        self.target_noise_clip = target_noise_clip
        self.exploration_noise = exploration_noise
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm

        self.buffer = ReplayBuffer(buffer_capacity, state_dim, action_dim)
        self.step_count = 0
        self._rng = np.random.default_rng(seed)

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        # Random exploration during warmup
        if not deterministic and self.step_count < self.warmup_steps:
            return self._rng.uniform(self.action_low_np, self.action_high_np).astype(np.float32)

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.actor(obs_t).squeeze(0).cpu().numpy()
        if not deterministic:
            scale = self.action_high_np - self.action_low_np
            noise = self._rng.normal(0.0, self.exploration_noise, size=self.action_dim) * scale
            action = action + noise.astype(np.float32)
            action = np.clip(action, self.action_low_np, self.action_high_np)
        return action.astype(np.float32)

    def store(self, obs, action, reward: float, next_obs, done: bool) -> None:
        self.buffer.add(obs, action, reward, next_obs, done)
        self.step_count += 1

    def update(self) -> dict[str, float]:
        """One gradient step on critic; actor + target sync every policy_delay steps."""
        if self.buffer.size < max(self.batch_size, self.warmup_steps):
            return {}

        obs, action, reward, next_obs, done = (
            t.to(self.device) for t in self.buffer.sample(self.batch_size, self._rng)
        )

        # --- Critic update with clipped double Q-learning + target policy smoothing ---
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.target_noise_std).clamp(
                -self.target_noise_clip, self.target_noise_clip
            )
            next_action = self.actor_target(next_obs) + noise
            # Clamp to action range
            low_t = torch.as_tensor(self.action_low_np, device=self.device)
            high_t = torch.as_tensor(self.action_high_np, device=self.device)
            next_action = next_action.clamp(low_t, high_t)
            q1_t = self.critic_1_target(next_obs, next_action)
            q2_t = self.critic_2_target(next_obs, next_action)
            target_q = reward + (1.0 - done) * self.gamma * torch.min(q1_t, q2_t)

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

        # --- Delayed actor update + target network Polyak update ---
        if self.step_count % self.policy_delay == 0:
            actor_action = self.actor(obs)
            actor_loss = -self.critic_1(obs, actor_action).mean()

            if not torch.isfinite(actor_loss):
                out["actor_loss_nan"] = 1.0
            else:
                self.opt_actor.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.opt_actor.step()
                out["actor_loss"] = float(actor_loss.item())

            # Polyak update target networks
            self._polyak_update(self.actor_target, self.actor)
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
                "actor_target": self.actor_target.state_dict(),
                "critic_1_target": self.critic_1_target.state_dict(),
                "critic_2_target": self.critic_2_target.state_dict(),
                "opt_actor": self.opt_actor.state_dict(),
                "opt_critic": self.opt_critic.state_dict(),
                "step_count": self.step_count,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic_1.load_state_dict(ckpt["critic_1"])
        self.critic_2.load_state_dict(ckpt["critic_2"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic_1_target.load_state_dict(ckpt["critic_1_target"])
        self.critic_2_target.load_state_dict(ckpt["critic_2_target"])
        if "opt_actor" in ckpt:
            self.opt_actor.load_state_dict(ckpt["opt_actor"])
        if "opt_critic" in ckpt:
            self.opt_critic.load_state_dict(ckpt["opt_critic"])
        self.step_count = int(ckpt.get("step_count", 0))
