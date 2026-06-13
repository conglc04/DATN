"""Worker agent (xApp π_L) for PA-CHRL-PPO.

Phase 3.3.3 / 3.3.4: low-level policy operating on full Worker state (33-dim
for K=1, F=4) every Worker step (T_L = 10 ms sim = 20 MAC ticks).

Output action dimensions (Phase 2.3.2):
    a_L = (Δr_min, Δr_max, r_ded_ratio, w_intra^C1, w_intra^C2, w_intra^C3)
    6-dim continuous. Decoded via per-component squashing:
        Δr_min, Δr_max ∈ [-0.1, +0.1]    (0.1 · tanh)
        r_ded_ratio    ∈ [0, 1]          (sigmoid; r_ded = r_ded_ratio · r_min)
        w_intra^C1..C3 ∈ Δ³ simplex       (softmax)

Reference:
    - docs/13_methodology_walkthrough.md Phase 3.3.3 (Worker arch)
    - Three-rate hierarchy locked: α_πL=1e-3 (Worker fastest)
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
    GAMMA_WORKER,
    LR_PI_L,
    LR_V_L,
    MINIBATCH_SIZE,
    PPO_CLIP_EPS,
    PPO_K_EPOCHS,
)

WORKER_STATE_DIM_DEFAULT: int = 33
WORKER_ACTION_DIM_DEFAULT: int = 6
# Component indices in the 6-dim raw action vector
IDX_DELTA_R_MIN: int = 0
IDX_DELTA_R_MAX: int = 1
IDX_R_DED_RATIO: int = 2
IDX_W_INTRA_START: int = 3   # w_intra^C1, w_intra^C2, w_intra^C3 (indices 3..5)
DELTA_R_SCALE: float = 0.1   # Δr_min, Δr_max bound (Phase 2.3.2)


def _make_mlp(in_dim: int, hidden: Sequence[int], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class WorkerActor(nn.Module):
    """π_L — Gaussian policy over 6-dim continuous a_L (raw, pre-squash)."""

    def __init__(
        self,
        state_dim: int = WORKER_STATE_DIM_DEFAULT,
        action_dim: int = WORKER_ACTION_DIM_DEFAULT,
        hidden: Sequence[int] = (512, 256, 128, 64),
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.mean_net = _make_mlp(state_dim, hidden, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)  # std ≈ 0.6

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.mean_net(obs)
        std = torch.exp(self.log_std.clamp(-5.0, 2.0))
        return Normal(mean, std)

    def forward(self, obs: torch.Tensor) -> Normal:
        return self.distribution(obs)


class WorkerCritic(nn.Module):
    """V_L — state-value of low-level state s_L."""

    def __init__(
        self,
        state_dim: int = WORKER_STATE_DIM_DEFAULT,
        hidden: Sequence[int] = (512, 256, 128, 64),
    ) -> None:
        super().__init__()
        self.net = _make_mlp(state_dim, hidden, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


def decode_worker_action(a_raw: np.ndarray) -> dict[str, float | np.ndarray]:
    """Squash 6-dim raw Gaussian sample → physical Worker action.

    Δr_min, Δr_max  →  0.1 · tanh(raw)
    r_ded_ratio     →  sigmoid(raw)
    w_intra^C1..C3  →  softmax over (raw_3, raw_4, raw_5)
    """
    a = np.asarray(a_raw, dtype=np.float32)
    if a.shape != (WORKER_ACTION_DIM_DEFAULT,):
        raise ValueError(f"a_raw shape {a.shape} != ({WORKER_ACTION_DIM_DEFAULT},)")
    d_r_min = float(DELTA_R_SCALE * np.tanh(a[IDX_DELTA_R_MIN]))
    d_r_max = float(DELTA_R_SCALE * np.tanh(a[IDX_DELTA_R_MAX]))
    r_ded_ratio = float(1.0 / (1.0 + np.exp(-a[IDX_R_DED_RATIO])))
    logits = a[IDX_W_INTRA_START : IDX_W_INTRA_START + 3]
    logits = logits - logits.max()  # numerical stability
    exp = np.exp(logits)
    w_intra = (exp / exp.sum()).astype(np.float32)
    return {
        "delta_r_min": d_r_min,
        "delta_r_max": d_r_max,
        "r_ded_ratio": r_ded_ratio,
        "w_intra": w_intra,           # shape (3,), simplex
    }


class WorkerAgent:
    """xApp Worker — Gaussian PPO actor-critic with γ_L=0.99 (clipped PPO + GAE).

    Safety is enforced downstream by the closed-form feasibility projection
    Π_feasible (no learnable params); the Worker policy is standard PPO.
    """

    def __init__(
        self,
        state_dim: int = WORKER_STATE_DIM_DEFAULT,
        action_dim: int = WORKER_ACTION_DIM_DEFAULT,
        actor_hidden: Sequence[int] = (512, 256, 128, 64),
        critic_hidden: Sequence[int] = (512, 256, 128, 64),
        lr_actor: float = LR_PI_L,
        lr_critic: float = LR_V_L,
        gamma: float = GAMMA_WORKER,
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
        self.actor = WorkerActor(state_dim, action_dim, actor_hidden).to(self.device)
        self.critic = WorkerCritic(state_dim, critic_hidden).to(self.device)
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
        s_L: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, float, float]:
        """Sample a_L ~ π_L(·|s_L). Returns (action_raw, log_prob, value)."""
        obs_t = torch.as_tensor(s_L, dtype=torch.float32, device=self.device).unsqueeze(0)
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
    # PPO update (clipped surrogate + GAE)
    # ------------------------------------------------------------------

    def update(
        self,
        obs: np.ndarray,
        actions_raw: np.ndarray,
        old_log_probs: np.ndarray,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_value: float,
    ) -> dict[str, float]:
        """One PPO epoch sweep with γ_L = γ_WORKER (clipped surrogate + entropy)."""
        n = len(rewards)
        if n == 0:
            return {}
        advantages, returns = compute_gae(
            rewards, values, dones, last_value, self.gamma, self.gae_lambda
        )
        if advantages.size > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(actions_raw, dtype=torch.float32, device=self.device)
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
            "worker_actor_loss": last_actor_loss,
            "worker_critic_loss": last_critic_loss,
            "worker_entropy": last_entropy,
            "worker_clip_fraction": last_clip_frac,
            "worker_n_samples": n,
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
