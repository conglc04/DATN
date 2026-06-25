"""Worker agent (xApp π_L) for PPO.

Phase 3.3.3 / 3.3.4: low-level policy operating on full Worker state
(obs_dim = 20 + 11K + F; 32-dim for K=1, F=1 — per-ambulance severity_k epic
2026-06-15, exposes delay_norm_k, AoI_norm_k, severity_k_norm, per-amb λ_local
slots and active_mask_k) every Worker step (T_L = 10 ms sim = 20 MAC ticks).

Output action dimensions (pure-RL intra-slice, 2026-06-21):
    Manager owns inter-slice URLLC/eMBB budget. Worker owns only intra-URLLC
    priority — emitted as PURE per-vehicle logits (no β temperature slot;
    pure-RL allocation has no urgency-temperature term).
        K=1: 1-dim no-op continuous action; the single active ambulance gets
             the full URLLC budget regardless of the action value.
        K>=2: a_L = (ℓ_0, ..., ℓ_{K-1}) — K raw logits; env applies
              softmax(ℓ) → weights → PRB split. No rule, no N_req, no λ in
              the allocation path (severity awareness is fully learned).

Reference:
    - docs/13_methodology_walkthrough.md Phase 3.3.3 (Worker arch)
    - Three-rate hierarchy locked: α_πL=3e-4 (Worker fastest; config.LR_PI_L SSOT)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from agents.ppo_core import (
    approx_kl, compute_gae, entropy_bonus, explained_variance,
    ppo_clip_loss, value_loss,
)
from utils.config import (
    GAE_LAMBDA,
    GAMMA_WORKER,
    LR_PI_L,
    LR_V_L,
    MINIBATCH_SIZE,
    PPO_CLIP_EPS,
    PPO_K_EPOCHS,
)

WORKER_STATE_DIM_DEFAULT: int = 32   # 20 + 11K + F for K=1, F=1 (+active_mask_k 2026-06-23)
WORKER_ACTION_DIM_DEFAULT: int = 1   # K=1 no-op; K>=2 uses K pure logits (no β slot)
WORKER_ACTION_DIM_K3: int = 3        # K=3 → 3 per-vehicle logits
IDX_PRB_LOGITS_START: int = 0        # logits start at index 0 (no β prefix)


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
    """π_L — Gaussian policy over raw intra-URLLC priority actions."""

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
        self._zero_init_output_layer()
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)  # std ≈ 0.6

    def _zero_init_output_layer(self) -> None:
        """Zero the final layer so all K per-vehicle logits start tied (audit
        2026-06-24, ĐX1).

        Default `nn.Linear` init (Kaiming uniform) gives each output unit a
        small random weight/bias, so the K logits differ by a seed-dependent
        ~1.05-1.25x factor before any training. PPO's self-reinforcing policy
        gradient turns that initial asymmetry into a persistent,
        severity-unconditional PRB allocation bias (one ambulance favored
        regardless of its actual severity). Zeroing only the output layer's
        weight and bias makes mean_net(obs) == 0 for every obs at init — all
        K logits start exactly equal (softmax → uniform) — while leaving the
        hidden layers' random init untouched for feature learning. Applied
        once at construction; not re-applied when the active-ambulance count
        changes during an episode.
        """
        output_layer = self.mean_net[-1]
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)

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


def decode_worker_action(a_raw: np.ndarray) -> dict[str, np.ndarray]:
    """Decode raw Worker action into intra-URLLC per-vehicle priority weights.

    Pure-RL layout (2026-06-21, matches env._apply_action / _prb_split_intra_slice):
        len=1:        K=1 no-op action; single ambulance gets full B_U (weight [1.0]).
        len=K, K>=2:  K raw per-vehicle logits ℓ_k → softmax → weights w_k.
    No β slot — pure-RL allocation has no urgency-temperature term.

    NOTE: this mirrors the env's decode (the single source of truth used in
    training is env._apply_action, called via env.step()). Kept as the
    documented decode contract; training does NOT route through this helper.
    """
    a = np.asarray(a_raw, dtype=np.float32)
    if a.ndim != 1 or a.shape[0] == 0:
        raise ValueError(
            "Worker action must have shape (1,) for K=1 or (K,) for K>=2"
        )
    logits = a[IDX_PRB_LOGITS_START:].astype(np.float32)
    if a.shape[0] == 1:
        # K=1 no-op: logit has no numeric effect; single ambulance → full budget.
        prb_weights = np.ones(1, dtype=np.float32)
    else:
        stable = logits - logits.max()
        exp = np.exp(stable)
        prb_weights = (exp / exp.sum()).astype(np.float32)
    return {
        "prb_logits": logits,
        "prb_weights": prb_weights,
    }


class WorkerAgent:
    """xApp Worker — Gaussian PPO actor-critic with γ_L=0.99 (clipped PPO + GAE).

    Intra-URLLC PRB split is a pure-RL softmax over the Worker's per-vehicle
    logits (env._prb_split_intra_slice — no rules, no N_req, no λ, no β). The
    only env-side guard is an anti-starvation floor (PRB_MIN_QOS) and the
    largest-remainder integer projection so Σ PRB = B_U exactly.
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
        """One PPO epoch sweep with γ_L = γ_WORKER (clipped surrogate + entropy).

        P1 fix: when action_dim == 1 (K=1), the Worker action is a no-op
        (single ambulance gets all URLLC PRBs regardless). Skip actor gradient
        updates; only train the critic (value function still needed for GAE).
        P2 fix: accumulate metrics across all minibatches/epochs, return mean.
        """
        n = len(rewards)
        if n == 0:
            return {}
        skip_actor = self.action_dim == 1
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

        sum_actor_loss = sum_critic_loss = sum_entropy = sum_clip_frac = 0.0
        sum_approx_kl = 0.0
        n_mb = 0
        idx_all = np.arange(n)
        for _ in range(self.k_epochs):
            np.random.shuffle(idx_all)
            for start in range(0, n, self.minibatch_size):
                idx = idx_all[start : start + self.minibatch_size]
                if len(idx) == 0:
                    continue
                b_obs = obs_t[idx]
                b_ret = ret_t[idx]

                if not skip_actor:
                    b_act = act_t[idx]
                    b_oldlp = oldlp_t[idx]
                    b_adv = adv_t[idx]

                    dist = self.actor(b_obs)
                    new_log_probs = dist.log_prob(b_act).sum(-1)
                    ent = entropy_bonus(dist)
                    policy_loss, clip_frac = ppo_clip_loss(
                        new_log_probs, b_oldlp, b_adv, self.clip_eps
                    )
                    actor_loss = policy_loss - self.ent_coef * ent

                    kl = approx_kl(b_oldlp, new_log_probs)

                    if torch.isfinite(actor_loss):
                        self.opt_actor.zero_grad()
                        actor_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                        self.opt_actor.step()
                        sum_actor_loss += float(actor_loss.item())
                        sum_entropy += float(ent.item())
                        sum_clip_frac += float(clip_frac.item())
                        sum_approx_kl += float(kl.item())

                v = self.critic(b_obs)
                critic_loss = value_loss(v, b_ret, self.vf_coef)
                if torch.isfinite(critic_loss):
                    self.opt_critic.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.opt_critic.step()
                    sum_critic_loss += float(critic_loss.item())

                n_mb += 1

        ev = explained_variance(values, returns)
        d = max(n_mb, 1)
        return {
            "worker_actor_loss": sum_actor_loss / d if not skip_actor else 0.0,
            "worker_critic_loss": sum_critic_loss / d,
            "worker_entropy": sum_entropy / d if not skip_actor else 0.0,
            "worker_clip_fraction": sum_clip_frac / d if not skip_actor else 0.0,
            "worker_approx_kl": sum_approx_kl / d if not skip_actor else 0.0,
            "worker_explained_variance": ev,
            "worker_n_samples": n,
            "worker_actor_skipped_k1": int(skip_actor),
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
