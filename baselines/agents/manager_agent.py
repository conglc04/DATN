"""Manager agent (high-level policy π_H, 100 ms cadence) for PPO.

Phase 3.3.1 / 3.3.2: high-level policy operating on coarse multi-cell aggregate
state every Manager step (T_H = 100 ms = 10 Worker steps).

Output action dimensions (Phase 2.3.2; MEC offload removed — B0b):
    a_H = (b_rrm,)             # 1-dim continuous
        b_rrm ∈ [B_RRM_MIN, B_RRM_MAX]   inter-slice URLLC budget (decode_manager_action:
                                          sigmoid → affine into the safe PRB range)

Reference:
    - docs/13_methodology_walkthrough.md Phase 3.3.1 (Manager arch)
    - docs/13_methodology_walkthrough.md Phase 3.2.4 (γ_MANAGER = γ_WORKER^W ≈ 0.904)
    - Three-rate hierarchy: α_πH=3e-5 < α_λ=2e-4 < α_πL=3e-4 (config.py SSOT; α_λ A/B
      5e-4 reverted to 2e-4 on 2026-06-22, see config.ALPHA_LAMBDA_DUAL history)
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
import copy

from utils.config import (
    B_RRM_MAX,
    B_RRM_MIN,
    GAE_LAMBDA,
    GAMMA,
    GAMMA_MANAGER,
    LR_PI_H,
    LR_V_H,
    MINIBATCH_SIZE,
    PPO_CLIP_EPS,
    PPO_K_EPOCHS,
)

MANAGER_ACTION_DIM_DEFAULT: int = 1


def manager_state_dim(K: int) -> int:
    """Manager state dim = 6 fixed scalars + (4K+1)-dim lambda_global.

    Fixed block: [rho_urllc, rho_emBB, bler, severity_ref_idx, aoi_mean, aoi_max].
    At K=1 this is 6 + 5 = 11 (numerically identical to the legacy 11-dim state).
    """
    return 6 + (4 * K + 1)


MANAGER_STATE_DIM_DEFAULT: int = manager_state_dim(1)  # 11 (K=1)


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
    """Map raw action → b_rrm ∈ [B_RRM_MIN, B_RRM_MAX].

    Applies sigmoid then affine shift so the decoded budget is always within
    the safe PRB range, regardless of which Manager variant produced it.
    """
    sig = float(1.0 / (1.0 + np.exp(-float(action_raw[0]))))
    b_rrm = B_RRM_MIN + (B_RRM_MAX - B_RRM_MIN) * sig
    return {"b_rrm": b_rrm}


class ManagerAgent:
    """Manager (high-level, 100 ms) — Gaussian PPO actor-critic with γ_H ≈ 0.904 (N1).

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
        """One PPO epoch sweep over a rollout buffer with γ_H = γ_MANAGER.

        P2 fix: accumulate metrics across all minibatches/epochs, return mean.
        P3 fix: cap effective epochs to min(k_epochs, max(1, n // 4)) to prevent
        overfitting when Manager gets very few transitions per rollout.
        """
        n = len(rewards)
        if n == 0:
            return {}
        k_epochs_eff = min(self.k_epochs, max(1, n // 4))
        advantages, returns = compute_gae(
            rewards, values, dones, last_value, self.gamma, self.gae_lambda
        )
        if advantages.size > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        oldlp_t = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        sum_actor_loss = sum_critic_loss = sum_entropy = sum_clip_frac = 0.0
        sum_approx_kl = 0.0
        n_mb = 0
        idx_all = np.arange(n)
        for _ in range(k_epochs_eff):
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
            "manager_actor_loss": sum_actor_loss / d,
            "manager_critic_loss": sum_critic_loss / d,
            "manager_entropy": sum_entropy / d,
            "manager_clip_fraction": sum_clip_frac / d,
            "manager_approx_kl": sum_approx_kl / d,
            "manager_explained_variance": ev,
            "manager_n_samples": n,
            "manager_k_epochs_eff": k_epochs_eff,
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


# ============================================================
# TD3 Manager (algorithm-matched Manager for TD3Solver)
# ============================================================


class TD3ManagerAgent:
    """TD3 Manager for the slow timescale (100 ms; b_rrm ∈ [B_RRM_MIN, B_RRM_MAX]).

    Uses the SAME TD3 algorithm as the Worker for absolute algorithm parity.
    Off-policy replay buffer; updates every Manager boundary once buffer ≥ warmup.
    Transition: (s_H, a_H_raw, r_H_acc, s_H_next, done) where r_H_acc is the
    SMDP-discounted intra-window return Σ γ_L^i · r_aug_i.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        hidden: tuple[int, ...] = (128, 64),
        lr_actor: float = 1e-5,
        lr_critic: float = 1e-4,
        gamma: float = GAMMA_MANAGER,
        tau: float = 0.005,
        policy_delay: int = 2,
        noise_std: float = 0.1,
        noise_clip: float = 0.3,
        buffer_capacity: int = 5_000,
        batch_size: int = 32,
        warmup_steps: int = 64,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        from agents.td3_agent import DeterministicActor, QCritic, ReplayBuffer

        torch.manual_seed(seed)
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.noise_std = noise_std
        self.noise_clip = noise_clip
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self._update_counter = 0

        act_low = np.array([-5.0] * action_dim, dtype=np.float32)
        act_high = np.array([5.0] * action_dim, dtype=np.float32)

        self.actor = DeterministicActor(state_dim, action_dim, act_low, act_high, hidden).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        for p in self.actor_target.parameters():
            p.requires_grad_(False)

        self.q1 = QCritic(state_dim, action_dim, hidden).to(self.device)
        self.q2 = QCritic(state_dim, action_dim, hidden).to(self.device)
        self.q1_target = copy.deepcopy(self.q1).to(self.device)
        self.q2_target = copy.deepcopy(self.q2).to(self.device)
        for p in list(self.q1_target.parameters()) + list(self.q2_target.parameters()):
            p.requires_grad_(False)

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.opt_critic = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr_critic
        )
        self.buffer = ReplayBuffer(buffer_capacity, state_dim, action_dim)

    @torch.no_grad()
    def act(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        """Return raw action in [-5, 5]; decode via decode_manager_action for b_rrm."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = self.actor(obs_t).squeeze(0).cpu().numpy().astype(np.float32)
        if explore:
            a += self.noise_std * np.random.randn(*a.shape).astype(np.float32)
            a = np.clip(a, -5.0, 5.0)
        return a

    def store(self, s_H: np.ndarray, a_H: np.ndarray, r_H: float,
              s_H_next: np.ndarray, done: bool) -> None:
        self.buffer.add(s_H, a_H, r_H, s_H_next, done)

    def update(self) -> dict[str, float]:
        if self.buffer.size < self.warmup_steps:
            return {}
        s, a, r, s_next, d = self.buffer.sample(self.batch_size)
        s = s.to(self.device); a = a.to(self.device)
        r = r.to(self.device); s_next = s_next.to(self.device); d = d.to(self.device)

        with torch.no_grad():
            noise = (torch.randn_like(a) * self.noise_std).clamp(-self.noise_clip, self.noise_clip)
            a_next = (self.actor_target(s_next) + noise).clamp(-5.0, 5.0)
            q1_t = self.q1_target(s_next, a_next)
            q2_t = self.q2_target(s_next, a_next)
            target = r + self.gamma * (1.0 - d) * torch.min(q1_t, q2_t)

        critic_loss = (
            torch.nn.functional.mse_loss(self.q1(s, a), target)
            + torch.nn.functional.mse_loss(self.q2(s, a), target)
        )
        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()

        self._update_counter += 1
        actor_loss_val = float("nan")
        if self._update_counter % self.policy_delay == 0:
            actor_loss = -self.q1(s, self.actor(s)).mean()
            self.opt_actor.zero_grad()
            actor_loss.backward()
            self.opt_actor.step()
            actor_loss_val = float(actor_loss.item())
            for p, pt in zip(self.actor.parameters(), self.actor_target.parameters()):
                pt.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)
        for p, pt in zip(self.q1.parameters(), self.q1_target.parameters()):
            pt.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)
        for p, pt in zip(self.q2.parameters(), self.q2_target.parameters()):
            pt.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)

        return {"mgr_critic_loss": float(critic_loss.item()), "mgr_actor_loss": actor_loss_val}

    def save(self, path: str) -> None:
        torch.save({
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor_target.load_state_dict(ckpt["actor"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.q1_target.load_state_dict(ckpt["q1"])
        self.q2_target.load_state_dict(ckpt["q2"])


# ============================================================
# SAC Manager (algorithm-matched Manager for SACSolver)
# ============================================================


class SACManagerAgent:
    """SAC Manager for the slow timescale (100 ms).

    Uses the SAME SAC algorithm as the Worker for absolute algorithm parity.
    Stochastic actor with automatic entropy temperature α_sac.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        hidden: tuple[int, ...] = (128, 64),
        lr_actor: float = 1e-5,
        lr_critic: float = 1e-4,
        lr_alpha: float = 3e-4,
        gamma: float = GAMMA_MANAGER,
        tau: float = 0.005,
        buffer_capacity: int = 5_000,
        batch_size: int = 32,
        warmup_steps: int = 64,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        from agents.sac_agent import GaussianTanhActor
        from agents.td3_agent import QCritic, ReplayBuffer

        torch.manual_seed(seed)
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps

        act_low = np.array([-5.0] * action_dim, dtype=np.float32)
        act_high = np.array([5.0] * action_dim, dtype=np.float32)

        self.actor = GaussianTanhActor(state_dim, action_dim, act_low, act_high, hidden).to(self.device)
        self.q1 = QCritic(state_dim, action_dim, hidden).to(self.device)
        self.q2 = QCritic(state_dim, action_dim, hidden).to(self.device)
        self.q1_target = copy.deepcopy(self.q1).to(self.device)
        self.q2_target = copy.deepcopy(self.q2).to(self.device)
        for p in list(self.q1_target.parameters()) + list(self.q2_target.parameters()):
            p.requires_grad_(False)

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.opt_critic = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr_critic
        )
        self.log_alpha_sac = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.opt_alpha = torch.optim.Adam([self.log_alpha_sac], lr=lr_alpha)
        self.target_entropy = float(-action_dim)
        self.buffer = ReplayBuffer(buffer_capacity, state_dim, action_dim)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> tuple[np.ndarray, float]:
        """Return (raw_action, log_prob); decode via decode_manager_action for b_rrm."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        a, lp = self.actor.sample(obs_t)
        return a.squeeze(0).cpu().numpy().astype(np.float32), float(lp.item())

    def store(self, s_H: np.ndarray, a_H: np.ndarray, r_H: float,
              s_H_next: np.ndarray, done: bool) -> None:
        self.buffer.add(s_H, a_H, r_H, s_H_next, done)

    def update(self) -> dict[str, float]:
        if self.buffer.size < self.warmup_steps:
            return {}
        s, a, r, s_next, d = self.buffer.sample(self.batch_size)
        s = s.to(self.device); a = a.to(self.device)
        r = r.to(self.device); s_next = s_next.to(self.device); d = d.to(self.device)

        alpha_sac = self.log_alpha_sac.exp().detach()

        with torch.no_grad():
            a_next, lp_next = self.actor.sample(s_next)
            q1_t = self.q1_target(s_next, a_next)
            q2_t = self.q2_target(s_next, a_next)
            target = r + self.gamma * (1.0 - d) * (torch.min(q1_t, q2_t) - alpha_sac * lp_next)

        critic_loss = (
            torch.nn.functional.mse_loss(self.q1(s, a), target)
            + torch.nn.functional.mse_loss(self.q2(s, a), target)
        )
        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()

        a_new, lp_new = self.actor.sample(s)
        q_new = torch.min(self.q1(s, a_new), self.q2(s, a_new))
        actor_loss = (alpha_sac * lp_new - q_new).mean()
        self.opt_actor.zero_grad()
        actor_loss.backward()
        self.opt_actor.step()

        alpha_loss = -(self.log_alpha_sac * (lp_new.detach() + self.target_entropy)).mean()
        self.opt_alpha.zero_grad()
        alpha_loss.backward()
        self.opt_alpha.step()

        for p, pt in zip(self.q1.parameters(), self.q1_target.parameters()):
            pt.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)
        for p, pt in zip(self.q2.parameters(), self.q2_target.parameters()):
            pt.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)

        return {
            "mgr_critic_loss": float(critic_loss.item()),
            "mgr_actor_loss": float(actor_loss.item()),
            "mgr_alpha_sac": float(alpha_sac.item()),
        }

    def save(self, path: str) -> None:
        torch.save({
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "log_alpha_sac": self.log_alpha_sac.item(),
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.q1_target.load_state_dict(ckpt["q1"])
        self.q2_target.load_state_dict(ckpt["q2"])
        if "log_alpha_sac" in ckpt:
            self.log_alpha_sac.data.fill_(float(ckpt["log_alpha_sac"]))
