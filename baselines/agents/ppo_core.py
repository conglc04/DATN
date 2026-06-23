"""PPO core utilities — pure functions reusable by Manager + Worker agents.

Phase 3.3 / 3.4: shared building blocks for hierarchical PPO. Manager and Worker
construct their own actor/critic + optimizer but call these primitives during
update steps.

Reference:
    - docs/13_methodology_walkthrough.md Phase 3.4.1 Algorithm 1 (GAE + clipped PPO)
    - docs/13_methodology_walkthrough.md Phase 3.4.4 N1 (γ_H = γ_L^W, distinct)
    - Schulman et al. 2017, "Proximal Policy Optimization Algorithms"
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalized Advantage Estimation (Schulman 2016).

    Computes per-timestep advantages and discounted returns from a single
    rollout. Bootstraps with ``last_value`` for the terminal step.

    Args:
        rewards: shape (T,)
        values: shape (T,) — V(s_t) under current critic
        dones: shape (T,) — 1 if terminal at t, else 0
        last_value: V(s_T) bootstrap (0 if terminal)
        gamma: discount (γ_L=0.99 for Worker, γ_H≈0.904 for Manager — see N1)
        lam: GAE λ (default 0.95)

    Returns:
        (advantages, returns) both shape (T,)
    """
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    gae = 0.0
    next_v = float(last_value)
    for t in reversed(range(n)):
        nonterm = 1.0 - float(dones[t])
        delta = float(rewards[t]) + gamma * next_v * nonterm - float(values[t])
        gae = delta + gamma * lam * nonterm * gae
        advantages[t] = gae
        next_v = float(values[t])
    returns = advantages + values.astype(np.float32)
    return advantages, returns


def ppo_clip_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clipped surrogate objective L^CLIP (Schulman 2017 eq. 7).

    Returns:
        (loss, clip_fraction) — loss is the **negative** mean (to minimize),
        clip_fraction is the proportion of ratios outside [1-ε, 1+ε].
    """
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    loss = -torch.min(surr1, surr2).mean()
    with torch.no_grad():
        clipped = (ratio - 1.0).abs() > clip_eps
        clip_fraction = clipped.float().mean()
    return loss, clip_fraction


def value_loss(
    values: torch.Tensor,
    returns: torch.Tensor,
    vf_coef: float = 0.5,
) -> torch.Tensor:
    """Scaled MSE critic loss (vf_coef · ||V(s) - R||²)."""
    return vf_coef * F.mse_loss(values, returns)


def entropy_bonus(dist) -> torch.Tensor:
    """Mean entropy across batch (sum across action dims, mean across batch).

    For continuous Normal: sum-over-action-dim then mean-over-batch.
    """
    return dist.entropy().sum(-1).mean()


def approx_kl(old_log_probs: torch.Tensor, new_log_probs: torch.Tensor) -> torch.Tensor:
    """Approximate KL(π_old || π_new) ≈ mean(old_lp - new_lp) (Schulman 2020 blog)."""
    return (old_log_probs - new_log_probs).mean()


def explained_variance(values: np.ndarray, returns: np.ndarray) -> float:
    """EV = 1 - Var(returns - values) / Var(returns). 1.0 = perfect critic."""
    var_ret = np.var(returns)
    if var_ret < 1e-8:
        return 0.0
    return float(1.0 - np.var(returns - values) / var_ret)
