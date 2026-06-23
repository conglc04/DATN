"""Reward normalization for numerically-stable PPO/TD3/SAC critics (audit 2026-06-22, fix D).

PROBLEM (part D of the smoke-train audit): the Lagrangian augmented reward
    r_aug = r_base - Σ_j λ_j · (c_j - d_j) / scale_j
has a structurally large penalty term, even after the delay/AoI caps (A+B+C)
bound c_vec. With scale_C1 = D_REF = 0.001 s, a capped 100 ms delay gives a
normalized deviation of ~99; times λ (up to LAMBDA_MAX=10) that is ~990 per
worker step, versus r_base ~ O(1-11). Summed/bootstrapped into the critic
target, raw returns reach O(1e4-1e7) and the critic MSE explodes (observed
1e8-1e20). This is a NUMERICAL conditioning problem, not a reward-design one.

FIX: scale the augmented reward by a running estimate of the standard deviation
of the discounted return (Engstrom et al. 2020 "Implementation Matters in Deep
Policy Gradients"; Stable-Baselines3 VecNormalize(norm_reward=True)). This is a
pure numerical rescale:
  - It scales reward AND penalty by the SAME running constant → preserves their
    relative weighting (does NOT touch the per-severity reward weights α_e, the
    C1-C5 thresholds, λ dynamics, LR, entropy coef, or PPO clip — none of the
    protected hyperparameters).
  - It does NOT change the optimal policy (a positive affine reward scaling
    leaves argmax over policies invariant).
  - It only keeps the critic target in a learnable range.

Used identically by all 3 sibling solvers (PPO train.py, TD3/SAC
train_offpolicy.py) so the fair-comparison property is preserved.
"""

from __future__ import annotations

import numpy as np


class RunningMeanStd:
    """Welford parallel running mean/variance (Chan et al. 1979).

    Tracks scalar (or vector) statistics online without storing samples.
    """

    def __init__(self, epsilon: float = 1e-4, shape: tuple[int, ...] = ()) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count: int) -> None:
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot
        self.var = m2 / tot
        self.count = tot

    def state_dict(self) -> dict:
        return {"mean": float(self.mean), "var": float(self.var), "count": float(self.count)}

    def load_state_dict(self, d: dict) -> None:
        self.mean = np.asarray(d["mean"], dtype=np.float64)
        self.var = np.asarray(d["var"], dtype=np.float64)
        self.count = float(d["count"])


class ReturnNormalizer:
    """SB3-style reward normalization by the running std of the discounted return.

    Maintain ret_t = gamma * ret_{t-1} + r_t; track Var(ret) online; emit
    r_t / sqrt(Var(ret) + eps). Call reset_episode() at each episode boundary so
    the discounted-return accumulator does not leak across episodes.

    NOTE: only the STD is used (not mean-centering) — centering rewards would
    change the effective discounting of the constant term; SB3 uses std-only.
    """

    def __init__(self, gamma: float, epsilon: float = 1e-8) -> None:
        self.ret_rms = RunningMeanStd(shape=())
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self._ret = 0.0

    def normalize(self, reward: float) -> float:
        """Update running return stats with this step's reward, return scaled reward."""
        self._ret = self._ret * self.gamma + float(reward)
        self.ret_rms.update(np.array([self._ret], dtype=np.float64))
        std = float(np.sqrt(self.ret_rms.var + self.epsilon))
        return float(reward) / std

    @property
    def std(self) -> float:
        return float(np.sqrt(self.ret_rms.var + self.epsilon))

    def reset_episode(self) -> None:
        self._ret = 0.0

    def state_dict(self) -> dict:
        return {"ret_rms": self.ret_rms.state_dict(), "ret": self._ret, "gamma": self.gamma}

    def load_state_dict(self, d: dict) -> None:
        self.ret_rms.load_state_dict(d["ret_rms"])
        self._ret = float(d.get("ret", 0.0))
        self.gamma = float(d.get("gamma", self.gamma))
