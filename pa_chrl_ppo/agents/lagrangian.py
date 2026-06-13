"""LambdaState — 5-dim Lagrangian multipliers + λ_warm[φ] phase table.

Standalone class independent of any RL algorithm (PPO/TD3/etc). Owns the dual
state evolution for PPO per docs/13_methodology_walkthrough.md:

    - Phase 2.3.3 Dual update rule (projected gradient ascent)
    - Phase 2.3.5 Subgradient interval-window Option b
    - Phase 3.2.6 λ_warm[φ] EMA refresh table
    - Phase 3.4.4 N9 phase transition handling (Fix Error 1: sync both global + local)

Key API:
    reset_episode(initial_phase)        — sync BOTH λ_global + λ_local from λ_warm[phase]
    on_manager_step_start(phi_now)      — phase transition check + sync
    accumulate(c_vec, d_phi)            — per-Worker-step interval-window add
    on_manager_step_end()               — dual ascent + reset win_c
    augmented_reward(r, c_vec, d_phi)   — r - Σ_j λ_local[j] · (c_j - d_j^φ_t)

Used by:
    - baselines/td3.py, baselines/sac.py (Phase 3 sibling solvers, W07/B7)
    - agents/worker_agent.py (PPO, W08+)
    - train.py Algorithm 1 main loop (W09)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from utils.config import (
    ALPHA_LAMBDA_DUAL,
    D_REF_URLLC,
    LAMBDA_MAX,
    LAMBDA_WARM,
    R_REF_EMBB_MBPS,
    WORKER_STEPS_PER_MANAGER,
)


N_HARD_CONSTRAINTS: int = 5            # C1-C5 (Phase 2.2.1)
DEFAULT_BETA_EMA: float = 0.05          # λ_warm slow EMA decay (Phase 3.2.6)
CONSTRAINT_DUAL_SCALES: np.ndarray = np.asarray(
    [D_REF_URLLC, 1.0, R_REF_EMBB_MBPS, 1.0, 1.0],
    dtype=np.float64,
)
# Audit Fix CF-4 (2026-05-28): scaling factors balance gradient magnitudes
# across heterogeneous units (seconds, Mbps, probabilities). PRELIMINARY
# empirical choice; no formal theoretical derivation. Future work: GradNorm
# (Chen 2018) or PCGrad (Yu 2020) adaptive normalization.
# See docs/13_methodology_walkthrough.md §2.3.3 disclaimer for details.


def _default_warm_table() -> dict[int, np.ndarray]:
    """Initial λ_warm table from utils.config.LAMBDA_WARM (Phase 3.2.6 initial values)."""
    return {phi: np.asarray(vals, dtype=np.float64).copy() for phi, vals in LAMBDA_WARM.items()}


@dataclass
class LambdaState:
    """Comprehensive Lagrangian state for PPO + flat baselines.

    Embeds Phase 3.4 critical fixes (docs/13):
      - Fix Error 1 (λ-Overwriting): phase transition syncs BOTH global + local
      - Fix Error 2 (interval-window): win_c reset sau mỗi Manager step,
        Option b (last W samples), NOT cumulative Option a

    Attributes:
        n_constraints     5 hard constraints (C1-C5)
        alpha_lambda       Dual learning rate (Phase 2.3.3 locked = 1e-4)
        beta_ema           λ_warm slow EMA decay
        worker_steps_per_manager  W = 10 (Phase 1.4 ratio)
        lambda_global      Current global λ (R^5)
        lambda_local       Worker-side λ snapshot (R^5)
        lambda_warm        Phase warm-start cache {1..5: R^5}
        win_c              Interval-window subgradient accumulator (R^5)
        win_steps          Worker steps elapsed in current Manager window
        phi_prev           Previously observed phase (for transition detection)
    """

    n_constraints: int = N_HARD_CONSTRAINTS
    alpha_lambda: float = ALPHA_LAMBDA_DUAL
    beta_ema: float = DEFAULT_BETA_EMA
    worker_steps_per_manager: int = WORKER_STEPS_PER_MANAGER

    # Dynamic state (set by reset_episode)
    lambda_global: np.ndarray = field(
        default_factory=lambda: np.zeros(N_HARD_CONSTRAINTS, dtype=np.float64)
    )
    lambda_local: np.ndarray = field(
        default_factory=lambda: np.zeros(N_HARD_CONSTRAINTS, dtype=np.float64)
    )
    lambda_warm: dict[int, np.ndarray] = field(default_factory=_default_warm_table)
    win_c: np.ndarray = field(
        default_factory=lambda: np.zeros(N_HARD_CONSTRAINTS, dtype=np.float64)
    )
    win_steps: int = 0
    phi_prev: int = 0  # 0 = uninitialized

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset_episode(self, initial_phase: int) -> None:
        """Sync BOTH λ_global AND λ_local from λ_warm[initial_phase] (Fix Error 1).

        Called at start of each episode. Cross-episode: λ_warm table carries over;
        λ_global is re-loaded from warm-start (no stale state from previous episode).
        """
        if initial_phase not in self.lambda_warm:
            raise ValueError(
                f"Phase {initial_phase} not in lambda_warm table (keys: {sorted(self.lambda_warm)})"
            )
        warm = self.lambda_warm[initial_phase].copy()
        self.lambda_global = warm
        self.lambda_local = warm.copy()
        self.win_c = np.zeros(self.n_constraints, dtype=np.float64)
        self.win_steps = 0
        self.phi_prev = initial_phase

    # ------------------------------------------------------------------
    # Manager step boundary (slow timescale)
    # ------------------------------------------------------------------

    def on_manager_step_start(self, phi_now: int) -> None:
        """Called at start of each Manager step. Handles phase transition.

        Per docs/13 Phase 3.4.4 N9 (Fix Error 1):
          1. EMA save λ_warm[phi_prev] ← (1-β_ema) · λ_warm[phi_prev] + β_ema · λ_global
          2. Sync BOTH λ_global ← λ_warm[phi_now] AND λ_local ← λ_global.copy()
          3. Reset win_c (window context changed)

        If phi_now == phi_prev, no-op (no transition).
        """
        if phi_now == self.phi_prev:
            return
        if phi_now not in self.lambda_warm:
            raise ValueError(
                f"Phase {phi_now} not in lambda_warm table (keys: {sorted(self.lambda_warm)})"
            )
        # EMA save current λ_global into λ_warm[phi_prev]
        self._ema_save_current_phase()
        # Sync BOTH global + local from new phase warm-start (Fix Error 1)
        new_warm = self.lambda_warm[phi_now].copy()
        self.lambda_global = new_warm
        self.lambda_local = new_warm.copy()
        # Reset interval-window
        self.win_c = np.zeros(self.n_constraints, dtype=np.float64)
        self.win_steps = 0
        self.phi_prev = phi_now

    def on_episode_end(self, final_phase: int | None = None) -> None:
        """Flush current lambda into the warm table at episode boundary.

        Phase transitions are checked only at Manager-step boundaries. The hard
        mission can enter the final phase inside the last Manager window; this
        hook preserves the learned lambda for the last active phase before the
        next reset_episode() reloads a phase warm-start.
        """
        if self.phi_prev in self.lambda_warm:
            self._ema_save_current_phase()
        if final_phase is not None and final_phase != self.phi_prev:
            if final_phase not in self.lambda_warm:
                raise ValueError(
                    f"Phase {final_phase} not in lambda_warm table (keys: {sorted(self.lambda_warm)})"
                )
            warm = self.lambda_warm[final_phase].copy()
            self.lambda_global = warm
            self.lambda_local = warm.copy()
            self.phi_prev = final_phase
        self.win_c = np.zeros(self.n_constraints, dtype=np.float64)
        self.win_steps = 0

    def _ema_save_current_phase(self) -> None:
        old_warm = self.lambda_warm[self.phi_prev]
        self.lambda_warm[self.phi_prev] = (
            (1.0 - self.beta_ema) * old_warm + self.beta_ema * self.lambda_global
        )

    def on_manager_step_end(self) -> dict[str, float]:
        """Dual ascent + reset win_c (Phase 2.3.3 + Fix Error 2).

        g_hat = win_c / win_steps     (mean interval-window deviation)
        λ_global ← max(0, λ_global + α_λ * g_hat)
        λ_local ← λ_global.copy()     (push to Worker)
        win_c, win_steps ← 0          (RESET — Option b interval-window)

        Returns diagnostic dict (subgradient + λ snapshot for logging).
        """
        if self.win_steps == 0:
            # No accumulation this window — skip dual update
            return {
                "subgradient_mean": 0.0,
                "lambda_global_mean": float(np.mean(self.lambda_global)),
            }
        g_hat = self.win_c / self.win_steps
        # Reviewer M4 (W06): bounded projection Π_Λ(·) — clip to [0, LAMBDA_MAX].
        # Prevents dual blow-up under sustained violations. Empirical λ ≤ 2.5
        # → LAMBDA_MAX=10 is soft safety net. See docs/13 §2.3.3.
        self.lambda_global = np.clip(
            self.lambda_global + self.alpha_lambda * g_hat,
            0.0,
            LAMBDA_MAX,
        )
        self.lambda_local = self.lambda_global.copy()
        out = {
            "subgradient_mean": float(np.mean(g_hat)),
            "lambda_global_mean": float(np.mean(self.lambda_global)),
        }
        # Reset interval-window (Option b — Phase 2.3.5)
        self.win_c = np.zeros(self.n_constraints, dtype=np.float64)
        self.win_steps = 0
        return out

    # ------------------------------------------------------------------
    # Per-Worker-step accumulator
    # ------------------------------------------------------------------

    def accumulate(self, c_vec: np.ndarray, d_phi: np.ndarray) -> None:
        """Add normalized per-step deviation (c_j - d_j^phi_t) / scale_j.

        Called every Worker step trong loop. c_vec + d_phi come from env.step()
        info dict (per Phase 2.2.1 Master Table lookup).
        """
        self.win_c += self._normalized_deviation(c_vec, d_phi)
        self.win_steps += 1

    def _normalized_deviation(self, c_vec: np.ndarray, d_phi: np.ndarray) -> np.ndarray:
        c = np.asarray(c_vec, dtype=np.float64)
        d = np.asarray(d_phi, dtype=np.float64)
        if c.shape != (self.n_constraints,):
            raise ValueError(f"c_vec shape {c.shape} != ({self.n_constraints},)")
        if d.shape != (self.n_constraints,):
            raise ValueError(f"d_phi shape {d.shape} != ({self.n_constraints},)")
        return (c - d) / CONSTRAINT_DUAL_SCALES

    # ------------------------------------------------------------------
    # Augmented reward (Phase 3.2.1)
    # ------------------------------------------------------------------

    def augmented_reward(
        self,
        reward: float,
        c_vec: np.ndarray,
        d_phi: np.ndarray,
    ) -> float:
        """Phase 3.2.1 augmented reward (pure Lagrangian, NO QP distillation here).

        r_aug = r - Σ_j λ_local[j] · (c_j - d_j^φ_t)

        Signed deviation (NOT max(0, ...)). QP penalty term is in actor loss
        (Phase 3.2.1 Fix Error 1 from earlier session).
        """
        penalty = float(np.dot(self.lambda_local, self._normalized_deviation(c_vec, d_phi)))
        return float(reward) - penalty

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_lambda_warm_table_snapshot(self) -> dict[int, np.ndarray]:
        """Deep copy of λ_warm table for logging (does not mutate)."""
        return {phi: arr.copy() for phi, arr in self.lambda_warm.items()}

    def get_lambda_global(self) -> np.ndarray:
        """Read-only view of current λ_global."""
        return self.lambda_global.copy()

    def get_lambda_local(self) -> np.ndarray:
        """Read-only view of current λ_local (Worker-side)."""
        return self.lambda_local.copy()

    def __repr__(self) -> str:
        return (
            f"LambdaState(n={self.n_constraints}, α_λ={self.alpha_lambda:.0e}, "
            f"β_ema={self.beta_ema}, W={self.worker_steps_per_manager}, "
            f"phi_prev={self.phi_prev}, win_steps={self.win_steps}, "
            f"λ_global={self.lambda_global.round(3).tolist()})"
        )
