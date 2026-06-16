"""LambdaState — K-aware (4K+1)-dim Lagrangian multipliers + λ_warm[severity_per_amb] table.

Standalone class independent of any RL algorithm (PPO/TD3/etc). Owns the dual
state evolution for PPO per docs/13_methodology_walkthrough.md:

    - Phase 2.3.3 Dual update rule (projected gradient ascent)
    - Phase 2.3.5 Subgradient interval-window Option b
    - Phase 3.2.6 λ_warm[severity] EMA refresh table
    - Phase 3.4.4 N9 context-sync handling (Fix Error 1: sync both global + local)

Per-ambulance severity_k epic (2026-06-15): each of the K ambulances carries an
independent severity_k in {1..5}, sampled independently and fixed for the
episode. The Lagrangian vectors are (4K+1)-dim:

    [C1_0.C1_{K-1}, C2_0..C2_{K-1}, C4_0..C4_{K-1}, C5_0..C5_{K-1}, C3_shared]

At K=1 this is the permutation [0,1,3,4,2] of the legacy 5-dim
[C1,C2,C3,C4,C5] order — exact numeric preservation of all prior behavior.

The warm-start table is keyed by the per-ambulance severity tuple
``severity_per_amb`` (exogenous, fixed per episode). Because severity does not
change within an episode, on_manager_step_start() is normally a no-op.

Key API:
    reset_episode(severity_per_amb, severity_ref)
        — sync BOTH λ_global + λ_local from λ_warm[severity_per_amb]
    on_manager_step_start(severity_per_amb, severity_ref)
        — severity-change check + sync (no-op if unchanged)
    accumulate(c_vec, d_phi)            — per-Worker-step interval-window add
    on_manager_step_end()               — dual ascent + reset win_c
    augmented_reward(r, c_vec, d_phi)   — r - Σ_j λ_local[j] · (c_j - d_j^sev)
    on_episode_end(severity_per_amb, severity_ref)
        — EMA-flush λ_warm at episode boundary

Used by:
    - solvers/td3.py, solvers/sac.py (Phase 3 sibling solvers, W07/B7)
    - agents/worker_agent.py (PPO, W08+)
    - train.py Algorithm 1 main loop (W09)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from utils.config import (
    ALPHA_LAMBDA_DUAL,
    AOI_REF_S,
    D_REF_URLLC,
    LAMBDA_MAX,
    R_REF_EMBB_MBPS,
    WORKER_STEPS_PER_MANAGER,
    build_dual_scales,
    build_lambda_warm_vector,
)

# Legacy 5-dim constants (C1-C5), retained for backward compat with standalone
# CMDPLagrangian(n=5, ...) usages (solvers/no_phase_ppo.py, ppo_cmdp_flat.py)
# and any pre-existing 5-dim tests.
N_HARD_CONSTRAINTS: int = 5
DEFAULT_BETA_EMA: float = 0.05          # λ_warm slow EMA decay (Phase 3.2.6)
CONSTRAINT_DUAL_SCALES: np.ndarray = np.asarray(
    [D_REF_URLLC, 1.0, R_REF_EMBB_MBPS, AOI_REF_S, 1.0],
    dtype=np.float64,
)


@dataclass
class LambdaState:
    """K-aware (4K+1)-dim Lagrangian state for PPO + flat solvers.

    Embeds Phase 3.4 critical fixes (docs/13):
      - Fix Error 1 (λ-Overwriting): severity sync sets BOTH global + local
      - Fix Error 2 (interval-window): win_c reset sau mỗi Manager step,
        Option b (last W samples), NOT cumulative Option a

    Attributes:
        K                  Number of ambulances (n_constraints = 4K+1)
        alpha_lambda       Dual learning rate (Phase 2.3.3 locked = 1e-4)
        beta_ema           λ_warm slow EMA decay
        worker_steps_per_manager  W = 10 (Phase 1.4 ratio)
        force_zero_warm    Exp3 ablation: always warm-start from zero and
                            never EMA-write the λ_warm table (disable_warm_start)
        lambda_global      Current global λ (R^(4K+1))
        lambda_local       Worker-side λ snapshot (R^(4K+1))
        lambda_warm        Severity-tuple warm-start cache {severity_per_amb: R^(4K+1)}
        win_c              Interval-window subgradient accumulator (R^(4K+1))
        win_steps          Worker steps elapsed in current Manager window
        sev_prev           Previously observed severity_per_amb tuple (None = uninitialized)
        sev_ref_prev       Previously observed severity_ref (= max(severity_per_amb))
    """

    K: int = 1
    alpha_lambda: float = ALPHA_LAMBDA_DUAL
    beta_ema: float = DEFAULT_BETA_EMA
    worker_steps_per_manager: int = WORKER_STEPS_PER_MANAGER
    force_zero_warm: bool = False

    # Dynamic state (set by reset_episode); None defaults are resized to
    # (4K+1,) zeros in __post_init__.
    lambda_global: np.ndarray | None = None
    lambda_local: np.ndarray | None = None
    lambda_warm: dict[tuple[int, ...], np.ndarray] = field(default_factory=dict)
    win_c: np.ndarray | None = None
    win_steps: int = 0
    sev_prev: tuple[int, ...] | None = None
    sev_ref_prev: int = 0

    def __post_init__(self) -> None:
        self.n_constraints = 4 * self.K + 1
        self.dual_scales = build_dual_scales(self.K)
        if self.lambda_global is None:
            self.lambda_global = np.zeros(self.n_constraints, dtype=np.float64)
        if self.lambda_local is None:
            self.lambda_local = np.zeros(self.n_constraints, dtype=np.float64)
        if self.win_c is None:
            self.win_c = np.zeros(self.n_constraints, dtype=np.float64)

    # ------------------------------------------------------------------
    # Warm-start table lookups
    # ------------------------------------------------------------------

    def _warm_for(self, severity_per_amb: Sequence[int], severity_ref: int) -> np.ndarray:
        if self.force_zero_warm:
            return np.zeros(self.n_constraints, dtype=np.float64)
        sev_key = tuple(severity_per_amb)
        warm = self.lambda_warm.get(sev_key)
        if warm is None:
            warm = build_lambda_warm_vector(severity_per_amb, severity_ref)
        return warm.copy()

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset_episode(self, severity_per_amb: Sequence[int], severity_ref: int) -> None:
        """Sync BOTH λ_global AND λ_local from λ_warm[severity_per_amb] (Fix Error 1).

        Called at start of each episode. Cross-episode: λ_warm table carries over;
        λ_global is re-loaded from warm-start (no stale state from previous episode).
        """
        warm = self._warm_for(severity_per_amb, severity_ref)
        self.lambda_global = warm
        self.lambda_local = warm.copy()
        self.win_c = np.zeros(self.n_constraints, dtype=np.float64)
        self.win_steps = 0
        self.sev_prev = tuple(severity_per_amb)
        self.sev_ref_prev = severity_ref

    # ------------------------------------------------------------------
    # Manager step boundary (slow timescale)
    # ------------------------------------------------------------------

    def on_manager_step_start(self, severity_per_amb: Sequence[int], severity_ref: int) -> None:
        """Called at start of each Manager step. Handles a severity change.

        Per docs/13 Phase 3.4.4 N9 (Fix Error 1):
          1. EMA save λ_warm[sev_prev] ← (1-β_ema) · λ_warm[sev_prev] + β_ema · λ_global
          2. Sync BOTH λ_global ← λ_warm[severity_now] AND λ_local ← λ_global.copy()
          3. Reset win_c (window context changed)

        If severity_per_amb == sev_prev, no-op. Severity is fixed per episode,
        so this is normally a no-op; the path is kept for manual overrides.
        """
        sev_key = tuple(severity_per_amb)
        if sev_key == self.sev_prev and severity_ref == self.sev_ref_prev:
            return
        self._ema_save_current_severity()
        warm = self._warm_for(severity_per_amb, severity_ref)
        self.lambda_global = warm
        self.lambda_local = warm.copy()
        self.win_c = np.zeros(self.n_constraints, dtype=np.float64)
        self.win_steps = 0
        self.sev_prev = sev_key
        self.sev_ref_prev = severity_ref

    def on_episode_end(
        self,
        severity_per_amb: Sequence[int] | None = None,
        severity_ref: int | None = None,
    ) -> None:
        """Flush current lambda into the warm table at episode boundary.

        EMA-saves the learned λ for the current severity back into λ_warm[sev]
        so the next episode at the same severity warm-starts from it.
        severity_per_amb/severity_ref are normally the same as at reset_episode
        (severity is fixed per episode); kept for generality (manual overrides).
        """
        self._ema_save_current_severity()
        if severity_per_amb is not None:
            sev_key = tuple(severity_per_amb)
            sev_ref = severity_ref if severity_ref is not None else self.sev_ref_prev
            if sev_key != self.sev_prev or sev_ref != self.sev_ref_prev:
                warm = self._warm_for(severity_per_amb, sev_ref)
                self.lambda_global = warm
                self.lambda_local = warm.copy()
                self.sev_prev = sev_key
                self.sev_ref_prev = sev_ref
        self.win_c = np.zeros(self.n_constraints, dtype=np.float64)
        self.win_steps = 0

    def _ema_save_current_severity(self) -> None:
        if self.force_zero_warm or self.sev_prev is None:
            return
        old_warm = self.lambda_warm.get(self.sev_prev)
        if old_warm is None:
            old_warm = build_lambda_warm_vector(self.sev_prev, self.sev_ref_prev)
        self.lambda_warm[self.sev_prev] = (
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
        info dict (per Phase 2.2.1 Master Table lookup), both (4K+1,).
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
        return (c - d) / self.dual_scales

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

    def get_lambda_warm_table_snapshot(self) -> dict[tuple[int, ...], np.ndarray]:
        """Deep copy of λ_warm table for logging (does not mutate)."""
        return {sev: arr.copy() for sev, arr in self.lambda_warm.items()}

    def get_lambda_global(self) -> np.ndarray:
        """Read-only view of current λ_global."""
        return self.lambda_global.copy()

    def get_lambda_local(self) -> np.ndarray:
        """Read-only view of current λ_local (Worker-side)."""
        return self.lambda_local.copy()

    def __repr__(self) -> str:
        return (
            f"LambdaState(K={self.K}, n={self.n_constraints}, α_λ={self.alpha_lambda:.0e}, "
            f"β_ema={self.beta_ema}, W={self.worker_steps_per_manager}, "
            f"sev_prev={self.sev_prev}, win_steps={self.win_steps}, "
            f"λ_global={self.lambda_global.round(3).tolist()})"
        )
