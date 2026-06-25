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
    augmented_reward(r, c_vec, d_phi)   — r - Σ_j λ_local[j] · max(0, c_j - d_j^sev)
    on_episode_end(severity_per_amb, severity_ref)
        — EMA-flush λ_warm at episode boundary

Constraint taxonomy (CMDP formulation — load-bearing definitions):
    C1  E[D_e2e^k] ≤ D_max^{sev_k}         MEAN    per-amb   delay
    C2  P(D > D_max) ≤ ε^{sev_k}            CHANCE  per-amb   delay-tail
    C3  E[R_eMBB] ≥ 10 Mbps                 MEAN    shared    eMBB floor
    C4  E[AoI_k] ≤ AoI_max^{sev_k}          MEAN    per-amb   AoI
    C5  P(AoI > AoI_max) ≤ ε_AoI^{sev_k}   CHANCE  per-amb   AoI-tail
    C3 is a MEAN-throughput constraint (NOT a chance constraint).

Dual-ascent estimator — hybrid Option-a/Option-b (audit 2026-06-21):
    C1 (delay mean), C4 (AoI mean), C3 (eMBB floor) use the Option-b
    interval-window (win_c/win_steps, reset every Manager step, N≈200
    MAC-tick samples). N=200 is adequate for MEAN-type constraints.

    C2 (delay tail), C5 (AoI tail) are CHANCE constraints
    Pr[violation] <= eps with eps as low as 1e-5 (severity 4-5). A
    Bernoulli rate is only resolvable to within ~1/N: N=200 cannot even
    register a single occurrence of a 1e-5 event (it would take ~500
    Manager-step windows on average to see ONE violation), so the
    Option-b gradient for C2/C5 is almost always a frozen "all-satisfied"
    signal that occasionally spikes 100-500x when a rare violation lands
    inside the 200-sample window — a high-variance, mostly-uninformative
    estimator at the target eps scale.
    Fix: C2/C5 use a separate EPISODE-CUMULATIVE estimator (cum_c/
    cum_steps, Option a — never reset mid-episode, only at episode
    boundaries) so N grows with elapsed episode time (N≈200 at the first
    Manager step, N≈200,000+ by mid-episode), which is the textbook
    consistent estimator for a fixed Bernoulli rate and is the only way
    to resolve probabilities at the 1e-4/1e-5 scale within an episode.
    Severity (hence eps) is fixed per episode, so an episode-cumulative
    mean never mixes samples from different constraint regimes.

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
    LAMBDA_MAX,
    WORKER_STEPS_PER_MANAGER,
    build_dual_scales,
    build_lambda_warm_vector,
)

# Legacy 5-dim constant (C1-C5), retained for backward compat with
# tests/test_lagrangian.py only (NOT used by ablation variants
# solvers/pa_ppo_soft.py/ppo_cmdp_flat.py — those use their own
# unnormalized solvers._common.CMDPLagrangian, not this module's scales).
N_HARD_CONSTRAINTS: int = 5
# λ-persistence (audit 2026-06-23): β_ema=1.0 = FULL persistence. λ_warm[sev]
# is the PERSISTENT per-severity dual variable — at episode end the full learned
# λ_global is saved (NOT a 5% EMA blend), and the next same-severity episode
# warm-starts from it. The LAMBDA_WARM constant table is the ONE-TIME initial
# value (first time a severity is seen per run). This fixes the starvation root
# cause: β_ema=0.05 diluted accumulation 20× → λ_C2 pinned at λ_warm≈2.2 < the
# equilibrium λ*≈4.0 needed to offset the eMBB reward gain → Manager starved
# URLLC. With full persistence λ accumulates monotonically toward equilibrium
# across episodes (reset only at a new run = new LambdaState).
DEFAULT_BETA_EMA: float = 1.0           # full persistence (was 0.05 slow EMA)


@dataclass
class LambdaState:
    """K-aware (4K+1)-dim Lagrangian state for all 3 sibling solvers (PPO/TD3/SAC).

    Embeds Phase 3.4 critical fixes (docs/13):
      - Fix Error 1 (λ-Overwriting): severity sync sets BOTH global + local
      - Fix Error 2 (interval-window): win_c reset sau mỗi Manager step,
        Option b (last W samples), NOT cumulative Option a — used for
        C1/C4/C3 (mean-type constraints; N≈200 is adequate)
      - Tail-estimator fix (audit 2026-06-21): cum_c/cum_steps, Option a
        (episode-cumulative, never reset mid-episode) — used for C2/C5
        (chance constraints with eps down to 1e-5; N≈200 cannot resolve
        that scale, see module docstring)

    Attributes:
        K                  Number of ambulances (n_constraints = 4K+1)
        alpha_lambda       Dual learning rate (2e-4; config.ALPHA_LAMBDA_DUAL SSOT —
                            A/B 5e-4 reverted to 2e-4 on 2026-06-22
        beta_ema           λ_warm persistence blend (1.0 = full persistence;
                            λ_warm[sev] = persistent per-severity dual variable)
        worker_steps_per_manager  W = 10 (Phase 1.4 ratio)
        force_zero_warm    Exp3 ablation: always warm-start from zero and
                            never EMA-write the λ_warm table (disable_warm_start)
        lambda_global      Current global λ (R^(4K+1))
        lambda_local       Worker-side λ snapshot (R^(4K+1))
        lambda_warm        Severity-tuple warm-start cache {severity_per_amb: R^(4K+1)}
        win_c              Interval-window subgradient accumulator (R^(4K+1)) — C1/C4/C3
        win_steps          Worker steps elapsed in current Manager window
        cum_c              Episode-cumulative subgradient accumulator (R^(4K+1)) — C2/C5
        cum_steps          Worker steps elapsed since episode start (never reset
                            mid-episode; only at reset_episode/on_episode_end)
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
    cum_c: np.ndarray | None = None
    cum_steps: int = 0
    sev_prev: tuple[int, ...] | None = None
    sev_ref_prev: int = 0
    # Most recent window deviation g_hat (the SAME vector dual ascent consumed in
    # on_manager_step_end). Exposed to the Manager via get_deviation_hat() so the
    # high-level policy observes the CURRENT constraint residual g_j — distinct
    # from λ_global (the long-run integral of past g). The Manager's augmented
    # reward r_aug = r − Σ λ_j·max(0, g_j) depends on g directly, so without g in
    # the state the Manager critic faces partial observability (V(s) cannot be fit
    # → negative explained variance). Audit 2026-06-23. Read-only for the Manager.
    last_g_hat: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.n_constraints = 4 * self.K + 1
        # Initial dual_scales without severity (fallback fixed-scale); rebuilt
        # per-episode in reset_episode() with the actual severity_per_amb so
        # C1/C4 scales match each ambulance's D_max/AoI_max threshold.
        self.dual_scales = build_dual_scales(self.K)
        if self.lambda_global is None:
            self.lambda_global = np.zeros(self.n_constraints, dtype=np.float64)
        if self.lambda_local is None:
            self.lambda_local = np.zeros(self.n_constraints, dtype=np.float64)
        if self.win_c is None:
            self.win_c = np.zeros(self.n_constraints, dtype=np.float64)
        if self.cum_c is None:
            self.cum_c = np.zeros(self.n_constraints, dtype=np.float64)
        if self.last_g_hat is None:
            self.last_g_hat = np.zeros(self.n_constraints, dtype=np.float64)
        # C2 (delay tail) at indices [K, 2K), C5 (AoI tail) at [3K, 4K).
        self._tail_mask = np.zeros(self.n_constraints, dtype=bool)
        self._tail_mask[self.K:2 * self.K] = True
        self._tail_mask[3 * self.K:4 * self.K] = True

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
        """Load λ_global/λ_local from the PERSISTENT λ_warm[severity_per_amb].

        Called at start of each episode. This is a CONTINUE, not a reset-to-zero:
        λ_warm[sev] persists across episodes (full persistence, β_ema=1.0), so a
        same-severity episode warm-starts from the accumulated dual, letting λ
        climb monotonically toward the CMDP equilibrium across episodes. The
        LAMBDA_WARM constant only seeds λ_warm[sev] the first time that severity
        is seen in the run. The within-episode estimators (win_c/cum_c) DO reset
        each episode — they measure this episode's window/tail, and severity (eps
        regime) can differ between episodes. λ persistence is reset only by
        constructing a new LambdaState (= new training run/seed).
        """
        # Rebuild dual_scales for this episode's severity_per_amb (ĐX2 audit
        # 2026-06-24): C1_k scale = D_max^{sev_k}, C4_k = AoI_max^{sev_k}.
        self.dual_scales = build_dual_scales(self.K, severity_per_amb)
        warm = self._warm_for(severity_per_amb, severity_ref)
        self.lambda_global = warm
        self.lambda_local = warm.copy()
        self.win_c = np.zeros(self.n_constraints, dtype=np.float64)
        self.win_steps = 0
        self.cum_c = np.zeros(self.n_constraints, dtype=np.float64)
        self.cum_steps = 0
        # No window measured yet this episode → Manager observes a zero residual on
        # its first action (correct: no QoS sample has arrived).
        self.last_g_hat = np.zeros(self.n_constraints, dtype=np.float64)
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
        # Rebuild dual_scales for new severity (ĐX2 — mirrors reset_episode).
        self.dual_scales = build_dual_scales(self.K, severity_per_amb)
        warm = self._warm_for(severity_per_amb, severity_ref)
        self.lambda_global = warm
        self.lambda_local = warm.copy()
        self.win_c = np.zeros(self.n_constraints, dtype=np.float64)
        self.win_steps = 0
        # Severity changed → eps regime changed → cumulative C2/C5 estimator
        # from the old severity is no longer valid; reset it too.
        self.cum_c = np.zeros(self.n_constraints, dtype=np.float64)
        self.cum_steps = 0
        self.last_g_hat = np.zeros(self.n_constraints, dtype=np.float64)
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
        self.cum_c = np.zeros(self.n_constraints, dtype=np.float64)
        self.cum_steps = 0

    def _ema_save_current_severity(self) -> None:
        """Persist the current severity's learned λ_global into λ_warm[sev].

        With β_ema=1.0 (default, full persistence) this is λ_warm[sev] = λ_global
        — the per-severity dual variable carries forward to the next same-severity
        episode. β_ema<1 (legacy/ablation) blends with the prior λ_warm.
        """
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

        Hybrid estimator (audit 2026-06-21 — see module docstring):
          C1/C4/C3 (mean constraints): g_hat = win_c/win_steps, Option-b
              interval-window, N≈200 — adequate, reset every Manager step.
          C2/C5 (tail/chance constraints, eps down to 1e-5): g_hat =
              cum_c/cum_steps, Option-a episode-cumulative — N grows with
              elapsed episode time; N≈200 cannot resolve a 1e-5 rate.

        λ_global ← max(0, λ_global + α_λ * g_hat)
        λ_local ← λ_global.copy()     (push to Worker)
        win_c, win_steps ← 0          (RESET — Option b interval-window)
        cum_c, cum_steps ← unchanged  (NOT reset — Option a, episode-cumulative)

        Returns diagnostic dict (subgradient + λ snapshot for logging).
        """
        if self.win_steps == 0:
            # No accumulation this window — skip dual update
            return {
                "subgradient_mean": 0.0,
                "lambda_global_mean": float(np.mean(self.lambda_global)),
            }
        g_hat_mean = self.win_c / self.win_steps
        g_hat_tail = self.cum_c / self.cum_steps if self.cum_steps > 0 else g_hat_mean
        g_hat = np.where(self._tail_mask, g_hat_tail, g_hat_mean)
        # Cache the residual the Manager will observe at the next decision (the SAME
        # vector dual ascent uses below). Stored BEFORE the win_c reset so it
        # survives the Option-b window reset. See get_deviation_hat().
        self.last_g_hat = g_hat.copy()
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
        # Reset interval-window (Option b — Phase 2.3.5). cum_c/cum_steps
        # (Option a, C2/C5) are intentionally NOT reset here.
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

        Feeds BOTH estimators: win_c (Option-b, reset every Manager step,
        used for C1/C4/C3) and cum_c (Option-a, episode-cumulative, used
        for C2/C5 — see module docstring for why tail constraints need it).
        """
        dev = self._normalized_deviation(c_vec, d_phi)
        self.win_c += dev
        self.win_steps += 1
        self.cum_c += dev
        self.cum_steps += 1

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
        """Phase 3.2.1 augmented reward — hinge penalty (one-sided).

        r_aug = r - Σ_j λ_local[j] · max(0, c_j - d_j^φ_t)

        Hinge, NOT signed deviation. Fix (2026-06-22, bonus-masking audit):
        signed deviation let slack mean-constraints (C1/C4 — mean delay/AoI
        usually far under threshold) contribute a large NEGATIVE penalty
        (= reward bonus) that swamped the much smaller, correctly-signed
        tail-constraint (C2/C5) penalties (measured 131x at severity 5 via
        penalty_breakdown). Hinge means a slack constraint contributes
        exactly 0 — no bonus, only real violations are penalized. The
        dual-ascent subgradient estimator (accumulate/on_manager_step_end,
        via _normalized_deviation) is UNCHANGED — it must stay signed so λ
        can still descend when a constraint is slack.
        """
        dev = self._normalized_deviation(c_vec, d_phi)
        penalty = float(np.dot(self.lambda_local, np.maximum(0.0, dev)))
        return float(reward) - penalty

    def penalty_breakdown(self, c_vec: np.ndarray, d_phi: np.ndarray) -> np.ndarray:
        """Per-constraint hinge penalty vector  λ_local[j]·max(0, (c_j−d_j)/scale_j)  (4K+1,).

        Element-wise terms whose sum is exactly the scalar penalty subtracted in
        augmented_reward (so Σ breakdown == r_base − r_aug). All entries are
        >= 0: a slack constraint (c < d) contributes exactly 0 (no bonus); a
        violated constraint (c > d) contributes its positive penalty. Does NOT
        mutate state.
        """
        dev = self._normalized_deviation(c_vec, d_phi)
        return self.lambda_local * np.maximum(0.0, dev)

    # ------------------------------------------------------------------
    # Checkpoint support (P5 fix)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Serialize full LambdaState for checkpoint."""
        warm_serial = {str(k): v.tolist() for k, v in self.lambda_warm.items()}
        return {
            "lambda_global": self.lambda_global.tolist(),
            "lambda_local": self.lambda_local.tolist(),
            "lambda_warm": warm_serial,
            "win_c": self.win_c.tolist(),
            "win_steps": self.win_steps,
            "cum_c": self.cum_c.tolist(),
            "cum_steps": self.cum_steps,
            "last_g_hat": self.last_g_hat.tolist(),
            "sev_prev": list(self.sev_prev) if self.sev_prev is not None else None,
            "sev_ref_prev": self.sev_ref_prev,
        }

    def load_state_dict(self, d: dict) -> None:
        """Restore LambdaState from checkpoint."""
        self.lambda_global = np.array(d["lambda_global"], dtype=np.float64)
        self.lambda_local = np.array(d["lambda_local"], dtype=np.float64)
        self.win_c = np.array(d["win_c"], dtype=np.float64)
        self.win_steps = d["win_steps"]
        self.cum_c = np.array(
            d.get("cum_c", np.zeros(self.n_constraints).tolist()), dtype=np.float64
        )
        self.cum_steps = d.get("cum_steps", 0)
        self.last_g_hat = np.array(
            d.get("last_g_hat", np.zeros(self.n_constraints).tolist()), dtype=np.float64
        )
        sp = d.get("sev_prev")
        self.sev_prev = tuple(sp) if sp is not None else None
        self.sev_ref_prev = d.get("sev_ref_prev", 0)
        warm_raw = d.get("lambda_warm", {})
        self.lambda_warm = {}
        for k_str, v in warm_raw.items():
            sev_key = tuple(int(x) for x in k_str.strip("()[] ").split(",") if x.strip())
            self.lambda_warm[sev_key] = np.array(v, dtype=np.float64)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_lambda_warm_table_snapshot(self) -> dict[tuple[int, ...], np.ndarray]:
        """Deep copy of λ_warm table for logging (does not mutate)."""
        return {sev: arr.copy() for sev, arr in self.lambda_warm.items()}

    def get_lambda_global(self) -> np.ndarray:
        """Read-only view of current λ_global."""
        return self.lambda_global.copy()

    def get_deviation_hat(self) -> np.ndarray:
        """Read-only view of the most recent window deviation g_hat = ĝ_{t-1} (4K+1).

        The residual of the LAST COMPLETED Manager window (C1/C3/C4 = Option-b
        window mean, C2/C5 = Option-a episode-cumulative tail) — the SAME vector
        dual ascent consumed at the last on_manager_step_end. Read BEFORE the
        Manager picks b_rrm_t, so it reflects b_rrm_{t-1}, never the not-yet-run
        window (no future-information leak). Exposed alongside λ_global (the
        long-run price) so the Manager is not blind to the per-window residual.

        Same-SOURCE proxy, NOT a literal equality with the reward's penalty:
        this is the SIGNED window MEAN of (c_j-d_j)/scale_j (via accumulate());
        augmented_reward() instead applies max(0,·) (hinge) to that SAME
        per-tick deviation, individually per Worker-step, before the SMDP
        discounted sum the Manager's critic actually learns from. Hinge is
        convex, so mean(max(0,dev)) >= max(0,mean(dev)) (Jensen) — the true
        accumulated penalty is >= what hinging g_hat after the fact would give.
        g_hat answers "which constraints were under pressure, in which
        direction" (same c_vec/d_phi/severity source as the reward), not
        "exactly how much reward was subtracted this window".

        Zero before any window has been measured (episode start), and reset on
        a severity change (on_manager_step_start) so it never carries a
        residual measured under a different severity's threshold regime. Does
        NOT mutate state.
        """
        return self.last_g_hat.copy()

    def get_lambda_local(self) -> np.ndarray:
        """Read-only view of current λ_local (Worker-side)."""
        return self.lambda_local.copy()

    def __repr__(self) -> str:
        return (
            f"LambdaState(K={self.K}, n={self.n_constraints}, α_λ={self.alpha_lambda:.0e}, "
            f"β_ema={self.beta_ema}, W={self.worker_steps_per_manager}, "
            f"sev_prev={self.sev_prev}, win_steps={self.win_steps}, cum_steps={self.cum_steps}, "
            f"λ_global={self.lambda_global.round(3).tolist()})"
        )
