"""Shared infrastructure for solvers.

Provides:
    - BaselineFlags: configuration toggles (severity visibility, CMDP, HRL)
    - CMDPLagrangian: dual-ascent multipliers for constrained variants
    - mask_severity: zero out the severity one-hot in obs for "w/o severity" solvers
    - estimate_constraints: derive c1..c5 from a rollout's diagnostics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from utils.config import (
    ALPHA_LAMBDA_DUAL,
    AMB_SEVERITY_NORM_OFFSET,
    OBS_FIXED_BLOCK_LEN,
    OBS_PER_AMB_BLOCK_LEN,
    SEVERITY_OH_OBS_INDEX,
    SEVERITY_QOS,
)


@dataclass
class BaselineFlags:
    """Single source of truth for ablation toggles."""

    use_phase: bool = True          # severity one-hot visible in obs? (legacy name)
    use_cmdp: bool = False          # Lagrangian-augmented reward?
    use_hrl: bool = False           # 2-level Manager/Worker scheduling? (placeholder)
    n_constraints: int = 0          # 0 (no CMDP), 2 (c1, c2 only), 5 (full)


@dataclass
class CMDPLagrangian:
    """Dual-ascent multipliers — λ_j ← max(0, λ_j + α·(J_Cj − d_j^sev))."""

    n: int = 5
    alpha: float = ALPHA_LAMBDA_DUAL
    lambdas: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __post_init__(self) -> None:
        if self.lambdas.size != self.n:
            self.lambdas = np.zeros(self.n, dtype=np.float64)

    def reset(self) -> None:
        self.lambdas = np.zeros(self.n, dtype=np.float64)

    def penalty(self, constraints: Sequence[float]) -> float:
        """Σ_j λ_j · max(0, c_j)."""
        if self.n == 0 or len(constraints) == 0:
            return 0.0
        penalty = 0.0
        for j in range(min(self.n, len(constraints))):
            penalty += float(self.lambdas[j]) * max(0.0, float(constraints[j]))
        return penalty

    def step(self, mean_constraints: Sequence[float]) -> None:
        """One dual-ascent step per (mean) constraint deviation."""
        for j in range(min(self.n, len(mean_constraints))):
            self.lambdas[j] = max(0.0, self.lambdas[j] + self.alpha * float(mean_constraints[j]))


def estimate_constraints(
    d_e2e_samples: Sequence[float],
    embb_mbps: float,
    aoi_samples: Sequence[float] | None,
    severity: int,
) -> np.ndarray:
    """Compute the 5-dim constraint deviation vector for a recent rollout window.

    c1 = mean(D_e2e) - D_max^sev
    c2 = mean(D_e2e > D_max^sev) - ε^sev
    c3 = R_min_eMBB - R_eMBB         (positive ⇒ short of floor)
    c4 = mean(AoI) - AoI_max^sev
    c5 = mean(AoI > AoI_max^sev) - ε_AoI^sev
    """
    from utils.config import CMDP_D_J_SEVERITY

    qos = SEVERITY_QOS[severity]
    d_max = qos["D_max"]
    eps_tail = qos["eps"]
    aoi_max = qos.get("AoI_max", 0.1)
    eps_aoi = qos.get("eps_aoi", 1e-3)
    # eMBB floor (Mbps) per severity — single source from CMDP_D_J_SEVERITY
    r_min_emBB = float(CMDP_D_J_SEVERITY[severity]["d3_embb_mbps"])

    if len(d_e2e_samples) == 0:
        c1 = c2 = 0.0
    else:
        arr = np.asarray(d_e2e_samples, dtype=float)
        c1 = float(arr.mean() - d_max)
        c2 = float((arr > d_max).mean() - eps_tail)

    c3 = r_min_emBB - embb_mbps

    if aoi_samples is None or len(aoi_samples) == 0:
        c4 = c5 = 0.0
    else:
        a = np.asarray(aoi_samples, dtype=float)
        c4 = float(a.mean() - aoi_max)
        c5 = float((a > aoi_max).mean() - eps_aoi)

    return np.array([c1, c2, c3, c4, c5], dtype=np.float64)


# ============================================================
# Manager state construction (shared by PPO train.py + TD3/SAC smoke_train.py)
# ============================================================


def build_manager_state(
    worker_obs: np.ndarray,
    lambda_global: np.ndarray,
) -> np.ndarray:
    """Construct (6 + 4K+1)-dim Manager state s_H from Worker obs + λ_global.

    Layout (obs is (20 + 10K + F)-dim):
        [0:2]        ρ_urllc, ρ_eMBB         (obs[0:2])
        [2]          mean BLER                (obs[9])
        [3]          severity_ref normalized  (argmax(obs[10:15]) + 1) / 5.0
        [4:6]        aoi_mean, aoi_max        (obs[18:20])
        [6:6+4K+1]   λ_global (4K+1)-dim
    At K=1: 6 + 5 = 11-dim (backward-compatible with legacy 11-dim state).
    """
    rho_urllc = float(worker_obs[0])
    rho_emBB = float(worker_obs[1])
    bler = float(worker_obs[9])
    sev_oh = worker_obs[SEVERITY_OH_OBS_INDEX: SEVERITY_OH_OBS_INDEX + 5]
    sev_idx = float((np.argmax(sev_oh) + 1) / 5.0)
    aoi_mean = float(worker_obs[18])
    aoi_max = float(worker_obs[19])
    return np.concatenate([
        np.array([rho_urllc, rho_emBB, bler, sev_idx, aoi_mean, aoi_max], dtype=np.float32),
        np.asarray(lambda_global, dtype=np.float32),
    ]).astype(np.float32)


def _manager_act(manager, s_H: np.ndarray) -> np.ndarray:
    """Unified Manager act() — extracts raw action array regardless of Manager type."""
    result = manager.act(s_H)
    if isinstance(result, tuple):
        return np.asarray(result[0], dtype=np.float32)
    return np.asarray(result, dtype=np.float32)


# ============================================================
# Observation masking — for "w/o severity" ablation solvers
# ============================================================


SEVERITY_OH_START_INDEX = 10  # 20+10K+F obs layout — severity_ref one-hot at indices [10:15] (fixed block, unaffected by K/F)
SEVERITY_OH_LEN = 5


def mask_severity(
    obs: np.ndarray,
    *,
    K: int | None = None,
    F: int = 1,
    start: int = SEVERITY_OH_START_INDEX,
    length: int = SEVERITY_OH_LEN,
) -> np.ndarray:
    """Return obs with all severity signals zeroed-out (does not mutate input).

    Zeroes the severity_ref one-hot (fixed block, [start:start+length]) AND,
    for each ambulance k, its per-ambulance severity_k_norm slot (B5 epic
    2026-06-15) — otherwise "w/o severity" ablations would still leak
    severity_k through the per-ambulance block. ``K`` is inferred from
    ``obs.shape`` (20 + 10K + F) when not given explicitly.
    """
    out = obs.copy()
    out[start : start + length] = 0.0
    if K is None:
        K = (obs.shape[-1] - OBS_FIXED_BLOCK_LEN - F) // OBS_PER_AMB_BLOCK_LEN
    for k in range(K):
        out[OBS_FIXED_BLOCK_LEN + OBS_PER_AMB_BLOCK_LEN * k + AMB_SEVERITY_NORM_OFFSET] = 0.0
    return out
