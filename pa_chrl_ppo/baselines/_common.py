"""Shared infrastructure for baselines.

Provides:
    - BaselineFlags: configuration toggles (phase visibility, CMDP, HRL)
    - CMDPLagrangian: dual-ascent multipliers for constrained variants
    - PhaseMaskWrapper: zero out the phase one-hot in obs for "w/o Phase" baselines
    - estimate_constraints: derive c1..c5 from a rollout's diagnostics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from utils.config import ALPHA_LAMBDA_DUAL, PHASE_QOS


@dataclass
class BaselineFlags:
    """Single source of truth for ablation toggles."""

    use_phase: bool = True          # phase one-hot visible in obs?
    use_cmdp: bool = False          # Lagrangian-augmented reward?
    use_hrl: bool = False           # 2-level Manager/Worker scheduling? (placeholder)
    use_safety_qp: bool = False     # NSF/OSQP runtime safety? (placeholder)
    n_constraints: int = 0          # 0 (no CMDP), 2 (c1, c2 only), 5 (full)


@dataclass
class CMDPLagrangian:
    """Dual-ascent multipliers — λ_j ← max(0, λ_j + α·(J_Cj − d_j^φ))."""

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
    phase: int,
) -> np.ndarray:
    """Compute the 5-dim constraint deviation vector for a recent rollout window.

    c1 = mean(D_e2e) - D_max^φ
    c2 = mean(D_e2e > D_max^φ) - ε^φ
    c3 = R_min_eMBB - R_eMBB         (positive ⇒ short of floor)
    c4 = mean(AoI) - AoI_max^φ
    c5 = mean(AoI > AoI_max^φ) - ε_AoI^φ
    """
    qos = PHASE_QOS[phase]
    d_max = qos["D_max"]
    eps_tail = qos["eps"]
    aoi_max = qos.get("AoI_max_HR", 0.1)
    eps_aoi = qos.get("eps_aoi", 1e-3)
    # Use a sensible default eMBB floor (Mbps) per phase
    r_min_emBB = 30.0 if phase in (3, 4) else 10.0

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
# Observation masking — for "w/o Phase" baselines
# ============================================================


PHASE_OH_START_INDEX = 10  # W05 40-dim obs layout — phase one-hot at indices [10:15]
PHASE_OH_LEN = 5


def mask_phase(obs: np.ndarray, *, start: int = PHASE_OH_START_INDEX, length: int = PHASE_OH_LEN) -> np.ndarray:
    """Return obs with the phase one-hot block zeroed-out (does not mutate input)."""
    out = obs.copy()
    out[start : start + length] = 0.0
    return out
