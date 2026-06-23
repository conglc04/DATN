"""Observation post-processing shared across ALL solvers (single source).

The CMDP-Lagrangian formulation conditions the policy on the current dual
variables λ (docs/13 §2.3 + §3.2). The Worker observation reserves slots for
λ_local (the shared C3 slot at obs[15] plus 4 per-ambulance slots inside each
ambulance's 10-dim block), but the environment itself does NOT know λ (λ
lives in the agent-side LambdaState). The training loop is therefore
responsible for injecting the *current* λ_local into the observation before
the policy sees it — this is what makes the augmented-reward MDP Markovian.

CRITICAL: this overlay MUST be applied identically by every solver driver
(train.py for PPO, solvers/train_offpolicy.py for TD3/SAC). A solver that skips
it observes λ=0 forever, optimizes a non-stationary target, and is no longer
solving the same problem as the others. Keeping one definition here prevents
the per-file drift that previously left TD3/SAC blind to λ.

B5 severity_k epic (2026-06-15): λ_local is now (4K+1,)-dim, laid out as
[C1_0..C1_{K-1}, C2_0..C2_{K-1}, C4_0..C4_{K-1}, C5_0..C5_{K-1}, C3_shared].
The overlay scatters this into the non-contiguous obs slots: the shared C3
slot at obs[LAMBDA_C3_SHARED_OBS_INDEX], and the per-ambulance C1/C2/C4/C5
slots inside each ambulance's OBS_PER_AMB_BLOCK_LEN-dim block.
"""

from __future__ import annotations

import numpy as np

from utils.config import (
    AMB_LAMBDA_C1_OFFSET,
    AMB_LAMBDA_C2_OFFSET,
    AMB_LAMBDA_C4_OFFSET,
    AMB_LAMBDA_C5_OFFSET,
    LAMBDA_C3_SHARED_OBS_INDEX,
    OBS_FIXED_BLOCK_LEN,
    OBS_PER_AMB_BLOCK_LEN,
)


def overlay_lambda_local(obs: np.ndarray, lambda_local: np.ndarray, K: int) -> np.ndarray:
    """Return a copy of ``obs`` with the λ_local slots set to ``lambda_local``.

    ``lambda_local`` is the (4K+1,)-dim dual snapshot
    ``[C1_0..C1_{K-1}, C2_0..C2_{K-1}, C4_0..C4_{K-1}, C5_0..C5_{K-1}, C3_shared]``
    — ALSO used to compute the augmented reward for the same decision step,
    so the state and its reward stay mutually consistent.

    Pure function — does not mutate the input. At K=1 this writes exactly
    4 per-ambulance slots + 1 shared slot, equivalent to the old contiguous
    5-slot overlay under the [0,1,3,4,2] permutation.
    """
    lam = np.asarray(lambda_local, dtype=np.float32)
    if lam.shape != (4 * K + 1,):
        raise ValueError(f"lambda_local shape {lam.shape} != ({4 * K + 1},)")

    out = obs.astype(np.float32, copy=True)
    out[LAMBDA_C3_SHARED_OBS_INDEX] = lam[4 * K]
    for k in range(K):
        base = OBS_FIXED_BLOCK_LEN + OBS_PER_AMB_BLOCK_LEN * k
        out[base + AMB_LAMBDA_C1_OFFSET] = lam[k]
        out[base + AMB_LAMBDA_C2_OFFSET] = lam[K + k]
        out[base + AMB_LAMBDA_C4_OFFSET] = lam[2 * K + k]
        out[base + AMB_LAMBDA_C5_OFFSET] = lam[3 * K + k]
    return out
