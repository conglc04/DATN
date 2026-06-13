"""NSF (Neural Safety Filter) — Identity stub for W07.

W07 scope: IdentityNSF returns ``a_raw.detach()`` so the β_qp distillation term
in Worker actor loss is identically zero. Lets us wire the full PA-CHRL-PPO
algorithm without yet training an actual safety projection net.

Full NSF MLP + OSQP-label offline pre-training (Algorithm 0B) is deferred to
sub-phase D (parallel with W11+).

Reference:
    - docs/13_methodology_walkthrough.md Phase 3.3.5 (NSF architecture)
    - docs/13_methodology_walkthrough.md Phase 3.4.4 N2 (detach a_safe)
"""

from __future__ import annotations

import numpy as np
import torch


class IdentityNSF:
    """No-op safety filter. Always returns ``a_raw`` unchanged (detached)."""

    name: str = "identity_nsf"

    def forward(
        self,
        s_L: np.ndarray | torch.Tensor,
        a_raw: np.ndarray | torch.Tensor,
    ) -> np.ndarray | torch.Tensor:
        """Return a_safe = a_raw (detached if Tensor).

        Behaviour:
            torch.Tensor in  → torch.Tensor out (detached, no grad)
            np.ndarray in    → np.ndarray out (copy)
        """
        if isinstance(a_raw, torch.Tensor):
            return a_raw.detach().clone()
        return np.asarray(a_raw, dtype=np.float32).copy()

    def __call__(self, s_L, a_raw):
        return self.forward(s_L, a_raw)

    def is_trained(self) -> bool:
        """Identity stub is by definition not a trained NSF."""
        return False
