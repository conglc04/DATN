"""Metric computations — function stubs (Week 1).

Full implementation comes in Week 2-4 alongside env modules.
Reference: docs/06_validation.md, docs/04_data_flow.md
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def e2e_delay_breakdown(
    d_tx: float, d_queue: float, d_fh: float = 0.1e-3, d_bh: float = 0.1e-3, d_det: float = 0.07e-3
) -> dict[str, float]:
    """Decompose E2E delay into components for sanity reporting.

    D_e2e = D_det + D_tx + D_queue + D_fh + D_bh
    Reference: docs/04_data_flow.md, target at φ₃ ≈ 0.7-0.9ms.
    """
    return {
        "D_det": d_det,
        "D_tx": d_tx,
        "D_queue": d_queue,
        "D_fh": d_fh,
        "D_bh": d_bh,
        "D_total": d_det + d_tx + d_queue + d_fh + d_bh,
    }


def violation_rate(delays: Sequence[float], d_max: float) -> float:
    """Fraction of packets with D_e2e > D_max."""
    if len(delays) == 0:
        return 0.0
    arr = np.asarray(delays, dtype=float)
    return float((arr > d_max).mean())


def jain_fairness(values: Sequence[float]) -> float:
    """Jain's fairness index ∈ [0, 1]. 1.0 = perfectly fair.

    Reference: docs/06_validation.md (Exp5 multi-ambulance).
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(arr == 0):
        return 0.0
    num = arr.sum() ** 2
    den = arr.size * (arr * arr).sum()
    return float(num / den)


def aoi_violation_rate(aoi_samples: Sequence[float], aoi_max: float) -> float:
    """Fraction of AoI samples exceeding the cap (for C5)."""
    if len(aoi_samples) == 0:
        return 0.0
    arr = np.asarray(aoi_samples, dtype=float)
    return float((arr > aoi_max).mean())


def embb_throughput_mbps(bits_delivered: float, duration_sec: float) -> float:
    """Throughput in Mbps."""
    if duration_sec <= 0:
        return 0.0
    return bits_delivered / duration_sec / 1e6


def queue_stability_check(arrival_rate: float, service_rate: float) -> bool:
    """ρ_s = λ/μ < 1 ⇒ stable (per C12)."""
    if service_rate <= 0:
        return False
    return (arrival_rate / service_rate) < 1.0


def hoeffding_sample_size(target_eps: float, observed_eps: float, confidence: float = 0.99) -> int:
    """Hoeffding bound sample size for true-violation upper bound.

    N ≥ ln(2/δ) / (2·(target − observed)²)
    Reference: docs/06_validation.md violation-rate.
    """
    delta = 1.0 - confidence
    if target_eps <= observed_eps:
        return 0
    gap = target_eps - observed_eps
    return int(np.ceil(np.log(2.0 / delta) / (2.0 * gap * gap)))


# ============================================================
# Sanity-check assertions (placeholders — wired up in Week 4)
# ============================================================
def assert_prb_budget(prb_urllc: int, prb_embb: int, p_total: int = 273) -> None:
    assert prb_urllc + prb_embb <= p_total, (
        f"PRB budget violated: {prb_urllc} + {prb_embb} > {p_total}"
    )


def assert_phase_constraint(phase: int, d_e2e_mean: float, d_max_phi: float) -> None:
    assert d_e2e_mean <= d_max_phi, (
        f"Phase φ{phase} constraint violated: mean={d_e2e_mean*1e3:.3f}ms > "
        f"D_max={d_max_phi*1e3:.3f}ms"
    )
