"""M/G/1 queue per slice for O-RAN simulator.

Implements:
    - Service rate computation: μ = PRB · C_avg / L_avg
    - Pollaczek-Khinchine formula: E[D_queue] = λ·E[S²] / (2(1-ρ))
    - Augmented service time with D_stoch (RLC + retx variance)
    - HOL delay
    - Stability check ρ < 0.9 (engineering margin under C12 theoretical ρ<1)

Reference:
    - docs/04_data_flow.md#queue-model (lines 97-117)
    - docs/04_data_flow.md (D_stoch handling in M/G/1 σ²)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from utils.config import D_STOCH

RHO_MAX_STABILITY: float = 0.9  # docs/04:113 — engineering margin (theory: ρ<1)


@dataclass(slots=True)
class MG1Queue:
    """Single M/G/1 queue tracking arrival rate, service rate, and delay stats.

    Service time has variance from RLC retx (D_stoch in docs/04). The
    augmented service time is:
        E[S]   = E[S_pure] + D_stoch
        E[S²]  = E[S_pure]² + Var(S_pure) + 2·E[S_pure]·D_stoch + D_stoch²
                 (approximating Var(S_total) ≈ Var(S_pure) + Var(D_stoch))

    For simplicity in Week 2 we set:
        Var(S_total) ≈ (D_stoch)² · cv² where cv ≈ 1.0 (exponential-ish retx)
    """

    name: str
    arrival_rate: float = 0.0       # λ (packets / second)
    service_rate: float = 1.0       # μ (packets / second)
    mean_packet_bits: float = 1000.0
    d_stoch_sec: float = D_STOCH    # default 0.05 ms (reviewer PB-C2 fix 2026-05-27)

    @property
    def rho(self) -> float:
        """Utilization ρ = λ/μ."""
        if self.service_rate <= 0:
            return float("inf")
        return self.arrival_rate / self.service_rate

    @property
    def is_stable(self) -> bool:
        """ρ < 0.9 (engineering margin per docs/04:113)."""
        return self.rho < RHO_MAX_STABILITY

    @property
    def mean_service_time(self) -> float:
        """E[S] = 1/μ + D_stoch."""
        return 1.0 / self.service_rate + self.d_stoch_sec

    @property
    def second_moment_service(self) -> float:
        """E[S²] for the augmented service time.

        Assume pure service is exponential (M/M/1 limit) → E[S²] = 2/μ²
        plus D_stoch contribution.
        """
        e_s_pure = 1.0 / self.service_rate
        var_pure = e_s_pure ** 2                   # exponential variance = mean²
        var_stoch = self.d_stoch_sec ** 2          # rough variance for retx
        e_s = self.mean_service_time
        var_total = var_pure + var_stoch
        return var_total + e_s ** 2                # E[S²] = Var(S) + (E[S])²

    def expected_queue_delay(self) -> float:
        """Pollaczek-Khinchine: E[D_queue] = λ·E[S²] / (2(1-ρ)).

        Returns inf if not stable.
        """
        rho = self.rho
        if rho >= 1.0:
            return float("inf")
        return self.arrival_rate * self.second_moment_service / (2.0 * (1.0 - rho))

    def hol_delay(self) -> float:
        """Head-of-line delay = queueing delay + service time."""
        return self.expected_queue_delay() + self.mean_service_time

    def update_service_rate(self, prb_count: int, capacity_per_prb_bps: float) -> None:
        """μ = (PRB · C_avg) / L_avg.

        Reference: docs/04_data_flow.md:103
        """
        if self.mean_packet_bits <= 0:
            raise ValueError("mean_packet_bits must be positive")
        self.service_rate = max(prb_count * capacity_per_prb_bps / self.mean_packet_bits, 1e-9)

    def set_arrival_rate(self, lam: float) -> None:
        if lam < 0:
            raise ValueError(f"arrival rate must be non-negative, got {lam}")
        self.arrival_rate = lam

    def summary(self) -> dict[str, float]:
        return {
            "name_idx": hash(self.name) & 0xFFFF,
            "lambda": self.arrival_rate,
            "mu": self.service_rate,
            "rho": self.rho,
            "E_S": self.mean_service_time,
            "E_S2": self.second_moment_service,
            "E_D_queue": self.expected_queue_delay(),
            "HOL": self.hol_delay(),
            "stable": float(self.is_stable),
        }


# ============================================================
# Helper: HOL delay tail bound via Chernoff / Markov inequality
# ============================================================


def hol_tail_bound_markov(expected_hol: float, threshold: float) -> float:
    """Markov upper bound: P(HOL > d) ≤ E[HOL]/d.

    Loose but always valid (no distributional assumption). For tight bounds
    use Chernoff with M/G/1 specific moments.
    """
    if threshold <= 0:
        return 1.0
    if expected_hol < 0 or expected_hol == float("inf"):
        return 1.0
    return min(expected_hol / threshold, 1.0)


def hol_tail_bound_chernoff_mm1(
    arrival_rate: float, service_rate: float, threshold: float
) -> float:
    """Tight tail bound for M/M/1: P(D > d) = ρ·exp(-(μ-λ)·d).

    Returns 1.0 if unstable (ρ ≥ 1).
    """
    if service_rate <= arrival_rate:
        return 1.0
    if threshold <= 0:
        return 1.0
    import math
    rho = arrival_rate / service_rate
    return min(rho * math.exp(-(service_rate - arrival_rate) * threshold), 1.0)


# ============================================================
# Multi-slice queue manager
# ============================================================


@dataclass
class SliceQueueManager:
    """Group of queues, one per slice (URLLC, eMBB, mMTC)."""

    queues: dict[str, MG1Queue] = field(default_factory=dict)

    def add(self, queue: MG1Queue) -> None:
        self.queues[queue.name] = queue

    def all_stable(self) -> bool:
        """C12: every slice must be stable simultaneously."""
        return all(q.is_stable for q in self.queues.values())

    def total_load(self) -> float:
        """Sum of utilizations (sanity check for PRB allocation)."""
        return sum(q.rho for q in self.queues.values())

    def __getitem__(self, name: str) -> MG1Queue:
        return self.queues[name]

    def __iter__(self):
        return iter(self.queues.values())

    def __len__(self) -> int:
        return len(self.queues)
