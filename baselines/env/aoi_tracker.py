"""Age of Information (AoI) tracker for the per-ambulance aggregated URLLC stream.

Implements:
    - STREAM_TYPES classification (single LCFS+drop_old aggregated stream)
    - AoI definition Δ(t) = t − U(t) where U(t) = generation time of the
      most recently delivered packet [Kangwei Qi et al. 2024 §III;
      Xianfu Chen et al. §II; Zoubeir Mlika et al. 2022 §II]
    - LCFS-with-drop-old queue (latest-status semantics: stale reports are
      superseded by the newest one before delivery)
      [Kangwei Qi et al. 2024; Xianfu Chen et al.]
    - AoI violation rate vs AoI_max^sev thresholds

Reference:
    - docs/04_data_flow.md AoI formula (lines 150-160)
    - Kangwei Qi et al. 2024 (primary corpus source for AoI definition + LCFS)
    - Xianfu Chen et al. (LCFS freshest-first approximation)
    - Zoubeir Mlika et al. 2022 §II (AoI formula)
    Note: expected_aoi_mm1() helper uses Kaul 2012 M/M/1 closed-form for
    unit tests only — NOT the primary AoI model of this simulator.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Literal

from utils.config import SEVERITY_QOS

StreamId = str  # e.g. "ambulance_status"
QueueDiscipline = Literal["LCFS", "FCFS"]


# Stream classification — single aggregated URLLC status stream per ambulance
# (periodic bundle, 1 packet/report-cycle; LCFS+drop_old = latest-status semantics).
STREAM_TYPES: dict[str, dict[str, bool | str]] = {
    "ambulance_status": {"queue": "LCFS", "drop_old": True, "aoi_aware": True},
}

# AoI threshold (from SEVERITY_QOS) — map stream → key into SEVERITY_QOS
AOI_THRESHOLD_KEY: dict[str, str] = {
    "ambulance_status": "AoI_max",
}


@dataclass(slots=True)
class AoIPacket:
    """Update packet with generation time."""

    gen_time: float        # when sensor generated
    deliver_time: float | None = None
    payload_id: int = 0


@dataclass
class AoIStreamTracker:
    """Per-stream AoI bookkeeping.

    LCFS+drop_old:
        Buffer holds at most the newest pending update.
        Newer arrival drops older queued one.
    FCFS:
        Standard FIFO, no drops.
    """

    stream_id: StreamId
    queue_kind: QueueDiscipline = "FCFS"
    drop_old: bool = False

    # Internal state
    queue: deque[AoIPacket] = field(default_factory=deque)
    last_delivered_gen_time: float | None = None
    dropped_count: int = 0
    delivered_count: int = 0
    aoi_samples: list[float] = field(default_factory=list)        # observed AoI at delivery instants

    @classmethod
    def from_spec(cls, stream_id: StreamId) -> "AoIStreamTracker":
        """Build a tracker from STREAM_TYPES classification."""
        spec = STREAM_TYPES.get(stream_id)
        if spec is None:
            raise KeyError(f"Unknown stream_id: {stream_id}")
        return cls(
            stream_id=stream_id,
            queue_kind=spec["queue"],         # type: ignore[arg-type]
            drop_old=bool(spec["drop_old"]),
        )

    def arrive(self, gen_time: float, payload_id: int = 0) -> None:
        """A new sensor update arrives at the queue."""
        pkt = AoIPacket(gen_time=gen_time, payload_id=payload_id)
        if self.queue_kind == "LCFS" and self.drop_old:
            # Drop any older pending packet and queue only the newest
            self.dropped_count += len(self.queue)
            self.queue.clear()
            self.queue.append(pkt)
        else:
            self.queue.append(pkt)

    def deliver_next(self, sim_time: float) -> AoIPacket | None:
        """MAC scheduler picks the next packet to transmit successfully.

        LCFS+drop_old picks the freshest queued packet (which is the only one);
        FCFS picks the head of the FIFO. Return the delivered packet (with
        deliver_time set) or None if queue is empty.
        """
        if not self.queue:
            return None
        if self.queue_kind == "LCFS":
            pkt = self.queue.pop()       # newest
            # Any older packets are stale; drop them.
            self.dropped_count += len(self.queue)
            self.queue.clear()
        else:
            pkt = self.queue.popleft()   # oldest (FCFS)
        pkt.deliver_time = sim_time
        self.delivered_count += 1
        self.last_delivered_gen_time = pkt.gen_time
        self.aoi_samples.append(self.current_aoi(sim_time))
        return pkt

    def current_aoi(self, sim_time: float) -> float:
        """AoI at receiver = sim_time − gen_time(most recently delivered)."""
        if self.last_delivered_gen_time is None:
            return sim_time         # never delivered ⇒ AoI = elapsed time
        return sim_time - self.last_delivered_gen_time

    def violation_rate(self, threshold_sec: float) -> float:
        """Fraction of observed AoI samples exceeding threshold."""
        if not self.aoi_samples:
            return 0.0
        return sum(1 for a in self.aoi_samples if a > threshold_sec) / len(self.aoi_samples)

    def reset(self) -> None:
        self.queue.clear()
        self.last_delivered_gen_time = None
        self.dropped_count = 0
        self.delivered_count = 0
        self.aoi_samples.clear()


def expected_aoi_mm1(arrival_rate: float, service_rate: float) -> float:
    """Average AoI for M/M/1 FCFS — Kaul 2012.

    E[AoI] = (1/μ) · (1 + ρ/(1-ρ) + ρ²/(1-ρ²))
    Returns inf if unstable.
    """
    if service_rate <= 0 or arrival_rate < 0:
        return float("inf")
    rho = arrival_rate / service_rate
    if rho >= 1.0:
        return float("inf")
    return (1.0 / service_rate) * (1.0 + rho / (1.0 - rho) + (rho ** 2) / (1.0 - rho ** 2))


def aoi_threshold_for_severity(severity: int, stream_id: StreamId) -> float:
    """Look up AoI_max^sev for a given aggregated-vital stream.

    Returns inf for streams without an aggregated AoI threshold (waveforms).
    """
    key = AOI_THRESHOLD_KEY.get(stream_id)
    if key is None:
        return float("inf")
    return float(SEVERITY_QOS[severity][key])
