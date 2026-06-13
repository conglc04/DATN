"""Traffic generators for URLLC / eMBB / mMTC classes.

Implements 6 generators per docs/02_requirements.md#traffic-classes:
    URLLC-C1 DENM        — event-driven Poisson burst (300-800 B)
    URLLC-C2 Vital signs — periodic 100 Hz   (100-500 B)
    URLLC-C3 CAM         — periodic 10 Hz    (200-400 B)
    eMBB-V1 Video 4K     — CBR/VBR           (large, 1500 B chunks)
    eMBB-V2 Image MEC    — aperiodic         (frame-based)
    mMTC IoT             — sparse Poisson    (50-100 B)

Each generator yields Packet(arrival_time, size_bits, ttype, deadline,
priority, payload_id) and is rng-seeded for reproducibility.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import numpy as np

PacketClass = Literal[
    "URLLC_C1_DENM",
    "URLLC_C2_VITAL",
    "URLLC_C3_CAM",
    "eMBB_V1_VIDEO4K",
    "eMBB_V2_IMAGE_MEC",
    "mMTC_IOT",
]


@dataclass(slots=True)
class Packet:
    """Single packet event."""

    arrival_time: float        # seconds (sim time)
    size_bits: int             # payload size
    ttype: PacketClass
    deadline_sec: float        # arrival_time + D_max
    priority: int              # 1=highest (URLLC C1) ... 5=lowest (mMTC)
    payload_id: int            # incrementing id within class


# Priority lookup — used by MAC scheduler ranking
PRIORITY: dict[str, int] = {
    "URLLC_C1_DENM": 1,
    "URLLC_C2_VITAL": 1,
    "URLLC_C3_CAM": 2,
    "eMBB_V1_VIDEO4K": 3,
    "eMBB_V2_IMAGE_MEC": 4,
    "mMTC_IOT": 5,
}

# Latency budget per class (sec) — from docs/02_requirements.md traffic-classes
DEADLINE_SEC: dict[str, float] = {
    "URLLC_C1_DENM": 1e-3,
    "URLLC_C2_VITAL": 5e-3,
    "URLLC_C3_CAM": 3e-3,
    "eMBB_V1_VIDEO4K": 100e-3,
    "eMBB_V2_IMAGE_MEC": 50e-3,
    "mMTC_IOT": 1.0,
}


def _make_packet(
    ttype: PacketClass,
    arrival_time: float,
    size_bytes: int,
    payload_id: int,
) -> Packet:
    return Packet(
        arrival_time=arrival_time,
        size_bits=size_bytes * 8,
        ttype=ttype,
        deadline_sec=arrival_time + DEADLINE_SEC[ttype],
        priority=PRIORITY[ttype],
        payload_id=payload_id,
    )


# ============================================================
# URLLC generators
# ============================================================


def gen_urllc_denm(
    duration_sec: float,
    lambda_base: float = 50.0,
    burst_lambda: float = 500.0,
    burst_start: float | None = None,
    burst_duration: float = 0.5,
    size_range_bytes: tuple[int, int] = (300, 800),
    rng: np.random.Generator | None = None,
) -> list[Packet]:
    """URLLC-C1 DENM: event-driven Poisson burst.

    Steady-state λ=50 pkt/s; during a collision burst λ jumps to 500 pkt/s
    for `burst_duration` seconds (default 500ms — docs/08:87).

    Args:
        burst_start: When the burst begins. None ⇒ no burst.
    """
    rng = rng or np.random.default_rng()
    packets: list[Packet] = []
    pid = 0
    t = 0.0

    while t < duration_sec:
        # Time-varying λ
        in_burst = (
            burst_start is not None and burst_start <= t < burst_start + burst_duration
        )
        lam = burst_lambda if in_burst else lambda_base
        if lam <= 0:
            break
        # Poisson interarrival
        dt = rng.exponential(1.0 / lam)
        t += dt
        if t >= duration_sec:
            break
        size = int(rng.integers(size_range_bytes[0], size_range_bytes[1] + 1))
        packets.append(_make_packet("URLLC_C1_DENM", t, size, pid))
        pid += 1
    return packets


def gen_urllc_vital(
    duration_sec: float,
    frequency_hz: float = 100.0,
    size_range_bytes: tuple[int, int] = (100, 500),
    jitter_sec: float = 1e-4,
    rng: np.random.Generator | None = None,
) -> list[Packet]:
    """URLLC-C2 Vital signs: 100 Hz periodic with small jitter."""
    rng = rng or np.random.default_rng()
    period = 1.0 / frequency_hz
    n = int(duration_sec * frequency_hz)
    packets: list[Packet] = []
    for pid in range(n):
        t = pid * period + (rng.uniform(-jitter_sec, jitter_sec) if jitter_sec > 0 else 0.0)
        t = max(t, 0.0)
        if t >= duration_sec:
            break
        size = int(rng.integers(size_range_bytes[0], size_range_bytes[1] + 1))
        packets.append(_make_packet("URLLC_C2_VITAL", t, size, pid))
    return packets


def gen_urllc_cam(
    duration_sec: float,
    frequency_hz: float = 10.0,
    size_range_bytes: tuple[int, int] = (200, 400),
    rng: np.random.Generator | None = None,
) -> list[Packet]:
    """URLLC-C3 CAM: 10 Hz periodic."""
    rng = rng or np.random.default_rng()
    period = 1.0 / frequency_hz
    n = int(duration_sec * frequency_hz)
    packets: list[Packet] = []
    for pid in range(n):
        t = pid * period
        size = int(rng.integers(size_range_bytes[0], size_range_bytes[1] + 1))
        packets.append(_make_packet("URLLC_C3_CAM", t, size, pid))
    return packets


# ============================================================
# eMBB generators
# ============================================================


def gen_embb_video(
    duration_sec: float,
    rate_mbps: float = 5.0,
    chunk_bytes: int = 1500,
    cv: float = 0.3,
    rng: np.random.Generator | None = None,
) -> list[Packet]:
    """eMBB-V1 4K Video: VBR around `rate_mbps`, chunked into packets.

    Interarrival jitter via lognormal with coefficient of variation `cv`.
    cv=0 → CBR exact.
    """
    rng = rng or np.random.default_rng()
    pkts_per_sec = (rate_mbps * 1e6) / (chunk_bytes * 8)
    mean_interarrival = 1.0 / pkts_per_sec
    packets: list[Packet] = []
    pid = 0
    t = 0.0
    while t < duration_sec:
        if cv > 0:
            sigma = (cv ** 2 + 1.0) ** 0.5  # not used directly; use lognormal scale param
            # Approximate VBR by multiplying mean by lognormal(0, log(1+cv²))
            mu_ln = -0.5 * np.log(1 + cv ** 2)
            sigma_ln = np.sqrt(np.log(1 + cv ** 2))
            scale = np.exp(rng.normal(mu_ln, sigma_ln))
            dt = mean_interarrival * scale
        else:
            dt = mean_interarrival
        t += dt
        if t >= duration_sec:
            break
        packets.append(_make_packet("eMBB_V1_VIDEO4K", t, chunk_bytes, pid))
        pid += 1
    return packets


def gen_embb_image_mec(
    duration_sec: float,
    images_per_sec: float = 5.0,
    image_size_bytes: int = 50_000,    # ~50KB JPEG
    chunk_bytes: int = 1500,
    rng: np.random.Generator | None = None,
) -> list[Packet]:
    """eMBB-V2 MEC analytics: aperiodic image uploads.

    Each image is fragmented into MTU-sized chunks (default 1500B).
    """
    rng = rng or np.random.default_rng()
    chunks_per_image = max(1, image_size_bytes // chunk_bytes)
    packets: list[Packet] = []
    pid = 0
    t = 0.0
    while t < duration_sec:
        # Image arrival (Poisson)
        dt = rng.exponential(1.0 / images_per_sec)
        t += dt
        if t >= duration_sec:
            break
        # Burst all chunks at this instant (then sub-jittered slightly)
        for k in range(chunks_per_image):
            t_chunk = t + k * 1e-6  # microsecond burst
            packets.append(_make_packet("eMBB_V2_IMAGE_MEC", t_chunk, chunk_bytes, pid))
            pid += 1
    return packets


# ============================================================
# mMTC generators
# ============================================================


def gen_mmtc(
    duration_sec: float,
    n_devices: int = 50,
    per_device_lambda: float = 0.1,
    size_range_bytes: tuple[int, int] = (50, 100),
    rng: np.random.Generator | None = None,
) -> list[Packet]:
    """mMTC sparse Poisson — each device sends at λ=0.1 pkt/s."""
    rng = rng or np.random.default_rng()
    total_lambda = n_devices * per_device_lambda
    packets: list[Packet] = []
    pid = 0
    t = 0.0
    while t < duration_sec:
        dt = rng.exponential(1.0 / total_lambda)
        t += dt
        if t >= duration_sec:
            break
        size = int(rng.integers(size_range_bytes[0], size_range_bytes[1] + 1))
        packets.append(_make_packet("mMTC_IOT", t, size, pid))
        pid += 1
    return packets


# ============================================================
# Multi-class mixer
# ============================================================


def mix_traffic(*streams: list[Packet]) -> list[Packet]:
    """Merge multiple packet streams and sort by arrival time."""
    merged: list[Packet] = []
    for s in streams:
        merged.extend(s)
    merged.sort(key=lambda p: p.arrival_time)
    return merged


def aggregate_arrival_rate(packets: list[Packet], duration_sec: float) -> float:
    """Empirical λ (packets/sec) from a generated stream."""
    if duration_sec <= 0:
        return 0.0
    return len(packets) / duration_sec


def aggregate_mean_size_bits(packets: list[Packet]) -> float:
    """Mean packet size in bits across a stream."""
    if not packets:
        return 0.0
    return float(np.mean([p.size_bits for p in packets]))


def iterate_packets_in_window(
    packets: list[Packet], t_start: float, t_end: float
) -> Iterator[Packet]:
    """Yield packets with arrival_time in [t_start, t_end)."""
    for p in packets:
        if t_start <= p.arrival_time < t_end:
            yield p
        elif p.arrival_time >= t_end:
            break
