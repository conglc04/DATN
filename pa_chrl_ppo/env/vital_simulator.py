"""Synthetic vital-sign and ambulance kinematic trace generator.

Used for:
    - LSTM training data (vital trace prediction / phase context)
    - Reward shaping debugging
    - Manual scenario seeding

Streams produced:
    - ECG waveform (250-500 Hz, ~1 mV peak, P-QRS-T composite)
    - SpO2 (slow varying 90-100%)
    - Heart rate (60-180 bpm with HRV)
    - Speed (km/h, ambulance kinematics)

Reference: docs/09_execution_plan.md:51 (LSTM training data)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ============================================================
# ECG synthetic waveform
# ============================================================


def _gaussian_bump(t: np.ndarray, center: float, width: float, amplitude: float) -> np.ndarray:
    return amplitude * np.exp(-((t - center) ** 2) / (2.0 * width ** 2))


def synth_ecg(
    duration_sec: float,
    sampling_hz: float = 250.0,
    heart_rate_bpm: float = 75.0,
    noise_std_mv: float = 0.02,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate synthetic ECG waveform (single lead) ~ mV.

    Uses Gaussian-bump approximation for P, Q, R, S, T waves.
    Returns array of length round(duration_sec * sampling_hz).
    """
    rng = rng or np.random.default_rng()
    n_samples = int(round(duration_sec * sampling_hz))
    if n_samples <= 0:
        return np.zeros(0, dtype=float)
    t = np.arange(n_samples) / sampling_hz
    rr = 60.0 / heart_rate_bpm  # R-R interval (sec)

    # Add HRV jitter (±5% per beat)
    hrv_factor = rng.uniform(0.95, 1.05, size=int(np.ceil(duration_sec / rr)) + 1)
    beat_times: list[float] = []
    t_curr = 0.0
    for jitter in hrv_factor:
        beat_times.append(t_curr)
        t_curr += rr * float(jitter)
        if t_curr >= duration_sec:
            break

    signal = np.zeros_like(t)
    # PQRST template offsets relative to R peak (sec)
    waves = [
        # (offset, width, amplitude_mV)
        (-0.20, 0.020,  0.10),   # P
        (-0.04, 0.010, -0.15),   # Q
        ( 0.00, 0.008,  1.00),   # R (dominant)
        ( 0.04, 0.010, -0.25),   # S
        ( 0.15, 0.040,  0.30),   # T
    ]
    for beat in beat_times:
        for off, w, amp in waves:
            signal += _gaussian_bump(t, beat + off, w, amp)

    signal += rng.normal(0.0, noise_std_mv, size=n_samples)
    return signal


# ============================================================
# SpO2 — slow varying 90-100%
# ============================================================


def synth_spo2(
    duration_sec: float,
    sampling_hz: float = 1.0,
    baseline: float = 98.0,
    noise_std: float = 0.3,
    drift_period_sec: float = 60.0,
    drift_amp: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate SpO2 percentage trace.

    Defaults: 98% baseline ± 1% slow drift + Gaussian noise.
    """
    rng = rng or np.random.default_rng()
    n_samples = int(round(duration_sec * sampling_hz))
    if n_samples <= 0:
        return np.zeros(0, dtype=float)
    t = np.arange(n_samples) / sampling_hz
    drift = drift_amp * np.sin(2.0 * np.pi * t / drift_period_sec)
    noise = rng.normal(0.0, noise_std, size=n_samples)
    return np.clip(baseline + drift + noise, 70.0, 100.0)


# ============================================================
# Heart-rate trace (bpm) — HRV via Ornstein-Uhlenbeck-like jitter
# ============================================================


def synth_heart_rate(
    duration_sec: float,
    sampling_hz: float = 1.0,
    baseline_bpm: float = 80.0,
    hrv_std: float = 3.0,
    mean_reversion: float = 0.05,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Heart-rate trace with HRV mean-reverting around `baseline_bpm`."""
    rng = rng or np.random.default_rng()
    n_samples = int(round(duration_sec * sampling_hz))
    if n_samples <= 0:
        return np.zeros(0, dtype=float)
    series = np.empty(n_samples, dtype=float)
    series[0] = baseline_bpm
    for i in range(1, n_samples):
        # OU: dx = θ·(μ − x)dt + σ·dW
        dx = mean_reversion * (baseline_bpm - series[i - 1]) + rng.normal(0.0, hrv_std)
        series[i] = max(40.0, min(220.0, series[i - 1] + dx * (1.0 / sampling_hz)))
    return series


# ============================================================
# Ambulance speed trace (km/h)
# ============================================================


def synth_speed_kmh(
    duration_sec: float,
    sampling_hz: float = 1.0,
    phase: int = 4,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Ambulance speed by phase.

    Phase mapping (rough Hanoi traffic profile):
        φ₁ STANDBY   → 0
        φ₂ DISPATCH  → 30-60 km/h (urgent, dense traffic)
        φ₃ SCENE     → 0 (parked)
        φ₄ TRANSPORT → 40-70 km/h
        φ₅ RETURN    → 20-40 km/h
    """
    rng = rng or np.random.default_rng()
    n_samples = int(round(duration_sec * sampling_hz))
    if n_samples <= 0:
        return np.zeros(0, dtype=float)
    ranges = {
        1: (0.0, 0.0),
        2: (30.0, 60.0),
        3: (0.0, 0.0),
        4: (40.0, 70.0),
        5: (20.0, 40.0),
    }
    lo, hi = ranges.get(phase, (0.0, 0.0))
    if hi == 0.0:
        return np.zeros(n_samples)
    baseline = rng.uniform(lo, hi)
    noise = rng.normal(0.0, 3.0, size=n_samples)
    speed = np.clip(baseline + noise.cumsum() * 0.05, 0.0, 90.0)
    return speed


# ============================================================
# Multi-stream LSTM trace bundle
# ============================================================


@dataclass(slots=True)
class VitalTrace:
    """Composite trace for LSTM training (one ambulance, one time window)."""

    duration_sec: float
    ecg: np.ndarray = field(default_factory=lambda: np.zeros(0))
    spo2: np.ndarray = field(default_factory=lambda: np.zeros(0))
    heart_rate: np.ndarray = field(default_factory=lambda: np.zeros(0))
    speed_kmh: np.ndarray = field(default_factory=lambda: np.zeros(0))
    ecg_sampling_hz: float = 250.0
    slow_sampling_hz: float = 1.0


def generate_trace_bundle(
    duration_sec: float,
    phase: int = 4,
    rng: np.random.Generator | None = None,
    ecg_sampling_hz: float = 250.0,
    slow_sampling_hz: float = 1.0,
    heart_rate_bpm: float = 80.0,
) -> VitalTrace:
    """Generate a full multi-stream trace for LSTM training.

    Returns four signals at two sampling rates (ECG fast, vital slow).
    """
    rng = rng or np.random.default_rng()
    return VitalTrace(
        duration_sec=duration_sec,
        ecg=synth_ecg(duration_sec, ecg_sampling_hz, heart_rate_bpm, rng=rng),
        spo2=synth_spo2(duration_sec, slow_sampling_hz, rng=rng),
        heart_rate=synth_heart_rate(
            duration_sec, slow_sampling_hz, baseline_bpm=heart_rate_bpm, rng=rng
        ),
        speed_kmh=synth_speed_kmh(duration_sec, slow_sampling_hz, phase=phase, rng=rng),
        ecg_sampling_hz=ecg_sampling_hz,
        slow_sampling_hz=slow_sampling_hz,
    )
