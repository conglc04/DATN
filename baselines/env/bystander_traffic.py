"""S2B sub-scenario — bystander livestream spike model.

Reference: docs/02_requirements.md:51-53 (Sub-scenario S2B)
    When a collision happens at a crowded intersection, 30-50% of passers-by
    livestream → eMBB UE arrival ~30 baseline → 80-120 UEs (2-5 Mbps UL/UE).

    This is the slice-isolation acid test: PPO must reject new eMBB via
    RAC and/or force handover to n28.

Implements:
    - BystanderArrivalModel: time-varying UE arrival process around a triggering event
    - Per-UE traffic generator (eMBB livestream at 2-5 Mbps)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from env.traffic_gen import Packet, gen_embb_video


@dataclass
class BystanderArrivalModel:
    """Spawn 80-120 livestream UEs in a burst window around `trigger_time`.

    Profile:
        - Before trigger:                 baseline UEs (constant)
        - Ramp-up (trigger ± ramp_sec):   exponential rise to peak
        - Sustained:                      peak UEs for sustain_sec
        - Ramp-down:                      decay back to baseline
    """

    trigger_time_sec: float = 2.0
    baseline_ues: int = 30
    peak_ues_range: tuple[int, int] = (80, 120)
    ramp_sec: float = 1.0
    sustain_sec: float = 30.0
    decay_sec: float = 10.0
    per_ue_rate_mbps_range: tuple[float, float] = (2.0, 5.0)
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    # populated by initialize()
    _peak_ues: int = 0
    _ue_rates_mbps: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def initialize(self) -> None:
        """Sample the realised peak count and per-UE rates."""
        self._peak_ues = int(self.rng.integers(*self.peak_ues_range, endpoint=True))
        self._ue_rates_mbps = self.rng.uniform(*self.per_ue_rate_mbps_range, size=self._peak_ues)

    @property
    def peak_ues(self) -> int:
        return self._peak_ues

    def active_ue_count(self, sim_time: float) -> int:
        """Number of bystander UEs active at sim_time."""
        if self._peak_ues == 0:
            self.initialize()

        t_relative = sim_time - self.trigger_time_sec
        if t_relative < -self.ramp_sec:
            return self.baseline_ues
        if t_relative < 0:
            # Ramp-up
            f = (t_relative + self.ramp_sec) / self.ramp_sec   # in [0,1]
            return int(self.baseline_ues + f * (self._peak_ues - self.baseline_ues))
        if t_relative <= self.sustain_sec:
            return self._peak_ues
        # Decay
        f = max(0.0, 1.0 - (t_relative - self.sustain_sec) / self.decay_sec)
        return int(self.baseline_ues + f * (self._peak_ues - self.baseline_ues))

    def generate_packets(self, sim_time_window: tuple[float, float]) -> list[Packet]:
        """Generate eMBB video packets for all active bystander UEs in the window.

        Each UE produces a CBR-ish video stream at its sampled rate.
        """
        if self._peak_ues == 0:
            self.initialize()
        t_start, t_end = sim_time_window
        if t_end <= t_start:
            return []
        duration = t_end - t_start
        n_active = self.active_ue_count(t_start)

        all_packets: list[Packet] = []
        for ue_idx in range(min(n_active, self._peak_ues)):
            rate = float(self._ue_rates_mbps[ue_idx])
            ue_pkts = gen_embb_video(
                duration_sec=duration,
                rate_mbps=rate,
                cv=0.2,
                rng=self.rng,
            )
            # Shift arrivals into the window
            for p in ue_pkts:
                p.arrival_time += t_start
            all_packets.extend(ue_pkts)
        all_packets.sort(key=lambda p: p.arrival_time)
        return all_packets

    def aggregate_load_mbps(self, sim_time: float) -> float:
        """Instantaneous aggregate eMBB UL demand in Mbps."""
        if self._peak_ues == 0:
            self.initialize()
        n_active = self.active_ue_count(sim_time)
        if n_active <= 0:
            return 0.0
        # All UEs that are active use their sampled rate
        return float(self._ue_rates_mbps[:n_active].sum())
