"""Ambulance 5-phase FSM with explicit signaling.

Implements:
    - 5-state FSM (STANDBY → DISPATCH → SCENE → TRANSPORT → RETURN)
    - Deterministic event-triggered transitions
    - Sudden-event auto-trigger to φ₃ (collision / cardiac alarm)
    - ETA-based predicted-next-phase
    - Pre-tightening trigger when ETA_to_next < 30s
    - Phase signaling delay model (UE FSM → xApp ~10-50ms with drop)

Reference:
    - docs/03_architecture.md#phase-fsm
    - docs/08_implementation_notes.md (Phase Signaling Delay Model)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Literal

import numpy as np

from utils.config import PHASE_QOS, PRE_TIGHTEN_ETA


class Phase(IntEnum):
    STANDBY = 1
    DISPATCH = 2
    SCENE = 3
    TRANSPORT = 4
    RETURN = 5


TransitionEvent = Literal[
    "dispatch_call_received",
    "arrived_at_scene",
    "patient_loaded",
    "arrived_at_hospital",
    "return_to_station",
    "collision_shock",
    "cardiac_alarm_critical",
]


# Normal forward transitions (state → event → next_state)
NORMAL_TRANSITIONS: dict[tuple[Phase, str], Phase] = {
    (Phase.STANDBY, "dispatch_call_received"): Phase.DISPATCH,
    (Phase.DISPATCH, "arrived_at_scene"): Phase.SCENE,
    (Phase.SCENE, "patient_loaded"): Phase.TRANSPORT,
    (Phase.TRANSPORT, "arrived_at_hospital"): Phase.RETURN,
    (Phase.RETURN, "return_to_station"): Phase.STANDBY,
}

# Emergency overrides — fire from ANY state to SCENE
SUDDEN_EVENTS: set[str] = {"collision_shock", "cardiac_alarm_critical"}


@dataclass
class PhaseDetector:
    """Per-ambulance Phase FSM with signaling delay simulator.

    State held at UE Mission Management System (MMS). Reads:
        - current_phase: actual phase at UE
        - reported_phase: phase observed at xApp (delayed)
        - last_transition_time: sim-time at last switch
        - eta_to_next_phase: from navigation system (None if unknown)
    """

    current_phase: Phase = Phase.STANDBY
    reported_phase: Phase = Phase.STANDBY
    last_transition_time: float = 0.0
    last_report_time: float = 0.0
    eta_to_next_phase: float | None = None    # seconds from MMS
    # Signaling delay model (from docs/08 Phase Signaling Delay Model)
    delay_min_sec: float = 10e-3
    delay_max_sec: float = 50e-3
    drop_probability: float = 0.001
    # Sudden-event fast path (MAC CE, single TTI)
    sudden_event_delay_sec: float = 0.5e-3
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    # transition history for analysis
    transition_history: list[tuple[float, Phase, Phase, str]] = field(default_factory=list)

    # ----------------------------------------------------------------
    # FSM core
    # ----------------------------------------------------------------

    def trigger(self, event: str, sim_time: float) -> bool:
        """Apply event. Returns True if a transition fired."""
        if event in SUDDEN_EVENTS:
            return self._sudden_transition(sim_time, event)

        key = (self.current_phase, event)
        if key not in NORMAL_TRANSITIONS:
            return False
        next_phase = NORMAL_TRANSITIONS[key]
        self._apply_transition(sim_time, next_phase, event, sudden=False)
        return True

    def _sudden_transition(self, sim_time: float, event: str) -> bool:
        if self.current_phase == Phase.SCENE:
            return False
        self._apply_transition(sim_time, Phase.SCENE, event, sudden=True)
        return True

    def _apply_transition(
        self, sim_time: float, next_phase: Phase, event: str, sudden: bool
    ) -> None:
        prev = self.current_phase
        self.current_phase = next_phase
        self.last_transition_time = sim_time
        self.eta_to_next_phase = None  # invalidated; navigation provides new ETA
        self.transition_history.append((sim_time, prev, next_phase, event))

        # Schedule the report time at xApp (fast path for sudden events).
        if sudden:
            delay = self.sudden_event_delay_sec
        else:
            if self.rng.random() < self.drop_probability:
                # Dropped — xApp keeps the stale "reported_phase" indefinitely
                # (until a future non-dropped transition arrives). Push the
                # report time to +inf so observed_phase() never picks up this
                # transition on its own.
                self.last_report_time = float("inf")
                return
            delay = self.rng.uniform(self.delay_min_sec, self.delay_max_sec)
        self.last_report_time = sim_time + delay

    # ----------------------------------------------------------------
    # xApp-side observation (delayed)
    # ----------------------------------------------------------------

    def observed_phase(self, sim_time: float) -> Phase:
        """Phase as seen by xApp at sim_time (after signaling delay)."""
        if sim_time >= self.last_report_time and self.reported_phase != self.current_phase:
            self.reported_phase = self.current_phase
        return self.reported_phase

    # ----------------------------------------------------------------
    # ETA + pre-tightening
    # ----------------------------------------------------------------

    def set_eta_to_next_phase(self, eta_sec: float | None) -> None:
        """MMS pushes navigation ETA when known."""
        if eta_sec is not None and eta_sec < 0:
            raise ValueError(f"ETA must be non-negative, got {eta_sec}")
        self.eta_to_next_phase = eta_sec

    def predicted_next_phase(self) -> Phase | None:
        """Walk the normal-transition graph: which state comes after the current one?"""
        nexts = [
            nxt for (state, _evt), nxt in NORMAL_TRANSITIONS.items() if state == self.current_phase
        ]
        return nexts[0] if nexts else None

    def should_pretighten(self, threshold_sec: float = PRE_TIGHTEN_ETA) -> bool:
        """Per docs/03:106 — if ETA < 30s and next ∈ {φ₃, φ₄}, pre-tighten."""
        if self.eta_to_next_phase is None:
            return False
        if self.eta_to_next_phase >= threshold_sec:
            return False
        nxt = self.predicted_next_phase()
        return nxt in (Phase.SCENE, Phase.TRANSPORT)

    # ----------------------------------------------------------------
    # Convenience: effective D_max + AoI_max
    # ----------------------------------------------------------------

    def effective_qos(self) -> dict[str, float]:
        """QoS budget the xApp should currently honor.

        Returns dict from PHASE_QOS, looking up the next phase if pre-tightening
        is active.
        """
        phase_idx: int = int(self.current_phase)
        if self.should_pretighten():
            nxt = self.predicted_next_phase()
            if nxt is not None:
                phase_idx = int(nxt)
        return dict(PHASE_QOS[phase_idx])

    def __repr__(self) -> str:
        return (
            f"PhaseDetector(phase={self.current_phase.name}, "
            f"reported={self.reported_phase.name}, "
            f"ETA={self.eta_to_next_phase}, "
            f"transitions={len(self.transition_history)})"
        )
