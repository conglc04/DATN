"""MEC offloading model.

Implements:
    - D_MEC^k = D_upload + D_comp = L_k / R_k^UL + (W_k · L_k) / f_k^MEC
    - F_MEC total CPU budget per O-DU edge server (10 GHz)
    - Resource constraint (C10): Σ_k x_k · f_k^MEC ≤ F_MEC
    - Rule-based offload decision (xApp Algorithm 2)

Reference:
    - docs/04_data_flow.md#mec-offloading-model (lines 208-239)
    - utils.config.F_MEC = 10 GHz
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from utils.config import F_MEC

TaskType = Literal[
    "video_analytics",
    "vital_sign_process",
    "denm_cam",
    "image_recognition",
]


# W_k = required CPU cycles per bit (estimated from typical workloads)
TASK_PROFILE: dict[str, dict[str, float]] = {
    "video_analytics":     {"W_cycles_per_bit": 1200.0, "size_bits_typical": 4_000_000},
    "vital_sign_process":  {"W_cycles_per_bit":   50.0, "size_bits_typical":    16_000},
    "denm_cam":            {"W_cycles_per_bit":   10.0, "size_bits_typical":     8_000},
    "image_recognition":   {"W_cycles_per_bit":  800.0, "size_bits_typical":   400_000},
}


@dataclass(slots=True)
class MECTask:
    """A compute task offloadable to MEC."""

    task_id: int
    task_type: TaskType
    size_bits: int                          # L_k (uplink payload)
    W_cycles_per_bit: float                 # CPU cycles per bit
    f_mec_allocation_hz: float = 1e9        # f_k^MEC (default 1 GHz/task)
    uplink_rate_bps: float = 50e6           # R_k^UL (default 50 Mbps)
    f_local_hz: float = 0.5e9               # f_local (UE CPU, much smaller)

    def delay_offloaded(self) -> float:
        """D_MEC = L/R_UL + W·L / f_MEC."""
        d_upload = self.size_bits / max(self.uplink_rate_bps, 1.0)
        d_comp = (self.W_cycles_per_bit * self.size_bits) / max(self.f_mec_allocation_hz, 1.0)
        return d_upload + d_comp

    def delay_local(self) -> float:
        """D_local = W·L / f_local."""
        return (self.W_cycles_per_bit * self.size_bits) / max(self.f_local_hz, 1.0)


@dataclass
class MECServer:
    """O-DU edge server with shared CPU budget.

    Reference: utils.config.F_MEC = 10 GHz total
    """

    f_total_hz: float = F_MEC               # 10 GHz
    active_tasks: list[MECTask] = field(default_factory=list)

    @property
    def utilization(self) -> float:
        used = sum(t.f_mec_allocation_hz for t in self.active_tasks)
        return min(used / max(self.f_total_hz, 1.0), 1.0)

    @property
    def available_hz(self) -> float:
        used = sum(t.f_mec_allocation_hz for t in self.active_tasks)
        return max(self.f_total_hz - used, 0.0)

    def can_admit(self, task: MECTask) -> bool:
        """C10 admission check."""
        return self.available_hz >= task.f_mec_allocation_hz

    def admit(self, task: MECTask) -> bool:
        if not self.can_admit(task):
            return False
        self.active_tasks.append(task)
        return True

    def release(self, task_id: int) -> bool:
        for i, t in enumerate(self.active_tasks):
            if t.task_id == task_id:
                self.active_tasks.pop(i)
                return True
        return False

    def reset(self) -> None:
        self.active_tasks.clear()


# ============================================================
# Rule-based offload decision (xApp Algorithm 2)
# Reference: docs/04_data_flow.md:230-239
# ============================================================


def offload_decision(
    task_type: TaskType,
    sinr_db: float,
    mec_load: float,
) -> bool:
    """Return True if xApp should offload (x_k = 1) per rule table."""
    if task_type == "video_analytics":
        return sinr_db > 5.0 and mec_load < 0.8
    if task_type == "image_recognition":
        return mec_load < 0.7
    if task_type in ("vital_sign_process", "denm_cam"):
        return False
    return False


def total_mec_delay(
    task: MECTask, server: MECServer, sinr_db: float, force_decision: bool | None = None
) -> tuple[float, bool, str]:
    """Compute the realized delay applying rule + admission.

    Returns (delay_sec, offloaded_flag, reason).
    """
    decision = (
        offload_decision(task.task_type, sinr_db, server.utilization)
        if force_decision is None
        else force_decision
    )
    if not decision:
        return task.delay_local(), False, "rule_says_local"
    if not server.can_admit(task):
        return task.delay_local(), False, "mec_overloaded"
    server.admit(task)
    return task.delay_offloaded(), True, "offloaded"
