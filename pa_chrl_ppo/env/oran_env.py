"""ORANEnv — Gymnasium-compatible environment for PA-CHRL-PPO.

Integrates all Week 2-3 modules into a TTI-level simulator:
    - Channel: SINR per UE per cell
    - Queue: M/G/1 per slice
    - Traffic: URLLC + eMBB generators
    - Phase: 5-state FSM (held fixed or trajectory-driven via reset opts)
    - AoI: stream-aware trackers
    - MEC: server budget + offload rules

Time hierarchy (sim — compressed per docs/08:305-336):
    1 episode = 1s of simulated time
    Manager action every T_H_sim = 10ms (Week 8+ wires hierarchical wrapper)
    Worker action every T_L_sim = 0.5ms = 1 TTI  ← env.step() unit

Action space (6-dim continuous, per docs/05:103-114):
    a[0] = Δr_min^URLLC   ∈ [-1, +1] → decoded ×0.1 → [-0.1, +0.1]
    a[1] = Δr_max^eMBB    ∈ [-1, +1] → decoded ×0.1
    a[2] = r_ded_ratio    ∈ [ 0,  1] → r_ded = min(0.2, r_ded_ratio × r_min)
    a[3..5] = w_C1, w_C2, w_C3 (Softmax later — env stores raw logits)

Note: δ (preemption) and x_k (MEC offload) are rule-based, NOT in this action.

Observation (Worker s_L, flattened — dim depends on K, F):
    Q_urllc, Q_eMBB                      (queue lengths, packets)
    HOL_urllc, HOL_eMBB                  (sec)
    PRB_alloc_urllc, PRB_alloc_eMBB      (fraction of P_TOTAL)
    SINR_per_ambulance                    (dB)
    arrival_rate_urllc, arrival_rate_eMBB
    mean_BLER_cell
    phase one-hot (5-dim)
    AoI_per_stream                        (sec)
    λ_local (5-dim, zeros until Week 9)
    rrm_budget (1-dim placeholder)
    LSTM 6-head outputs (zeros until Week 10)

Reference:
    - docs/08_implementation_notes.md TTI Simulation Loop
    - docs/05_agent_workflow.md State + Action + Reward
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.channel_model import (
    BaseStation,
    ChannelModel,
    capacity_per_prb_bps,
    thermal_noise_dbm,
)
from env.queue_model import MG1Queue, SliceQueueManager
from env.aoi_tracker import AoIStreamTracker, STREAM_TYPES, aoi_threshold_for_phase
from env.mec_model import MECServer
from env.phase_detector import Phase, PhaseDetector
from utils.config import (
    B_PRB,
    CMDP_D_J_PHI,
    D_BH,
    D_DET,
    D_FH,
    D_REF_URLLC,
    D_STOCH,
    MAC_TICKS_PER_WORKER,
    PHASE_ALPHA,
    PHASE_QOS,
    P_TOTAL,
    R_REF_EMBB_MBPS,
    SHANNON_ETA,
    TTI_SEC,
    get_phase_alpha,
    get_phase_thresholds,
)


# ============================================================
# Helpers
# ============================================================


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _expit(x: float) -> float:
    """Numerically stable sigmoid scalar."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# Streams tracked per ambulance (subset for Week 4 — full set in Week 8)
DEFAULT_AOI_STREAMS: tuple[str, ...] = (
    "HR_aggregated",
    "SpO2_aggregated",
    "ECG_waveform",
    "DENM",
)


@dataclass
class EnvConfig:
    """All tunables for a single-cell single-ambulance setup (Exp1 / Gate P2).

    The defaults preserve the easy single-phase scenario from Week 4. To run
    the hard-mission scenario (S1 collision burst + S2B bystander spike +
    phase trajectory), use :func:`hard_mission_config`.
    """

    K_ambulances: int = 1
    M_eMBB: int = 30
    num_streams: int = 4
    initial_phase: int = 3                  # default φ₃ SCENE for sanity
    episode_duration_sec: float = 1.0
    tti_sec: float = TTI_SEC
    # Traffic rates (steady state, before phase scaling)
    urllc_arrival_rate: float = 50.0        # DENM steady (pkt/s/ambulance)
    embb_arrival_rate: float = 1000.0       # per eMBB UE (pkt/s)
    urllc_packet_bits: int = 400 * 8        # 400B → 3200 bits
    embb_packet_bits: int = 1500 * 8        # MTU
    # Geometry — single-cell UMi, gNB fixed at origin (0,0) = local Cartesian
    # convergence point of 3 ambulances on Giải Phóng (đường hội tụ về BV
    # Bạch Mai, GPS 21.002966, 105.840780 — anchor cho lớp OSM/SUMO, W15;
    # KHÔNG dùng GPS trực tiếp trong env). R_cell=300m, no handover
    # (docs/03_architecture.md, REFERENCE_MAP M2.0).
    cell_radius_m: float = 300.0
    ambulance_speed_kmh: float = 40.0
    ambulance_start_distance_m: float | None = None  # None = random; set to fix
    # Channel
    bs_tx_power_dbm: float = 46.0           # macro default; reduce for hard mission
    sinr_clamp_max_db: float = 40.0         # cap to avoid log saturation
    sinr_clamp_min_db: float = -10.0
    # Phase Lagrangian placeholders (filled by HRL wrapper in Week 8/9)
    lambda_local: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    rrm_budget_hint: float = 0.6            # Manager hint for r_min^URLLC

    # ---- Hard-mission features (opt-in, default disabled) -----------------
    phase_trajectory: tuple[tuple[float, int], ...] | None = None
    urllc_burst_at_sec: float | None = None
    urllc_burst_duration_sec: float = 0.10
    urllc_burst_factor: float = 10.0
    enable_bystander: bool = False
    bystander_trigger_sec: float = 0.4
    bystander_peak_range: tuple[int, int] = (80, 120)
    bystander_per_ue_mbps: tuple[float, float] = (2.0, 5.0)
    # ---- Phase 2.1 reward (W05 refactor + post-critique restructure W12) ---
    # Reward is eMBB log-utility ONLY (single-term objective):
    #   r = α_e(φ) · U_eMBB(t),  U_eMBB = log(1 + R_eMBB / R_REF_EMBB_MBPS)
    # URLLC enforced via Lagrangian C1, C2 (LambdaState), NOT via reward penalty.
    # Pre-restructure form r = -α_U · L_URLLC + α_e · U_eMBB caused double-counting
    # with λ_1, λ_2 (W11 audit found λ_1, λ_2 stagnated). L_URLLC retained in info
    # dict for diagnostics only. See docs/13 §2.1, docs/05 #reward-rl.


# Map (current_phase, next_phase) → event name (per docs/03 phase FSM).
_NEXT_PHASE_EVENT: dict[tuple[Phase, Phase], str] = {
    (Phase.STANDBY, Phase.DISPATCH): "dispatch_call_received",
    (Phase.DISPATCH, Phase.SCENE): "arrived_at_scene",
    (Phase.SCENE, Phase.TRANSPORT): "patient_loaded",
    (Phase.TRANSPORT, Phase.RETURN): "arrived_at_hospital",
    (Phase.RETURN, Phase.STANDBY): "return_to_station",
}


def hard_mission_config(
    *,
    K_ambulances: int = 1,
    seed: int = 0,
) -> EnvConfig:
    """Pre-built "hard" scenario per docs/02 S1 + S2B and docs/03 phase FSM.

    Compressed to 1s sim time:
        0.00s  φ₁ STANDBY
        0.20s  φ₂ DISPATCH
        0.40s  φ₃ SCENE (+ bystander spike, eMBB jumps to 80-120 UEs)
        0.45s  URLLC burst window (DENM ×10 for 100ms)
        0.70s  φ₄ TRANSPORT
        0.95s  φ₅ RETURN

    Reductions vs easy config that force baselines to actually fight:
        - SINR clamp 40 → 30 dB; TX power 46 → 38 dBm (micro cell typical)
        - Ambulance starts at 100m and moves at 60 km/h → SINR drifts
        - URLLC burst ×10 at φ₃ → tail probability stressed
        - Bystander S2B 80-120 UEs at 2-5 Mbps → eMBB demand surge 200-500 Mbps
    """
    return EnvConfig(
        K_ambulances=K_ambulances,
        initial_phase=1,                            # overridden by trajectory[0]
        phase_trajectory=(
            (0.0,  1),
            (0.2,  2),
            (0.4,  3),
            (0.7,  4),
            (0.95, 5),
        ),
        # Aggressive URLLC burst so a static r_min=0.05 hint cannot absorb it
        urllc_burst_at_sec=0.45,
        urllc_burst_duration_sec=0.10,
        urllc_burst_factor=50.0,
        # Bystander S2B spike concurrent with φ₃
        enable_bystander=True,
        bystander_trigger_sec=0.4,
        bystander_peak_range=(80, 120),
        bystander_per_ue_mbps=(2.0, 5.0),
        # Hard channel — lower clamp + lower TX so capacity / PRB drops 5-10×.
        # SINR cap 15 dB ⇒ 0.75·360 kHz·log2(1+31.6) ≈ 1.36 Mbps/PRB.
        sinr_clamp_max_db=15.0,
        bs_tx_power_dbm=30.0,
        ambulance_speed_kmh=60.0,
        ambulance_start_distance_m=150.0,
        cell_radius_m=300.0,
        M_eMBB=30,
        urllc_arrival_rate=50.0,
        # Critical: tiny URLLC PRB budget at start (mirrors φ₁ STANDBY profile).
        # Static policy keeps this throughout the whole 1-second mission and
        # therefore cannot serve the φ₃ burst. PA-CHRL-PPO is expected to
        # raise r_min during φ₃ via the phase-aware action.
        # Calibration: at hint=0.02 with SINR clamp 15 dB and burst factor 50,
        # peak ρ ≈ 0.85 during φ₃ → mean D_e2e ≈ 1.5-2 ms on burst TTI
        # → mean(D_e2e > 1ms) ≈ 4-10% across full episode for Static.
        rrm_budget_hint=0.02,
    )


class ORANEnv(gym.Env):
    """Single-cell PA-CHRL-PPO TTI-level environment (Worker timescale)."""

    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None, seed: int | None = None):
        super().__init__()
        self.config = config or EnvConfig()
        self._seed = seed
        self.rng = np.random.default_rng(seed)

        # ---------------- Spaces ----------------
        # 6-dim action: Δr_min, Δr_max, r_ded_ratio, w_C1, w_C2, w_C3
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([+1.0, +1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        K = self.config.K_ambulances
        F = self.config.num_streams
        # Phase 1.1 formal Worker state s_t^L = 33 + 3K + F dims (docs/13 Phase 1.1)
        #   33-dim fixed block (queue, HOL, PRB_util, arr_rates, BLER, n_active,
        #                       phase one-hot, λ_local, rrm_budget, MEC util, n_bys,
        #                       AoI summary, LSTM 6-head, ETA_next, t_phi)
        #   3K per-ambulance block (SINR, distance, speed)
        #   F per-stream block (AoI per stream)
        # For K=1, F=4 → 33 + 3 + 4 = 40 dim.
        obs_dim = (
            # === Fixed 33-dim block ===
            2     # Q_urllc, Q_eMBB
            + 2   # HOL_urllc, HOL_eMBB
            + 3   # PRB ratios: r_min^URLLC, r_max^eMBB, r_ded^URLLC (Phase 1.1)
            + 2   # arrival_rate_urllc, arrival_rate_eMBB (normalized)
            + 1   # mean BLER
            + 5   # phase one-hot
            + 1   # t_phi (normalized time elapsed in current phase)
            + 1   # ETA_next (s, normalized)
            + 5   # λ_local (5 hard constraints)
            + 1   # rrm_budget hint (Manager → Worker)
            + 1   # u_MEC (MEC utilization)
            + 1   # n_bys (active bystander UE count, normalized)
            + 2   # mean AoI + max AoI (across F streams)
            + 6   # LSTM 6-head placeholder
            # = 33 fixed
            # === 3K per-ambulance block ===
            + 3 * K  # SINR_k (dB clamped, normalized), d_k (distance to BS), v_k (speed)
            # === F per-stream block ===
            + F   # AoI per stream
        )
        self._obs_dim = obs_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # ---------------- Persistent components ----------------
        self.base_station = BaseStation(
            cell_id=0, x=0.0, y=0.0, h=10.0, layer="micro",
            tx_power_dbm=self.config.bs_tx_power_dbm,
        )
        self.channel = ChannelModel(shadowing=True, rng=np.random.default_rng(seed))
        self.mec = MECServer()
        # Bystander spike model — built lazily in reset() if enabled
        self.bystander = None

        # ---------------- Per-episode state (set by reset) ----------------
        self.ambulance_pos: np.ndarray            # (K, 2)
        self.ambulance_vel: np.ndarray            # (K, 2)
        self.phase_dets: list[PhaseDetector]
        self.queues: SliceQueueManager
        self.aoi_trackers: dict[str, AoIStreamTracker]
        self.tti_idx: int
        self.sim_time: float
        self.r_min_urllc: float
        self.r_max_emBB: float
        self.r_ded_urllc: float
        self.last_sinr_db: np.ndarray
        self.last_bler: float
        # Diagnostics
        self.e2e_history: list[float]
        self.viol_history: list[bool]
        self.prb_alloc_history: list[tuple[int, int]]
        self.embb_mbps_history: list[float]
        self.c3_viol_history: list[bool]
        self.last_embb_mbps: float = 0.0

    # ----------------------------------------------------------------
    # Gym API
    # ----------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
            self.rng = np.random.default_rng(seed)
            self.channel.rng = np.random.default_rng(seed)

        K = self.config.K_ambulances
        # Ambulance start positions. Fixed-distance preset if requested
        # (used by hard-mission to put ambulance near the cell edge).
        if self.config.ambulance_start_distance_m is not None:
            radii = np.full(K, float(self.config.ambulance_start_distance_m))
        else:
            radii = self.rng.uniform(30.0, self.config.cell_radius_m, size=K)
        angles = self.rng.uniform(0.0, 2.0 * math.pi, size=K)
        self.ambulance_pos = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
        # Velocity vector in m/s
        speed_ms = self.config.ambulance_speed_kmh / 3.6
        head = self.rng.uniform(0.0, 2.0 * math.pi, size=K)
        self.ambulance_vel = speed_ms * np.stack([np.cos(head), np.sin(head)], axis=1)

        # Phase detectors. Initial phase comes from (in order of precedence):
        #   1) options["initial_phase"] (manual override)
        #   2) phase_trajectory[0][1] (hard-mission preset)
        #   3) config.initial_phase
        if options and "initial_phase" in options:
            phase = options["initial_phase"]
        elif self.config.phase_trajectory:
            phase = self.config.phase_trajectory[0][1]
        else:
            phase = self.config.initial_phase
        self.phase_dets = [
            PhaseDetector(current_phase=Phase(phase), reported_phase=Phase(phase),
                          rng=np.random.default_rng(self._seed if self._seed is not None else 0))
            for _ in range(K)
        ]
        self._trajectory_idx = 0 if self.config.phase_trajectory else -1

        # Bystander S2B model
        if self.config.enable_bystander:
            from env.bystander_traffic import BystanderArrivalModel
            self.bystander = BystanderArrivalModel(
                trigger_time_sec=self.config.bystander_trigger_sec,
                baseline_ues=self.config.M_eMBB,
                peak_ues_range=self.config.bystander_peak_range,
                per_ue_rate_mbps_range=self.config.bystander_per_ue_mbps,
                rng=np.random.default_rng(self._seed if self._seed is not None else 0),
            )
            self.bystander.initialize()
        else:
            self.bystander = None

        # Queues — start empty, μ comes from initial PRB hint
        self.queues = SliceQueueManager()
        self.queues.add(MG1Queue(name="urllc", arrival_rate=0.0,
                                  mean_packet_bits=self.config.urllc_packet_bits))
        self.queues.add(MG1Queue(name="eMBB", arrival_rate=0.0,
                                  mean_packet_bits=self.config.embb_packet_bits))

        # AoI trackers
        self.aoi_trackers = {
            sid: AoIStreamTracker.from_spec(sid) for sid in DEFAULT_AOI_STREAMS
        }

        # MAC ratios — initialise from Manager hint
        self.r_min_urllc = self.config.rrm_budget_hint
        self.r_max_emBB = 1.0 - self.r_min_urllc
        self.r_ded_urllc = 0.1

        # Sim time + diagnostics
        self.tti_idx = 0
        self.sim_time = 0.0
        self.last_sinr_db = np.full(K, 15.0)
        self.last_bler = 0.0
        self.e2e_history = []
        self.viol_history = []
        self.prb_alloc_history = []
        self.embb_mbps_history = []
        self.c3_viol_history = []
        self.last_embb_mbps = 0.0
        # Per-Worker-step constraint accumulator (5 hard, reset each Worker step in step())
        self._worker_c_accum = np.zeros(5, dtype=np.float64)
        self._worker_tick_count = 0
        # Last computed Worker-step c_vec + d_phi (for info dict)
        self._last_c_vec = np.zeros(5, dtype=np.float32)
        self._last_l_urllc: float = 0.0    # URLLC mean delay (D/D_ref) — diagnostics only
        # Initialize d_phi from current phase (so reset() returns valid info immediately)
        th0 = get_phase_thresholds(int(self.phase_dets[0].current_phase))
        self._last_d_phi = np.array(
            [th0["d1"], th0["d2"], th0["d3"], th0["d4"], th0["d5"]], dtype=np.float32
        )
        # Phase entry time tracker (for t_phi normalization in observation)
        self._phase_entry_time: float = 0.0
        self._last_observed_phase: int = int(self.phase_dets[0].current_phase)

        # First channel sample
        self._update_channel()
        self._update_queue_service_rates()

        return self._observe(), self._info()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """One Worker/xApp step = 10 ms = 20 MAC ticks (per O-RAN spec).

        xApp decides ONE RRMPolicyRatio (from `action`) at start of step,
        then O-DU MAC executes that action constant for 20 TTI internal
        ticks. Reward is the sum of per-TTI rewards across the window.

        Returns observation aggregated AT END of Worker step (post 20 ticks).
        See docs/13_methodology_walkthrough.md Phase 1.4 for full hierarchy.
        """
        # 1. Decode action ONCE per Worker step → RRMPolicyRatios
        self._apply_action(action)

        # 1b. Reset per-Worker-step constraint accumulator (aggregated across 20 MAC ticks)
        self._worker_c_accum = np.zeros(5, dtype=np.float64)
        self._worker_tick_count = 0

        # 2. Run MAC_TICKS_PER_WORKER (=20) internal MAC ticks with same action
        reward_accumulated = 0.0
        for _ in range(MAC_TICKS_PER_WORKER):
            reward_accumulated += self._mac_tick()
            if self.tti_idx >= self._max_tti_for_episode():
                break  # episode truncated mid-Worker-step (rare)

        # 3. Aggregate c_vec across MAC ticks → mean per-step constraint signal (5 hard)
        if self._worker_tick_count > 0:
            self._last_c_vec = (
                self._worker_c_accum / self._worker_tick_count
            ).astype(np.float32)
        else:
            self._last_c_vec = np.zeros(5, dtype=np.float32)
        # 4. Phase trajectory advance once per Worker step
        self._advance_phase_trajectory()
        # 5. Per-step phase threshold lookup (post-transition phase, docs/13 Phase 2.2)
        phase_now = int(self.phase_dets[0].current_phase)
        th = get_phase_thresholds(phase_now)
        self._last_d_phi = np.array(
            [th["d1"], th["d2"], th["d3"], th["d4"], th["d5"]], dtype=np.float32
        )

        terminated = False
        truncated = self.tti_idx >= self._max_tti_for_episode()
        return self._observe(), float(reward_accumulated), terminated, truncated, self._info()

    def _max_tti_for_episode(self) -> int:
        """Episode duration converted to integer TTI count (avoids float drift)."""
        return int(round(self.config.episode_duration_sec / self.config.tti_sec))

    def _mac_tick(self) -> float:
        """One MAC TTI (= 0.5 ms). Internal O-DU step — NOT exposed to xApp.

        Action is already decoded (set via _apply_action). MAC holds it
        constant. Returns the per-TTI reward (env-defined).
        """
        # Channel + traffic arrivals
        self._update_channel()
        n_urllc, _n_emBB = self._sample_arrivals()

        # MAC scheduling (service rates from current PRB ratios)
        self._update_queue_service_rates()

        # Queue evolution → D_e2e for any URLLC packet this TTI
        d_e2e = self._compute_e2e_delay()
        phase_idx = int(self.phase_dets[0].current_phase)
        d_max_phi = float(PHASE_QOS[phase_idx]["D_max"])
        viol = d_e2e > d_max_phi

        # HARQ + bookkeeping
        self.last_bler = self._sample_bler()

        # eMBB throughput (Mbps) — capacity-limited when queue unstable
        self.last_embb_mbps = self._compute_embb_throughput_mbps()
        r_min_embb_phi = float(CMDP_D_J_PHI[phase_idx]["d3_embb_mbps"])
        embb_gap_mbps = r_min_embb_phi - self.last_embb_mbps
        embb_deficit_mbps = max(0.0, embb_gap_mbps)
        c3_viol = embb_gap_mbps > 0.0

        # ---------------- Phase 2.1 Restructured Reward (post-critique, docs/13 §2.1) ----------------
        # r_t = α_e(φ) · U_eMBB(t)   (eMBB log-utility ONLY)
        #   U_eMBB = log(1 + R_eMBB / R_REF_EMBB_MBPS)   (bounded, R_REF = 100 Mbps)
        # URLLC enforced via Lagrangian C1, C2 in LambdaState — NOT in reward.
        # Removes double-counting with λ_1, λ_2 that caused dual stagnation (W11 audit).
        # Single-term reward: only α_eMBB is used. α_URLLC is intentionally NOT
        # applied to the reward (URLLC enforced via Lagrangian λ_1, λ_2). The
        # urllc weight is ignored here by design (post-restructure 2026-05-26).
        _, alpha_e = get_phase_alpha(phase_idx)
        l_urllc = d_e2e / D_REF_URLLC                       # diagnostics only, exported via info dict
        self._last_l_urllc = float(l_urllc)
        u_embb = math.log(1.0 + self.last_embb_mbps / R_REF_EMBB_MBPS)
        reward = alpha_e * u_embb

        # ---------------- AoI tracking for C4, C5 (Worker-step aggregation) ----------------
        # Aggregated vital streams = HR + SpO2 (LCFS+drop_old per docs/04)
        aoi_max_phi = float(PHASE_QOS[phase_idx]["AoI_max_HR"])
        eps_aoi_phi = float(PHASE_QOS[phase_idx]["eps_aoi"])
        agg_aoi_vals = [
            self.aoi_trackers["HR_aggregated"].current_aoi(self.sim_time),
            self.aoi_trackers["SpO2_aggregated"].current_aoi(self.sim_time),
        ]
        aoi_mean_tick = float(np.mean(agg_aoi_vals)) if agg_aoi_vals else 0.0
        aoi_tail_viol = float(np.mean([1.0 if a > aoi_max_phi else 0.0 for a in agg_aoi_vals]))

        # ---------------- Per-MAC-tick c_vec accumulator (5 hard constraints) ----------------
        # Aggregate across MAC ticks → reported as Worker-step c_vec in info dict.
        # c1 = mean D_e2e (seconds); c2 = tail viol fraction;
        # c3 = signed eMBB gap R_min^phi - R_eMBB (Mbps);
        # c4 = mean AoI (seconds); c5 = AoI tail viol fraction.
        self._worker_c_accum[0] += float(d_e2e)
        self._worker_c_accum[1] += float(1.0 if viol else 0.0)
        self._worker_c_accum[2] += float(embb_gap_mbps)
        self._worker_c_accum[3] += float(aoi_mean_tick)
        self._worker_c_accum[4] += float(aoi_tail_viol)
        self._worker_tick_count += 1

        # State update + history
        self._advance_ambulance_positions()
        self._update_aoi_trackers(n_urllc)
        self.tti_idx += 1
        self.sim_time += self.config.tti_sec

        self.e2e_history.append(d_e2e)
        self.viol_history.append(viol)
        self.embb_mbps_history.append(self.last_embb_mbps)
        self.c3_viol_history.append(c3_viol)
        prb_urllc, prb_emBB = self._prb_allocation()
        self.prb_alloc_history.append((prb_urllc, prb_emBB))

        return float(reward)

    def close(self) -> None:
        pass

    # ----------------------------------------------------------------
    # Action decoder
    # ----------------------------------------------------------------

    def _apply_action(self, action: np.ndarray) -> None:
        """Decode action into r_min, r_max, r_ded (and stash w_intra)."""
        a = np.asarray(action, dtype=np.float64)
        delta_r_min = float(np.clip(a[0], -1.0, 1.0)) * 0.1
        delta_r_max = float(np.clip(a[1], -1.0, 1.0)) * 0.1
        r_ded_ratio = float(np.clip(a[2], 0.0, 1.0))

        self.r_min_urllc = float(np.clip(self.r_min_urllc + delta_r_min, 0.0, 1.0))
        self.r_max_emBB = float(np.clip(self.r_max_emBB + delta_r_max, 0.0, 1.0))
        # Enforce r_min + r_max ≤ 1 (C6)
        if self.r_min_urllc + self.r_max_emBB > 1.0:
            excess = self.r_min_urllc + self.r_max_emBB - 1.0
            self.r_max_emBB = max(0.0, self.r_max_emBB - excess)
        # C7: r_ded ≤ r_min (by design via r_ded_ratio)
        self.r_ded_urllc = min(0.2, r_ded_ratio * self.r_min_urllc)
        # w_intra stored implicitly via action; full use in Week 9

    def _prb_allocation(self) -> tuple[int, int]:
        prb_urllc = int(self.r_min_urllc * P_TOTAL)
        prb_emBB = int(self.r_max_emBB * P_TOTAL)
        # Ensure sum doesn't exceed (guard against rounding)
        if prb_urllc + prb_emBB > P_TOTAL:
            prb_emBB = P_TOTAL - prb_urllc
        return prb_urllc, max(prb_emBB, 0)

    # ----------------------------------------------------------------
    # Channel + arrivals + queues
    # ----------------------------------------------------------------

    def _update_channel(self) -> None:
        """Compute SINR per ambulance against the single cell."""
        K = self.config.K_ambulances
        n_dbm = thermal_noise_dbm(B_PRB)
        sinrs = np.empty(K)
        for k in range(K):
            rx = self.channel.receive_power_dbm(
                (float(self.ambulance_pos[k, 0]), float(self.ambulance_pos[k, 1])),
                self.base_station,
                tx_power_dbm=self.base_station.tx_power_dbm,
            )
            sinrs[k] = rx - n_dbm
        self.last_sinr_db = np.clip(
            sinrs, self.config.sinr_clamp_min_db, self.config.sinr_clamp_max_db
        )

    def _advance_phase_trajectory(self) -> None:
        """If a phase_trajectory is configured, fire transitions at scheduled times."""
        traj = self.config.phase_trajectory
        if not traj:
            return
        while (
            self._trajectory_idx + 1 < len(traj)
            and traj[self._trajectory_idx + 1][0] <= self.sim_time
        ):
            self._trajectory_idx += 1
            target = Phase(traj[self._trajectory_idx][1])
            for det in self.phase_dets:
                event = _NEXT_PHASE_EVENT.get((det.current_phase, target))
                if event is not None:
                    det.trigger(event, self.sim_time)
        # Phase entry time tracking (used by _observe() for t_phi normalization)
        cur = int(self.phase_dets[0].current_phase)
        if cur != self._last_observed_phase:
            self._phase_entry_time = self.sim_time
            self._last_observed_phase = cur

    def _urllc_burst_active(self) -> bool:
        bs = self.config.urllc_burst_at_sec
        if bs is None:
            return False
        return bs <= self.sim_time < bs + self.config.urllc_burst_duration_sec

    def _sample_bler(self) -> float:
        """BLER from SINR (simple logistic approximation, 3GPP-style)."""
        sinr = float(np.mean(self.last_sinr_db))
        # ~10% BLER at SINR=0dB, asymptote ~0 at high SINR
        bler = 1.0 / (1.0 + math.exp(0.5 * (sinr - 2.0)))
        return float(np.clip(bler, 1e-4, 0.5))

    def _sample_arrivals(self) -> tuple[int, int]:
        """Sample per-TTI arrivals (Poisson) with burst + bystander spikes."""
        K = self.config.K_ambulances

        # URLLC: apply burst factor if inside the burst window
        burst_factor = self.config.urllc_burst_factor if self._urllc_burst_active() else 1.0
        eff_urllc_rate = self.config.urllc_arrival_rate * burst_factor
        lam_urllc_per_tti = eff_urllc_rate * self.config.tti_sec * K

        # eMBB: M_eMBB → bystander.active_ue_count(sim_time) if S2B enabled
        if self.bystander is not None:
            n_active = self.bystander.active_ue_count(self.sim_time)
        else:
            n_active = self.config.M_eMBB
        eff_embb_total_rate = self.config.embb_arrival_rate * n_active
        lam_emBB_per_tti = eff_embb_total_rate * self.config.tti_sec

        n_urllc = int(self.rng.poisson(max(lam_urllc_per_tti, 0.0)))
        n_emBB = int(self.rng.poisson(max(lam_emBB_per_tti, 0.0)))

        # Update queue arrival-rate estimate (per-second basis)
        self.queues["urllc"].set_arrival_rate(eff_urllc_rate * K)
        self.queues["eMBB"].set_arrival_rate(eff_embb_total_rate)
        return n_urllc, n_emBB

    def _update_queue_service_rates(self) -> None:
        prb_urllc, prb_emBB = self._prb_allocation()
        sinr_avg = float(np.mean(self.last_sinr_db))
        c_per_prb = capacity_per_prb_bps(sinr_avg, eta=SHANNON_ETA)
        self.queues["urllc"].update_service_rate(prb_urllc, c_per_prb)
        self.queues["eMBB"].update_service_rate(prb_emBB, c_per_prb)

    def _compute_embb_throughput_mbps(self) -> float:
        """eMBB realized throughput in Mbps.

        Stable queue: served rate = arrival rate (no backlog accumulation).
        Unstable queue (ρ ≥ 0.9): capacity-limited at service_rate.
        """
        q = self.queues["eMBB"]
        if q.service_rate <= 0:
            return 0.0
        served_pkts_per_sec = min(q.arrival_rate, q.service_rate)
        return served_pkts_per_sec * q.mean_packet_bits / 1e6

    def _compute_e2e_delay(self) -> float:
        """Return current mean D_e2e for URLLC slice in this TTI."""
        q = self.queues["urllc"]
        if not q.is_stable:
            return float(PHASE_QOS[3]["D_max"]) * 2.0   # clamp at 2× D_max if unstable

        hol = q.hol_delay()                            # ~queue + service
        d_tx = q.mean_service_time - D_STOCH           # subtract stoch component
        d_queue = q.expected_queue_delay()
        d_e2e = D_DET + d_tx + d_queue + D_FH + D_BH
        return float(d_e2e)

    # ----------------------------------------------------------------
    # AoI + position
    # ----------------------------------------------------------------

    def _update_aoi_trackers(self, n_urllc: int) -> None:
        """Inject one HR + one SpO2 sample per TTI if URLLC traffic arrives."""
        if n_urllc <= 0:
            return
        for sid in self.aoi_trackers:
            self.aoi_trackers[sid].arrive(gen_time=self.sim_time)
            self.aoi_trackers[sid].deliver_next(sim_time=self.sim_time + self.config.tti_sec)

    def _advance_ambulance_positions(self) -> None:
        self.ambulance_pos = self.ambulance_pos + self.ambulance_vel * self.config.tti_sec
        # Bounce off the cell edge (cheap reflection)
        r2 = (self.ambulance_pos ** 2).sum(axis=1)
        out = r2 > self.config.cell_radius_m ** 2
        if out.any():
            self.ambulance_vel[out] *= -1.0

    # ----------------------------------------------------------------
    # Observation + info
    # ----------------------------------------------------------------

    def _observe(self) -> np.ndarray:
        """Return Phase 1.1 formal Worker state s_t^L (40-dim for K=1, F=4).

        Layout (matches docs/13 Phase 1.1):
            == Fixed 33-dim block ==
            [0:2]   Q_urllc, Q_eMBB                      (queue arrival rates, normalized)
            [2:4]   HOL_urllc, HOL_eMBB                   (head-of-line delay, ms)
            [4:7]   r_min^URLLC, r_max^eMBB, r_ded^URLLC  (PRB ratios)
            [7:9]   arr_rate_urllc, arr_rate_eMBB         (per-second rate, normalized)
            [9]     mean BLER
            [10:15] phase one-hot (φ_1..φ_5)
            [15]    t_phi (normalized time elapsed in current phase)
            [16]    ETA_next (s, normalized; 0 if unknown)
            [17:22] λ_local (5 hard constraints)
            [22]    rrm_budget hint (Manager → Worker)
            [23]    u_MEC (MEC utilization)
            [24]    n_bys (active bystander UE count, normalized by M_eMBB)
            [25:27] mean AoI + max AoI (s, across F streams)
            [27:33] LSTM 6-head placeholder (zeros until Week 10)
            == 3K per-ambulance block ==
            [33:33+K]      SINR_k (dB normalized)
            [33+K:33+2K]   d_k (distance to serving O-RU, normalized by cell_radius)
            [33+2K:33+3K]  v_k (speed m/s, normalized by 60 m/s)
            == F per-stream block ==
            [33+3K:33+3K+F]  AoI per stream (s)

        Normalisations chosen to keep components O(1) for PPO stability.
        """
        K = self.config.K_ambulances
        F = self.config.num_streams
        phase_idx = int(self.phase_dets[0].current_phase)
        phase_oh = np.zeros(5, dtype=np.float32)
        phase_oh[phase_idx - 1] = 1.0
        aoi_vec = np.array(
            [t.current_aoi(self.sim_time) for t in self.aoi_trackers.values()],
            dtype=np.float32,
        )

        # Queue load: utilization ρ = λ/μ ∈ [0, 1] (NOT duplicate of arrival rate)
        rho_urllc = float(np.clip(self.queues["urllc"].rho, 0.0, 1.0))
        rho_emBB = float(np.clip(self.queues["eMBB"].rho, 0.0, 1.0))
        # Arrival rates (per second, normalized differently from ρ)
        arr_urllc = float(self.queues["urllc"].arrival_rate) / 1e3
        arr_emBB = float(self.queues["eMBB"].arrival_rate) / 1e4
        # Clip HOL: unstable queue returns +inf; cap at sane upper bounds.
        hol_urllc_ms = min(float(self.queues["urllc"].hol_delay()) * 1e3, 100.0)
        hol_emBB_ms = min(float(self.queues["eMBB"].hol_delay()) * 1e3, 1000.0)

        # PRB ratios (Phase 1.1: r_min, r_max, r_ded directly — KHÔNG split per slice)
        prb_ratios = np.array(
            [self.r_min_urllc, self.r_max_emBB, self.r_ded_urllc],
            dtype=np.float32,
        )

        # Time-in-phase (normalized by episode duration as proxy)
        phase_elapsed = self.sim_time - self._phase_entry_time
        t_phi = float(np.clip(phase_elapsed / self.config.episode_duration_sec, 0.0, 1.0))

        # ETA to next phase transition (0 if unknown / no trajectory)
        eta_next = self._compute_eta_next()
        eta_next_norm = float(np.clip(eta_next, 0.0, 10.0)) / 10.0  # clip 10s, normalize

        # MEC utilization (property, not method)
        mec_util = float(self.mec.utilization) if hasattr(self.mec, "utilization") else 0.0

        # Bystander UE count (normalized by baseline M_eMBB)
        if self.bystander is not None:
            n_bys = float(self.bystander.active_ue_count(self.sim_time)) / max(self.config.M_eMBB, 1)
        else:
            n_bys = 1.0  # baseline M_eMBB → ratio 1.0

        # AoI summary (mean + max across F streams)
        aoi_mean = float(aoi_vec.mean()) if aoi_vec.size > 0 else 0.0
        aoi_max = float(aoi_vec.max()) if aoi_vec.size > 0 else 0.0

        # Per-ambulance kinematics (sinr already cached, recompute distance + speed)
        sinr_norm = self.last_sinr_db.astype(np.float32) / 40.0     # [-10, 40] dB → [-0.25, 1.0]
        amb_dist = np.linalg.norm(self.ambulance_pos, axis=1) / max(self.config.cell_radius_m, 1.0)
        amb_speed = np.linalg.norm(self.ambulance_vel, axis=1) / 60.0  # normalize by 60 m/s cap

        obs = np.concatenate(
            [
                # === Fixed 33-dim block ===
                np.array([rho_urllc, rho_emBB,                       # [0:2] queue utilization
                          hol_urllc_ms, hol_emBB_ms], dtype=np.float32),  # [2:4] HOL ms
                prb_ratios,                                          # [4:7] r_min, r_max, r_ded
                np.array([arr_urllc, arr_emBB,                       # [7:9] arrival rates
                          float(self.last_bler)], dtype=np.float32), # [9]   BLER
                phase_oh,                                            # [10:15] phase one-hot
                np.array([t_phi, eta_next_norm], dtype=np.float32),  # [15:17]
                np.asarray(self.config.lambda_local, dtype=np.float32),  # [17:22] λ_local
                np.array([self.config.rrm_budget_hint, mec_util, n_bys,  # [22:25]
                          aoi_mean, aoi_max], dtype=np.float32),     # [25:27]
                np.zeros(6, dtype=np.float32),                       # [27:33] LSTM placeholder
                # === 3K per-amb block ===
                sinr_norm,                                           # [33:33+K]   SINR
                amb_dist.astype(np.float32),                         # [33+K:33+2K] distance
                amb_speed.astype(np.float32),                        # [33+2K:33+3K] speed
                # === F per-stream block ===
                aoi_vec,                                             # [33+3K:33+3K+F]
            ]
        )
        # Final safety: replace any residual NaN/inf with 0
        return np.nan_to_num(obs.astype(np.float32), nan=0.0, posinf=1e3, neginf=-1e3)

    def _compute_eta_next(self) -> float:
        """ETA (seconds) until next phase transition. Returns 0 if unknown."""
        traj = self.config.phase_trajectory
        if not traj or self._trajectory_idx < 0:
            return 0.0
        next_idx = self._trajectory_idx + 1
        if next_idx >= len(traj):
            return 0.0
        return max(0.0, float(traj[next_idx][0]) - self.sim_time)

    def _info(self) -> dict[str, Any]:
        phase_idx = int(self.phase_dets[0].current_phase)
        return {
            "tti": self.tti_idx,
            "sim_time": self.sim_time,
            "phase": phase_idx,
            "phase_name": Phase(phase_idx).name,
            "phase_now": phase_idx,                      # alias for docs/13 compatibility
            "r_min_urllc": self.r_min_urllc,
            "r_max_eMBB": self.r_max_emBB,
            "r_ded_urllc": self.r_ded_urllc,
            "sinr_db": float(np.mean(self.last_sinr_db)),
            "mean_BLER": self.last_bler,
            "urllc_stable": self.queues["urllc"].is_stable,
            "eMBB_stable": self.queues["eMBB"].is_stable,
            # Phase 2.2.1 — 5 hard constraint signals aggregated per Worker step
            # c_vec[0] = mean D_e2e (s)         | d_phi[0] = D_max^φ (s)
            # c_vec[1] = URLLC tail viol frac    | d_phi[1] = ε^φ
            # c_vec[2] = signed eMBB gap (Mbps) | d_phi[2] = 0
            # c_vec[3] = mean AoI (s, agg)       | d_phi[3] = AoI_max^φ_HR (s)
            # c_vec[4] = AoI tail viol frac      | d_phi[4] = ε_AoI^φ
            "c_vec": self._last_c_vec.copy(),
            "d_phi": self._last_d_phi.copy(),
            # Phase 2.1 reward restructure (post-critique W12): URLLC mean delay
            # exported for diagnostics — NOT used in reward computation. Reward is
            # alpha_e * log1p(R_eMBB/R_REF) only; URLLC enforced via λ_1, λ_2.
            "l_urllc_mean": float(self._last_l_urllc),
            # Reviewer M2 (Gemini W02, 2026-05-27): expose M/G/1 queue diagnostics
            # so reviewers can audit Pollaczek-Khinchine formula application.
            # Returns dict {lambda, mu, rho, E_S, E_S2, E_D_queue, HOL, stable}.
            # See docs/13 §1.3 service-time distribution + env/queue_model.py:62-73.
            "queue_diag_urllc": self.queues["urllc"].summary(),
            "queue_diag_embb": self.queues["eMBB"].summary(),
        }

    # ----------------------------------------------------------------
    # Sanity helpers
    # ----------------------------------------------------------------

    def episode_violation_rate(self) -> float:
        if not self.viol_history:
            return 0.0
        return sum(self.viol_history) / len(self.viol_history)

    def mean_e2e_ms(self) -> float:
        if not self.e2e_history:
            return 0.0
        return float(np.mean(self.e2e_history) * 1e3)

    def mean_embb_mbps(self) -> float:
        if not self.embb_mbps_history:
            return 0.0
        return float(np.mean(self.embb_mbps_history))

    def c3_violation_rate(self) -> float:
        if not self.c3_viol_history:
            return 0.0
        return sum(self.c3_viol_history) / len(self.c3_viol_history)
