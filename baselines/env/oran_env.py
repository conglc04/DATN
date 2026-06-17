"""ORANEnv — Gymnasium-compatible environment for PPO.

Integrates all Week 2-3 modules into a TTI-level simulator:
    - Channel: SINR per UE per cell
    - Queue: M/G/1 per slice
    - Traffic: URLLC + eMBB generators
    - Severity: 5-level patient-urgency tier (exogenous, fixed per episode)
    - AoI: stream-aware trackers

Time hierarchy (sim — compressed per docs/08:305-336):
    1 episode = 1s of simulated time
    Manager action every T_H_sim = 10ms (Week 8+ wires hierarchical wrapper)
    Worker action every T_L_sim = 0.5ms = 1 TTI  ← env.step() unit

Action space (6-dim for K=1, 7-dim for K>=2, per docs/05:103-114 + B5
severity_k epic 2026-06-15):
    a[0] = Δr_min^URLLC   ∈ [-1, +1] → decoded ×0.1 → [-0.1, +0.1]
    a[1] = Δr_max^eMBB    ∈ [-1, +1] → decoded ×0.1
    a[2] = r_ded_ratio    ∈ [ 0,  1] → r_ded = min(0.2, r_ded_ratio × r_min)
    a[3..5] = w_C1, w_C2, w_C3 (Softmax later — env stores raw logits)
    a[6] = β priority temperature (K>=2 only) → sigmoid → [BETA_MIN, BETA_MAX],
           drives the intra-slice Π_feasible PRB split across ambulances.

Observation (Worker s_L, flattened, obs_dim = 20 + 10K + F):
    == Fixed 20-dim block ==
    Q_urllc, Q_eMBB                      (queue lengths, packets)
    HOL_urllc, HOL_eMBB                  (sec)
    PRB_alloc_urllc, PRB_alloc_eMBB, PRB_ded_urllc (fraction of P_TOTAL)
    arrival_rate_urllc, arrival_rate_eMBB
    mean_BLER_cell
    severity_ref one-hot (5-dim)
    λ_local^C3 (shared, 1-dim)
    rrm_budget (1-dim placeholder), n_bys, mean AoI, max AoI
    == 10K per-ambulance block (interleaved) ==
    SINR_k, d_k, v_k, delay_norm_k, AoI_norm_k, severity_k_norm,
    λ_local^C1_k, λ_local^C2_k, λ_local^C4_k, λ_local^C5_k
    == F per-stream block ==
    AoI_per_stream                        (sec)

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
from env.aoi_tracker import AoIStreamTracker
from utils.config import (
    AMB_AOI_NORM_OFFSET,
    AMB_DELAY_NORM_OFFSET,
    AMB_DIST_OFFSET,
    AMB_LAMBDA_C1_OFFSET,
    AMB_LAMBDA_C2_OFFSET,
    AMB_LAMBDA_C4_OFFSET,
    AMB_LAMBDA_C5_OFFSET,
    AMB_SEVERITY_NORM_OFFSET,
    AMB_SINR_OFFSET,
    AMB_SPEED_OFFSET,
    B_PRB,
    B_RRM_MAX,
    B_RRM_MIN,
    BETA_MAX,
    BETA_MIN,
    CMDP_D_J_SEVERITY,
    D_BH,
    D_DET,
    D_FH,
    D_REF_URLLC,
    D_STOCH,
    INTRA_SLICE_KAPPA,
    LAMBDA_C3_SHARED_OBS_INDEX,
    MAC_TICKS_PER_WORKER,
    OBS_AOI_MAX_IDX,
    OBS_AOI_MEAN_IDX,
    OBS_ARR_EMBB_IDX,
    OBS_ARR_URLLC_IDX,
    OBS_BLER_IDX,
    OBS_FIXED_BLOCK_LEN,
    OBS_HOL_EMBB_IDX,
    OBS_HOL_URLLC_IDX,
    OBS_LAMBDA_C3_IDX,
    OBS_N_BYS_IDX,
    OBS_PER_AMB_BLOCK_LEN,
    OBS_R_DED_URLLC_IDX,
    OBS_R_MAX_EMBB_IDX,
    OBS_R_MIN_URLLC_IDX,
    OBS_RHO_EMBB_IDX,
    OBS_RHO_URLLC_IDX,
    OBS_RMIN_ANCHOR_IDX,
    OBS_SEVERITY_OH_IDX,
    OBS_SEVERITY_OH_LEN,
    PRB_MIN_QOS,
    RHO_URGENCY_TIEBREAK,
    SEVERITY_OH_OBS_INDEX,
    SEVERITY_QOS,
    P_TOTAL,
    R_REF_EMBB_MBPS,
    SHANNON_ETA,
    TTI_SEC,
    build_d_phi_vector,
    get_severity_alpha,
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


# Single consolidated AoI-tracked stream per ambulance (F=1, 2026-06-14
# stream consolidation): replaces the prior 4-stream HR/SpO2/ECG/DENM split.
# Conceptually a periodic status bundle (🔴 declared envelope: ~500-1500B,
# ~10-20Hz) carrying both patient-monitoring and V2X telemetry as one
# aggregated URLLC payload per report-cycle — see docs/02_requirements.md.
DEFAULT_AOI_STREAMS: tuple[str, ...] = (
    "ambulance_status",
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
    num_streams: int = 1
    initial_severity: int = 5               # default IMMEDIATE (tightest) for sanity
    # Severity_k epic (2026-06-15): each of the K ambulances carries an
    # INDEPENDENT severity_k in {1..5}, sampled independently and fixed for the
    # episode (severity_per_amb). severity_ref := max(severity_per_amb) drives
    # all SHARED quantities (alpha_e reward weight, C3 R_min, severity one-hot,
    # info["severity"]). For training diversity, set sample_severity=True to
    # draw a fresh independent level per ambulance each reset() from
    # severity_sample_weights. When False, every ambulance uses initial_severity
    # (=> severity_per_amb = [initial_severity]*K, severity_ref = initial_severity,
    # exact K=1 legacy behaviour).
    sample_severity: bool = False
    severity_sample_weights: tuple[float, ...] = (0.20, 0.20, 0.20, 0.20, 0.20)
    episode_duration_sec: float = 1.0
    tti_sec: float = TTI_SEC
    # Traffic rates (steady state, before phase scaling)
    urllc_arrival_rate: float = 50.0        # ambulance_status steady (pkt/s/ambulance)
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
    # λ_local dual variables live in the agent's LambdaState; the env caches
    # the latest snapshot via set_lambda_local() (used by _observe() and the
    # Π_feasible urgency tiebreaker). The training loop additionally overlays
    # λ_local onto the returned obs via utils.obs.overlay_lambda_local (single
    # source, applied by every solver driver) so the policy sees the value
    # computed for the SAME decision step (the env's cached copy lags by one
    # step at the start of each Worker decision).
    rrm_budget_hint: float = 0.6            # Manager hint for r_min^URLLC

    # ---- Hard-mission features (opt-in, default disabled) -----------------
    urllc_burst_at_sec: float | None = None
    urllc_burst_duration_sec: float = 0.10
    urllc_burst_factor: float = 10.0
    enable_bystander: bool = False
    bystander_trigger_sec: float = 0.4
    bystander_peak_range: tuple[int, int] = (80, 120)
    bystander_per_ue_mbps: tuple[float, float] = (2.0, 5.0)
    # ---- Phase 2.1 reward (W05 refactor + post-critique restructure W12) ---
    # Reward is eMBB log-utility ONLY (single-term objective):
    #   r = α_e(sev) · U_eMBB(t),  U_eMBB = log(1 + R_eMBB / R_REF_EMBB_MBPS)
    # URLLC enforced via Lagrangian C1, C2 (LambdaState), NOT via reward penalty.
    # Pre-restructure form r = -α_U · L_URLLC + α_e · U_eMBB caused double-counting
    # with λ_1, λ_2 (W11 audit found λ_1, λ_2 stagnated). L_URLLC retained in info
    # dict for diagnostics only. See docs/13 §2.1, docs/05 #reward-rl.

    def __post_init__(self) -> None:
        # Guard against a degenerate K=0 problem: the (4K+1)-dim constraint
        # vectors would collapse to a C3-only (1,)-dim vector that silently
        # "passes" while modelling no ambulance at all. (audit 2026-06-16)
        if not isinstance(self.K_ambulances, int) or self.K_ambulances < 1:
            raise ValueError(f"K_ambulances must be an int >= 1; got {self.K_ambulances!r}")
        if not isinstance(self.num_streams, int) or self.num_streams < 1:
            raise ValueError(f"num_streams must be an int >= 1; got {self.num_streams!r}")
        if not (1 <= self.initial_severity <= 5):
            raise ValueError(f"initial_severity must be in 1..5; got {self.initial_severity}")


def hard_mission_config(
    *,
    K_ambulances: int = 1,
    seed: int = 0,
) -> EnvConfig:
    """Pre-built "hard" scenario per docs/02 S1 + S2B.

    Fixed IMMEDIATE severity (severity 5 — tightest QoS) for the whole 1s episode,
    plus the channel/traffic stressors that force solvers to actually fight:
        0.45s  URLLC burst window (DENM ×50 for 100ms)
        0.40s  bystander S2B spike (eMBB jumps to 80-120 UEs)

    Reductions vs easy config:
        - SINR clamp 40 → 15 dB; TX power 46 → 30 dBm (micro cell typical)
        - Ambulance starts at 150m and moves at 60 km/h → SINR drifts
        - URLLC burst ×50 → tail probability stressed
        - Bystander S2B 80-120 UEs at 2-5 Mbps → eMBB demand surge 200-500 Mbps
    """
    return EnvConfig(
        K_ambulances=K_ambulances,
        initial_severity=5,                         # IMMEDIATE — tightest QoS
        # Aggressive URLLC burst so a static r_min=0.05 hint cannot absorb it
        urllc_burst_at_sec=0.45,
        urllc_burst_duration_sec=0.10,
        urllc_burst_factor=50.0,
        # Bystander S2B spike (eMBB demand surge)
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
        # Critical: tiny URLLC PRB budget at start. A static policy keeps this
        # throughout the whole 1-second mission and therefore cannot serve the
        # URLLC burst. PPO is expected to raise r_min in response to the burst.
        # Calibration: at hint=0.02 with SINR clamp 15 dB and burst factor 50,
        # peak ρ ≈ 0.85 during the burst → mean D_e2e ≈ 1.5-2 ms on burst TTI
        # → mean(D_e2e > 1ms) ≈ 4-10% across full episode for Static.
        rrm_budget_hint=0.02,
    )


class ORANEnv(gym.Env):
    """Single-cell PPO TTI-level environment (Worker timescale)."""

    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None, seed: int | None = None):
        super().__init__()
        self.config = config or EnvConfig()
        self._seed = seed
        self.rng = np.random.default_rng(seed)

        # ---------------- Spaces ----------------
        K = self.config.K_ambulances
        F = self.config.num_streams
        # Action space: 6-dim for K=1 (unchanged legacy), 7-dim for K>=2
        # (adds a[6] = β priority temperature, B5 severity_k epic 2026-06-15).
        # Δr_min, Δr_max, r_ded_ratio, w_C1, w_C2, w_C3[, β]
        if K >= 2:
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0, 0.0, 0.0, 0.0, 0.0, -3.0], dtype=np.float32),
                high=np.array([+1.0, +1.0, 1.0, 1.0, 1.0, 1.0, +3.0], dtype=np.float32),
                dtype=np.float32,
            )
        else:
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([+1.0, +1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )

        # Formal Worker state s_t^L = 20 + 10K + F dims (per-ambulance
        # severity_k epic 2026-06-15):
        #   20-dim fixed block (queue, HOL, PRB_util, arr_rates, BLER,
        #                       severity_ref one-hot, λ_local^C3 shared,
        #                       rrm_budget, n_bys, AoI summary)
        #   10K per-ambulance block, INTERLEAVED per ambulance:
        #     SINR_k, d_k, v_k, delay_norm_k, AoI_norm_k, severity_k_norm,
        #     λ_local^C1_k, λ_local^C2_k, λ_local^C4_k, λ_local^C5_k
        #   F per-stream block (AoI per stream)
        # For K=1, F=1 → 20 + 10 + 1 = 31 dim (was 30).
        obs_dim = OBS_FIXED_BLOCK_LEN + OBS_PER_AMB_BLOCK_LEN * K + F
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
        # Bystander spike model — built lazily in reset() if enabled
        self.bystander = None

        # ---------------- Per-episode state (set by reset) ----------------
        self.ambulance_pos: np.ndarray            # (K, 2)
        self.ambulance_vel: np.ndarray            # (K, 2)
        self.severity_per_amb: np.ndarray         # (K,) independent severity_k 1..5 (fixed/episode)
        self.severity: int                        # severity_ref = max(severity_per_amb), drives SHARED quantities
        self._lambda_local: np.ndarray            # (4K+1,) λ_local, set via set_lambda_local()
        self._beta: float                         # priority temperature for Π_feasible (K>=2)
        self.queues: SliceQueueManager
        self.aoi_trackers: list[dict[str, AoIStreamTracker]]
        self.tti_idx: int
        self.sim_time: float
        self.r_min_urllc: float
        self.r_min_urllc_anchor: float      # Manager setpoint; obs[16]; fixed per window
        self.r_max_emBB: float
        self.r_ded_urllc: float
        self.last_sinr_db: np.ndarray
        self.last_bler: float
        # PRB allocation cache — set once per step() after _apply_action(),
        # read by _info() to avoid re-calling impure _prb_allocation/_prb_split_intra_slice.
        self._last_prb_urllc: int
        self._last_prb_embb: int
        self._last_prb_per_amb: np.ndarray  # (K,) int64
        # Feasibility bounds for set_rrm_budget() — computed at reset()
        self._feasible_rrm_floor: float
        self._feasible_rrm_cap: float
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

        # Per-ambulance severity_k (exogenous patient urgency, independent per
        # ambulance, fixed for the whole episode — B5 severity_k epic
        # 2026-06-15). Precedence:
        #   1) options["severity_per_amb"] (manual override, shape (K,))
        #   2) options["initial_severity"] (manual override, broadcast to all K)
        #   3) sampled independently per ambulance from severity_sample_weights
        #      (if sample_severity)
        #   4) config.initial_severity (broadcast to all K)
        # severity_ref := max(severity_per_amb) drives all SHARED quantities
        # (alpha_e reward weight, C3 R_min, severity one-hot, info["severity"]).
        # At K=1 this reduces exactly to the legacy scalar severity.
        if options and "severity_per_amb" in options:
            severity_per_amb = np.asarray(options["severity_per_amb"], dtype=np.int64)
            if severity_per_amb.shape != (K,):
                raise ValueError(
                    f"options['severity_per_amb'] shape {severity_per_amb.shape} != ({K},)"
                )
        elif options and "initial_severity" in options:
            severity_per_amb = np.full(K, int(options["initial_severity"]), dtype=np.int64)
        elif self.config.sample_severity:
            w = np.asarray(self.config.severity_sample_weights, dtype=np.float64)
            severity_per_amb = self.rng.choice(
                np.arange(1, 6), size=K, p=w / w.sum()
            ).astype(np.int64)
        else:
            severity_per_amb = np.full(K, int(self.config.initial_severity), dtype=np.int64)
        for sev in severity_per_amb:
            if int(sev) not in SEVERITY_QOS:
                raise ValueError(f"Invalid severity {sev}; must be 1..5")
        self.severity_per_amb = severity_per_amb
        self.severity = int(severity_per_amb.max())  # severity_ref

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

        # Queues — start empty, μ comes from initial PRB hint.
        # URLLC is split per-ambulance (urllc_0..urllc_{K-1}) so each
        # ambulance's D_e2e can be observed independently (2026-06-14 fix).
        # eMBB stays a single pooled queue (bystander traffic, not per-ambulance).
        self.queues = SliceQueueManager()
        for k in range(K):
            self.queues.add(MG1Queue(name=f"urllc_{k}", arrival_rate=0.0,
                                      mean_packet_bits=self.config.urllc_packet_bits))
        self.queues.add(MG1Queue(name="eMBB", arrival_rate=0.0,
                                  mean_packet_bits=self.config.embb_packet_bits))

        # AoI trackers — one dict-of-streams per ambulance (2026-06-14 fix)
        self.aoi_trackers = [
            {sid: AoIStreamTracker.from_spec(sid) for sid in DEFAULT_AOI_STREAMS}
            for _ in range(K)
        ]

        # MAC ratios — initialise from Manager hint
        self.r_min_urllc = self.config.rrm_budget_hint
        self.r_min_urllc_anchor = self.r_min_urllc   # Manager setpoint anchor; obs[16]
        self.r_max_emBB = 1.0 - self.r_min_urllc
        self.r_ded_urllc = 0.1

        # Feasibility bounds for set_rrm_budget() (conservative, recomputed each reset).
        # Floor: K × min-PRBs-per-ambulance at SINR=0dB with ×5 QoS safety margin.
        # Cap:   ensures enough PRBs remain for max-severity eMBB floor at SINR=0dB.
        sinr_0db_cap_bps = B_PRB * SHANNON_ETA * math.log2(2.0)   # ≈ 270 kbps/PRB at 0dB
        safety_factor = 5.0
        need_bps = self.config.urllc_arrival_rate * self.config.urllc_packet_bits * safety_factor
        min_prb_per_amb = math.ceil(need_bps / max(sinr_0db_cap_bps, 1.0))
        self._feasible_rrm_floor = min(
            B_RRM_MAX,
            K * min_prb_per_amb / max(P_TOTAL, 1),
        )
        max_sev = max(CMDP_D_J_SEVERITY.keys())
        d3_mbps = float(CMDP_D_J_SEVERITY[max_sev]["d3_embb_mbps"])
        min_embb_prb = math.ceil(d3_mbps * 1e6 / max(sinr_0db_cap_bps, 1.0))
        self._feasible_rrm_cap = max(
            B_RRM_MIN,
            1.0 - min_embb_prb / max(P_TOTAL, 1),
        )

        # PRB allocation cache — zeros until first step()
        self._last_prb_urllc = 0
        self._last_prb_embb = 0
        self._last_prb_per_amb = np.zeros(K, dtype=np.int64)

        # Sim time + diagnostics
        self.tti_idx = 0
        self.sim_time = 0.0
        self.last_sinr_db = np.full(K, 15.0)
        self.last_bler = 0.0
        # Per-ambulance D_e2e / AoI cache, refreshed every MAC tick — read by
        # _observe() for delay_norm_k / AoI_norm_k (2026-06-14 fix).
        self._last_d_e2e_per_amb = np.zeros(K, dtype=np.float64)
        self._last_aoi_per_amb = np.zeros(K, dtype=np.float64)
        self.e2e_history = []
        self.viol_history = []
        self.prb_alloc_history = []
        self.embb_mbps_history = []
        self.c3_viol_history = []
        self.last_embb_mbps = 0.0
        # Per-Worker-step constraint accumulator ((4K+1)-dim, reset each
        # Worker step in step()) — B5 severity_k epic 2026-06-15 layout:
        #   [C1_0..C1_{K-1}, C2_0..C2_{K-1}, C4_0..C4_{K-1}, C5_0..C5_{K-1}, C3_shared]
        n_c = 4 * K + 1
        self._worker_c_accum = np.zeros(n_c, dtype=np.float64)
        self._worker_tick_count = 0
        # Last computed Worker-step c_vec + d_phi (for info dict)
        self._last_c_vec = np.zeros(n_c, dtype=np.float32)
        self._last_l_urllc: float = 0.0    # URLLC mean delay (D/D_ref) — diagnostics only
        # Initialize d_phi from severity_per_amb (so reset() returns valid info immediately)
        self._last_d_phi = build_d_phi_vector(self.severity_per_amb).astype(np.float32)

        # λ_local — env-internal storage, default zeros; overwritten by
        # set_lambda_local() each Worker step (train.py / solver drivers).
        self._lambda_local = np.zeros(n_c, dtype=np.float64)
        # β priority temperature for Π_feasible (K=1: unused, softmax([x])=[1.0]
        # always ⟹ K=1-preserving regardless of β; default BETA_MIN).
        self._beta = BETA_MIN

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

        # 1b. Cache PRB allocation for this step (r_min_urllc is fixed after _apply_action;
        #     must NOT be recomputed inside _info() since state may have advanced).
        self._last_prb_urllc, self._last_prb_embb = self._prb_allocation()
        self._last_prb_per_amb = self._prb_split_intra_slice(self._last_prb_urllc)

        # 1c. Reset per-Worker-step constraint accumulator (aggregated across 20 MAC ticks)
        n_c = 4 * self.config.K_ambulances + 1
        self._worker_c_accum = np.zeros(n_c, dtype=np.float64)
        self._worker_tick_count = 0

        # 2. Run MAC_TICKS_PER_WORKER (=20) internal MAC ticks with same action
        reward_accumulated = 0.0
        for _ in range(MAC_TICKS_PER_WORKER):
            reward_accumulated += self._mac_tick()
            if self.tti_idx >= self._max_tti_for_episode():
                break  # episode truncated mid-Worker-step (rare)

        # 3. Aggregate c_vec across MAC ticks → mean per-step constraint signal ((4K+1)-dim)
        if self._worker_tick_count > 0:
            self._last_c_vec = (
                self._worker_c_accum / self._worker_tick_count
            ).astype(np.float32)
        else:
            self._last_c_vec = np.zeros(n_c, dtype=np.float32)
        # 4. Per-step severity threshold lookup (severity fixed/episode, docs/13 Phase 2.2)
        self._last_d_phi = build_d_phi_vector(self.severity_per_amb).astype(np.float32)

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
        n_urllc_per_amb, _n_emBB = self._sample_arrivals()

        # MAC scheduling (service rates from current PRB ratios)
        self._update_queue_service_rates()

        # Queue evolution → D_e2e for each ambulance's URLLC stream this TTI.
        # Per-ambulance C1/C2 violation thresholds use severity_per_amb[k]
        # (each ambulance held to its OWN severity's QoS budget — B5 epic).
        K = self.config.K_ambulances
        d_e2e_per_amb = self._compute_e2e_delay_per_amb()
        d_max_phi_per_amb = np.array(
            [SEVERITY_QOS[int(s)]["D_max"] for s in self.severity_per_amb], dtype=np.float64
        )
        viol_per_amb = d_e2e_per_amb > d_max_phi_per_amb
        d_e2e = float(np.mean(d_e2e_per_amb))     # diagnostics use the K-mean
        viol = bool(np.mean(viol_per_amb) > 0.0)  # any-violation flag (history list)

        # HARQ + bookkeeping
        self.last_bler = self._sample_bler()

        # eMBB throughput (Mbps) — capacity-limited when queue unstable.
        # SHARED C3 quantity: R_min floor + alpha_e keyed by severity_ref
        # (= max(severity_per_amb)), per locked design decision.
        sev_ref = self.severity
        self.last_embb_mbps = self._compute_embb_throughput_mbps()
        r_min_embb_phi = float(CMDP_D_J_SEVERITY[sev_ref]["d3_embb_mbps"])
        embb_gap_mbps = r_min_embb_phi - self.last_embb_mbps
        embb_deficit_mbps = max(0.0, embb_gap_mbps)
        c3_viol = embb_gap_mbps > 0.0

        # ---------------- Phase 2.1 Restructured Reward (post-critique, docs/13 §2.1) ----------------
        # r_t = α_e(sev_ref) · U_eMBB(t)   (eMBB log-utility ONLY)
        #   U_eMBB = log(1 + R_eMBB / R_REF_EMBB_MBPS)   (bounded, R_REF = 100 Mbps)
        # URLLC enforced via Lagrangian C1, C2 in LambdaState — NOT in reward.
        # Removes double-counting with λ_1, λ_2 that caused dual stagnation (W11 audit).
        # Single-term reward: only α_eMBB is used. α_URLLC is intentionally NOT
        # applied to the reward (URLLC enforced via Lagrangian λ_1, λ_2). The
        # urllc weight is ignored here by design (post-restructure 2026-05-26).
        _, alpha_e = get_severity_alpha(sev_ref)
        l_urllc = d_e2e / D_REF_URLLC                       # diagnostics only, exported via info dict
        self._last_l_urllc = float(l_urllc)
        u_embb = math.log(1.0 + self.last_embb_mbps / R_REF_EMBB_MBPS)
        reward = alpha_e * u_embb

        # ---------------- AoI tracking for C4, C5 (Worker-step aggregation) ----------------
        # Single consolidated per-ambulance AoI stream (F=1, LCFS+drop_old per docs/04),
        # one tracker per ambulance. Per-ambulance C4/C5 thresholds use
        # severity_per_amb[k] (B5 epic).
        aoi_max_phi_per_amb = np.array(
            [SEVERITY_QOS[int(s)]["AoI_max"] for s in self.severity_per_amb], dtype=np.float64
        )
        aoi_per_amb = np.array(
            [t["ambulance_status"].current_aoi(self.sim_time) for t in self.aoi_trackers],
            dtype=np.float64,
        )
        aoi_viol_per_amb = aoi_per_amb > aoi_max_phi_per_amb
        aoi_mean_tick = float(np.mean(aoi_per_amb))
        aoi_tail_viol = float(np.mean(aoi_viol_per_amb) > 0.0)

        # Cache per-ambulance values for _observe()'s delay_norm_k / AoI_norm_k
        self._last_d_e2e_per_amb = d_e2e_per_amb
        self._last_aoi_per_amb = aoi_per_amb

        # ---------------- Per-MAC-tick c_vec accumulator ((4K+1)-dim, B5 epic) ----------------
        # Aggregate across MAC ticks → reported as Worker-step c_vec in info dict.
        # Layout: [C1_0..C1_{K-1}, C2_0..C2_{K-1}, C4_0..C4_{K-1}, C5_0..C5_{K-1}, C3_shared]
        #   C1_k = D_e2e_k (seconds); C2_k = viol_k (0/1);
        #   C4_k = AoI_k (seconds);   C5_k = AoI_viol_k (0/1);
        #   C3_shared = signed eMBB gap R_min^sev_ref - R_eMBB (Mbps).
        # At K=1 this is the permutation [0,1,3,4,2] of the legacy 5-dim
        # [C1,C2,C3,C4,C5] — exact numeric preservation.
        self._worker_c_accum[0:K] += d_e2e_per_amb
        self._worker_c_accum[K:2 * K] += viol_per_amb.astype(np.float64)
        self._worker_c_accum[2 * K:3 * K] += aoi_per_amb
        self._worker_c_accum[3 * K:4 * K] += aoi_viol_per_amb.astype(np.float64)
        self._worker_c_accum[4 * K] += embb_gap_mbps
        self._worker_tick_count += 1

        # State update + history
        self._advance_ambulance_positions()
        self._update_aoi_trackers(n_urllc_per_amb)
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
        """Decode action into r_min, r_max, r_ded (and β for K>=2)."""
        a = np.asarray(action, dtype=np.float64)
        delta_r_min = float(np.clip(a[0], -1.0, 1.0)) * 0.1
        delta_r_max = float(np.clip(a[1], -1.0, 1.0)) * 0.1
        r_ded_ratio = float(np.clip(a[2], 0.0, 1.0))

        self.r_min_urllc = float(np.clip(self.r_min_urllc + delta_r_min, 0.0, 1.0))
        self.r_max_emBB = float(np.clip(self.r_max_emBB + delta_r_max, 0.0, 1.0))
        self._renormalize_prb_ratios()   # enforce r_min + r_max ≤ 1 (C6)
        # C7: r_ded ≤ r_min (by design via r_ded_ratio)
        self.r_ded_urllc = min(0.2, r_ded_ratio * self.r_min_urllc)
        # w_intra stored implicitly via action; full use in Week 9

        # β priority temperature (K>=2 only, B5 severity_k epic 2026-06-15):
        # a[6] → sigmoid → [BETA_MIN, BETA_MAX], drives Π_feasible PRB split.
        # K=1: softmax([x])=[1.0] always, so β has no numeric effect — keep
        # at BETA_MIN for exact K=1 preservation.
        if self.config.K_ambulances >= 2 and a.shape[0] >= 7:
            self._beta = BETA_MIN + (BETA_MAX - BETA_MIN) * _expit(float(a[6]))
        else:
            self._beta = BETA_MIN

    def _renormalize_prb_ratios(self) -> None:
        """Enforce r_min_urllc + r_max_emBB ≤ 1 by trimming r_max_emBB (C6)."""
        if self.r_min_urllc + self.r_max_emBB > 1.0:
            excess = self.r_min_urllc + self.r_max_emBB - 1.0
            self.r_max_emBB = max(0.0, self.r_max_emBB - excess)

    def set_rrm_budget(self, b_rrm: float) -> None:
        """Re-anchor r_min_urllc to Manager setpoint at the start of a Manager window.

        Two-tier clipping:
          1. [B_RRM_MIN, B_RRM_MAX] — guaranteed by decode_manager_action upstream.
          2. [feasible_rrm_floor, feasible_rrm_cap] — computed at reset() per K/QoS.
        The tighter of the two bounds applies.  For typical K=1 scenarios, tier-1 is
        always binding; tier-2 only tightens at large K or very low SINR.
        Must be called BEFORE the Worker loop for each Manager window.
        """
        lo = max(B_RRM_MIN, self._feasible_rrm_floor)
        hi = min(B_RRM_MAX, self._feasible_rrm_cap)
        hi = max(hi, lo)   # safety: ensure hi ≥ lo
        clipped = float(np.clip(b_rrm, lo, hi))
        self.r_min_urllc = clipped
        self.r_min_urllc_anchor = clipped
        self._renormalize_prb_ratios()

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

    def _sample_arrivals(self) -> tuple[np.ndarray, int]:
        """Sample per-TTI arrivals (Poisson) with burst + bystander spikes.

        URLLC arrivals are drawn independently per ambulance (K draws of
        Poisson(eff_urllc_rate * tti_sec)) so each ambulance's queue/AoI
        tracker can be fed separately — statistically equivalent in aggregate
        to the prior single Poisson(rate*K) draw, but enables per-ambulance
        attribution (2026-06-14 fix).
        """
        K = self.config.K_ambulances

        # URLLC: apply burst factor if inside the burst window
        burst_factor = self.config.urllc_burst_factor if self._urllc_burst_active() else 1.0
        eff_urllc_rate = self.config.urllc_arrival_rate * burst_factor
        lam_urllc_per_tti = eff_urllc_rate * self.config.tti_sec

        # eMBB: M_eMBB → bystander.active_ue_count(sim_time) if S2B enabled
        if self.bystander is not None:
            n_active = self.bystander.active_ue_count(self.sim_time)
        else:
            n_active = self.config.M_eMBB
        eff_embb_total_rate = self.config.embb_arrival_rate * n_active
        lam_emBB_per_tti = eff_embb_total_rate * self.config.tti_sec

        n_urllc_per_amb = self.rng.poisson(max(lam_urllc_per_tti, 0.0), size=K).astype(np.int64)
        n_emBB = int(self.rng.poisson(max(lam_emBB_per_tti, 0.0)))

        # Update per-ambulance queue arrival-rate estimate (per-second basis)
        for k in range(K):
            self.queues[f"urllc_{k}"].set_arrival_rate(eff_urllc_rate)
        self.queues["eMBB"].set_arrival_rate(eff_embb_total_rate)
        return n_urllc_per_amb, n_emBB

    def _update_queue_service_rates(self) -> None:
        """Split prb_urllc across K ambulances via intra-slice Π_feasible.

        Each ambulance's per-PRB capacity is derived from its own SINR
        (`self.last_sinr_db[k]`), so a worse channel ⇒ lower service rate ⇒
        higher D_e2e_k — the signal `delay_norm_k` exposes (2026-06-14 fix).
        eMBB stays a single pooled queue served at the cell-average SINR.
        """
        K = self.config.K_ambulances
        prb_urllc, prb_emBB = self._prb_allocation()
        sinr_avg = float(np.mean(self.last_sinr_db))
        c_per_prb_avg = capacity_per_prb_bps(sinr_avg, eta=SHANNON_ETA)

        prb_per_amb = self._prb_split_intra_slice(prb_urllc)
        for k in range(K):
            c_per_prb_k = capacity_per_prb_bps(float(self.last_sinr_db[k]), eta=SHANNON_ETA)
            self.queues[f"urllc_{k}"].update_service_rate(int(prb_per_amb[k]), c_per_prb_k)

        self.queues["eMBB"].update_service_rate(prb_emBB, c_per_prb_avg)

    def _prb_split_intra_slice(self, prb_urllc: int) -> np.ndarray:
        """Intra-slice Π_feasible PRB split across K ambulances (B5 epic).

        b = max(floor(κ·B_U/K), PRB_MIN_QOS); feasibility fallback b = B_U//K
        if K·b > B_U. Remainder S = B_U - K·b is distributed via
        w = softmax(β·(severity_per_amb/5) + δ·ũ), δ = ρ·β, where ũ is the
        per-ambulance C1 λ_local urgency tiebreaker normalized to [0,1].

        Severity is NORMALIZED to [0.2, 1.0] (÷5, same convention as obs
        severity_k_norm) BEFORE entering the softmax (audit 2026-06-16) so β acts
        as a pure global gain on a unit-scale priority — decoupled from the raw
        1..5 magnitude. β ∈ [BETA_MIN, BETA_MAX]; BETA_MIN>0 guarantees a minimum
        severity ordering (agent cannot fully flatten priority).

        At K=1, softmax([x]) = [1.0] always ⟹ PRB_0 = b + S = B_U regardless
        of β/severity/urgency — exact K=1 preservation.
        """
        K = self.config.K_ambulances
        B_U = int(prb_urllc)
        if K == 0:
            return np.zeros(0, dtype=np.int64)

        b = max(int(math.floor(INTRA_SLICE_KAPPA * B_U / K)), PRB_MIN_QOS)
        if K * b > B_U:
            b = B_U // K
        S = B_U - K * b
        if S <= 0:
            return np.full(K, b, dtype=np.int64)

        lam_c1 = self._lambda_local[0:K]
        max_lam_c1 = float(np.max(lam_c1))
        u_tilde = lam_c1 / max_lam_c1 if max_lam_c1 > 0.0 else np.zeros(K, dtype=np.float64)

        beta = self._beta
        delta = RHO_URGENCY_TIEBREAK * beta
        sev_norm = self.severity_per_amb.astype(np.float64) / 5.0   # [0.2,1.0], β = global gain
        logits = beta * sev_norm + delta * u_tilde
        w = _softmax(logits)

        shares = np.floor(S * w).astype(np.int64)
        remainder = S - int(shares.sum())
        if remainder > 0:
            fracs = S * w - shares
            order = np.argsort(-fracs)
            for i in range(remainder):
                shares[order[i % K]] += 1

        return np.full(K, b, dtype=np.int64) + shares

    def set_lambda_local(self, lambda_local: np.ndarray) -> None:
        """Store the (4K+1,)-dim λ_local vector (B5 epic, 2026-06-15).

        Called by train.py / solver drivers each Worker step. Feeds both
        _observe()'s per-ambulance + shared λ_local slots and the
        Π_feasible urgency tiebreaker in _prb_split_intra_slice().
        """
        K = self.config.K_ambulances
        arr = np.asarray(lambda_local, dtype=np.float64)
        if arr.shape != (4 * K + 1,):
            raise ValueError(f"lambda_local shape {arr.shape} != ({4 * K + 1},)")
        self._lambda_local = arr

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

    def _compute_e2e_delay_per_amb(self) -> np.ndarray:
        """Return current D_e2e (s) per ambulance's URLLC queue, shape (K,)."""
        K = self.config.K_ambulances
        d_e2e = np.empty(K, dtype=np.float64)
        for k in range(K):
            q = self.queues[f"urllc_{k}"]
            if not q.is_stable:
                d_e2e[k] = float(SEVERITY_QOS[5]["D_max"]) * 2.0   # clamp at 2× tightest D_max if unstable
                continue
            d_tx = q.mean_service_time - D_STOCH        # subtract stoch component
            d_queue = q.expected_queue_delay()
            d_e2e[k] = D_DET + d_tx + d_queue + D_FH + D_BH
        return d_e2e

    # ----------------------------------------------------------------
    # AoI + position
    # ----------------------------------------------------------------

    def _update_aoi_trackers(self, n_urllc_per_amb: np.ndarray) -> None:
        """Inject one consolidated status sample per ambulance per TTI if its
        URLLC traffic arrives this tick (one tracker dict per ambulance,
        2026-06-14 fix)."""
        for k in range(self.config.K_ambulances):
            if n_urllc_per_amb[k] <= 0:
                continue
            for sid, tracker in self.aoi_trackers[k].items():
                tracker.arrive(gen_time=self.sim_time)
                tracker.deliver_next(sim_time=self.sim_time + self.config.tti_sec)

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
        """Return formal Worker state s_t^L (31-dim for K=1, F=1).

        Layout (per-ambulance severity_k epic, B5, 2026-06-15):
            == Fixed 20-dim block ==
            [0:2]   Q_urllc, Q_eMBB                      (queue arrival rates, normalized)
            [2:4]   HOL_urllc, HOL_eMBB                   (head-of-line delay, ms)
            [4:7]   r_min^URLLC, r_max^eMBB, r_ded^URLLC  (PRB ratios)
            [7:9]   arr_rate_urllc, arr_rate_eMBB         (per-second rate, normalized)
            [9]     mean BLER
            [10:15] severity_ref one-hot (levels 1..5 NON_URGENT..IMMEDIATE,
                    severity_ref = max(severity_per_amb))
            [15]    λ_local for shared C3 (eMBB throughput floor)
            [16]    r_min_urllc_anchor (Manager setpoint, FIXED within window;
                    obs[4]=live r_min drifts around it — see set_rrm_budget)
            [17]    n_bys (active bystander UE count, normalized by M_eMBB)
            [18:20] mean AoI + max AoI (s, across K ambulances × F streams)
            == 10K per-ambulance block (interleaved per k) ==
            For each k in 0..K-1, 10 contiguous dims:
              SINR_k, d_k, v_k, delay_norm_k, AoI_norm_k, severity_k_norm,
              λC1_k, λC2_k, λC4_k, λC5_k
            where delay_norm_k = D_e2e_k / D_max^{sev_k}, AoI_norm_k =
            AoI_k / AoI_max^{sev_k} (per-ambulance severity thresholds),
            severity_k_norm = severity_per_amb[k] / 5.0, and λC*_k are the
            per-ambulance Lagrangian multipliers from set_lambda_local().
            == F per-stream block ==
            [20+10K:20+10K+F]  AoI per stream (s), mean over K — F=1, the
                               consolidated "ambulance_status" stream

        Normalisations chosen to keep components O(1) for PPO stability.
        At K=1, this layout reduces to exactly the pre-redesign scalar
        values via the documented [0,1,3,4,2] permutation (no numeric
        change to existing K=1 behaviour).
        """
        K = self.config.K_ambulances
        F = self.config.num_streams
        sev_oh = np.zeros(5, dtype=np.float32)
        sev_oh[self.severity - 1] = 1.0

        # Queue load: utilization ρ = λ/μ ∈ [0, 1], mean over K URLLC queues
        # (NOT duplicate of arrival rate). Identical to old scalar at K=1.
        rho_urllc = float(np.clip(
            np.mean([self.queues[f"urllc_{k}"].rho for k in range(K)]), 0.0, 1.0
        ))
        rho_emBB = float(np.clip(self.queues["eMBB"].rho, 0.0, 1.0))
        # Arrival rates (per second, normalized differently from ρ).
        # arr_urllc = sum over K ambulances (total system rate, K=1-identical).
        arr_urllc = float(sum(self.queues[f"urllc_{k}"].arrival_rate for k in range(K))) / 1e3
        arr_emBB = float(self.queues["eMBB"].arrival_rate) / 1e4
        # Clip HOL: unstable queue returns +inf; cap at sane upper bounds.
        # Mean over K URLLC queues (identical to old scalar at K=1).
        hol_urllc_ms = min(
            float(np.mean([self.queues[f"urllc_{k}"].hol_delay() for k in range(K)])) * 1e3,
            100.0,
        )
        hol_emBB_ms = min(float(self.queues["eMBB"].hol_delay()) * 1e3, 1000.0)

        # PRB ratios (Phase 1.1: r_min, r_max, r_ded directly — KHÔNG split per slice)
        prb_ratios = np.array(
            [self.r_min_urllc, self.r_max_emBB, self.r_ded_urllc],
            dtype=np.float32,
        )

        # Bystander UE count (normalized by baseline M_eMBB)
        if self.bystander is not None:
            n_bys = float(self.bystander.active_ue_count(self.sim_time)) / max(self.config.M_eMBB, 1)
        else:
            n_bys = 1.0  # baseline M_eMBB → ratio 1.0

        # AoI per ambulance for the consolidated "ambulance_status" stream
        # (one tracker dict per ambulance, 2026-06-14 fix).
        aoi_per_amb = np.array(
            [t["ambulance_status"].current_aoi(self.sim_time) for t in self.aoi_trackers],
            dtype=np.float64,
        )
        # AoI summary (mean + max over K ambulances × F=1 stream — identical
        # to old scalar at K=1).
        aoi_mean = float(aoi_per_amb.mean()) if aoi_per_amb.size > 0 else 0.0
        aoi_max = float(aoi_per_amb.max()) if aoi_per_amb.size > 0 else 0.0

        # Per-ambulance kinematics (sinr already cached, recompute distance + speed).
        # NB: v_k (speed) is now the sole signal distinguishing on-scene (v≈0) vs
        # in-transport (v high) mobility — phase one-hot no longer carries it.
        sinr_norm = self.last_sinr_db.astype(np.float32) / 40.0     # [-10, 40] dB → [-0.25, 1.0]
        amb_dist = np.linalg.norm(self.ambulance_pos, axis=1) / max(self.config.cell_radius_m, 1.0)
        amb_speed = np.linalg.norm(self.ambulance_vel, axis=1) / 60.0  # normalize by 60 m/s cap

        # Per-ambulance proximity to QoS violation, using each ambulance's OWN
        # severity_k threshold (B5 epic 2026-06-15) — dimensionless ratios that
        # hit ~1.0 at the violation boundary, comparable across ambulances even
        # when severity_k differs.
        d_max_phi_per_amb = np.array(
            [SEVERITY_QOS[int(s)]["D_max"] for s in self.severity_per_amb], dtype=np.float64
        )
        aoi_max_phi_per_amb = np.array(
            [SEVERITY_QOS[int(s)]["AoI_max"] for s in self.severity_per_amb], dtype=np.float64
        )
        delay_norm = (self._last_d_e2e_per_amb / d_max_phi_per_amb).astype(np.float32)
        aoi_norm = (self._last_aoi_per_amb / aoi_max_phi_per_amb).astype(np.float32)
        severity_k_norm = (self.severity_per_amb.astype(np.float32) / 5.0)

        # Per-ambulance Lagrangian multipliers from set_lambda_local(), laid
        # out as [C1_0..C1_{K-1}, C2_0..C2_{K-1}, C4_0..C4_{K-1}, C5_0..C5_{K-1}, C3_shared].
        lam = self._lambda_local
        lam_c1 = lam[0:K]
        lam_c2 = lam[K:2 * K]
        lam_c4 = lam[2 * K:3 * K]
        lam_c5 = lam[3 * K:4 * K]
        lam_c3_shared = float(lam[4 * K])

        # F per-stream block (F=1): mean over K ambulances of the consolidated
        # "ambulance_status" AoI — identical to the old aoi_vec at K=1.
        aoi_stream_vec = np.full(F, aoi_mean, dtype=np.float32)

        # 10K per-amb block, INTERLEAVED per k:
        # [SINR_k, d_k, v_k, delay_norm_k, AoI_norm_k, severity_k_norm,
        #  λC1_k, λC2_k, λC4_k, λC5_k]
        per_amb = np.stack(
            [
                sinr_norm,
                amb_dist.astype(np.float32),
                amb_speed.astype(np.float32),
                delay_norm,
                aoi_norm,
                severity_k_norm,
                lam_c1.astype(np.float32),
                lam_c2.astype(np.float32),
                lam_c4.astype(np.float32),
                lam_c5.astype(np.float32),
            ],
            axis=1,
        ).reshape(-1)

        # === Fixed 20-dim block — assembled by NAMED INDEX (single source of
        # truth = OBS_*_IDX in utils.config; do NOT hardcode positions here). ===
        fixed = np.zeros(OBS_FIXED_BLOCK_LEN, dtype=np.float32)
        fixed[OBS_RHO_URLLC_IDX] = rho_urllc
        fixed[OBS_RHO_EMBB_IDX] = rho_emBB
        fixed[OBS_HOL_URLLC_IDX] = hol_urllc_ms
        fixed[OBS_HOL_EMBB_IDX] = hol_emBB_ms
        fixed[OBS_R_MIN_URLLC_IDX] = prb_ratios[0]
        fixed[OBS_R_MAX_EMBB_IDX] = prb_ratios[1]
        fixed[OBS_R_DED_URLLC_IDX] = prb_ratios[2]
        fixed[OBS_ARR_URLLC_IDX] = arr_urllc
        fixed[OBS_ARR_EMBB_IDX] = arr_emBB
        fixed[OBS_BLER_IDX] = float(self.last_bler)
        fixed[OBS_SEVERITY_OH_IDX:OBS_SEVERITY_OH_IDX + OBS_SEVERITY_OH_LEN] = sev_oh
        fixed[OBS_LAMBDA_C3_IDX] = lam_c3_shared
        fixed[OBS_RMIN_ANCHOR_IDX] = self.r_min_urllc_anchor
        fixed[OBS_N_BYS_IDX] = n_bys
        fixed[OBS_AOI_MEAN_IDX] = aoi_mean
        fixed[OBS_AOI_MAX_IDX] = aoi_max

        obs = np.concatenate(
            [
                fixed,                                               # [0:20]
                per_amb,                                             # [20:20+10K]
                aoi_stream_vec,                                      # [20+10K:20+10K+F]
            ]
        )
        # Final safety: replace any residual NaN/inf with 0
        return np.nan_to_num(obs.astype(np.float32), nan=0.0, posinf=1e3, neginf=-1e3)

    def _info(self) -> dict[str, Any]:
        K = self.config.K_ambulances
        d_max_phi_per_amb = np.array(
            [SEVERITY_QOS[int(s)]["D_max"] for s in self.severity_per_amb], dtype=np.float64
        )
        aoi_max_phi_per_amb = np.array(
            [SEVERITY_QOS[int(s)]["AoI_max"] for s in self.severity_per_amb], dtype=np.float64
        )
        return {
            "tti": self.tti_idx,
            "sim_time": self.sim_time,
            # severity = severity_ref = max(severity_per_amb), drives SHARED
            # quantities (alpha_e reward weight, C3 R_min, fixed-block one-hot).
            "severity": self.severity,
            "severity_name": SEVERITY_QOS[self.severity]["name"],
            # severity_per_amb[k] = independent severity_k in {1..5}, fixed for
            # the episode (B5 epic 2026-06-15).
            "severity_per_amb": self.severity_per_amb.copy(),
            "r_min_urllc": self.r_min_urllc,
            "r_max_eMBB": self.r_max_emBB,
            "r_ded_urllc": self.r_ded_urllc,
            "sinr_db": float(np.mean(self.last_sinr_db)),
            "mean_BLER": self.last_bler,
            # All K per-ambulance URLLC queues stable (C12 semantics; identical
            # to old scalar at K=1).
            "urllc_stable": all(self.queues[f"urllc_{k}"].is_stable for k in range(K)),
            "eMBB_stable": self.queues["eMBB"].is_stable,
            # (4K+1)-dim hard constraint signals aggregated per Worker step
            # (B5 epic 2026-06-15). Layout:
            # c_vec[0:K]      = D_e2e_k (s)          | d_phi[0:K]      = D_max^{sev_k} (s)
            # c_vec[K:2K]     = URLLC tail viol_k     | d_phi[K:2K]     = ε^{sev_k}
            # c_vec[2K:3K]    = AoI_k (s)             | d_phi[2K:3K]    = AoI_max^{sev_k} (s)
            # c_vec[3K:4K]    = AoI tail viol_k       | d_phi[3K:4K]    = ε_AoI^{sev_k}
            # c_vec[4K]       = signed eMBB gap (Mbps)| d_phi[4K]       = 0
            # At K=1 this is exactly the old [C1,C2,C3,C4,C5] under the
            # permutation [0,1,3,4,2].
            "c_vec": self._last_c_vec.copy(),
            "d_phi": self._last_d_phi.copy(),
            # Phase 2.1 reward restructure (post-critique W12): URLLC mean delay
            # exported for diagnostics — NOT used in reward computation. Reward is
            # alpha_e * log1p(R_eMBB/R_REF) only; URLLC enforced via λ_1, λ_2.
            "l_urllc_mean": float(self._last_l_urllc),
            # Reviewer M2 (internal review, W02, 2026-05-27): expose M/G/1 queue diagnostics
            # so reviewers can audit Pollaczek-Khinchine formula application.
            # Returns dict {lambda, mu, rho, E_S, E_S2, E_D_queue, HOL, stable}.
            # See docs/13 §1.3 service-time distribution + env/queue_model.py:62-73.
            # queue_diag_urllc = ambulance-0's queue (backward-compat shape;
            # identical to the old pooled queue at K=1).
            "queue_diag_urllc": self.queues["urllc_0"].summary(),
            "queue_diag_embb": self.queues["eMBB"].summary(),
            # Per-ambulance diagnostics: one queue summary, delay_norm, AoI_norm
            # per ambulance (using each ambulance's OWN severity_k threshold,
            # B5 epic 2026-06-15) — lets reviewers/policies audit per-ambulance
            # proximity to QoS violation when severity_k differs.
            "queue_diag_urllc_per_amb": [self.queues[f"urllc_{k}"].summary() for k in range(K)],
            "delay_norm_per_amb": (self._last_d_e2e_per_amb / d_max_phi_per_amb).astype(np.float32),
            "aoi_norm_per_amb": (self._last_aoi_per_amb / aoi_max_phi_per_amb).astype(np.float32),
            # Manager setpoint anchor and PRB breakdown — read from step() cache,
            # never recomputed here (both functions are not pure).
            "r_min_urllc_anchor": self.r_min_urllc_anchor,
            "prb_urllc": self._last_prb_urllc,
            "prb_embb": self._last_prb_embb,
            "prb_per_amb": self._last_prb_per_amb.tolist(),
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
