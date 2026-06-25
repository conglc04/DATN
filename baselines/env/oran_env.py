"""ORANEnv — Gymnasium-compatible environment for PPO.

Integrates all Week 2-3 modules into a TTI-level simulator:
    - Channel: SINR per UE per cell
    - Queue: M/G/1 per slice
    - Traffic: URLLC + eMBB generators
    - Severity: 5-level patient-urgency tier (exogenous, fixed per episode)
    - AoI: stream-aware trackers

Time hierarchy (single real-time clock — sim_time advances one MAC tick/step):
    MAC tick     = 0.5 ms                  (internal O-DU scheduler; not exposed)
    Worker step  = 10 ms = 20 MAC ticks    ← env.step() unit
    Manager step = 100 ms = 10 Worker steps
    Episode      ≤ 400 s                   (ends early when all ambulances arrive)

Action ownership (clean HRL split, 2026-06-19):
    Manager chooses the inter-slice budget:
        273 PRBs → URLLC slice + eMBB slice via set_rrm_budget().
    Worker/xApp chooses only the intra-URLLC split:
        K=1: 1-dim no-op action; the single active ambulance receives all URLLC PRBs.
        K>=2: a[0:K] = per-ambulance priority logits ℓ_k → softmax → PRB weights
              (pure-RL: no β slot, no rule, no N_req, no λ in the allocation).

Observation (Worker s_L, flattened, obs_dim = 20 + 11K + F):
    == Fixed 20-dim block ==
    Q_urllc, Q_eMBB                      (queue lengths, packets)
    HOL_urllc, HOL_eMBB                  (sec)
    PRB_alloc_urllc, PRB_alloc_eMBB, PRB_ded_urllc (fraction of P_TOTAL)
    arrival_rate_urllc, arrival_rate_eMBB
    mean_BLER_cell
    severity_ref one-hot (5-dim)
    λ_local^C3 (shared, 1-dim)
    rrm_budget (1-dim placeholder), n_bys, mean AoI, max AoI
    == 11K per-ambulance block (interleaved) ==
    SINR_k, d_k, v_k, delay_norm_k, AoI_norm_k, severity_k_norm,
    λ_local^C1_k, λ_local^C2_k, λ_local^C4_k, λ_local^C5_k, active_mask_k
    == F per-stream block ==
    AoI_per_stream                        (sec)

Reference:
    - docs/08_implementation_notes.md TTI Simulation Loop
    - docs/05_agent_workflow.md State + Action + Reward
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.channel_model import (
    BaseStation,
    ChannelModel,
    capacity_per_prb_bps,
    noise_plus_interference_dbm,
    thermal_noise_dbm,
)
from env.queue_model import MG1Queue, SliceQueueManager
from env.aoi_tracker import AoIStreamTracker
from env.sumo_mobility import (
    PooledSumoMobilityProvider,
    SumoMobilityProvider,
    default_fcd_path as _default_sumo_fcd_path,
    default_route_pool as _default_route_pool,
    density_fcd_path as _density_fcd_path,  # noqa: F401 — re-exported for callers
    gps_to_metric as _sumo_gps_to_metric,
)
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
    URLLC_OFFERED_LOAD_BPS,
    URLLC_PKT_BITS,
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
    SEVERITY_OH_OBS_INDEX,
    SEVERITY_QOS,
    P_TOTAL,
    R_REF_EMBB_MBPS,
    SHANNON_ETA,
    TTI_SEC,
    ARRIVAL_RADIUS_M,
    DEST_LAT,
    DEST_LON,
    build_d_phi_vector,
)


# ============================================================
# Constants
# ============================================================

# E1 fix: overload delay for rho >= 1.0 — must exceed ALL D_max values
# to guarantee C1/C2 violations regardless of severity level.
# Also caps the Pollaczek-Khinchine queueing delay near rho→1 (audit 2026-06-22):
# PK = λE[S²]/(2(1-ρ)) diverges as ρ→1⁻, producing delays of millions of seconds
# that blow up the Lagrangian penalty and the critic target. Capping at this
# sentinel keeps the violation registered without numerical explosion.
OVERLOAD_DELAY_SEC: float = 0.1   # 100 ms >> D_max_sev1 = 20 ms

# AoI overload sentinel (audit 2026-06-22, parallel to OVERLOAD_DELAY_SEC):
# current_aoi grows unbounded (= sim_time since last delivery) when a vehicle's
# delivery stalls — observed reaching 28-53 s in smoke train, which blows up the
# C4/C5 penalty and the critic loss to 1e18-1e20. Cap at a sentinel that exceeds
# ALL AoI_max values (loosest = AoI_max_sev1 = 1.0 s) so the violation is still
# registered for every severity, while bounding the penalty magnitude.
OVERLOAD_AOI_SEC: float = 2.0   # 2 s >> AoI_max_sev1 = 1.0 s

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
    """All tunables for a single-cell ambulance setup.

    Defaults reflect the DECIDED macro-cell problem (W15-B2, 2026-06-18):
        R_CELL = 1000m, UMa path loss, interference_margin = -86 dBm/PRB,
        ambulance_speed = 60 km/h, sinr_clamp_min = -15 dB.
    episode_duration_sec stays at 1.0 for fast unit-test execution; production
    training uses macro_mission_config() which sets 400.0.
    For hard-mission (S1 burst + S2B spike), use :func:`hard_mission_config`.
    """

    K_ambulances: int = 1
    M_eMBB: int = 30
    num_streams: int = 1
    initial_severity: int = 5               # default IMMEDIATE (tightest) for sanity
    # Severity_k epic (2026-06-15): each of the K ambulances carries an
    # INDEPENDENT severity_k in {1..5}, sampled independently and fixed for the
    # episode (severity_per_amb). severity_ref := max(severity_per_amb) drives
    # all SHARED quantities (C3 R_min, severity one-hot,
    # info["severity"]). For training diversity, set sample_severity=True to
    # draw a fresh independent level per ambulance each reset() from
    # severity_sample_weights. When False, every ambulance uses initial_severity
    # (=> severity_per_amb = [initial_severity]*K, severity_ref = initial_severity,
    # exact K=1 legacy behaviour).
    sample_severity: bool = True
    severity_sample_weights: tuple[float, ...] = (0.20, 0.20, 0.20, 0.20, 0.20)
    episode_duration_sec: float = 1.0       # short for unit tests; production uses 400.0
    tti_sec: float = TTI_SEC
    # Traffic rates (steady state, before phase scaling)
    urllc_arrival_rate: float = 50.0        # ambulance_status steady (pkt/s/ambulance)
    embb_arrival_rate: float = 1000.0       # per eMBB UE (pkt/s)
    urllc_packet_bits: int = 400 * 8        # 400B → 3200 bits
    embb_packet_bits: int = 1500 * 8        # MTU
    # Geometry — single-cell, gNB fixed at origin (0,0) = local Cartesian
    # convergence point of 3 ambulances on Giải Phóng (đường hội tụ về BV
    # Bạch Mai, GPS 21.002966, 105.840780 — anchor cho lớp OSM/SUMO, W15;
    # KHÔNG dùng GPS trực tiếp trong env).
    # R_CELL = 1000m (macro, W15-B2). No handover (docs/03_architecture.md, M2.0).
    cell_radius_m: float = 1000.0
    ambulance_speed_kmh: float = 60.0       # emergency vehicle speed (vClass=emergency)
    ambulance_start_distance_m: float | None = None  # None = random; set to fix
    # Channel — macro-cell calibrated values (W15-B2, 2026-06-18)
    bs_tx_power_total_dbm: float = 46.0      # total BS TX power over whole carrier
    bs_n_prb: int = P_TOTAL                  # number of PRBs power is spread over
    sinr_clamp_max_db: float = 40.0
    sinr_clamp_min_db: float = -15.0        # UMa cell-edge floor (was -10 for UMi)
    # Aggregate inter-cell interference margin (dBm/PRB). -86 dBm/PRB calibrated
    # for UMa 1km macro cell (W15-B2). None = noise-limited SNR-only (legacy).
    interference_margin_dbm_per_prb: float | None = -86.0
    # Propagation scenario for the serving cell's path loss (3GPP TR 38.901):
    #   "micro" → UMi Street Canyon (LOS/NLOS sampled)
    #   "macro" → UMa Urban Macro (pl_uma)  [decided W15-B2]
    bs_layer: Literal["macro", "micro"] = "macro"
    # λ_local dual variables live in the agent's LambdaState; the env caches
    # the latest snapshot via set_lambda_local() (used by _observe() for the
    # per-ambulance λ slots in obs). The training loop additionally overlays
    # λ_local onto the returned obs via utils.obs.overlay_lambda_local (single
    # source, applied by every solver driver) so the policy sees the value
    # computed for the SAME decision step. Pure-RL allocation does NOT use
    # λ in _prb_split_intra_slice — allocation is 100% from policy logits.
    rrm_budget_hint: float = 0.10           # 10% → 27 PRBs; matches macro working point

    # ---- Hard-mission features (opt-in, default disabled) -----------------
    urllc_burst_at_sec: float | None = None
    urllc_burst_duration_sec: float = 0.10
    urllc_burst_factor: float = 10.0
    enable_bystander: bool = False
    bystander_trigger_sec: float = 0.4
    bystander_peak_range: tuple[int, int] = (80, 120)
    bystander_per_ue_mbps: tuple[float, float] = (2.0, 5.0)
    # ---- SUMO mobility (W15) --- -----------------------------------------
    # When set, SumoMobilityProvider replaces the legacy RWP bounce model.
    # hard_mission_config() sets this automatically to the bundled K-specific
    # trace.  Set to None to use the old RWP model (legacy / unit-test only).
    sumo_fcd_path: str | None = None
    # Route pool: list of FCD paths to sample from per episode.
    # Mutually exclusive with sumo_fcd_path (validated in __post_init__).
    # PooledSumoMobilityProvider samples one path per reset(seed).
    sumo_route_pool: list[str] | None = None

    # ---- Ambulance lifecycle (F1/F4, W15 future) --------------------------
    # When True, ambulances are marked "arrived" once they reach the destination.
    # SUMO: arrival via FCD exit within arrival_radius_m of destination only
    # (no live dist check — prevents false positive on pass-through routes).
    # RWP:  legacy dist_to_gnb < arrival_radius_m fallback.
    # Arrived vehicles receive 0 PRB and zeroed obs block.
    # Episode terminates (terminated=True) when ALL K ambulances have arrived.
    # Default False → exact legacy behavior (all always active, fixed duration).
    enable_arrival: bool = False
    arrival_radius_m: float = 25.0          # FCD-exit arrival threshold (m)
    # GPS centroid of the ambulance stopping zone (edge 37370971#0).
    # Converted to local Cartesian in reset() via gps_to_metric.
    # SSOT: DEST_LAT/LON in utils/config.py.
    destination_lat: float = DEST_LAT
    destination_lon: float = DEST_LON

    # ---- Background traffic density (F5, W15 future) ----------------------
    # Synthetic speed overlay: "light"→1.2×, "medium"→1.0× (no-op), "heavy"→0.7×
    # Tier-A (real SUMO density traces) flagged as separate SUMO re-run task.
    traffic_density: str = "medium"         # {"light", "medium", "heavy"}

    # ---- Phase 2.1 reward (W05 refactor + post-critique restructure W12) ---
    # Reward is eMBB log-utility ONLY (single-term objective), NO α_e weight:
    #   r = U_eMBB(t),  U_eMBB = log(1 + R_eMBB / R_REF_EMBB_MBPS)
    # α_e(sev) severity-weighting REMOVED 2026-06-23 — severity enters the system
    # ONLY via constraints C1–C5 + λ dual ascent, never via reward weighting
    # (sev5 had α_e=0.05 → reward ≈ 0, killing the Manager's gradient signal).
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
        w = np.asarray(self.severity_sample_weights, dtype=np.float64)
        if w.shape != (5,):
            raise ValueError(
                f"severity_sample_weights must contain 5 weights; got shape {w.shape}"
            )
        if np.any(w < 0.0) or float(w.sum()) <= 0.0:
            raise ValueError("severity_sample_weights must be non-negative with positive sum")
        if self.sumo_fcd_path is not None and self.sumo_route_pool is not None:
            raise ValueError("Cannot set both sumo_fcd_path and sumo_route_pool simultaneously.")
        if self.sumo_route_pool is not None:
            if len(self.sumo_route_pool) == 0:
                raise ValueError("sumo_route_pool must be non-empty.")
        # RWP removed (2026-06-20): the env ALWAYS uses SUMO+OSM mobility. When no
        # trace is configured, auto-discover the density route pool for this K.
        # Pool discovery order: density variants (93 files) → legacy single file.
        if self.sumo_fcd_path is None and self.sumo_route_pool is None:
            import os as _os
            _pool = _default_route_pool(self.K_ambulances)
            if _pool and all(_os.path.exists(p) for p in _pool):
                self.sumo_route_pool = _pool
            else:
                _p = _default_sumo_fcd_path(self.K_ambulances)
                if not _os.path.exists(_p):
                    raise ValueError(
                        f"No SUMO+OSM FCD trace for K={self.K_ambulances} ({_p}). "
                        f"Traces exist for K in {{1, 3}}; the legacy RWP fallback was removed — "
                        f"supply sumo_fcd_path/sumo_route_pool or use K in {{1, 3}}."
                    )
                self.sumo_fcd_path = _p
        _VALID_DENSITIES = {"light", "medium", "heavy"}
        if self.traffic_density not in _VALID_DENSITIES:
            raise ValueError(f"traffic_density must be one of {_VALID_DENSITIES}; got {self.traffic_density!r}")


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
        initial_severity=5,                         # IMMEDIATE — tightest QoS hint (sampled each episode)
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
        bs_tx_power_total_dbm=30.0,
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
        # W15: SUMO mobility — rotate across 3 density scenarios per episode.
        sumo_route_pool=_default_route_pool(K_ambulances),
    )


def macro_mission_config(
    *,
    K_ambulances: int = 3,
    seed: int = 0,
) -> EnvConfig:
    """W15-B2 macro-cell scenario — calibration-locked parameters.

    Locked values (2026-06-18):
        P_TOTAL = 273  bs_tx_power_total_dbm = 46  bs_n_prb = 273
        interference_margin_dbm_per_prb = -86     alpha = 0.5 (in bler_effective)
        R_CELL_M = 1000m  UMa pathloss  K = 3 ambulances

    Working point at cell edge (1000m):
        tx_per_prb = 46 - 10*log10(273) = 21.6 dBm/PRB
        SINR = 2.7 dB  BLER_single_tx = 0.41
        min_prb for QoS (alpha=0.5): 11 PRBs (r_min = 4.0%)
        K=3 collective: 21 PRBs (7.7%) — leaves 92.3% for eMBB
    """
    return EnvConfig(
        K_ambulances=K_ambulances,
        initial_severity=3,                          # fallback if sample_severity=False
        # Physics params (cell_radius_m=1000, bs_layer="macro",
        # interference_margin_dbm_per_prb=-86, ambulance_speed_kmh=60,
        # sinr_clamp_min_db=-15, sample_severity=True, rrm_budget_hint=0.10)
        # are now EnvConfig defaults — no need to repeat here.
        # W15: SUMO mobility — rotate across 3 density scenarios per episode.
        sumo_route_pool=_default_route_pool(K_ambulances),
        # Episode ends when all K ambulances arrive at the destination
        # (edge 37370971#0 centroid). amb_2 arrives ~372s in K3-light traces.
        # Fallback truncation at 400s if not all have arrived.
        enable_arrival=True,
        episode_duration_sec=400.0,
        # ARRIVAL_RADIUS_M=15m covers all 4 lanes on edge 37370971#0
        # (max lane deviation 6.38m from centroid); excludes gNB area (47.7m away).
        arrival_radius_m=ARRIVAL_RADIUS_M,
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
        # Clean HRL action space:
        #   Manager controls inter-slice r_min_urllc/r_max_eMBB via set_rrm_budget().
        #   Worker controls only intra-URLLC priority via per-vehicle logits.
        # K=1: one no-op scalar keeps continuous-control agents well-defined
        #      (softmax([ℓ_0])=[1.0] always — single ambulance gets full B_U).
        # K>=2: a[0:K] = per-vehicle priority logits ℓ_k → softmax → PRB_k.
        # No β slot (pure-RL allocation has no urgency-temperature term).
        if K >= 2:
            self.action_space = spaces.Box(
                low=np.full(K, -3.0, dtype=np.float32),
                high=np.full(K, 3.0, dtype=np.float32),
                dtype=np.float32,
            )
        else:
            self.action_space = spaces.Box(
                low=np.array([-3.0], dtype=np.float32),
                high=np.array([3.0], dtype=np.float32),
                dtype=np.float32,
            )

        # Formal Worker state s_t^L = 20 + 11K + F dims (per-ambulance
        # severity_k epic 2026-06-15; +active_mask_k 2026-06-23):
        #   20-dim fixed block (queue, HOL, PRB_util, arr_rates, BLER,
        #                       severity_ref one-hot, λ_local^C3 shared,
        #                       rrm_budget, n_bys, AoI summary)
        #   11K per-ambulance block, INTERLEAVED per ambulance:
        #     SINR_k, d_k, v_k, delay_norm_k, AoI_norm_k, severity_k_norm,
        #     λ_local^C1_k, λ_local^C2_k, λ_local^C4_k, λ_local^C5_k, active_mask_k
        #   F per-stream block (AoI per stream)
        # For K=1, F=1 → 20 + 11 + 1 = 32 dim.
        obs_dim = OBS_FIXED_BLOCK_LEN + OBS_PER_AMB_BLOCK_LEN * K + F
        self._obs_dim = obs_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # ---------------- Persistent components ----------------
        self.base_station = BaseStation(
            cell_id=0, x=0.0, y=0.0, h=10.0, layer=self.config.bs_layer,
            tx_power_dbm=self.config.bs_tx_power_total_dbm,
        )
        self.channel = ChannelModel(shadowing=True, rng=np.random.default_rng(seed))
        # Bystander spike model — built lazily in reset() if enabled
        self.bystander = None
        # SUMO mobility provider — initialised in reset() when sumo_fcd_path set
        self._mobility: SumoMobilityProvider | PooledSumoMobilityProvider | None = None

        # ---------------- Per-episode state (set by reset) ----------------
        self.ambulance_pos: np.ndarray            # (K, 2)
        self.ambulance_vel: np.ndarray            # (K, 2)
        self.severity_per_amb: np.ndarray         # (K,) independent severity_k 1..5 (fixed/episode)
        self.severity: int                        # severity_ref = max(severity_per_amb), drives SHARED quantities
        self._lambda_local: np.ndarray            # (4K+1,) λ_local, set via set_lambda_local()
        self._beta: float                         # reserved (unused in pure-RL allocation)
        self._prb_weights: np.ndarray             # (K,) per-vehicle priority logits from action[0:K]
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
        # Ambulance lifecycle masks (F1/F4, enable_arrival=False → all-True/all-False)
        self.active_mask: np.ndarray         # (K,) bool — entered cell AND not yet arrived
        self.arrived_mask: np.ndarray        # (K,) bool — latched True on arrival
        self.entered_mask: np.ndarray        # (K,) bool — latched True once dist ≤ R_CELL_M
        # Feasibility bounds for set_rrm_budget() — computed at reset()
        self._feasible_rrm_floor: float
        self._feasible_rrm_cap: float
        # Diagnostics
        self.e2e_history: list[float]
        self.viol_history: list[bool]
        self.prb_alloc_history: list[tuple[int, int]]
        self.embb_mbps_history: list[float]
        self.c3_viol_history: list[bool]
        self.aoi_history: list[float]        # C4: mean AoI (s) per MAC tick
        self.aoi_viol_history: list[bool]    # C5: AoI-tail violation per MAC tick
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

        # Compute destination in local Cartesian (metric relative to gNB = origin).
        # Used by SumoMobilityProvider arrival detection and _update_arrival_masks().
        _dest_east, _dest_north = _sumo_gps_to_metric(
            self.config.destination_lat, self.config.destination_lon
        )
        self._destination_xy_m = np.array([_dest_east, _dest_north], dtype=np.float64)
        _dest_xy_tuple = (float(_dest_east), float(_dest_north))

        # Ambulance positions — SUMO trace / route pool / legacy RWP.
        if self.config.sumo_route_pool is not None:
            self._mobility = PooledSumoMobilityProvider(
                self.config.sumo_route_pool,
                K=K,
                tti_sec=self.config.tti_sec,
                destination_xy_m=_dest_xy_tuple,
                arrival_radius_m=self.config.arrival_radius_m,
            )
            # Random density scenario per episode (light/medium/heavy background
            # traffic).  Route is non-cyclic origin→BV Bạch Mai so always t=0;
            # fast-forward in the block below advances to first cell entry.
            _trace_idx = int(self.rng.integers(0, len(self.config.sumo_route_pool)))
            init_pos = self._mobility.reset(trace_idx=_trace_idx)
            self.ambulance_pos = init_pos.astype(np.float64)
            _, vel = self._mobility.step()
            self.ambulance_vel = vel.astype(np.float64)
            self._mobility.reset(trace_idx=_trace_idx)
        elif self.config.sumo_fcd_path is not None:
            self._mobility = SumoMobilityProvider(
                self.config.sumo_fcd_path,
                K=K,
                tti_sec=self.config.tti_sec,
                destination_xy_m=_dest_xy_tuple,
                arrival_radius_m=self.config.arrival_radius_m,
            )
            init_pos = self._mobility.reset()
            self.ambulance_pos = init_pos.astype(np.float64)
            _, vel = self._mobility.step()
            self.ambulance_vel = vel.astype(np.float64)
            self._mobility.reset()
        else:
            # Legacy RWP bounce (preserved for unit-test / non-hard-mission use)
            self._mobility = None
            if self.config.ambulance_start_distance_m is not None:
                radii = np.full(K, float(self.config.ambulance_start_distance_m))
            else:
                radii = self.rng.uniform(30.0, self.config.cell_radius_m, size=K)
            angles = self.rng.uniform(0.0, 2.0 * math.pi, size=K)
            self.ambulance_pos = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
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
        # (C3 R_min, severity one-hot, info["severity"]).
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

        # MAC ratios — initialise from Manager hint. Worker actions no longer
        # mutate these inter-slice ratios; set_rrm_budget() is the sole HRL hook.
        self.r_min_urllc = self.config.rrm_budget_hint
        self.r_min_urllc_anchor = self.r_min_urllc   # Manager setpoint anchor; obs[16]
        self.r_max_emBB = 1.0 - self.r_min_urllc
        self.r_ded_urllc = 0.0

        # Feasibility bounds for set_rrm_budget() (conservative, recomputed each reset).
        # Floor: K × min-PRBs-per-ambulance at SINR=0dB with ×5 QoS safety margin.
        # Cap:   ensures enough PRBs remain for the fixed 10 Mbps eMBB floor at SINR=0dB.
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

        # Ambulance lifecycle masks (F1/F6).
        # enable_arrival=False: arrived_mask stays all-False; active_mask all-True.
        # sumo_fcd_path set: fast-forward trace until first ambulance enters cell;
        #   entered_mask starts all-False and latches per-ambulance on cell entry.
        # Legacy RWP / no SUMO: entered_mask starts all-True (positions already in cell).
        self.arrived_mask = np.zeros(K, dtype=bool)
        if self._mobility is not None:
            # Fast-forward SUMO until at least one ambulance enters cell (dist ≤ R_CELL_M).
            # This defines the RL episode start; tti_idx/sim_time reset after fast-forward.
            # Vectorized O(n_timesteps) seek (was a per-TTI step loop ~4e5 steps = ~13s/reset).
            pos, vel, _entered = self._mobility.advance_until_within(self.config.cell_radius_m)
            self.ambulance_pos = pos.astype(np.float64)
            self.ambulance_vel = vel.astype(np.float64)
            dist = np.linalg.norm(self.ambulance_pos, axis=1)
            self.entered_mask = dist <= self.config.cell_radius_m
            # Precompute trajectory cache: vectorized interpolation for the
            # entire episode at 0.5ms resolution, replacing ~735K per-tick
            # Python function calls with a single NumPy pass + array lookup.
            # Positions are deterministic from the FCD trace (not RL-dependent).
            max_ticks = self._max_tti_for_episode()
            if hasattr(self._mobility, 'build_trajectory_cache'):
                self._mobility.build_trajectory_cache(max_ticks)
        else:
            # Legacy RWP: positions already initialised inside cell.
            self.entered_mask = np.ones(K, dtype=bool)
        self.active_mask = self.entered_mask & ~self.arrived_mask

        # Arrival diagnostic state (reset each episode)
        self._last_dist_to_gnb = np.zeros(K, dtype=np.float64)
        self._last_dist_to_dest = np.zeros(K, dtype=np.float64)
        self._arrival_reason = np.full(K, "", dtype=object)
        self._arrival_rl_time = np.full(K, np.nan, dtype=np.float64)

        # Sim time + diagnostics
        self.tti_idx = 0
        self.sim_time = 0.0
        self.last_sinr_db = np.full(K, 15.0)
        self.last_bler = 0.0
        self.last_bler_per_amb = np.zeros(K, dtype=np.float64)
        self._aoi_pkt_arrived = np.zeros(K, dtype=np.int64)
        self._aoi_pkt_delivered = np.zeros(K, dtype=np.int64)
        self._aoi_pkt_failed_bler = np.zeros(K, dtype=np.int64)
        self._aoi_pkt_failed_no_prb = np.zeros(K, dtype=np.int64)
        self._aoi_pkt_failed_no_capacity = np.zeros(K, dtype=np.int64)
        # Raw C2/C5 counters — violation_count / sample_count per ambulance.
        # C2: MAC ticks where PK-expected D_e2e > D_max (queue-state proxy).
        # C5: MAC ticks where current AoI > AoI_max (true observation).
        # Only active ticks are counted (inactive → excluded).
        self._c2_violation_count = np.zeros(K, dtype=np.int64)
        self._c2_sample_count = np.zeros(K, dtype=np.int64)
        self._c5_violation_count = np.zeros(K, dtype=np.int64)
        self._c5_sample_count = np.zeros(K, dtype=np.int64)
        # C-fix (audit 2026-06-22): RLC-style cross-TTI service-bit accumulator.
        # A URLLC packet may span multiple TTIs when one TTI's transport block is
        # smaller than the packet (low SINR / few PRB). Without accumulation the
        # all-or-nothing per-TTI gate stalls delivery forever → unbounded AoI.
        self._partial_service_bits = np.zeros(K, dtype=np.float64)
        # Per-ambulance D_e2e / AoI cache, refreshed every MAC tick — read by
        # _observe() for delay_norm_k / AoI_norm_k (2026-06-14 fix).
        self._last_d_e2e_per_amb = np.zeros(K, dtype=np.float64)
        self._last_aoi_per_amb = np.zeros(K, dtype=np.float64)
        self.e2e_history = []
        self.viol_history = []
        self.prb_alloc_history = []
        self.embb_mbps_history = []
        self.c3_viol_history = []
        self.aoi_history = []
        self.aoi_viol_history = []
        self.last_embb_mbps = 0.0
        # Per-Worker-step constraint accumulator ((4K+1)-dim, reset each
        # Worker step in step()) — B5 severity_k epic 2026-06-15 layout:
        #   [C1_0..C1_{K-1}, C2_0..C2_{K-1}, C4_0..C4_{K-1}, C5_0..C5_{K-1}, C3_shared]
        n_c = 4 * K + 1
        self._worker_c_accum = np.zeros(n_c, dtype=np.float64)
        self._worker_tick_count = 0
        # Per-ambulance active-sample counter (K,): counts MAC ticks where active_mask[k]=True.
        # Used as per-ambulance denominator for C1/C2/C4/C5 to normalize over active window.
        self._worker_active_count = np.zeros(K, dtype=np.float64)
        # Last computed Worker-step c_vec + d_phi (for info dict)
        self._last_c_vec = np.zeros(n_c, dtype=np.float32)
        self._last_l_urllc: float = 0.0    # URLLC mean delay (D/D_ref) — diagnostics only
        # Initialize d_phi from severity_per_amb (so reset() returns valid info immediately)
        self._last_d_phi = build_d_phi_vector(self.severity_per_amb).astype(np.float32)

        # λ_local — env-internal storage, default zeros; overwritten by
        # set_lambda_local() each Worker step (train.py / solver drivers).
        self._lambda_local = np.zeros(n_c, dtype=np.float64)
        # β reserved (pure-RL allocation ignores it; kept for action-dim compat).
        self._beta = BETA_MIN
        # Per-vehicle priority logits (K≥2 only); zero init → uniform softmax →
        # no per-vehicle bias; updated by _apply_action each step.
        self._prb_weights = np.zeros(K, dtype=np.float64)

        # First channel sample
        self._update_channel()
        self._update_queue_service_rates()

        return self._observe(), self._info()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """One Worker/xApp step = 10 ms = 20 MAC ticks (per O-RAN spec).

        xApp decides only intra-URLLC priority at start of step; the inter-slice
        URLLC/eMBB budget is the latest Manager setpoint. O-DU MAC executes that
        schedule for 20 TTI internal ticks. Reward is the MEAN of per-TTI rewards.

        Reward = MEAN (not SUM) over MAC ticks: the constraint c_vec is a per-tick
        MEAN (delay rate, violation rate, AoI), so reward must share the same
        temporal basis for the augmented Lagrangian r − Σλⱼ·gⱼ to be balanced.
        SUM-vs-MEAN mismatch (×20) made the eMBB reward gradient swamp the
        constraint penalty → Manager starved URLLC (audit 2026-06-23, see
        docs/OPTIMIZATION_PROBLEM_FINAL §reward).

        Returns observation aggregated AT END of Worker step (post 20 ticks).
        See docs/13_methodology_walkthrough.md Phase 1.4 for full hierarchy.
        """
        # 1. Decode Worker action ONCE per step → intra-URLLC priority only.
        self._apply_action(action)

        # 1b. Cache PRB allocation for this step (r_min_urllc is Manager-owned;
        #     must NOT be recomputed inside _info() since state may have advanced).
        self._last_prb_urllc, self._last_prb_embb = self._prb_allocation()
        self._last_prb_per_amb = self._prb_split_intra_slice(self._last_prb_urllc)

        # 1c. Reset per-Worker-step constraint accumulator (aggregated across 20 MAC ticks)
        n_c = 4 * self.config.K_ambulances + 1
        K = self.config.K_ambulances
        self._worker_c_accum = np.zeros(n_c, dtype=np.float64)
        self._worker_tick_count = 0
        self._worker_active_count = np.zeros(K, dtype=np.float64)

        # 2. Run MAC_TICKS_PER_WORKER (=20) internal MAC ticks with same action.
        #    Reward = MEAN over executed ticks (temporal-basis match with c_vec mean).
        reward_accumulated = 0.0
        n_reward_ticks = 0
        for _ in range(MAC_TICKS_PER_WORKER):
            reward_accumulated += self._mac_tick()
            n_reward_ticks += 1
            if self.tti_idx >= self._max_tti_for_episode():
                break  # episode truncated mid-Worker-step (rare)
        reward_accumulated /= max(n_reward_ticks, 1)   # SUM → MEAN

        # 3. Aggregate c_vec across MAC ticks → mean per-step constraint signal ((4K+1)-dim).
        # C1/C2/C4/C5: normalize per ambulance by its own active-sample count so a
        #   late-entering ambulance is evaluated on its active window only, not diluted
        #   by idle ticks before it entered the cell.
        # C3 (eMBB, index 4K): slice-level constraint, normalize by total tick count.
        if self._worker_tick_count > 0:
            # Per-ambulance denominator: active_count[k] if > 0, else 1 (avoid ÷0; numerator=0).
            per_amb_denom = np.where(self._worker_active_count > 0,
                                     self._worker_active_count, 1.0)
            # Build full (4K+1) denominator: replicate per-amb 4×, then total ticks for C3.
            denom = np.concatenate([
                per_amb_denom, per_amb_denom, per_amb_denom, per_amb_denom,
                [float(self._worker_tick_count)],
            ])
            self._last_c_vec = (self._worker_c_accum / denom).astype(np.float32)
        else:
            self._last_c_vec = np.zeros(n_c, dtype=np.float32)
        # 4. Per-step severity threshold lookup (severity fixed/episode, docs/13 Phase 2.2)
        self._last_d_phi = build_d_phi_vector(self.severity_per_amb).astype(np.float32)

        # F4: early termination when ALL ambulances have arrived (enable_arrival=True only).
        all_arrived = bool(self.config.enable_arrival and self.arrived_mask.all())
        terminated = all_arrived
        truncated = (not terminated) and (self.tti_idx >= self._max_tti_for_episode())
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
        #
        # LIMITATION (C2 semantic proxy): d_e2e_per_amb is the M/G/1 PK
        # EXPECTED queue delay — a queue-state metric, NOT an observed
        # per-packet delay. The true chance constraint P(D_packet > D_max)
        # requires packet-level delay samples, which are unavailable in this
        # analytical simulation (M/G/1 PK). The proxy P(E[D_state] > D_max)
        # is monotonically correlated but not equivalent at the boundary.
        # For packet-level fidelity, use ns-3 (future work, docs/01).
        K = self.config.K_ambulances
        d_e2e_per_amb = self._compute_e2e_delay_per_amb()
        d_max_phi_per_amb = np.array(
            [SEVERITY_QOS[int(s)]["D_max"] for s in self.severity_per_amb], dtype=np.float64
        )
        viol_per_amb = d_e2e_per_amb > d_max_phi_per_amb
        # RC-1 fix: diagnostics only over ACTIVE ambulances (inactive have
        # rho=inf → OVERLOAD delay, inflating metrics 10-400x at K>1).
        if self.active_mask.any():
            d_e2e = float(np.mean(d_e2e_per_amb[self.active_mask]))
            viol = bool(np.any(viol_per_amb[self.active_mask]))
        else:
            d_e2e = 0.0
            viol = False

        # HARQ + bookkeeping
        self.last_bler = self._sample_bler()
        self.last_bler_per_amb = self._sample_bler_per_amb()

        # eMBB throughput (Mbps) — capacity-limited when queue unstable.
        # SHARED C3 quantity: R_min is a FIXED, severity-INDEPENDENT floor
        # (10 Mbps at every severity — CMDP_D_J_SEVERITY[*]["d3_embb_mbps"]).
        # sev_ref (= max(severity_per_amb)) below is only the TABLE LOOKUP
        # KEY, not a value-driver — kept so C3 shares one per-severity table
        # shape with C1/C2/C4/C5 (which DO genuinely vary by severity).
        sev_ref = self.severity
        self.last_embb_mbps = self._compute_embb_throughput_mbps()
        r_min_embb_phi = float(CMDP_D_J_SEVERITY[sev_ref]["d3_embb_mbps"])
        embb_gap_mbps = r_min_embb_phi - self.last_embb_mbps
        embb_deficit_mbps = max(0.0, embb_gap_mbps)
        c3_viol = embb_gap_mbps > 0.0

        # ---------------- Reward: pure eMBB log-utility (no α_e) ----------------
        # r_t = log(1 + R_eMBB / R_REF)
        # Severity differentiation is ENTIRELY via constraints C1–C5 + λ dual
        # ascent — NOT via reward weighting. Removing α_e(sev) eliminates the
        # double-count: constraints already force higher b_rrm at high severity
        # (large penalty if QoS violated), so α_e was redundant and obscured
        # the Manager's gradient signal (sev=5 had α_e=0.05 → reward ≈ 0).
        l_urllc = d_e2e / D_REF_URLLC                       # diagnostics only
        self._last_l_urllc = float(l_urllc)
        u_embb = math.log(1.0 + self.last_embb_mbps / R_REF_EMBB_MBPS)
        reward = u_embb

        # ---------------- AoI tracking for C4, C5 (Worker-step aggregation) ----------------
        # Single consolidated per-ambulance AoI stream (F=1, LCFS+drop_old per docs/04),
        # one tracker per ambulance. Per-ambulance C4/C5 thresholds use
        # severity_per_amb[k] (B5 epic).
        aoi_max_phi_per_amb = np.array(
            [SEVERITY_QOS[int(s)]["AoI_max"] for s in self.severity_per_amb], dtype=np.float64
        )
        # Cap raw AoI at OVERLOAD_AOI_SEC (audit 2026-06-22): a stalled-delivery
        # vehicle's AoI grows unbounded (= time since last delivery); capping
        # keeps the C4/C5 violation registered without exploding the penalty.
        aoi_per_amb = np.minimum(
            np.array(
                [t["ambulance_status"].current_aoi(self.sim_time) for t in self.aoi_trackers],
                dtype=np.float64,
            ),
            OVERLOAD_AOI_SEC,
        )
        aoi_viol_per_amb = aoi_per_amb > aoi_max_phi_per_amb
        # RC-1 fix: AoI diagnostics only over ACTIVE ambulances.
        if self.active_mask.any():
            aoi_mean_tick = float(np.mean(aoi_per_amb[self.active_mask]))
            aoi_tail_viol = float(np.any(aoi_viol_per_amb[self.active_mask]))
        else:
            aoi_mean_tick = 0.0
            aoi_tail_viol = 0.0

        # Cache per-ambulance values for _observe()'s delay_norm_k / AoI_norm_k
        self._last_d_e2e_per_amb = d_e2e_per_amb
        self._last_aoi_per_amb = aoi_per_amb

        # ---------------- Per-MAC-tick c_vec accumulator ((4K+1)-dim, B5 epic) ----------------
        # Aggregate across MAC ticks → reported as Worker-step c_vec in info dict.
        # Layout: [C1_0..C1_{K-1}, C2_0..C2_{K-1}, C4_0..C4_{K-1}, C5_0..C5_{K-1}, C3_shared]
        #
        # Constraint taxonomy (CMDP formulation — load-bearing, do not change):
        #   C1_k = E[D_e2e^k]         MEAN-type  (per-amb delay, seconds)
        #   C2_k = P(D>D_max)         CHANCE-type (per-amb delay-tail, 0/1)
        #   C3    = E[R_min - R_eMBB] MEAN-type  (shared eMBB floor, Mbps)
        #   C4_k = E[AoI_k]           MEAN-type  (per-amb AoI, seconds)
        #   C5_k = P(AoI>AoI_max)     CHANCE-type (per-amb AoI-tail, 0/1)
        #
        # C1/C3/C4 use Option-b (interval-window mean, reset per Manager step).
        # C2/C5   use Option-a (episode-cumulative, N grows with time).
        # See agents/lagrangian.py:_tail_mask for the dispatch.
        # At K=1 this is the permutation [0,1,3,4,2] of the legacy 5-dim
        # [C1,C2,C3,C4,C5] — exact numeric preservation.
        # F6: mask ambulance constraints by active_mask (not-yet-entered OR arrived → skip).
        # Per-ambulance active-sample counter enables per-active-window normalization.
        # C3 (eMBB) is slice-level — always accumulates regardless of ambulance state.
        am = self.active_mask.astype(np.float64)
        self._worker_active_count += am           # (K,) count of active MAC ticks per ambulance
        if self.active_mask.any():
            self._worker_c_accum[0:K] += d_e2e_per_amb * am
            self._worker_c_accum[K:2 * K] += viol_per_amb.astype(np.float64) * am
            self._worker_c_accum[2 * K:3 * K] += aoi_per_amb * am
            self._worker_c_accum[3 * K:4 * K] += aoi_viol_per_amb.astype(np.float64) * am
            # Raw C2/C5 counters (active ticks only)
            active_int = self.active_mask.astype(np.int64)
            self._c2_sample_count += active_int
            self._c2_violation_count += (viol_per_amb & self.active_mask).astype(np.int64)
            self._c5_sample_count += active_int
            self._c5_violation_count += (aoi_viol_per_amb & self.active_mask).astype(np.int64)
        self._worker_c_accum[4 * K] += embb_gap_mbps   # C3: slice-level, never masked
        self._worker_tick_count += 1

        # State update + history
        self._advance_ambulance_positions()
        self._update_aoi_trackers(n_urllc_per_amb)
        self.tti_idx += 1
        self.sim_time += self.config.tti_sec

        # URLLC histories: only when ≥1 ambulance active (avoid dilution by zeros).
        if self.active_mask.any():
            self.e2e_history.append(d_e2e)
            self.viol_history.append(viol)
            self.aoi_history.append(aoi_mean_tick)
            self.aoi_viol_history.append(bool(aoi_tail_viol))
        # Slice-level histories: always (eMBB/C3 independent of ambulances).
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
        """Decode Worker action into intra-URLLC priority only.

        Inter-slice ratios are intentionally not touched here:
            Manager: r_min_urllc / eMBB remainder
            Worker: per-ambulance logits → softmax → PRB weights

        a[0:K] = per-vehicle priority logits ℓ_k, raw ∈ ℝ (K>=2 only).
        No β slot — pure-RL allocation has no urgency-temperature term.
        """
        a = np.asarray(action, dtype=np.float64)
        K = self.config.K_ambulances
        self._beta = BETA_MIN  # reserved, unused in allocation

        if K >= 2 and a.shape[0] >= K:
            self._prb_weights = a[0:K].astype(np.float64)
        else:
            self._prb_weights = np.zeros(K, dtype=np.float64)

    def _renormalize_prb_ratios(self) -> None:
        """Enforce r_min_urllc + r_max_emBB ≤ 1 by trimming r_max_emBB (C6)."""
        if self.r_min_urllc + self.r_max_emBB > 1.0:
            excess = self.r_min_urllc + self.r_max_emBB - 1.0
            self.r_max_emBB = max(0.0, self.r_max_emBB - excess)

    def set_rrm_budget(self, b_rrm: float) -> None:
        """Re-anchor r_min_urllc to Manager setpoint at the start of a Manager window.

        Two-tier clipping (no hard severity floor — severity differentiation
        is entirely via constraints C1–C5 + λ dual ascent):
          1. [B_RRM_MIN, B_RRM_MAX] — analytical bounds from decode_manager_action.
          2. [feasible_rrm_floor, feasible_rrm_cap] — computed at reset() per K/QoS.
        The tighter of all bounds applies.  Must be called BEFORE the Worker loop.
        """
        lo = max(B_RRM_MIN, self._feasible_rrm_floor)
        hi = min(B_RRM_MAX, self._feasible_rrm_cap)
        hi = max(hi, lo)   # safety: ensure hi ≥ lo
        clipped = float(np.clip(b_rrm, lo, hi))
        self.r_min_urllc = clipped
        self.r_min_urllc_anchor = clipped
        self.r_max_emBB = 1.0 - clipped
        self.r_ded_urllc = 0.0

    def _prb_allocation(self) -> tuple[int, int]:
        prb_urllc = int(self.r_min_urllc * P_TOTAL)
        prb_emBB = P_TOTAL - prb_urllc   # remainder → sum always = P_TOTAL
        return prb_urllc, max(prb_emBB, 0)

    # ----------------------------------------------------------------
    # Channel + arrivals + queues
    # ----------------------------------------------------------------

    def _update_channel(self) -> None:
        """Compute SINR per ambulance against the single cell."""
        K = self.config.K_ambulances
        # Per-PRB TX power: total power divided over all PRBs in the carrier.
        # Correct power accounting (W15-B2): tx_per_prb = total_dbm - 10*log10(N_prb).
        # Legacy path (bs_n_prb == P_TOTAL, interference_margin_dbm_per_prb == None)
        # is unchanged: tx_per_prb = 46 - 10*log10(273) = 21.6 dBm/PRB.
        import math as _math
        tx_per_prb_dbm = (
            self.config.bs_tx_power_total_dbm
            - 10.0 * _math.log10(max(self.config.bs_n_prb, 1))
        )
        # Effective noise floor: thermal noise, plus a constant inter-cell
        # interference margin for interference-limited macro scenarios (None =
        # legacy noise-limited SNR; preserves W12/micro exactly).
        n_dbm = noise_plus_interference_dbm(
            thermal_noise_dbm(B_PRB), self.config.interference_margin_dbm_per_prb
        )
        sinrs = np.empty(K)
        for k in range(K):
            rx = self.channel.receive_power_dbm(
                (float(self.ambulance_pos[k, 0]), float(self.ambulance_pos[k, 1])),
                self.base_station,
                tx_power_dbm=tx_per_prb_dbm,
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
        """Cell-average BLER: logistic(mean SINR across K ambulances), 3GPP-style.

        The mean is over all K ambulance SINR samples, producing a single scalar
        representing the aggregate link quality seen by the scheduler. This is
        NOT per-UE BLER (no per-UE CQI feedback is modelled). Calibration:
        ~50% BLER at SINR=0 dB; asymptotes to ~0% at high SINR.
        """
        sinr = float(np.mean(self.last_sinr_db))
        # logistic curve: BLER ≈ 0.5 at SINR ≈ 0 dB, asymptotes → 0 at high SINR
        bler = 1.0 / (1.0 + math.exp(0.5 * (sinr - 2.0)))
        return float(np.clip(bler, 1e-4, 0.5))

    def _sample_bler_per_amb(self) -> np.ndarray:
        """Per-ambulance BLER from per-ambulance SINR, same logistic as cell-average."""
        K = self.config.K_ambulances
        bler = np.empty(K, dtype=np.float64)
        for k in range(K):
            sinr_k = float(self.last_sinr_db[k])
            b = 1.0 / (1.0 + math.exp(0.5 * (sinr_k - 2.0)))
            bler[k] = float(np.clip(b, 1e-4, 0.5))
        return bler

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
        # RC-1 upstream: suppress arrivals for inactive ambulances so their
        # queues stay idle (rho=0) and AoI trackers don't accumulate packets.
        n_urllc_per_amb[~self.active_mask] = 0
        n_emBB = int(self.rng.poisson(max(lam_emBB_per_tti, 0.0)))

        # Update per-ambulance queue arrival-rate estimate (per-second basis)
        for k in range(K):
            rate_k = eff_urllc_rate if self.active_mask[k] else 0.0
            self.queues[f"urllc_{k}"].set_arrival_rate(rate_k)
        self.queues["eMBB"].set_arrival_rate(eff_embb_total_rate)
        return n_urllc_per_amb, n_emBB

    def _update_queue_service_rates(self) -> None:
        """Split prb_urllc across K ambulances via the pure-RL intra-slice softmax.

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
        """Pure-RL intra-slice PRB allocation — fully learned, no rules.

        Pipeline (matches formulation diagram):
          severity_k → obs_k → neural policy → ℓ_k → softmax → w_k → PRB_k

        The Worker neural network outputs per-ambulance logits ℓ_k
        (action[0:K] for K≥2). softmax(ℓ) produces weights w_k.

        Reserve-first split (audit 2026-06-24, fixes a floor violation):
          1. Reserve PRB_MIN_QOS for every active ambulance FIRST:
             reserved = K_active × PRB_MIN_QOS.
          2. Split only the remainder (B_U − reserved) by softmax weight,
             largest-remainder integer projection.
          3. PRB_k = PRB_MIN_QOS + remainder_share_k.
        This guarantees PRB_k >= PRB_MIN_QOS for every active k by
        construction. The prior order (floor the full-budget proportional
        split → force each entry up to the minimum → rescale down on
        overflow) could zero out an ambulance when one softmax weight
        dominated extremely — e.g. raw logits [10,-5,-5] with B_U=27 gave
        [26,1,0], violating the floor for the third ambulance.
        Feasibility precondition: B_U >= K_active × PRB_MIN_QOS — holds with
        wide margin under current bounds (B_RRM_MIN=0.05 → B_U>=13 PRBs vs
        K_active<=3 → reserved<=3 PRBs); raises if a future config breaks it.

        NO hard-coded severity ordering.
        NO N_req formula.
        NO dual-ascent urgency (λ) in allocation.
        NO β temperature multiplier.

        Severity awareness is FULLY LEARNED by the policy:
          - Policy observes severity_k_norm in per-ambulance obs block
          - Policy observes λ_C1..C5 per amb (constraint violation signal)
          - r_aug = r - λ^T·(c-d) penalizes violations (Lagrangian reward)
          - PPO gradient teaches the policy to output higher ℓ_k for
            ambulances that need more resources

        K=1: full B_U goes to the single active ambulance.
        K_active=0: all-zero array.
        """
        K = self.config.K_ambulances
        B_U = int(prb_urllc)

        active_idx = np.where(self.active_mask)[0]
        K_active = len(active_idx)
        if K_active == 0:
            return np.zeros(K, dtype=np.int64)
        if K_active == 1:
            prb_out = np.zeros(K, dtype=np.int64)
            prb_out[active_idx[0]] = B_U
            return prb_out

        # --- Reserve PRB_MIN_QOS for every active ambulance FIRST ---
        reserved = K_active * PRB_MIN_QOS
        if B_U < reserved:
            raise ValueError(
                f"Infeasible URLLC PRB split: B_U={B_U} < K_active={K_active} "
                f"* PRB_MIN_QOS={PRB_MIN_QOS} (reserved={reserved}). "
                "B_RRM_MIN / K_ambulances must guarantee "
                "B_U >= K_active * PRB_MIN_QOS."
            )
        remaining = B_U - reserved

        # --- Pure softmax over the remaining budget only ---
        w = _softmax(self._prb_weights[active_idx])

        # --- Largest-remainder integer allocation of the remainder ---
        extra = np.floor(remaining * w).astype(np.int64)
        rem = remaining - int(extra.sum())
        if rem > 0:
            fracs = remaining * w - extra.astype(float)
            for i in np.argsort(-fracs)[:rem]:
                extra[i] += 1

        allocs = np.full(K_active, PRB_MIN_QOS, dtype=np.int64) + extra

        prb_out = np.zeros(K, dtype=np.int64)
        for i, k in enumerate(active_idx):
            prb_out[k] = int(allocs[i])
        return prb_out

    def set_lambda_local(self, lambda_local: np.ndarray) -> None:
        """Store the (4K+1,)-dim λ_local vector (B5 epic, 2026-06-15).

        Called by train.py / solver drivers each Worker step. Feeds
        _observe()'s per-ambulance λ slots (policy can see constraint state).
        Pure-RL allocation does NOT use λ in _prb_split_intra_slice().
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
        """Return current D_e2e (s) per ambulance's URLLC queue, shape (K,).

        E1 fix: use PK formula for all rho < 1.0 (was incorrectly gated by
        is_stable at rho < 0.9).  For rho >= 1.0, use OVERLOAD_DELAY_SEC
        which exceeds all D_max values, guaranteeing C1/C2 violations.
        """
        K = self.config.K_ambulances
        d_e2e = np.empty(K, dtype=np.float64)
        for k in range(K):
            q = self.queues[f"urllc_{k}"]
            rho_k = q.rho
            if rho_k >= 1.0:
                d_e2e[k] = OVERLOAD_DELAY_SEC
                continue
            d_tx = q.mean_service_time - D_STOCH
            d_queue = q.expected_queue_delay()
            d_e2e[k] = min(D_DET + d_tx + d_queue + D_FH + D_BH, OVERLOAD_DELAY_SEC)
        return d_e2e

    # ----------------------------------------------------------------
    # AoI + position
    # ----------------------------------------------------------------

    def _update_aoi_trackers(self, n_urllc_per_amb: np.ndarray) -> None:
        """Receiver-side AoI: arrive on generation, deliver on PRB + capacity + BLER.

        E2 fix: separate arrival phase from service phase — pending packets
        are retried on every TTI with allocated PRBs, not just on arrival ticks.
        E3 fix: delivery requires service_bits >= URLLC_PKT_BITS (capacity gate).
        E4 fix: arrival counter tracks actual Poisson count, not coalesced 1.
        """
        tti_sec = self.config.tti_sec
        K = self.config.K_ambulances

        # --- Phase 1: Arrivals (only ticks with new packets) ---
        for k in range(K):
            n_arr = int(n_urllc_per_amb[k])
            if n_arr <= 0:
                continue
            for _sid, tracker in self.aoi_trackers[k].items():
                tracker.arrive(gen_time=self.sim_time)
            self._aoi_pkt_arrived[k] += n_arr

        # --- Phase 2: Service (all ambulances with pending packets) ---
        # C-fix (audit 2026-06-22): RLC segmentation — accumulate service bits
        # across TTIs instead of an all-or-nothing per-TTI capacity gate. A
        # vehicle with too few PRB / low SINR to fit a whole packet in one TTI
        # now makes partial progress each TTI and eventually delivers, instead
        # of stalling forever (which drove AoI to tens of seconds).
        for k in range(K):
            tracker = self.aoi_trackers[k]["ambulance_status"]
            if len(tracker.queue) == 0:
                self._partial_service_bits[k] = 0.0   # no pending packet → drop partial
                continue

            prb_k = int(self._last_prb_per_amb[k])
            if prb_k <= 0:
                self._aoi_pkt_failed_no_prb[k] += 1
                continue

            sinr_k = float(self.last_sinr_db[k])
            cap_k = capacity_per_prb_bps(sinr_k, eta=SHANNON_ETA)
            self._partial_service_bits[k] += prb_k * cap_k * tti_sec
            if self._partial_service_bits[k] < URLLC_PKT_BITS:
                # Not enough bits accumulated for a full packet yet — partial
                # progress this TTI (counted as a no-capacity-this-TTI event).
                self._aoi_pkt_failed_no_capacity[k] += 1
                continue

            # Enough bits for one packet → consume them + roll HARQ/BLER once.
            self._partial_service_bits[k] -= URLLC_PKT_BITS
            bler_k = self.last_bler_per_amb[k]
            if self.rng.random() >= bler_k:
                tracker.deliver_next(sim_time=self.sim_time + tti_sec)
                self._aoi_pkt_delivered[k] += 1
            else:
                self._aoi_pkt_failed_bler[k] += 1

    _DENSITY_SPEED_SCALE: dict[str, float] = {
        "light": 1.2,   # less background traffic → ambulance moves faster
        "medium": 1.0,  # baseline (Tier-A default; Tier-B synthetic proxy)
        "heavy": 0.7,   # heavy congestion → ambulance slowed
    }

    def _advance_ambulance_positions(self) -> None:
        if self._mobility is not None:
            # SUMO trace replay — positions and velocities from pre-generated FCD.
            pos, vel = self._mobility.step()
            # F5: Tier-B synthetic traffic-density speed overlay (medium=no-op).
            # Applied BEFORE SINR/obs computation so kinematics stay consistent.
            # NOTE: This is a synthetic proxy. Tier-A (real density-variant SUMO
            # traces) requires re-running SUMO — flagged as a separate task.
            scale = self._DENSITY_SPEED_SCALE.get(self.config.traffic_density, 1.0)
            if scale != 1.0:
                # Scale velocity; recompute position from scaled velocity increment.
                vel = vel * scale
                pos = self.ambulance_pos + vel * self.config.tti_sec
            self.ambulance_pos = pos
            self.ambulance_vel = vel
        else:
            # Legacy RWP bounce (unit-tests / non-hard-mission only).
            self.ambulance_pos = self.ambulance_pos + self.ambulance_vel * self.config.tti_sec
            r2 = (self.ambulance_pos ** 2).sum(axis=1)
            out = r2 > self.config.cell_radius_m ** 2
            if out.any():
                self.ambulance_vel[out] *= -1.0
        # F1: update arrival masks after position update.
        self._update_arrival_masks()

    def _update_arrival_masks(self) -> None:
        """F1/F6: update entered_mask and arrived_mask after each position step.

        entered_mask[k]: latches True when dist_to_gnb[k] ≤ R_CELL_M (cell entry).
        arrived_mask[k]: latches True when vehicle reaches destination (SUMO path:
            FCD exit within arrival_radius_m of destination only;
            RWP path: legacy dist_to_gnb < arrival_radius_m fallback).
        active_mask[k] = entered_mask[k] AND NOT arrived_mask[k].

        enable_arrival=False → arrived_mask stays all-False (legacy exact behavior).
        Legacy RWP (no SUMO): entered_mask is all-True from reset, never changes here.
        """
        dist_to_gnb = np.linalg.norm(self.ambulance_pos, axis=1)
        self._last_dist_to_gnb = dist_to_gnb.copy()

        # F6: latch cell entry via dist_to_gnb (SUMO path only; RWP: all-True from reset).
        if self._mobility is not None:
            newly_entered = (~self.entered_mask) & (dist_to_gnb <= self.config.cell_radius_m)
            if newly_entered.any():
                for k in np.where(newly_entered)[0]:
                    for tracker in self.aoi_trackers[k].values():
                        tracker.reset()
            self.entered_mask |= newly_entered

        # F1: latch arrival — separate logic for SUMO vs RWP.
        #   SUMO: arrival ONLY via reached_destination_mask (FCD exit near dest).
        #     No live dist_to_dest check — prevents false positive if a future route
        #     passes through the 15m circle around dest without stopping.
        #   RWP:  legacy dist_to_gnb < arrival_radius_m fallback (no FCD).
        if self.config.enable_arrival:
            if self._mobility is not None:
                dist_to_dest = np.linalg.norm(
                    self.ambulance_pos - self._destination_xy_m, axis=1
                )
                self._last_dist_to_dest = dist_to_dest.copy()
                newly_arrived = self._mobility.reached_destination_mask.copy()
            else:
                dist_to_dest = dist_to_gnb
                self._last_dist_to_dest = dist_to_dest.copy()
                newly_arrived = dist_to_gnb < self.config.arrival_radius_m

            just_arrived = newly_arrived & ~self.arrived_mask
            for k in np.where(just_arrived)[0]:
                if self._mobility is not None:
                    self._arrival_reason[k] = "fcd_exit_dest"
                else:
                    self._arrival_reason[k] = "dist_to_gnb_rwp"
                self._arrival_rl_time[k] = self.sim_time
            self.arrived_mask |= newly_arrived
        else:
            self._last_dist_to_dest = dist_to_gnb.copy()

        self.active_mask = self.entered_mask & ~self.arrived_mask

    # ----------------------------------------------------------------
    # Observation + info
    # ----------------------------------------------------------------

    def _observe(self) -> np.ndarray:
        """Return formal Worker state s_t^L (32-dim for K=1, F=1).

        Layout (per-ambulance severity_k epic, B5, 2026-06-15; +active_mask_k 2026-06-23):
            == Fixed 20-dim block ==
            [0:2]   Q_urllc, Q_eMBB                      (queue arrival rates, normalized)
            [2:4]   HOL_urllc, HOL_eMBB                   (head-of-line delay, ms)
            [4:7]   r_min^URLLC, r_max^eMBB, r_ded^URLLC  (PRB ratios)
            [7:9]   arr_rate_urllc, arr_rate_eMBB         (per-second rate, normalized)
            [9]     mean BLER
            [10:15] severity_ref one-hot (levels 1..5 NON_URGENT..IMMEDIATE,
                    severity_ref = max(severity_per_amb))
            [15]    λ_local for shared C3 (eMBB throughput floor)
            [16]    r_min_urllc_anchor (Manager setpoint; obs[4] should match it
                    unless an external test manually edits env.r_min_urllc)
            [17]    n_bys (active bystander UE count, normalized by M_eMBB)
            [18:20] mean AoI + max AoI (s, across K ambulances × F streams)
            == 11K per-ambulance block (interleaved per k) ==
            For each k in 0..K-1, 11 contiguous dims:
              SINR_k, d_k, v_k, delay_norm_k, AoI_norm_k, severity_k_norm,
              λC1_k, λC2_k, λC4_k, λC5_k, active_mask_k
            where delay_norm_k = D_e2e_k / D_max^{sev_k}, AoI_norm_k =
            AoI_k / AoI_max^{sev_k} (per-ambulance severity thresholds),
            severity_k_norm = severity_per_amb[k] / 5.0, λC*_k are the
            per-ambulance Lagrangian multipliers from set_lambda_local(), and
            active_mask_k ∈ {0,1} = entered_k & ~arrived_k (explicit active
            flag; 0 for inactive xe so the all-zero sentinel is unambiguous).
            == F per-stream block ==
            [20+11K:20+11K+F]  AoI per stream (s), mean over K — F=1, the
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
        # (one tracker dict per ambulance, 2026-06-14 fix). Capped at
        # OVERLOAD_AOI_SEC (audit 2026-06-22) so a stalled-delivery vehicle's
        # unbounded AoI does not inflate the observation / obs-AoI summary.
        aoi_per_amb = np.minimum(
            np.array(
                [t["ambulance_status"].current_aoi(self.sim_time) for t in self.aoi_trackers],
                dtype=np.float64,
            ),
            OVERLOAD_AOI_SEC,
        )
        # RC-3 fix: AoI summary over ACTIVE ambulances only (inactive have
        # AoI=sim_time, inflating the fixed-block signal to the Worker).
        if self.active_mask.any():
            active_aoi = aoi_per_amb[self.active_mask]
            aoi_mean = float(active_aoi.mean())
            aoi_max = float(active_aoi.max())
        else:
            aoi_mean = 0.0
            aoi_max = 0.0

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

        # Explicit per-vehicle active flag (2026-06-23): 1=active, 0=inactive.
        # Placed LAST in the block so the zeroing step below naturally drives it
        # to 0 for inactive vehicles (= correct) and leaves it 1 for active ones.
        active_flag = self.active_mask.astype(np.float32)

        # 11K per-amb block, INTERLEAVED per k:
        # [SINR_k, d_k, v_k, delay_norm_k, AoI_norm_k, severity_k_norm,
        #  λC1_k, λC2_k, λC4_k, λC5_k, active_mask_k]
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
                active_flag,
            ],
            axis=1,
        ).reshape(-1)

        # F1/F6: zero out the per-amb block for ALL inactive ambulances so the
        # policy sees a clean sentinel (all-zeros, incl. active_mask_k=0) and
        # doesn't track stale kinematics. Inactive = not-yet-entered (outside
        # cell) OR already-arrived. active_mask = entered_mask & ~arrived_mask;
        # ~active_mask covers both. (active_mask_k stays 1 only for active xe.)
        if (~self.active_mask).any():
            per_amb_2d = per_amb.reshape(K, -1)  # (K, OBS_PER_AMB_BLOCK_LEN)
            per_amb_2d[~self.active_mask] = 0.0
            per_amb = per_amb_2d.reshape(-1)

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
                per_amb,                                             # [20:20+11K]
                aoi_stream_vec,                                      # [20+11K:20+11K+F]
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
            # quantities (C3 R_min, fixed-block one-hot).
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
            # log(1 + R_eMBB/R_REF) pure utility; URLLC enforced via λ_1, λ_2.
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
            "bler_per_amb": self.last_bler_per_amb.tolist(),
            "aoi_pkt_arrived": self._aoi_pkt_arrived.copy(),
            "aoi_pkt_delivered": self._aoi_pkt_delivered.copy(),
            "aoi_pkt_failed_bler": self._aoi_pkt_failed_bler.copy(),
            "aoi_pkt_failed_no_prb": self._aoi_pkt_failed_no_prb.copy(),
            "aoi_pkt_failed_no_capacity": self._aoi_pkt_failed_no_capacity.copy(),
            # Raw C2/C5 counters — for sweep/audit (violation_count / sample_count).
            # C2 limitation: uses PK-expected delay (queue-state proxy), not
            # per-packet delay — inherent to analytical M/G/1 simulation.
            "c2_violation_count": self._c2_violation_count.copy(),
            "c2_sample_count": self._c2_sample_count.copy(),
            "c5_violation_count": self._c5_violation_count.copy(),
            "c5_sample_count": self._c5_sample_count.copy(),
            # F1: per-ambulance lifecycle (masks in info only; obs sentinel-zeroed above).
            "active_mask": self.active_mask.copy(),
            "arrived_mask": self.arrived_mask.copy(),
            "entered_mask": self.entered_mask.copy(),
            "n_active": int(self.active_mask.sum()),
            "active_count_per_amb": self._worker_active_count.copy(),
            "all_arrived": bool(self.arrived_mask.all()),
            "episode_end_reason": (
                "all_arrived" if (self.config.enable_arrival and self.arrived_mask.all())
                else "truncated"
            ),
            # Arrival distance diagnostics (updated each step by _update_arrival_masks)
            "dist_to_gnb_per_amb": self._last_dist_to_gnb.copy(),
            "dist_to_destination_per_amb": self._last_dist_to_dest.copy(),
            "present_mask": (
                self._mobility.present_mask
                if self._mobility is not None
                else np.ones(K, dtype=bool)
            ),
            "reached_destination_mask": (
                self._mobility.reached_destination_mask
                if self._mobility is not None
                else np.zeros(K, dtype=bool)
            ),
            "arrival_reason_per_amb": list(self._arrival_reason),
            "arrival_rl_time_per_amb": self._arrival_rl_time.copy(),
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

    def mean_aoi_ms(self) -> float:
        """C4 metric: episode-mean AoI in ms (mean over K, over MAC ticks)."""
        if not self.aoi_history:
            return 0.0
        return float(np.mean(self.aoi_history) * 1e3)

    def aoi_violation_rate(self) -> float:
        """C5 metric: fraction of MAC ticks with an AoI-tail violation."""
        if not self.aoi_viol_history:
            return 0.0
        return sum(self.aoi_viol_history) / len(self.aoi_viol_history)
