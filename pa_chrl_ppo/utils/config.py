"""Simulation parameters — single source of truth.

Tất cả hằng số kéo trực tiếp từ docs/. KHÔNG hardcode lại ở module khác.

Cross-reference:
    - Hardware:           docs/03_architecture.md#hardware-spec
    - Phase QoS table:    docs/02_requirements.md#phase-table
    - CMDP thresholds:    docs/05_agent_workflow.md#cmdp-thresholds
    - Delay components:   docs/04_data_flow.md, docs/08_implementation_notes.md
    - RL hyperparams:     docs/09_execution_plan.md (Reference Table)
"""

from __future__ import annotations

from typing import Final

# ============================================================
# Hardware spec (3GPP TS 38.101-1, Viettel C-Band reference)
# ============================================================
P_TOTAL: Final[int] = 273              # Total PRB @ 100MHz μ=1 (3GPP TS 38.101-1 Table 5.3.2-1)
B_PRB: Final[float] = 360e3            # 360 kHz per PRB (12 subcarriers × 30 kHz SCS)
B_TOTAL: Final[float] = 100e6          # 100 MHz total bandwidth (Viettel C-Band)
TTI_SEC: Final[float] = 0.5e-3         # 0.5 ms TTI (μ=1, 30 kHz SCS)
F_CARRIER: Final[float] = 3.5e9        # 3.5 GHz carrier (FR1 n78)

# ============================================================
# Geometry anchor — single-cell @ Bệnh viện Bạch Mai, đường Giải Phóng
# Reference: docs/03_architecture.md, REFERENCE_MAP.md M2.0 (D25)
# ============================================================
# GPS thật (Google Maps) — map center / hospital anchor / route destination
# cho lớp mobility SUMO/OSM (W15). KHÔNG dùng trực tiếp trong RL env (env
# dùng local Cartesian mét, gNB tại gốc (0,0) = điểm hội tụ 3 xe trên
# Giải Phóng, đặt bên trong vùng OSM được anchor bởi tọa độ này).
BACH_MAI_LAT: Final[float] = 21.002965894776974
BACH_MAI_LON: Final[float] = 105.84078002433277
R_CELL_M: Final[float] = 300.0         # single-cell UMi radius, gNB=(0,0), no handover
NUM_RU: Final[int] = 20                # 20 O-RU in 3×3 km Hanoi grid (5×4)
F_MEC: Final[float] = 10e9             # 10 GHz MEC CPU budget per O-DU edge server.
                                        # Engineering assumption (edge server scale); ref Mlika 2021
                                        # "Network Slicing with MEC and DRL for IoV". See REFERENCE_MAP §2.
C_FH_BPS: Final[float] = 25e9          # 25 Gbps eCPRI fronthaul capacity per O-RU
                                        # (O-RAN.WG4 standard cho 32T32R Massive MIMO sub-6 GHz;
                                        # raw IQ ~94 Gbps cho 32T32R @ 100 MHz, sau 7.2x compression
                                        # ~24 Gbps. 25 GbE cho ~25% headroom)
HARQ_MAX_RETX: Final[int] = 3          # 3GPP NR standard
SHANNON_ETA: Final[float] = 0.75       # Link-adaptation efficiency (MCS/coding overhead vs Shannon limit).
                                        # 3GPP TR 36.942 §A reports ~0.7–0.8 for typical NR deployments.
                                        # ASSUMPTION: 0.75 is engineering mid-point; sensitivity {0.6,0.75,0.9}
                                        # in REFERENCE_MAP §5. See also Ji 2017 NR physical-layer efficiency.
TX_POWER_UE_DBM: Final[float] = 23.0   # 3GPP TS 38.101-1
MIMO_T: Final[int] = 32                # Qualcomm QRU100 32T32R
MIMO_R: Final[int] = 32

# ============================================================
# Delay components (deterministic + stochastic)
# Reference: docs/04_data_flow.md, docs/08_implementation_notes.md
# ============================================================
D_FH: Final[float] = 0.1e-3             # Fronthaul one-way latency (eCPRI/IEEE 802.1CM, O-RAN.WG4).
                                        # Typical eCPRI transport target ≤ 100 μs (O-RAN.WG4.CTR v03.00
                                        # Table 4-2); ASSUMPTION within this range. REFERENCE_MAP §2.
D_BH: Final[float] = 0.1e-3             # Backhaul O-CU→5GC one-way latency.
                                        # 3GPP TR 38.801 §6.3 transport budget: BH ≤ 1–10 ms for eMBB;
                                        # 0.1 ms is engineering lower-bound for Metro MEC topology.
                                        # ASSUMPTION — REFERENCE_MAP §2.
D_DET: Final[float] = 0.07e-3           # Deterministic processing delay (PDCCH decode + PDCP header).
                                        # 3GPP TS 38.214 §5.3 / TS 38.213 UE processing 3–4 OFDM symbols
                                        # ≈ 0.04–0.14 ms @ μ=1 (30 kHz SCS). 0.07 ms is midpoint.
                                        # ASSUMPTION — REFERENCE_MAP §2.
D_STOCH: Final[float] = 0.05e-3         # Stochastic RLC + retx mean (in M/G/1 σ²).
                                         # Reviewer PB-C2 fix (2026-05-27): chốt 0.05 ms
                                         # align với docs/13 §1.3 (Gemini M2: D_stoch ~ Uniform(0, 2·0.05ms)
                                         # → E[D_stoch] = 0.05 ms). Old value 0.15e-3 mâu thuẫn với
                                         # service-time distribution specification.
SAFETY_QP_PERIOD: Final[float] = 10e-3  # xApp QP control cycle
ODU_LOCAL_CHECK: Final[float] = 0.5e-3  # O-DU local 1-TTI check

# D_max_QP per phase = D_max^φ - D_det - D_fh - D_bh (air-side budget for QP)
# Reference: Patch fix in plan file
# D_max_QP^φ = D_max^φ − D_det(0.07) − D_fh(0.1) − D_bh(0.1) = D_max^φ − 0.27 ms.
# Only DETERMINISTIC fixed delays are subtracted. E[D_stoch]=0.05ms is NOT subtracted
# here because it is part of the M/G/1 service-time variance σ² (queue delay) that the
# air-side QP/scheduler controls — see oran_env.py:692 (d_tx = service − D_stoch; D_stoch
# enters d_queue via P-K σ²). Subtracting it again would double-count. (docs/04 §D_max_QP
# previously subtracted it in error — that line is the stale one, not this.)
# NOTE: currently UNUSED in training (IdentityNSF smoke regime); activates with real NSF/QP
# (sub-phase D).
D_MAX_QP_PHI: Final[dict[int, float]] = {
    1: 19.73e-3,       # 20 - 0.27
    2: 4.73e-3,        # 5  - 0.27
    3: 0.73e-3,        # ← critical: 1 - 0.07 - 0.1 - 0.1
    4: 1.73e-3,        # 2  - 0.27
    5: 19.73e-3,
}

# ============================================================
# Phase QoS table (5-phase FSM)
# References:
#   D_max^φ₃=1ms, ε=1e-5: 3GPP TS 22.261 V17.14.0 Annex D §D.1
#     "cycle time as low as 2ms → end-to-end latency constraint 1ms;
#      communication service availability 99.9999%" (Discrete automation)
#   D_max^φ₁,φ₅=20ms: 3GPP TS 22.261 Annex A Table A.1-1
#     "Real-time command/control for remote healthcare: 10-100ms"
#   Reliability ε=1e-5: 3GPP TS 22.261 §7.2 + Table 7.2.3.2-1
#     "wireless ITS infrastructure backhaul: reliability 99.999%"
#   Confirmed by Zexian Li 2018 §II: "1ms / 99.9999%" for URLLC.
#   See also: docs/02_requirements.md#phase-qos-table, docs/REFERENCE_MAP.md §2
# AoI_max medical thresholds (0.1–1.0 s): ENGINEERING/USE-CASE ASSUMPTION for
#   emergency telemetry (no direct medical-standards paper in corpus). Sensitivity
#   analysis ±50% scheduled in REFERENCE_MAP §5. See docs/REFERENCE_MAP.md §2.
# Units: seconds
# ============================================================
PHASE_QOS: Final[dict[int, dict[str, float]]] = {
    # φ₁ STANDBY — relaxed
    1: {
        "name": "STANDBY",
        "D_max": 20e-3, "eps": 1e-3,
        "AoI_max_HR": 1.0, "AoI_max_SpO2": 2.0, "AoI_max_BP": 5.0,
        "eps_aoi": 1e-2,
    },
    # φ₂ DISPATCH — tightening
    2: {
        "name": "DISPATCH",
        "D_max": 5e-3, "eps": 1e-4,
        "AoI_max_HR": 0.2, "AoI_max_SpO2": 0.5, "AoI_max_BP": 1.0,
        "eps_aoi": 1e-3,
    },
    # φ₃ SCENE — critical (cardiac, accident response)
    3: {
        "name": "SCENE",
        "D_max": 1e-3, "eps": 1e-5,
        "AoI_max_HR": 0.1, "AoI_max_SpO2": 0.2, "AoI_max_BP": 0.5,
        "eps_aoi": 1e-3,
    },
    # φ₄ TRANSPORT — high but slightly relaxed
    4: {
        "name": "TRANSPORT",
        "D_max": 2e-3, "eps": 1e-5,
        "AoI_max_HR": 0.1, "AoI_max_SpO2": 0.2, "AoI_max_BP": 0.5,
        "eps_aoi": 1e-3,
    },
    # φ₅ RETURN — relaxed
    5: {
        "name": "RETURN",
        "D_max": 20e-3, "eps": 1e-3,
        "AoI_max_HR": 1.0, "AoI_max_SpO2": 2.0, "AoI_max_BP": 5.0,
        "eps_aoi": 1e-2,
    },
}

# Phase reward weights α(φ). Post-restructure (2026-05-26) reward is SINGLE-TERM
# eMBB log-utility: r = α_eMBB(φ) · log(1 + R_eMBB/R_REF)  (oran_env.py:515).
# Only the "embb" weight enters the reward. The "urllc" weight is RETAINED for
# diagnostics / legacy ablation only — URLLC is enforced via Lagrangian λ_1, λ_2
# (C1, C2 hard constraints), NOT via a reward penalty. See docs/13 §2.1.
# Reference: docs/05_agent_workflow.md; docs/REFERENCE_MAP.md §2 (α_φ = design assumption).
PHASE_ALPHA: Final[dict[int, dict[str, float]]] = {
    1: {"urllc": 0.3, "embb": 0.7},
    2: {"urllc": 0.6, "embb": 0.4},
    3: {"urllc": 0.95, "embb": 0.05},
    4: {"urllc": 0.8, "embb": 0.2},
    5: {"urllc": 0.3, "embb": 0.7},
}

# ============================================================
# CMDP constraint thresholds d_j^φ
# Reference: docs/05_agent_workflow.md#cmdp-thresholds
# Used in: λ_j ← max(0, λ_j + α_λ · (J_Cj − d_j^φ))
# ============================================================
CMDP_D_J_PHI: Final[dict[int, dict[str, float]]] = {
    1: {"d1_lat_mean": 20e-3, "d2_lat_tail": 1e-3,  "d3_embb_mbps": 10.0, "d4_aoi_mean": 1.0, "d5_aoi_tail": 1e-2},
    2: {"d1_lat_mean": 5e-3,  "d2_lat_tail": 1e-4,  "d3_embb_mbps": 20.0, "d4_aoi_mean": 0.2, "d5_aoi_tail": 1e-3},
    3: {"d1_lat_mean": 1e-3,  "d2_lat_tail": 1e-5,  "d3_embb_mbps": 30.0, "d4_aoi_mean": 0.1, "d5_aoi_tail": 1e-3},
    4: {"d1_lat_mean": 2e-3,  "d2_lat_tail": 1e-5,  "d3_embb_mbps": 30.0, "d4_aoi_mean": 0.1, "d5_aoi_tail": 1e-3},
    5: {"d1_lat_mean": 20e-3, "d2_lat_tail": 1e-3,  "d3_embb_mbps": 10.0, "d4_aoi_mean": 1.0, "d5_aoi_tail": 1e-2},
}

# λ_warm table — post-training expected values
# Reference: docs/05_agent_workflow.md:174-180
LAMBDA_WARM: Final[dict[int, list[float]]] = {
    1: [0.02, 0.01, 0.00, 0.01, 0.00],
    2: [0.15, 0.08, 0.02, 0.05, 0.02],
    3: [1.80, 2.20, 0.10, 1.50, 2.00],
    4: [1.20, 1.50, 0.08, 1.20, 1.50],
    5: [0.02, 0.01, 0.00, 0.01, 0.00],
}

# ============================================================
# Traffic classes
# Reference: docs/02_requirements.md#traffic-classes
# ============================================================
TRAFFIC_CLASSES: Final[dict[str, dict]] = {
    "URLLC_C1_DENM": {
        "size_bytes": (300, 800), "arrival": "event_burst",
        "D_max": 1e-3, "eps": 1e-5, "lambda_base": 50.0,
    },
    "URLLC_C2_VITAL": {
        "size_bytes": (100, 500), "arrival": "100Hz_periodic",
        "D_max": 5e-3, "eps": 1e-5, "lambda_base": 100.0,
    },
    "URLLC_C3_CAM": {
        "size_bytes": (200, 400), "arrival": "10Hz_periodic",
        "D_max": 3e-3, "eps": 1e-4, "lambda_base": 10.0,
    },
    "eMBB_V1_VIDEO4K": {
        "size_bytes": (1500, 1500), "arrival": "CBR_VBR",
        "D_max": 100e-3, "eps": 1e-3, "rate_mbps": 5.0,
    },
    "eMBB_V2_IMAGE_MEC": {
        "size_bytes": (1500, 1500), "arrival": "aperiodic",
        "D_max": 50e-3, "eps": None, "rate_mbps": 1.0,
    },
    "mMTC_IOT": {
        "size_bytes": (50, 100), "arrival": "sparse",
        "D_max": 1.0, "eps": None, "lambda_base": 0.1,
    },
}

# ============================================================
# RL hyperparameters
# Reference: docs/09_execution_plan.md (Reference Table)
# ============================================================
T_MAX_EPISODES: Final[int] = 10000
STEPS_PER_EPISODE_LOW: Final[int] = 100   # 1 second / T_int=10ms = 100 low-level steps
PPO_CLIP_EPS: Final[float] = 0.2
GAMMA: Final[float] = 0.99
GAE_LAMBDA: Final[float] = 0.95
PPO_K_EPOCHS: Final[int] = 10
MINIBATCH_SIZE: Final[int] = 64

# Dual ascent for Lagrangian λ (Phase 2.3.3 locked value, 2026-05-20).
# Hierarchy: α_πH (1e-5) < α_λ (1e-4) < α_πL (1e-3)
# Rationale (two-timescale dual ascent — Spoor 2025 / Ding 2023; HRL timescale Akyıldız 2024):
#   - α_λ must be slower than α_πL → primal stability
#   - α_λ = 0.1 × α_πL provides timescale separation while remaining responsive
#     to constraint violations (faster than Manager policy update)
# Old value 0.01 (10× too fast vs primal) → corrected to 1e-4.
ALPHA_LAMBDA_DUAL: Final[float] = 1e-4

# Learning rates per network
# Borkar 2008 two-timescale alignment: α_πH ≪ α_πL (2 orders of magnitude).
# Old values 1e-4 / 3e-4 had ratio 1/3 ≈ 0.33 → too close; Worker noise leaks
# into Manager slow update. New 1e-5 / 1e-3 → ratio 0.01 ≪ 1 (heuristic OK).
# Reference: docs/13_methodology_walkthrough.md Phase 1.4 Borkar correction.
LR_PI_H: Final[float] = 1e-5            # Manager / rApp (slow)
LR_V_H: Final[float] = 5e-5             # Manager critic (slow)
LR_PI_L: Final[float] = 1e-3            # Worker / xApp (fast)
LR_V_L: Final[float] = 1e-3             # Worker critic (fast)
LR_LSTM: Final[float] = 5e-4
LR_NSF: Final[float] = 1e-3

# Safety / β_qp anneal
# W09 β_qp diagnostic verdict (2026-05-24, updated post-n=15):
# Constant β_qp = 0.6 over the anneal schedule for IdentityNSF smoke regime.
# Empirical justification: at 300 ep × n=15/15/10 A/B, β_qp = 0.6 delivers
# 80% balance rate (12/15) vs 66.7% (10/15) at 0.5 vs 70% (7/10) at 0.7.
# β_qp = 0.6 also has lowest total trap count (3 traps vs 5 at 0.5, 3 at 0.7)
# and second-tightest balanced-cluster reward variance.
# Working regime is the [0.5, 0.6] plateau; difference NOT statistically
# significant (z=-0.83, p>0.05) but trend favors 0.6.
# See logs_w9/summary.md §6 for full analysis.
# Revert to anneal logic when real NSF MLP arrives (sub-phase D).
BETA_QP_INIT: Final[float] = 0.6
BETA_QP_FINAL: Final[float] = 0.6
BETA_QP_ANNEAL_EPISODES: Final[int] = 5000  # T_anneal for linear schedule (Phase 3.2.2)
BETA_QP_T_ANNEAL: Final[int] = BETA_QP_ANNEAL_EPISODES  # alias used by train.py

# Worker observation layout indices (33-dim formal spec, post LSTM+MEC removal)
# Used by train.py + ablation baselines to overlay λ_local + mask phase
PHASE_OH_OBS_INDEX: Final[int] = 10        # phase one-hot at obs[10:15]
LAMBDA_LOCAL_OBS_INDEX: Final[int] = 17    # λ_local at obs[17:22]

# ============================================================
# Phase 2.1 reward normalization (docs/13 Phase 2.1, post-restructure 2026-05-26)
# SINGLE-TERM reward (oran_env.py:515):
#   r = α_eMBB(φ) · U_eMBB ,  U_eMBB = log(1 + R_eMBB / R_REF_EMBB)   (bounded)
# URLLC enforced ONLY via Lagrangian C1, C2 (λ_1, λ_2) — NOT in reward.
# L_URLLC = mean(D_e2e)/D_REF_URLLC is computed for DIAGNOSTICS only (info dict).
# (DEPRECATED 2-term form r = -α_U·L_URLLC + α_e·U_eMBB removed: double-counted
#  URLLC with λ_1, λ_2 → dual stagnation. See docs/13 §2.1 restructure note.)
# ============================================================
D_REF_URLLC: Final[float] = 1e-3        # 1 ms = tightest D_max (φ_3 SCENE)
R_REF_EMBB_MBPS: Final[float] = 100.0   # eMBB log-utility normalization anchor (worst-case φ_3 capacity).
                                        # ASSUMPTION: 100 Mbps is engineering estimate for available eMBB
                                        # throughput after URLLC slice reservation. docs/13:671 justifies
                                        # as "eMBB cap at worst-case φ_3". Sensitivity sweep {50,100,200,300}
                                        # scheduled in REFERENCE_MAP §5.

# Reviewer M4 (Gemini Section II W06, 2026-05-27):
# Lagrangian projection upper bound — Π_Λ(λ) = clip(λ, 0, LAMBDA_MAX).
# Prevents dual ascent blow-up under sustained constraint violations
# (Exp11 Robustness sensor failure scenarios). Empirical λ ≤ 2.5 across
# W11 10-seed × 1000-episode runs → LAMBDA_MAX = 10.0 is a soft safety
# net, NOT an active constraint at convergence.
# See docs/13 §2.3.3 + agents/lagrangian.py:191.
LAMBDA_MAX: Final[float] = 10.0

# Reviewer Mn1 (Gemini Section II W08, 2026-05-27):
# β_qp anneal floor — residual policy-distillation pull preventing
# catastrophic forgetting of NSF/QP safety boundaries at end-of-training.
# Without floor, β_qp → 0 would let PPO drift away from QP imitation →
# safety boundary violations under out-of-distribution conditions.
# See docs/13 §3.2.2 + agents/pa_chrl_ppo.py.
BETA_QP_FLOOR: Final[float] = 0.05

# Hierarchical time scales (corrected 2026-05-20 to comply with O-RAN spec)
#
#   Real deployment (O-RAN.WG3 Near-RT RIC + O-RAN.WG2 Non-RT RIC):
#     MAC TTI       = 0.5 ms (3GPP TS 38.211 μ=1, O-DU internal)
#     Worker (xApp) = 10 ms (1 RRMPolicyRatio update per 10 ms)
#     Manager (rApp)= 1 s (policy + Lagrangian dual update)
#
#   Simulation (Hướng B — compressed Manager 10× for tractable training):
#     MAC TTI       = 0.5 ms (unchanged)
#     Worker (xApp) = 10 ms (= 20 MAC ticks, unchanged from real)
#     Manager (rApp)= 100 ms sim (= 10 Worker steps; 1 s real)
#
#   Borkar 2008 two-timescale theorem applies: α_πH ≪ α_πL preserved
#   regardless of T_H / T_L ratio. Compression is conservative direction.

T_TTI_SEC: Final[float] = 0.5e-3            # MAC tick (3GPP TS 38.211)
T_L_SEC: Final[float] = 10e-3               # Worker/xApp step (O-RAN Near-RT RIC)
T_H_SEC: Final[float] = 100e-3              # Manager/rApp step (sim — compressed)
T_H_REAL_SEC: Final[float] = 1.0            # Manager/rApp step (real deployment)

# Derived ratios
MAC_TICKS_PER_WORKER: Final[int] = 20       # T_L / T_TTI = 10 ms / 0.5 ms
WORKER_STEPS_PER_MANAGER: Final[int] = 10   # T_H / T_L = 100 ms / 10 ms (sim)

# Worker / Manager discount factors (Phase 3.2.4 — distinct notation, no clash)
# γ_H = γ_L^W ensures both levels see same effective horizon in wall-clock time.
GAMMA_WORKER: Final[float] = GAMMA                                # = 0.99 per Worker step (10 ms)
GAMMA_MANAGER: Final[float] = GAMMA ** WORKER_STEPS_PER_MANAGER   # = 0.99^10 ≈ 0.904 per Manager step

# Legacy aliases (DO NOT use in new code — kept for backward compat in tests)
T_H_REAL: Final[float] = T_H_REAL_SEC       # was 1.0
T_H_SIM: Final[float] = T_H_SEC             # was 10e-3 — now 100e-3 (semantic fix)
T_L_SIM: Final[float] = T_TTI_SEC           # was 0.5e-3 — same value, semantic was wrong

T_INT_RANGE: Final[tuple[float, float]] = (10e-3, 100e-3)  # xApp T_int learned range

# ============================================================
# Hysteresis (anti ping-pong handover)
# Reference: docs/05_agent_workflow.md
# ============================================================
HYSTERESIS_RSRP_DB: Final[float] = 3.0
T_GUARD_SEC: Final[float] = 2.0

# Proactive handover trigger (per Algorithm 3)
HANDOVER_ETA_TRIGGER: Final[float] = 10.0   # seconds; pre-allocate when ETA < 10s

# Pre-tightening
PRE_TIGHTEN_ETA: Final[float] = 30.0        # seconds; apply D_max^φ_next if ETA_next < 30s

# ============================================================
# NSF / OSQP runtime
# Reference: docs/08_implementation_notes.md Bảng 4.2C
# ============================================================
NSF_TARGET_RUNTIME_MS: Final[float] = 1.0       # NSF target < 1ms
OSQP_FALLBACK_RUNTIME_MS: Final[float] = 2.0    # OSQP fallback budget
OSQP_MAX_ITER: Final[int] = 200
OSQP_TIME_LIMIT_SEC: Final[float] = 2.0e-3
OSQP_EPS_ABS: Final[float] = 1e-4
OSQP_EPS_REL: Final[float] = 1e-4

# ============================================================
# LSTM specs
# Reference: docs/05_agent_workflow.md#lstm
# ============================================================
LSTM_HIDDEN_1: Final[int] = 64
LSTM_HIDDEN_2: Final[int] = 32
LSTM_HEADS: Final[int] = 6   # P_overload, PRB_demand, BLER, trajectory, ETA, target_cell_id
LSTM_LOOKBACK_SEC: Final[float] = 30.0
LSTM_INFERENCE_PERIOD_MS: Final[float] = 500.0


def get_phase_thresholds(phi: int) -> dict[str, float]:
    """Master Table helper — per-step phase threshold lookup (docs/13 Phase 1.3).

    Returns 5-key dict matching constraint signals C1-C5 (Phase 2.2.1):
        d1: D_max^φ     (URLLC mean latency, seconds)
        d2: ε^φ          (URLLC tail probability)
        d3: 0            (eMBB signed throughput gap threshold; R_min is in CMDP_D_J_PHI)
        d4: AoI_max^φ_HR (AoI mean for aggregated vitals, seconds)
        d5: ε_AoI^φ      (AoI tail probability)

    Used trong:
        - env.step() info dict (per Worker tick lookup)
        - LambdaState.accumulate() (dual subgradient per-step deviation)
        - Worker augmented reward r^aug = r - Σ λ_j · (c_j - d_j^φ_t)
    """
    if phi not in PHASE_QOS:
        raise ValueError(f"Invalid phase {phi}; must be 1..5")
    qos = PHASE_QOS[phi]
    return {
        "d1": float(qos["D_max"]),               # URLLC mean latency budget (seconds)
        "d2": float(qos["eps"]),                 # URLLC tail probability budget
        "d3": 0.0,                                # eMBB signed gap threshold; floor is in CMDP_D_J_PHI
        "d4": float(qos["AoI_max_HR"]),          # AoI mean budget (HR aggregated, seconds)
        "d5": float(qos["eps_aoi"]),             # AoI tail probability budget
    }


def get_phase_alpha(phi: int) -> tuple[float, float]:
    """Phase-weighted reward coefficients α_U(φ), α_e(φ) (Phase 2.1).

    Returns (alpha_urllc, alpha_embb) tuple. Sum to 1.0 per phase.
    """
    if phi not in PHASE_ALPHA:
        raise ValueError(f"Invalid phase {phi}; must be 1..5")
    pa = PHASE_ALPHA[phi]
    return float(pa["urllc"]), float(pa["embb"])


def summary() -> str:
    """One-line summary for sanity checking."""
    return (
        f"PA-CHRL-PPO config: P_total={P_TOTAL} PRB, "
        f"B_PRB={B_PRB/1e3:.0f}kHz, "
        f"f_c={F_CARRIER/1e9:.1f}GHz, "
        f"F_MEC={F_MEC/1e9:.0f}GHz, "
        f"phases={len(PHASE_QOS)}, "
        f"D_max^φ₃={PHASE_QOS[3]['D_max']*1e3:.2f}ms, "
        f"γ_L={GAMMA_WORKER:.3f}, γ_H={GAMMA_MANAGER:.4f}"
    )


if __name__ == "__main__":
    print(summary())
