"""Simulation parameters — single source of truth.

Tất cả hằng số kéo trực tiếp từ docs/. KHÔNG hardcode lại ở module khác.

Cross-reference:
    - Hardware:           docs/03_architecture.md#hardware-spec
    - Severity QoS table: docs/02_requirements.md#severity-qos-table
    - CMDP thresholds:    docs/05_agent_workflow.md#cmdp-thresholds
    - Delay components:   docs/04_data_flow.md, docs/08_implementation_notes.md
    - RL hyperparams:     docs/09_execution_plan.md (Reference Table)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np

# ============================================================
# Hardware spec (3GPP TS 38.101-1, Viettel C-Band reference)
# ============================================================
P_TOTAL: Final[int] = 273              # Total PRB @ 100MHz μ=1 (3GPP TS 38.101-1 Table 5.3.2-1)
B_PRB: Final[float] = 360e3            # 360 kHz per PRB (12 subcarriers × 30 kHz SCS)
B_TOTAL: Final[float] = 100e6          # 100 MHz total bandwidth (Viettel C-Band)
TTI_SEC: Final[float] = 0.5e-3         # 0.5 ms TTI (μ=1, 30 kHz SCS)
F_CARRIER: Final[float] = 3.5e9        # 3.5 GHz carrier (FR1 n78)

# ============================================================
# Geometry anchor — single-cell gNB @ Bệnh viện Bạch Mai, đường Giải Phóng
# Reference: docs/03_architecture.md, REFERENCE_MAP.md M2.0 (D25)
# ============================================================
# GPS thật (Google Maps) — tọa độ gNB TRÙNG tọa độ BV Bạch Mai (distance=0):
# gNB đặt tại chính vị trí BV, 3 xe hội tụ về đây khi đến viện. Dùng làm map
# center / route destination cho lớp mobility SUMO/OSM (W15). KHÔNG dùng trực
# tiếp trong RL env (env dùng local Cartesian mét, gNB tại gốc (0,0) tương
# ứng đúng tọa độ GPS này).
BACH_MAI_LAT: Final[float] = 21.002965894776974
BACH_MAI_LON: Final[float] = 105.84078002433277
R_CELL_M: Final[float] = 300.0         # single-cell UMi radius quanh gNB=BV=(0,0), no handover
NUM_RU: Final[int] = 20                # RESERVED — multi-cell grid (5×4 O-RU); UNUSED in the
                                        # current single-cell UMi env. Provenance for a future
                                        # multi-cell extension. See REFERENCE_MAP §2. (audit 2026-06-16)
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
                                         # align với docs/13 §1.3 (internal review M2: D_stoch ~ Uniform(0, 2·0.05ms)
                                         # → E[D_stoch] = 0.05 ms). Old value 0.15e-3 mâu thuẫn với
                                         # service-time distribution specification.
SAFETY_QP_PERIOD: Final[float] = 10e-3  # xApp QP control cycle
ODU_LOCAL_CHECK: Final[float] = 0.5e-3  # O-DU local 1-TTI check

# ============================================================
# Severity QoS table (5-level patient-urgency tier — replaces the old 5-phase
# FSM, 2026-06-14). Severity is an EXOGENOUS per-episode attribute of the patient
# on board (fixed within an episode, re-sampled across episodes); it — not the
# ambulance mission stage — selects how strict the QoS targets are. Monotonic:
# level 1 (loosest) → level 5 (tightest).
# References (endpoints reuse the calibrated old-phase values):
#   D_max(Immediate)=1ms, ε=1e-5: 3GPP TS 22.261 V17.14.0 Annex D §D.1
#     "cycle time as low as 2ms → end-to-end latency constraint 1ms;
#      communication service availability 99.9999%" (Discrete automation)
#   D_max(Non-urgent)=20ms: 3GPP TS 22.261 Annex A Table A.1-1
#     "Real-time command/control for remote healthcare: 10-100ms"
#   Reliability ε=1e-5: 3GPP TS 22.261 §7.2 + Table 7.2.3.2-1
#     "wireless ITS infrastructure backhaul: reliability 99.999%"
#   Confirmed by Zexian Li 2018 §II: "1ms / 99.9999%" for URLLC.
#   See also: docs/02_requirements.md#severity-qos-table, docs/REFERENCE_MAP.md §2
# AoI_max medical thresholds (0.1–1.0 s): ENGINEERING/USE-CASE ASSUMPTION for
#   emergency telemetry (no direct medical-standards paper in corpus). The tightest
#   value (0.1 s @ severity 4-5) mirrors the order of D_max^sev-5 (1 ms latency →
#   0.1 s freshness, 100× looser, status-update vs packet-delay timescales); the
#   loosest (1.0 s @ severity 1) matches relaxed non-urgent monitoring. Absolute
#   values NOT claimed as clinical fact. Sensitivity ±50% scheduled in REFERENCE_MAP §5.
# Units: seconds
# ============================================================
SEVERITY_QOS: Final[dict[int, dict[str, float]]] = {
    # 1 NON_URGENT — stable patient, low immediate risk (loosest)
    1: {
        "name": "NON_URGENT",
        "D_max": 20e-3, "eps": 1e-3,
        "AoI_max": 1.0,
        "eps_aoi": 1e-2,
    },
    # 2 SEMI_URGENT — symptomatic / needs monitoring, not yet pressing
    2: {
        "name": "SEMI_URGENT",
        "D_max": 10e-3, "eps": 1e-4,
        "AoI_max": 0.5,
        "eps_aoi": 1e-3,
    },
    # 3 URGENT — significant priority, may deteriorate if delayed
    3: {
        "name": "URGENT",
        "D_max": 5e-3, "eps": 1e-4,
        "AoI_max": 0.2,
        "eps_aoi": 1e-3,
    },
    # 4 EMERGENCY — high risk, time-critical handling
    4: {
        "name": "EMERGENCY",
        "D_max": 2e-3, "eps": 1e-5,
        "AoI_max": 0.1,
        "eps_aoi": 1e-3,
    },
    # 5 IMMEDIATE — life-threatening, strongest QoS protection (tightest)
    5: {
        "name": "IMMEDIATE",
        "D_max": 1e-3, "eps": 1e-5,
        "AoI_max": 0.1,
        "eps_aoi": 1e-3,
    },
}

# Severity reward weights α(severity). Post-restructure (2026-05-26) reward is
# SINGLE-TERM eMBB log-utility: r = α_eMBB(sev) · log(1 + R_eMBB/R_REF) (oran_env).
# Only the "embb" weight enters the reward. The "urllc" weight is RETAINED for
# diagnostics / legacy ablation only — URLLC is enforced via Lagrangian λ_1, λ_2
# (C1, C2 hard constraints), NOT via a reward penalty. See docs/13 §2.1.
# Monotonic: higher severity → lower α_embb (eMBB deprioritized, PRB → URLLC).
# Reference: docs/05_agent_workflow.md; docs/REFERENCE_MAP.md §2 (design assumption).
SEVERITY_ALPHA: Final[dict[int, dict[str, float]]] = {
    1: {"urllc": 0.30, "embb": 0.70},   # NON_URGENT
    2: {"urllc": 0.45, "embb": 0.55},   # SEMI_URGENT
    3: {"urllc": 0.60, "embb": 0.40},   # URGENT
    4: {"urllc": 0.80, "embb": 0.20},   # EMERGENCY
    5: {"urllc": 0.95, "embb": 0.05},   # IMMEDIATE
}

# ============================================================
# CMDP constraint thresholds d_j^sev (per severity level)
# Reference: docs/05_agent_workflow.md#cmdp-thresholds
# Used in: λ_j ← max(0, λ_j + α_λ · (J_Cj − d_j^sev))
# ============================================================
# d3_embb_mbps (eMBB throughput floor) is NON-INCREASING in severity (audit
# 2026-06-16 fix): higher severity → URLLC prioritized → LESS eMBB guaranteed,
# consistent with SEVERITY_ALPHA["embb"] which also drops (0.70→0.05). The
# previous increasing table [10,15,20,30,30] pulled AGAINST URLLC priority
# (reward deprioritized eMBB while C3 demanded MORE eMBB) — a contradiction.
CMDP_D_J_SEVERITY: Final[dict[int, dict[str, float]]] = {
    1: {"d1_lat_mean": 20e-3, "d2_lat_tail": 1e-3,  "d3_embb_mbps": 30.0, "d4_aoi_mean": 1.0, "d5_aoi_tail": 1e-2},
    2: {"d1_lat_mean": 10e-3, "d2_lat_tail": 1e-4,  "d3_embb_mbps": 25.0, "d4_aoi_mean": 0.5, "d5_aoi_tail": 1e-3},
    3: {"d1_lat_mean": 5e-3,  "d2_lat_tail": 1e-4,  "d3_embb_mbps": 20.0, "d4_aoi_mean": 0.2, "d5_aoi_tail": 1e-3},
    4: {"d1_lat_mean": 2e-3,  "d2_lat_tail": 1e-5,  "d3_embb_mbps": 15.0, "d4_aoi_mean": 0.1, "d5_aoi_tail": 1e-3},
    5: {"d1_lat_mean": 1e-3,  "d2_lat_tail": 1e-5,  "d3_embb_mbps": 10.0, "d4_aoi_mean": 0.1, "d5_aoi_tail": 1e-3},
}

# λ_warm table [C1, C2, C3, C4, C5] — per-severity warm-start. Overall λ grows
# with severity (mean), BUT the C3 slot (index 2, eMBB-floor dual) is
# NON-INCREASING [0.10→0.02] (audit 2026-06-17): co-directional with the
# now non-increasing d3_embb floor — at high severity the eMBB floor is low and
# easily met, so C3 is rarely binding ⟹ lower warm-start dual. (Previously the
# C3 slot was increasing, matching the old increasing d3_embb — flipped together.)
# Reference: docs/05_agent_workflow.md:174-180
LAMBDA_WARM: Final[dict[int, list[float]]] = {
    1: [0.02, 0.01, 0.10, 0.01, 0.00],   # NON_URGENT  (C3 highest: eMBB floor 30 Mbps)
    2: [0.15, 0.08, 0.08, 0.05, 0.02],   # SEMI_URGENT
    3: [0.60, 0.70, 0.05, 0.50, 0.60],   # URGENT
    4: [1.20, 1.50, 0.03, 1.20, 1.50],   # EMERGENCY
    5: [1.80, 2.20, 0.02, 1.50, 2.00],   # IMMEDIATE   (C3 lowest: eMBB floor 10 Mbps)
}

# ============================================================
# Traffic classes — REFERENCE TABLE ONLY (NOT wired into the env).
# ⚠️ The per-class D_max here is documentation/provenance; the live CMDP
# latency constraint C1 uses SEVERITY_QOS[severity]["D_max"] (per-ambulance
# severity), NOT these traffic-class values. Do not read constraint budgets
# from this dict. (audit 2026-06-16)
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

# ============================================================
# Observation layout — SINGLE SOURCE OF TRUTH (obs_dim = OBS_FIXED_BLOCK_LEN
# + OBS_PER_AMB_BLOCK_LEN·K + F; K=1,F=1 → 31). Every consumer — env._observe(),
# mask_severity(), overlay_lambda_local(), build_manager_state(), docs/07_api_spec
# — MUST import these constants. Do NOT hardcode the integer indices anywhere.
# A layout-lock test (tests/test_obs_layout.py) asserts _observe() places each
# field at its named index. (audit 2026-06-17, full-SSOT Phương án 1)
#
# Fixed block obs[0:20] — Index | Field | Meaning | _observe() source
#   0  rho_urllc          URLLC queue util ρ=λ/μ (mean/K, clip[0,1])  np.mean(queues[urllc_k].rho)
#   1  rho_emBB           eMBB queue util ρ (clip[0,1])               queues["eMBB"].rho
#   2  hol_urllc_ms       URLLC head-of-line delay (ms, mean/K, ≤100) mean(queues[urllc_k].hol_delay)·1e3
#   3  hol_emBB_ms        eMBB HOL delay (ms, ≤1000)                  queues["eMBB"].hol_delay·1e3
#   4  r_min_urllc        PRB ratio r_min^URLLC (LIVE, drifts/window) prb_ratios[0]=self.r_min_urllc
#   5  r_max_emBB         PRB ratio r_max^eMBB                        prb_ratios[1]=self.r_max_emBB
#   6  r_ded_urllc        PRB ratio r_ded^URLLC                       prb_ratios[2]=self.r_ded_urllc
#   7  arr_urllc          URLLC arrival (Σ over K)/1e3                Σ queues[urllc_k].arrival_rate/1e3
#   8  arr_emBB           eMBB arrival /1e4                           queues["eMBB"].arrival_rate/1e4
#   9  bler               mean BLER                                   float(self.last_bler)
#   10:15 severity_oh[5]  severity_ref one-hot (lvl 1..5)             sev_oh[self.severity-1]=1
#   15 lambda_c3_shared   λ_local shared C3 (eMBB-floor dual)         self._lambda_local[4K]
#   16 r_min_urllc_anchor Manager setpoint anchor (FIXED/window)      self.r_min_urllc_anchor
#   17 n_bys              bystander UE count / M_eMBB                 bystander.active_ue_count/M_eMBB
#   18 aoi_mean           mean AoI over K (s)                         aoi_per_amb.mean()
#   19 aoi_max            max AoI over K (s)                          aoi_per_amb.max()
# ============================================================
OBS_RHO_URLLC_IDX: Final[int] = 0
OBS_RHO_EMBB_IDX: Final[int] = 1
OBS_HOL_URLLC_IDX: Final[int] = 2
OBS_HOL_EMBB_IDX: Final[int] = 3
OBS_R_MIN_URLLC_IDX: Final[int] = 4
OBS_R_MAX_EMBB_IDX: Final[int] = 5
OBS_R_DED_URLLC_IDX: Final[int] = 6
OBS_ARR_URLLC_IDX: Final[int] = 7
OBS_ARR_EMBB_IDX: Final[int] = 8
OBS_BLER_IDX: Final[int] = 9
OBS_SEVERITY_OH_IDX: Final[int] = 10       # one-hot occupies [10:15]
OBS_SEVERITY_OH_LEN: Final[int] = 5
OBS_LAMBDA_C3_IDX: Final[int] = 15
OBS_RMIN_ANCHOR_IDX: Final[int] = 16
OBS_N_BYS_IDX: Final[int] = 17
OBS_AOI_MEAN_IDX: Final[int] = 18
OBS_AOI_MAX_IDX: Final[int] = 19
OBS_FIXED_BLOCK_LEN: Final[int] = 20       # fixed block obs[0:20]
OBS_PER_AMB_BLOCK_LEN: Final[int] = 10     # per-ambulance block, 10 dims/amb

# Backward-compat aliases (existing importers reference these names)
SEVERITY_OH_OBS_INDEX: Final[int] = OBS_SEVERITY_OH_IDX        # severity_ref one-hot at obs[10:15]
SEVERITY_OH_LEN: Final[int] = OBS_SEVERITY_OH_LEN
LAMBDA_C3_SHARED_OBS_INDEX: Final[int] = OBS_LAMBDA_C3_IDX     # shared C3 λ_local at obs[15]

# Per-ambulance block offsets (relative to OBS_FIXED_BLOCK_LEN + 10*k):
#   SINR_k, d_k, v_k, delay_norm_k, AoI_norm_k, severity_k_norm,
#   λC1_k, λC2_k, λC4_k, λC5_k
AMB_SINR_OFFSET: Final[int] = 0
AMB_DIST_OFFSET: Final[int] = 1
AMB_SPEED_OFFSET: Final[int] = 2
AMB_DELAY_NORM_OFFSET: Final[int] = 3
AMB_AOI_NORM_OFFSET: Final[int] = 4
AMB_SEVERITY_NORM_OFFSET: Final[int] = 5
AMB_LAMBDA_C1_OFFSET: Final[int] = 6
AMB_LAMBDA_C2_OFFSET: Final[int] = 7
AMB_LAMBDA_C4_OFFSET: Final[int] = 8
AMB_LAMBDA_C5_OFFSET: Final[int] = 9

# ============================================================
# B5: severity_k priority temperature (β) + intra-slice Π_feasible PRB split
# (per-ambulance severity epic, 2026-06-15). β is Worker action a[6] (K≥2 only):
#   beta = BETA_MIN + (BETA_MAX - BETA_MIN) * sigmoid(a[6])
# Intra-slice split: b = max(floor(κ·B_U/K), PRB_MIN_QOS), feasibility fallback
# b = B_U // K if K·b > B_U; remainder S = B_U - K·b distributed via
# w = softmax(β·(severity_per_amb/5) + δ·ũ), δ = ρ·β. Severity is NORMALIZED
# (÷5, same as obs severity_k_norm) so β is a pure GLOBAL GAIN on unit-scale
# priority. At K=1, softmax([x]) = [1.0] always ⟹ PRB_0 = B_U regardless of
# parameters (exact K=1 preservation).
# ============================================================
# BETA_MIN > 0 (main method, audit 2026-06-16): guarantees a MINIMUM severity
# ordering in the PRB split — the agent cannot learn β→0 to flatten priority
# (severity-aware prioritization is a core contribution, not optional). At
# BETA_MIN=0.5 with normalized severity (spread 0.8) the sev5:sev1 weight ratio
# floor ≈ exp(0.5·0.8) ≈ 1.49×; at BETA_MAX=5 it reaches ≈ exp(4) ≈ 55×.
BETA_MIN: Final[float] = 0.5
BETA_MAX: Final[float] = 5.0
INTRA_SLICE_KAPPA: Final[float] = 0.5      # floor fraction: b = floor(κ·B_U/K)
PRB_MIN_QOS: Final[int] = 1                # minimum PRB floor per ambulance
RHO_URGENCY_TIEBREAK: Final[float] = 0.15  # δ = ρ·β; severity term (β·0.8) dominates tiebreaker (β·0.15)

# ============================================================
# Phase 2.1 reward normalization (docs/13 Phase 2.1, post-restructure 2026-05-26)
# SINGLE-TERM reward (oran_env.py:515):
#   r = α_eMBB(φ) · U_eMBB ,  U_eMBB = log(1 + R_eMBB / R_REF_EMBB)   (bounded)
# URLLC enforced ONLY via Lagrangian C1, C2 (λ_1, λ_2) — NOT in reward.
# L_URLLC = mean(D_e2e)/D_REF_URLLC is computed for DIAGNOSTICS only (info dict).
# (DEPRECATED 2-term form r = -α_U·L_URLLC + α_e·U_eMBB removed: double-counted
#  URLLC with λ_1, λ_2 → dual stagnation. See docs/13 §2.1 restructure note.)
# ============================================================
D_REF_URLLC: Final[float] = 1e-3        # 1 ms = tightest D_max (severity 5 IMMEDIATE)
R_REF_EMBB_MBPS: Final[float] = 100.0   # eMBB log-utility normalization anchor (worst-case φ_3 capacity).
                                        # ASSUMPTION: 100 Mbps is engineering estimate for available eMBB
                                        # throughput after URLLC slice reservation. docs/13:671 justifies
                                        # as "eMBB cap at worst-case φ_3". Sensitivity sweep {50,100,200,300}
                                        # scheduled in REFERENCE_MAP §5.
# AoI dual-gradient normalization scale (NOT a constraint threshold — the AoI
# budgets live in SEVERITY_QOS["AoI_max"] / CMDP_D_J_SEVERITY["d4_aoi_mean"]).
# Mirrors D_REF_URLLC: 0.1 s = tightest AoI_max (severity 5 IMMEDIATE). Used ONLY
# to scale the C4 subgradient so its magnitude matches C1/C2/C3/C5; without it
# C4's deviation (raw seconds) is ~10× weaker than C1's and AoI is under-weighted.
# See agents/lagrangian.py CONSTRAINT_DUAL_SCALES + docs/13 §2.3.3.
AOI_REF_S: Final[float] = 0.1

# Reviewer M4 (internal review, W06, 2026-05-27):
# Lagrangian projection upper bound — Π_Λ(λ) = clip(λ, 0, LAMBDA_MAX).
# Prevents dual ascent blow-up under sustained constraint violations
# (Exp11 Robustness sensor failure scenarios). Empirical λ ≤ 2.5 across
# W11 10-seed × 1000-episode runs → LAMBDA_MAX = 10.0 is a soft safety
# net, NOT an active constraint at convergence.
# See docs/13 §2.3.3 + agents/lagrangian.py:191.
LAMBDA_MAX: Final[float] = 10.0

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

# Manager PRB-budget bounds — outer [B_RRM_MIN, B_RRM_MAX] from decode_manager_action;
# set_rrm_budget() further clips to [feasible_rrm_floor, feasible_rrm_cap] at runtime.
B_RRM_MIN: Final[float] = 0.05   # lower bound: avoids URLLC starvation
B_RRM_MAX: Final[float] = 0.85   # upper bound: leaves ≥15% PRBs for eMBB

# Legacy aliases (DO NOT use in new code — kept for backward compat in tests)
T_H_REAL: Final[float] = T_H_REAL_SEC       # was 1.0
T_H_SIM: Final[float] = T_H_SEC             # was 10e-3 — now 100e-3 (semantic fix)
T_L_SIM: Final[float] = T_L_SEC             # Worker/xApp step = 10 ms (FIXED 2026-06-16: was
                                            # aliased to T_TTI_SEC=0.5ms — a 20× semantic bug;
                                            # T_L is 20 MAC ticks, not one. Unused, fixed defensively.)

# RESERVED — learnable xApp decision interval T_int ∈ [10,100] ms. UNUSED and
# CONFLICTS with the current FIXED two-timescale design (T_L=10ms, T_H=100ms,
# W=WORKER_STEPS_PER_MANAGER=10 → GAMMA_MANAGER=γ^10 assumes a fixed W). A
# learnable T_int would break that fixed SMDP ratio. Kept only as provenance.
# (audit 2026-06-17)
T_INT_RANGE: Final[tuple[float, float]] = (10e-3, 100e-3)

# ============================================================
# RESERVED — multi-cell handover + pre-tightening (OUT OF SINGLE-CELL SCOPE,
# NOT WIRED). The env is single-cell UMi (R_CELL_M=300, no handover); the
# constants below are unused by env/agents. PRE_TIGHTEN_ETA additionally
# references "φ_next" (the removed 5-phase FSM — phase→severity swap 2026-06-14),
# so it is doubly stale: severity is fixed per episode, there is no next phase to
# pre-tighten toward. Kept only as design provenance. (audit 2026-06-17)
# ============================================================
HYSTERESIS_RSRP_DB: Final[float] = 3.0      # reserved (anti ping-pong handover)
T_GUARD_SEC: Final[float] = 2.0             # reserved
HANDOVER_ETA_TRIGGER: Final[float] = 10.0   # reserved (proactive handover ETA trigger)
PRE_TIGHTEN_ETA: Final[float] = 30.0        # reserved + phase-FSM vestige (φ_next no longer exists)


def get_severity_thresholds(sev: int) -> dict[str, float]:
    """Master Table helper — per-step severity threshold lookup (docs/13 Phase 1.3).

    Returns 5-key dict matching constraint signals C1-C5 (Phase 2.2.1):
        d1: D_max^sev   (URLLC mean latency, seconds)
        d2: ε^sev        (URLLC tail probability)
        d3: 0            (eMBB signed throughput gap threshold; R_min in CMDP_D_J_SEVERITY)
        d4: AoI_max^sev (AoI mean for aggregated URLLC traffic stream, seconds)
        d5: ε_AoI^sev    (AoI tail probability)

    Used trong:
        - env.step() info dict (per Worker tick lookup)
        - LambdaState.accumulate() (dual subgradient per-step deviation)
        - Worker augmented reward r^aug = r - Σ λ_j · (c_j - d_j^sev)
    """
    if sev not in SEVERITY_QOS:
        raise ValueError(f"Invalid severity {sev}; must be 1..5")
    qos = SEVERITY_QOS[sev]
    return {
        "d1": float(qos["D_max"]),               # URLLC mean latency budget (seconds)
        "d2": float(qos["eps"]),                 # URLLC tail probability budget
        "d3": 0.0,                                # eMBB signed gap threshold; floor is in CMDP_D_J_SEVERITY
        "d4": float(qos["AoI_max"]),              # AoI mean budget (aggregated URLLC stream, seconds)
        "d5": float(qos["eps_aoi"]),             # AoI tail probability budget
    }


def get_severity_alpha(sev: int) -> tuple[float, float]:
    """Severity-weighted reward coefficients α_U(sev), α_e(sev) (Phase 2.1).

    Returns (alpha_urllc, alpha_embb) tuple. Sum to 1.0 per severity level.
    """
    if sev not in SEVERITY_ALPHA:
        raise ValueError(f"Invalid severity {sev}; must be 1..5")
    pa = SEVERITY_ALPHA[sev]
    return float(pa["urllc"]), float(pa["embb"])


# ============================================================
# K-aware (4K+1)-dim Lagrangian vector builders (per-ambulance severity_k epic,
# 2026-06-15). Index convention:
#   [C1_0..C1_{K-1}, C2_0..C2_{K-1}, C4_0..C4_{K-1}, C5_0..C5_{K-1}, C3_shared]
# At K=1 this is the permutation [0,1,3,4,2] of the legacy 5-dim
# [C1,C2,C3,C4,C5] order — exact numeric preservation (see docs/13).
# ============================================================


def build_dual_scales(K: int) -> np.ndarray:
    """Return (4K+1,)-dim CONSTRAINT_DUAL_SCALES for K ambulances.

    At K=1: [D_REF_URLLC, 1.0, AOI_REF_S, 1.0, R_REF_EMBB_MBPS] — the
    permutation [0,1,3,4,2] of the legacy 5-dim CONSTRAINT_DUAL_SCALES.
    """
    if not isinstance(K, (int, np.integer)) or K < 1:
        raise ValueError(f"K must be an int >= 1 (at least one ambulance); got {K!r}")
    return np.concatenate([
        np.full(K, D_REF_URLLC, dtype=np.float64),     # C1_k
        np.full(K, 1.0, dtype=np.float64),             # C2_k
        np.full(K, AOI_REF_S, dtype=np.float64),       # C4_k
        np.full(K, 1.0, dtype=np.float64),             # C5_k
        np.array([R_REF_EMBB_MBPS], dtype=np.float64),  # C3 shared
    ])


def build_lambda_warm_vector(severity_per_amb: Sequence[int], severity_ref: int) -> np.ndarray:
    """Build (4K+1,)-dim λ_warm vector from per-ambulance severities.

    C1_k/C2_k/C4_k/C5_k warm values come from ``LAMBDA_WARM[severity_per_amb[k]]``
    (indices 0,1,3,4 of the legacy 5-list). The shared C3 slot uses
    ``LAMBDA_WARM[severity_ref][2]``. At K=1 with severity_ref==severity_per_amb[0]
    this is the permutation [0,1,3,4,2] of ``LAMBDA_WARM[sev]`` — exact match
    to the legacy reset_episode() warm-start.

    Note: the C3 slot (LAMBDA_WARM[sev][2]) is NON-INCREASING in severity,
    co-directional with the non-increasing d3_embb floor (audit 2026-06-17).
    """
    if len(severity_per_amb) < 1:
        raise ValueError("severity_per_amb must have >= 1 entry (K >= 1)")
    for sev in (*severity_per_amb, severity_ref):
        if sev not in LAMBDA_WARM:
            raise ValueError(f"Severity {sev} not in LAMBDA_WARM table (keys: {sorted(LAMBDA_WARM)})")
    c1 = np.array([LAMBDA_WARM[s][0] for s in severity_per_amb], dtype=np.float64)
    c2 = np.array([LAMBDA_WARM[s][1] for s in severity_per_amb], dtype=np.float64)
    c4 = np.array([LAMBDA_WARM[s][3] for s in severity_per_amb], dtype=np.float64)
    c5 = np.array([LAMBDA_WARM[s][4] for s in severity_per_amb], dtype=np.float64)
    c3 = np.array([LAMBDA_WARM[severity_ref][2]], dtype=np.float64)
    return np.concatenate([c1, c2, c4, c5, c3])


def build_d_phi_vector(severity_per_amb: Sequence[int]) -> np.ndarray:
    """Build (4K+1,)-dim d_phi threshold vector.

    d1_k/d2_k/d4_k/d5_k come from ``get_severity_thresholds(severity_per_amb[k])``
    — each ambulance is held to its OWN severity's QoS budget.

    Note (no ``severity_ref`` parameter, unlike ``build_lambda_warm_vector``):
    the shared C3 slot (index 4K) is **0.0 unconditionally** — the threshold is
    for the SIGNED eMBB gap, and the R_min^sev_ref floor is carried inside the
    env-computed ``c_vec[4K] = R_min^sev_ref − R_eMBB`` (oran_env step()), NOT in
    d_phi. So C3 reduces to ``deviation = c3 − 0 = signed_gap``. ``severity_ref``
    is only relevant for locating that floor in c_vec; it is not needed here.
    ⚠️ Load-bearing coupling: if c_vec[4K] were ever changed to raw R_eMBB (not
    the gap), d3=0 would become WRONG — tests/test_env_c3.py locks the sign.
    """
    if len(severity_per_amb) < 1:
        raise ValueError("severity_per_amb must have >= 1 entry (K >= 1)")
    d1 = np.array([get_severity_thresholds(s)["d1"] for s in severity_per_amb], dtype=np.float64)
    d2 = np.array([get_severity_thresholds(s)["d2"] for s in severity_per_amb], dtype=np.float64)
    d4 = np.array([get_severity_thresholds(s)["d4"] for s in severity_per_amb], dtype=np.float64)
    d5 = np.array([get_severity_thresholds(s)["d5"] for s in severity_per_amb], dtype=np.float64)
    d3 = np.array([0.0], dtype=np.float64)
    return np.concatenate([d1, d2, d4, d5, d3])


def summary() -> str:
    """One-line summary for sanity checking."""
    return (
        f"PPO config: P_total={P_TOTAL} PRB, "
        f"B_PRB={B_PRB/1e3:.0f}kHz, "
        f"f_c={F_CARRIER/1e9:.1f}GHz, "
        f"severity_levels={len(SEVERITY_QOS)}, "
        f"D_max^IMMEDIATE={SEVERITY_QOS[5]['D_max']*1e3:.2f}ms, "
        f"γ_L={GAMMA_WORKER:.3f}, γ_H={GAMMA_MANAGER:.4f}"
    )


if __name__ == "__main__":
    print(summary())
