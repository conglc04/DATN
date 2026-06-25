# 07 — API Specification

## Observation space
**Quy tắc**: field-set = nguồn-sự-thật; số = derived; binding `assert env.observation_space.shape` tại B5 ([W18](weeks/W18_pha3_algorithm_code.md)). KHÔNG hard-code số.

- **K=1 = 32-dim** (per-ambulance `severity_k` epic 2026-06-15; **+`active_mask_k`** 2026-06-23): `obs_dim = 20 + 11·K + F`. K=1,F=1 → 32; K=3,F=1 → 54.
  - **20 fixed block — SSOT = `OBS_*_IDX` trong `utils/config.py`** (mọi consumer import, KHÔNG hardcode số; `tests/test_obs_layout.py` khóa layout):

    | idx | field | ý nghĩa | OBS_*_IDX |
    |----:|-------|---------|-----------|
    | 0 | rho_urllc | URLLC queue util ρ (mean/K, clip[0,1]) | `OBS_RHO_URLLC_IDX` |
    | 1 | rho_emBB | eMBB queue util ρ | `OBS_RHO_EMBB_IDX` |
    | 2 | hol_urllc_ms | URLLC HOL delay (ms, ≤100) | `OBS_HOL_URLLC_IDX` |
    | 3 | hol_emBB_ms | eMBB HOL delay (ms, ≤1000) | `OBS_HOL_EMBB_IDX` |
    | 4 | r_min_urllc | PRB ratio r_min (LIVE, drift/window) | `OBS_R_MIN_URLLC_IDX` |
    | 5 | r_max_emBB | PRB ratio r_max | `OBS_R_MAX_EMBB_IDX` |
    | 6 | r_ded_urllc | PRB ratio r_ded | `OBS_R_DED_URLLC_IDX` |
    | 7 | arr_urllc | URLLC arrival (Σ/K)/1e3 | `OBS_ARR_URLLC_IDX` |
    | 8 | arr_emBB | eMBB arrival /1e4 | `OBS_ARR_EMBB_IDX` |
    | 9 | bler | mean BLER | `OBS_BLER_IDX` |
    | 10:15 | severity_oh[5] | severity_ref one-hot (lvl 1..5) | `OBS_SEVERITY_OH_IDX` (=`SEVERITY_OH_OBS_INDEX`) |
    | 15 | lambda_c3_shared | λ_local shared C3 (eMBB-floor dual) | `OBS_LAMBDA_C3_IDX` (=`LAMBDA_C3_SHARED_OBS_INDEX`) |
    | 16 | r_min_urllc_anchor | Manager setpoint (FIXED/window) | `OBS_RMIN_ANCHOR_IDX` |
    | 17 | n_bys | bystander UE count / M_eMBB | `OBS_N_BYS_IDX` |
    | 18 | aoi_mean | mean AoI over K (s) | `OBS_AOI_MEAN_IDX` |
    | 19 | aoi_max | max AoI over K (s) | `OBS_AOI_MAX_IDX` |

  - **obs[16] = `r_min_urllc_anchor`** (Manager setpoint cho cửa sổ hiện tại, set bởi `env.set_rrm_budget(b_rrm)` mỗi Manager window; **cố định trong window**). obs[4] = `r_min_urllc` = live ratio (= anchor, Worker KHÔNG drift — Worker chỉ điều khiển intra-URLLC priority, KHÔNG thay đổi inter-slice). Trước W18 đây là hằng số tĩnh `config.rrm_budget_hint`; nay là setpoint động của Manager.
  - **per-amb (×K, 11-dim block, `OBS_PER_AMB_BLOCK_LEN=11`, base=`20+11k`)**: `{SINR_k, d_k, v_k, delay_norm_k, AoI_norm_k, severity_norm_k, λ_C1_k, λ_C2_k, λ_C4_k, λ_C5_k, active_mask_k}` — `severity_norm_k = severity_k/5` (per-ambulance, độc lập); λ_C{1,2,4,5}_k = λ_local cho 4 constraint per-xe; **`active_mask_k ∈ {0,1}` = `entered_k & ~arrived_k`** (cờ active tường minh: phân biệt xe inactive (block toàn 0) với xe active queue-rỗng — zero-sentinel đơn thuần mơ hồ). Khối của xe inactive bị zero-out → `active_mask_k=0`.
- **severity_per_amb** ∈ {1..5}^K: sampled độc lập per ambulance, **cố định trong 1 episode**. **severity_ref := max(severity_per_amb)** lái mọi đại lượng SHARED (severity one-hot [10:15], `info["severity"]`; C3 R_min cố định 10 Mbps). **Reward KHÔNG còn α_e** (bỏ 2026-06-23). `info["severity_per_amb"]` = nguồn-sự-thật per-xe.

## Worker Action space (intra-URLLC only)

- **K=1 = 1-dim**: no-op scalar (xe duy nhất nhận toàn bộ URLLC PRBs). Bounds `[-3]→[3]`.
- **K≥2 = K-dim** (pure-RL, gỡ β 2026-06-21): `a[0:K]`→per-vehicle priority logits `ℓ_k` (raw ∈[−3,+3], decode qua `softmax(ℓ)` trong `_prb_split_intra_slice`). KHÔNG còn β slot. K=3 → 3-dim (was 4-dim khi còn β).
- **Worker KHÔNG điều khiển inter-slice**: `r_min/r_max/r_ded` trong obs[4:7] là Manager-owned (read-only cho Worker). `{Δr_min, Δr_max, r_ded, w_C1, w_C2, w_C3}` 6-dim = **legacy, ĐÃ GỠ** — Worker action hiện chỉ gồm per-vehicle logits.
- Anti-starvation floor `PRB_MIN_QOS=1` (per xe ACTIVE, PHẲNG — KHÔNG severity-tiered) = hyperparam (KHÔNG action), structural guarantee duy nhất, giữ **by construction** qua reserve-first split order (`oran_env.py::_prb_split_intra_slice`, audit 2026-06-24 — reserve `K_active·PRB_MIN_QOS` TRƯỚC khi softmax chia phần còn lại; feasibility precondition `B_U≥K_active·PRB_MIN_QOS`). `BETA_MIN/BETA_MAX, INTRA_SLICE_KAPPA, RHO_URGENCY_TIEBREAK` = legacy constant, giữ CHỈ để import/test compat — KHÔNG được đọc bởi allocation. `urgency_k`/N_req KHÔNG tồn tại trong allocation — severity-awareness học hoàn toàn qua λ_C1..C5 trong obs + r_aug gradient.

## Mapping → 3GPP RRMPolicyRatio (✅[TS 28.541])

Manager `b_rrm`→`RRMPolicyMinRatio` (URLLC fraction); `1−b_rrm`→`RRMPolicyMaxRatio` (eMBB remainder). Worker per-vehicle logits → intra-slice pure-RL softmax split (không mapping trực tiếp 3GPP — đây là scheduler-internal).

## Manager → env hook (HRL thật, W18+)
- **`env.set_rrm_budget(b_rrm: float) -> None`**: re-anchor `r_min_urllc` về setpoint Manager tại đầu mỗi Manager window (gọi TRƯỚC vòng Worker). Two-tier clipping: (1) `[B_RRM_MIN, B_RRM_MAX] = [0.05, 0.85]` đảm bảo bởi `decode_manager_action` upstream; (2) `[feasible_rrm_floor, feasible_rrm_cap]` tính ở `reset()` theo K/QoS. Bound chặt hơn thắng. Cập nhật cả `r_min_urllc` lẫn `r_min_urllc_anchor`, rồi `_renormalize_prb_ratios()` (C6: r_min+r_max≤1).
- **`decode_manager_action(a_H_raw) -> {"b_rrm"}`**: `b_rrm = B_RRM_MIN + (B_RRM_MAX−B_RRM_MIN)·sigmoid(a_H_raw[0])` — chung cho cả 3 Manager (PPO/TD3/SAC). raw=0 → b_rrm=0.45 (trung điểm).
- **Manager state** `build_manager_state(obs, λ_global, g_hat)` (audit 2026-06-23, was `(6+4K+1)`): `(8 + 2·(4K+1))`-dim = `[ρ_urllc, ρ_eMBB, BLER, sev_ref_norm, sev_mean_norm, n_active_norm, AoI_mean, AoI_max]` ⊕ (λ_global**/LAMBDA_MAX**) ⊕ g_hat. K=1 → 18-dim, K=3 → 34-dim. **λ-slot normalized** by the LAMBDA_MAX clip ceiling → [0,1] (audit 2026-06-24, fixes Manager-critic input-scale imbalance: λ∈[0,10] vs fixed block∈[0,1]); pure linear rescale, dual ascent/penalty use the RAW `LambdaState.lambda_global` not this obs copy. **g_hat kept RAW** (signed, symmetric around 0 — sign/zero-crossing is load-bearing for the critic; not rescaled by a one-sided ceiling). `g_hat = LambdaState.get_deviation_hat()` (= `g_hat_{t-1}`, the residual from the LAST COMPLETED Manager window — same vector dual ascent consumed; NOT the residual of the window about to run under the action being chosen now, which would leak future information) — exposed alongside the long-run dual price λ_global so the Manager sees the per-window residual `r_aug` is sensitive to. **Same-source proxy, not literal equality** (see `lagrangian.py::get_deviation_hat` docstring): `g_hat` is the SIGNED window MEAN of `(c-d)/scale`; `r_aug` hinges (`max(0,·)`) that SAME per-tick deviation individually before the SMDP-discounted sum the critic learns from — `mean(max(0,dev)) ≥ max(0,mean(dev))` (Jensen), so `g_hat` is directional signal, not the exact subtracted reward amount. `sev_mean_norm`/`n_active_norm` disambiguate K≥2 states that alias under `sev_ref=max(·)` (e.g. (5,1,1) vs (5,5,5)).

## info dict (mỗi step)
`{c_vec, d_phi, severity (=severity_ref), severity_per_amb, r_min_urllc_anchor, prb_urllc, prb_embb, prb_per_amb, ...}` (KHÔNG có `phase_now` — đã gỡ cùng phase FSM) — `c_vec`/`d_phi` shape `(4K+1,)`, layout `[C1_0..C1_{K-1}, C2_0..C2_{K-1}, C4_0..C4_{K-1}, C5_0..C5_{K-1}, C3_shared]`. K=1 → `(5,)`, là permutation `[0,1,3,4,2]` của thứ tự cũ `[C1,C2,C3,C4,C5]` (số trị giống hệt K=1 cũ). **`prb_urllc/prb_embb/prb_per_amb`** = PRB cache trong `step()` (đọc-thuần từ cache, KHÔNG tính lại trong `_info()` vì `_prb_allocation`/`_prb_split_intra_slice` đọc state mutable); invariant `sum(prb_per_amb) == prb_urllc`.

## ⚠️ Đã GỠ khỏi API
`u_MEC` obs field, MEC offload action, LSTM predictor heads, β_qp/NSF params. **2026-06-21**: Π_feasible severity-tier safety filter (N_req 2-pha protection) + intra-slice priority temperature β — Worker action K≥2 đổi `(1+K)`-dim → `K`-dim (xem `agents/worker_agent.py` "pure-RL intra-slice").

## Cross-reference
[05](05_agent_workflow.md) (agent) · [02](02_requirements.md) (constraints) · [W18](weeks/W18_pha3_algorithm_code.md) (assert obs).
