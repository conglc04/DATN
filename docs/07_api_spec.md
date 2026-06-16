# 07 — API Specification

## Observation space
**Quy tắc**: field-set = nguồn-sự-thật; số = derived; binding `assert env.observation_space.shape` tại B5 ([W18](weeks/W18_pha3_algorithm_code.md)). KHÔNG hard-code số.

- **K=1 = 31-dim** (per-ambulance `severity_k` epic 2026-06-15: thay khối 24-fixed+5K bằng **20 fixed + 10·K + F(=1)**): `obs_dim = 20 + 10K + F`. K=1,F=1 → 31; K=3,F=1 → 51.
  - **20 fixed** = ρ_urllc/ρ_eMBB(2) + HOL_urllc/HOL_eMBB(2) + PRB-ratio(3) + arr_urllc/arr_eMBB(2) + BLER(1) + **severity_ref one-hot[5]** (idx 10:15, `SEVERITY_OH_OBS_INDEX=10`) + **λ_local_C3_shared[1]** (idx 15, `LAMBDA_C3_SHARED_OBS_INDEX=15`) + **r_min_urllc_anchor[1]** (idx 16, Manager setpoint anchor) + n_bys(1) + AoI mean/max(2).
  - **obs[16] = `r_min_urllc_anchor`** (Manager setpoint cho cửa sổ hiện tại, set bởi `env.set_rrm_budget(b_rrm)` mỗi Manager window; **cố định trong window**), KHÁC với obs[4] = `r_min_urllc` *live* (Worker drift mỗi step qua Δr_min). Hiệu obs[4]−obs[16] = cumulative Worker drift (tín hiệu phối hợp Manager↔Worker). Trước W18 đây là hằng số tĩnh `config.rrm_budget_hint`; nay là setpoint động của Manager.
  - **per-amb (×K, 10-dim block, `OBS_PER_AMB_BLOCK_LEN=10`, base=`20+10k`)**: `{SINR_k, d_k, v_k, delay_norm_k, AoI_norm_k, severity_norm_k, λ_C1_k, λ_C2_k, λ_C4_k, λ_C5_k}` — `severity_norm_k = severity_k/5` (per-ambulance, độc lập); λ_C{1,2,4,5}_k = λ_local cho 4 constraint per-xe.
- **severity_per_amb** ∈ {1..5}^K: sampled độc lập per ambulance, **cố định trong 1 episode**. **severity_ref := max(severity_per_amb)** lái mọi đại lượng SHARED (severity one-hot [10:15], `α_eMBB(sev)`, C3 R_min^sev, `info["severity"]`). `info["severity_per_amb"]` = nguồn-sự-thật per-xe.

## Action space
- **K=1 = 6-dim**: `{Δr_min^URLLC, Δr_max^eMBB, r_ded, w_C1, w_C2, w_C3}`, bounds `[-1,-1,0,0,0,0]→[1,1,1,1,1,1]`.
- **K≥2 = 7-dim**: + **β** = priority temperature; `β = BETA_MIN + (BETA_MAX−BETA_MIN)·sigmoid(a[6])`, `BETA_MIN=0.0, BETA_MAX=5.0`. K=1: `β := BETA_MIN` cố định (a[6] không tồn tại, không ảnh hưởng số — Π_feasible K=1 luôn `PRB_0=B_U`).
- ρ/δ/β_min/β_max = hyperparam (KHÔNG action; `INTRA_SLICE_KAPPA=0.5`, `PRB_MIN_QOS=1`, `RHO_URGENCY_TIEBREAK=0.15`). ũ_k DERIVED từ λ trong obs (KHÔNG thêm field).

## Mapping → 3GPP RRMPolicyRatio (✅[TS 28.541])
`r_min^URLLC`→`RRMPolicyMinRatio`; `r_max^eMBB`→`RRMPolicyMaxRatio`; `r_ded`→`RRMPolicyDedicatedRatio` (⊆ Min). Δ = thay đổi tương đối → tích lũy thành ratio; ratio chịu projection (`r_ded≤r_min≤r_max`, `Σ≤1`). w_C1/C2/C3 = trọng số constraint (Lagrangian warm), KHÔNG phải RRMPolicyRatio.

## Manager → env hook (HRL thật, W18+)
- **`env.set_rrm_budget(b_rrm: float) -> None`**: re-anchor `r_min_urllc` về setpoint Manager tại đầu mỗi Manager window (gọi TRƯỚC vòng Worker). Two-tier clipping: (1) `[B_RRM_MIN, B_RRM_MAX] = [0.05, 0.85]` đảm bảo bởi `decode_manager_action` upstream; (2) `[feasible_rrm_floor, feasible_rrm_cap]` tính ở `reset()` theo K/QoS. Bound chặt hơn thắng. Cập nhật cả `r_min_urllc` lẫn `r_min_urllc_anchor`, rồi `_renormalize_prb_ratios()` (C6: r_min+r_max≤1).
- **`decode_manager_action(a_H_raw) -> {"b_rrm"}`**: `b_rrm = B_RRM_MIN + (B_RRM_MAX−B_RRM_MIN)·sigmoid(a_H_raw[0])` — chung cho cả 3 Manager (PPO/TD3/SAC). raw=0 → b_rrm=0.45 (trung điểm).
- **Manager state** `build_manager_state(obs, λ_global)`: `(6 + 4K+1)`-dim = `[ρ_urllc, ρ_eMBB, BLER, sev_norm, AoI_mean, AoI_max]` ⊕ λ_global. K=1 → 11-dim.

## info dict (mỗi step)
`{c_vec, d_phi, severity (=severity_ref), severity_per_amb, r_min_urllc_anchor, prb_urllc, prb_embb, prb_per_amb, ...}` (KHÔNG có `phase_now` — đã gỡ cùng phase FSM) — `c_vec`/`d_phi` shape `(4K+1,)`, layout `[C1_0..C1_{K-1}, C2_0..C2_{K-1}, C4_0..C4_{K-1}, C5_0..C5_{K-1}, C3_shared]`. K=1 → `(5,)`, là permutation `[0,1,3,4,2]` của thứ tự cũ `[C1,C2,C3,C4,C5]` (số trị giống hệt K=1 cũ). **`prb_urllc/prb_embb/prb_per_amb`** = PRB cache trong `step()` (đọc-thuần từ cache, KHÔNG tính lại trong `_info()` vì `_prb_allocation`/`_prb_split_intra_slice` đọc state mutable); invariant `sum(prb_per_amb) == prb_urllc`.

## ⚠️ Đã GỠ khỏi API
`u_MEC` obs field, MEC offload action, LSTM predictor heads, β_qp/NSF params.

## Cross-reference
[05](05_agent_workflow.md) (agent) · [02](02_requirements.md) (constraints) · [W18](weeks/W18_pha3_algorithm_code.md) (assert obs).
