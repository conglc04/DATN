# 05 — Agent Workflow

## Tổng quan 2 tầng (HRL two-timescale)
- **Manager (100ms)**: discount γ_H; quyết `b_rrm` budget inter-slice (1-dim action → sigmoid → `[B_RRM_MIN, B_RRM_MAX]`).
- **Worker (xApp, 10ms)**: discount γ_L; điều khiển intra-URLLC priority (β + per-vehicle logits) + Lagrangian λ.
- `W = WORKER_STEPS_PER_MANAGER = 10` (`config.py`); `γ_H ≈ γ_L^W`. ✅[`Akyıldız 2024`] *(Borkar/FeUdal vắng corpus)*. ⚠️ `γ_H/γ_L` = RL discount, KHÁC `β` priority temperature.

## Manager MDP

- **Action**: 1-dim → `sigmoid` → `b_rrm ∈ [B_RRM_MIN, B_RRM_MAX]` = tỷ lệ PRB cho URLLC slice (inter-slice budget). KHÔNG có Δλ_direction / T_int trong action (dual ascent deterministic; T_int static).
- **State**: `s_H` = 6 + (4K+1) dim = `[ρ_u, ρ_e, BLER, sev_norm, AoI_mean, AoI_max, λ_global]`.

## Worker MDP (xApp)

- **State**: `s_L` = 20 + 11K + F dim (obs + overlay λ_local, per-amb block incl. active_mask_k) qua E2SM-KPM.
- **Action**: K=1: 1-dim (no-op scalar, xe duy nhất nhận toàn bộ URLLC PRBs). K≥2: (1+K)-dim — `a[0]`→β (priority temperature), `a[1:1+K]`→per-vehicle logits w_k (intra-URLLC Π_feasible). **Worker KHÔNG điều khiển inter-slice** (r_min/r_max/r_ded là Manager-owned, Worker chỉ đọc qua obs).
- **Reward** Lagrangian-augmented: `r_aug = r − Σ_c λ_c·max(0,g_c)` (hinge; fixed 2026-06-22 bonus-masking audit — slack constraint, g_c<0, đóng góp đúng 0, KHÔNG được bonus reward), với `r = α_e·log(1+R_eMBB/R_REF)` (single-term; URLLC/severity KHÔNG vào r — tránh double-counting với λ). **Lưu ý**: dual ascent (mục dưới) vẫn dùng `g_c` **signed** (không hinge) để λ còn relax được khi constraint slack — hai công thức dùng cùng ký hiệu `g_c` nhưng áp dụng khác nhau.

## CMDP Lagrangian update
`λ_c ← clip(λ_c + α_λ·g_c, 0, Λ_max)`, `α_λ=1e-4` (`ALPHA_LAMBDA_DUAL`), `Λ_max=10` (`LAMBDA_MAX`) — ✅[`Spoor 2025`; `Ding 2023`]. **λ-saturation logging** bắt buộc (sweep W18–W23): %step `λ==Λ_max` + cờ saturation-without-convergence.

## λ_warm — bridge Manager↔Worker cadence gap
Init λ mỗi episode từ `λ_warm[severity_per_amb]` (`build_lambda_warm_vector` — keyed theo tuple severity per-xe; slot C3_shared dùng `severity_ref`): severity-aware prior, rule-based, đơn điệu tăng theo severity. EMA flush `λ_warm[sev] ← (1−β_ema)·λ_warm + β_ema·λ_global` ở **biên episode** (`on_episode_end`). Severity **cố định/episode** ⟹ KHÔNG transition trong-episode (path đổi-severity `on_manager_step_start` là no-op). **Đây là cơ chế proactivity** (thay LSTM — đã loại).

## Intra-slice priority (Worker, K≥2)
**Severity-ordered protection** (2 pha trên `B_U`): Pha 1 cấp `N_req[k]=ceil(C_req[sev_k]/cap_per_prb(SINR_k))` theo tier severity giảm dần (INVARIANT với β/logits); thiếu trong tier → chia theo `score[k]=N_req[k]·(1+β·urgency[k])·softmax(a[1:1+K])[k]`. Pha 2: surplus theo cùng score. `β=BETA_MIN+(BETA_MAX−BETA_MIN)·sigmoid(a[0])`, `BETA_MIN=0.5`. *(Split `κ/δ-softmax` cũ + `INTRA_SLICE_KAPPA`/`RHO_URGENCY_TIEBREAK` = legacy unused.)* Structural guarantee (feasibility + no-starvation + tier-ordering) by construction.

## Safety filter = Π_feasible (intra-slice, K≥2)

Severity-ordered tier-protection + score-based surplus distribution — structural guarantee by construction (no learnable params trong pha protection). Worker logits chỉ ảnh hưởng **surplus** và **tiebreak cùng tier**.

- ⚠️ **GỠ HOÀN TOÀN**: LSTM Network QoS predictor; Hybrid Safety NSF + QP; β_qp distillation; MEC offload; RRMPolicyRatio Δr_min/Δr_max/r_ded ở Worker action (legacy — giờ Worker chỉ output β + per-vehicle logits).

## Cross-reference
[02](02_requirements.md) (CMDP) · [13](13_methodology_walkthrough.md) §3 (thuật toán) · [W17](weeks/W17_pha2_cmdp_formulation.md) (formulation) · [W18](weeks/W18_pha3_algorithm_code.md) (code).
