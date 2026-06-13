# 05 — Agent Workflow

## Tổng quan 2 tầng (HRL two-timescale)
- **Manager (rApp, ~1s)**: discount γ_H; quyết budget inter-slice + temperature β.
- **Worker (xApp, 10ms)**: discount γ_L; điều chỉnh RRMPolicyRatio + Lagrangian λ.
- `W = WORKER_STEPS_PER_MANAGER = 10` (`config.py:315`); `γ_H ≈ γ_L^W`. ✅[`Akyıldız 2024`] *(Borkar/FeUdal vắng corpus)*. ⚠️ `γ_H/γ_L` = RL discount, KHÁC `β` priority temperature.

## Manager MDP (rApp)
- **Action** (K=1): `{Δr_min^URLLC, Δr_max^eMBB, r_ded, w_C1, w_C2, w_C3}` = 6-dim. K=3: +**β** = 7-dim.
- KHÔNG có Δλ_direction / T_int trong action (dual ascent deterministic; T_int static).

## Worker MDP (xApp)
- **State** S_L qua E2SM-KPM (Q, HOL, PRB-ratio, BLER, λ_local…).
- **Action** a_raw (PPO actor, continuous) → RRMPolicyRatio. δ-preemption / x_k-offload = **KHÔNG dùng** (MEC gỡ).
- **Reward** Lagrangian-augmented: `r_aug = r − Σ_c λ_c·g_c`, với `r = α_e·log(1+R_eMBB/R_REF)` (single-term; URLLC/severity KHÔNG vào r — tránh double-counting với λ).

## CMDP Lagrangian update
`λ_c ← clip(λ_c + α_λ·g_c, 0, Λ_max)`, `α_λ=1e-4` (`ALPHA_LAMBDA_DUAL`), `Λ_max=10` (`LAMBDA_MAX`) — ✅[`Spoor 2025`; `Ding 2023`]. **λ-saturation logging** bắt buộc (sweep W18–W23): %step `λ==Λ_max` + cờ saturation-without-convergence.

## λ_warm — bridge rApp 1s gap
Init λ mỗi episode từ `LAMBDA_WARM[phase]` (phase-aware prior, rule-based); EMA cập nhật qua phase transition. **Đây là cơ chế proactivity** (thay LSTM — đã loại).

## Intra-slice priority (Worker, K≥2)
`w_k = softmax(β·sev_k + δ·ũ_k)`, `PRB_k = b + S·w_k`, `b = max(κB_U/K, PRB_min^QoS)`. β squash `β_min+(β_max−β_min)·sigmoid(a_β)`. Structural guarantee (feasibility + ordering) by construction.

## Safety filter = closed-form feasibility projection
`a_safe = Π_feasible(a_raw)` lên đa diện RRMPolicyRatio: ratios∈[0,1]; `r_ded≤r_min≤r_max`; `Σr_min≤1`, `Σr_ded≤1`; `r_min≥K·PRB_min^QoS/P_TOTAL`. Thuật toán: **projection-onto-simplex [Duchi 2008] + isotonic/PAV + clip + floor** — Euclidean projection chính xác, **no learnable params**.
- ⚠️ **GỠ HOÀN TOÀN**: LSTM Network QoS predictor; Hybrid Safety NSF + QP; β_qp distillation; MEC offload. KHÔNG claim NSF novel (Kim 2026 = neural QP, chỉ cite).

## Cross-reference
[02](02_requirements.md) (CMDP) · [13](13_methodology_walkthrough.md) §3 (thuật toán) · [W17](weeks/W17_pha2_cmdp_formulation.md) (formulation) · [W18](weeks/W18_pha3_algorithm_code.md) (code).
