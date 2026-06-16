# W19 — Pha 3: PPO solver run, K=3 (severity + intra-slice)

> **Pha**: 3 · **Status**: 📅 PLANNED · **Gate**: **GATE 3B** · **Solver**: PPO · **K**: 3 (3 xe cứu thương) · **Build**: B5 (đã code ở W18) · **Deps**: GATE 3A

## Env config (giống W18, KHÔNG đổi)
gNB/cell-center = `(0,0)`, R_cell=300m, UMi 3GPP TR 38.901, single-cell, no handover. Chỉ đổi `K_ambulances: 1 → 3` → obs 31→51 (đã code+assert ở W18/B5).

## C — PPO solver run, K=3
- **C1** Train PPO (Manager+Worker) trên env K=3, ≥10 seeds, MCI single-cell @ Bạch Mai (3 xe đồng trú 1 cell, hội tụ, severity khác nhau → triage contention).
- **C2** Log: episode reward, c_vec/λ **(4K+1)=13 ở K=3** (layout nhóm-theo-constraint `[C1_0..2, C2_0..2, C4_0..2, C5_0..2, C3_shared]`, KHÔNG interleave theo xe), λ-trajectory + saturation-rate (= %step `λ_c==Λ_max=10`); **C6 = structural metric** (priority-inversion rate, KHÔNG λ_C6, KHÔNG Lagrangian).
- **C3** Verify cấu trúc K=3 (chuẩn bị cho so sánh PPO vs TD3 vs SAC ở các tuần sau, KHÔNG so sánh chéo solver trong tuần này):
  - **ordering-compliance**: `sev_i>sev_j ∧ S>0 ⟹ PRB_i>PRB_j` (khi surplus S>0).
  - **no-starvation min-share**: `min_k PRB_k ≥ b ≥ PRB_min^QoS`.
  - **priority-inversion rate** (C6 slack-gated) — log để so sánh chéo solver ở W24.

## ⟲ RÀ SOÁT PPO K=3
`test_multi_ambulance` + `test_intra_slice` pass với K=3 (no-starvation, structural ordering, C6 inversion-rate metric); λ-trajectory + saturation-rate đã log cho cả **(4K+1)=13 λ**; KHÔNG so sánh chéo solver trong tuần này (để dành Table I/II ở W24).

## GATE 3B
PPO K=3 train hội tụ trên obs=51; 2 structural guarantee (ordering-compliance, no-starvation) xác nhận ≥1 kịch bản; λ-trajectory ((4K+1)=13 thành phần) đã log, không có saturation-without-convergence chưa-giải-thích.

## Liên kết
Master plan PHẦN 11/W19 · tiếp TD3 K=1 → [W20](W20_pha3_td3_k1.md).
