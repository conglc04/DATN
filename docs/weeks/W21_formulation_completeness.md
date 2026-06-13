# W21 — Pha 3: TD3 solver run, K=3 (severity + intra-slice)

> **Pha**: 3 · **Status**: 📅 PLANNED · **Gate**: **GATE 3D** · **Solver**: TD3 (off-policy) · **K**: 3 (3 xe cứu thương) · **Build**: B6 (đã có sẵn) + B5 (đã code ở W18) · **Deps**: GATE 3C

## Env config (giống W19, KHÔNG đổi)
gNB/cell-center = `(0,0)`, R_cell=300m, UMi 3GPP TR 38.901, single-cell, no handover. `K_ambulances=3` → obs=58 (B5, code đã xong ở W18).

## A-TD3 — TD3 solver, K=3 (severity + intra-slice + C6)
- **A-TD3.4** Train `TD3Baseline` trên env K=3, ≥10 seeds, MCI single-cell @ Bạch Mai (3 xe đồng trú 1 cell, hội tụ, severity khác nhau → triage contention).
- **A-TD3.5** Log: episode reward, 5+1 constraint costs C1–C5 (+ C6 per-pair, K=3 có C(3,2)=3 cặp), λ-trajectory + saturation-rate cho mọi λ kể cả `λ_C6^{ij}`, critic/actor loss.
- **A-TD3.6** Verify cấu trúc K=3 cho TD3 (chuẩn bị so sánh chéo solver ở W24):
  - **ordering-compliance**: `sev_i>sev_j ∧ S>0 ⟹ PRB_i>PRB_j` (khi surplus S>0).
  - **no-starvation min-share**: `min_k PRB_k ≥ b ≥ PRB_min^QoS`.
  - **priority-inversion rate** (C6 slack-gated).

## ⟲ RÀ SOÁT TD3 K=3
`test_multi_ambulance` + `test_intra_slice` pass với K=3 cho TD3; λ-trajectory + saturation-rate đã log cho cả 6 λ (C1-C5+C6); KHÔNG so sánh chéo solver trong tuần này (để dành Table I/II ở W24).

## GATE 3D
TD3 K=3 train hội tụ trên obs=58; 2 structural guarantee (ordering-compliance, no-starvation) xác nhận ≥1 kịch bản cho TD3; λ-trajectory (6 thành phần) đã log, không có saturation-without-convergence chưa-giải-thích.

## Liên kết
Master plan PHẦN 11/W21 · tiếp SAC K=1 → [W22](W22_sac_k1.md).
