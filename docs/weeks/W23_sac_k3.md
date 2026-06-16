# W23 — Pha 3: SAC solver run, K=3 (severity + intra-slice) — Table I/II compilation

> **Pha**: 3 · **Status**: 📅 PLANNED · **Gate**: **GATE 3F** · **Solver**: SAC (off-policy, max-entropy) · **K**: 3 (3 xe cứu thương) · **Build**: B7 (đã code ở W22) + B5 (đã code ở W18) · **Deps**: GATE 3E

## Env config (giống W19/W21, KHÔNG đổi)
gNB/cell-center = `(0,0)`, R_cell=300m, UMi 3GPP TR 38.901, single-cell, no handover. `K_ambulances=3` → obs=51 (B5, code đã xong ở W18).

## A-SAC — SAC solver run, K=3 (severity + intra-slice + C6)
- **A-SAC.3** Train `SACBaseline` trên env K=3, ≥10 seeds, MCI single-cell @ Bạch Mai (3 xe đồng trú 1 cell, hội tụ, severity khác nhau → triage contention).
- **A-SAC.4** Log: episode reward, c_vec/λ **(4K+1)=13 ở K=3** (layout nhóm-theo-constraint `[C1_0..2, C2_0..2, C4_0..2, C5_0..2, C3_shared]`, KHÔNG interleave theo xe), λ-trajectory + saturation-rate; **C6 = structural metric** (priority-inversion rate, KHÔNG λ_C6, KHÔNG Lagrangian), critic/actor/alpha loss, α-trajectory.
- **A-SAC.5** Verify cấu trúc K=3 cho SAC:
  - **ordering-compliance**: `sev_i>sev_j ∧ S>0 ⟹ PRB_i>PRB_j` (khi surplus S>0).
  - **no-starvation min-share**: `min_k PRB_k ≥ b ≥ PRB_min^QoS`.
  - **priority-inversion rate** (C6 slack-gated).

## A-T — Table I/II compilation (3 solver × K∈{1,3} — toàn bộ sweep W18–W23)
- **A-T.1 Table I** (so sánh solver, ≥10 seeds): PPO vs TD3 vs SAC, mỗi solver × K∈{1,3} (6 cell) — reward, 5 constraint-violation rates (C1–C5), λ-saturation-rate. Chuẩn đánh giá: **Holm–Bonferroni** (p<0.01) + **Hedges' g** + **bootstrap 95% CI** *(Holm 1979/Hedges 1985 vắng corpus → ghi công thức)*. 10 seeds = TỐI THIỂU; CI rộng/overlap → tăng 20-30 seeds; **KHÔNG claim "vượt trội" nếu CI chồng lấn**.
- **A-T.2 Table II** (K=3 only — severity/intra-slice headline): ordering-compliance, no-starvation min-share, priority-inversion rate, fairness trong-tier (Jain trên nhóm cùng-severity hoặc weighted-Jain `PRB_k/w_k` 🟡[Jain 1984 vắng corpus → ghi công thức]) — cho cả 3 solver.
- **A-T.3** **ε=1e-5 KHÔNG validate bằng 10 seeds** (rare-event, #7): cần ~10⁶–10⁷ mẫu → báo cáo observed violation-rate + CI + rule-of-three `ε≤3/N`; KHÔNG vẽ "đạt ε=1e-5" nếu mẫu không đủ — declare "C2 enforced qua Lagrangian; ε validate đến mức N mẫu cho phép". IS/EVT = future work.

## ⟲ RÀ SOÁT SAC K=3 + Table I/II
`test_multi_ambulance` + `test_intra_slice` pass với K=3 cho SAC; λ-trajectory + saturation-rate đã log cho cả **(4K+1)=13 λ**; mỗi p-value kèm test-name + n_seeds + hiệu chỉnh + Hedges' g + bootstrap CI; Table II KHÔNG Jain toàn cục.

## GATE 3F
SAC K=3 train hội tụ trên obs=51; 2 structural guarantee xác nhận cho SAC; Table I (6 cell × 3 solver) + Table II (severity/intra-slice, K=3) hoàn tất với CI + p-value hiệu chỉnh; KHÔNG báo cáo "thắng" nếu CI chồng lấn.

## Liên kết
Master plan PHẦN 11/W23 · formulation audit + luận án → [W24](W24_thesis_writing_defense.md).
