# W09 — 100-ep Smoke + Phase 3 Verify Gate

> **Pha**: GĐ A (code foundation) · **Status**: ✅ DONE · **Gate**: **G3 — Pha 3 verified empirically** · **Deps**: W08/G3.2

## Đã xây / verify
- 100-ep smoke cho PPO + TD3 + SAC (B3-RCPO cũ đã loại).
- Verify constraint dynamics: **URLLC corner** (viol_C1+C2 thấp, viol_C3 cao) vs **eMBB corner** (ngược lại); λ-trajectory khớp active-set theo phase (λ_1+λ_2 > λ_3 @ φ₃).

## Đã GỠ (post-cleanup, master plan D13)
- ❌ **β_qp sweep** (113-run β_qp diagnostic) — gỡ hoàn toàn cùng NSF-distillation; safety filter → closed-form projection (no learnable β_qp). KHÔNG còn trong scope.

## Gate G3 ✅
- λ healthy (no blow-up tới Λ_max=10 vô hạn); corner behaviours reproduced; constraint binding đúng active-set.

## Liên kết
- λ-saturation logging (rủi ro diagnostic) chính thức hoá ở [W19](W19_pha3_e1_baseline_sumo.md)/A-E1.2d. Tune → [W10](W10_tune_multiseed_prep.md).
