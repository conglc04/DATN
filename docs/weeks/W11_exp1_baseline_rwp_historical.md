# W11 — Exp1 Baseline (RWP) — LỊCH SỬ, KHÔNG tái dùng

> **Pha**: GĐ A · **Status**: ✅ DONE (lịch sử) · **Deps**: W10/G3.3

## Đã chạy (mobility = Random Waypoint)
- Exp1 K=1 inter-slice baseline: PA-CHRL-PPO vs TD3-Lag vs B3, 1500-ep × 30 run (3 method × 10 seed).
- Số đã thu (tail-100): PA=1059±47, TD3=981±1.4, B3=685±356; Mann-Whitney + Holm-Bonferroni p<0.01.

## ⚠️ TRẠNG THÁI (master plan D8/D9/D20, PHẦN 9)
- **RWP BỎ HOÀN TOÀN** — số RWP (PA=1059, −42% latency vs TD3, …) = **LỊCH SỬ, KHÔNG dùng làm kết quả/tham chiếu/sanity**. obs đã đổi 40→33-dim (gỡ LSTM+MEC) ⟹ không dim-compatible.
- ❌ **Exp1B (LSTM accuracy) — GỠ HOÀN TOÀN** (LSTM loại, master plan D10).
- → Baseline chạy LẠI TỪ ĐẦU trên **SUMO** = **E1** ở [W19](W19_pha3_e1_baseline_sumo.md) (KHÔNG cite số RWP cũ).

## Liên kết
- Thay thế bởi E1 SUMO → [W19](W19_pha3_e1_baseline_sumo.md). Lý do bỏ RWP → master plan D8/D20, `docs/10_risks.md`.
