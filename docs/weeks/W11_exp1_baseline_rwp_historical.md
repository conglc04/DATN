# W11 — Exp1 Baseline (RWP) — LỊCH SỬ, KHÔNG tái dùng

> **Pha**: GĐ A · **Status**: ✅ DONE (lịch sử) · **Deps**: W10/G3.3

## Đã chạy (mobility = Random Waypoint)
- Exp1 K=1 inter-slice baseline (mobility RWP), 1500-ep × 30 run. **Toàn bộ số RWP = LỊCH SỬ, KHÔNG tái dùng** (B3-RCPO + RWP đã loại; obs đã đổi 40→33-dim ⟹ không dim-compatible).

## ⚠️ TRẠNG THÁI (master plan D8/D9/D20, PHẦN 9)
- **RWP BỎ HOÀN TOÀN** — mọi số RWP = **LỊCH SỬ, KHÔNG dùng làm kết quả/tham chiếu/sanity**. obs đã đổi 40→33-dim (gỡ LSTM+MEC) ⟹ không dim-compatible.
- ❌ **Exp1B (LSTM accuracy) — GỠ HOÀN TOÀN** (LSTM loại, master plan D10). ❌ **B3-RCPO — loại HOÀN TOÀN** (sweep dùng PPO / TD3 / SAC).
- → Solver chạy LẠI TỪ ĐẦU trên **SUMO** = sweep W18–W23 (KHÔNG cite số RWP cũ).

## Liên kết
- Thay thế bởi PPO K=3 (SUMO) → [W19](W19_pha3_ppo_k3.md). Lý do bỏ RWP → master plan D8/D20, `docs/10_risks.md`.
