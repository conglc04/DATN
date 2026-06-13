# W24 — Formulation completeness audit + Luận án + Defense

> **Pha**: 3 · **Status**: 📅 PLANNED · **Build**: B8 · **Deps**: GATE 3F
> **REVISED**: thay cho W21 (formulation audit) + W22-23 (thesis/defense) cũ — gộp vào tuần cuối sau khi sweep 6-tuần (PPO/TD3/SAC × K∈{1,3}) hoàn tất ở W18-W23.

## A-E6 — Formulation completeness audit + buffer
- **A-E6.1** Rà soát toàn-tuyến Pha 1-3 (M1-M11/P1-P5/A1-A4 + A-TD3*/A-SAC*): 0 mục 🟡 còn sót (mọi 🟡 đã ✅ hoặc chuyển 🔴-declared có sensitivity); rà soát notation β/γ/α (§1.3) đã sạch ở mọi nơi liệt kê.
- **A-E6.2** Buffer cho Table I/II (W23): nếu CI rộng/overlap → chạy thêm seeds (20-30) cho solver×K liên quan; nếu λ-saturation cao mà violation chưa→0 → sweep `LAMBDA_MAX`, re-run.
- **A-E6.3** Polish formulation cho luận án: hoàn thiện đoạn văn đặc tả CMDP (P1-P5) + 2 structural guarantee (P5.2) + C6 soft-nudge framing (§1.4) + SAC entropy-temperature framing (mới, B7) — đảm bảo phát biểu bài toán tự-đầy-đủ.

## A6 — Viết + bảo vệ
- **A6.1** Viết theo **3 pha** (Mô hình hệ thống → Bài toán tối ưu → 3 solver giải: PA-CHRL-PPO, TD3-Lag, SAC-Lag) + Related Work (master plan PHẦN 10) + `REFERENCE_MAP` đầy đủ (mọi M*/P*/A* → file `documents/` hoặc chuẩn).
- **A6.2** Trình bày sweep K∈{1,3} × 3 solver (Table I + Table II từ W23) như nội dung thực nghiệm chính.
- **A6.3** Defense slides (master plan PHẦN 12); đánh dấu rõ future work (E3 AoI re-run SUMO, E4 stress/robustness, feasibility-projection polish / ns-3 / cardiac-event severity). [LSTM/MEC đã loại — KHÔNG nhắc.]

## Khung đóng góp (master plan PHẦN 8)
- **C1 — Context-aware constraints (phase + severity)**: **★ HEADLINE — Severity NOVEL** (triage → intra-slice priority + ordering).
- **C2 — CMDP-Lagrangian → 3 solver (PA-CHRL-PPO HRL, TD3-Lag, SAC-Lag)** (✅ built, sweep W18-W23).
- **C3 — AoI hard constraints** (✅ built).
- *Feasibility projection* (honest, KHÔNG claim novel như Kim 2026).

## Future work
- **E3** AoI re-run SUMO (LCFS vs FCFS) — C3 vẫn built+tested và active trong sweep W18-W23.
- **E4** stress/robustness (load/burst/noise/sensor-fail).

## ⟲ RÀ SOÁT cuối
Quét toàn luận án: 0 đại lượng/ngưỡng/số liệu thiếu nhãn ✅/🟡/🔴; 0 citation gán cho file chưa mở; KHÔNG over-claim (no zero-duality-gap, no regret bound, no fake vitals, no Viettel-data, severity→latency = design principle declared); 0 claim nào ngụ ý E3/E4 đã chạy.

## GATE 3-FINAL
Formulation Pha 1-3 hoàn chỉnh (0 mục 🟡 sót); Table I/II (W23) có CI hẹp đủ kết luận (không overlap, hoặc overlap được giải thích trung thực); E3/E4 declared "future work" trong luận án — **KHÔNG yêu cầu E3/E4 chạy để pass gate**.

## Liên kết
Master plan PHẦN 11/W24 + PHẦN 10 + PHẦN 12 · `docs/01_overview.md` (đóng góp) · `docs/REFERENCE_MAP.md` · `docs/06_validation.md`.
