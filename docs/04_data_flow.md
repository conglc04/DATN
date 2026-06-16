# 04 — Data Flow

## Transaction flow (xe → bệnh viện)
1. **UE (xe)** sinh gói telemetry (F=1 luồng tổng hợp `ambulance_status`, 2026-06-14 consolidation — gộp HR_agg/SpO2_agg/ECG_waveform/DENM cũ) — point process, payload+rate (M8.1b). **KHÔNG sinh giá trị sinh hiệu** (chỉ timestamp + size).
2. **Air interface (Uu, UPLINK)**: P_tx^UE=23dBm; SINR từ UMi channel (M2); capacity `C=η·PRB·B_PRB·log₂(1+SINR)`.
3. **Fronthaul (O-RU→O-DU)**: D_FH=0.1ms.
4. **Backhaul → Core/Bệnh viện**: D_BH=0.1ms.
- ⚠️ **KHÔNG MEC processing step** (MEC đã gỡ — D23). Đầu nhận = ED / dispatch centre / bác sĩ từ xa (lý do AoI phục vụ "đầu nhận ở xa" — master plan PHẦN 12).

## End-to-End Delay
`D_e2e = D_DET + d_tx + d_queue + D_FH + D_BH` (KHÔNG D_MEC)
- `D_DET=0.07ms` (xử lý) 🟡[TS 38.214]; `D_FH=D_BH=0.1ms` 🟡[O-RAN WG4; Nie Cheng 2022]; `D_stoch=0.05ms` 🔴 (jitter, ±50%).
- `d_tx, d_queue`: **M/G/1 Pollaczek–Khinchine** ✅[`Kleinrock 9780470316887.pdf` §5.6] — upper-bound trung bình (Poisson giả định); burst non-Poisson (DENM) bắt bằng Monte Carlo, KHÔNG closed-form.
- Reliability: `P(D_e2e^k>D_max^{sev_k})≤ε^{sev_k}` ✅[TS 22.261 §7.2].

## AoI Model (Age of Information)
- **Định nghĩa**: `Δ_k(t) = t − U_k(t)`, U = timestamp gói mới nhất nhận được — ✅[`Qi 2024`; `Chen`; `Mlika 2022`] *(KHÔNG Kaul, vắng corpus)*.
- **Tại sao AoI ≠ latency thuần**: latency thuần có thể retx gói CŨ trong khi gói mới hơn đã sinh → AoI phạt đúng "độ tươi".
- **Discrete-time evolution (LCFS + drop-old)**:
  - Không giao gói trong (t−Δt, t]: `Δ_k(t) = Δ_k(t−Δt) + Δt` (aging tuyến tính).
  - Gói gen_time u giao tại t: `Δ_k(t) = t − u` (reset).
  - `drop_rate = dropped/(dropped+delivered)`.
- **Scope**: AoI optimization cho 1 luồng tổng hợp `ambulance_status` (F=1, 2026-06-14 consolidation; LCFS+drop_old, AoI-aware) có ngưỡng `AoI_max^sev`. Code: `env/aoi_tracker.py`.
- **obs per-amb (∀K)**: `AoI_norm_k = AoI_k/AoI_max^{sev_k}` = **1 dim/xe** (#21, offset `AMB_AOI_NORM_OFFSET` trong khối 10-dim per-amb — [08](08_implementation_notes.md)) — worst-NORMALIZED là sufficient statistic; F=1 ⟹ 1 luồng duy nhất nên "worst" = giá trị đó, công thức tổng quát cho F>1 là `max_s(AoI_s/AoI_max_s^{sev_k})`.

## ⚠️ Đã GỠ
- **MEC offloading model** (decision var x_k, D_MEC, f_MEC, C_FH) — vestigial, gỡ hoàn toàn (D23).
- **Energy efficiency model** — out-of-scope hiện tại.
- **D_max_QP / NSF safety theorem** — NSF+QP gỡ; safety = closed-form feasibility projection ([05](05_agent_workflow.md)).

## Cross-reference
[03](03_architecture.md) (channel) · delay/AoI chi tiết [W13](weeks/W13_pha1_delay_reliability_qos.md) (M4-M6) + [W14](weeks/W14_pha1_aoi_traffic.md) (M7-M8) · constraints [02](02_requirements.md).
