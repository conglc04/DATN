# 04 — Data Flow

## Transaction flow (xe → bệnh viện)
1. **UE (xe)** sinh gói telemetry (F=4 luồng: HR_agg, SpO2_agg, ECG_waveform, DENM) — point process, payload+rate per-stream (M8.1b). **KHÔNG sinh giá trị sinh hiệu** (chỉ timestamp + size).
2. **Air interface (Uu, UPLINK)**: P_tx^UE=23dBm; SINR từ UMi channel (M2); capacity `C=η·PRB·B_PRB·log₂(1+SINR)`.
3. **Fronthaul (O-RU→O-DU)**: D_FH=0.1ms.
4. **Backhaul → Core/Bệnh viện**: D_BH=0.1ms.
- ⚠️ **KHÔNG MEC processing step** (MEC đã gỡ — D23). Đầu nhận = ED / dispatch centre / bác sĩ từ xa (lý do AoI phục vụ "đầu nhận ở xa" — master plan PHẦN 12).

## End-to-End Delay
`D_e2e = D_DET + d_tx + d_queue + D_FH + D_BH` (KHÔNG D_MEC)
- `D_DET=0.07ms` (xử lý) 🟡[TS 38.214]; `D_FH=D_BH=0.1ms` 🟡[O-RAN WG4; Nie Cheng 2022]; `D_stoch=0.05ms` 🔴 (jitter, ±50%).
- `d_tx, d_queue`: **M/G/1 Pollaczek–Khinchine** ✅[`Kleinrock 9780470316887.pdf` §5.6] — upper-bound trung bình (Poisson giả định); burst non-Poisson (DENM) bắt bằng Monte Carlo, KHÔNG closed-form.
- Reliability: `P(D_e2e>D_max^φ)≤ε^φ` ✅[TS 22.261 §7.2].

## AoI Model (Age of Information)
- **Định nghĩa**: `Δ_k(t) = t − U_k(t)`, U = timestamp gói mới nhất nhận được — ✅[`Qi 2024`; `Chen`; `Mlika 2022`] *(KHÔNG Kaul, vắng corpus)*.
- **Tại sao AoI ≠ latency thuần**: latency thuần có thể retx gói CŨ trong khi gói mới hơn đã sinh → AoI phạt đúng "độ tươi".
- **Discrete-time evolution (LCFS + drop-old)**:
  - Không giao gói trong (t−Δt, t]: `Δ_k(t) = Δ_k(t−Δt) + Δt` (aging tuyến tính).
  - Gói gen_time u giao tại t: `Δ_k(t) = t − u` (reset).
  - `drop_rate = dropped/(dropped+delivered)`.
- **Scope**: AoI optimization cho aggregated streams (HR/SpO2/BP) có ngưỡng; ECG-waveform = latency-aware, no-drop (không ngưỡng AoI). Code: `env/aoi_tracker.py`.
- **K=3 obs**: `{AoI_worstnorm_k = max_s(AoI_s/AoI_max_s^φ), AoI_mean_k}` = 2 dims/xe (#21) — worst-NORMALIZED là sufficient statistic.

## ⚠️ Đã GỠ
- **MEC offloading model** (decision var x_k, D_MEC, f_MEC, C_FH) — vestigial, gỡ hoàn toàn (D23).
- **Energy efficiency model** — out-of-scope hiện tại.
- **D_max_QP / NSF safety theorem** — NSF+QP gỡ; safety = closed-form feasibility projection ([05](05_agent_workflow.md)).

## Cross-reference
[03](03_architecture.md) (channel) · delay/AoI chi tiết [W13](weeks/W13_pha1_delay_reliability_qos.md) (M4-M6) + [W14](weeks/W14_pha1_aoi_traffic.md) (M7-M8) · constraints [02](02_requirements.md).
