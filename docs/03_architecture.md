# 03 — Architecture

## O-RAN topology (3-tier)
- **Non-RT RIC (rApp / Manager)**: timescale ~1s; tối ưu budget inter-slice (r_min^URLLC, r_max^eMBB) + intra-slice temperature β.
- **Near-RT RIC (xApp / Worker)**: timescale 10ms; điều chỉnh RRMPolicyRatio + Lagrangian λ.
- **O-DU/O-RU**: MAC TTI 0.5ms; thực thi PRB allocation.
- Mapping action → **RRMPolicy{Min/Max/Dedicated}Ratio** (E2SM-CCC) ✅[3GPP TS 28.541].

## Information delay (E2/A1 timescale)
rApp 1s vs xApp 10ms gap → bridge bằng **λ_warm** (severity-aware prior, `λ_warm[severity_ref]`, [05](05_agent_workflow.md)). KHÔNG predict (LSTM đã loại).

## Topology mô phỏng — single-cell @ Bạch Mai
- **1 gNB cố định** tại ổ-MCI cạnh BV Bạch Mai (đường Giải Phóng), **R_cell = 300m** (UMi), KHÔNG sweep (D25). Phục vụ **uplink xe** (xe→gNB).
- 3 xe đồng trú 1 cell suốt cửa sổ contention (no handover — out-of-scope). Map OSM 5km×5km = topology/route extract (KHÁC cell 300m).
- ⚠️ **KHÔNG** multi-cell N=20; **KHÔNG** Viettel hardware (Qualcomm X100/QRU100 — đã gỡ, không có data thật).

## Severity (exogenous — thay Phase FSM, swap 2026-06-14)
Mỗi xe có **severity ∈ {1..5}** (ATS triage); `severity_per_amb` sampled độc lập per-ambulance, **cố định trong 1 episode**, random giữa các episode — **KHÔNG FSM/transition, KHÔNG explicit signaling, KHÔNG ML inference** (xe luôn có bệnh nhân ⟹ pha STANDBY/DISPATCH/RETURN vô nghĩa). `severity_ref := max(severity_per_amb)` chọn bộ QoS-tier **SHARED** (D_max, ε, R_min, AoI_max) + trọng số reward `α_eMBB`; `severity_per_amb[k]` chọn ngưỡng QoS **per-xe** (C1/C2/C4/C5) + thứ tự ưu tiên intra-slice (β/Π_feasible, K≥2) [02](02_requirements.md).

## Channel model — UMi (3GPP TR 38.901)
- **Path-loss UMi LOS/NLOS** ✅[TR 38.901 §7.4.1 Table 7.4.1-1]; **P_LOS(d)** ✅[§7.4.2]; **shadow σ_SF** ✅[§7.4.1].
- `SINR(d)` từ vị trí thật (SUMO FCD → Haversine → d). Single-cell + PRB trực giao ⟹ I≈0 ⟹ **SNR-based** (declare). Capacity `C=η·PRB·B_PRB·log₂(1+SINR)`, SINR linear.
- **VERIFY** (M2): NLOS chi phối; SINR @300m≈+13dB, @200m≈+19dB ∈[−10,+30]dB ✓.
- ⚠️ **GỠ** UMa hybrid + KNN "Keangnam" (sai site — Cầu Giấy ≠ Bạch Mai, #11). Nếu dùng KNN phải đúng site Bạch Mai, HOẶC UMi generic. 300m vẫn trong validity UMi (KHÔNG cần UMa).

## Hanoi data strategy
**Mô hình hóa được** (OSM hình học + UMi 3GPP + SUMO Tầng 1) ≠ **calibrate số đo thực** (out-of-scope, declare honest). KHÔNG claim "khớp số đo Hà Nội thật".

## Cross-reference
[01](01_overview.md) · channel chi tiết [W12](weeks/W12_pha1_radio_channel_capacity.md) (M1-M3) · mobility [W15](weeks/W15_pha1_sumo_mobility.md) (M10) · delay [04](04_data_flow.md).
