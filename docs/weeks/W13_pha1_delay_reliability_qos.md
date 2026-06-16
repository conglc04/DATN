# W13 — Pha 1: Delay E2E + Reliability + QoS Thresholds

> **Pha**: 1 · **Status**: 📅 PLANNED · **Gate**: micro-GATE 1B · **Nhóm**: M4–M6 · **Deps**: W12/1A

## M4 — Phân rã độ trễ `D_e2e = D_DET + d_tx + d_queue + D_FH + D_BH`
- **M4.1** `D_DET = 0.07 ms` (xử lý N1+N2) — 🟡→[3GPP TS 38.214 §5.1] — [ms]
- **M4.2** `d_tx, d_queue` — hàng đợi **M/G/1 Pollaczek–Khinchine** — ✅[`9780470316887.pdf` (Kleinrock) §5.6] — [ms]
- **M4.3** `D_FH = 0.1 ms` (fronthaul) — 🟡→[`O-RAN.pdf` WG4] (eCPRI = chuẩn ngoài corpus → ghi kèm, KHÔNG làm ✅ chính)
- **M4.4** `D_BH = 0.1 ms` (backhaul) — 🟡→[`Nie Cheng…[2022].pdf` §III] / [3GPP TR 38.801]
- **M4.5** `D_stoch = 0.05 ms` (jitter) — 🔴 NEEDS-CALIBRATION (declared PB-C2; sensitivity ±50%)
- ⚠️ **MEC**: KHÔNG có `D_MEC` trong tổng (MEC đã GỠ — [W03](W03_env_phase_aoi.md)/[W18](W18_pha3_algorithm_code.md)).

## M5 — Độ tin cậy
- **M5.1** `Reliability: P(D_e2e > D_max) ≤ ε` — ✅[3GPP TS 22.261 §7.2] — reliability = 1−P(quá hạn), KHÔNG nhầm BLER thuần.

## M6 — Ngưỡng QoS theo severity
- **M6.1** `D_max^{sev5} = 1 ms`, `ε^{sev5} = 1e-5` (IMMEDIATE) — ✅[3GPP TS 22.261 Annex D §D.1 + Annex A]
- **M6.2** `D_max^{sev4} = 2 ms` (EMERGENCY) + các mức khác (sev1 NON_URGENT 20ms → sev5 IMMEDIATE 1ms, bảng [02](02_requirements.md)) — 🟡→[3GPP TS 22.261 Annex A Table A.1-1] (đối chiếu đúng dòng use-case)
- ⚠️ **Severity = bộ chọn QoS-tier** (swap 2026-06-14): `severity_per_amb[k]` chọn `D_max^{sev_k}/ε^{sev_k}/AoI_max^{sev_k}` per-xe (C1/C2/C4/C5); trọng số reward dùng `α_eMBB(sev_ref)` riêng. (KHÔNG còn "severity chỉ là weight, không đổi target" — đó là thiết kế pre-swap.)

## PRB_min^QoS — thủ tục nghịch đảo (sàn cứng Option B, audit #6)
Giải `PRB_k` nhỏ nhất sao cho `D_e2e(PRB_k) ≤ D_max^{sev_k}`, với `C_k(PRB_k)=η·PRB_k·B_PRB·log₂(1+SINR_cell-edge)` (M3.1 @SINR cell-edge ≈ NLOS@300m +13dB) và `d_queue` từ M/G/1 PK (M4.2) tại tải M8. `D_e2e` đơn điệu giảm theo PRB_k ⟹ nghiệm duy nhất. 🟡 (form đã rõ; numeric solve khi M3/M6/M8 khóa). **KHÔNG** claim "guarantee 3GPP ∀ channel" — chỉ sàn PRB-allocation @SINR_ref (audit #6).

## ⟲ RÀ SOÁT M4–M6
Cùng đơn vị [ms]; M/G/1 ổn định ρ<1 (kiểm tải biên); D_e2e không sót thành phần so với `oran_env._compute_e2e_delay`; mỗi mức severity (D_max, ε) gắn đúng dòng 3GPP; D_max^{sev4} > D_max^{sev5} (severity lỏng hơn ⟹ ngưỡng lớn hơn).

## micro-GATE 1B
Mọi thành phần D_e2e + ε + D_max đều ✅ hoặc 🔴-declared (0 mục 🟡 sót).

## Liên kết
Master plan PHẦN 11/W13 · `docs/04_data_flow.md` (delay) · PRB_min^QoS → master plan §1.3 + constraint C1/C2 [W17](W17_pha2_cmdp_formulation.md).
