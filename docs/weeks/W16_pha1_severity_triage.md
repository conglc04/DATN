# W16 — Pha 1: Severity Model (ATS 5-level triage, exogenous, cố định/episode)

> **Pha**: 1 · **Status**: 📅 PLANNED · **Gate**: **GATE 1E (kết thúc Pha 1)** · **Nhóm**: M11 · **Build**: B2 · **Deps**: W12-W15

## M11 — Mô hình độ nặng bệnh nhân (ngoại sinh, KHÔNG vitals giả)

Severity = **thang phân loại triage 5 mức** theo **ATS — Australasian Triage Scale** — ✅[`ATS — Australasian College for Emergency Medicine (ACEM)`]. Đây là **trục ưu tiên DUY NHẤT** (thay 5-pha cũ, phase→severity swap 2026-06-14): xe luôn có bệnh nhân ⟹ severity quyết định độ chặt QoS + thứ tự ưu tiên giữa các xe.

- **M11.1** 5 mức (internal `1..5` = lỏng→chặt). Số hiệu ATS gốc đánh ngược (ATS 1 = nguy kịch nhất) → ánh xạ:

  | internal sev | nhãn | ATS | ngữ nghĩa ATS |
  |---|---|---|---|
  | 1 | NON_URGENT | ATS 5 | non-urgent (đánh giá ≤120 phút) |
  | 2 | SEMI_URGENT | ATS 4 | semi-urgent (≤60 phút) |
  | 3 | URGENT | ATS 3 | urgent — potentially life-threatening (≤30 phút) |
  | 4 | EMERGENCY | ATS 2 | emergency — imminently life-threatening (≤10 phút) |
  | 5 | IMMEDIATE | ATS 1 | immediate / resuscitation — life-threatening |

- **M11.2** **Exogenous + cố định trong 1 episode** (episode = 1s — quá ngắn để severity đổi; gán 1 lần khi ekip tiếp cận bệnh nhân). **KHÔNG mô phỏng giá trị sinh hiệu** (severity là category rời rạc). KHÔNG CTMC/birth-death/phase-event — bỏ hoàn toàn (không còn pha trong-episode để drive transition).
- **M11.3** **Sampling**: TRAIN random/episode từ `severity_sample_weights` (uniform `(.20,.20,.20,.20,.20)` — không giả định phân phối, mọi mức xác suất bằng nhau); EVAL scripted cố định (seed) — (a) `[1,3,5]` ordering 3 xe; (b) `[5,5,5]` MCI toàn nguy kịch; (c) `[5,1]` money-shot Immediate vs Non-urgent.
- **M11.4** Severity → **QoS tier** (config.py): `SEVERITY_QOS[sev]` (D_max, ε, AoI_max, R_min) + `α_eMBB(sev)` (đơn điệu ngược: 0.70→0.05) + `λ_warm[sev]` (warm-start, đơn điệu tăng).
- **M11.5** Severity → **network priority** (K≥2): intra-slice PRB weight `softmax(β·sev+δ·ũ)` + C6 ordering — 🔴 **design principle** (analogy từ ED triage prioritization [ATS]; KHÔNG ref mạng trực tiếp). Khi 2 xe đồng severity: tie-break theo constraint-slack → channel (SINR) → AoI-ratio (đều đã có trong obs, KHÔNG thêm hyperparam).

## ⟲ RÀ SOÁT M11

Severity ngoại sinh (cố định/episode ⟹ không transition đọc QoS/SINR); 5 mức rời rạc one-hot obs[10:15]; severity vào ALLOCATION/constraint (`SEVERITY_QOS`) + reward weight `α_e(sev)`; KHÔNG mô phỏng vitals; phân biệt scene/transport (mobility) đã nằm trong `speed` obs (severity KHÔNG kiêm việc đó).

## GATE 1E (kết thúc Pha 1)

Mọi 🟡 của M1–M11 đã thành ✅; mọi 🔴 đã declared + có sensitivity/calibration plan; rà soát chéo: 0 đại lượng nào trong code đi vào kết quả mà thiếu nhãn. **→ Pha 1 hoàn tất.**

## Liên kết

Master plan PHẦN 11/W16 + §1.1 · `severity.py` ([docs/08](../08_implementation_notes.md)) · severity → intra-slice PRB weight `softmax(β·sev+δ·ũ)` formal hoá ở [W17](W17_pha2_cmdp_formulation.md)/P5 (β = priority temp, KHÔNG RL discount).
