# W03 — Env Modules II: AoI Tracker (Phase FSM removed)

> **Pha**: GĐ A (code foundation) · **Status**: ✅ DONE · **Gate**: G1.2 — unit tests pass · **Deps**: W02/G1.1

## Đã xây
- `env/aoi_tracker.py` — AoI per stream, **LCFS + drop-old**; `AoIPacket{gen_time, deliver_time, payload_id}` → AoI thuần timestamp (point process), **KHÔNG sinh giá trị sinh hiệu**.

## ⚠️ ĐÃ GỠ (phase→severity swap 2026-06-14)
- ❌ `env/phase_detector.py` — **5-phase FSM XÓA HOÀN TOÀN**. Thay bằng **severity (ATS 5-level) exogenous** ([W16](W16_pha1_severity_triage.md)): severity sampled độc lập per-ambulance, cố định/episode. Mọi mô tả "phase FSM φ₁..φ₅" trong các doc cũ = lịch sử. Gate G1.2 nay chỉ còn phần AoI (KHÔNG còn "FSM transitions").

## Đã GỠ (post-cleanup, master plan D10/D23/M7.4)
- ❌ `env/mec_model.py` — **XÓA HOÀN TOÀN** (MEC vestigial: D_MEC không vào D_e2e; gỡ ở B0b/[W18](W18_pha3_algorithm_code.md)).
- ❌ `env/vital_simulator.py` — **XÓA HOÀN TOÀN** (sinh vitals giả + "LSTM training data" — trùng 2 lý do loại; nhất quán no-fake-vitals).

## Gate G1.2 ✅
- AoI stream classification (LCFS+drop-old: latest-status, drop accounting) đúng; `pytest tests/test_env_week3.py` pass (11 tests). *(Clause "FSM transitions đúng" cũ đã bỏ — FSM gỡ.)*

## Liên kết
- AoI định nghĩa/ngưỡng ground ref ở [W14](W14_pha1_aoi_traffic.md) (M7, refs **Qi/Chen/Mlika** — KHÔNG Kaul, vắng corpus). Phase QoS → [W13](W13_pha1_delay_reliability_qos.md) (M6). Severity (ATS 5-level, lớp độc lập với AoI) → [W16](W16_pha1_severity_triage.md).
