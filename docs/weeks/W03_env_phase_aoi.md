# W03 — Env Modules II: Phase FSM + AoI Tracker

> **Pha**: GĐ A (code foundation) · **Status**: ✅ DONE · **Gate**: G1.2 — unit tests pass · **Deps**: W02/G1.1

## Đã xây
- `env/phase_detector.py` — **5-phase FSM** (φ₁ STANDBY → φ₂ DISPATCH → φ₃ SCENE → φ₄ TRANSPORT → φ₅ RETURN); phase từ **explicit signaling**, KHÔNG ML.
- `env/aoi_tracker.py` — AoI per stream, **LCFS + drop-old**; `AoIPacket{gen_time, deliver_time, payload_id}` → AoI thuần timestamp (point process), **KHÔNG sinh giá trị sinh hiệu**.

## Đã GỠ (post-cleanup, master plan D10/D23/M7.4)
- ❌ `env/mec_model.py` — **XÓA HOÀN TOÀN** (MEC vestigial: D_MEC không vào D_e2e; gỡ ở B0b/[W18](W18_pha3_algorithm_code.md)).
- ❌ `env/vital_simulator.py` — **XÓA HOÀN TOÀN** (sinh vitals giả + "LSTM training data" — trùng 2 lý do loại; nhất quán no-fake-vitals).

## Gate G1.2 ✅
- FSM transitions đúng; AoI stream classification (LCFS) đúng; `pytest tests/test_env_week3.py` pass.

## Liên kết
- AoI định nghĩa/ngưỡng ground ref ở [W14](W14_pha1_aoi_traffic.md) (M7, refs **Qi/Chen/Mlika** — KHÔNG Kaul, vắng corpus). Phase QoS → [W13](W13_pha1_delay_reliability_qos.md) (M6). Severity (ATS 5-level, lớp độc lập với AoI) → [W16](W16_pha1_severity_triage.md).
