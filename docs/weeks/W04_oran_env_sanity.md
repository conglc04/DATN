# W04 — ORANEnv Complete + Sanity

> **Pha**: GĐ A (code foundation) · **Status**: ✅ DONE · **Gate**: G1 — D_e2e < 1ms @ φ₃ · **Deps**: W03/G1.2

## Đã xây
- `env/oran_env.py` — Gym `ORANEnv` gộp tất cả env modules; `step()` advance 20 MAC tick / Worker step (10ms = T_L); `reset()` init phase/queue/ambulance.
- `env/bystander_traffic.py` — nền eMBB bystander.
- `tests/test_env_week4.py`, `tests/test_env_hard.py` (hard mission φ₃).

## Sửa (audit post-cleanup)
- ⚠️ **obs K=1 = 33-dim** tại thời điểm W04 (post gỡ LSTM 6 + MEC 1). **Lịch sử: nay = 31 (K=1) / 51 (K=3)** sau refactor 2026-06-14/15 (phase→severity, F=4→1, per-amb queue/AoI + severity_k) — xem [08_implementation_notes.md](../08_implementation_notes.md). Số chuẩn = derived từ field-set + `assert observation_space.shape`; chốt lại ở [W18](W18_pha3_algorithm_code.md) (audit #2).

## Gate G1 ✅
- D_e2e @ φ₃ (r_min=0.6) = D_tx+D_queue+D_FH+D_BH ≈ 0.7–0.9ms < 1ms ✓; ΣPRB ≤ 273 mọi tick; queue ρ<1 mọi slice.

## Liên kết
- Phân rã D_e2e ground ref ở [W13](W13_pha1_delay_reliability_qos.md) (M4); obs/action spec đầy đủ ở [W18](W18_pha3_algorithm_code.md) + `docs/07_api_spec.md`.
