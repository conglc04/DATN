# W04 — ORANEnv Complete + Sanity

> **Pha**: GĐ A (code foundation) · **Status**: ✅ DONE · **Gate**: G1 — D_e2e < 1ms @ φ₃ · **Deps**: W03/G1.2

## Đã xây
- `env/oran_env.py` — Gym `ORANEnv` gộp tất cả env modules; `step()` advance 20 MAC tick / Worker step (10ms = T_L); `reset()` init phase/queue/ambulance.
- `env/bystander_traffic.py` — nền eMBB bystander.
- `tests/test_env_week4.py`, `tests/test_env_hard.py` (hard mission φ₃).

## Sửa (audit post-cleanup)
- ⚠️ **obs K=1 = 33-dim** tại thời điểm W04 (post gỡ LSTM 6 + MEC 1). **Lịch sử: nay = 31 (K=1) / 51 (K=3)** sau refactor 2026-06-14/15 (phase→severity, F=4→1, per-amb queue/AoI + severity_k) — xem [08_implementation_notes.md](../08_implementation_notes.md). Số chuẩn = derived từ field-set + `assert observation_space.shape`; chốt lại ở [W18](W18_pha3_algorithm_code.md) (audit #2).

## Gate G1 ✅
- Kiểm tra **D_e2e < 1ms theo hồ sơ QoS của severity-5 (IMMEDIATE)** — ngưỡng 1ms = `SEVERITY_QOS[5]["D_max"]`, ca chặt nhất nên là điều kiện ràng buộc. D_e2e = D_DET+D_tx+D_queue+D_FH+D_BH (r_min=0.6) < 1ms ✓ (audit 2026-06-16: đo 0.279ms); ΣPRB ≤ 273 mọi tick (đo 272). **Ổn định ρ<1 chỉ yêu cầu cho slice URLLC** (đảm bảo D_e2e); **eMBB = best-effort greedy** (reward=log(1+R_eMBB), KHÔNG ràng buộc latency) ⟹ eMBB CÓ THỂ bão hòa ρ≥1 by-design, KHÔNG phải vi phạm. *(Doc cũ ghi "ρ<1 mọi slice" — overreach, đã thu hẹp về URLLC.)*

## Liên kết
- Phân rã D_e2e ground ref ở [W13](W13_pha1_delay_reliability_qos.md) (M4); obs/action spec đầy đủ ở [W18](W18_pha3_algorithm_code.md) + `docs/07_api_spec.md`.
