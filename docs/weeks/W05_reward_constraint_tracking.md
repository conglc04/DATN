# W05 — Reward + 5-Constraint Tracking (Pha 2 code I)

> **Pha**: GĐ A (code foundation) · **Status**: ✅ DONE · **Gate**: G2.1 — reward + c_vec/d_phi exposed · **Deps**: W04/G1

## Đã xây
- `env/oran_env.py` — **reward single-term**: `reward = α_e·log(1 + R_eMBB_aggregate / R_REF)` (R_REF=100Mbps). **Severity & URLLC KHÔNG vào reward** — enforced qua Lagrangian constraint (tránh double-counting với λ).
- 5-constraint tracking: `info{c_vec, d_phi, phase_now}` mọi `step()` (W05-era; sau phase→severity swap 2026-06-14: KHÔNG còn `phase_now`, thay bằng `severity`(=`severity_ref`)/`severity_per_amb`).
- `tests/test_env_phase2.py`.

## Sửa (audit post-cleanup)
- ⚠️ Reward = eMBB log-utility ONLY (khớp [Alsenwi 2022 Eq.13; Sohaib 2024 Eq.9]); bỏ mọi biến thể reward đa-term/α-per-phase cũ.
- ⚠️ obs extend: **33-dim** (K=1) sau gỡ LSTM+MEC tại thời điểm W05 (KHÔNG 40); **lịch sử: nay 31** sau refactor 2026-06-14/15 — xem [08_implementation_notes.md](../08_implementation_notes.md).

## Gate G2.1 ✅
- `info` có `c_vec, d_phi, phase_now` (nay `severity`/`severity_per_amb`); reward verify analytic = single-term; `get_phase_thresholds(3)` (nay `get_severity_thresholds(3)`) khớp 3GPP TS 22.261.

## Liên kết
- Objective/constraint formal → [W17](W17_pha2_cmdp_formulation.md) (P1, P3 C1-C6). QoS ngưỡng → [W13](W13_pha1_delay_reliability_qos.md) (M6).
