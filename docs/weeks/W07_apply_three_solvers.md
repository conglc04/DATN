# W07 — Apply ALL 3 Solvers (PA-CHRL-PPO + TD3-Lag + B3)

> **Pha**: GĐ A (code foundation) · **Status**: ✅ DONE · **Gate**: G3.1 — 3 solvers instantiate, dims đúng · **Deps**: W06/G2

## Đã xây (3 solver = SIBLINGS, áp dụng SAU khi Pha 2 statement complete)
- `agents/ppo_core.py`, `agents/manager_agent.py` (rApp, action 4-dim K=1), `agents/worker_agent.py` (xApp, 12-dim state).
- `agents/nsf.py` — slot safety-filter (**hiện `IdentityNSF` no-op**).
- `baselines/td3_lag.py` — TD3 + Lagrangian [Fujimoto 2018, `fujimoto18a.pdf`].
- `baselines/b3.py` — **Lagrangian-penalty PPO baseline**.

## Sửa (audit post-cleanup)
- ⚠️ **B3 = "Lagrangian-penalty", KHÔNG đặt tên "RCPO/Tessler"** (RCPO gốc vắng corpus; mô tả cơ chế theo [Spoor 2025]).
- ⚠️ `nsf.py` `IdentityNSF` → **sẽ thay bằng closed-form `Π_feasible`** (projection-onto-simplex Duchi + isotonic, [TS 28.541]) ở [W18](W18_pha3_algorithm_code.md)/B5; gỡ `β_qp`/`LR_NSF` (no learnable params). KHÔNG claim NSF novel.

## Gate G3.1 ✅
- 3 solver instantiate không crash; forward dims đúng (Manager 4 / Worker 12 / baseline 6-action); `LambdaState` tích hợp cả 3.

## Liên kết
- Thuật toán formal (PPO/GAE/HRL two-timescale) → [W18](W18_pha3_algorithm_code.md) (A1-A2). Projection #17 → [W18](W18_pha3_algorithm_code.md)/B5 + master plan PHẦN 8.
