# W07 — Apply ALL 3 Solvers (PPO + TD3 + SAC)

> **Pha**: GĐ A (code foundation) · **Status**: ✅ DONE · **Gate**: G3.1 — 3 solvers instantiate, dims đúng · **Deps**: W06/G2
> **REVISED**: B3-RCPO (Lagrangian-penalty PPO) đã loại HOÀN TOÀN; sibling off-policy thứ 2 = SAC (B7). β_qp/NSF-distillation/`nsf.py` đã gỡ — safety = closed-form `Π_feasible`.

## Đã xây (3 solver = SIBLINGS, áp dụng SAU khi Pha 2 statement complete)
- `agents/ppo_core.py`, `agents/manager_agent.py` (rApp, action 1-dim K=1 sau gỡ MEC), `agents/worker_agent.py` (xApp).
- `solvers/td3.py` — TD3 + Lagrangian (deterministic actor) [Fujimoto 2018, `fujimoto18a.pdf`].
- `solvers/sac.py` — SAC + Lagrangian (max-entropy stochastic actor) [Haarnoja 2018, `1812.05905v2.pdf`].

## Safety filter
- Closed-form `Π_feasible` (projection-onto-simplex Duchi + isotonic, [TS 28.541]) — no learnable params. KHÔNG claim novel.

## Gate G3.1 ✅
- 3 solver instantiate không crash; forward dims đúng; `LambdaState` (5-dim λ) tích hợp cả 3.

## Liên kết
- Thuật toán formal (PPO/GAE/HRL two-timescale) → [W18](W18_pha3_algorithm_code.md) (A1-A2). Projection #17 → [W18](W18_pha3_algorithm_code.md)/B5 + master plan PHẦN 8.
