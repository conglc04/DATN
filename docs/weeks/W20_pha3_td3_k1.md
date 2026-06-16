# W20 — Pha 3: TD3 solver run, K=1

> **Pha**: 3 · **Status**: 📅 PLANNED · **Gate**: **GATE 3C** · **Solver**: TD3 (off-policy) · **K**: 1 (1 xe cứu thương) · **Build**: B6 (`agents/td3_agent.py` + `solvers/td3.py`, đã có sẵn) · **Deps**: GATE 3B

## Env config (giống W18/W19, KHÔNG đổi)
gNB/cell-center = `(0,0)`, R_cell=300m, UMi 3GPP TR 38.901, single-cell, no handover. `K_ambulances=1` → obs=31 (như W18).

## A-TD3 — TD3 solver (sibling off-policy, (4K+1)=5-dim λ ở K=1)
- **A-TD3.1** Baseline TD3 — ✅[`fujimoto18a.pdf`]. Off-policy backbone (twin Q, target policy smoothing, delayed policy update) + (4K+1)=5-dim λ ở K=1 (layout `[C1,C2,C4,C5,C3_shared]`) qua `LambdaState` (cùng cơ chế PPO, dual ascent 1 lần/Manager-step boundary).
- **A-TD3.2** Train `TD3Baseline` trên env K=1, ≥10 seeds, replay buffer + warmup (500 steps) như cấu hình mặc định `td3.py`.
- **A-TD3.3** Log: episode reward, constraint costs (c_vec (4K+1)=5 ở K=1, layout `[C1,C2,C4,C5,C3_shared]`), λ-trajectory + saturation-rate, critic/actor loss.

## ⟲ RÀ SOÁT TD3 K=1
`test_oran_env_sanity` + `test_reward_constraint_tracking` pass với obs=31 (như W18, env không đổi — chỉ solver đổi); λ-trajectory + saturation-rate đã log; KHÔNG so sánh chéo solver trong tuần này (để dành Table I/II ở W24).

## GATE 3C
TD3 K=1 train hội tụ (critic/actor loss giảm, reward tăng, λ ổn định); 5 constraint costs C1–C5 trong ngưỡng hoặc λ phản ứng đúng hướng; không có saturation-without-convergence chưa-giải-thích.

## Liên kết
Master plan PHẦN 11/W20 · tiếp TD3 K=3 → [W21](W21_pha3_td3_k3.md).
