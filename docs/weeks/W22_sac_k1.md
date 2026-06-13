# W22 — Pha 3: SAC-Lag solver code + run, K=1

> **Pha**: 3 · **Status**: 📅 PLANNED · **Gate**: **GATE 3E** · **Solver**: SAC-Lag (off-policy, max-entropy) · **K**: 1 (1 xe cứu thương) · **Build**: B7 (NEW: `agents/sac_agent.py` + `baselines/sac_lag.py`) · **Deps**: GATE 3D

## Env config (giống W18/W20, KHÔNG đổi)
gNB/cell-center = `(0,0)`, R_cell=300m, UMi 3GPP TR 38.901, single-cell, no handover. `K_ambulances=1` → obs=33.

## B7 — Code SAC-Lag (sibling thứ 3, mirror TD3-Lag)
- **B7.1** `agents/sac_agent.py` — ✅[Haarnoja et al. 2018 "Soft Actor-Critic" (ICML); Haarnoja et al. 2018 "SAC Algorithms and Applications" (arXiv:1812.05905)]. Stochastic tanh-Gaussian actor (reparameterization, log_prob với tanh-correction), twin Q-critic + target (Polyak τ=0.005), automatic entropy temperature `α` (target entropy = `-action_dim`), replay buffer (reuse `ReplayBuffer` từ `td3_agent.py`).
- **B7.2** `baselines/sac_lag.py` — `SACLagBaseline` (`name="sac_lag"`), mirror `TD3LagBaseline`: cùng `BaselineFlags(use_phase=False, use_cmdp=True, use_hrl=False, n_constraints=5)`, cùng `LambdaState` 5-dim λ lifecycle (`on_episode_start`, `on_manager_step_start`, `accumulate_constraint`, `on_manager_step_end`, `augment_reward`), `mask_phase`, `select_action()` trả `(action, log_prob, value)` để khớp PPO API.
- **B7.3** Smoke test: actor/critic forward pass đúng shape, `update()` trả `critic_loss`/`actor_loss`/`alpha_loss`/`alpha`, `save`/`load` roundtrip.

## A-SAC — SAC-Lag solver run, K=1
- **A-SAC.1** Train `SACLagBaseline` trên env K=1, ≥10 seeds, replay buffer + warmup (500 steps) như cấu hình mặc định `sac_lag.py`.
- **A-SAC.2** Log: episode reward, 5 constraint costs C1–C5, λ-trajectory (λ_C1..C5) + saturation-rate, critic/actor/alpha loss, α-trajectory (entropy temperature).

## ⟲ RÀ SOÁT SAC K=1
`test_oran_env_sanity` + `test_reward_constraint_tracking` pass với obs=33 (env không đổi — chỉ solver mới SAC-Lag); SAC-Lag code mirror đúng convention TD3-Lag (cùng `BaselineFlags`, cùng `LambdaState`); KHÔNG so sánh chéo solver trong tuần này (để dành Table I/II ở W24).

## GATE 3E
SAC-Lag K=1 train hội tụ (critic/actor/alpha loss giảm hoặc ổn định, α hội tụ về giá trị hợp lý, reward tăng); 5 constraint costs C1–C5 trong ngưỡng hoặc λ phản ứng đúng hướng; không có saturation-without-convergence chưa-giải-thích.

## Liên kết
Master plan PHẦN 11/W22 · tiếp SAC K=3 → [W23](W23_sac_k3.md).
