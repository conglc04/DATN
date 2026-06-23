# W22 — Pha 3: SAC solver code + run, K=1

> **Pha**: 3 · **Status**: 📅 PLANNED · **Gate**: **GATE 3E** · **Solver**: SAC (off-policy, max-entropy) · **K**: 1 (1 xe cứu thương) · **Build**: B7 (NEW: `agents/sac_agent.py` + `solvers/sac.py`) · **Deps**: GATE 3D

## Env config (giống W18/W20, KHÔNG đổi)
gNB/cell-center = `(0,0)`, R_cell=1km, UMa 3GPP TR 38.901 + interference margin −86 dBm/PRB, single-cell, no handover. `K_ambulances=1` → obs=31.

## B7 — Code SAC (sibling thứ 3, mirror TD3)
- **B7.1** `agents/sac_agent.py` — ✅[Haarnoja et al. 2018 "Soft Actor-Critic" (ICML); Haarnoja et al. 2018 "SAC Algorithms and Applications" (arXiv:1812.05905)]. Stochastic tanh-Gaussian actor (reparameterization, log_prob với tanh-correction), twin Q-critic + target (Polyak τ=0.005), automatic entropy temperature `α` (target entropy = `-action_dim`), replay buffer (reuse `ReplayBuffer` từ `td3_agent.py`).
- **B7.2** `solvers/sac.py` — `SACSolver` (`name="sac"`), mirror `TD3Solver`: cùng `BaselineFlags(use_phase=True, use_cmdp=True, use_hrl=True, n_constraints=5)` (**HRL thật**: `self.manager = SACManagerAgent`, parity với PPO/TD3 — W18+), cùng `LambdaState` (4K+1)-dim λ lifecycle (=5 ở K=1) (`on_episode_start`, `on_manager_step_start`, `accumulate_constraint`, `on_manager_step_end`, `augment_reward`), `mask_severity`, `select_action()` trả `(action, log_prob, value)` để khớp PPO API.
- **B7.3** Smoke test: actor/critic forward pass đúng shape, `update()` trả `critic_loss`/`actor_loss`/`alpha_loss`/`alpha`, `save`/`load` roundtrip.

## A-SAC — SAC solver run, K=1
- **A-SAC.1** Train `SACSolver` trên env K=1, ≥10 seeds, replay buffer + warmup (500 steps) như cấu hình mặc định `sac.py`.
- **A-SAC.2** Log: episode reward, constraint costs (c_vec (4K+1)=5 ở K=1, layout `[C1,C2,C4,C5,C3_shared]`), λ-trajectory + saturation-rate, critic/actor/alpha loss, α-trajectory (entropy temperature).

## ⟲ RÀ SOÁT SAC K=1
`test_oran_env_sanity` + `test_reward_constraint_tracking` pass với obs=31 (env không đổi — chỉ solver mới SAC); SAC code mirror đúng convention TD3 (cùng `BaselineFlags`, cùng `LambdaState`); KHÔNG so sánh chéo solver trong tuần này (để dành Table I/II ở W24).

## GATE 3E
SAC K=1 train hội tụ (critic/actor/alpha loss giảm hoặc ổn định, α hội tụ về giá trị hợp lý, reward tăng); 5 constraint costs C1–C5 trong ngưỡng hoặc λ phản ứng đúng hướng; không có saturation-without-convergence chưa-giải-thích.

## Liên kết
Master plan PHẦN 11/W22 · tiếp SAC K=3 → [W23](W23_sac_k3.md).
