# W08 — Algorithm 1 Training Loop + Integration

> **Pha**: GĐ A (code foundation) · **Status**: ✅ DONE · **Gate**: G3.2 — 5-ep smoke no NaN (cả 3 solver) · **Deps**: W07/G3.1

## Đã xây
- `train.py` — Algorithm 1 đầy đủ (Manager/Worker rollout, GAE, PPO update, LambdaState augmented reward).
- `solvers/smoke_train.py` — smoke cho td3 + sac.
- `tests/test_train_loop.py` — 5-ep smoke.

## Gate G3.2 ✅
- `train.py --algo ppo --episodes 5` chạy không crash; `metrics.csv` 5 rows finite (no NaN/Inf); λ_global đôi khi >0 (constraint binding visible); baseline smoke (td3, sac) chạy với 5-dim λ.

## Liên kết
- Training loop pseudocode → `docs/13_methodology_walkthrough.md` §3.4. Verify đầy đủ → [W09](W09_smoke_phase3_gate.md) (G3).
