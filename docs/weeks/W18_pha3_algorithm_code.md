# W18 — Pha 3: Thuật toán (PA-CHRL-PPO) + Code K≥2 + PPO solver run, K=1

> **Pha**: 3 · **Status**: 📅 PLANNED · **Gate**: **GATE 3A** · **Solver**: PA-CHRL-PPO · **K**: 1 (1 xe cứu thương) · **Nhóm**: A1–A4 · **Build**: B0/B0b + B5 · **Deps**: GATE 2

## Env config khóa (áp dụng XUYÊN SUỐT W18–W23, KHÔNG đổi giữa các tuần)
- **gNB / cell-center**: cố định tại local origin `(0,0)` m — điểm hội tụ 3 xe trên đường Giải Phóng, anchor bởi GPS BV Bạch Mai (`config.py: BACH_MAI_LAT/LON`, dùng cho lớp OSM/SUMO W15, KHÔNG dùng trực tiếp trong RL env).
- **R_cell = 300 m** (`cell_radius_m`, `config.py`/`oran_env.py: EnvConfig`, đổi từ 200→300).
- **Channel**: UMi (Urban Micro Street Canyon), 3GPP TR 38.901, single-cell, **no handover**.
- K thay đổi theo tuần (K=1 ở W18/W20/W22; K=3 ở W19/W21/W23) — MỌI thông số geometry khác giữ nguyên.

## A — Thuật toán (gắn nguồn corpus)
- **A1.1** PPO clipped surrogate — ✅[`1707.06347v2.pdf` (Schulman 2017)]
- **A1.2** GAE — ✅[`Foundations_of_Deep_Reinforcement_Learning….pdf` §GAE] *(Schulman 2016 vắng corpus)*
- **A2.1** Two-timescale HRL (Manager `γ_H`, Worker `γ_L`; `γ_H≈γ_L^W`) — ✅[`Hasan Anıl Akyıldız…[2024].pdf`; `Hierarchical_RL_based_Resource.pdf`] *(Borkar/FeUdal vắng corpus)*. ⚠️ `γ_H/γ_L` = RL discount, KHÁC `β` priority temp.
- **A2.1b** `W = WORKER_STEPS_PER_MANAGER = 10` (`config.py:315`): MAC TTI=0.5ms ✅[TS 38.211 μ=1]; Worker(xApp)=10ms; Manager(rApp)=1s — chuẩn O-RAN [`O-RAN.pdf`].
- **A3.1** Dual ascent λ-update — ✅[`Spoor…[2025].pdf`; `Ding…[2023].pdf`] (nối P4).
- **A4.1** Hạ tầng K=3 + intra-slice + C6 → code (dùng cho [W19](W19_pha3_e1_baseline_sumo.md), build 1 lần ở đây).

## B5 — Code K≥2 (+ removals B0/B0b)
- `oran_env.py`: obs +severity_k +AoI_worstnorm/mean +λ_C{1,2,4,5}^k; action +**β** (squash [β_min,β_max]); intra-slice PRB split Option B; phase-event severity MAP. **`assert observation_space.shape == (58,)`** (K=3) — KHÔNG hard-code số (audit #2).
- `lagrangian.py`: C6 per-pair (chỉ K≥2).
- **Safety filter**: `IdentityNSF` → **closed-form `Π_feasible`** = projection-onto-simplex [Duchi 2008] + isotonic/PAV (`r_ded≤r_min≤r_max`) + clip [0,1] + floor `r_min≥K·PRB_min/P_TOTAL` (#17, Euclidean projection CHÍNH XÁC). **Gỡ `β_qp` + `LR_NSF`** (no learnable params); KHÔNG claim NSF novel.
- **B0 LSTM removal** + **B0b MEC removal**: gỡ `lstm_*`, `vital_simulator.py`, `mec_model.py`, `u_MEC` obs, `F_MEC`; verify `observation_space.shape==(33,)` (K=1).

## C — PPO solver run, K=1
- **C1** Train PA-CHRL-PPO (Manager+Worker) trên env K=1, R_cell=300m, UMi single-cell, no handover, ≥10 seeds.
- **C2** Log: episode reward, 5 constraint costs C1–C5, λ-trajectory (λ_C1..C5), λ-saturation-rate (= %step `λ_c==Λ_max=10`).
- **C3** Sanity: `test_oran_env_sanity`, `test_reward_constraint_tracking` pass với obs=33.

## ⟲ RÀ SOÁT A + B5 + C
Tên thuật toán/baseline KHÔNG gán citation chưa kiểm chứng; PPO file đúng; two-timescale ghi rõ W; projection trả action LUÔN khả thi; grep 0 ref `lstm`/`mec`/`vital`/`β_qp`; env config (gNB=(0,0), R_cell=300, UMi, no handover) áp dụng đúng cho cả K=1 và K=3.

## GATE 3A
`test_multi_ambulance` + `test_intra_slice` pass (no-starvation, structural ordering); **assert obs 33/58 đúng**; PPO K=1 train hội tụ (reward tăng, λ ổn định, no saturation-without-convergence); ~261 tests xanh; mọi nhãn A* ✅ hoặc 🟡-đã-xử-lý.

## Liên kết
Master plan PHẦN 11/W18 + PHẦN 8 (projection) · `docs/05_agent_workflow.md`, `docs/07_api_spec.md` · tiếp PPO K=3 → [W19](W19_pha3_e1_baseline_sumo.md).
