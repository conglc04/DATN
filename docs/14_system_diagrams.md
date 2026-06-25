# 14 — System Diagrams

> Sơ đồ hệ thống dựng trực tiếp từ source `baselines/` (audited 2026-06-13), từ tổng quát → chi tiết.
> Nguồn vẽ: `figures/*.dot` (Graphviz); regenerate bằng `bash figures/_gen_diagrams.sh` → SVG + PNG.
> SVG = vector (dùng cho luận văn); PNG @150dpi = preview nhanh.

## 1. Pipeline tổng quát (3 pha phương pháp luận)
![Pipeline](figures/01_pipeline.svg)

3 pha: Mô hình hệ thống → Bài toán tối ưu CMDP → 3 solver ngang hàng (PPO/TD3/SAC). RL là bước cuối; đóng góp = bài toán, không phải thuật toán. Xem [13 §Pipeline](13_methodology_walkthrough.md).

## 2. Kiến trúc O-RAN 3 tầng + thang thời gian
![O-RAN architecture](figures/02_oran_arch.svg)

Near-RT RIC (Manager 100ms → Worker 10ms) → O-DU/O-RU MAC (TTI 0.5ms); W=10 Worker/Manager. Single-cell UMa R=1km @ Bạch Mai (+ interference margin). Xem [03_architecture](03_architecture.md).

## 3. Bên trong môi trường `oran_env.py` (1 Worker step = 20 MAC tick)
![Env internal](figures/03_env_internal.svg)

Vòng 20 MAC tick: channel → arrivals → M/G/1 queue → delay/throughput → reward + c_vec; severity (cố định/episode) + AoI. Reward = `mean_tick log(1+R_eMBB/R_REF)` (MEAN over ticks, KHÔNG α_e — bỏ 2026-06-23). Xem [04_data_flow](04_data_flow.md).

## 4. Bài toán tối ưu CMDP (objective + ràng buộc + Lagrangian)
![CMDP](figures/04_cmdp.svg)

Objective single-term `mean_tick log(1+R_eMBB/R_REF)` (KHÔNG α_e — severity vào hệ qua constraint+λ); C1–C5 hard với ngưỡng `d_j^sev` theo severity (C6 demote→metric, K≥2); dual ascent `λ←clip(λ+α_λ·ĝ,0,10)`, subgradient chuẩn hóa `ĝ=(c−d)/scale`. λ_local + severity inject vào obs (Markov). Xem [02_requirements](02_requirements.md#severity-qos-table) + [13 §2](13_methodology_walkthrough.md).

## 5. Không gian quan sát (32-dim, K=1) & hành động (1-dim K=1 / K-dim K≥2)
![State/Action](figures/05_state_action.svg)

obs = 20 fixed + 11K + F(=1) = 32 (K=1; K=3,F=1 → 54); severity_ref one-hot tại [10:15], λ_local_C3_shared (scalar) tại [15]; per-amb 11-dim block (`SINR_k,d_k,v_k,delay_norm_k,AoI_norm_k,severity_norm_k,λ_C1_k,λ_C2_k,λ_C4_k,λ_C5_k,active_mask_k`) (2026-06-15 severity_per_amb epic: per-xe `severity_per_amb[k]∈{1..5}` độc lập/cố định-episode, `severity_ref:=max` cho shared; obs 30→31, formula `24+5K+F`→`20+10K+F`; 2026-06-23 +active_mask_k: `20+10K+F`→`20+11K+F`, 31→32/51→54; lịch sử trước: phase→severity swap 33→31, F=4→F=1 31→28, per-ambulance queue/AoI split 28→30). Action K=1: 1-dim (no-op scalar, xe duy nhất nhận toàn bộ URLLC PRBs); K≥2: K-dim per-vehicle logits (pure-RL, no β). Worker KHÔNG điều khiển inter-slice (r_min/r_max/r_ded là Manager-owned obs, không phải Worker action). Xem [07_api_spec](07_api_spec.md).

## 6. Vòng huấn luyện & điểm đối xứng 3 solver
![Training loop](figures/06_training_loop.svg)

Ô xanh = λ-overlay dùng chung (`utils/obs.py`) cho cả 3 solver; ô vàng = dual ascent quyết định timing `s_next`. PPO (on-policy HRL) vs TD3/SAC (flat off-policy) — đều thấy λ + pha như nhau (audit fix 2026-06-13, xem [08 §audit](08_implementation_notes.md)).

## 7. Một vòng chạy hoàn chỉnh — 1 rollout (1 s) trong 1 episode (= trọn mission)
![One episode](figures/07_one_episode.svg)

Phân cấp 3 thang thời gian lồng nhau:
**Episode = trọn hành trình MCI** (reset → sample severity cố định cả episode + load `λ_warm[sev]`; env chạy tới all-arrived / 400s). Bên trong, **1 rollout = cửa sổ env-interaction CHUNG cho cả 3 solver** (= **Manager step ×10** (Manager 100ms: `on_manager_step_start(sev)` no-op vì severity cố định + `manager.act → b_rrm`) → **Worker step ×W=10** (xApp 10ms: overlay λ → `worker.act` → `env.step`) → **20 MAC tick** (O-DU 0.5ms)). Cuối mỗi Manager step: **dual ascent** `λ←clip(λ+α_λ·ĝ,0,10)` với `ĝ=win_c/win_steps`. Cuối mỗi rollout: **PPO update** (Worker+Manager buf, GAE bootstrap V(s) **trừ khi terminated** — timeout/truncation vẫn bootstrap) rồi **tiếp tục cùng env**; TD3/SAC update mỗi step (mask `(1−terminated)`). Cuối episode (mission done): flush `λ_warm[sev]`.
Tổng **1 rollout**: `10 × 10 × 20 = 2000 TTI × 0.5ms = 1.0 s`; **1 episode = N rollouts** tới khi cả K xe arrived (hoặc 400s). Xem [13 §1.4](13_methodology_walkthrough.md) (thang thời gian) + `train.py`.
