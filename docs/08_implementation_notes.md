# 08 — Implementation Notes

> Repo code root: `/home/cong/Desktop/USB_BACKUP/Do-an/baselines/` (Linux; đổi tên từ `pa_chrl_ppo/`). Path Windows `f:\Do an\` cũ = bỏ.

## Code map
```
baselines/                       # repo code root (đổi tên từ pa_chrl_ppo/)
├── env/
│   ├── channel_model.py      # UMa@1km (macro, config-driven bs_layer) 3GPP TR 38.901 + interference margin −86 dBm/PRB
│   ├── queue_model.py        # M/G/1 Pollaczek–Khinchine
│   ├── traffic_gen.py        # URLLC (F=1, "ambulance_status") + eMBB bystander, payload/rate
│   ├── aoi_tracker.py        # AoI LCFS+drop (timestamp only, no vitals; 1 luồng "ambulance_status"); per-ambulance K tracker
│   ├── sumo_mobility.py      # SUMO FCD trace reader (GPS→metric); default mobility (RWP removed)
│   └── oran_env.py           # Gym ORANEnv; severity sampling inline (config.sample_severity=True, KHÔNG phải file riêng env/severity.py)
├── agents/                   # ppo_core, manager_agent (action 1-dim), worker_agent, lagrangian, td3_agent, sac_agent
├── solvers/                  # 3 solver ngang hàng: PPO (= train_ppo + agents/) + td3.py + sac.py; + ablation variants + train_offpolicy.py
├── scripts/audit_gate3.py    # post-training Gate 3x scorecard (đọc metrics.csv: convergence/C1-C5/λ-sat/health)
├── run_30runs.py             # production batch runner: N method × N seed
├── stats_analysis.py         # so sánh thống kê sweep (Mann-Whitney, Holm-Bonferroni, Cohen's d/Hedges' g)
├── data/sumo/                # hanoi_bachmai.osm, *.net.xml, ambulance_routes.xml, *.fcd.xml
└── utils/config.py           # single source: P_TOTAL=273, SEVERITY_QOS, LAMBDA_WARM, ALPHA_LAMBDA_DUAL=1e-4, LAMBDA_MAX=10, WORKER_STEPS_PER_MANAGER=10
```

> `experiments/` (package rỗng, chỉ còn `__init__.py`, không import ở đâu) đã XÓA 2026-06-25 — dọn dự án trước khi viết báo cáo.
>
> **phase_detector.py XÓA (2026-06-14)** — 5-pha FSM (chu kỳ trực của xe) bị thay bằng **5 mức severity bệnh nhân** (Non-urgent…Immediate) làm trục ưu tiên duy nhất. Xem mục "Phase→Severity swap" bên dưới.

## Removals (B0/B0b — ✅ ĐÃ THỰC THI 2026-06-13)
> Loại khỏi **scope/design** VÀ khỏi **code** (đã xóa thật, test xanh). obs K=1: 40 → **33**.
- ✅ **LSTM** (D10): obs −6; `LSTM_*` config gỡ.
- ✅ **MEC** (D23): `env/mec_model.py` XÓA; `MECServer`/`u_MEC` obs (−1)/`F_MEC` gỡ; Manager action 2→1 (bỏ f_MEC). D_FH/D_BH delay GIỮ.
- ✅ **vital_simulator.py** XÓA (sinh vitals giả). GIỮ telemetry — gộp thành 1 luồng `ambulance_status` (F=1, traffic_gen; xem "F=4→F=1 stream consolidation" bên dưới, thay cho HR/SpO2/ECG/DENM cũ).
- ✅ **β_qp / NSF distillation / nsf.py / LR_NSF** XÓA: kế hoạch ban đầu (W18/B5) là safety filter → closed-form `Π_feasible` (Duchi simplex + isotonic) — **nhưng outcome thực tế khác kế hoạch**: implement thành severity-tier N_req 2-pha (KHÔNG Duchi/isotonic), rồi **gỡ tiếp 2026-06-21** thành pure-RL softmax thuần (xem "Per-ambulance severity_k epic" dưới + `agents/worker_agent.py` docstring). Π_feasible (mọi biến thể) KHÔNG còn trong code hiện tại.
- ✅ **B3-RCPO** XÓA hoàn toàn (3 solver = PPO/TD3/SAC).
- → obs K=1: 40 → **33** (B0/B0b) → **31** (phase→severity swap) → **28** (F=4→F=1 stream consolidation) → **30** (per-ambulance queue/AoI split) → **31** (per-ambulance `severity_k` epic, `20+10K+F`) → **32** (+active_mask_k, `20+11K+F`, 2026-06-23, xem dưới); K=3,F=1 → **54**.

## Phase→Severity swap (✅ 2026-06-14) — severity thay phase làm trục ưu tiên

> **Lý do**: kịch bản "xe luôn có bệnh nhân" ⟹ 3/5 pha cũ (STANDBY/DISPATCH/RETURN, không có bệnh nhân) vô nghĩa. Phase (chu kỳ trực của xe, chạy bằng tín hiệu sự kiện) bị thay bằng **5 mức severity bệnh nhân** (NON_URGENT→IMMEDIATE) — exogenous, **cố định trong 1 episode, random giữa các episode**.

- **Trục độc lập**: severity (ưu tiên/độ chặt QoS, sample/episode) ⊥ mobility (kênh, từ `speed` obs[24:27] — kiêm luôn vai trò scene/transport mà phase từng dán nhãn; SUMO/OSM B3 vẫn đóng góp ở đây).
- **config**: `PHASE_QOS`→`SEVERITY_QOS`, `PHASE_ALPHA`→`SEVERITY_ALPHA`, `CMDP_D_J_PHI`→`CMDP_D_J_SEVERITY`, `LAMBDA_WARM` re-key — **đơn điệu** (sev 1 lỏng nhất → sev 5 chặt nhất = giá trị φ3 SCENE cũ). `get_phase_*`→`get_severity_*`. `D_REF_URLLC`/`AOI_REF_S` không đổi (= mức IMMEDIATE).
- **obs 33→31**: phase one-hot [10:15] → **severity one-hot** (cùng vị trí); **bỏ** t_φ/ETA [15:17] (severity cố định ⟹ không có đồng hồ pha/ETA) ⟹ λ_local dịch **17→15** (`LAMBDA_LOCAL_OBS_INDEX=15`, obs[15:20]).
- **xóa**: `phase_detector.py`, `EnvConfig.phase_trajectory`, `_advance_phase_trajectory`/`_compute_eta_next`, `experiments/exp3_phase_transition.py` (severity cố định ⟹ không còn transition để test).
- **lagrangian**: rename `reset_episode(initial_severity)`, `on_manager_step_start(severity_now)` (no-op vì severity cố định/episode — giữ cho K≥2), `phi_prev`→`sev_prev`. `hard_mission_config` = severity 5 IMMEDIATE + giữ burst/bystander.
- **ablation**: `mask_phase`→`mask_severity` (zero obs[10:15]); `no_phase_ppo`/`b2_hrl_ppo_soft` = severity-blind.
- VERIFIED: **221 test xanh**; PPO/TD3/SAC chạy e2e (obs=31, severity one-hot, λ overlay). Regression guard `test_td3.py::TestSolverObservesLambda`.

## F=4→F=1 stream consolidation (✅ 2026-06-14) — 1 luồng `ambulance_status`/xe

> **Lý do**: 4 luồng AoI cũ (HR_aggregated/SpO2_aggregated/ECG_waveform/DENM) là tách rời mức-bookkeeping AoI phủ lên CÙNG một mô hình queue/Poisson đơn-luồng — không gộp trung bình 4 luồng (giữ semantics), mà mỗi xe có **đúng 1 luồng task/traffic tổng hợp**: `ambulance_status`.

- **config.py**: `SEVERITY_QOS["AoI_max_HR"|"AoI_max_SpO2"|"AoI_max_BP"]` → gộp 1 key **`AoI_max`** (giữ giá trị cũ của `AoI_max_HR`: {1:1.0, 2:0.5, 3:0.2, 4:0.1, 5:0.1}, KHÔNG averaging). `get_severity_thresholds()["d4"]` đọc từ `AoI_max`.
- **aoi_tracker.py**: `STREAM_TYPES` 9 entries → 1 entry `{"ambulance_status": {"queue": "LCFS", "drop_old": True, "aoi_aware": True}}`; `AOI_THRESHOLD_KEY = {"ambulance_status": "AoI_max"}`.
- **oran_env.py**: `DEFAULT_AOI_STREAMS = ("ambulance_status",)`; `EnvConfig.num_streams = 1`; `_mac_tick` đọc 1 scalar `aoi_tracker["ambulance_status"].current_aoi(...)` (không còn list/mean qua nhiều luồng); `_observe` AoI per-stream block [24+3K : 24+3K+F] = F=1.
- **obs K=1**: **31 → 28** (`24 + 3K + F`, F: 4→1). `WORKER_STATE_DIM_DEFAULT` (agents/worker_agent.py): 31→28.
- **Traffic params KHÔNG đổi**: `urllc_arrival_rate=50.0`, `urllc_packet_bits=400*8` (Poisson queue/PRB model M8.1b) giữ nguyên — khung "periodic status bundle ~500–1500B @ ~10–20Hz" cho `ambulance_status` là 🔴 declared/conceptual (docs only), KHÔNG recalibrate queue numerics (tránh re-validate 221 test).
- **xóa**: `experiments/exp8_aoi.py` (ablation LCFS-vs-FCFS trên 4 luồng HR/SpO2/BP/Temperature cũ — obsolete vì `STREAM_TYPES` chỉ còn 1 entry, không còn FCFS stream để so sánh).
- **K=3 tie-break không đổi**: constraint-slack → channel quality (SINR/distance) → AoI-ratio; F=1 ⟹ `AoI_worstnorm_k == AoI_mean_k` (đơn giản hoá, công thức vẫn tổng quát).
- VERIFIED: **219 test xanh** (221 − 2 từ xóa `exp8_aoi.py`/FCFS-specific test cases); PPO/TD3/SAC obs=28.

## Per-ambulance queue/AoI split (✅ 2026-06-14) — fix lỗi nghiêm trọng K≥2 không phân biệt được xe nào gần vi phạm QoS

> **Lý do**: với K≥2 xe cùng severity, `self.queues["urllc"]` là **1 hàng đợi M/G/1 pooled** chung cho K xe, và `self.aoi_trackers` là **1 tracker AoI global** — `_compute_e2e_delay()`/AoI chỉ trả về 1 scalar cho cả slice. Policy chỉ thấy khác biệt vật lý (SINR_k/d_k/v_k), KHÔNG thấy xe nào gần ngưỡng vi phạm C1/C2 (delay)/C4/C5 (AoI) hơn ⟹ **lỗi quan sát nghiêm trọng** cho mọi quyết định ưu tiên intra-slice.

- **K hàng đợi URLLC riêng**: `self.queues["urllc_0".."urllc_{K-1}"]` (mỗi `MG1Queue(mean_packet_bits=urllc_packet_bits)`); `"eMBB"` vẫn 1 hàng đợi pooled chung (bystander không gắn với xe cụ thể).
- **K Poisson arrivals độc lập**: `_sample_arrivals()` rút K mẫu `Poisson(eff_urllc_rate·tti_sec)` (1/xe) thay cho 1 mẫu `Poisson(rate·K)` — tương đương về thống kê tổng hợp, cho phép gán nguồn gốc theo xe.
- **K AoI tracker riêng**: `self.aoi_trackers: list[dict[str, AoIStreamTracker]]` — 1 dict/xe (mỗi dict vẫn `{"ambulance_status": AoIStreamTracker}`, F=1).
- **PRB split theo SINR riêng từng xe**: `_update_queue_service_rates()` chia `prb_urllc` cho K xe (`prb_urllc // K` + dư cho các xe đầu), mỗi xe dùng `capacity_per_prb_bps(self.last_sinr_db[k])` riêng — channel xấu hơn ⟹ service rate thấp hơn ⟹ D_e2e_k cao hơn.
- **obs K=1**: **28 → 30** (`24 + 5K + F`, khối per-amb 3K→5K). Khối per-amb mới = `{SINR_k, d_k, v_k, delay_norm_k, AoI_norm_k}` với `delay_norm_k = D_e2e_k/D_max^sev`, `AoI_norm_k = AoI_k/AoI_max^sev` — tỉ số O(1), ≈1.0 tại ngưỡng vi phạm, so sánh trực tiếp được giữa các xe cùng severity. `WORKER_STATE_DIM_DEFAULT` (agents/worker_agent.py): 28→30.
- **K=1 numerics KHÔNG đổi**: tất cả aggregation mean-over-K mới (`rho_urllc`, `hol_urllc_ms`, `aoi_mean`, `aoi_max`, c_vec[0,1,3,4]) quy về đúng giá trị scalar cũ khi K=1; `arr_urllc` = sum-over-K cũng quy về scalar cũ.
- **`info` backward-compat + additive**: `queue_diag_urllc` giữ = `queues["urllc_0"].summary()` (cùng shape dict cũ); thêm `queue_diag_urllc_per_amb` (list[K]), `delay_norm_per_amb`/`aoi_norm_per_amb` (shape (K,)) cho diagnostics.
- VERIFIED: **221 test xanh** (219 cũ + 2 test K=3 mới: obs shape (40,) cho K=3,F=1; `delay_norm_k` khác nhau giữa các xe khi SINR khác nhau dù cùng severity).

> Lưu ý lịch sử: bản "30-dim / 40-dim K=3" trên là điểm dừng tạm trước khi epic `severity_k` (dưới) đổi tiếp layout sang `20+10K+F` (31/51).

## Per-ambulance severity_k epic (✅ 2026-06-15) — severity độc lập từng xe + Lagrangian (4K+1)-dim *(β/Π_feasible mô tả dưới đã gỡ tiếp 2026-06-21 — xem ghi chú UPDATE trong từng mục)*

> **Lý do**: `self.severity` trước đó là 1 giá trị scalar dùng chung cho mọi xe (K xe cùng severity ⟹ C6/triage contention không có nghĩa). Epic này thêm `severity_per_amb ∈ {1..5}^K` (sampled độc lập per ambulance, cố định/episode); `severity_ref := max(severity_per_amb)` giữ vai trò scalar cũ cho mọi đại lượng SHARED (C3 R_min, severity one-hot, `info["severity"]`; **reward KHÔNG còn α_e** — bỏ 2026-06-23).

- **`(4K+1)`-dim Lagrangian/c_vec/d_phi**: layout `[C1_0..C1_{K-1}, C2_0..C2_{K-1}, C4_0..C4_{K-1}, C5_0..C5_{K-1}, C3_shared]` (C3 = eMBB throughput floor, shared/global, luôn ở slot cuối). K=1 = permutation `[0,1,3,4,2]` của thứ tự cũ `[C1,C2,C3,C4,C5]` — numerically identical, verify bằng `build_lambda_warm_vector`/`build_dual_scales`/`build_d_phi_vector` (`utils/config.py`).
- **`LambdaState` (agents/lagrangian.py)**: K-aware, `n_constraints=4K+1`; API `reset_episode(severity_per_amb, severity_ref)`, `on_manager_step_start(severity_per_amb, severity_ref)`, `on_episode_end(severity_per_amb, severity_ref)` (2-arg: tuple `(K,)` + int `severity_ref` — transition/EMA-save keyed bởi `severity_per_amb` tuple). `lam.dual_scales` (K-aware, instance attr) thay `CONSTRAINT_DUAL_SCALES` module-level (5-dim, OLD order).
- **obs `20+11K+F`** (K=1,F=1→32; K=3,F=1→54): 20 fixed = ρ/HOL/PRB-ratio/arrival/BLER(10) + severity_ref one-hot[10:15] + λ_local_C3_shared[15] + rrm_budget/n_bys/AoI mean/max(4); per-amb (×K, 11-dim, base=`20+11k`) = `{SINR_k, d_k, v_k, delay_norm_k, AoI_norm_k, severity_norm_k, λ_C1_k, λ_C2_k, λ_C4_k, λ_C5_k, active_mask_k}` (active_mask_k∈{0,1}=entered_k&~arrived_k, offset 10, 2026-06-23). `overlay_lambda_local(obs, lambda_local, K)` (utils/obs.py) scatter non-contiguous: per-amb λ vào offset 6-9 của mỗi khối 11-dim, λ_C3_shared vào index 15.
- **Worker action K-dependent** [UPDATE 2026-06-21]: K=1 = **1-dim** (no-op scalar — xe duy nhất nhận toàn bộ URLLC PRBs); K≥2 = **K-dim** (KHÔNG còn `(1+K)`-dim/β — đã gỡ) — `a[0:K]`→per-vehicle logits ℓ_k. Worker KHÔNG điều khiển inter-slice (Δr_min/Δr_max/r_ded = **legacy ĐÃ GỠ**; inter-slice do Manager `set_rrm_budget` duy nhất).
- **Worker actor zero-init output layer (ĐX1)** [UPDATE 2026-06-24, audit fix; mở rộng sang TD3/SAC cùng ngày]: default `nn.Linear` Kaiming-uniform init cho output layer khiến K logit lệch nhau ~1.05-1.25× ngẫu nhiên (seed-dependent) trước khi train; policy gradient (PPO clip / TD3 deterministic / SAC max-entropy — cả 3 đều tự khuếch đại lệch ban đầu qua gradient) biến lệch này thành PRB allocation bias dai dẳng, KHÔNG phụ thuộc severity (1 xe được ưu ái bất kể severity thực). **PPO**: `WorkerActor.mean_net` layer cuối (`agents/worker_agent.py::WorkerActor._zero_init_output_layer`, luôn áp dụng — class này CHỈ dùng cho Worker). **TD3**: `DeterministicActor.net` layer cuối (`agents/td3_agent.py`) — tham số `zero_init_output: bool=False` (default giữ nguyên hành vi cũ), Worker truyền `True` (`solvers/td3.py::TD3Solver.__init__`, `self.td3=TD3Agent(...)`), Manager (`TD3ManagerAgent`) KHÔNG truyền → giữ random init (action_dim=1, không có cross-dim bias để fix). **SAC**: CHỈ `GaussianTanhActor.mean_head` (KHÔNG `log_std_head` — giữ nguyên random, tương đương `WorkerActor.log_std` vốn đã là constant không bị lệch); tham số `zero_init_output` tương tự, Worker truyền `True` (`solvers/sac.py::SACSolver.__init__`, `self.sac=SACAgent(...)`), Manager (`SACManagerAgent`) KHÔNG truyền. Cả 3: `net(obs)≡0` (hoặc `mean_head(...)≡0`) tại init nên K logit khởi đầu bằng nhau; hidden layers giữ random init bình thường. Áp dụng MỘT LẦN lúc khởi tạo actor, KHÔNG re-apply khi số xe active thay đổi trong episode. Test: `tests/test_mutation_guards.py::test_m20/m21/m22` (xác nhận Worker zero, Manager vẫn random).
- **`_prb_split_intra_slice(prb_urllc)`** [UPDATE 2026-06-21 — pure-RL, gỡ tier-protection dưới; UPDATE 2026-06-24 — reserve-first order]: reserve `K_active·PRB_MIN_QOS` cho mọi xe ACTIVE TRƯỚC, rồi `softmax(ℓ) → w_k → extra_k=floor(w_k·(B_U−reserved))` + largest-remainder integer correction trên PHẦN CÒN LẠI (Σ PRB=B_U); `PRB_k=PRB_MIN_QOS+extra_k`. KHÔNG còn N_req formula, KHÔNG severity-tier 2-pha, KHÔNG β. Structural guarantee DUY NHẤT: `PRB_k≥PRB_MIN_QOS=1` cho xe ACTIVE — nay đúng **by construction** (trước đó chỉ đúng khi softmax không quá lệch). Feasibility precondition `B_U≥K_active·PRB_MIN_QOS` (raise `ValueError` nếu vi phạm — luôn thỏa dư margin lớn với bound hiện tại: `B_RRM_MIN=0.05→B_U≥13` PRB vs `K_active≤3→reserved≤3`). K=1: `PRB_0=B_U` luôn (numeric preservation, không đổi). *(`BETA_MIN/MAX`, `INTRA_SLICE_KAPPA`/`RHO_URGENCY_TIEBREAK` = legacy, giữ CHỈ để import/test compat.)* ~~Order cũ (2026-06-21→2026-06-24, ĐÃ GỠ): floor TOÀN BỘ B_U theo tỷ lệ (`PRB_k=floor(w_k·B_U)`) → ép tối thiểu từng xe → sửa overflow bằng rescale. Bug: rescale có thể đưa 1 xe về 0 khi 1 logit áp đảo cực độ (ví dụ raw `[10,−5,−5]`, `B_U=27` → `[26,1,0]`, vi phạm floor xe thứ 3).~~ ~~Mô tả gốc (2026-06-15, ĐÃ GỠ): severity-ordered N_req tier-protection 2 pha — Pha 1 cấp `N_req[k]=ceil(C_req[sev_k]/cap_per_prb(SINR_k))` theo tier giảm dần; thiếu trong tier → `score[k]=N_req·(1+β·urgency)·softmax(w)[k]`; Pha 2 surplus cùng score; `β∈[BETA_MIN,BETA_MAX]=[0.5,5]`.~~
- **C6 demoted hoàn toàn thành metric** [UPDATE 2026-06-21]: KHÔNG λ_C6/Lagrangian; severity-ordering⟹delay-ordering nay là **empirical/learned tendency** qua gradient (KHÔNG còn algebraic property của `_prb_split_intra_slice` — tier-protection đã gỡ), verify bằng `tests/test_env_severity_k.py`.
- VERIFIED: **237 test xanh** (226 sau permutation/dim fixes + 11 test mới `tests/test_env_severity_k.py`).

## Formulation audit fixes (✅ 2026-06-13) — 3 solver giải CÙNG bài toán
- **λ trong state (Markov)**: `overlay_lambda_local` → `utils/obs.py` = **1 nguồn dùng chung**; PPO (train.py) + TD3/SAC (smoke_train.py) đều inject λ_local vào obs (post-`severity_k` epic 2026-06-15: per-amb λ_C{1,2,4,5}_k ở offset 6-9 khối 10-dim + λ_C3_shared ở obs[15]; layout cũ block 5-dim ở obs[17:22]). Trước fix, TD3/SAC nhận obs thô → λ=0 vĩnh viễn (mục tiêu phi tĩnh, bất công). `EnvConfig.lambda_local` **xóa**; env chỉ chừa slot zeros.
- **Off-policy timing**: `s` mang λ trước dual-update (khớp `r_aug`); `next_obs` mang λ sau dual-update → tuple `(s,a,r,s')` nhất quán cho critic.
- **TD3/SAC severity-aware**: `use_phase=True` (tên flag **legacy** trong `_common.py` — nay gate **severity** one-hot, KHÔNG còn pha) + `maybe_mask` trả obs nguyên; nếu `use_phase=False` thì `mask_severity` che one-hot → mù severity, không thỏa QoS-theo-severity. Ngang hàng PPO.
- **C4 dual-scale**: `AOI_REF_S=0.1s` (scale chuẩn-hóa subgradient, **không** phải ngưỡng); tại thời điểm 2026-06-13, `CONSTRAINT_DUAL_SCALES[3]` 1.0→`AOI_REF_S` (C4 hết yếu ~10× so C1) — *constant module-level này đã bị XÓA 2026-06-24 (dead code từ sau ĐX2, xem dòng trên); cơ chế tương đương nay là `build_dual_scales()[2K:3K]` (slot C4, K-aware, per-instance, per-severity từ ĐX2)*. **C5**: m=1 (`P(AoI>AoI_max)≤ε_AoI^{sev_k}`; code `eps_aoi`/`d5_aoi_tail`).

## Code changes K≥2 (B5)
`oran_env.py`: obs per-amb block +severity_norm_k +λ_C{1,2,4,5}_k; action K=1 1-dim (no-op), K≥2 **K-dim** (pure-RL `a[0:K]`→per-vehicle logits, KHÔNG β slot — gỡ 2026-06-21); intra-slice PRB split = **pure-RL softmax** (`softmax(ℓ)→w_k→PRB_k`, largest-remainder; KHÔNG còn severity-ordered N_req tier-protection — legacy `κ/δ-softmax` + N_req tier ĐÃ GỠ); `severity_per_amb` exogenous per-ambulance (xem "Per-ambulance severity_k epic" trên). `lagrangian.py`: `(4K+1)`-dim λ/c_vec/d_phi; C6 = empirical metric, learned (no λ_C6, no structural guarantee). `cell_radius_m`: 200→300 (D25) → **1000** (W15-B2 macro UMa 2026-06-18).

## KHÔNG đổi (W11 backward-compat history)
`train.py` (K=1 default), `run_30runs.py`, `stats_analysis.py` — nhưng số RWP cũ KHÔNG tái dùng (sweep W18–W23 chạy lại trên SUMO mobility).

## Cross-reference
[03](03_architecture.md) · [05](05_agent_workflow.md) · [07](07_api_spec.md) · [weeks/](weeks/README.md) (build B0-B9).
