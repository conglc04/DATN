# 13 — Methodology Walkthrough (3-Phase Pipeline)

> Spec toán hợp nhất, mirror master plan `~/.claude/plans/s-p-x-p-l-i-plan-jaunty-toast.md` PHẦN 1 (thiết kế toán) + PHẦN 2 (I/O). Chi tiết atomic theo tuần ở [`weeks/`](weeks/README.md). Mọi đại lượng có nhãn ✅/🟡/🔴; nguồn vắng corpus flag rõ.
>
> **Pipeline**: Pha 1 Mô hình hệ thống → Pha 2 Bài toán tối ưu (CMDP) → Pha 3: 3 solver NGANG HÀNG (PPO/TD3/SAC) giải. RL/DL là **bước CUỐI** (solver), sau khi phát biểu bài toán hoàn tất; KHÔNG đề cao thuật toán nào.

---

## Pha 1 — Mô hình hệ thống

### 1.1 Bài toán General-K, 2 lớp
- **Lớp 1 INTER-SLICE** (K-agnostic): chia P_TOTAL giữa URLLC slice (xe) vs eMBB slice (bystander) → `r_min^URLLC, r_max^eMBB`.
- **Lớp 2 INTRA-URLLC** (K≥2): chia `B_U = r_min^URLLC·P_TOTAL` giữa K xe theo **pure-RL softmax split, reserve-first order** (audit 2026-06-21, gỡ N_req tier-protection; ĐỔI ORDER 2026-06-24): reserve `K_active·PRB_MIN_QOS` cho mọi xe ACTIVE TRƯỚC, rồi `softmax(ℓ_k) → w_k → extra_k=floor(w_k·(B_U−reserved))` + largest-remainder integer correction trên phần còn lại; `PRB_k=PRB_MIN_QOS+extra_k`. Structural guarantee duy nhất: `PRB_k≥PRB_MIN_QOS` (anti-starvation phẳng), giữ **by construction** (order cũ 2026-06-21→2026-06-24 — floor toàn bộ B_U rồi ép-tối-thiểu-rồi-rescale-overflow — có thể đưa 1 xe về 0 PRB khi 1 logit áp đảo cực độ). Severity-awareness HỌC HOÀN TOÀN qua obs (`severity_norm_k`, λ_C1..C5 per-amb) + `r_aug` gradient — KHÔNG còn N_req formula, KHÔNG β, KHÔNG tier-protection structural.
- K=1 (E1) = chỉ Lớp 1; K=3 (E2) = Lớp 1+2 đầy đủ. Xe = **URLLC-only** (1 luồng `ambulance_status` tổng hợp, F=1, 2026-06-14 consolidation — gộp vital + DENM/CAM cũ); eMBB = nền bystander.

### 1.2 State / Action (PHẦN 2)
- **Observation**: K=1 = **32-dim** (per-ambulance `severity_k` epic 2026-06-15; **+active_mask_k** 2026-06-23: `obs_dim = 20 + 11K + F`, K=1,F=1→32; K=3,F=1→54). Fixed 20 = ρ/HOL/PRB-ratio/arrival/BLER(10) + **severity_ref one-hot [10:15]** + **λ_local_C3_shared [15]** + rrm_budget/n_bys/AoI mean/max(4). per-amb (×K, 11-dim block) = {SINR_k, d_k, v_k, delay_norm_k, AoI_norm_k, severity_norm_k, λ_C1_k, λ_C2_k, λ_C4_k, λ_C5_k, **active_mask_k**} — mỗi xe có URLLC queue + AoI tracker + **severity_k riêng (sampled độc lập, cố định/episode)** + λ_local per-constraint riêng + **active_mask_k∈{0,1}=entered_k&~arrived_k** (cờ active tường minh, phân biệt xe inactive với xe active queue-rỗng), nên policy phân biệt được xe nào gần vi phạm QoS hơn ngay cả khi severity khác nhau. **severity_ref := max(severity_per_amb)** lái các đại lượng SHARED (one-hot). **Reward KHÔNG còn α_e** (bỏ 2026-06-23). C3 R_min **KHÔNG** còn phụ thuộc severity — eMBB floor cố định 10 Mbps (Gate 7, 2026-06-20). Field-set = nguồn-sự-thật; binding `assert observation_space.shape` ([W18](weeks/W18_pha3_algorithm_code.md)).
- **Worker Action**: K=1 = **1-dim** (no-op scalar — xe duy nhất nhận toàn bộ URLLC PRBs); K≥2 = **K-dim** (pure-RL, gỡ β 2026-06-21 — KHÔNG còn `(1+K)`-dim): `a[0:K]`→**per-vehicle priority logits ℓ_k** (raw ∈[−3,+3], decode qua `softmax(ℓ)` trong intra-slice split). **Worker KHÔNG điều khiển inter-slice** — `r_min/r_max/r_ded` trong obs[4:7] là Manager-owned (read-only). ⚠️ RL discount `γ_H/γ_L` (#3) ≠ `β_ema` (λ-persistence EMA, §2.3.5) — `β` priority-temperature đã gỡ, không còn nguy cơ nhầm 3 ký hiệu.
- **Manager Action**: 1-dim → `sigmoid` → `b_rrm ∈ [B_RRM_MIN, B_RRM_MAX]` = tỷ lệ PRB cho URLLC slice (inter-slice budget).

### 1.3 Định luật vật lý / miền (nhóm M, [W12](weeks/W12_pha1_radio_channel_capacity.md)–[W16](weeks/W16_pha1_severity_triage.md))
- **Channel** **UMa single-cell 1km @ Bạch Mai** (3GPP TR 38.901 `pl_uma`) **+ interference margin −86 dBm/PRB** (mô phỏng interference-limited macro qua noise-rise floor, KHÔNG sim gNB láng giềng → giữ single-cell) — ✅[3GPP TR 38.901 §7.4]; P_tx^UE=23dBm uplink ✅[TS 38.101-1]. Điểm làm việc: SINR ≈ 2.7 dB @cell-edge 1km (`macro_mission_config`, W15-B2). *(Scenario 300m micro/UMi = legacy default, dùng cho unit-test; sweep W18–W23 = UMa 1km.)*
- **Capacity** `C_k=η·PRB_k·B_PRB·log₂(1+SINR_k)`, SINR linear, η=0.75 🟡 — ✅[Shannon; `Hyoungju Ji 2017`]. Sweep macro = **interference-limited** (I≠0 qua margin −86 dBm/PRB); legacy micro single-cell = SNR-only (I≈0).
- **Delay** `D_e2e = D_DET + d_tx + d_queue + D_FH + D_BH` (KHÔNG D_MEC); `d_queue` = **M/G/1 PK** ✅[`Kleinrock 9780470316887.pdf` §5.6] (upper-bound trung bình; burst bắt bằng Monte Carlo).
- **AoI** `Δ(t)=t−U(t)`, LCFS+drop — ✅[`Qi 2024`; `Chen`; `Mlika 2022`] *(KHÔNG Kaul, vắng corpus)*; AoI thuần timestamp (no fake vitals).
- **Severity** 5 mức exogenous (Non-urgent…Immediate) theo **ATS — Australasian Triage Scale**, **`severity_per_amb ∈ {1..5}^K` sampled độc lập per ambulance, cố định/episode, random giữa các episode** — ✅[`ATS — Australasian College for Emergency Medicine (ACEM)`]; `severity_ref := max(severity_per_amb)` là **bộ chọn QoS-tier cho đại lượng SHARED** (severity one-hot; **reward KHÔNG còn α_e** — bỏ 2026-06-23; **C3 R_min nay cố định 10 Mbps, KHÔNG theo severity** — Gate 7) + `severity_per_amb` lái priority ordering **HỌC HOÀN TOÀN** qua per-vehicle softmax logits + λ-penalty gradient (pure-RL, K≥2 — KHÔNG còn β/Π_feasible structural, gỡ 2026-06-21) = design principle declared 🔴.
- **Traffic** Poisson + payload/rate, 1 luồng tổng hợp `ambulance_status` (F=1, 2026-06-14 consolidation; periodic status bundle ~500–1500B @ ~10–20Hz 🔴 declared, conceptual — Poisson queue params M8.1b giữ nguyên) — ✅[`Alsenwi 2022`; `Sohaib 2024`]; R_REF=100Mbps 🟡[`Weijian Zhou`].

### 1.4 Thang thời gian
MAC TTI=0.5ms [TS 38.211 μ=1] · Worker(xApp)=10ms · Manager=100ms; `W=WORKER_STEPS_PER_MANAGER=10` (`config.py`). Two-timescale `γ_H≈γ_L^W`.

---

## Pha 2 — Bài toán tối ưu (CMDP) — [W17](weeks/W17_pha2_cmdp_formulation.md)

### 2.1 Objective
`max E[Σ_t r_t]`, `r_t = mean_{tick∈step} log(1 + R_eMBB,tick/R_REF)` — ✅[`Alsenwi 2022` Eq.13; `Sohaib 2024` Eq.9]. **Single-term, pure eMBB log-utility — KHÔNG α_e** (bỏ 2026-06-23). Severity differentiation **hoàn toàn qua constraints C1–C5 + λ**, KHÔNG qua trọng số reward (bỏ α_e xoá double-count: constraint đã ép b_rrm cao ở severity cao; α_e=0.05 ở sev5 còn triệt tín hiệu Manager gradient). URLLC KHÔNG vào reward (enforced qua Lagrangian → tránh double-counting). **`r_t` = MEAN over MAC ticks** (KHÔNG sum): c_vec cũng là per-tick MEAN, nên reward phải cùng thang thời gian để `r − Σλⱼ·gⱼ` cân bằng. Mismatch SUM-vs-MEAN (×20) khiến reward gradient nuốt constraint penalty → Manager bỏ đói URLLC (audit 2026-06-23).

### 2.2 Constraints (CMDP `(S,A,P,r,{g_c},γ)` — ✅[`Yongshuai Liu 2020`; `Wen Wu 2020`; `Qiang Liu 2021`])
| | Ràng buộc | Nguồn |
|---|---|---|
| C1 | `E[D_e2e^k] ≤ D_max^{sev_k}` | ✅[TS 22.261 §7.2/Annex A] |
| C2 | `P(D_e2e^k > D_max^{sev_k}) ≤ ε^{sev_k}` | ✅[TS 22.261 §7.2] |
| C3 | `R_eMBB ≥ R_min` | ✅[Alsenwi; Sohaib] |
| C4 | `E[AoI_k] ≤ AoI_max^{sev_k}` | 🔴-declared (AoI_max theo severity) |
| C5 | `P(AoI_k > AoI_max) ≤ ε_AoI^{sev_k}` (m=1: cùng ngưỡng `AoI_max` như C4, đối xứng cặp C1/C2 dùng `D_max`; `ε_AoI` đối xứng `ε` của C2; code `eps_aoi`/`d5_aoi_tail`) | 🔴 (form grounded; demote optional) |
| C6 | `severity_per_amb[i]>severity_per_amb[j] ⟹ E[D^i] ≤ E[D^j]` | **DEMOTE→metric** (#13); KHÔNG còn structural guarantee (§2.4 pure-RL, gỡ Π_feasible 2026-06-21) — empirical/learned tendency, verify test (no Lagrangian λ_C6) |

> **Độ phân giải ngưỡng theo severity** (audit 2026-06-20): mọi cột ngưỡng đều **đơn điệu chặt hơn** (non-increasing) theo severity → 5 mức tạo phân cấp QoS nhất quán, KHÔNG mâu thuẫn. 5 mức được phân biệt **đầy đủ** qua `D_max` (20/10/5/2/1 ms — 5 giá trị riêng); reward KHÔNG còn α_e (severity vào hệ qua constraint thresholds + λ, không qua trọng số reward — bỏ 2026-06-23). Các cột tin cậy/tươi-mới thô hơn **có chủ đích**: **ε (C2)** = 3 lớp "nines" chuẩn 3GPP {1e‑3,1e‑4,1e‑5} (99.9/99.99/99.999%) — đặt 5 giá trị riêng sẽ *bịa* lớp phi-chuẩn; **AoI_max (C4)** = 4 mức, bão hoà ở sàn tươi-mới 0.1 s cho sev4=sev5; **ε_AoI (C5)** = 2 mức {non-urgent 1e‑2, urgent 1e‑3} — phân biệt yếu nhất, nhất quán với C5 🔴-declared/demote-optional. C3 `R_min` nay **cố định 10 Mbps** (Gate 7), không theo severity.

### 2.3 Lagrangian
`L = J − Σ_c λ_c·g_c`; `λ_c ← clip(λ_c + α_λ·ĝ_c, 0, Λ_max=10)`, α_λ=2e-4 (`ALPHA_LAMBDA_DUAL` SSOT) — ✅[`Spoor 2025`; `Ding 2023`] *(KHÔNG Boyd/Tessler)*. Subgradient chuẩn hóa `ĝ_c = (c_c−d_c)/scale_c`. **λ PERSIST qua episode** (audit 2026-06-23): `λ_warm[sev]` là biến đối ngẫu bền per-severity, lưu FULL ở `on_episode_end` (β_ema=1.0), warm-start episode kế từ giá trị tích lũy → λ leo đơn điệu tới cân bằng CMDP (trước: β_ema=0.05 pha loãng 20× → λ_C2 ghim ở 2.2 < λ*≈4.0 → starvation). Reset chỉ khi `LambdaState` mới (run/seed mới). `c_vec/d_phi/λ` đều **`(4K+1,)`**, layout `[C1_0..C1_{K-1}, C2_0..C2_{K-1}, C4_0..C4_{K-1}, C5_0..C5_{K-1}, C3_shared]` (`build_dual_scales(K)`, `build_lambda_warm_vector`, `build_d_phi_vector` — `utils/config.py`). K=1 là permutation `[0,1,3,4,2]` của thứ tự cũ `[C1,C2,C3,C4,C5]`, numerically identical. Disclaimer **weak duality** (no zero-duality-gap, deep-NN). **Lưu ý r_aug vs dual ascent** (fix 2026-06-22 bonus-masking audit): dual ascent ở đây dùng `ĝ_c` **signed** (đúng lý thuyết, để λ relax được khi slack); nhưng reward per-step thực tế đưa vào policy gradient là `r_aug = r − Σ_c λ_c·max(0,ĝ_c)` (**hinge**) — slack constraint đóng góp đúng 0, KHÔNG tạo bonus che lấp violation của constraint khác (xem `lagrangian.py::augmented_reward`).
- **State observability (Markov)**: λ_local được inject qua `overlay_lambda_local(obs, lambda_local, K)` (utils/obs.py — **1 nguồn dùng chung** cho PPO + TD3 + SAC): per-amb λ_C{1,2,4,5}_k vào khối 11-dim per-amb (offset 6-9; active_mask_k ở offset 10), λ_C3_shared vào `LAMBDA_C3_SHARED_OBS_INDEX=15`. Cả 3 solver thấy λ + severity_ref one-hot [10:15] như nhau → so sánh công bằng. Off-policy (TD3/SAC): `s` mang λ trước dual-update (đúng λ tạo `r_aug`), `next_obs` mang λ sau dual-update (λ điều kiện hành động kế).

### 2.4 Intra-slice — pure-RL (audit 2026-06-21, gỡ Π_feasible severity-tier protection; ĐỔI ORDER 2026-06-24)
**Pure-RL softmax split, reserve-first** (`_prb_split_intra_slice`, oran_env.py — xem docstring "Pure-RL intra-slice PRB allocation — fully learned, no rules"; thay severity-tier N_req protection cũ VÀ split `κ/δ-softmax` cũ hơn nữa — `BETA_MIN/MAX`, `INTRA_SLICE_KAPPA`/`RHO_URGENCY_TIEBREAK` nay đều = legacy, giữ CHỈ để import/test compat):
- **Cơ chế**: reserve `K_active·PRB_MIN_QOS` cho mọi xe ACTIVE TRƯỚC (`reserved`); `a[0:K]` = per-vehicle priority logits `ℓ_k` (raw ∈[−3,3], K≥2) → `w_k=softmax(ℓ)` → `extra_k=floor(w_k·(B_U−reserved))` + largest-remainder integer correction trên phần còn lại (Σ PRB = B_U chính xác); `PRB_k=PRB_MIN_QOS+extra_k`. Feasibility precondition `B_U≥K_active·PRB_MIN_QOS` (raise `ValueError` nếu vi phạm — luôn thỏa dư margin với bound hiện tại). KHÔNG N_req formula, KHÔNG severity-tier-descending phase, KHÔNG β urgency multiplier, KHÔNG λ trong allocation.
- **Order cũ (2026-06-21→2026-06-24, ĐÃ GỠ)**: `PRB_k=floor(w_k·B_U)` trên toàn bộ `B_U` → ép tối thiểu từng xe → sửa overflow bằng rescale `B_U·allocs//Σallocs`. Bug: rescale có thể đưa 1 xe ACTIVE về 0 PRB khi 1 logit áp đảo cực độ (vd raw `[10,−5,−5]`, B_U=27 → `[26,1,0]`), vi phạm floor I5.
- **Severity-awareness HỌC HOÀN TOÀN qua gradient** (không qua rule): policy quan sát `severity_norm_k` + `λ_C1/C2/C4/C5_k` per-amb trong obs; `r_aug = r − Σλ_j·max(0,g_j)` phạt vi phạm QoS → gradient dạy policy gán `ℓ_k` cao hơn cho xe cần nhiều PRB hơn.
- **Structural guarantee DUY NHẤT** (verify `tests/test_env_severity_k.py`, audit gate `audit/closure_checks.py::g7_nreq`): (1) feasibility `ΣPRB=B_U` (budget conservation, by construction); (2) anti-starvation floor PHẲNG `PRB_k≥PRB_MIN_QOS=1` cho mọi xe ACTIVE (KHÔNG severity-tiered), nay đúng **vô điều kiện** nhờ reserve-first order (trước 2026-06-24 chỉ đúng khi softmax không quá lệch). KHÔNG còn severity tier-ordering structural — C6 (§2.2 bảng) demote thành **empirical/learned tendency**, KHÔNG phải algebraic property. K=1: 1 xe active, `PRB_0=B_U` luôn (numeric preservation, không đổi qua refactor này).
- **Worker actor zero-init, cả 3 solver** (audit 2026-06-24, ĐX1; mở rộng TD3/SAC cùng ngày): `nn.Linear` Kaiming-uniform init mặc định cho output layer tạo lệch ngẫu nhiên ~1.05-1.25× giữa K logit trước khi train; policy gradient (PPO/TD3/SAC) khuếch đại lệch này thành PRB bias dai dẳng KHÔNG phụ thuộc severity — vi phạm yêu cầu "3 solver giải CÙNG bài toán" công bằng nếu chỉ fix 1 solver. Fix, zero-init weight+bias CHỈ layer cuối, hidden layers giữ random init: **PPO** `WorkerActor.mean_net[-1]` (luôn áp dụng, class chỉ dùng cho Worker); **TD3** `DeterministicActor.net[-1]` qua flag `zero_init_output` (Worker truyền `True` ở `solvers/td3.py`, Manager KHÔNG truyền); **SAC** CHỈ `GaussianTanhActor.mean_head` qua flag tương tự (KHÔNG `log_std_head`, tương đương `WorkerActor.log_std` vốn đã là constant), Worker truyền `True` ở `solvers/sac.py`, Manager KHÔNG truyền. Manager (`action_dim=1`, không có cross-dim bias) giữ random init ở cả 3 solver. Áp dụng 1 lần lúc khởi tạo actor, KHÔNG re-apply khi active-count đổi trong episode. Test: `tests/test_mutation_guards.py::test_m20/m21/m22`.

---

## Pha 3 — Giải bằng 3 solver ngang hàng (PPO/TD3/SAC) — [W18](weeks/W18_pha3_algorithm_code.md)

### 3.1 Khung chung + 3 solver ngang hàng
- **Khung CHUNG** (KHÔNG gắn 1 thuật toán): two-timescale HRL Manager/Worker ✅[`Akyıldız 2024`] + CMDP-Lagrangian (§2.3) + intra-slice pure-RL softmax split (§2.4/§3.2) + GAE ✅[`Foundations_of_Deep_RL.pdf`].
- **3 RL core NGANG HÀNG** giải cùng bài toán (so sánh công bằng): **PPO** clipped ✅[`1707.06347v2.pdf`] (on-policy); **TD3** ✅[`fujimoto18a.pdf`] (off-policy deterministic); **SAC** ✅[`1812.05905v2.pdf` Haarnoja 2018] (off-policy max-entropy). *(B3-RCPO cũ loại khỏi Table I.)*

### 3.2 Kiến trúc & safety
- Manager + Worker (xApp) actor-critic; action continuous → RRMPolicyRatio.
- **HRL thật — Manager có tác động NHÂN QUẢ lên env (W18+)**. Mỗi solver ghép Manager CÙNG thuật toán với Worker (parity tuyệt đối): PPO→PPO-Manager (on-policy), TD3→TD3-Manager (off-policy det.), SAC→SAC-Manager (off-policy stoch.). Vòng lặp Algorithm 1 mỗi Manager window (W=10 Worker steps):
  1. `s_H,t = build_manager_state(obs, λ_global, g_hat) = [x_t, λ_t, ĝ_{t-1}]` — `8 + 2·(4K+1)`-dim (audit 2026-06-23, was 6+(4K+1)). `g_hat = ĝ_{t-1} = LambdaState.get_deviation_hat()`, đọc TRƯỚC khi Manager chọn `b_rrm_t` ở bước 2 dưới ⟹ là residual của window TRƯỚC (chạy dưới `b_rrm_{t-1}`), KHÔNG phải residual của window sẽ chạy dưới `b_rrm_t` (mới chọn) — tránh leak thông tin tương lai. Cache bởi `on_manager_step_end()` (lời gọi cuối vòng lặp window t-1) trước khi `s_H,t` được build. Exposed alongside λ_global (cũng post-window t-1) so the Manager observes the per-window residual, not just the long-run dual price. **Chuẩn hóa scale (audit 2026-06-24)**: λ-slot chia `LAMBDA_MAX` → [0,1] (sửa lệch scale input critic: λ∈[0,10] vs fixed-block∈[0,1]; dual ascent/penalty dùng `LambdaState.lambda_global` RAW, KHÔNG phải bản obs này); **g_hat giữ RAW/signed** (đối xứng quanh 0 — dấu/điểm cắt-0 là tín hiệu load-bearing cho critic, không squash bằng ceiling một phía). **Lưu ý (same-source proxy, KHÔNG đẳng thức tuyệt đối)**: `g_hat` là **SIGNED window MEAN** của `(c-d)/scale` (qua `accumulate()`); `r_aug = r − Σλ·max(0,dev)` lại **hinge** đúng `dev` đó **per-tick** (từng bước Worker) rồi mới SMDP-discount-sum vào critic. Vì `max(0,·)` convex, `mean(max(0,dev)) ≥ max(0,mean(dev))` (Jensen) — `g_hat` cho biết constraint nào đang chịu áp lực và chiều nào (cùng nguồn `c_vec/d_phi`), KHÔNG phải con số chính xác đã bị trừ khỏi reward.
  2. `a_H = manager.act(s_H)`; `b_rrm = decode_manager_action(a_H) = B_RRM_MIN + (B_RRM_MAX−B_RRM_MIN)·σ(a_H[0])`, `[B_RRM_MIN,B_RRM_MAX]=[0.05,0.85]`.
  3. **`env.set_rrm_budget(b_rrm)`** — re-anchor `r_min_urllc` (two-tier clip), tác động trực tiếp lên PRB split URLLC/eMBB ⟹ obs[16]=anchor. Worker KHÔNG drift inter-slice — chỉ điều khiển intra-URLLC priority (per-vehicle logits, pure-RL — KHÔNG β).
  4. Worker chạy W bước; Manager return SMDP-discounted `r_H = Σ_{i=0}^{W−1} γ_L^i · r_aug_i` (KHÔNG undiscounted), `γ_H = γ_L^W ≈ 0.904 = GAMMA_MANAGER` cho return/GAE Manager.
  5. PPO-Manager: rollout buffer + update **mỗi rollout** (= MANAGER_STEPS_PER_EPISODE windows = 100 Worker steps; GAE bootstrap V(s) **trừ khi terminated** — timeout/truncation vẫn bootstrap) rồi tiếp tục cùng env tới khi mission done. TD3/SAC-Manager: replay buffer nhỏ (5k), store `(s_H, a_H, r_H, s_H', done)` + update mỗi boundary khi `buffer ≥ warmup`. **Episode = trọn mission** (all-arrived/400s) cho cả 3 solver. Checkpoint sidecar `*_manager.pt`.
  - Three-rate hierarchy khóa: `α_πH=3e-5 < α_λ=2e-4 < α_πL=3e-4` (two-timescale ordering — Akyıldız 2024; Borkar 2008 vắng corpus. Định lý chỉ biện minh THỨ TỰ/separation, KHÔNG quy định giá trị — con số là heuristic + tinh chỉnh).
- **Intra-slice split = pure-RL softmax** (audit 2026-06-21 — KHÔNG còn `Π_feasible` Duchi-simplex/isotonic projection: đó chỉ là kế hoạch ban đầu W18/B5, outcome thực tế đi thẳng severity-tier N_req 2-pha (2026-06-15) → pure-softmax (2026-06-21), không qua Duchi/PAV). Structural guarantee duy nhất: budget conservation (`ΣPRB=B_U`) + anti-starvation floor `PRB_MIN_QOS` phẳng — **no learnable params** ở 2 ràng buộc này (gỡ β_qp/NSF-distillation); KHÔNG claim novel (Kim 2026 = neural QP, chỉ cite).
- δ-preemption / offload = KHÔNG dùng (MEC gỡ).

### 3.3 Verification (sweep W18–W23 → Table I/II; E3/E4 = future work D26)
Sweep 3 solver × K∈{1,3} → **Table I** (reward + C1–C5 + λ-saturation) + **Table II** (K=3 severity/intra-slice + ablation 2×2). Holm-Bonferroni + Hedges' g + bootstrap CI; **ε rule-of-three** (KHÔNG claim 1e-5 empirical); **KHÔNG Jain toàn cục** (mâu thuẫn priority) → ordering-compliance + no-starvation + within-tier fairness; **λ-saturation logging**. E3 (AoI LCFS/FCFS) và E4 (stress/robustness) → future work — C3 (AoI hard constraints) vẫn active trong CMDP của sweep. Chi tiết [`06_validation.md`](06_validation.md).

---

## Bibliography (corpus-only; vắng-corpus đánh dấu)
- **RL/CRL**: CMDP [`Yongshuai Liu 2020`; `Wen Wu 2020`; `Qiang Liu 2021`] *(KHÔNG Altman)*; Lagrangian [`Spoor 2025`; `Ding 2023`] *(KHÔNG Boyd/Tessler)*; HRL [`Akyıldız 2024`] *(KHÔNG Borkar/FeUdal)*.
- **DRL**: PPO [`1707.06347v2`]; TD3 [`fujimoto18a`]; SAC [`1812.05905v2` Haarnoja 2018]; GAE [`Foundations_of_Deep_RL`] *(Schulman 2016 vắng corpus)*.
- **Queue/AoI**: M/G/1 [`Kleinrock 9780470316887`]; AoI [`Qi 2024`; `Chen`; `Mlika 2022`] *(KHÔNG Kaul)*.
- **3GPP**: TS 22.261 (QoS), TR 38.901 (channel), TS 38.101-1/211/214 (radio), TS 23.501 §5.7 (5QI priority), TS 28.541 (RRMPolicyRatio).
- **Medical (triage)**: ATS — Australasian Triage Scale [`ATS — Australasian College for Emergency Medicine (ACEM)`].
- **O-RAN/slicing**: [`O-RAN.pdf`]; [`Alsenwi 2022`; `Sohaib 2024`; `Filali 2022`; `Weijian Zhou`].

## Cross-reference
`weeks/` (atomic per-week) · `01_overview.md` · `02_requirements.md` · `05_agent_workflow.md` · `06_validation.md` · `REFERENCE_MAP.md`.
