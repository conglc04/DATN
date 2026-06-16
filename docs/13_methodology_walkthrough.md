# 13 — Methodology Walkthrough (3-Phase Pipeline)

> Spec toán hợp nhất, mirror master plan `~/.claude/plans/s-p-x-p-l-i-plan-jaunty-toast.md` PHẦN 1 (thiết kế toán) + PHẦN 2 (I/O). Chi tiết atomic theo tuần ở [`weeks/`](weeks/README.md). Mọi đại lượng có nhãn ✅/🟡/🔴; nguồn vắng corpus flag rõ.
>
> **Pipeline**: Pha 1 Mô hình hệ thống → Pha 2 Bài toán tối ưu (CMDP) → Pha 3: 3 solver NGANG HÀNG (PPO/TD3/SAC) giải. RL/DL là **bước CUỐI** (solver), sau khi phát biểu bài toán hoàn tất; KHÔNG đề cao thuật toán nào.

---

## Pha 1 — Mô hình hệ thống

### 1.1 Bài toán General-K, 2 lớp
- **Lớp 1 INTER-SLICE** (K-agnostic): chia P_TOTAL giữa URLLC slice (xe) vs eMBB slice (bystander) → `r_min^URLLC, r_max^eMBB`.
- **Lớp 2 INTRA-URLLC** (K≥2): chia `B_U = r_min^URLLC·P_TOTAL` giữa K xe theo severity → `w_k = softmax(β·severity_k)`.
- K=1 (E1) = chỉ Lớp 1; K=3 (E2) = Lớp 1+2 đầy đủ. Xe = **URLLC-only** (1 luồng `ambulance_status` tổng hợp, F=1, 2026-06-14 consolidation — gộp vital + DENM/CAM cũ); eMBB = nền bystander.

### 1.2 State / Action (PHẦN 2)
- **Observation**: K=1 = **31-dim** (per-ambulance `severity_k` epic 2026-06-15: `obs_dim = 20 + 10K + F`, K=1,F=1→31; K=3,F=1→51). Fixed 20 = ρ/HOL/PRB-ratio/arrival/BLER(10) + **severity_ref one-hot [10:15]** + **λ_local_C3_shared [15]** + rrm_budget/n_bys/AoI mean/max(4). per-amb (×K, 10-dim block) = {SINR_k, d_k, v_k, delay_norm_k, AoI_norm_k, severity_norm_k, λ_C1_k, λ_C2_k, λ_C4_k, λ_C5_k} — mỗi xe có URLLC queue + AoI tracker + **severity_k riêng (sampled độc lập, cố định/episode)** + λ_local per-constraint riêng, nên policy phân biệt được xe nào gần vi phạm QoS hơn ngay cả khi severity khác nhau. **severity_ref := max(severity_per_amb)** lái mọi đại lượng SHARED (one-hot, α_eMBB, C3 R_min). Field-set = nguồn-sự-thật; binding `assert observation_space.shape` ([W18](weeks/W18_pha3_algorithm_code.md)).
- **Action**: K=1 = 6-dim {Δr_min, Δr_max, r_ded, w_C1, w_C2, w_C3}; K≥2 = 7-dim (+**β** = priority temperature, `β=BETA_MIN+(BETA_MAX−BETA_MIN)·sigmoid(a_β)`, `BETA_MIN=0.0, BETA_MAX=5.0`). K=1: β cố định `BETA_MIN` (không ảnh hưởng số trị — Π_feasible K=1 luôn `PRB_0=B_U`). ⚠️ β ≠ RL discount `γ_H/γ_L` (#3). ρ/δ = hyperparam (KHÔNG action). Mapping → **RRMPolicyRatio** [3GPP TS 28.541] (Min/Max/Dedicated).

### 1.3 Định luật vật lý / miền (nhóm M, [W12](weeks/W12_pha1_radio_channel_capacity.md)–[W16](weeks/W16_pha1_severity_triage.md))
- **Channel** UMi single-cell 300m @ Bạch Mai — ✅[3GPP TR 38.901 §7.4]; P_tx^UE=23dBm uplink ✅[TS 38.101-1].
- **Capacity** `C_k=η·PRB_k·B_PRB·log₂(1+SINR_k)`, SINR linear, single-cell SNR (I≈0), η=0.75 🟡 — ✅[Shannon; `Hyoungju Ji 2017`].
- **Delay** `D_e2e = D_DET + d_tx + d_queue + D_FH + D_BH` (KHÔNG D_MEC); `d_queue` = **M/G/1 PK** ✅[`Kleinrock 9780470316887.pdf` §5.6] (upper-bound trung bình; burst bắt bằng Monte Carlo).
- **AoI** `Δ(t)=t−U(t)`, LCFS+drop — ✅[`Qi 2024`; `Chen`; `Mlika 2022`] *(KHÔNG Kaul, vắng corpus)*; AoI thuần timestamp (no fake vitals).
- **Severity** 5 mức exogenous (Non-urgent…Immediate) theo **ATS — Australasian Triage Scale**, **`severity_per_amb ∈ {1..5}^K` sampled độc lập per ambulance, cố định/episode, random giữa các episode** — ✅[`ATS — Australasian College for Emergency Medicine (ACEM)`]; `severity_ref := max(severity_per_amb)` là **bộ chọn QoS-tier cho đại lượng SHARED** (thay 5-pha cũ: α_eMBB, C3 R_min, severity one-hot) + `severity_per_amb` lái priority ordering qua β/Π_feasible (K≥2) = design principle declared 🔴.
- **Traffic** Poisson + payload/rate, 1 luồng tổng hợp `ambulance_status` (F=1, 2026-06-14 consolidation; periodic status bundle ~500–1500B @ ~10–20Hz 🔴 declared, conceptual — Poisson queue params M8.1b giữ nguyên) — ✅[`Alsenwi 2022`; `Sohaib 2024`]; R_REF=100Mbps 🟡[`Weijian Zhou`].

### 1.4 Thang thời gian
MAC TTI=0.5ms [TS 38.211 μ=1] · Worker(xApp)=10ms · Manager(rApp)=1s; `W=WORKER_STEPS_PER_MANAGER=10` (`config.py`). Two-timescale `γ_H≈γ_L^W`.

---

## Pha 2 — Bài toán tối ưu (CMDP) — [W17](weeks/W17_pha2_cmdp_formulation.md)

### 2.1 Objective
`max E[Σ_t α_e(sev)·log(1 + R_eMBB/R_REF)]` — ✅[`Alsenwi 2022` Eq.13; `Sohaib 2024` Eq.9]. **Single-term**; `α_e(sev)` = trọng số eMBB phụ thuộc severity (`get_severity_alpha`, code `oran_env.py`; đơn điệu Non-urgent 0.70→Immediate 0.05). URLLC KHÔNG vào reward (enforced qua Lagrangian → tránh double-counting).

### 2.2 Constraints (CMDP `(S,A,P,r,{g_c},γ)` — ✅[`Yongshuai Liu 2020`; `Wen Wu 2020`; `Qiang Liu 2021`])
| | Ràng buộc | Nguồn |
|---|---|---|
| C1 | `E[D_e2e^k] ≤ D_max^{sev_k}` | ✅[TS 22.261 §7.2/Annex A] |
| C2 | `P(D_e2e^k > D_max^{sev_k}) ≤ ε^{sev_k}` | ✅[TS 22.261 §7.2] |
| C3 | `R_eMBB ≥ R_min` | ✅[Alsenwi; Sohaib] |
| C4 | `E[AoI_k] ≤ AoI_max^{sev_k}` | 🔴-declared (AoI_max theo severity) |
| C5 | `P(AoI_k > AoI_max) ≤ ε_AoI^{sev_k}` (m=1: cùng ngưỡng `AoI_max` như C4, đối xứng cặp C1/C2 dùng `D_max`; `ε_AoI` đối xứng `ε` của C2; code `eps_aoi`/`d5_aoi_tail`) | 🔴 (form grounded; demote optional) |
| C6 | `severity_per_amb[i]>severity_per_amb[j] ⟹ E[D^i] ≤ E[D^j]` | **DEMOTE→metric** (#13); structural property của §2.4 Π_feasible weight-ordering, verify test (no Lagrangian λ_C6) |

### 2.3 Lagrangian
`L = J − Σ_c λ_c·g_c`; `λ_c ← clip(λ_c + α_λ·ĝ_c, 0, Λ_max=10)`, α_λ=1e-4 — ✅[`Spoor 2025`; `Ding 2023`] *(KHÔNG Boyd/Tessler)*. Subgradient chuẩn hóa `ĝ_c = (c_c−d_c)/scale_c`. `c_vec/d_phi/λ` đều **`(4K+1,)`**, layout `[C1_0..C1_{K-1}, C2_0..C2_{K-1}, C4_0..C4_{K-1}, C5_0..C5_{K-1}, C3_shared]` (`build_dual_scales(K)`, `build_lambda_warm_vector`, `build_d_phi_vector` — `utils/config.py`). K=1 là permutation `[0,1,3,4,2]` của thứ tự cũ `[C1,C2,C3,C4,C5]`, numerically identical. Disclaimer **weak duality** (no zero-duality-gap, deep-NN).
- **State observability (Markov)**: λ_local được inject qua `overlay_lambda_local(obs, lambda_local, K)` (utils/obs.py — **1 nguồn dùng chung** cho PPO + TD3 + SAC): per-amb λ_C{1,2,4,5}_k vào khối 10-dim per-amb (offset 6-9), λ_C3_shared vào `LAMBDA_C3_SHARED_OBS_INDEX=15`. Cả 3 solver thấy λ + severity_ref one-hot [10:15] như nhau → so sánh công bằng. Off-policy (TD3/SAC): `s` mang λ trước dual-update (đúng λ tạo `r_aug`), `next_obs` mang λ sau dual-update (λ điều kiện hành động kế).

### 2.4 Intra-slice (cấu trúc, LUÔN khả thi)
`b = max(floor(κ·B_U/K), PRB_min^QoS)` (`κ=INTRA_SLICE_KAPPA=0.5`, `PRB_MIN_QOS=1`; fallback `b=B_U//K` nếu `K·b>B_U`), `S=B_U−K·b`, `w_k=softmax(β·severity_k+δ·ũ_k)`, `PRB_k=b+S·w_k`. `β∈[BETA_MIN,BETA_MAX]=[0,5]` từ action a[6] (K≥2); `severity_k`=`severity_per_amb[k]` (per-ambulance, độc lập). **2 structural guarantee** (đại số, verify test `tests/test_env_severity_k.py`): feasibility `ΣPRB=B_U` + no-starvation `PRB_k≥b` + weight-ordering đơn điệu theo severity (δ=ρ·β, `ρ=RHO_URGENCY_TIEBREAK=0.15`). Tiebreaker `ũ_k=λ_C1_k/max(λ_C1)` (normalized URLLC-delay urgency). K=1: `softmax([x])=[1.0]` ∀x ⟹ `PRB_0=B_U` luôn (numeric preservation).

---

## Pha 3 — Giải bằng 3 solver ngang hàng (PPO/TD3/SAC) — [W18](weeks/W18_pha3_algorithm_code.md)

### 3.1 Khung chung + 3 solver ngang hàng
- **Khung CHUNG** (KHÔNG gắn 1 thuật toán): two-timescale HRL Manager/Worker ✅[`Akyıldız 2024`] + CMDP-Lagrangian (§2.3) + Π_feasible (§3.2) + GAE ✅[`Foundations_of_Deep_RL.pdf`].
- **3 RL core NGANG HÀNG** giải cùng bài toán (so sánh công bằng): **PPO** clipped ✅[`1707.06347v2.pdf`] (on-policy); **TD3** ✅[`fujimoto18a.pdf`] (off-policy deterministic); **SAC** ✅[`1812.05905v2.pdf` Haarnoja 2018] (off-policy max-entropy). *(B3-RCPO cũ loại khỏi Table I.)*

### 3.2 Kiến trúc & safety
- Manager (rApp) + Worker (xApp) actor-critic; action continuous → RRMPolicyRatio.
- **HRL thật — Manager có tác động NHÂN QUẢ lên env (W18+)**. Mỗi solver ghép Manager CÙNG thuật toán với Worker (parity tuyệt đối): PPO→PPO-Manager (on-policy), TD3→TD3-Manager (off-policy det.), SAC→SAC-Manager (off-policy stoch.). Vòng lặp Algorithm 1 mỗi Manager window (W=10 Worker steps):
  1. `s_H = build_manager_state(obs, λ_global)` — (6+4K+1)-dim.
  2. `a_H = manager.act(s_H)`; `b_rrm = decode_manager_action(a_H) = B_RRM_MIN + (B_RRM_MAX−B_RRM_MIN)·σ(a_H[0])`, `[B_RRM_MIN,B_RRM_MAX]=[0.05,0.85]`.
  3. **`env.set_rrm_budget(b_rrm)`** — re-anchor `r_min_urllc` (two-tier clip), tác động trực tiếp lên PRB split URLLC ⟹ obs[16]=anchor; Worker drift quanh anchor qua Δr_min.
  4. Worker chạy W bước; Manager return SMDP-discounted `r_H = Σ_{i=0}^{W−1} γ_L^i · r_aug_i` (KHÔNG undiscounted), `γ_H = γ_L^W ≈ 0.904 = GAMMA_MANAGER` cho return/GAE Manager.
  5. PPO-Manager: rollout buffer + update cuối episode. TD3/SAC-Manager: replay buffer nhỏ (5k), store `(s_H, a_H, r_H, s_H', done)` + update mỗi boundary khi `buffer ≥ warmup`. Checkpoint sidecar `*_manager.pt`.
  - Three-rate hierarchy khóa: `α_πH=1e-5 < α_λ=1e-4 < α_πL=1e-3` (Borkar two-timescale).
- **Safety filter = closed-form `Π_feasible`** (projection-onto-simplex [Duchi 2008] + isotonic/PAV + clip + floor) — Euclidean projection chính xác, **no learnable params** (gỡ β_qp/NSF-distillation); KHÔNG claim novel (Kim 2026 = neural QP, chỉ cite).
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
