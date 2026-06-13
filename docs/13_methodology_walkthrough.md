# 13 — Methodology Walkthrough (3-Phase Pipeline)

> Spec toán hợp nhất, mirror master plan `~/.claude/plans/s-p-x-p-l-i-plan-jaunty-toast.md` PHẦN 1 (thiết kế toán) + PHẦN 2 (I/O). Chi tiết atomic theo tuần ở [`weeks/`](weeks/README.md). Mọi đại lượng có nhãn ✅/🟡/🔴; nguồn vắng corpus flag rõ.
>
> **Pipeline**: Pha 1 Mô hình hệ thống → Pha 2 Bài toán tối ưu (CMDP) → Pha 3: 3 solver NGANG HÀNG (PPO/TD3/SAC) giải. RL/DL là **bước CUỐI** (solver), sau khi phát biểu bài toán hoàn tất; KHÔNG đề cao thuật toán nào.

---

## Pha 1 — Mô hình hệ thống

### 1.1 Bài toán General-K, 2 lớp
- **Lớp 1 INTER-SLICE** (K-agnostic): chia P_TOTAL giữa URLLC slice (xe) vs eMBB slice (bystander) → `r_min^URLLC, r_max^eMBB`.
- **Lớp 2 INTRA-URLLC** (K≥2): chia `B_U = r_min^URLLC·P_TOTAL` giữa K xe theo severity → `w_k = softmax(β·severity_k)`.
- K=1 (E1) = chỉ Lớp 1; K=3 (E2) = Lớp 1+2 đầy đủ. Xe = **URLLC-only** (vital + DENM/CAM); eMBB = nền bystander.

### 1.2 State / Action (PHẦN 2)
- **Observation**: K=1 = **33-dim** (sau gỡ LSTM+MEC); K=3 = `15K + 10 + C(K,2)` = **58-dim**. Field-set = nguồn-sự-thật; số = derived; binding `assert observation_space.shape` ([W18](weeks/W18_pha3_algorithm_code.md)). Per-amb (×K): {SINR, d, v, phase[5], severity, λ_C{1,2,4,5}[4], AoI_worstnorm, AoI_mean}=15.
- **Action**: K=1 = 6-dim {Δr_min, Δr_max, r_ded, w_C1, w_C2, w_C3}; K=3 = 7-dim (+**β** = priority temperature, squash `β_min+(β_max−β_min)·sigmoid`, β_min≈0.5, β_max≈4). ⚠️ β ≠ RL discount `γ_H/γ_L` (#3). ρ/δ = hyperparam (KHÔNG action). Mapping → **RRMPolicyRatio** [3GPP TS 28.541] (Min/Max/Dedicated).

### 1.3 Định luật vật lý / miền (nhóm M, [W12](weeks/W12_pha1_radio_channel_capacity.md)–[W16](weeks/W16_pha1_severity_naca.md))
- **Channel** UMi single-cell 300m @ Bạch Mai — ✅[3GPP TR 38.901 §7.4]; P_tx^UE=23dBm uplink ✅[TS 38.101-1].
- **Capacity** `C_k=η·PRB_k·B_PRB·log₂(1+SINR_k)`, SINR linear, single-cell SNR (I≈0), η=0.75 🟡 — ✅[Shannon; `Hyoungju Ji 2017`].
- **Delay** `D_e2e = D_DET + d_tx + d_queue + D_FH + D_BH` (KHÔNG D_MEC); `d_queue` = **M/G/1 PK** ✅[`Kleinrock 9780470316887.pdf` §5.6] (upper-bound trung bình; burst bắt bằng Monte Carlo).
- **AoI** `Δ(t)=t−U(t)`, LCFS+drop — ✅[`Qi 2024`; `Chen`; `Mlika 2022`] *(KHÔNG Kaul, vắng corpus)*; AoI thuần timestamp (no fake vitals).
- **Severity** NACA-S 6 mức exogenous (birth-death ±1 phase-independent + phase-event MAP theo loại) — ✅[`Prehospital_emergency_anesthesia`; `Champion 1989`]; severity→priority = design principle declared 🔴.
- **Traffic** Poisson + payload/rate per-stream (F=4 luồng) — ✅[`Alsenwi 2022`; `Sohaib 2024`]; R_REF=100Mbps 🟡[`Weijian Zhou`].

### 1.4 Thang thời gian
MAC TTI=0.5ms [TS 38.211 μ=1] · Worker(xApp)=10ms · Manager(rApp)=1s; `W=WORKER_STEPS_PER_MANAGER=10` (`config.py:315`). Two-timescale `γ_H≈γ_L^W`.

---

## Pha 2 — Bài toán tối ưu (CMDP) — [W17](weeks/W17_pha2_cmdp_formulation.md)

### 2.1 Objective
`max E[Σ_t α_e·log(1 + R_eMBB/R_REF)]` — ✅[`Alsenwi 2022` Eq.13; `Sohaib 2024` Eq.9]. **Single-term**; severity & URLLC KHÔNG vào reward (enforced qua Lagrangian → tránh double-counting).

### 2.2 Constraints (CMDP `(S,A,P,r,{g_c},γ)` — ✅[`Yongshuai Liu 2020`; `Wen Wu 2020`; `Qiang Liu 2021`])
| | Ràng buộc | Nguồn |
|---|---|---|
| C1 | `E[D_e2e^k] ≤ D_max^{φ_k}` | ✅[TS 22.261 §7.2/Annex A] |
| C2 | `P(D_e2e^k > D_max^{φ_k}) ≤ ε^{φ_k}` | ✅[TS 22.261 §7.2] |
| C3 | `R_eMBB ≥ R_min` | ✅[Alsenwi; Sohaib] |
| C4 | `E[AoI_k] ≤ AoI_max` | 🔴-declared (AoI_max=500ms placeholder) |
| C5 | `P(AoI_k > A_th) ≤ δ_tail` (A_th=m·AoI_max) | 🔴 (form grounded; demote optional) |
| C6 | `s_i>s_j ⟹ E[D^i] ≤ E[D^j]` | **DEMOTE→metric** (#13); design principle declared |

### 2.3 Lagrangian
`L = J − Σ_c λ_c·g_c`; `λ_c ← clip(λ_c + α_λ·g_c, 0, Λ_max=10)`, α_λ=1e-4 — ✅[`Spoor 2025`; `Ding 2023`] *(KHÔNG Boyd/Tessler)*. Disclaimer **weak duality** (no zero-duality-gap, deep-NN).

### 2.4 Intra-slice (cấu trúc, LUÔN khả thi)
`b = max(κ·B_U/K, PRB_min^QoS@SINR_ref)` (Option B), `S=B_U−K·b`, `w_k=softmax(β·sev_k+δ·ũ_k)`, `PRB_k=b+S·w_k`. **2 structural guarantee** (đại số, verify test): feasibility `ΣPRB=B_U` (cần `B_U≥K·PRB_min^QoS`) + no-starvation `PRB_k≥b` + weight-ordering đơn điệu (δ=ρ·β, ρ<0.2). Tiebreaker `ũ_k=urgency_k/max`, `urgency_k=Σλ_Cc^k`.

---

## Pha 3 — Giải bằng 3 solver ngang hàng (PPO/TD3/SAC) — [W18](weeks/W18_pha3_algorithm_code.md)

### 3.1 Khung chung + 3 solver ngang hàng
- **Khung CHUNG** (KHÔNG gắn 1 thuật toán): two-timescale HRL Manager/Worker ✅[`Akyıldız 2024`] + CMDP-Lagrangian (§2.3) + Π_feasible (§3.2) + GAE ✅[`Foundations_of_Deep_RL.pdf`].
- **3 RL core NGANG HÀNG** giải cùng bài toán (so sánh công bằng): **PPO** clipped ✅[`1707.06347v2.pdf`] (on-policy); **TD3** ✅[`fujimoto18a.pdf`] (off-policy deterministic); **SAC** ✅[`1812.05905v2.pdf` Haarnoja 2018] (off-policy max-entropy). *(B3-RCPO cũ loại khỏi Table I.)*

### 3.2 Kiến trúc & safety
- Manager (rApp) + Worker (xApp) actor-critic; action continuous → RRMPolicyRatio.
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
- **Medical**: NACA [`Prehospital_emergency_anesthesia`; `s12245-024-00605-5`]; RTS [`Champion 1989`]; triage [`news2-executive-summary_0`].
- **O-RAN/slicing**: [`O-RAN.pdf`]; [`Alsenwi 2022`; `Sohaib 2024`; `Filali 2022`; `Weijian Zhou`].

## Cross-reference
`weeks/` (atomic per-week) · `01_overview.md` · `02_requirements.md` · `05_agent_workflow.md` · `06_validation.md` · `REFERENCE_MAP.md`.
