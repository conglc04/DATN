# W17 — Pha 2: Phát biểu bài toán tối ưu (CMDP)

> **Pha**: 2 · **Status**: 📅 PLANNED · **Gate**: **GATE 2** · **Nhóm**: P1–P5 · **Build**: B4 · **Deps**: GATE 1E

## P1 — Hàm mục tiêu
- **P1.1** `max E[Σ_t α_e·log(1 + R_eMBB/R_REF)]` (tối đa eMBB nền, single-term, giữ W11) — ✅[`Madyan Alsenwi…[2022].pdf` Eq.13; `R. Sohaib…[2024].pdf` Eq.9]; R_REF từ M8.3.
- ⟲ severity KHÔNG vào mục tiêu (chỉ constraint/allocation — §1.5).

## P2 — Khung CMDP
- **P2.1** `(S, A, P, r, {g_c}, γ)` primal-dual — ✅[`Yongshuai Liu…[2020].pdf`; `Wen Wu…[2020].pdf`; `Qiang Liu…[2021].pdf`] *(thay Altman 1999 — vắng corpus)*. ⚠️ `γ` ở đây = **RL discount của CMDP**, KHÁC `β` priority temperature (#3).
- ⟲ state/action khớp PHẦN 2 (obs K=1=31 / K=3=51 assert @[W18](W18_pha3_algorithm_code.md); Worker action 1-dim K=1 / (1+K)-dim K≥2; Manager action 1-dim b_rrm).

## P3 — Ràng buộc (mỗi cái 1 nguồn)
- **P3.1** `C1: E[D_e2e^k] ≤ D_max^{sev_k}` — ✅[3GPP TS 22.261 §7.2/Annex A] (nối M6)
- **P3.2** `C2: P(D_e2e^k > D_max^{sev_k}) ≤ ε^{sev_k}` — ✅[3GPP TS 22.261 §7.2] (nối M5)
- **P3.3** `C3: R_eMBB ≥ R_min` — ✅[`Alsenwi…pdf`; `Sohaib…pdf`]
- **P3.4** `C4: E[AoI_k] ≤ AoI_max^{sev_k}` — 🔴-declared (AoI_max M7.3=500ms placeholder); AoI→hard-constraint là khác biệt declared so với Qi/Chen (dùng AoI objective).
- **P3.5** `C5: P(AoI_k > AoI_max^{sev_k}) ≤ ε_AoI^{sev_k}` — **cùng ngưỡng `AoI_max^{sev_k}` như C4** (m=1, đối xứng cặp C1/C2 dùng `D_max`; `ε_AoI` đối xứng `ε` của C2); budget `ε_AoI^{sev_k}` per-severity = code `eps_aoi` (`SEVERITY_QOS`) / `d5_aoi_tail` (`CMDP_D_J_SEVERITY`). 🔴-declared, FORM grounded.
- **P3.6** `C6: severity_per_amb[i]>severity_per_amb[j] ⟹ E[D_e2e^i] ≤ E[D_e2e^j]` — **DEMOTE → metric chẩn đoán** (#13): bảo đảm severity = (1) weight-ordering structural + (2) C1/C2 per-xe; C6 chỉ báo cáo **priority-inversion rate** (KHÔNG λ_C6, KHÔNG Lagrangian — bảo đảm bằng §2.4 weight-ordering structural + C1/C2 per-xe). Giới hạn channel-infeasibility declared.

## P4 — Lagrangian relaxation
- **P4.1** `L = J − Σ_c λ_c·g_c`; `λ_c ← clip(λ_c + α_λ·g_c, 0, Λ_max)` — ✅[`Lindsay Spoor…[2025].pdf`; `Dongsheng Ding…[2023].pdf`] *(thay Boyd/Tessler)*.
- **P4.2** Disclaimer **đối ngẫu yếu** (no zero-duality-gap, deep-NN) — ✅[`Lindsay Spoor…[2025].pdf`].

## P5 — Tầng intra-slice (cấu trúc)
- **P5.1** ⚠️ **Thuật toán intra-slice ĐÃ ĐỔI** (W18+): `κ/δ-softmax` bên dưới là **lịch sử design**; code thực thi = **severity-ordered N_req tier-protection** (2 pha), xem [05_agent_workflow.md](../05_agent_workflow.md#intra-slice-priority-worker-k2) và `oran_env.py:_prb_split_intra_slice`. ~~`b=max(floor(κ·B_U/K), PRB_min^QoS)`, `w_k=softmax(β·sev+δ·ũ)`, `PRB_k=b+S·w_k`~~ → Pha 1: `N_req[k]=ceil(C_req[sev_k]/cap_per_prb(SINR_k))` theo tier severity giảm dần; Pha 2: surplus theo `score[k]=N_req·(1+β·urgency)·softmax(w)[k]`. `β∈[BETA_MIN,BETA_MAX]=[0.5,5]` (squash sigmoid từ `a[0]`, K≥2; K=1: β≡BETA_MIN). — ✅ nguyên lý weighted-priority [3GPP TS 23.501 §5.7 5QI].
- **P5.2** 2 structural guarantee (đại số tự chứa, verify bằng test `tests/test_env_severity_k.py`): feasibility `ΣPRB=B_U` (cần `B_U≥K·PRB_min^QoS`); no-starvation CỨNG `PRB_k≥PRB_MIN_QOS`; tier-ordering (severity cao → cấp trước). K=1: `PRB_0=B_U` luôn (numeric preservation). *(Legacy `δ=ρ·β, ρ=RHO_URGENCY_TIEBREAK=0.15` = unused.)*
- **P5.3** Tiebreaker cấp 2 `urgency_k=λ_C1_k` (per-amb λ_local trong obs per-amb block, normalized ∈[0,1]) — phá hòa CHỈ trong cùng tier severity.

## ⟲ RÀ SOÁT P1–P5
softmax đơn điệu theo sev (test mọi pair, kể cả ũ biên); Σw=1; b≤PRB_k≤B_U; C6 ghi rõ "soft/metric, không zero-violation"; ký hiệu khớp 1-1 obs/action PHẦN 2; γ(RL)≠β(priority).

## GATE 2 (kết thúc Pha 2)
Mọi objective + C1–C6 + Lagrangian + intra-slice ✅/🔴-declared; disclaimer đối ngẫu yếu ghi rõ; rà chéo ký hiệu khớp obs/action. **→ Pha 2 hoàn tất.**

## Liên kết
Master plan PHẦN 11/W17 + §1.2/1.3/1.4 · `docs/02_requirements.md` (constraints) · `docs/05_agent_workflow.md` (Lagrangian update) · giải bằng thuật toán → [W18](W18_pha3_algorithm_code.md).
