# W17 — Pha 2: Phát biểu bài toán tối ưu (CMDP)

> **Pha**: 2 · **Status**: 📅 PLANNED · **Gate**: **GATE 2** · **Nhóm**: P1–P5 · **Build**: B4 · **Deps**: GATE 1E

## P1 — Hàm mục tiêu
- **P1.1** `max E[Σ_t α_e·log(1 + R_eMBB/R_REF)]` (tối đa eMBB nền, single-term, giữ W11) — ✅[`Madyan Alsenwi…[2022].pdf` Eq.13; `R. Sohaib…[2024].pdf` Eq.9]; R_REF từ M8.3.
- ⟲ severity KHÔNG vào mục tiêu (chỉ constraint/allocation — §1.5).

## P2 — Khung CMDP
- **P2.1** `(S, A, P, r, {g_c}, γ)` primal-dual — ✅[`Yongshuai Liu…[2020].pdf`; `Wen Wu…[2020].pdf`; `Qiang Liu…[2021].pdf`] *(thay Altman 1999 — vắng corpus)*. ⚠️ `γ` ở đây = **RL discount của CMDP**, KHÁC `β` priority temperature (#3).
- ⟲ state/action khớp PHẦN 2 (obs K=1=33 / K=3=58 assert @[W18](W18_pha3_algorithm_code.md); action 7-dim K=3).

## P3 — Ràng buộc (mỗi cái 1 nguồn)
- **P3.1** `C1: E[D_e2e^k] ≤ D_max^{φ_k}` — ✅[3GPP TS 22.261 §7.2/Annex A] (nối M6)
- **P3.2** `C2: P(D_e2e^k > D_max^{φ_k}) ≤ ε^{φ_k}` — ✅[3GPP TS 22.261 §7.2] (nối M5)
- **P3.3** `C3: R_eMBB ≥ R_min` — ✅[`Alsenwi…pdf`; `Sohaib…pdf`]
- **P3.4** `C4: E[AoI_k] ≤ AoI_max` — 🔴-declared (AoI_max M7.3=500ms placeholder); AoI→hard-constraint là khác biệt declared so với Qi/Chen (dùng AoI objective).
- **P3.5** `C5: P(AoI_k > A_th) ≤ δ_tail` — `A_th=m·AoI_max` (m=2, sweep {1.5,2,3}), `δ_tail=1e-2` (sweep {1e-2,1e-3,ε^φ}). 🔴 trên (m,δ_tail), FORM grounded; demote optional nếu bất định.
- **P3.6** `C6: s_i>s_j ⟹ E[D_e2e^i] ≤ E[D_e2e^j]` — **DEMOTE → metric chẩn đoán** (#13): bảo đảm severity = (1) weight-ordering structural + (2) C1/C2 per-xe; C6 chỉ báo cáo **priority-inversion rate** (soft-nudge dead-band+slack-gate optional, λ_C6 **per-pair** #14). Giới hạn channel-infeasibility declared.

## P4 — Lagrangian relaxation
- **P4.1** `L = J − Σ_c λ_c·g_c`; `λ_c ← clip(λ_c + α_λ·g_c, 0, Λ_max)` — ✅[`Lindsay Spoor…[2025].pdf`; `Dongsheng Ding…[2023].pdf`] *(thay Boyd/Tessler)*.
- **P4.2** Disclaimer **đối ngẫu yếu** (no zero-duality-gap, deep-NN) — ✅[`Lindsay Spoor…[2025].pdf`].

## P5 — Tầng intra-slice (cấu trúc)
- **P5.1** Guaranteed-min Option B: `b=max(κ·B_U/K, PRB_min^QoS@SINR_ref)`, `S=B_U−K·b`, `w_k=softmax(β·sev_k + δ·ũ_k)`, `PRB_k=b+S·w_k`. β = priority temp ∈[β_min,β_max] (squash sigmoid; β_min≈0.5 chống collapse; β_max≈4) — ✅ nguyên lý weighted-priority [3GPP TS 23.501 §5.7 5QI] *(WFQ Parekh&Gallager vắng corpus → "tương tự WFQ" KHÔNG ✅)*.
- **P5.2** 2 structural guarantee (đại số tự chứa, verify bằng test): feasibility `ΣPRB=B_U` (cần `B_U≥K·PRB_min^QoS`); no-starvation CỨNG `PRB_k≥b≥PRB_min^QoS`; weight-ordering đơn điệu `sev_i>sev_j⟹w_i>w_j` (đảm bảo bởi `δ=ρ·β, ρ<0.2`).
- **P5.3** Tiebreaker cấp 2 `urgency_k=Σ_c λ_Cc^k` (đã trong obs, ũ_k=urgency/max ∈[0,1], #16) — phá hòa CHỈ trong cùng tier.

## ⟲ RÀ SOÁT P1–P5
softmax đơn điệu theo sev (test mọi pair, kể cả ũ biên); Σw=1; b≤PRB_k≤B_U; C6 ghi rõ "soft/metric, không zero-violation"; ký hiệu khớp 1-1 obs/action PHẦN 2; γ(RL)≠β(priority).

## GATE 2 (kết thúc Pha 2)
Mọi objective + C1–C6 + Lagrangian + intra-slice ✅/🔴-declared; disclaimer đối ngẫu yếu ghi rõ; rà chéo ký hiệu khớp obs/action. **→ Pha 2 hoàn tất.**

## Liên kết
Master plan PHẦN 11/W17 + §1.2/1.3/1.4 · `docs/02_requirements.md` (constraints) · `docs/05_agent_workflow.md` (Lagrangian update) · giải bằng thuật toán → [W18](W18_pha3_algorithm_code.md).
