# 02 — Requirements

## Phase QoS Table (single source of truth)
Ngưỡng từ **3GPP TS 22.261** — severity **KHÔNG** modulate D_max (severity = priority weight, KHÔNG đổi target).

| Phase | D_max | ε (tail) | R_min^eMBB | Nguồn |
|---|---|---|---|---|
| φ₁ STANDBY | 20 ms | 1e-3 | 10 Mbps | TS 22.261 Annex A (remote healthcare 10–100ms) |
| φ₂ DISPATCH | 5 ms | 1e-4 | 20 Mbps | TS 22.261 Annex A |
| φ₃ SCENE | **1 ms** | **1e-5** | 30 Mbps | TS 22.261 Annex D §D.1 + §7.2 |
| φ₄ TRANSPORT | 2 ms | 1e-5 | 30 Mbps | TS 22.261 Annex A Table A.1-1 |
| φ₅ RETURN | 20 ms | 1e-3 | 10 Mbps | TS 22.261 Annex A |

- **AoI_max** (aggregated vital streams HR/SpO2/BP): 🔴 placeholder **500ms** (≈5×period SpO2/BP @10Hz), sweep {250,500,750}ms (M7.3). KHÔNG modulate theo pha ngoài mức đã chốt.
- **ε rare-event**: KHÔNG claim đạt 1e-5 empirical (rule-of-three, [06](06_validation.md)).

## Severity (NACA-S)
6 mức exogenous {0,0.2,…,1.0} ↔ NACA 1-6 (NACA7=death out-of-scope); event-driven (birth-death ±1 phase-independent + phase-event MAP theo loại). **KHÔNG mô phỏng giá trị sinh hiệu**. Severity → intra-slice priority weight + ordering (KHÔNG vào reward). Chi tiết [W16](weeks/W16_pha1_severity_naca.md).

## Traffic Classes
- **URLLC (xe)**: F=4 luồng {HR_agg, SpO2_agg, ECG_waveform, DENM} — payload+rate per-stream (M8.1b); Poisson arrival.
- **eMBB (bystander nền)**: throughput slice, reward target. URLLC ∩ eMBB = ∅ [Alsenwi §II.A].

## Scenarios tới hạn
- **S1 — MCI hội tụ @ Bạch Mai** (chính, D22): 3 xe đồng trú **1 cell 300m** trên đường Giải Phóng, severity khác nhau, hội tụ BV Bạch Mai → triage contention. Cửa sổ = SCENE(φ₃)+TRANSPORT(φ₄).
- **S2 — Collision burst**: DENM spike @φ₃ + severity spike (collision→clip(s+Δ)).
- **S3 — Rush hour**: density sweep {light/medium/heavy} (SUMO).
- **S4 — Bystander spike**: eMBB nền tăng đột biến.

## CMDP Constraints (Lagrangian relaxation)
- **C1** `E[D_e2e^k] ≤ D_max^{φ_k}` · **C2** `P(D_e2e^k>D_max)≤ε^{φ_k}` · **C3** `R_eMBB≥R_min` · **C4** `E[AoI_k]≤AoI_max` · **C5** `P(AoI_k>A_th)≤δ_tail`.
- **C6** `s_i>s_j ⟹ E[D^i]≤E[D^j]` — **DEMOTE → metric** (#13): severity enforced bởi (1) weight-ordering structural + (2) C1/C2 per-xe; C6 báo cáo inversion-rate (soft-nudge optional, λ_C6 per-pair).
- **Structural (by construction, KHÔNG Lagrangian)**: feasibility `ΣPRB=B_U`; no-starvation `PRB_k≥b≥PRB_min^QoS`; ratio constraints `r_ded≤r_min≤r_max`, `Σr≤1` (closed-form projection [TS 28.541]). ⚠️ **KHÔNG NSF/QP** (gỡ).

## Phát biểu bài toán (formal)
`max E[Σ α_e·log(1+R_eMBB/R_REF)]` s.t. C1–C6. CMDP `(S,A,P,r,{g_c},γ)`, primal-dual. Chi tiết [W17](weeks/W17_pha2_cmdp_formulation.md) + [13](13_methodology_walkthrough.md) §2.

## Convergence (Approximate Lagrangian)
Weak duality — **KHÔNG zero-duality-gap** (deep-NN + hierarchical phá convexity); hội tụ local saddle point verified empirically. ✅[`Spoor 2025`; `Ding 2023`].

## Cross-reference
[01](01_overview.md) · [05](05_agent_workflow.md) (Lagrangian) · [06](06_validation.md) (sweep W18–W23; E3/E4 future work) · [weeks/](weeks/README.md).
