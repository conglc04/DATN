# 02 — Requirements

## Severity QoS Table (single source of truth)
Ngưỡng từ **3GPP TS 22.261**. **Severity bệnh nhân** (5 mức, exogenous, cố định/episode) là **bộ chọn QoS-tier DUY NHẤT** — thay vai trò của 5-pha cũ (phase→severity swap 2026-06-14: xe luôn có bệnh nhân ⟹ pha STANDBY/DISPATCH/RETURN vô nghĩa). Đơn điệu: NON_URGENT lỏng nhất → IMMEDIATE chặt nhất. **2026-06-15**: `severity_per_amb ∈ {1..5}^K` sampled độc lập per ambulance (per-xe C1/C2/C4/C5 dùng `severity_per_amb[k]`); `severity_ref := max(severity_per_amb)` áp bảng này cho đại lượng SHARED (α_eMBB, C3 R_min, severity one-hot trong obs).

| Severity | D_max | ε (tail) | R_min^eMBB | AoI_max | Nguồn |
|---|---|---|---|---|---|
| 1 NON_URGENT | 20 ms | 1e-3 | 10 Mbps | 1.0 s | TS 22.261 Annex A (remote healthcare 10–100ms) |
| 2 SEMI_URGENT | 10 ms | 1e-4 | 15 Mbps | 0.5 s | TS 22.261 Annex A |
| 3 URGENT | 5 ms | 1e-4 | 20 Mbps | 0.2 s | TS 22.261 Annex A |
| 4 EMERGENCY | 2 ms | 1e-5 | 30 Mbps | 0.1 s | TS 22.261 Annex A Table A.1-1 |
| 5 IMMEDIATE | **1 ms** | **1e-5** | 30 Mbps | 0.1 s | TS 22.261 Annex D §D.1 + §7.2 |

- **α_eMBB(sev)** (reward weight): đơn điệu NGƯỢC — NON_URGENT 0.70 → IMMEDIATE 0.05 (sev cao ⟹ eMBB nhường PRB cho URLLC). Chỉ `α_eMBB` vào reward; URLLC ép qua Lagrangian.
- **AoI_max** (luồng URLLC tổng hợp `ambulance_status`, F=1, 2026-06-14 consolidation): theo severity (cột trên); placeholder, sweep {250,500,750}ms (M7.3).
- **ε rare-event**: KHÔNG claim đạt 1e-5 empirical (rule-of-three, [06](06_validation.md)).

## Severity (5-level triage, exogenous)
5 mức **Non-urgent / Semi-urgent / Urgent / Emergency / Immediate** theo **ATS — Australasian Triage Scale** ✅[`ATS — Australasian College for Emergency Medicine (ACEM)`] (internal `sev 1..5` = lỏng→chặt, ánh xạ ngược số hiệu ATS: sev 1=ATS 5 … sev 5=ATS 1). `severity_per_amb` (K,) sampled độc lập per ambulance, **cố định trong 1 episode** (1s; severity không đổi kịp trong 1s), **random giữa các episode** (lệch trung-nặng). `severity_ref := max(severity_per_amb)` là biến **exogenous Markov shared** đưa vào obs (one-hot [10:15]) ⟹ quyết định ngưỡng QoS shared (`SEVERITY_QOS`, `α_eMBB(sev)`, `λ_warm[sev]`); `severity_per_amb[k]` per-xe (obs per-amb block `severity_norm_k`) ⟹ ngưỡng C1/C2/C4/C5 per-xe + thứ tự ưu tiên C6 qua β/Π_feasible (K≥2). **KHÔNG mô phỏng giá trị sinh hiệu**. Phân biệt scene/transport (mobility) đã nằm trong `speed` obs. Chi tiết [W16](weeks/W16_pha1_severity_triage.md).

## Traffic Classes
- **URLLC (xe)**: F=1 luồng tổng hợp `ambulance_status`/ambulance (2026-06-14 consolidation, gộp HR/SpO2/ECG/DENM cũ) — periodic status bundle; LCFS+drop_old, AoI-aware.
  - 🔴 **Cấu hình canonical (mô phỏng dùng ĐÚNG bộ tham số này)**: `urllc_arrival_rate = 50 pkt/s`, `urllc_packet_bits = 3200` (**= 400 B/packet**) ⟹ offered load **160 kbps/xe** (mô hình hàng đợi Poisson M/G/1). **Kết quả PHẢI báo cáo theo cấu hình 50 pkt/s × 400 B này.**
  - ⚠️ Mô tả khái niệm "~500–1500B @ ~10–20Hz" (và biến thể "20 Hz × 1000 B") chỉ là **envelope conceptual**: có cùng offered load nhưng **phân phối kích thước/tần suất KHÁC** ⟹ động học queue delay và AoI khác (E[D]=λE[S²]/(2(1−ρ)) phụ thuộc E[S²], không chỉ offered load). KHÔNG dùng các con số này khi báo cáo số liệu.
- **eMBB (bystander nền)**: throughput slice, reward target. URLLC ∩ eMBB = ∅ [Alsenwi §II.A].

## Scenarios tới hạn
- **S1 — MCI hội tụ @ Bạch Mai** (chính, D22): 3 xe đồng trú **1 cell 300m** trên đường Giải Phóng, **severity khác nhau** (mỗi xe 1 bệnh nhân), hội tụ BV Bạch Mai → triage contention (C6 ordering).
- **S2 — Collision burst**: DENM spike + severity cao (Emergency/Immediate).
- **S3 — Rush hour**: density sweep {light/medium/heavy} (SUMO).
- **S4 — Bystander spike**: eMBB nền tăng đột biến.

## CMDP Constraints (Lagrangian relaxation)
- **C1** `E[D_e2e^k] ≤ D_max^{sev_k}` · **C2** `P(D_e2e^k>D_max)≤ε^{sev_k}` · **C3** `R_eMBB≥R_min` (shared, dùng `sev_ref`) · **C4** `E[AoI_k]≤AoI_max^{sev_k}` · **C5** `P(AoI_k>AoI_max)≤ε_AoI^{sev_k}` (cùng ngưỡng `AoI_max`, m=1 — đối xứng cặp C1/C2 dùng `D_max`; `ε_AoI` đối xứng `ε` của C2). `sev_k = severity_per_amb[k]`; budget per-severity = code `eps_aoi` (`SEVERITY_QOS`) / `d5_aoi_tail` (`CMDP_D_J_SEVERITY`). `c_vec/d_phi/λ` đều `(4K+1,)`, layout `[C1_0..C1_{K-1}, C2_0..C2_{K-1}, C4_0..C4_{K-1}, C5_0..C5_{K-1}, C3_shared]`.
- **C6** `severity_per_amb[i]>severity_per_amb[j] ⟹ E[D^i]≤E[D^j]` — **DEMOTE → metric** (#13): enforced bởi (1) β/Π_feasible weight-ordering structural (§2.4) + (2) C1/C2 per-xe; C6 báo cáo inversion-rate (KHÔNG λ_C6, KHÔNG Lagrangian).
- **Structural (by construction, KHÔNG Lagrangian)**: feasibility `ΣPRB=B_U`; no-starvation `PRB_k≥b≥PRB_min^QoS`; ratio constraints `r_ded≤r_min≤r_max`, `Σr≤1` (closed-form projection [TS 28.541]). ⚠️ **KHÔNG NSF/QP** (gỡ).

## Phát biểu bài toán (formal)
`max E[Σ α_e(sev)·log(1+R_eMBB/R_REF)]` s.t. C1–C6. CMDP `(S,A,P,r,{g_c},γ)`, primal-dual. Chi tiết [W17](weeks/W17_pha2_cmdp_formulation.md) + [13](13_methodology_walkthrough.md) §2.

## Convergence (Approximate Lagrangian)
Weak duality — **KHÔNG zero-duality-gap** (deep-NN + hierarchical phá convexity); hội tụ local saddle point verified empirically. ✅[`Spoor 2025`; `Ding 2023`].

## Cross-reference
[01](01_overview.md) · [05](05_agent_workflow.md) (Lagrangian) · [06](06_validation.md) (sweep W18–W23; E3/E4 future work) · [weeks/](weeks/README.md).
