# REFERENCE MAP — Đại lượng / Công thức → Nguồn

> Mirror master plan Reference Ledger. **Quy tắc thép**: chỉ file thật trong `documents/` hoặc chuẩn 3GPP/IEEE/ETSI/ns-3/SUMO mới làm ✅. Nguồn cổ điển **vắng corpus** (Kaul/Altman/Boyd/Tessler/Borkar/Parekh/Jain/Schulman-2016) phải ghi rõ "vắng corpus", KHÔNG trích như đã có.

## 1. Công thức / đại lượng → nguồn + nhãn
| ID | Đại lượng | Nhãn | Nguồn |
|---|---|---|---|
| M1.1/1.2 | P_TOTAL=273, B_PRB=360kHz | ✅ | TS 38.101-1 Table 5.3.2-1; TS 38.211 §4.4.4 |
| M1.3 | P_tx^UE=23dBm (uplink) | ✅ | TS 38.101-1 §6.2.1 |
| M1.5/1.6 | η=0.75, NF=7dB | 🟡 | `Hyoungju Ji 2017` §II-B; `Zexian Li 2018` §III / TR 38.101-4 |
| M2.1-2.3 | UMa path-loss/shadow (sweep) `pl_uma`; UMi LOS/NLOS (legacy micro) | ✅ | TR 38.901 §7.4.1/§7.4.2 |
| M2.0 | R_cell=1km single-cell UMa macro Bạch Mai + interference margin −86 dBm/PRB (W15-B2) | 🔴 | UMa-norm declared (D25); KHÔNG đo gNB thật; SINR≈2.7dB@edge |
| M3.1 | Shannon C=η·PRB·B·log₂(1+SINR) | ✅ | Shannon; η `Hyoungju Ji 2017`; MCS TS 38.214 |
| M4.1/4.3/4.4 | D_DET/D_FH/D_BH | 🟡 | TS 38.214 §5.1; `O-RAN.pdf` WG4; `Nie Cheng 2022` §III |
| M4.2 | M/G/1 Pollaczek–Khinchine | ✅ | `9780470316887.pdf` (Kleinrock) §5.6 |
| M4.5 | D_stoch=0.05ms | 🔴 | declared PB-C2 ±50% |
| M5/M6 | reliability + QoS thresholds | ✅ | TS 22.261 §7.2 + Annex A/D |
| M7.1 | AoI Δ(t)=t−U(t) | ✅ | `Qi 2024`; `Chen`; `Mlika 2022` *(KHÔNG Kaul)* |
| M7.3 | AoI_max | 🔴 | placeholder 500ms, sweep {250,500,750}ms; IEEE 11073 ngoài corpus |
| M8.1/8.1b | traffic Poisson + payload/rate | ✅/🟡 | `Alsenwi 2022`; `Sohaib 2024`; size/rate 🟡 (IEEE 11073/V2X) |
| M8.3 | R_REF=100Mbps | 🟡 | `Weijian Zhou` §IV; sweep {50,100,200,300} |
| M11.1 | ATS 5-level triage scale | ✅ | `ATS — Australasian College for Emergency Medicine (ACEM)` |
| M11.2 | `severity_per_amb` exogenous per-ambulance, cố định/episode + sampling weights; `severity_ref:=max(severity_per_amb)` cho đại lượng shared | 🔴 | abstraction + sensitivity (CTMC/birth-death + phase-event MAP loại bỏ — severity cố định/episode) |
| M11.6 | `severity_per_amb`→priority (pure-RL per-vehicle logits + λ-penalty gradient, K≥2 — KHÔNG còn β/Π_feasible weight-ordering, gỡ 2026-06-21) | 🔴 | design principle (ATS triage analogy) |
| P1.1 | objective eMBB log-utility | ✅ | `Alsenwi 2022` Eq.13; `Sohaib 2024` Eq.9 |
| P2.1 | CMDP framework | ✅ | `Yongshuai Liu 2020`; `Wen Wu 2020`; `Qiang Liu 2021` *(KHÔNG Altman)* |
| P4.1 | Lagrangian dual ascent | ✅ | `Spoor 2025`; `Ding 2023` *(KHÔNG Boyd/Tessler)* |
| §1.3 | PRB_min^QoS (Option B floor) | 🟡 | thủ tục nghịch đảo M3→M4/M6/M8 (W13) |
| §1.3 | κ, γ_max/β_max, ρ, β_min | 🔴 | sweep declared (no medical ref); **β_max/β_min nay LEGACY/unused** trong intra-slice allocation (gỡ 2026-06-21, giữ constant chỉ để import/test compat) |
| §1.4 | λ_c init/α/clip `(4K+1,)` | ✅ | reuse ALPHA_LAMBDA_DUAL=1e-4, LAMBDA_MAX=10; C6 = structural metric (no λ_C6) |
| A1.1/1.2 | PPO; GAE | ✅ | `1707.06347v2`; `Foundations_of_Deep_RL` *(Schulman 2016 vắng corpus)* |
| A2.1 | two-timescale HRL | ✅ | `Akyıldız 2024` *(Borkar/FeUdal vắng corpus)* |
| A3.2/3.3 | TD3; SAC (sibling solvers) | ✅ | `fujimoto18a`; `1812.05905v2` (Haarnoja 2018 SAC) *(B3-RCPO loại khỏi Table I)* |
| A-E2.3 | fairness (KHÔNG Jain toàn cục) | 🟡 | within-tier/weighted-Jain; Jain 1984 vắng corpus |
| P5.1 | softmax priority weight | ✅ | TS 23.501 §5.7 5QI *(WFQ Parekh&Gallager vắng corpus)* |

## 2. Per-week verdict
- **W01–W11 (GĐ A, done)**: foundation code grounded; số RWP (W11) = lịch sử, KHÔNG tái dùng.
- **W12–W16 (Pha 1)**: mọi 🟡 → ✅ trước GATE 1A–1E; 🔴 declared + sensitivity.
- **W17 (Pha 2)**: objective + C1–C6 + Lagrangian + intra-slice gắn nguồn.
- **W18–W23 (Pha 3)**: thuật toán + stats discipline (rule-of-three ε, no Jain toàn cục, λ-saturation).

## 3. Nguồn vắng corpus (KHÔNG dùng làm ✅)
Kaul 2012 (AoI) · Altman 1999 (CMDP) · Boyd 2004 / Tessler 2018 RCPO (Lagrangian) · Borkar 2008 / FeUdal 2017 (HRL) · Parekh&Gallager 1993 (WFQ) · Jain 1984 (fairness) · Schulman 2016 (GAE) · Holm 1979 / Hedges 1985 (stats). → đã thay bằng nguồn corpus tương đương hoặc ghi công thức trực tiếp.

## Cross-reference
[13](13_methodology_walkthrough.md) (Bibliography) · [weeks/](weeks/README.md) (nhãn per-tuần) · master plan Reference Ledger (PHẦN 11).
