# 06 — Validation

## Solvers (3 — so sánh trong Table I)
| Solver | Họ | Nguồn |
|---|---|---|
| **PPO** | on-policy (clipped PPO + GAE) + Lagrangian | ✅[`1707.06347v2.pdf`] |
| **TD3** | off-policy TD3 + Lagrangian (deterministic actor) | ✅[`fujimoto18a.pdf`] |
| **SAC** | off-policy SAC + Lagrangian (max-entropy stochastic actor) | ✅[`1812.05905v2.pdf` Haarnoja 2018] |

Ablation variants (equal-weight / phase-only / severity-only / full) → Table II (component study), KHÔNG phải solver. *(B3-RCPO cũ đã gỡ HOÀN TOÀN khỏi code + Table I; chỉ còn trong week docs lịch sử W07/W09/W11.)*

## Solver sweep (W18–W23, mobility = SUMO duy nhất, RWP bỏ)
Sweep tuần tự **3 solver × K∈{1,3}** (6 cell), env config khóa (gNB=`(0,0)`, R_cell=300m, UMi single-cell, no handover):
- **PPO** K=1 [W18](weeks/W18_pha3_algorithm_code.md) → K=3 [W19](weeks/W19_pha3_ppo_k3.md); **TD3** K=1 [W20](weeks/W20_pha3_td3_k1.md) → K=3 [W21](weeks/W21_pha3_td3_k3.md); **SAC** K=1 [W22](weeks/W22_sac_k1.md) → K=3 [W23](weeks/W23_sac_k3.md).
- **Table I** = 3 solver × K∈{1,3}: reward + C1–C5 violation-rate + λ-saturation. **Table II** (K=3) = severity/intra-slice metrics + ablation 2×2.

### Future work (D26, 2026-06-12)
- **E3** AoI (LCFS vs FCFS, re-run SUMO) — C3 (AoI hard constraints, P3.4/P3.5) vẫn **built+tested và active** trong sweep; chỉ thực nghiệm minh chứng riêng bị demote.
- **E4** stress/robust (load/burst/noise/sensor-fail).
- Formulation completeness audit + buffer Table I/II → [W24](weeks/W24_thesis_writing_defense.md).

## Statistical validity
- **Multiple-comparison**: Holm–Bonferroni (primary, p<0.01); effect size **Hedges' g**; **bootstrap 95% CI**. ✅[phương pháp chuẩn] *(Holm 1979/Hedges 1985 vắng corpus → ghi công thức)*.
- **Sample size**: 10 seeds = TỐI THIỂU; tăng 20–30 nếu CI rộng/overlap. **KHÔNG claim "thắng" nếu CI chồng lấn**; pre-register cặp so sánh.
- **ε rare-event (#7)**: 1e-5 KHÔNG validate được bằng 10 seeds (cần ~10⁶–10⁷ mẫu). → báo cáo observed violation-rate + CI; **rule-of-three** `ε≤3/N`; KHÔNG vẽ "đạt ε=1e-5". IS/EVT = future.
- **λ-saturation logging**: λ-trajectory + %step `λ==Λ_max`; flag saturation-without-convergence.

## Metrics severity (Table II, K=3) — KHÔNG Jain toàn cục
Jain thuần thưởng phân bổ ĐỀU ⟹ **mâu thuẫn** severity priority. Bộ metric đúng:
1. **ordering-compliance** STRICT khi surplus (#15): `sev_i>sev_j ∧ S>0 ⟹ PRB_i>PRB_j`; báo cáo kèm `(PRB_i−PRB_j)` và `S`.
2. **no-starvation min-share**: `min_k PRB_k ≥ b ≥ PRB_min^QoS`.
3. **fairness TRONG tier**: Jain trên nhóm cùng-severity HOẶC weighted-Jain `PRB_k/w_k` 🟡[Jain 1984 vắng corpus].
4. **priority-inversion rate** (C6 slack-gated, <1% empirical declared); **adaptation lag** (<5 Manager steps).

## Ablation 2×2 factorial (#22)
Trục phase∈{off,on} × severity∈{off,on} → equal/phase-only/severity-only/full. Mỗi cặp kề khác đúng 1 trục → đo đóng góp riêng từng thành phần.

## Verification checklist
- [ ] obs K=1=31 / K=3=51 (assert); action 6/7-dim (+β); reward single-term; 3 solver (PPO + TD3 + SAC).
- [ ] Mọi M*/P*/A* có nhãn ✅/🟡/🔴; 0 citation gán file chưa mở.
- [ ] structural guarantee (feasibility + ordering) verify bằng test.
- [ ] KHÔNG over-claim: no zero-duality-gap, no regret bound, no fake vitals, no ε=1e-5 empirical.

## Cross-reference
[02](02_requirements.md) (constraints) · [13](13_methodology_walkthrough.md) §3.3 · [weeks/](weeks/README.md) (sweep W18–W23 + GATE 3A–3F; E3/E4 = future work).
