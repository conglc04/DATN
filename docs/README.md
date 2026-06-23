# PPO — Documentation

Severity-Aware Intra-slice Scheduling cho 5G O-RAN ambulance (Hà Nội, single-cell UMa 1km @ Bạch Mai).

> **Nguồn-sự-thật**: master plan `~/.claude/plans/s-p-x-p-l-i-plan-jaunty-toast.md`. Bộ docs này = phản chiếu repo-resident, audited, reference-disciplined (✅/🟡/🔴). Pipeline 3 pha: Mô hình hệ thống → Bài toán tối ưu → 3 solver ngang hàng (PPO/TD3/SAC) giải.

## Doc map
| Doc | Nội dung |
|---|---|
| [01_overview](01_overview.md) | Bối cảnh, đóng góp (severity headline), scope |
| [02_requirements](02_requirements.md) | Severity QoS, severity, constraints C1–C6, scenario MCI |
| [03_architecture](03_architecture.md) | O-RAN topology, channel UMa single-cell 1km, severity exogenous |
| [04_data_flow](04_data_flow.md) | D_e2e decomposition (no MEC), AoI model |
| [05_agent_workflow](05_agent_workflow.md) | HRL 2-tier, RRMPolicyRatio, Lagrangian, projection |
| [06_validation](06_validation.md) | 3 solver (PPO, TD3, SAC), sweep K∈{1,3} → Table I/II (E3/E4 future work), stats discipline |
| [07_api_spec](07_api_spec.md) | obs (31/51) + Worker action (1-dim K=1 / (1+K)-dim K≥2) + Manager action (1-dim b_rrm) |
| [08_implementation_notes](08_implementation_notes.md) | Code map, removals (LSTM/MEC/vital/β_qp) |
| [09_execution_plan](09_execution_plan.md) | Tổng quan lịch tuần W01–W24 |
| [10_risks](10_risks.md) | Risk register (🔴/🟡) |
| [11_roadmap](11_roadmap.md) | Build-order B0–B9 + dependency graph + gates |
| [13_methodology_walkthrough](13_methodology_walkthrough.md) | Spec toán 3 pha (master) |
| [REFERENCE_MAP](REFERENCE_MAP.md) | Đại lượng → nguồn (corpus-only; vắng-corpus flagged) |
| [weeks/](weeks/README.md) | Kế hoạch atomic từng tuần W01–W24 |

## Nguyên tắc (no over-claim)
- Mọi đại lượng có nhãn ✅ VERIFIED / 🟡 NEEDS-REF / 🔴 NEEDS-CALIBRATION; nguồn vắng corpus flag rõ.
- KHÔNG: zero-duality-gap, regret bound, vitals giả, ε=1e-5 empirical, dữ liệu Viettel thật, calibrate giao thông HN thật.
- ĐÃ GỠ khỏi scope: LSTM, MEC, NSF+QP "hybrid safety", β_qp, RWP mobility.
- Đóng góp headline = **severity-aware intra-slice** (triage → network priority + ordering).
