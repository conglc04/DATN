# Kế hoạch từng tuần — PPO (W01–W24)

> **Cấu trúc**: 1 file = 1 tuần, mỗi tuần có GATE riêng — KHÔNG start Wn nếu GATE W(n-1) chưa pass.
> **Nguồn chuẩn**: master plan `~/.claude/plans/s-p-x-p-l-i-plan-jaunty-toast.md` (PHẦN 11 = lịch tuần, PHẦN 6 = build-order). Mọi đại lượng gắn nhãn ✅ VERIFIED / 🟡 NEEDS-REF / 🔴 NEEDS-CALIBRATION; nguồn vắng corpus phải flag rõ.
> **Pipeline 3 pha**: Mô hình hệ thống → Bài toán tối ưu (CMDP) → 3 solver (PPO, TD3, SAC) giải + thực nghiệm sweep K∈{1,3}.
> **REVISED 2026-06-13**: W18–W23 = sweep tuần tự **3 solver × K∈{1,3}** (PPO K=1 → PPO K=3 → TD3 K=1 → TD3 K=3 → SAC K=1 → SAC K=3), env config khóa: gNB/cell-center=`(0,0)`, R_cell=300m, UMi single-cell, no handover.

## GĐ A — Code foundation (W01–W11, ✅ DONE)
| Tuần | Nội dung | Gate |
|---|---|---|
| [W01](W01_foundation.md) | Repo + config + test infra | G0 |
| [W02](W02_env_channel_queue_traffic.md) | Env I: channel + queue M/G/1 + traffic | G1.1 |
| [W03](W03_env_phase_aoi.md) | Env II: phase 5-FSM + AoI (MEC+vital GỠ) | G1.2 |
| [W04](W04_oran_env_sanity.md) | ORANEnv + sanity (obs K=1=33) | G1 |
| [W05](W05_reward_constraint_tracking.md) | Reward single-term + 5-constraint tracking | G2.1 |
| [W06](W06_lagrangian_standalone.md) | Lagrangian standalone (Pha 2 complete) | G2 |
| [W07](W07_apply_three_solvers.md) | 3 solver: PPO + TD3 + SAC | G3.1 |
| [W08](W08_training_loop.md) | Algorithm 1 training loop | G3.2 |
| [W09](W09_smoke_phase3_gate.md) | 100-ep smoke + Pha 3 verify (β_qp GỠ) | G3 |
| [W10](W10_tune_multiseed_prep.md) | Tune + multiseed prep | G3.3 |
| [W11](W11_exp1_baseline_rwp_historical.md) | Exp1 baseline RWP — **LỊCH SỬ, KHÔNG tái dùng** | — |

## GĐ B — Reference-disciplined + severity/SUMO + solver sweep (W12–W24, 📅 PLANNED)
| Tuần | Pha | Nội dung | Gate |
|---|---|---|---|
| [W12](W12_pha1_radio_channel_capacity.md) | 1 | Radio + channel UMi 300m Bạch Mai + capacity (M1–M3) | 1A |
| [W13](W13_pha1_delay_reliability_qos.md) | 1 | Delay E2E + reliability + QoS 3GPP (M4–M6) | 1B |
| [W14](W14_pha1_aoi_traffic.md) | 1 | AoI + traffic (MEC GỠ) (M7–M8) | 1C |
| [W15](W15_pha1_sumo_mobility.md) | 1 | Mobility SUMO/OSM (RWP bỏ) (M10) | 1D |
| [W16](W16_pha1_severity_naca.md) | 1 | Severity NACA-S exogenous (M11) | **1E** |
| [W17](W17_pha2_cmdp_formulation.md) | 2 | CMDP: objective + C1–C6 + Lagrangian + intra-slice (P1–P5) | **2** |
| [W18](W18_pha3_algorithm_code.md) | 3 | Code K≥2 (B0/B0b+B5, assert obs 33/58) + **PPO K=1** | **3A** |
| [W19](W19_pha3_e1_baseline_sumo.md) | 3 | **PPO K=3** (severity + intra-slice) | **3B** |
| [W20](W20_pha3_e2_severity_headline.md) | 3 | **TD3 K=1** | **3C** |
| [W21](W21_formulation_completeness.md) | 3 | **TD3 K=3** (severity + intra-slice) | **3D** |
| [W22](W22_sac_k1.md) | 3 | Code SAC (B7, NEW) + **SAC K=1** | **3E** |
| [W23](W23_sac_k3.md) | 3 | **SAC K=3** + Table I/II compilation (3 solver × K∈{1,3}) | **3F** |
| [W24](W24_thesis_writing_defense.md) | 3 | Formulation completeness audit + Luận án + defense | 3-FINAL |

## Solver sweep (W18–W23, env config khóa)
- **Env config (XUYÊN SUỐT)**: gNB/cell-center=`(0,0)` local Cartesian, R_cell=300m, UMi (3GPP TR 38.901) single-cell, no handover. GPS thật BV Bạch Mai (`config.py: BACH_MAI_LAT/LON`) chỉ dùng cho lớp OSM/SUMO (W15), KHÔNG dùng trực tiếp trong RL env.
- **Thứ tự**: PPO K=1 (W18) → PPO K=3 (W19) → TD3 K=1 (W20) → TD3 K=3 (W21) → SAC K=1 (W22) → SAC K=3 (W23).
- **Table I/II** (W23): so sánh 3 solver × K∈{1,3} (6 cell) — reward, constraint-violation rates C1–C5(+C6 K=3), λ-saturation, severity/intra-slice metrics (K=3).
- **E3** AoI (LCFS vs FCFS) và **E4** stress/robustness → **future work**; C3 (AoI hard constraints) vẫn built+tested và active trong sweep.

## Solvers (3 — NGANG HÀNG, KHÔNG đề cao thuật toán nào)
| Solver | Họ | Nguồn |
|---|---|---|
| **PPO** | on-policy (clipped PPO + GAE) + Lagrangian | [`1707.06347v2.pdf`] |
| **TD3** | off-policy TD3 + Lagrangian (deterministic actor) | [`fujimoto18a.pdf`] |
| **SAC** | off-policy SAC + Lagrangian (max-entropy stochastic actor) | [Haarnoja 2018 SAC + arXiv:1812.05905] |

## Doc tham chiếu
- `../13_methodology_walkthrough.md` — spec toán 3 pha · `../06_validation.md` — Table I/II + stats (E3/E4 future work) · `../11_roadmap.md` — build-order + gates · `../REFERENCE_MAP.md` — nguồn.
