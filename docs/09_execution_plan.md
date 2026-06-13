# 09 — Execution Plan

> Mirror master plan PHẦN 11 (lịch tuần) + PHẦN 6 (build-order). Chi tiết atomic per-tuần ở [`weeks/`](weeks/README.md) — file này = tổng quan.

## GĐ A — Code foundation (W01–W11, ✅ DONE)
Repo/config → env (channel/queue/traffic/phase/AoI) → ORANEnv → reward+constraint → Lagrangian standalone → 3 solver → training loop → smoke+verify → tune. **W11 Exp1 RWP = LỊCH SỬ** (kết quả KHÔNG tái dùng). MEC/LSTM/vital/β_qp: loại khỏi **scope/design**; xóa code = B0/B0b (W18, **chưa thực thi** — code hiện vẫn chứa).

## GĐ B — Reference-disciplined + severity/SUMO + solver sweep (W12–W24, 📅)
| Tuần | Pha | Build | Nội dung | Gate |
|---|---|---|---|---|
| W12 | 1 | B1 | Radio + UMi channel + capacity (M1–M3) | 1A |
| W13 | 1 | B1 | Delay E2E + reliability + QoS (M4–M6) | 1B |
| W14 | 1 | B1 | AoI + traffic (MEC gỡ) (M7–M8) | 1C |
| W15 | 1 | B3 | SUMO/OSM mobility (RWP bỏ) (M10) | 1D |
| W16 | 1 | B2 | Severity NACA-S (M11) | **1E** |
| W17 | 2 | B4 | CMDP: obj + C1–C6 + Lagrangian + intra-slice (P1–P5) | **2** |
| W18 | 3 | B0/B0b+B5 | Code K≥2 (assert obs 33/58; removals) + **PPO K=1** | **3A** |
| W19 | 3 | B5 | **PPO K=3** (severity + intra-slice) | **3B** |
| W20 | 3 | B6 | **TD3-Lag K=1** | **3C** |
| W21 | 3 | B6 | **TD3-Lag K=3** (severity + intra-slice) | **3D** |
| W22 | 3 | B7 | Code SAC-Lag (NEW) + **SAC-Lag K=1** | **3E** |
| W23 | 3 | B7 | **SAC-Lag K=3** + Table I/II (3 solver × K∈{1,3}) | **3F** |
| W24 | 3 | B8 | Formulation completeness audit + Luận án + defense | 3-FINAL |

## Env config khóa (W18–W23, KHÔNG đổi giữa các tuần)
gNB/cell-center=`(0,0)` (local Cartesian, điểm hội tụ 3 xe trên Giải Phóng), R_cell=300m, UMi 3GPP TR 38.901 single-cell, no handover. GPS thật BV Bạch Mai (`config.py: BACH_MAI_LAT/LON`) chỉ dùng cho lớp OSM/SUMO (W15). Chỉ `K_ambulances` (1↔3) và solver (PPO/TD3-Lag/SAC-Lag) thay đổi theo tuần.

## Phụ thuộc
B0/B0b→B1; B1→B4 (refs trước formulation); B2,B3→B5 (model trước code K≥2); B5→{W18 PPO K=1, W19 PPO K=3}→B6 (TD3-Lag, W20-21)→B7 (SAC-Lag NEW, W22-23)→B8 (W24); B4→B5. Quy tắc: KHÔNG start Wn nếu GATE W(n-1) chưa pass.

## Cross-reference
[weeks/README.md](weeks/README.md) (chi tiết) · [11_roadmap.md](11_roadmap.md) (dependency graph + gates) · [10_risks.md](10_risks.md).
