# 09 — Execution Plan

> Mirror master plan PHẦN 11 (lịch tuần) + PHẦN 6 (build-order). Chi tiết atomic per-tuần ở [`weeks/`](weeks/README.md) — file này = tổng quan.

## GĐ A — Code foundation (W01–W11, ✅ DONE)
Repo/config → env (channel/queue/traffic/severity/AoI) → ORANEnv → reward+constraint → Lagrangian standalone → 3 solver → training loop → smoke+verify → tune. **W11 Exp1 RWP = LỊCH SỬ** (kết quả KHÔNG tái dùng). MEC/LSTM/vital/β_qp/B3: loại khỏi **scope/design** — B0/B0b **✅ đã thực thi 2026-06-13**. **Phase→Severity swap ✅ 2026-06-14**: 5-pha FSM bị thay bằng **5 mức severity bệnh nhân** làm trục ưu tiên duy nhất (xe luôn có bệnh nhân ⟹ pha STANDBY/DISPATCH/RETURN vô nghĩa); `phase_detector.py` xóa, obs K=1 **33→31**, **221 test xanh** (chi tiết [08_implementation_notes.md](08_implementation_notes.md#phasesseverity-swap--2026-06-14--severity-thay-phase-làm-trục-ưu-tiên)). **F=4→F=1 stream consolidation ✅ 2026-06-14**: 4 luồng AoI cũ (HR/SpO2/ECG/DENM) gộp thành 1 luồng `ambulance_status`/xe; obs K=1 **31→28**, `experiments/exp8_aoi.py` xóa, **219 test xanh** (chi tiết [08_implementation_notes.md](08_implementation_notes.md#f4f1-stream-consolidation--2026-06-14--1-luồng-ambulance_statusxe)). **Per-ambulance queue/AoI split ✅ 2026-06-14**: mỗi xe có hàng đợi URLLC + AoI tracker riêng (trước đó pooled chung K xe ⟹ policy không phân biệt được xe nào gần vi phạm QoS); obs K=1 **28→30** (thêm `delay_norm_k`/`AoI_norm_k`), **221 test xanh** (chi tiết [08_implementation_notes.md](08_implementation_notes.md#per-ambulance-queueaoi-split--2026-06-14--fix-lỗi-nghiêm-trọng-k2-không-phân-biệt-được-xe-nào-gần-vi-phạm-qos)). **Per-ambulance severity_k ✅ 2026-06-15**: `severity_per_amb∈{1..5}^K` sampled độc lập/cố định-episode (per-xe C1/C2/C4/C5), `severity_ref:=max(severity_per_amb)` cho đại lượng shared (α_eMBB, C3, severity one-hot); obs K=1 **30→31** (`24+5K+F`→`20+10K+F`, K=3,F=1→51), `(4K+1,)`-dim λ/c_vec/d_phi (layout `[C1_0..,C2_0..,C4_0..,C5_0..,C3_shared]`, permutation `[0,1,3,4,2]` ở K=1), action K≥2 +7-dim β∈[BETA_MIN,BETA_MAX]=[0,5] điều khiển `_prb_split_intra_slice` (C6 demote→structural metric, no λ_C6), **237 test xanh** (chi tiết [08_implementation_notes.md](08_implementation_notes.md#per-ambulance-severity_k-epic--2026-06-15---severity-độc-lập-từng-xe--lagrangianβπ_feasible-4k1-dim)).

## GĐ B — Reference-disciplined + severity/SUMO + solver sweep (W12–W24, 📅)
| Tuần | Pha | Build | Nội dung | Gate |
|---|---|---|---|---|
| W12 | 1 | B1 | Radio + UMi channel + capacity (M1–M3) | 1A |
| W13 | 1 | B1 | Delay E2E + reliability + QoS (M4–M6) | 1B |
| W14 | 1 | B1 | AoI + traffic (MEC gỡ) (M7–M8) | 1C |
| W15 | 1 | B3 | SUMO/OSM mobility (RWP bỏ) (M10) | 1D |
| W16 | 1 | B2 | Severity ATS 5-level (M11) | **1E** |
| W17 | 2 | B4 | CMDP: obj + C1–C6 + Lagrangian + intra-slice (P1–P5) | **2** |
| W18 | 3 | B0/B0b+B5 | Code K≥2 (assert obs 31/51; removals) + **PPO K=1** | **3A** |
| W19 | 3 | B5 | **PPO K=3** (severity + intra-slice) | **3B** |
| W20 | 3 | B6 | **TD3 K=1** | **3C** |
| W21 | 3 | B6 | **TD3 K=3** (severity + intra-slice) | **3D** |
| W22 | 3 | B7 | Code SAC (NEW) + **SAC K=1** | **3E** |
| W23 | 3 | B7 | **SAC K=3** + Table I/II (3 solver × K∈{1,3}) | **3F** |
| W24 | 3 | B8 | Formulation completeness audit + Luận án + defense | 3-FINAL |

## Env config khóa (W18–W23, KHÔNG đổi giữa các tuần)
gNB/cell-center=`(0,0)` (local Cartesian, điểm hội tụ 3 xe trên Giải Phóng), R_cell=300m, UMi 3GPP TR 38.901 single-cell, no handover. GPS thật BV Bạch Mai (`config.py: BACH_MAI_LAT/LON`) chỉ dùng cho lớp OSM/SUMO (W15). Chỉ `K_ambulances` (1↔3) và solver (PPO/TD3/SAC) thay đổi theo tuần.

## Phụ thuộc
B0/B0b→B1; B1→B4 (refs trước formulation); B2,B3→B5 (model trước code K≥2); B5→{W18 PPO K=1, W19 PPO K=3}→B6 (TD3, W20-21)→B7 (SAC NEW, W22-23)→B8 (W24); B4→B5. Quy tắc: KHÔNG start Wn nếu GATE W(n-1) chưa pass.

## Cross-reference
[weeks/README.md](weeks/README.md) (chi tiết) · [11_roadmap.md](11_roadmap.md) (dependency graph + gates) · [10_risks.md](10_risks.md).
