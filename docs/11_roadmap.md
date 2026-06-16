# 11 — Implementation Roadmap (Critical Path & Gates)

> Mirror master plan PHẦN 6 (build-order B0–B9). Lịch tuần chi tiết = [09](09_execution_plan.md) + [weeks/](weeks/README.md).

## Dependency graph (must-build-before)
```
B0/B0b (gỡ LSTM+MEC+vital+β_qp)
   └─► B1 (references nền + audit config.py/REFERENCE_MAP)  ─► B4 (formulation CMDP)
B2 (severity.py — ATS) ┐
B3 (sumo_mobility.py) ─┴─► B5 (oran_env K≥2 + intra-slice + Π_feasible projection + assert obs 31/51)
                            ├─► W18 PPO K=1 ─► W19 PPO K=3
                            ├─► B6 (TD3, đã có) ─► W20 TD3 K=1 ─► W21 TD3 K=3
                            └─► B7 (SAC, NEW) ─► W22 SAC K=1 ─► W23 SAC K=3 (Table I/II)
                                                                              └─► B8 (W24: formulation completeness audit + luận án)
```

## Env config khóa (W18–W23)
gNB/cell-center=`(0,0)` (local Cartesian, điểm hội tụ 3 xe trên Giải Phóng), R_cell=300m, UMi 3GPP TR 38.901 single-cell, no handover — KHÔNG đổi giữa các tuần. GPS thật BV Bạch Mai (`config.py: BACH_MAI_LAT/LON`) chỉ dùng cho lớp OSM/SUMO (W15).

## Go/No-Go Gates
| Gate | Tuần | Điều kiện pass |
|---|---|---|
| 1A | W12 | M1.5/M1.6 hết 🟡; 0 đại lượng vô-nhãn M1–M3 |
| 1B | W13 | mọi thành phần D_e2e + ε + D_max ✅/🔴-declared |
| 1C | W14 | AoI ref đổi corpus (no Kaul); MEC removal xong; M8.3/AoI_max có sensitivity |
| 1D | W15 | SUMO trace chạy, dải SINR hợp lý; realism (KHÔNG vs RWP); scope 3-tầng declared |
| **1E** | W16 | mọi 🟡 M1–M11 → ✅; mọi 🔴 declared+plan → **Pha 1 xong** |
| **2** | W17 | obj+C1–C6+Lagrangian+intra-slice ✅/🔴; ký hiệu khớp obs/action → **Pha 2 xong** |
| **3A** | W18 | test_intra_slice + test_multi_amb pass; assert obs 31/51; PPO K=1 hội tụ; ~261 tests xanh |
| **3B** | W19 | PPO K=3 hội tụ trên obs=51; ≥1 kịch bản structural guarantee (ordering + no-starvation) |
| **3C** | W20 | TD3 K=1 hội tụ; C1–C5 trong ngưỡng hoặc λ phản ứng đúng hướng |
| **3D** | W21 | TD3 K=3 hội tụ trên obs=51; ≥1 kịch bản structural guarantee cho TD3 |
| **3E** | W22 | SAC code mirror TD3 (BaselineFlags/LambdaState); SAC K=1 hội tụ (α hội tụ hợp lý) |
| **3F** | W23 | SAC K=3 hội tụ trên obs=51; Table I (6 cell, 3 solver × K∈{1,3}) + Table II (severity/intra-slice K=3) có CI + p-value hiệu chỉnh |
| **3-FINAL** | W24 | formulation Pha 1-3 hoàn chỉnh (0 🟡 sót); Table I/II CI đủ (không overlap hoặc giải thích trung thực); E3/E4 declared future work — KHÔNG yêu cầu E3/E4 chạy |

## Critical path
B1 (refs) là cổng cho toàn bộ formulation; B5 (code K≥2 + assert obs 31/51) là cổng cho cả 6 tuần sweep W18–W23; W23 (Table I/II, 3 solver × K∈{1,3}) = headline. SUMO/OSM (B3) là hạ tầng GPS cho mobility (W15), KHÔNG phải eval harness của sweep W18–W23. **E3/E4 → future work**, KHÔNG còn trên critical path.

## Cross-reference
[09](09_execution_plan.md) · [weeks/README.md](weeks/README.md) · [06](06_validation.md) · [10](10_risks.md).
