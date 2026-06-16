# 10 — Risk Register

> Rủi ro hiện tại (post-cleanup). Nhãn 🔴 = cần calibrate/sensitivity; 🟡 = cần gắn ref trước gate.

| # | Rủi ro | Mức | Giảm thiểu |
|---|---|---|---|
| R1 | **PRB_min^QoS** chưa có giá trị số (load-bearing cho floor + feasibility) | 🟡 | Thủ tục nghịch đảo M3→M4/M6/M8 ([W13](weeks/W13_pha1_delay_reliability_qos.md)); numeric solve khi M3/M6/M8 khóa |
| R2 | **AoI_max** không có ref y khoa | 🔴 | Placeholder 500ms (5×period SpO2/BP) + sweep {250,500,750}ms; IEEE 11073 nếu tải được |
| R3 | **λ-saturation**: λ chạm Λ_max=10 mà viol chưa→0 ⟹ URLLC under-served | 🔴 | Log λ-trajectory + saturation-rate (sweep W18–W23); sweep Λ_max↑ nếu cần |
| R4 | **ε=1e-5 rare-event**: 10 seeds KHÔNG validate được | 🔴 | Rule-of-three `ε≤3/N`; KHÔNG claim 1e-5 empirical; IS/EVT future |
| R5 | **SUMO fidelity**: chỉ Tầng 1 (signals+speed-limit) + density sweep | 🔴 | Declare honest: KHÔNG calibrate số đo thực HN; OD-synthetic/timing-đèn-thật = future/out-of-scope |
| R6 | **severity→priority** không có ref y khoa trực tiếp | 🔴 | Design principle declared (triage analogy NEWS2/RTS) |
| R7 | **κ, γ_max/β_max, ρ, λ_det/λ_stab, D_stoch, s_C6, m, ε_AoI** = tham số chưa có ref | 🔴 | Sensitivity sweep declared (đã nêu dải mỗi cái ở Reference Ledger) |
| R8 | **Compute**: K=3 ×10+ seeds × sweep W18–W23 (3 solver × K∈{1,3}; E3/E4 → future work, D26) | 🟡 | 10 seeds tối thiểu; tăng nếu CI rộng; pre-register |
| R9 | **Novelty Related Work**: nếu paper khác đã làm intra-slice user-differentiation | 🟡 | Verify trực tiếp từng paper (PHẦN 10); hedge "to best of our knowledge" |

## Scope cuts nếu trễ
- C5 tail-AoI → demote optional (dựa C4 mean).
- C6 → metric thuần (đã demote).
- KNN site-specific channel → bỏ (dùng UMi generic TR 38.901).
- (E3/E4 đã → future work toàn bộ, D26 — không còn là "scope cut", là default.)

## Cross-reference
[09](09_execution_plan.md) · [06](06_validation.md) (stats discipline) · master plan Reference Ledger.
