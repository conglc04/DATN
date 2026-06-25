# 01 — Overview

## Bối cảnh
Tại Hà Nội (mật độ giao thông cao), **đa xe cứu thương** cùng truyền dữ liệu y tế khẩn (vital signs ECG/SpO2/HR/BP + cảnh báo DENM/CAM) qua mạng **5G O-RAN network slicing**, yêu cầu độ trễ cực thấp + độ tin cậy cực cao. Khi nhiều xe cạnh tranh tài nguyên trong cùng cell, cần **ưu tiên theo độ nguy kịch bệnh nhân** (medical triage) — không phải chia đều.

- **Xe = URLLC** (vital + DENM/CAM). Video 4K = future/bundled, KHÔNG eMBB per-xe.
- **eMBB = nền bystander** (dân băng rộng) — reward target.
- O-RAN điều phối tài nguyên qua Near-RT RIC + O-DU theo chuẩn **RRMPolicyRatio** [3GPP TS 28.541].

## Phương pháp mô phỏng
**Analytical simulation**: M/G/1 Pollaczek–Khinchine cho queue delay (upper-bound trung bình) + Monte Carlo rollouts cho stochastic dynamics (fading, arrival jitter, sensor fail) + 3GPP TR 38.901 **UMa** path-loss cho SINR. **KHÔNG packet-level** (ns-3 = future work). Cho phép train RL hàng triệu episode; M/G/1 PK giả định Poisson → URLLC burst (DENM) bắt bằng Monte Carlo.

## Giải pháp — Khung CMDP + 3 solver ngang hàng
**Đóng góp = BÀI TOÁN tối ưu (khung CHUNG, KHÔNG gắn với 1 thuật toán)**. Các thành phần dựng nên bài toán:
- **Severity-aware (trục ưu tiên chính)**: 5 mức **ATS — Australasian Triage Scale** (Non-urgent→Immediate; exogenous, **cố định/episode**, **KHÔNG vitals giả**) → chọn ngưỡng QoS (`SEVERITY_QOS`) + intra-slice priority/ordering (K≥2). Severity vào hệ **chỉ qua constraint C1–C5 + λ** (reward KHÔNG còn trọng số α_e — bỏ 2026-06-23). Thay 5-pha cũ (xe luôn có bệnh nhân ⟹ pha vô nghĩa). Scene/transport (mobility) đã có trong `speed` obs.
- **CMDP-Lagrangian**: constraint qua dual ascent (`λ←clip(λ+α·g,0,Λ_max)`), KHÔNG soft penalty trong reward.
- **Two-timescale Manager/Worker**: điều phối Manager (100ms) / Worker (10ms) — khung thời gian chung. **Cả 3 solver đều HRL Manager+Worker** (PPO→ManagerAgent on-policy; TD3→TD3ManagerAgent, SAC→SACManagerAgent off-policy) — khác biệt CHỈ ở RL core (on/off-policy + cadence update), KHÔNG ở bài toán/khung HRL.
- **Structural safety**: closed-form feasibility projection (Option B floor + ordering by construction) — KHÔNG cần NN/QP.

**3 method RL NGANG HÀNG giải cùng bài toán này** (KHÔNG đề cao thuật toán nào): **PPO** (on-policy), **TD3** (off-policy, deterministic), **SAC** (off-policy, max-entropy) → so sánh công bằng ở Table I/II.

## Đóng góp (3 build-thật + 1 headline novel)
1. **★ C1 — Context-aware constraints (severity)**: **Severity = ngữ cảnh bệnh nhân (ATS 5-level → chọn QoS tier + intra-slice priority + ordering, NOVEL)** — lấp gap "no user-criticality TRONG slice".
2. **C2 — CMDP-Lagrangian HRL → RRMPolicyRatio** (✅ built+tested): Manager/Worker → chuẩn 3GPP RRMPolicy{Min/Max/Dedicated}Ratio.
3. **C3 — AoI hard constraints** (✅ built+tested): freshest-data priority (C4/C5) thay latency thuần.
4. *Feasibility projection* (honest, **KHÔNG claim novel** như Kim 2026).

> **LSTM, MEC, NSF+QP "hybrid safety" — ĐÃ LOẠI HOÀN TOÀN** (standard/vestigial/trùng λ_warm). KHÔNG nhắc như đóng góp lẫn future.

## Bối cảnh triển khai
**Kiểu O-RAN đô thị Hà Nội**: channel UMa (mô hình 3GPP TR 38.901) + hình học phố từ OSM (Bạch Mai) + mobility SUMO. **KHÔNG claim dữ liệu/hạ tầng/collaboration Viettel thật** (chưa có). f_c=3.5GHz (FR1 n78) ✅[TS 38.101-1].

## Phạm vi
**Trong scope**: severity-aware intra-slice cho K xe URLLC same-cell (K∈{1,3}); **single-cell** UMa 1km @ Bạch Mai; sweep tuần tự **3 solver × K∈{1,3}** (PPO, TD3, SAC → Table I/II); pipeline 3 pha (Mô hình → Tối ưu → Giải). Luận án.
**Ngoài scope**: handover liên cell; MEC/compute offload; packet-level ns-3; calibrate số đo giao thông thực HN; mô phỏng giá trị sinh hiệu.
**Future work (D26, 2026-06-12)**: E3 (AoI re-run SUMO, LCFS vs FCFS) và E4 (stress/robustness) — C3 (AoI hard constraints) vẫn active trong CMDP của sweep W18–W23.

## Tài liệu liên quan
- Yêu cầu: [02_requirements.md](02_requirements.md) · Kiến trúc + channel: [03_architecture.md](03_architecture.md) · Luồng + delay: [04_data_flow.md](04_data_flow.md)
- Thuật toán: [05_agent_workflow.md](05_agent_workflow.md) · Validation: [06_validation.md](06_validation.md) · Spec toán 3 pha: [13_methodology_walkthrough.md](13_methodology_walkthrough.md)
- Kế hoạch tuần: [weeks/](weeks/README.md) (W01–W24) · Nguồn: [REFERENCE_MAP.md](REFERENCE_MAP.md)
