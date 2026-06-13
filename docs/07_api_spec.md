# 07 — API Specification

## Observation space
**Quy tắc**: field-set = nguồn-sự-thật; số = derived; binding `assert env.observation_space.shape` tại B5 ([W18](weeks/W18_pha3_algorithm_code.md)). KHÔNG hard-code số.

- **K=1 = 33-dim** (sau gỡ LSTM 6 + MEC 1): 26 fixed + 3·K + F(=4 AoI).
  - 26 fixed = Q(2)+HOL(2)+PRB-ratio(3)+arrival(2)+BLER(1)+phase-OH(5)+t_phi(1)+ETA(1)+λ_local(5)+rrm_budget(1)+n_bys(1)+AoI mean/max(2).
- **K=3 = 58-dim** = `15·K + 10 + C(K,2)` = 45+10+3.
  - per-amb (×K)=15: {SINR, d, v, phase[5], severity, λ_C{1,2,4,5}[4], **AoI_worstnorm**, **AoI_mean**}.
  - shared=10: λ_C3[1] + rrm_budget[1] + Q[2] + HOL[2] + arrival[2] + BLER[1] + n_bys[1]; + λ_C6 pairs[C(K,2)].
  - `severity_k` ∈ {0,0.2,0.4,0.6,0.8,1.0} (NACA 1-6, KHÔNG vitals).

## Action space
- **K=1 = 6-dim**: `{Δr_min^URLLC, Δr_max^eMBB, r_ded, w_C1, w_C2, w_C3}`, bounds `[-1,-1,0,0,0,0]→[1,1,1,1,1,1]`.
- **K=3 = 7-dim**: + **β** (priority temperature; squash `β=β_min+(β_max−β_min)·sigmoid(a_β)`, β_min≈0.5, β_max≈4). ⚠️ β ≠ RL discount.
- ρ/δ/β_min/β_max = hyperparam (KHÔNG action). ũ_k DERIVED từ λ trong obs (KHÔNG thêm field).

## Mapping → 3GPP RRMPolicyRatio (✅[TS 28.541])
`r_min^URLLC`→`RRMPolicyMinRatio`; `r_max^eMBB`→`RRMPolicyMaxRatio`; `r_ded`→`RRMPolicyDedicatedRatio` (⊆ Min). Δ = thay đổi tương đối → tích lũy thành ratio; ratio chịu projection (`r_ded≤r_min≤r_max`, `Σ≤1`). w_C1/C2/C3 = trọng số constraint (Lagrangian warm), KHÔNG phải RRMPolicyRatio.

## info dict (mỗi step)
`{c_vec, d_phi, phase_now, ...}` — c_vec = 5/6 constraint values; d_phi = thresholds theo pha.

## ⚠️ Đã GỠ khỏi API
`u_MEC` obs field, MEC offload action, LSTM predictor heads, β_qp/NSF params.

## Cross-reference
[05](05_agent_workflow.md) (agent) · [02](02_requirements.md) (constraints) · [W18](weeks/W18_pha3_algorithm_code.md) (assert obs).
