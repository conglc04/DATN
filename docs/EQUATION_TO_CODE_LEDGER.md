# Equation-to-Code Ledger

> Baseline `audit-baseline-e565846-20260620`. Each row: equation → units → code
> file:line → config source → independent test → runtime field → status.
> Independent evidence (not docs↔code): `audit/closure_checks.py` (30 checks),
> `audit/feasibility_oracle.py`, `audit/runtime_oracle.py`, `tests/test_formulation_*`.

## System model

| ID | Equation | Units | Code file:line | Config | Independent test | Status |
|----|----------|-------|----------------|--------|------------------|--------|
| M1 PL | UMa path loss `pl_uma(d,fc)` monotone ↑ in d | dB | `env/channel_model.py:29` | `F_CARRIER=3.5e9` | closure G2.pl_monotone | ✅ |
| M2 noise | `N = −174 + 10log₁₀(B) + NF(7)` | dBm | `env/channel_model.py:182` | `B_PRB=360e3` | closure G2.thermal_noise | ✅ |
| M3 SINR | `rx_dbm − (N⊕I)`; dB→lin `10^(x/10)` | dB | `env/oran_env.py:993-1003` | `sinr_clamp_[min,max]` | closure G2.cap (lin) | ✅ |
| M4 cap | `C_PRB = η·B_PRB·log₂(1+SINR_lin)` | bps/PRB | `env/channel_model.py:223` | `SHANNON_ETA=0.75` | closure G2.cap@{−5..20}dB | ✅ |
| M5 PRB | `PRB_URLLC + PRB_eMBB = 273` | PRB | `env/oran_env.py:966` | `P_TOTAL=273` | closure G5.inter_slice_273 | ✅ |
| M6 queue | `E[Dq] = λE[S²]/(2(1−ρ))` PK | s | `env/queue_model.py:75` | `D_STOCH=0.05ms` | closure G3.pk_delay | ✅ |
| M7 delay | `D_e2e = D_DET+1/μ+E[Dq]+D_FH+D_BH` | s | `env/oran_env.py:1213` | `D_DET/FH/BH` | runtime_oracle (a) | ✅ |
| M8 AoI | `Δ(t)=t−U(t)`, LCFS+drop, reset on update | s | `env/aoi_tracker.py` | `AoI_max^sev` | runtime_oracle | ✅ |
| M9 R_eMBB | `min(λ,μ)·bits/1e6` (slice total) | Mbps | `env/oran_env.py:1201` | `embb_packet_bits` | test_formulation_audit G7 | ✅ |

## Timescale (Gate 2)

| ID | Equation | Code | Independent test | Status |
|----|----------|------|------------------|--------|
| T1 | MAC tick 0.5 ms; Worker = 20 ticks = 10 ms | `MAC_TICKS_PER_WORKER=20` | closure G4 / timescale test | ✅ |
| T2 | Manager window = 10 Worker steps = 100 ms | `WORKER_STEPS_PER_MANAGER=10` | test_timescale | ✅ |
| T3 | `γ_H = γ_L^W ≈ 0.904` | `GAMMA_MANAGER` | closure G4.gamma_manager | ✅ |
| T4 | episode = entry→all-arrived/400 s ≠ 1 s rollout | `train.py` loop | test_timescale (2 s≠1 s) | ✅ |
| T5 | severity fixed/episode, no rollout-boundary resample | `oran_env.py:601` | test_mutation_guards m09 | ✅ |

## Objective + reward (Gate 6)

| ID | Equation | Units | Code | Independent | Status |
|----|----------|-------|------|-------------|--------|
| O1 | `r_t = α_e(sev_ref)·log(1+R_eMBB/R_REF)` | — | `env/oran_env.py:848-852` | runtime_oracle reward-form | ✅ |
| O2 | `α_e ∈ {.70,.55,.40,.20,.05}` (sev↓) | — | `config.py:160 SEVERITY_ALPHA` | test_formulas_config | ✅ |
| O3 | `sev_ref = max(severity_per_amb)` | — | `env/oran_env.py:601` | test_formulation_audit G7 | ✅ |

## Constraints C1–C5 (Gate 7) — `g_j ≤ 0` satisfied, `g_j > 0` violated

| Cj | `g_j` | unit | threshold (sev1→5) | c/d code | scale | test | Status |
|----|-------|------|---------------------|----------|-------|------|--------|
| C1 | `E[D_e2e^k]−D_max^{sev_k}` | s | 20/10/5/2/1 ms | `oran_env.py:887` / `build_d_phi[0:K]` | `D_REF=1e-3` | closure G11 / audit G2 | ✅ |
| C2 | `P(D>D_max^{sev_k})−ε^{sev_k}` **delay-tail, not BLER** | prob | 1e-3/1e-4/1e-4/1e-5/1e-5 | `oran_env.py:888` | `1.0` | mutation m12 | ✅ |
| C3 | `R_min−R_eMBB`, **fixed R_min=10 Mbps** | Mbps | 10 (all sev) | `oran_env.py:836,891` / `d=0` | `R_REF=100` | audit G7 / feasibility | ✅ |
| C4 | `E[AoI_k]−AoI_max^{sev_k}` | s | 1.0/0.5/0.2/0.1/0.1 | `oran_env.py:889` | `AOI_REF=0.1` | audit_gate3 C4 | ✅ |
| C5 | `P(AoI>AoI_max^{sev_k})−ε_AoI^{sev_k}` | prob | 1e-2/1e-3×4 | `oran_env.py:890` | `1.0` | mutation m16 | ✅ |

C5 `A_th = AoI_max^{sev}` (same threshold as C4, m=1; `config.py:484` `d5=eps_aoi`). `ε_AoI` is a probability (`0<d[3]<1`), not seconds — locked by mutation m16.

## Constraint vector + Lagrangian (Gate 8/9)

| ID | Equation | Code | Independent | Status |
|----|----------|------|-------------|--------|
| L1 | vector `[C1₀..C2..C4..C5..C3]` dim `4K+1` (K=1:5, K=3:13) | `config.py:508-572` | closure G11.dim | ✅ |
| L2 | `dev_j = (c_j−d_j)/scale_j` (threshold once) | `lagrangian.py:262` | audit G2 single-subtract | ✅ |
| L3 | `r_aug = r − Σ λ_j·max(0,dev_j)` (hinge, pre-update λ; fixed 2026-06-22 bonus-masking audit — was raw signed dev_j) | `lagrangian.py:275` | runtime_oracle (c) | ✅ |
| L4 | `λ ← clip(λ+α_λ·ĝ, 0, 10)` | `lagrangian.py:234` | runtime_oracle (d) | ✅ |
| L5 | C1/C2/C4/C5 active-denominator; C3 total-ticks | `oran_env.py:777-791` | audit G3 / mutation m04 | ✅ |

## Manager SMDP (Gate 10)

| ID | Equation | Code | Independent | Status |
|----|----------|------|-------------|--------|
| S1 | `R_H = Σ_{i<W} γ_L^i·r_aug,i` (PPO) | `train.py:299` | test_formulas_gae_smdp | ✅ |
| S2 | same SMDP sum (TD3/SAC, post-fix) | `train_offpolicy.py:259` | mutation m07 | ✅ |
| S3 | `γ_H = GAMMA_MANAGER` Bellman/GAE all 3 | `manager_agent.py:206,384,510` | test_solver_equivalence | ✅ |

## Intra-slice + N_req (Gate 4)

| ID | Equation | Code | Independent | Status |
|----|----------|------|-------------|--------|
| I1 | `N_req=⌈C_req/(η·B·log₂(1+SINR))⌉`, `C_req=load+pkt/D_max` (full units bps/[bps/PRB]=PRB) | `oran_env.py:1111-1119` | closure G7 / mutation m15 | ✅ |
| I2 | Phase-1 severity-tier-descending protection | `oran_env.py:1156-1178` | test_env_severity_k | ✅ |
| I3 | Phase-2 surplus `score=N_req·(1+β·urg)·softmax(w)` | `oran_env.py:1133-1184` | test_env_severity_k | ✅ |
| I4 | `Σ_k B_k = B_URLLC`, inactive=0 | `oran_env.py:1130-1186` | closure G5 / mutation m06,m17 | ✅ |
