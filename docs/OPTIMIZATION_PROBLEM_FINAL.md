# Optimization Problem — Final (CMDP, pre-training closure)

> Baseline `audit-baseline-e565846-20260620`. Cross-checked against
> `EQUATION_TO_CODE_LEDGER.md` and the independent oracles. SoT = `utils/config.py`
> + `agents/lagrangian.py` + `env/oran_env.py`.

## Decision variables

- **Manager** `a_H ∈ ℝ` → `b_rrm = 0.05 + 0.80·σ(a_H)` → inter-slice PRB split (`B_URLLC=⌊b_rrm·273⌋`, `B_eMBB=273−B_URLLC`). Re-decided every 100 ms.
- **Worker** (K≥2) `a_L = (β, w₀..w_{K−1}) ∈ ℝ^{1+K}`, `β=0.5+4.5·σ(a_L[0])` → intra-URLLC priority over active ambulances. K=1: trivial no-op.

## State

- **Worker obs** `(20 + 11K + F)`-dim (K=1,F=1 → 32; K=3 → 54): fixed block (ρ, HOL, PRB ratios, arrivals, BLER, severity_ref one-hot, λ_C3, anchor, n_bys, AoI mean/max) + per-amb 11-dim block (SINR, dist, speed, delay_norm, AoI_norm, sev_norm, λ_C1/C2/C4/C5, **active_mask_k**) + F AoI-stream. `active_mask_k ∈ {0,1} = entered_k & ~arrived_k` (explicit active flag — disambiguates inactive xe from active-empty-queue). λ overlaid by `overlay_lambda_local` (single source, all solvers).
- **Manager state** `(6+4K+1)`-dim: (ρ_U, ρ_e, BLER, sev_ref_norm, AoI mean/max) + λ_global.
- Severity `severity_per_amb ∈ {1..5}^K` sampled **once at reset**, fixed for the mission (locked, `m09`). **`sev_ref := max(severity_per_amb)`** drives shared quantities (α_e reward weight, severity one-hot). **C3 floor is severity-independent.**

## Objective (Gate 6)

```
max  E[ Σ_t α_e(sev_ref) · log(1 + R_eMBB,t / R_REF) ],   R_REF = 100 Mbps,
     sev_ref := max_k severity_per_amb[k]
```
Single-term eMBB log-utility. `α_e ∈ {0.70,0.55,0.40,0.20,0.05}` (decreasing in severity). URLLC/AoI enforced via constraints only (no reward double-count). **Note**: the reward is still severity-dependent via `α_e` even though the C3 floor is fixed — eMBB is NOT fully decoupled from severity; only the C3 SLA is.

**Why `sev_ref = max(·)` (explicit modeling choice, K≥2).** The system objective is a SINGLE scalar, so the per-mission eMBB-deprioritization weight `α_e` must collapse the K per-ambulance severities to one. We take the **max** = the most-urgent patient on the cell: as soon as any one ambulance is critical, eMBB is deprioritized to that level (most conservative for patients). This is a deliberate decision, not a degenerate "one representative severity" — each ambulance's **own** QoS is still enforced per-vehicle via its own `severity_per_amb[k]` in the C1/C2/C4/C5 constraints and per-amb λ; only the shared scalar reward weight + C3 floor + severity one-hot use `max`. Alternatives (mean / per-amb-weighted sum) would dilute the priority of the single critical patient and are explicitly rejected.

## Constraints (CMDP, `g_j ≤ 0` feasible)

| Cj | Type | Scope | Constraint | Threshold (sev1→5) | Subgradient | Grounding |
|----|------|-------|------------|---------------------|-------------|-----------|
| C1 | **MEAN** | per-amb | `E[D_e2e^k] ≤ D_max^{sev_k}` | 20/10/5/2/1 ms | Option-b (window) | TS 22.261 |
| C2 | **CHANCE** | per-amb | `P(D_e2e^k > D_max^{sev_k}) ≤ ε^{sev_k}` | 1e-3/1e-4/1e-4/1e-5/1e-5 | Option-a (cumulative) | TS 22.261 |
| C3 | **MEAN** | shared | `E[R_eMBB] ≥ R_min = 10 Mbps` (severity-independent) | 10 (all sev) | Option-b (window) | Alsenwi/Sohaib |
| C4 | **MEAN** | per-amb | `E[AoI_k] ≤ AoI_max^{sev_k}` | 1.0/0.5/0.2/0.1/0.1 s | Option-b (window) | declared |
| C5 | **CHANCE** | per-amb | `P(AoI_k > AoI_max^{sev_k}) ≤ ε_AoI^{sev_k}` | 1e-2/1e-3×4 | Option-a (cumulative) | declared |

**Type definitions**: MEAN = expectation over the trajectory (feasible when time-average ≤ budget); CHANCE = probability of exceedance ≤ ε (feasible when tail rate stays below threshold). C3 is a mean-throughput floor — it constrains E[R_eMBB], NOT P(R_eMBB < 10). Option-b (interval-window, N≈200, reset each Manager step) suits mean-type; Option-a (episode-cumulative, N grows) is mandatory for chance-type at ε ≤ 1e-4 — see `agents/lagrangian.py` docstring.

**Severity-tier granularity** (documented, not a bug): C1 + α_e fully distinguish 5 levels; C2 uses 3 standardized reliability tiers (making 5 would fabricate non-standard nines); C4 saturates at a 0.1 s freshness floor; C5 is near-binary (non-urgent vs urgent). See `config.py:SEVERITY_QOS` rationale.

## Lagrangian (Gate 8/9)

- Constraint vector `c_vec`, threshold `d_phi`, multipliers `λ` all **`(4K+1)`-dim**, layout `[C1₀..C1_{K-1}, C2.., C4.., C5.., C3_shared]` (K=1→5, K=3→13).
- Normalized deviation `dev_j = (c_j − d_j)/scale_j`, `scale = [D_REF, 1, AoI_REF, 1, R_REF]`-blocked. Threshold subtracted **exactly once**.
- **Augmented reward** `r_aug = r − Σ_j λ_j·max(0, dev_j)` (hinge — a slack constraint contributes exactly 0, never a bonus; fixed 2026-06-22 bonus-masking audit) using the **pre-update** λ (the λ in the state); `λ_{t+1}` created only after the transition. Dual ascent (below) still uses the raw **signed** `dev_j` so λ can relax when a constraint is slack.
- **Dual ascent** `λ_j ← clip(λ_j + α_λ·ĝ_j, 0, Λ_max)`, `α_λ=1e-4`, `Λ_max=10`, `ĝ_j` = mean window deviation. Fixed-rate projected ascent — *inspired by* multi-timescale stochastic approximation; **no formal convergence claim**.
- Active-time normalization: C1/C2/C4/C5 per-ambulance active-sample denominator; **C3 over total ticks** (slice-level, never active-masked). Inactive ambulance → c=0 (constraint trivially satisfied), dual relaxes ~1e-4/window (negligible, correct CMDP behavior).

## Manager SMDP (Gate 10)

`R_H = Σ_{i=0}^{W−1} γ_L^i · r_aug,i` over each 100 ms window — identical formula in PPO (`train.py:299`), TD3 & SAC (`train_offpolicy.py:259`). Bellman/GAE discount `γ_H = GAMMA_MANAGER ≈ 0.904` in all three Manager variants. Truncated final window flushed with actual length.

## Intra-slice safety projection Π_feasible (Gate 4)

Two-phase, always-feasible: Phase-1 severity-tier-descending protection (`N_req[k]=⌈C_req[sev_k]/C_PRB(SINR_k)⌉`, full units bps/[bps/PRB]=PRB); Phase-2 surplus by `score=N_req·(1+β·urgency)·softmax(w)`, largest-remainder integer projection. `Σ B_k = B_URLLC` exactly.

## C3 structural enforcement (Gate 3.2) — important nuance

`_feasible_rrm_cap` (`oran_env.py:652`) reserves `⌈10·1e6 / C_PRB@0dB⌉ = 38` PRB for eMBB at the conservative 0 dB rate and clips the Manager budget. Consequence: **C3 is structurally satisfied at SINR ≥ 0 dB for the fixed 10 Mbps floor** (feasibility oracle: eMBB ≥ 16.8 Mbps even at cell-edge worst case). The learned λ_C3 activates only in **deep fade (SINR < 0 dB)**, where the 0 dB-conservative reservation under-delivers; residual C3 violation is still measured from realized throughput. With floor 10 Mbps the effective ceiling stays `B_RRM_MAX=0.85`.
