# Optimization Problem — Final (CMDP, pre-training closure)

> Baseline `audit-baseline-e565846-20260620`. Cross-checked against
> `EQUATION_TO_CODE_LEDGER.md` and the independent oracles. SoT = `utils/config.py`
> + `agents/lagrangian.py` + `env/oran_env.py`.

## Decision variables

- **Manager** `a_H ∈ ℝ` → `b_rrm = 0.05 + 0.80·σ(a_H)` → inter-slice PRB split (`B_URLLC=⌊b_rrm·273⌋`, `B_eMBB=273−B_URLLC`). Re-decided every 100 ms.
- **Worker** (K≥2) `a_L = (ℓ_0..ℓ_{K−1}) ∈ ℝ^K` (pure-RL, audit 2026-06-21 — NO β slot, was `ℝ^{1+K}` with `β=0.5+4.5·σ(a_L[0])`) → `softmax(ℓ) → w_k → PRB_k` intra-URLLC priority over active ambulances. K=1: trivial no-op.

## State

- **Worker obs** `(20 + 11K + F)`-dim (K=1,F=1 → 32; K=3 → 54): fixed block (ρ, HOL, PRB ratios, arrivals, BLER, severity_ref one-hot, λ_C3, anchor, n_bys, AoI mean/max) + per-amb 11-dim block (SINR, dist, speed, delay_norm, AoI_norm, sev_norm, λ_C1/C2/C4/C5, **active_mask_k**) + F AoI-stream. `active_mask_k ∈ {0,1} = entered_k & ~arrived_k` (explicit active flag — disambiguates inactive xe from active-empty-queue). λ overlaid by `overlay_lambda_local` (single source, all solvers).
- **Manager state** `s_H,t = [x_t, λ_t, ĝ_{t-1}]`, `(8 + 2·(4K+1))`-dim (audit 2026-06-23 — was `6+(4K+1)`): fixed 8-scalar block `[ρ_U, ρ_e, BLER, severity_ref_norm, severity_mean_norm, n_active_norm, AoI_mean, AoI_max]`, then **λ_global/LAMBDA_MAX** `(4K+1)` (long-run dual price, integral of past violations, post the last-completed window; **normalized to [0,1]** by the LAMBDA_MAX clip ceiling — audit 2026-06-24, fixes Manager-critic input-scale imbalance λ∈[0,10] vs fixed-block∈[0,1]; the dual ascent/penalty use the RAW `LambdaState.lambda_global`, not this obs copy) AND **g_hat = ĝ_{t-1}** `(4K+1)` (**kept RAW/signed**, symmetric around 0 — the sign/zero-crossing is load-bearing for the critic and must not be squashed by a one-sided ceiling) (the residual of the LAST COMPLETED Manager window — `LambdaState.get_deviation_hat()`, the SAME vector dual ascent consumed at the last `on_manager_step_end`; read BEFORE the Manager picks `b_rrm_t`, so it is the result of `b_rrm_{t-1}`, NEVER of the action about to be chosen — no future-information leak). K=1 → 18, K=3 → 34. Two fixes from a formulation critique: (1) without `ĝ_{t-1}` the Manager observed only λ (history), never the per-window residual `r_aug = r − Σλ_j·max(0,g_j)` is sensitive to — partial observability, the likely root cause of a negative-EV Manager critic. **Same-source proxy, not literal equality**: `g_hat` is the SIGNED window MEAN of `(c-d)/scale` (`accumulate()`); `r_aug` instead hinges (`max(0,·)`) that SAME per-tick deviation individually, before the SMDP-discounted sum the critic learns from — `mean(max(0,dev)) ≥ max(0,mean(dev))` (Jensen, `max(0,·)` convex) — so `g_hat` signals which constraint is under pressure and in which direction (same `c_vec`/`d_phi`/severity source as the reward), not the exact amount subtracted that window; (2) `severity_ref = max(severity_per_amb)` aliases K=3 states like `(5,1,1)` and `(5,5,5)` (same max, very different resource need) — `severity_mean_norm` (mean over **active** ambulances, falls back to `severity_ref_norm` when none are active) plus `n_active_norm = n_active/K` break the alias without adding K-dim state (permutation-invariant; per-amb ordering is already handled by the Worker's own per-amb λ/obs).
- Severity `severity_per_amb ∈ {1..5}^K` sampled **once at reset**, fixed for the mission (locked, `m09`). **`sev_ref := max(severity_per_amb)`** drives shared quantities (severity one-hot). **C3 floor is severity-independent.** **Reward has NO α_e** — severity differentiation is entirely via constraints.

## Objective (Gate 6)

```
max  E[ Σ_t  r_t ],   r_t = mean_{tick∈step} log(1 + R_eMBB,tick / R_REF),   R_REF = 100 Mbps
```
Single-term **pure** eMBB log-utility — **no α_e weight** (removed 2026-06-23). Severity differentiation is enforced ENTIRELY via constraints C1–C5 + λ dual ascent, not via reward weighting. Removing α_e eliminated a double-count: constraints already force higher b_rrm at high severity (large penalty if QoS violated), so α_e was redundant and obscured the Manager gradient (sev=5 had α_e=0.05 → reward ≈ 0).

**Reward is the MEAN over MAC ticks** (not the sum). The constraint `c_vec` is a per-tick MEAN (delay rate, violation rate, AoI), so the reward must share the same temporal basis for the augmented Lagrangian `r − Σλⱼ·gⱼ` to be balanced. A SUM-vs-MEAN mismatch (×20) made the eMBB reward gradient (~+4.2 when dropping b_rrm) swamp the constraint penalty (~+1.0 even at sev=5 maxed λ) → Manager starved URLLC. With the MEAN basis the per-step reward gain shrinks to ~+0.21 while the penalty stays ~+1.0 → NET negative at sev=5 → URLLC protected; at sev=1 (loose QoS, penalty=0) the Manager still frees budget for eMBB → correct severity ordering with NO hard floor (audit 2026-06-23).

**Why `sev_ref = max(·)` (explicit modeling choice, K≥2).** Shared scalar quantities (severity one-hot, C3 floor) must collapse the K per-ambulance severities to one. We take the **max** = the most-urgent patient on the cell. Each ambulance's **own** QoS is still enforced per-vehicle via its own `severity_per_amb[k]` in the C1/C2/C4/C5 constraints and per-amb λ; only the shared one-hot + C3 floor use `max`. Alternatives (mean / per-amb-weighted sum) would dilute the priority of the single critical patient and are explicitly rejected.

## Constraints (CMDP, `g_j ≤ 0` feasible)

| Cj | Type | Scope | Constraint | Threshold (sev1→5) | Subgradient | Grounding |
|----|------|-------|------------|---------------------|-------------|-----------|
| C1 | **MEAN** | per-amb | `E[D_e2e^k] ≤ D_max^{sev_k}` | 20/10/5/2/1 ms | Option-b (window) | TS 22.261 |
| C2 | **CHANCE** | per-amb | `P(D_e2e^k > D_max^{sev_k}) ≤ ε^{sev_k}` | 1e-3/1e-4/1e-4/1e-5/1e-5 | Option-a (cumulative) | TS 22.261 |
| C3 | **MEAN** | shared | `E[R_eMBB] ≥ R_min = 10 Mbps` (severity-independent) | 10 (all sev) | Option-b (window) | Alsenwi/Sohaib |
| C4 | **MEAN** | per-amb | `E[AoI_k] ≤ AoI_max^{sev_k}` | 1.0/0.5/0.2/0.1/0.1 s | Option-b (window) | declared |
| C5 | **CHANCE** | per-amb | `P(AoI_k > AoI_max^{sev_k}) ≤ ε_AoI^{sev_k}` | 1e-2/1e-3×4 | Option-a (cumulative) | declared |

**Type definitions**: MEAN = expectation over the trajectory (feasible when time-average ≤ budget); CHANCE = probability of exceedance ≤ ε (feasible when tail rate stays below threshold). C3 is a mean-throughput floor — it constrains E[R_eMBB], NOT P(R_eMBB < 10). Option-b (interval-window, N≈200, reset each Manager step) suits mean-type; Option-a (episode-cumulative, N grows) is mandatory for chance-type at ε ≤ 1e-4 — see `agents/lagrangian.py` docstring.

**Severity-tier granularity** (documented, not a bug): C1 fully distinguishes 5 levels (reward is pure eMBB utility without α_e — severity differentiation is entirely via constraints); C2 uses 3 standardized reliability tiers (making 5 would fabricate non-standard nines); C4 saturates at a 0.1 s freshness floor; C5 is near-binary (non-urgent vs urgent). See `config.py:SEVERITY_QOS` rationale.

**C2 implementation limitation (analytical simulation proxy)**: The formulation defines C2 as P(D_packet > D_max) ≤ ε — a per-packet chance constraint. The implementation uses M/G/1 Pollaczek–Khinchine expected delay (a queue-state metric) as proxy: `viol_k = E[D_e2e^k] > D_max` per MAC tick. This is monotonically correlated with the true per-packet tail probability but not equivalent at the boundary. True per-packet delay requires packet-level simulation (ns-3 — future work). Raw violation/sample counters (`c2_violation_count`, `c2_sample_count`) are exported in the env info dict for post-hoc audit. C5 (AoI-tail) does NOT have this limitation — `current_aoi(t)` is a true observation, not an estimate.

## Lagrangian (Gate 8/9)

- Constraint vector `c_vec`, threshold `d_phi`, multipliers `λ` all **`(4K+1)`-dim**, layout `[C1₀..C1_{K-1}, C2.., C4.., C5.., C3_shared]` (K=1→5, K=3→13).
- Normalized deviation `dev_j = (c_j − d_j)/scale_j`, `scale = [D_REF, 1, AoI_REF, 1, R_REF]`-blocked. Threshold subtracted **exactly once**.
- **Augmented reward** `r_aug = r − Σ_j λ_j·max(0, dev_j)` (hinge — a slack constraint contributes exactly 0, never a bonus; fixed 2026-06-22 bonus-masking audit) using the **pre-update** λ (the λ in the state); `λ_{t+1}` created only after the transition. Dual ascent (below) still uses the raw **signed** `dev_j` so λ can relax when a constraint is slack.
- **Dual ascent** `λ_j ← clip(λ_j + α_λ·ĝ_j, 0, Λ_max)`, `α_λ=2e-4`, `Λ_max=10`, `ĝ_j` = mean window deviation. Fixed-rate projected ascent — *inspired by* multi-timescale stochastic approximation; **no formal convergence claim**.
- **λ persistence (audit 2026-06-23)** — `λ_warm[sev]` is the PERSISTENT per-severity dual variable: at `on_episode_end` the full learned `λ_global` is saved (β_ema=1.0), and the next same-severity episode warm-starts from it, so λ accumulates monotonically toward the CMDP equilibrium across episodes. The `LAMBDA_WARM` constant table only seeds λ the first time a severity is seen per run; reset happens only at a new run/seed (new `LambdaState`). Fixes the starvation root cause: β_ema=0.05 (old) diluted accumulation 20× → λ_C2 pinned at λ_warm≈2.2 < equilibrium λ*≈4.0 (= eMBB-reward-gain / C2-residual ≈ 0.25/0.063) → penalty too weak → Manager starved URLLC. Note: accumulation rate is α_λ-bound (~1.2e-3/episode for C2 at sev5), so reaching λ* needs a long run (~1.5k sev5 episodes); the fix makes it monotonic (was flat).
- Active-time normalization: C1/C2/C4/C5 per-ambulance active-sample denominator; **C3 over total ticks** (slice-level, never active-masked). Inactive ambulance → c=0 (constraint trivially satisfied), dual relaxes ~1e-4/window (negligible, correct CMDP behavior).

## Manager SMDP (Gate 10)

`R_H = Σ_{i=0}^{W−1} γ_L^i · r_aug,i` over each 100 ms window — identical formula in PPO (`train.py:299`), TD3 & SAC (`train_offpolicy.py:259`). Bellman/GAE discount `γ_H = GAMMA_MANAGER ≈ 0.904` in all three Manager variants. Truncated final window flushed with actual length.

## Intra-slice split — pure-RL (Gate 4, audit 2026-06-21 — SUPERSEDES Π_feasible severity-tier projection)

`_prb_split_intra_slice` (reserve-first order, audit 2026-06-24): reserve `K_active·PRB_MIN_QOS` for every active ambulance FIRST, then split only the remainder `B_U−reserved` via `softmax(ℓ_k) → w_k → extra_k=floor(w_k·remainder)` + largest-remainder integer correction (`Σ B_k = B_URLLC` exactly). `PRB_k = PRB_MIN_QOS + extra_k`. Structural guarantee DUY NHẤT: budget conservation + flat anti-starvation floor `PRB_k≥PRB_MIN_QOS=1` per ACTIVE ambulance, held **by construction** (not just when the softmax happens to be balanced) — KHÔNG severity-tiered. Feasibility precondition `B_U≥K_active·PRB_MIN_QOS` (raises `ValueError` if violated; holds with wide margin under current bounds). Severity-awareness is FULLY LEARNED via obs (`severity_norm_k`, λ_C1/C2/C4/C5_k per-amb) + `r_aug` gradient, NOT a rule.

~~**Prior order (2026-06-21 → 2026-06-24, REMOVED)**: floor the full-budget proportional split (`PRB_k=floor(w_k·B_U)`) → force each entry up to `PRB_MIN_QOS` → rescale down on overflow. Bug: with extreme logit skew the overflow rescale could zero out an ambulance (e.g. raw logits `[10,−5,−5]`, `B_U=27` → `[26,1,0]`), violating the floor. Fixed by reserving the floor before any softmax split.~~

~~**Historical (2026-06-15 → 2026-06-21, REMOVED)**: two-phase always-feasible projection — Phase-1 severity-tier-descending protection (`N_req[k]=⌈C_req[sev_k]/C_PRB(SINR_k)⌉`, full units bps/[bps/PRB]=PRB); Phase-2 surplus by `score=N_req·(1+β·urgency)·softmax(w)`. No longer in code — see `EQUATION_TO_CODE_LEDGER.md` I2/I3 (⛔ SUPERSEDED).~~

## C3 structural enforcement (Gate 3.2) — important nuance

`_feasible_rrm_cap` (`oran_env.py:652`) reserves `⌈10·1e6 / C_PRB@0dB⌉ = 38` PRB for eMBB at the conservative 0 dB rate and clips the Manager budget. Consequence: **C3 is structurally satisfied at SINR ≥ 0 dB for the fixed 10 Mbps floor** (feasibility oracle: eMBB ≥ 16.8 Mbps even at cell-edge worst case). The learned λ_C3 activates only in **deep fade (SINR < 0 dB)**, where the 0 dB-conservative reservation under-delivers; residual C3 violation is still measured from realized throughput. With floor 10 Mbps the effective ceiling stays `B_RRM_MAX=0.85`.
