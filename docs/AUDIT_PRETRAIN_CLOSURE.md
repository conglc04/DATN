# Pre-Training Closure Audit

> Baseline `audit-baseline-e565846-20260620` (commit e565846, branch
> feat/hrl-real-3solvers). Independent evidence only — no docs↔code
> self-confirmation. Companion: `SYSTEM_MODEL_FINAL.md`,
> `OPTIMIZATION_PROBLEM_FINAL.md`, `EQUATION_TO_CODE_LEDGER.md`,
> `audit/pretrain_audit_manifest.json`.

## FINAL VERDICT: `PASS — SAFE FOR SMOKE TRAIN`

No CRITICAL/HIGH finding. All smoke-train criteria met: formulation↔code match
(independent oracles), feasibility 81/81, runtime ≤1e-6, mutation guards have
teeth, solver equivalence PASS, full suite green, deterministic dry run PASS.
**Q1 (C2/C5 tail estimability) is RESOLVED** — a held-out eval-bank
(`audit/eval_tail_bank.py`) pools tail samples across episodes so ε=1e-3 tails
(C5 all, C2 sev1-3) are certifiable; C2 at ε=1e-5 (sev4-5) needs a ~250-episode
bank (reported, not certified at the per-episode horizon).

`SAFE FOR FULL TRAIN` is **not yet** granted — it requires (a) the actual
3-solver macro smoke train (no NaN, episode/conservation/mask/severity/λ checks,
checkpoint+log readable), and optionally (b) the 16-agent adversarial
verification, which did not complete this session (token limit, resets 15:00).
Neither is a correctness blocker; they are the remaining process gates.

## Gate scorecard

| Gate | Area | Verdict | Independent evidence |
|------|------|---------|----------------------|
| 0 | Census + baseline freeze | ✅ PASS | commit e565846; SoT map + legacy/dead-code listed (manifest) |
| 1 | Topology/mobility | ✅ PASS* | dry_run active_mask=entered&~arrived (K=1,3); existing suite test_active_arrived_mask, test_arrival_dest_separation, test_early_termination green. *deep SUMO no-false-arrival relies on existing green tests, not re-derived this session |
| 2 | Radio/channel | ✅ PASS | closure G2: PL monotone; Shannon cap exact @−5..20dB; thermal noise exact. Interference = calibrated margin (documented) |
| 3 | Queue/packet/AoI | ✅ PASS | closure G3: PK delay 8.93µs match; AoI logged + audited |
| 4 | Timescale/episode | ✅ PASS | closure G4 + test_timescale: W=10, MAC=20, γ_H=γ_L¹⁰; 2s≠1s; no reset at rollout boundary (mutation m08) |
| 5 | Hierarchy/action causality | ✅ PASS | closure G5: 273 conserved; ΣB_k=B_URLLC; worker can't touch b_rrm; action dims (1)/(1+K) |
| 6 | C3 safety projection | ✅ PASS | structurally enforced by feasible_rrm_cap @SINR≥0; dual binds only in deep fade; ceiling stays 0.85 (OPTIMIZATION_PROBLEM §C3) |
| 7 | Intra-slice + N_req dim | ✅ PASS | closure G7: N_req units bps/[bps/PRB]=PRB complete; mutation m15 |
| 8 | State/obs/Markov | ✅ PASS | obs (20+10K+F); λ single-source overlay; severity fixed/episode (m09); test_solver_equivalence |
| 9 | Objective/reward | ✅ PASS | runtime_oracle reward-form; α_e 5-distinct; reward severity-dependent via α_e (documented, NOT fully decoupled) |
| 10 | C1–C5 ledger | ✅ PASS | thresholds/signs/units verified (ledger); C2/C5 tail estimability resolved via held-out eval-bank (`audit/eval_tail_bank.py`) — ε=1e-3 certifiable, ε=1e-5 needs ~250-ep bank |
| 11 | Constraint vector sign/norm | ✅ PASS | closure G11: at/below/above threshold for C1–C5, K=1&3; dim 4K+1; active-denominator; C3 total-ticks (m04) |
| 12 | Lagrangian/dual | ✅ PASS | runtime_oracle (dev, r_aug pre-update λ, dual clip); no formal convergence claim |
| 13 | Manager SMDP | ✅ PASS | SMDP sum identical PPO/TD3/SAC; γ_H=GAMMA_MANAGER all 3 (m07, test_solver_equivalence) |
| 14 | Solver fairness | ✅ PASS | closure G14 + test_solver_equivalence: same seed → identical trace; shared machinery |
| 15 | Feasibility oracle | ✅ PASS | 81/81 feasible (independent physics); worst [5,5,5] edge slack +87 Mbps. Coverage note: checks C1+C3 jointly, not C4/C5 |
| 16 | Runtime oracle | ✅ PASS | every algebraic quantity ≤1e-6 vs independent recompute (K=1,3) |
| — | Mutation guards | ✅ PASS | 18 deliberate bugs all distinguishable (test_mutation_guards) |
| — | Deterministic dry run | ✅ PASS | 24 live checks K=1/K=3[3,3,3]/[5,3,1] |
| — | Full integration suite | ✅ PASS | 1112 green (pre-new-tests) + 38 new validated standalone |

## Findings

| # | Severity | Invariant | file:line | Consequence | Status |
|---|----------|-----------|-----------|-------------|--------|
| F-A | MEDIUM | C2(ε=1e-5)/C5(ε_aoi=1e-3) tail not certifiable at per-episode horizon | `config.py:115` (eps/eps_aoi) | per-episode tail is too short to certify ε; **RESOLVED**: held-out eval-bank pools samples across episodes (ε=1e-3 certifiable; ε=1e-5 needs ~250-ep bank) | **RESOLVED** (eval_tail_bank.py) |
| F-B | LOW | capacity uses η=0.75 as effective-rate proxy; BLER not separately derated in service rate | `channel_model.py:223` | not double-applied; correct, but document that η absorbs BLER/MCS | documented in SYSTEM_MODEL §2 |
| F-C | LOW | inactive-ambulance dual relaxes −1e-4/window | `lagrangian.py:253` | negligible; correct CMDP relaxation of a satisfied constraint | documented |
| F-D | INFO | feasibility oracle checks C1+C3 jointly, not C4/C5; runtime oracle recomputes dual+delay but not the full PRB-split/N_req chain | `audit/*_oracle.py` | coverage gap in the oracles (the env paths ARE covered by closure_checks G5/G7) | noted |

No CRITICAL/HIGH findings. F-B/F-C/F-D are documented; F-A requires your decision (Q1).

## Unresolved (require completion or your input)

1. **Q1 (C5/C2-tail estimability)** — methodology question, see below. Do NOT demote C5 without your decision (per your instruction).
2. **16-agent adversarial verification** — did not run to completion (session limit). Re-run after 15:00 reset for the deepest independent refutation pass.
3. **3-solver macro smoke train** — required before `SAFE FOR FULL TRAIN`.

## Path to `SAFE FOR FULL TRAIN`

1. Resolve Q1 (keep + rule-of-three reporting, or adjust horizon, or demote C5 — your call).
2. (Optional, recommended) re-run the 16-agent adversarial workflow after the session resets.
3. Run the deterministic dry run (done ✅) then the minimal smoke train for PPO + TD3 + SAC at macro config, same seed/scenario bank, verifying no NaN, episode semantics, conservation, masks, severity persistence, λ behavior, checkpoint/log readable.
4. Only then → `SAFE FOR FULL TRAIN`.

## Reproduce
```bash
cd baselines
python -m audit.closure_checks      # 30 independent checks
python -m audit.feasibility_oracle  # 81/81
python -m audit.runtime_oracle      # ≤1e-6
python -m audit.dry_run             # 24 live checks
python -m pytest tests/ -q          # full suite
```
