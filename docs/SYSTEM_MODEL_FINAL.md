# System Model — Final (pre-training closure)

> Baseline `audit-baseline-e565846-20260620`. Verified by independent oracles
> (`audit/closure_checks.py`, `feasibility_oracle.py`, `runtime_oracle.py`) and
> `tests/test_timescale_episode_semantics.py`. SoT = `env/oran_env.py` +
> `utils/config.py`.

## 1. Topology & mobility

- Single macro cell, **R_CELL = 1000 m**, gNB at local origin (0,0,h=10 m); 3GPP **UMa @ 3.5 GHz**.
- Mobility from **SUMO/OSM** pre-generated FCD traces (Bạch Mai MCI scenario), `env/sumo_mobility.py`. Legacy RWP bounce kept only for unit tests / non-SUMO (`oran_env.py:561`).
- Ambulance lifecycle masks: `entered_mask` (latches on cell entry), `arrived_mask` (latches at destination), **`active_mask = entered & ~arrived`** (`oran_env.py:684`, locked by `test_timescale_episode_semantics`). Out-of-cell vehicles still move in SUMO but contribute **no** reward/constraint (masked).
- Arrival: vehicle leaves FCD near destination within tolerance (`dist_to_destination`); dual-distance split (`dist_to_gNB` vs `dist_to_destination`) prevents false arrival from a route merely passing the gNB.
- **Episode** = first/configured vehicle entry → **all-K-arrived OR 400 s timeout** (`oran_env.py:792-795`). The 1 s PPO rollout chunk is NOT the episode (locked: a 2 s episode runs 200 Worker steps, not 100).

## 2. Radio / channel (`env/channel_model.py`)

- PRB total **273** (100 MHz, μ=1); `B_PRB = 360 kHz`.
- Path loss `pl_uma` (monotone ↑ in distance, verified); shadowing seeded; **interference = calibrated noise-rise margin** (`noise_plus_interference_dbm`) — a CALIBRATED assumption modelling reuse-1 macro, NOT a direct 3GPP value (single-cell preserved).
- Noise `N = −174 + 10log₁₀(B) + NF(7)` dBm; SINR dB↔linear `10^(x/10)`; clamped `[−10,40]` dB (config) / `[min,max]` per scenario.
- Capacity **`C_PRB = η·B_PRB·log₂(1+SINR_lin)`**, η=0.75 (link-adaptation / effective-rate proxy — BLER is **not** separately derated in the service rate; η absorbs MCS/coding overhead). Verified exactly at −5/0/2.7/10/20 dB.
- Channel randomness seeded per env and synchronized across solvers (same seed → same trace, `test_solver_equivalence`).

## 3. Queue / packet / AoI

- Per-ambulance URLLC **M/G/1** queue + one pooled eMBB queue (`env/queue_model.py`). PK delay `E[Dq]=λE[S²]/(2(1−ρ))`, service time augmented by `D_STOCH` (RLC/retx). Stability margin ρ<0.9.
- `D_e2e = D_DET + 1/μ + E[Dq] + D_FH + D_BH` (no MEC).
- Traffic: Poisson `ambulance_status` (F=1 consolidated stream, ~50 pkt/s, 400 B); eMBB bystander background (`bystander_traffic.py`).
- **AoI** `Δ(t)=t−U(t)`, LCFS + drop-old; resets only on successful update; one tracker per ambulance, reset on entry. Inactive vehicle generates no fake AoI/constraint (active-masked).

## 4. Timescales (Gate 2 — exact)

| layer | period | ratio | code |
|------|--------|-------|------|
| MAC TTI | 0.5 ms | — | `MAC_TTI_SEC` |
| Worker (xApp) | 10 ms | 20 MAC ticks | `MAC_TICKS_PER_WORKER=20` |
| Manager | 100 ms | 10 Worker steps | `WORKER_STEPS_PER_MANAGER=10` |
| PPO rollout chunk | 1 s | 100 Worker steps | (update only; no env reset) |

`γ_H = γ_L^W ≈ 0.904`. PPO update after the 1 s rollout does **not** reset SUMO/queue/AoI/severity/λ/masks/env (locked, `test_mutation_guards.m08`).

## 5. Hierarchy (Manager / Worker)

- **Manager** is the SOLE controller of inter-slice budget: `b_rrm = B_MIN + (B_MAX−B_MIN)·σ(a_H)`, `[B_MIN,B_MAX]=[0.05,0.85]`; `B_URLLC + B_eMBB = 273` exactly; new action every 100 ms; no other path writes `b_rrm` (locked, `m05`).
- **Worker (xApp)** splits `B_URLLC` among active ambulances only: K=1 → 1-dim no-op; K≥2 → **`K`-dim** `(ℓ_0..ℓ_{K−1})` (pure-RL, audit 2026-06-21 — NO β slot, was `(1+K)`-dim `(β, w₀..w_{K−1})`) → `softmax(ℓ)→w_k→PRB_k`. Worker **cannot** touch `b_rrm`/inter-slice. `Σ_k B_k = B_URLLC`, inactive → 0 PRB (locked, `m06/m17`).
- Two-tier safety clip: `b_rrm` clipped to `[max(B_MIN, feasible_floor), min(B_MAX, feasible_cap)]`; `feasible_cap` is derived from the C3 eMBB floor (`oran_env.py:652`), so eMBB is structurally protected at SINR ≥ 0 dB (see OPTIMIZATION_PROBLEM_FINAL §C3).
