# 08 — Implementation Notes

> Repo code root: `/home/cong/Desktop/USB_BACKUP/Do-an/baselines/` (Linux; đổi tên từ `pa_chrl_ppo/`). Path Windows `f:\Do an\` cũ = bỏ.

## Code map
```
baselines/                       # repo code root (đổi tên từ pa_chrl_ppo/)
├── env/
│   ├── channel_model.py      # UMi 3GPP TR 38.901 (single-cell 300m Bạch Mai)
│   ├── queue_model.py        # M/G/1 Pollaczek–Khinchine
│   ├── traffic_gen.py        # URLLC (F=4) + eMBB bystander, payload/rate
│   ├── phase_detector.py     # 5-phase FSM (explicit signaling)
│   ├── aoi_tracker.py        # AoI LCFS+drop (timestamp only, no vitals)
│   ├── naca_severity.py      # [MỚI, B2 — chưa tạo] NACA-S 6-state exogenous
│   ├── sumo_mobility.py      # [MỚI, B3 — chưa tạo] SUMO FCD trace reader (GPS→metric)
│   └── oran_env.py           # Gym ORANEnv (obs K=1=33; K≥2: +severity, +β, intra-slice Option B)
├── agents/                   # ppo_core, manager_agent (action 1-dim), worker_agent, lagrangian, td3_agent, sac_agent
├── solvers/                  # 3 solver ngang hàng: PPO (= train_ppo + agents/) + td3.py + sac.py; + ablation variants
├── experiments/              # sweep W18-W23 → Table I/II (E3/E4 = future work, D26); stats_analysis (Holm-Bonferroni)
├── data/sumo/                # [MỚI] hanoi_bachmai.osm, *.net.xml, ambulance_routes.xml, *.fcd.xml
└── utils/config.py           # single source: P_TOTAL=273, PHASE_QOS, LAMBDA_WARM, ALPHA_LAMBDA_DUAL=1e-4, LAMBDA_MAX=10, WORKER_STEPS_PER_MANAGER=10
```

## Removals (B0/B0b — ✅ ĐÃ THỰC THI 2026-06-13)
> Loại khỏi **scope/design** VÀ khỏi **code** (đã xóa thật, test xanh). obs K=1: 40 → **33**.
- ✅ **LSTM** (D10): obs −6; `LSTM_*` config gỡ.
- ✅ **MEC** (D23): `env/mec_model.py` XÓA; `MECServer`/`u_MEC` obs (−1)/`F_MEC` gỡ; Manager action 2→1 (bỏ f_MEC). D_FH/D_BH delay GIỮ.
- ✅ **vital_simulator.py** XÓA (sinh vitals giả). GIỮ telemetry stream HR/SpO2/ECG/DENM (traffic_gen).
- ✅ **β_qp / NSF distillation / nsf.py / LR_NSF** XÓA: safety filter → closed-form `Π_feasible` (Duchi simplex + isotonic) — CHƯA implement code (W18/B5).
- ✅ **B3-RCPO** XÓA hoàn toàn (3 solver = PPO/TD3/SAC).
- → obs K=1: 40 → **33** (`assert observation_space.shape==(33,)`); K=3 = **58** (sau B5).

## Code changes K≥2 (B5)
`oran_env.py`: obs +severity_k +AoI_worstnorm/mean +λ_C{1,2,4,5}^k; action +β (squash [β_min,β_max]); intra-slice PRB split Option B `b=max(κB_U/K, PRB_min^QoS)`; phase-event severity MAP. `lagrangian.py`: C6 per-pair (K≥2). `cell_radius_m`: 200→**300** (D25).

## KHÔNG đổi (W11 backward-compat history)
`train.py` (K=1 default), `run_30runs.py`, `stats_analysis.py` — nhưng số RWP cũ KHÔNG tái dùng (sweep W18–W23 chạy lại trên SUMO mobility).

## Cross-reference
[03](03_architecture.md) · [05](05_agent_workflow.md) · [07](07_api_spec.md) · [weeks/](weeks/README.md) (build B0-B9).
