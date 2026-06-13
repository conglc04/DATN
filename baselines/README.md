# O-RAN Ambulance Slicing — PPO / TD3 / SAC

**Severity-Aware Intra-slice Scheduling cho 5G O-RAN (Hà Nội, single-cell UMi @ Bạch Mai)**

Implementation reference cho luận văn: O-RAN URLLC slicing với 5-phase ambulance FSM + severity-aware intra-slice + CMDP-Lagrangian constraints + closed-form Π_feasible safety. Bài toán tối ưu được giải bằng **3 solver ngang hàng (PPO, TD3, SAC)** — KHÔNG đề cao thuật toán nào.

Toàn bộ specs nằm trong `../docs/` (11 file). File này chỉ là quick-start.

## Setup

```bash
# Tạo virtualenv (Python ≥3.10)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt
pip install -e .                 # dev mode

# Verify
python -c "from utils.config import P_TOTAL, PHASE_QOS; print(f'P_total={P_TOTAL}, PHASE3.D_max={PHASE_QOS[3][\"D_max\"]*1e3}ms')"
python train.py --smoke-test --seed 42
pytest tests/ -v
```

## Cấu trúc

```
baselines/          # repo code root (đổi tên từ pa_chrl_ppo/)
├── env/            # Simulator O-RAN (channel, queue, traffic, phase, AoI) — MEC/vital gỡ
├── agents/         # ppo_core, manager/worker (PPO HRL), td3_agent, sac_agent, lagrangian
├── solvers/        # 3 solver ngang hàng (PPO via train_ppo, td3.py, sac.py) + ablation variants
├── experiments/    # sweep W18-W23 → Table I/II + stats_analysis (Holm-Bonferroni)
├── utils/          # config / logger / metrics
├── data/           # synthetic traces + Hanoi calibration (placeholder)
├── tests/          # unit tests
└── train.py        # main training entry point (--algo ppo|td3|sac)
```

## Lịch trình 16 tuần

Xem `../docs/09_execution_plan.md` cho deliverables từng tuần.

| Phase | Tuần | Module chính |
|---|---|---|
| P1 Foundation | 1-2 | Setup + env modules cơ bản |
| P2 Simulator | 3-4 | Phase/AoI + ORANEnv (MEC gỡ) |
| P3 Solvers | 5-6 | 3 solver (PPO/TD3/SAC) + ablation variants |
| P4 Solvers | 8-10 | Worker/Manager + CMDP-Lagrangian (Π_feasible safety) |
| P5 Experiments | 11-13 | Exp1-Exp8 core |
| P6 Advanced | 14 | Exp10/11 + statistical analysis |
| P7 Writing | 15-16 | IEEE TWC manuscript |

## Tham chiếu nhanh

- Architecture spec: `../docs/03_architecture.md`
- Action space (RRMPolicyRatio): `../docs/07_api_spec.md`
- CMDP 5 constraints: `../docs/02_requirements.md#cmdp-constraints`
- Code tree chi tiết: `../docs/08_implementation_notes.md#code-tree`
- Algorithm 1 pseudocode: `../docs/08_implementation_notes.md#pseudocode`

## License

Internal research code. Not for distribution.
