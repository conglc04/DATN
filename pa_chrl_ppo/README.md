# PA-CHRL-PPO

**Phase-Aware Constrained Hierarchical RL with PPO for O-RAN Ambulance Slicing (Hanoi)**

Implementation reference cho luận văn nghiên cứu O-RAN URLLC slicing với 5-phase ambulance FSM, CMDP-Lagrangian constraints, NSF safety filter, và multi-task LSTM QoS predictor.

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
pa_chrl_ppo/
├── env/            # Simulator O-RAN (channel, queue, traffic, phase, AoI, MEC)
├── agents/         # PA-CHRL-PPO + safety (NSF) + LSTM
├── baselines/      # Static 50/50 + B2 HRL-PPO + B3 RCPO + 3 ablation variants
├── experiments/    # Exp1-Exp11 + statistical analysis
├── utils/          # config / logger / metrics
├── data/           # synthetic traces (Track A) + Hanoi calibration (Track B)
├── tests/          # unit tests
├── figures/        # output figures
├── checkpoints/    # model weights
├── logs/           # training logs
└── train.py        # main training entry point
```

## Lịch trình 16 tuần

Xem `../docs/09_execution_plan.md` cho deliverables từng tuần.

| Phase | Tuần | Module chính |
|---|---|---|
| P1 Foundation | 1-2 | Setup + env modules cơ bản |
| P2 Simulator | 3-4 | Phase/AoI/MEC + ORANEnv |
| P3 Baselines | 5-6 | 3 baselines + 3 ablation variants |
| P4 PA-CHRL-PPO | 8-10 | Worker/Manager/CMDP + NSF + LSTM |
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
