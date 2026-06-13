# W01 — Foundation Setup

> **Pha**: GĐ A (code foundation) · **Status**: ✅ DONE · **Gate**: G0 — pytest pass, imports OK

## Đã xây
- Cấu trúc package: `env/ agents/ baselines/ experiments/ utils/ data/ tests/ figures/ checkpoints/ logs/`
- `utils/config.py` — **single source of truth**: `P_TOTAL=273` PRB, `PHASE_QOS`, `LAMBDA_WARM`, `ALPHA_LAMBDA_DUAL=1e-4`, `LAMBDA_MAX=10`, `WORKER_STEPS_PER_MANAGER=10`
- `utils/logger.py` (CSV/TensorBoard), `utils/metrics.py` (D_e2e breakdown, viol_rate, eMBB_tput, AoI)
- `train.py` (CLI stub), `tests/test_imports.py`, `requirements.txt` (torch≥2.6, numpy≥2.0, gymnasium 0.29)

## Gate G0 ✅
- `pytest tests/test_imports.py` pass; `assert P_TOTAL == 273` pass; `train.py --smoke-test` exit 0.

## Liên kết
- Hằng số vô tuyến (P_TOTAL, B_PRB) được **ground reference** ở [W12](W12_pha1_radio_channel_capacity.md) (M1, 3GPP TS 38.101-1 / TS 38.211).
- Master plan: PHẦN 9 (ARCHIVE).
