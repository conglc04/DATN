# W10 — Tuning + Multi-seed Prep

> **Pha**: GĐ A (code foundation) · **Status**: ✅ DONE · **Gate**: G3.3 — mitigations tested + budget locked · **Deps**: W09/G3

## Đã xây / verify
- `--worker-ent-coef` + `--manager-ent-coef` CLI flags (tách riêng Worker/Manager; audit 2026-06-16 — wired vào agent qua `ent_coef=`); ent_coef sweep + α_λ sweep (ổn định seed khó).
- Wall-clock per-run đo + logging spec finalize cho experiment đa-seed.

## Gate G3.3 ✅
- ≥1 orthogonal mitigation tested; budget multi-seed locked; tests pass.

## Liên kết
- Đây là mốc CUỐI của GĐ A (code foundation). **GĐ B (reference-disciplined + severity/SUMO) bắt đầu từ [W12](W12_pha1_radio_channel_capacity.md)**. [W11](W11_exp1_baseline_rwp_historical.md) (Exp1 RWP) = LỊCH SỬ, kết quả KHÔNG tái dùng.
