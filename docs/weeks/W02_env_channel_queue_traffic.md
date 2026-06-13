# W02 — Env Modules I: Channel + Queue + Traffic

> **Pha**: GĐ A (code foundation) · **Status**: ✅ DONE · **Gate**: G1.1 — unit tests pass · **Deps**: W01/G0

## Đã xây
- `env/channel_model.py` — path-loss + LOS probability + shadow fading; Shannon capacity `C = η·PRB·B_PRB·log₂(1+SINR)` (SINR **tỉ số tuyến tính**, KHÔNG dB).
- `env/queue_model.py` — **M/G/1 Pollaczek–Khinchine** per slice, ổn định `ρ<0.9` margin.
- `env/traffic_gen.py` — URLLC (DENM/Vital/CAM) + eMBB (bystander) generators theo payload-size + arrival-rate.
- `tests/test_env_week2.py`.

## Sửa/scope (audit post-cleanup)
- ⚠️ Channel hiện code = UMa+UMi-SC hybrid → **sẽ thu về UMi single-cell 300m @ Bạch Mai** [3GPP TR 38.901] ở [W12](W12_pha1_radio_channel_capacity.md)/M2 (gỡ UMa + KNN Keangnam — D25, audit #11).
- ⚠️ Traffic chỉ **URLLC xe + eMBB bystander** (mMTC KHÔNG trong scope hiện tại); payload/rate ground ref ở [W14](W14_pha1_aoi_traffic.md)/M8.1b.

## Gate G1.1 ✅
- Shannon: linear-ratio conversion đúng, đơn điệu theo SINR; M/G/1 ổn định ρ<0.9; arrival-rate khớp spec; `pytest tests/test_env_week2.py` pass.

## Liên kết
- Capacity/channel → [W12](W12_pha1_radio_channel_capacity.md) (M1-M3); queue delay → [W13](W13_pha1_delay_reliability_qos.md) (M4.2, Kleinrock); traffic → [W14](W14_pha1_aoi_traffic.md) (M8).
