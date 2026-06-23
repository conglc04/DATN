# W14 — Pha 1: Age of Information + Traffic (MEC đã GỠ)

> **Pha**: 1 · **Status**: ✅ DONE · **Gate**: micro-GATE 1C ✅ · **Nhóm**: M7–M8 (M9 MEC GỠ) · **Deps**: W13/1B ✅

## M7 — Age of Information
- **M7.1** `Δ(t) = t − U(t)` (U = timestamp gói mới nhất) — ✅[`Kangwei Qi…[2024].pdf` §III; `Xianfu Chen…pdf` §II; `Zoubeir Mlika…[2022].pdf` §II] — [s] *(thay Kaul 2012 — vắng corpus)*
- **M7.2** Xấp xỉ LCFS (freshest-first) — ✅[`Kangwei Qi…[2024].pdf`; `Xianfu Chen…pdf`]
- **M7.3** `AoI_max` y khoa — 🔴 NEEDS-CALIBRATION. **Placeholder = 500 ms** (≈5×period SpO2/BP @10Hz, M8.1b; use-case "vital quá 5 chu kỳ ⟹ cũ với triage real-time" — declared, KHÔNG ref y khoa trực tiếp); **sensitivity {250, 500, 750} ms**. IEEE 11073 = chuẩn ngoài corpus → tải nếu dùng để thay placeholder.
- **M7.4** [AUDIT no-fake-vitals] `aoi_tracker.py` `AoIPacket{gen_time, deliver_time, payload_id}` → AoI thuần timestamp (point process), KHÔNG sinh giá trị sinh hiệu. AoI-traffic ⟂ severity (ATS) (2 lớp độc lập).
- **M7.5** `F = 1 stream/xe` (`oran_env.py:DEFAULT_AOI_STREAMS`, **2026-06-14 stream consolidation**): `ambulance_status` = NHÃN LOẠI luồng tổng hợp duy nhất (payload type), KHÔNG giá trị sinh hiệu — gộp 4 luồng cũ {HR_agg, SpO2_agg, ECG_waveform, DENM} (KHÔNG averaging tham số, mỗi URLLC end-device 1 luồng task/traffic). **Per-amb (∀K) aggregation ĐÃ CHỐT** (audit #21/D24): `AoI_norm_k = AoI_k/AoI_max^{sev_k}` = **1 dim/xe** (offset `AMB_AOI_NORM_OFFSET` trong khối 10-dim per-amb, [08_implementation_notes.md](08_implementation_notes.md)) — F=1 ⟹ "worst" trùng giá trị duy nhất; công thức tổng quát cho F>1 là `max_s(AoI_s/AoI_max_s^{sev_k})`.

## M8 — Lưu lượng
- **M8.1** URLLC arrival Poisson λ (xe) — ✅[`Madyan Alsenwi…[2022].pdf` §II; `R. Sohaib…[2024].pdf` §III]
- **M8.1b** **payload + rate** (`TRAFFIC_CLASSES` `config.py:176-201`, dùng cho `traffic_gen.py`/`bystander_traffic.py` — KHÔNG phải nhãn AoI stream): `URLLC_C1_DENM`(300–800B, event_burst), `URLLC_C2_VITAL`(100–500B, 100Hz periodic), `URLLC_C3_CAM`(200–400B, 10Hz periodic) — **tham số MẠNG** (size+rate), KHÔNG giá trị sinh hiệu. AoI stream `ambulance_status` (`oran_env.py: urllc_arrival_rate=50.0`, `urllc_packet_bits=400B`) là tham số queue/PRB riêng, **KHÔNG đổi** theo F=4→F=1 consolidation (2026-06-14) — xem [08_implementation_notes.md](08_implementation_notes.md#f4f1-stream-consolidation--2026-06-14--1-luồng-ambulance_statusxe). 🟡 cần ref size/rate (IEEE 11073 device rates / 3GPP V2X message size).
- **M8.2** eMBB nền M UEs (bystander) — ✅[`Weijian Zhou…pdf` §IV]
- **M8.3** `R_REF = 100 Mbps` (chuẩn hoá reward eMBB) — 🟡→[`Weijian Zhou…pdf` §IV]; sensitivity {50,100,200,300}

## M9 — MEC: ĐÃ GỠ HOÀN TOÀN (D23)
`D_MEC`/`f_MEC`/`C_FH` — vestigial (D_MEC không vào D_e2e). `mec_model.py` XÓA; `u_MEC` obs gỡ ([W18](W18_pha3_algorithm_code.md)/B0b). Comm+compute = future (Filali-style). **Verify** `C_FH` không dùng ngoài MEC trước khi xóa config (D_FH delay GIỮ — khác C_FH).

## ⟲ RÀ SOÁT M7–M8
URLLC∩eMBB user-set rời nhau (Alsenwi §II.A); λ [gói/s]; R_REF chỉ vào reward KHÔNG vào QoS target; AoI_max neo M8.1b period; F=1 nhãn-loại (`ambulance_status`, gộp từ F=4 cũ) nhất quán no-fake-vitals.

## micro-GATE 1C
M8.3 + AoI_max có sensitivity plan; M7.1 đổi nguồn sang corpus (KHÔNG Kaul); MEC removal quyết định xong; 0 đại lượng vô-nhãn.

## Liên kết
Master plan PHẦN 11/W14 · `docs/04_data_flow.md` (AoI) · C4/C5 (AoI constraint) → [W17](W17_pha2_cmdp_formulation.md) · AoI obs dims → [W18](W18_pha3_algorithm_code.md).
