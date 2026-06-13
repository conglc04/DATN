# W14 — Pha 1: Age of Information + Traffic (MEC đã GỠ)

> **Pha**: 1 · **Status**: 📅 PLANNED · **Gate**: micro-GATE 1C · **Nhóm**: M7–M8 (M9 MEC GỠ) · **Deps**: W13/1B

## M7 — Age of Information
- **M7.1** `Δ(t) = t − U(t)` (U = timestamp gói mới nhất) — ✅[`Kangwei Qi…[2024].pdf` §III; `Xianfu Chen…pdf` §II; `Zoubeir Mlika…[2022].pdf` §II] — [s] *(thay Kaul 2012 — vắng corpus)*
- **M7.2** Xấp xỉ LCFS (freshest-first) — ✅[`Kangwei Qi…[2024].pdf`; `Xianfu Chen…pdf`]
- **M7.3** `AoI_max` y khoa — 🔴 NEEDS-CALIBRATION. **Placeholder = 500 ms** (≈5×period SpO2/BP @10Hz, M8.1b; use-case "vital quá 5 chu kỳ ⟹ cũ với triage real-time" — declared, KHÔNG ref y khoa trực tiếp); **sensitivity {250, 500, 750} ms**. IEEE 11073 = chuẩn ngoài corpus → tải nếu dùng để thay placeholder.
- **M7.4** [AUDIT no-fake-vitals] `aoi_tracker.py` `AoIPacket{gen_time, deliver_time, payload_id}` → AoI thuần timestamp (point process), KHÔNG sinh giá trị sinh hiệu. AoI-traffic ⟂ NACA-S (2 lớp độc lập).
- **M7.5** `F = 4 streams/xe` (`oran_env.py:103-122`): {HR_agg, SpO2_agg, ECG_waveform, DENM} = NHÃN LOẠI luồng (payload type), KHÔNG giá trị sinh hiệu. **K=3 aggregation ĐÃ CHỐT** (audit #21/D24): `{AoI_worstnorm_k = max_s(AoI_s/AoI_max_s^φ), AoI_mean_k}` = **2 dims/xe** (KHÔNG F×K=12).

## M8 — Lưu lượng
- **M8.1** URLLC arrival Poisson λ (xe) — ✅[`Madyan Alsenwi…[2022].pdf` §II; `R. Sohaib…[2024].pdf` §III]
- **M8.1b** **payload + rate** per-stream (`config.py:181-201`): HR(100–500B,100Hz), SpO2/BP(200–400B,10Hz), ECG-waveform(1500B,CBR/VBR), DENM(50–100B sparse) — **tham số MẠNG** (size+rate), KHÔNG giá trị sinh hiệu. 🟡 cần ref size/rate (IEEE 11073 device rates / 3GPP V2X message size).
- **M8.2** eMBB nền M UEs (bystander) — ✅[`Weijian Zhou…pdf` §IV]
- **M8.3** `R_REF = 100 Mbps` (chuẩn hoá reward eMBB) — 🟡→[`Weijian Zhou…pdf` §IV]; sensitivity {50,100,200,300}

## M9 — MEC: ĐÃ GỠ HOÀN TOÀN (D23)
`D_MEC`/`f_MEC`/`C_FH` — vestigial (D_MEC không vào D_e2e). `mec_model.py` XÓA; `u_MEC` obs gỡ ([W18](W18_pha3_algorithm_code.md)/B0b). Comm+compute = future (Filali-style). **Verify** `C_FH` không dùng ngoài MEC trước khi xóa config (D_FH delay GIỮ — khác C_FH).

## ⟲ RÀ SOÁT M7–M8
URLLC∩eMBB user-set rời nhau (Alsenwi §II.A); λ [gói/s]; R_REF chỉ vào reward KHÔNG vào QoS target; AoI_max neo M8.1b period; F=4 nhãn-loại nhất quán no-fake-vitals.

## micro-GATE 1C
M8.3 + AoI_max có sensitivity plan; M7.1 đổi nguồn sang corpus (KHÔNG Kaul); MEC removal quyết định xong; 0 đại lượng vô-nhãn.

## Liên kết
Master plan PHẦN 11/W14 · `docs/04_data_flow.md` (AoI) · C4/C5 (AoI constraint) → [W17](W17_pha2_cmdp_formulation.md) · AoI obs dims → [W18](W18_pha3_algorithm_code.md).
