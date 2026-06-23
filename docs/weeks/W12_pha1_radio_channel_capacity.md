# W12 — Pha 1: Radio + Channel (UMa Bạch Mai) + Capacity

> **Pha**: 1 (Mô hình hệ thống) · **Status**: ✅ DONE · **Gate**: micro-GATE 1A ✅ · **Nhóm**: M1–M3
> **Build**: B1 (ground references `config.py` + `REFERENCE_MAP`). Mọi đại lượng phải có nhãn ✅/🟡/🔴.

## M1 — Tham số vô tuyến (per-cell)
- **M1.1** `P_TOTAL = 273 PRB` @100MHz/30kHz SCS — ✅[3GPP TS 38.101-1 Table 5.3.2-1] — [PRB]
- **M1.2** `B_PRB = 360 kHz` (12×30kHz) — ✅[3GPP TS 38.211 §4.4.4] — [Hz]
- **M1.3** `P_tx` — **tách UL/DL (audit #10)**: telemetry xe→mạng = **UPLINK**, `P_tx^UE = 23 dBm` ✅[TS 38.101-1 §6.2.1] (link CHÍNH của URLLC). eMBB→bystander = DOWNLINK `P_tx^gNB` ~43–49 dBm ✅[TS 38.104].
- **M1.4** `N0 = −174 dBm/Hz` — ✅[kT well-known] — [dBm/Hz]
- **M1.5** `η = 0.75` — hiệu suất phổ thực dụng — 🟡→[`Hyoungju Ji…[2017].pdf` §II-B]; sensitivity {0.6,0.75,0.9}
- **M1.6** `NF = 7 dB` — 🟡→[`Zexian Li…[2018].pdf` §III] / [3GPP TR 38.101-4]

## M2 — Kênh lan truyền (UMa single-cell 1km @ Bạch Mai)
- **M2.0** Serving cell CỐ ĐỊNH: gNB @ ổ-MCI cạnh **BV Bạch Mai (Giải Phóng)** 🔴 (tọa độ OSM, declare). **R_cell = 1km — UMa macro, GIÁ TRỊ ĐƠN, KHÔNG SWEEP** (D25, W15-B2 2026-06-18). Tiêu chí đặt gNB: trên **đoạn route cả 3 xe cùng qua** (convergence segment), KHÔNG sát cổng BV/rìa map; verify cả 3 quỹ đạo nằm trong 1km đồng thời trong cửa sổ contention (cùng [W15](W15_pha1_sumo_mobility.md)/M10.2).
- **M2.1** Path-loss **UMa** `pl_uma(d)=28+22·log10(d)+20·log10(f_c)` — ✅[3GPP TR 38.901 §7.4.1] — [dB]. *(LOS/NLOS branch + P_LOS = legacy micro/UMi; UMa dùng 1 công thức.)*
- **M2.2** Interference margin **−86 dBm/PRB** (noise-rise floor mô phỏng interference-limited macro; KHÔNG sim gNB láng giềng) — 🔴 declared (W15-B2 calibration sweet spot)
- **M2.3** Shadow fading σ_SF (UMa=4 dB) — ✅[3GPP TR 38.901 §7.4.1] — [dB]
- **M2.4** KNN site-specific — 🔴 (**GỠ "Keangnam"** — sai site; nếu dùng phải lấy đúng site Bạch Mai, HOẶC UMa generic). Optional, KHÔNG bắt buộc kết quả chính.
- **VERIFY 2026-06-19** (từ `channel_model.py` `pl_uma` + `macro_mission_config`, tx_per_prb=21.6dBm = 46−10log10(273), n_eff=−86dBm): SINR **+2.7dB @1km cell-edge**, +18.1dB @200m, +31.4dB @50m — gradient có ý nghĩa khắp cell, clamp [−15,+40]dB. Khớp đúng `test_macro_calibration` (2.7dB@1km). *(Số UMi@300m cũ +13dB đã thay; env trước đó chạy UMi-clamp −15dB @1km — audit FIX C2.)*

## M3 — Dung lượng
- **M3.1** `C_k = η·PRB_k·B_PRB·log₂(1+SINR_k)` — ✅[Shannon–Hartley; η `Hyoungju Ji 2017` §II-B; MCS-map TS 38.214 §5.1.3] — [bit/s]. SINR = **tỉ số TUYẾN TÍNH** (KHÔNG dB); bandwidth=PRB_k·B_PRB; macro sweep **interference-limited** (I≠0 qua noise-rise floor −86 dBm/PRB, KHÔNG sim gNB láng giềng → vẫn single-cell). *(Legacy micro: I≈0 ⟹ SNR.)*

## ⟲ RÀ SOÁT M1–M3
dBm↔mW nhất quán; N0+NF cộng đúng miền dB; P_TOTAL khớp numerology (273×360kHz≈98.28MHz ✓); SINR(d) đơn điệu giảm; SINR linear trong log₂; C=0 khi PRB=0; capacity capped by SINR (cơ sở channel-infeasibility C6); d_k ≤ R_cell=1km.

## micro-GATE 1A
M1.5, M1.6 hết 🟡 (đã gắn file/chuẩn); 0 đại lượng vô-nhãn ở M1–M3.

## Liên kết
Master plan PHẦN 11/W12 · `docs/03_architecture.md` (channel) · capacity → [W13](W13_pha1_delay_reliability_qos.md) (D_e2e) · cell/route → [W15](W15_pha1_sumo_mobility.md) (M10).
