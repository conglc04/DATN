# W12 — Pha 1: Radio + Channel (UMi Bạch Mai) + Capacity

> **Pha**: 1 (Mô hình hệ thống) · **Status**: 📅 PLANNED · **Gate**: micro-GATE 1A · **Nhóm**: M1–M3
> **Build**: B1 (ground references `config.py` + `REFERENCE_MAP`). Mọi đại lượng phải có nhãn ✅/🟡/🔴.

## M1 — Tham số vô tuyến (per-cell)
- **M1.1** `P_TOTAL = 273 PRB` @100MHz/30kHz SCS — ✅[3GPP TS 38.101-1 Table 5.3.2-1] — [PRB]
- **M1.2** `B_PRB = 360 kHz` (12×30kHz) — ✅[3GPP TS 38.211 §4.4.4] — [Hz]
- **M1.3** `P_tx` — **tách UL/DL (audit #10)**: telemetry xe→mạng = **UPLINK**, `P_tx^UE = 23 dBm` ✅[TS 38.101-1 §6.2.1] (link CHÍNH của URLLC). eMBB→bystander = DOWNLINK `P_tx^gNB` ~43–49 dBm ✅[TS 38.104].
- **M1.4** `N0 = −174 dBm/Hz` — ✅[kT well-known] — [dBm/Hz]
- **M1.5** `η = 0.75` — hiệu suất phổ thực dụng — 🟡→[`Hyoungju Ji…[2017].pdf` §II-B]; sensitivity {0.6,0.75,0.9}
- **M1.6** `NF = 7 dB` — 🟡→[`Zexian Li…[2018].pdf` §III] / [3GPP TR 38.101-4]

## M2 — Kênh lan truyền (UMi single-cell @ Bạch Mai)
- **M2.0** Serving cell CỐ ĐỊNH: gNB @ ổ-MCI cạnh **BV Bạch Mai (Giải Phóng)** 🔴 (tọa độ OSM, declare). **R_cell = 300m — GIÁ TRỊ ĐƠN, KHÔNG SWEEP** (D25). Tiêu chí đặt gNB: trên **đoạn route cả 3 xe cùng qua** (convergence segment), KHÔNG sát cổng BV/rìa map; verify cả 3 quỹ đạo nằm trong 300m đồng thời trong cửa sổ contention (cùng [W15](W15_pha1_sumo_mobility.md)/M10.2).
- **M2.1** Path-loss UMi LOS/NLOS — ✅[3GPP TR 38.901 §7.4.1 Table 7.4.1-1] — [dB]
- **M2.2** P_LOS(d) — ✅[3GPP TR 38.901 §7.4.2 Table 7.4.2-1]
- **M2.3** Shadow fading σ_SF — ✅[3GPP TR 38.901 §7.4.1] — [dB]
- **M2.4** KNN site-specific — 🔴 (**GỠ "Keangnam"** — sai site; nếu dùng phải lấy đúng site Bạch Mai, HOẶC UMi generic). Optional, KHÔNG bắt buộc kết quả chính.
- **VERIFY 2026-06-12** (từ `channel_model.py`, P_tx=23dBm, N=−111.44dBm @360kHz/NF=7dB): NLOS chi phối (P_LOS@300m≈6%); SINR NLOS @200m≈+19dB, @300m≈+13dB — trong [−10,+30]dB ✓. LOS hiếm (6–9%) cho SINR>+30dB = hiện tượng vật lý thật, declare 🔴, KHÔNG do 300m.

## M3 — Dung lượng
- **M3.1** `C_k = η·PRB_k·B_PRB·log₂(1+SINR_k)` — ✅[Shannon–Hartley; η `Hyoungju Ji 2017` §II-B; MCS-map TS 38.214 §5.1.3] — [bit/s]. SINR = **tỉ số TUYẾN TÍNH** (KHÔNG dB); bandwidth=PRB_k·B_PRB; **single-cell + PRB trực giao ⟹ I≈0 ⟹ thực chất SNR** (declare; nếu thêm inter-cell → đổi tên + mô hình nhiễu).

## ⟲ RÀ SOÁT M1–M3
dBm↔mW nhất quán; N0+NF cộng đúng miền dB; P_TOTAL khớp numerology (273×360kHz≈98.28MHz ✓); SINR(d) đơn điệu giảm; SINR linear trong log₂; C=0 khi PRB=0; capacity capped by SINR (cơ sở channel-infeasibility C6); d_k ≤ R_cell=300m.

## micro-GATE 1A
M1.5, M1.6 hết 🟡 (đã gắn file/chuẩn); 0 đại lượng vô-nhãn ở M1–M3.

## Liên kết
Master plan PHẦN 11/W12 · `docs/03_architecture.md` (channel) · capacity → [W13](W13_pha1_delay_reliability_qos.md) (D_e2e) · cell/route → [W15](W15_pha1_sumo_mobility.md) (M10).
