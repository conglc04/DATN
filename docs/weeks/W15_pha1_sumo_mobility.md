# W15 — Pha 1: Mobility SUMO/OSM Hà Nội (thay RWP)

> **Pha**: 1 · **Status**: ✅ DONE · **Gate**: micro-GATE 1D ✅ · **Nhóm**: M10 · **Build**: B3 · **Deps**: W12/1A (channel)

## M10 — Di động (SUMO + OSM, RWP BỎ HOÀN TOÀN)

- **M10.1** Network từ OSM (**Bạch Mai 5.91km×6.32km = kích thước thực sau netconvert**; origBoundary 105.805°E–105.873°E × 20.983°N–21.040°N; convBoundary 0–5906m × 0–6320m; trục Giải Phóng) qua `netconvert` — ✅[`traffic-modeling-with-sumo-a-tutorial…pdf` (Guastella 2023); `sumo-user.pdf` §netconvert]. ⚠️ serving cell = 1km UMa (W15-B2) ≪ map 5.91×6.32km.
- **M10.1b** **FIDELITY = Tầng 1 + density sweep** (LOCKED):
  - Traffic signals: `netconvert --tls.guess` + node `highway=traffic_signals` → `tlLogic` — ✅ free OSM [`sumo-user.pdf` §TLS]. KHÔNG claim khớp timing thật HN.
  - Speed-limit: giữ `maxspeed` tag — ✅ free [`sumo-user.pdf` §edge speed].
  - Background density: `randomTrips` {light=50/medium=200/heavy=500 xe nền} — 🔴 (ảnh hưởng tắc→tốc độ→SINR); KHÔNG calibrate đếm-xe thật.
  - **Route pool**: **6 bộ FCD** = {K=1, K=3} × {light, medium, heavy} — ✅ `data/sumo/density/bachmaiHN_mci_k{1,3}_{light,medium,heavy}.fcd.xml`. Mỗi episode chọn ngẫu nhiên 1 trong 3 density (theo K); route luôn bắt đầu t=0 (origin→BV Bạch Mai, không wrap).
  - Tầng 2 OD-synthetic (`od2trips`) = **FUTURE**; Tầng 3 (calibrate số đo thực HN) = **ngoài scope** (declare honest).
- **M10.2** **Kịch bản MCI hội tụ @ Bạch Mai** (D22): 3 xe **severity khác nhau** hội tụ **BV Bạch Mai** (trauma center). Route một chiều origin→BV; RL episode bắt đầu khi xe đầu tiên vào cell (R_CELL=1km), kết thúc khi cả 3 xe arrived (dist_to_destination ≤ ARRIVAL_RADIUS_M=15m) hoặc truncate 400s. Đích đến: `37370971#0` (cổng Giải Phóng, 44–54m từ gNB; đường nội viện không có trong OSM/SUMO). FCD traces 400s để amb_2 kịp đến (~372s). 🔴 tọa độ/tuyến = minh hoạ (OSM thật). ⚠️ **BỎ** "3 BV khác quận" (mâu thuẫn single-cell→handover).
- **M10.3** `sumo_mobility.py` đọc FCD; GPS→metric Haversine — ✅[`sumo-user.pdf` §FCD; Haversine well-known]
- **M10.4** Tích hợp: thay `_advance_ambulance_positions` (RWP) → SUMO trace mặc định; SINR từ vị trí thật (nối M2).

## ⟲ RÀ SOÁT M10
Timestep FCD khớp TTI sim; tọa độ trong bbox OSM (convBoundary 5906m×6320m); SINR từ M2 dùng đúng d từ FCD (KHÔNG còn RWP); 6 bộ route pool đủ (K=1,3 × light/medium/heavy); random trace_idx per episode → RL học policy tổng quát.

## micro-GATE 1D
SUMO trace chạy, dải SINR hợp lý (≈[−10,+30]dB NLOS-chi phối); validation = **realism** (KHÔNG "vs RWP" — RWP bỏ, audit FIX); scope/limit 3-tầng declared. Số RWP cũ KHÔNG tái dùng; solver sweep (W18–W23) chạy trên SUMO mobility. E3 (AoI LCFS/FCFS) / E4 (stress) → future work ([W24](W24_thesis_writing_defense.md)).

## Liên kết
Master plan PHẦN 11/W15 + M10.1b/M10.2 · `docs/03_architecture.md` (topology) · channel ← [W12](W12_pha1_radio_channel_capacity.md)/M2 · files `data/sumo/` ([docs/08](../08_implementation_notes.md)).
