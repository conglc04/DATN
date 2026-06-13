# W15 — Pha 1: Mobility SUMO/OSM Hà Nội (thay RWP)

> **Pha**: 1 · **Status**: 📅 PLANNED · **Gate**: micro-GATE 1D · **Nhóm**: M10 · **Build**: B3 · **Deps**: W12/1A (channel)

## M10 — Di động (SUMO + OSM, RWP BỎ HOÀN TOÀN)
- **M10.1** Network từ OSM (**Bạch Mai 5km×5km = MAP extract** topology/route; trục Giải Phóng) qua `netconvert` — ✅[`traffic-modeling-with-sumo-a-tutorial…pdf` (Guastella 2023); `sumo-user.pdf` §netconvert]. ⚠️ serving cell = 300m (M2.0) ≠ map 5km.
- **M10.1b** **FIDELITY = Tầng 1 + density sweep** (LOCKED):
  - Traffic signals: `netconvert --tls.guess` + node `highway=traffic_signals` → `tlLogic` — ✅ free OSM [`sumo-user.pdf` §TLS]. KHÔNG claim khớp timing thật HN.
  - Speed-limit: giữ `maxspeed` tag — ✅ free [`sumo-user.pdf` §edge speed].
  - Background density: `randomTrips` {light/medium/heavy} — 🔴 (ảnh hưởng tắc→tốc độ→SINR); KHÔNG calibrate đếm-xe thật.
  - Tầng 2 OD-synthetic (`od2trips`) = **FUTURE**; Tầng 3 (calibrate số đo thực HN) = **ngoài scope** (declare honest).
- **M10.2** **Kịch bản MCI hội tụ @ Bạch Mai** (D22): 3 xe đồng trú **1 cell 300m** (gNB @ Bạch Mai, M2.0), hội tụ **BV Bạch Mai** (trauma center). Cửa sổ contention = SCENE(φ₃)+TRANSPORT(φ₄) khi cả 3 trong 300m. 3 xe **severity khác nhau** → triage contention. 🔴 tọa độ/tuyến = minh hoạ (OSM thật). ⚠️ **BỎ** "3 BV khác quận" (mâu thuẫn single-cell→handover). **CLARIFY**: "route trong 300m" CHỈ verify khi SUMO route build xong (B3, mục này); bounce-reflection của RWP code cũ **KHÔNG** phải bằng chứng hợp lệ (RWP BỎ).
- **M10.3** `sumo_mobility.py` đọc FCD; GPS→metric Haversine — ✅[`sumo-user.pdf` §FCD; Haversine well-known]
- **M10.4** Tích hợp: thay `_advance_ambulance_positions` (RWP) → SUMO trace mặc định; SINR từ vị trí thật (nối M2).

## ⟲ RÀ SOÁT M10
Timestep FCD khớp TTI sim; tọa độ trong bbox OSM; SINR từ M2 dùng đúng d từ FCD (KHÔNG còn RWP); quỹ đạo 3 xe trong footprint 1 gNB suốt cửa sổ quyết định.

## micro-GATE 1D
SUMO trace chạy, dải SINR hợp lý (≈[−10,+30]dB NLOS-chi phối); validation = **realism** (KHÔNG "vs RWP" — RWP bỏ, audit FIX); scope/limit 3-tầng declared. Số RWP cũ KHÔNG tái dùng; solver sweep (W18–W23) chạy trên SUMO mobility. E3 (AoI LCFS/FCFS) / E4 (stress) → future work ([W24](W24_thesis_writing_defense.md)).

## Liên kết
Master plan PHẦN 11/W15 + M10.1b/M10.2 · `docs/03_architecture.md` (topology) · channel ← [W12](W12_pha1_radio_channel_capacity.md)/M2 · files `data/sumo/` ([docs/08](../08_implementation_notes.md)).
