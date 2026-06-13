# W16 — Pha 1: Severity Model NACA-S (exogenous, event-driven)

> **Pha**: 1 · **Status**: 📅 PLANNED · **Gate**: **GATE 1E (kết thúc Pha 1)** · **Nhóm**: M11 · **Build**: B2 · **Deps**: W12-W15

## M11 — Mô hình độ nặng (ngoại sinh, KHÔNG vitals giả)
- **M11.1** Thang NACA 7 mức — ✅[`Prehospital_emergency_anesthesia…pdf` p.3; `s12245-024-00605-5.pdf`]; dùng **6 mức NACA 1-6** {0,0.2,0.4,0.6,0.8,1.0}; NACA 7 (death) = OUT-OF-SCOPE declared.
- **M11.2** RTS nền (GCS/SBP/RR) — ✅[`a_revision_of_the_trauma_score.17.pdf` (Champion 1989)]
- **M11.3** Kernel **birth-death CTMC** mỗi tick: `P(+1)=λ_det·dt`, `P(−1)=λ_stab·dt`; biên s∈[0,5] phản xạ. **CHỈ ±1 bậc/transition** (đa bậc chỉ qua tích lũy tick hoặc event M11.5). `λ_det` 🔴 sens {0.001,0.01,0.1} s⁻¹.
- **M11.4** `λ_stab` 🔴 sens {0.005,0.02,0.1} s⁻¹.
- **M11.4b** LOCKED: λ_det/λ_stab **phase-INDEPENDENT** (hằng số mọi pha); hiệu ứng pha CHỈ qua event rời rạc tường minh (M11.5). Phương án λ_det^φ **loại** (thêm params 🔴 không data).
- **M11.5** **Phase-event → severity: MAP THEO LOẠI** (state-conditional, KHÔNG 1 luật chung):
  - `collision_shock` → `s ← clip(s+Δ, 0, 5)`, Δ=1 default (sens {1,2}) 🔴; anchor "MCI ~+1 NACA" 🟡 (KHÔNG ghi tên tác giả chưa kiểm chứng).
  - `cardiac_arrest` → `s ← max(s, NACA6)` SET-TO-FLOOR — ✅ grounded (def M11.1).
  - cardiac non-arrest → **CHƯA model** (nếu thêm: luật riêng + ref lâm sàng 🔴, KHÔNG áp luật ngừng-tim).
- **M11.6** severity → network priority — 🔴 **design principle** (KHÔNG ref y khoa trực tiếp); analogy mở rộng từ triage [`news2-executive-summary_0.pdf`; `a_revision_of_the_trauma_score.17.pdf`].
- **M11.7** s0 + λ gán 3 xe — **HYBRID**: TRAIN ngẫu nhiên/episode (tổng quát hóa); EVAL scripted scenarios cố định (seed cố định) — (a)[0.2,0.6,1.0] ordering; (b)[1,1,1] MCI toàn nguy kịch; (c) φ₄-crit(1.0) vs φ₃-mild(0.0) "money shot".

## ⟲ RÀ SOÁT M11
Transition KHÔNG đọc QoS/SINR (đảm bảo ngoại sinh); clip [0,5] đúng 6 mức; KHÔNG nguồn nào bị gán tên tác giả khi chưa mở đối chiếu; severity vào ALLOCATION/constraint (KHÔNG vào reward — §1.5).

## GATE 1E (kết thúc Pha 1)
Mọi 🟡 của M1–M11 đã thành ✅; mọi 🔴 đã declared + có sensitivity/calibration plan; rà soát chéo: 0 đại lượng nào trong code đi vào kết quả mà thiếu nhãn. **→ Pha 1 hoàn tất.**

## Liên kết
Master plan PHẦN 11/W16 + §1.1 · `naca_severity.py` ([docs/08](../08_implementation_notes.md)) · severity → intra-slice PRB weight `softmax(β·sev+δ·ũ)` formal hoá ở [W17](W17_pha2_cmdp_formulation.md)/P5 (β = priority temp, KHÔNG RL discount).
