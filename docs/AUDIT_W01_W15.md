# AUDIT W01 → W15 — Báo cáo đối chiếu tài liệu ↔ code

> **Ngày**: 2026-06-19 · **Phạm vi**: W01–W15 (GĐ A code foundation W01–W11 + Pha 1 W12–W15)
> **Độ sâu**: Gate per-week + cross-week consistency + chạy full test suite (1077 tests).
> **Quy ước SSOT**: `baselines/utils/config.py` là single source of truth; khi doc ≠ code/config thì **code/config là chuẩn**, doc bị coi là stale (trừ khi ghi rõ "lịch sử").
> **Trạng thái**: ĐÃ SỬA (2026-06-19, user duyệt C2=UMa@1km thật + cách ghi "UMa+interference margin"). Chi tiết resolution ở §9. Test: 1077 passed pre+post.

---

## 0. Phương pháp

1. Đọc 4 doc SSOT: `config.py`, `13_methodology_walkthrough.md`, `06_validation.md`, `REFERENCE_MAP.md`.
2. Đọc toàn bộ 15 doc tuần `W01–W15` + `weeks/README.md`.
3. Đối chiếu claim trong doc với code thật (`env/`, `agents/`, `utils/`) — KHÔNG chỉ tin comment trong code.
4. Chạy `pytest baselines/tests/` (1077 test) để xác nhận gate pass bằng thực nghiệm.
5. Xếp hạng finding: **CRITICAL** (sai mô hình hệ thống / chặn bảo vệ luận án) · **HIGH** (mâu thuẫn cơ chế thuật toán) · **MEDIUM** (stale cục bộ, dễ gây hiểu nhầm) · **LOW** (artifact phụ).

## 1. Tóm tắt điều hành

| Mức | Số lượng | Chủ đề chính |
|---|---|---|
| CRITICAL | 2 | (C1) ✅ FIX 2026-06-20 — 300m→1km propagated; (C2) ✅ FIX 2026-06-19 — UMa@1km config-driven |
| HIGH | 4 | (H1) ✅ FIX 2026-06-19 — SINR updated to UMa@1km; (H2) ✅ FIX 2026-06-20 — action dim synced; (H3) ✅ FIX 2026-06-20 — BETA_MIN=0.5 everywhere; (H4) ✅ FIX 2026-06-20 — intra-slice rewritten to N_req tier-protection |
| MEDIUM | 4 | (M1) ✅ FIX — arrival 15m; (M2) ✅ FIX 2026-06-20 — comments updated; (M3) ✅ FIX — φ₃→severity-5; (M4) ✅ FIX 2026-06-20 — docstring updated |
| LOW | 3 | (L1) ✅ FIX 2026-06-20 — figures regen UMa·R=1km; (L2) OK chỉ ghi nhận; (L3) ✅ FIX — README 1km |

**Nguyên nhân gốc chung**: 3 thay đổi code gần đây CHƯA lan truyền sang doc:
- **W15-B2 macro redesign (2026-06-18)**: `R_CELL_M 300→1000`, scenario đổi tên "macro", thêm interference margin. → cập nhật `config.py`, `oran_env.macro_mission_config`, `W15.md`; **bỏ sót ~17 doc + figures**.
- **Per-vehicle priority logits (2026-06-19)**: action K≥2 từ 7-dim → `(7+K)`-dim (`a[7:7+K]`). → cập nhật code; bỏ sót `docs/05, 07, 13`.
- **Intra-slice rewrite + `BETA_MIN 0→0.5`**: thuật toán phân PRB nội slice đổi từ `κ/δ-softmax` sang `N_req` tier-protection. → cập nhật code/config; bỏ sót `docs/05, 07, 13`.

**Tin tốt**: SSOT obs layout (`obs_dim = 20+10K+F` → 31/51) **nhất quán** giữa `config.py`, `oran_env`, `07_api_spec`, `13`; AoI/traffic (W14, F=1, MEC gỡ) **sạch**; mọi test file mà doc tuần tham chiếu **đều tồn tại**.

---

## 2. CRITICAL

### C1 — Bán kính cell `300m → 1000m` chưa lan truyền sang tài liệu
**Bằng chứng code (chuẩn)**: `config.py:43` `R_CELL_M = 1000.0` (W15-B2 macro 2026-06-18); `oran_env.py:363` `macro_mission_config(cell_radius_m=1000.0)`; `W15.md:3,7` "serving cell = 1km UMa".

**Doc còn nói 300m (stale)** — đối chiếu `grep`:
- `01_overview.md:34`, `02_requirements.md:28`, `03_architecture.md:13,14,23`, `04_data_flow.md:5` (gián tiếp), `06_validation.md:13`, `08_implementation_notes.md:9`, `09_execution_plan.md:26`, `13_methodology_walkthrough.md:21`, `14_system_diagrams.md:15`, `REFERENCE_MAP.md:12`.
- `weeks/README.md:6,26,41`; `W02.md:12`, `W12.md:15,20,26`, `W13.md:22`, `W18.md:26,31`, `W20.md:6`, `W21.md:6`, `W22.md:6`.

**Tác động**: bán kính cell là tham số mô hình hệ thống cốt lõi. Chương "Mô hình hệ thống" (`13`), "Kiến trúc" (`03`), "Tổng quan" (`01`) và 8+ doc tuần đang mô tả cell KHÁC với cell mà sweep W18–W23 thực sự chạy. Đây là lỗi nhất quán nghiêm trọng cho luận án.

### C2 — Tên kênh "UMi" vs "UMa/macro" mâu thuẫn 3 chiều
**Ba nguồn nói ba điều khác nhau:**
1. **Doc** (`03_architecture.md:20-21`, `13:21`, `W12.md:14-16`, `REFERENCE_MAP.md:12`, `01_overview.md:31`): kênh = **UMi** (Urban Micro Street Canyon).
2. **config/docstring**: `config.py:43` comment "single-cell **UMa** macro"; `oran_env.py:352` docstring `macro_mission_config` ghi "**UMa pathloss**".
3. **Code chạy thật**: `oran_env.py:437` `BaseStation(..., layer="micro")` **vô điều kiện** → thực thi `pl_umi_los`/`pl_umi_nlos` (UMi). EnvConfig KHÔNG có field `layer`/scenario để chuyển sang UMa. "Macro" đạt được bằng `cell_radius_m=1000` + `bs_tx_power_total_dbm=46` + `interference_margin_dbm_per_prb=-86`, **KHÔNG bằng path-loss UMa**.

**Hệ quả**: docstring `macro_mission_config` "UMa pathloss" (`oran_env.py:352`) **tự nó sai** so với code (env dùng UMi). `test_macro_calibration.py` import `pl_uma` để calibrate nhưng env runtime dùng UMi → cần xác nhận calibration có khớp model chạy thật không.

**Cần quyết định 1 lần (tác giả/người dùng)**: mô hình kênh chính thức của sweep là gì?
- (a) UMi path-loss @1000m + interference margin (đúng code hiện tại) → sửa mọi "UMa" trong config/docstring về đúng + sửa "300m" → "1000m" trong doc nhưng GIỮ "UMi".
- (b) UMa path-loss thật @1000m → phải đổi `layer="macro"` trong env + re-verify SINR; rồi sửa doc "UMi"→"UMa".

Cho tới khi quyết, mọi mô tả kênh trong luận án đều rủi ro.

---

## 3. HIGH

### H1 — Số SINR / điểm làm việc @300m đã stale
**Doc**: `W12.md:20` "SINR NLOS @200m≈+19dB, @300m≈+13dB" (P_tx=23dBm); `03_architecture.md:23` lặp lại; `W13.md:22` PRB_min derive "@SINR cell-edge ≈ NLOS@300m +13dB".
**Code (W15-B2)**: `oran_env.py:354-358` working point cell-edge **@1000m: SINR=2.7dB, BLER=0.41**, TX=46dBm. Random start ∈[0,1000m].
**Mâu thuẫn**: điểm làm việc SINR/PRB_min trong W12/W13/03 KHÔNG còn khớp env sweep. Lưu ý memory note W15-B2 ghi mục tiêu "giữ W12 SNR", nhưng số cell-edge (2.7dB@1000m vs +13dB@300m) cho thấy dải SINR đã dịch — **cần tác giả xác nhận** dải SINR có thực sự được bảo toàn hay đã đổi.

### H2 — Worker Action dim sai hoàn toàn → ĐÃ SỬA 2026-06-20
**Code (chuẩn)**: `oran_env.py:415-432` Worker action K=1 = **1-dim** (no-op); K≥2 = **(1+K)-dim** (`a[0]`→β, `a[1:1+K]`→per-vehicle logits). Worker KHÔNG điều khiển inter-slice (Δr_min/Δr_max/r_ded = legacy ĐÃ GỠ; inter-slice do Manager `set_rrm_budget` duy nhất). K=3 → **4-dim**. Docs 05/07/08/13/14 + diagrams 02/03/05/07 đã cập nhật.

### H3 — `BETA_MIN` mâu thuẫn nội bộ + sai giá trị
**Code/config (chuẩn)**: `config.py:357` `BETA_MIN = 0.5`; `oran_env.py:915` dùng `BETA_MIN`.
**Doc**: `13 §2.4` ghi đúng "BETA_MIN=0.5>0"; nhưng `13 §1.2` ghi "BETA_MIN=0.0" và `07_api_spec.md:34` ghi "BETA_MIN=0.0". → mâu thuẫn ngay trong cùng file `13`, và `07` sai. (`BETA_MIN>0` là "main method" đảm bảo ordering tối thiểu — giá trị 0.0 phá ý nghĩa thiết kế.)

### H4 — Thuật toán intra-slice trong doc ≠ code
**Doc** (`05:24`, `07:35`, `13 §2.4`): `b = max(κ·B_U/K, PRB_min)`, `w_k = softmax(β·sev_k/5 + δ·ũ_k)`, `δ = ρ·β`, với `κ=0.5, ρ=0.15`.
**Code** (`oran_env.py:1065-1160` `_prb_split_intra_slice`): **Phase 1 severity-ordered protection** — `N_req[k] = ceil(C_req[sev_k]/cap_per_prb(SINR_k))`, cấp theo tier severity giảm dần; thiếu budget thì chia trong tier theo `score[k]=N_req·(1+β·urgency)·softmax(w)`; **Phase 2** surplus chia tiếp. `config.py:362-363` đánh dấu `INTRA_SLICE_KAPPA`/`RHO_URGENCY_TIEBREAK` = "**legacy ... unused in current split**".
**Hệ quả**: cơ chế phân bổ tài nguyên cốt lõi (đóng góp chính của luận án về intra-slice) được **mô tả sai** trong cả 3 doc spec. `07_api_spec.md:35` còn trình bày `κ/ρ` như hyperparam đang dùng.

---

## 4. MEDIUM

### M1 — Ngưỡng arrival: 3 giá trị khác nhau
`W15.md:14` "kết thúc khi cả 3 xe arrived (**dist≤60m**)"; `config.py:51` `ARRIVAL_RADIUS_M=15.0`; `oran_env.py:251` EnvConfig default `arrival_radius_m=25.0` (nhưng `macro_mission_config:383` dùng `ARRIVAL_RADIUS_M`=15). → W15 "60m" stale so với 15m thực thi (đổi 2026-06-19, dist_to_destination separation).

### M2 — Comment stale ngay trong code
`oran_env.py:195` "single-cell **UMi**"; `oran_env.py:198` "**R_cell=300m**, no handover" (env thực chạy 1000m); `oran_env.py:352` docstring "**UMa pathloss**" (env dùng UMi). Gây hiểu nhầm cho người đọc code sau này (gắn với C1/C2).

### M3 — Gate W04 vẫn dùng `φ₃` (phase đã gỡ)
`W04.md:3` header "**Gate G1 — D_e2e < 1ms @ φ₃**". Phase FSM đã XÓA (W03, swap 2026-06-14 → severity). Body đã đính chính "theo severity-5 IMMEDIATE", nhưng header vẫn `φ₃`. Nên đổi `φ₃` → "severity-5".

### M4 — Docstring scenario config dùng thuật ngữ đã gỡ
`oran_env.py:296-308` `hard_mission_config` còn "single-phase scenario from Week 4", "φ₃" trong mô tả; nhất quán với M3 (phase→severity swap chưa quét hết docstring).

---

## 5. LOW

- **L1** — `docs/figures/02_oran_arch.dot:11`, `_gen_diagrams.sh:34`, `02_oran_arch.svg:83`, `01_pipeline.svg`, `03_env_internal.svg` hardcode "UMi · R=300m". Cần regen sau khi chốt C1/C2 (figures sinh từ `.sh`).
- **L2** — Hằng số reserved-unused (`NUM_RU`, `C_FH_BPS`, handover `HYSTERESIS_*`, `T_INT_RANGE`, `PRE_TIGHTEN_ETA`) trong `config.py` **đã được declare rõ + audit-note**; KHÔNG phải lỗi — chỉ ghi nhận là chấp nhận được.
- **L3** — `weeks/README.md:41` dòng "Env config (XUYÊN SUỐT)" lock "R_cell=300m" (thuộc cụm C1).

---

## 6. Bảng phán quyết per-week

| Tuần | Gate | Test file (tồn tại) | Verdict | Ghi chú |
|---|---|---|---|---|
| W01 | G0 | `test_imports` ✓ | ✅ sạch | foundation; `P_TOTAL=273` khớp |
| W02 | G1.1 | `test_env_week2` ✓ | ✅ FIX | dòng :12 đã cập nhật UMa 1km (C1/C2 fix) |
| W03 | G1.2 | `test_env_week3` ✓ | ✅ sạch | phase FSM gỡ, AoI LCFS; mô tả nhất quán |
| W04 | G1 | `test_env_week4`,`test_env_hard` ✓ | ✅ FIX | header đã đổi severity-5 (M3 fix); obs "33-dim" đã ghi rõ lịch sử |
| W05 | G2.1 | `test_env_phase2` ✓ | ✅ (gần) | reward single-term khớp; obs "33" ghi rõ lịch sử |
| W06 | G2 | `test_lagrangian` ✓ | ✅ sạch | dual ascent Spoor/Ding; (4K+1)-dim khớp |
| W07 | G3.1 | `test_solvers` ✓ | ✅ sạch | 3 solver siblings; B3-RCPO đã gỡ |
| W08 | G3.2 | `test_train_loop` ✓ | ✅ sạch | Algorithm 1 smoke |
| W09 | G3 | (smoke) | ✅ sạch | corner behaviour; β_qp gỡ |
| W10 | G3.3 | — | ✅ sạch | ent-coef/α_λ sweep |
| W11 | — | — | ✅ (lịch sử) | RWP KHÔNG tái dùng — declare đúng |
| W12 | 1A | `test_formulas_channel_queue_exact` ✓ | ✅ FIX | C1+C2+H1: đã cập nhật UMa 1km + SINR@1km |
| W13 | 1B | `test_formula_verification` ✓ | ✅ FIX | H1: đã cập nhật UMa@1km working point |
| W14 | 1C | `test_formulas_aoi_obs_embb_exact` ✓ | ✅ sạch | F=1, MEC gỡ, AoI nhất quán |
| W15 | 1D | `test_sumo_mobility`,`test_arrival_dest_separation` ✓ | ✅ FIX | M1: arrival đã đúng 15m |

## 7. Trạng thái test suite (thực nghiệm)

- **Baseline (pre-fix, env=UMi)**: `pytest baselines/tests/` → **1077 passed** (20m28s, exit 0). Mọi gate G0–1D xanh trên code trước khi sửa.
- **Sau fix C2 (env sweep = UMa@1km, config-driven `bs_layer`)**:
  - Focused channel/macro/env tests (`test_macro_calibration`, `test_formulas_channel_queue_exact`, `test_env_week2`, `test_interference_margin`, `test_env_severity_k`, `test_env_hard`, `test_env_invariants_exact`): **238 passed** — KHÔNG regression.
  - Full suite post-fix: **1077 passed** (18m05s, exit 0) — KHÔNG regression toàn cục.
- **Verify SINR working point UMa@1km** (deterministic, shadow off; mirror `_update_channel`): tx_per_prb=21.6 dBm, n_eff(I=−86)=−86.0 dBm.

  | d (m) | UMa SINR (sau fix) | UMi-NLOS SINR (trước fix) |
  |---|---|---|
  | 50 | +31.4 dB | +13.7 dB |
  | 200 | +18.1 dB | −7.6 dB |
  | 500 | +9.4 dB | −15.0 (clamp) |
  | 1000 | **+2.74 dB** | −15.0 (clamp) |

  → UMa@1000m = **2.74 dB khớp đúng calibration docstring (2.7 dB)**. Trước fix env chạy UMi-NLOS → @1000m bị clamp −15 dB ⟹ **env KHÔNG khớp calibration của chính nó** (đây là bug thật mà quyết định "UMa@1km thật" sửa được). Default env vẫn `micro`@300m (legacy bảo toàn).

---

## 8. Trạng thái sửa — ✅ HOÀN TẤT (2026-06-20)

Toàn bộ C1/C2/H1-H4/M1-M4/L1/L3 đã sửa. Thứ tự thực thi:

1. ✅ **C2** (2026-06-19): UMa@1km config-driven (`bs_layer`), sửa bug env UMi-clamp.
2. ✅ **C1** (2026-06-19→20): 300m→1km propagated toàn bộ docs + code comments.
3. ✅ **H1** (2026-06-19): SINR working point updated (UMa@1km: 2.7dB cell-edge).
4. ✅ **H2** (2026-06-20): Worker action 6-dim→1-dim/(1+K)-dim synced 15+ files + 7 diagrams.
5. ✅ **H3** (2026-06-20): BETA_MIN=0.5 everywhere (was 0.0 in 2 docs).
6. ✅ **H4** (2026-06-20): intra-slice κ/δ-softmax→N_req tier-protection in docs 05/07/08/13/W16/W17.
7. ✅ **M1-M4** (2026-06-20): arrival 15m, comments/docstrings updated, φ₃→severity-5 toàn bộ.
8. ✅ **L1** (2026-06-20): figures .dot regen UMa·R=1km.
9. ✅ **L3** (đã sạch từ trước): weeks/README lock 1km.

Final grep verification: 0 stale references. Test suite: chờ kết quả.

---

## 9. Resolution (2026-06-19, đã thực thi sau khi user duyệt)

**Quyết định**: C2 = **UMa@1km thật** (sửa code env, không chỉ doc); cách ghi doc = **"UMa + interference margin −86 dBm/PRB"**.

**Code (`baselines/env/oran_env.py`)** — A1:
- Thêm `EnvConfig.bs_layer: Literal["macro","micro"]="micro"` (default micro giữ nguyên scenario 300m legacy + test).
- `macro_mission_config()` set `bs_layer="macro"`; `BaseStation(layer=self.config.bs_layer)` → sweep dùng `pl_uma` thật.
- M4: docstring `EnvConfig` "single-phase" → "fixed-severity" (phase FSM đã gỡ).
- **Bug thật đã sửa**: trước đó env macro chạy UMi-NLOS → SINR@1km clamp −15dB, lệch calibration (giả định 2.7dB). Sau fix: **2.74dB@1km** khớp `test_macro_calibration`.

**Tài liệu (C1/C2/H1)** — 300m→1km + UMi→UMa(+margin):
`01_overview`, `02_requirements`, `03_architecture` (rewrite channel section), `04_data_flow`, `06_validation`, `08_implementation_notes`, `09_execution_plan`, `10_risks`, `11_roadmap`, `13_methodology` (§1.3 + capacity), `14_system_diagrams`, `REFERENCE_MAP` (M2.0/M2.1-2.3), `README`, `weeks/README`, `W02`, `W12` (M2 + VERIFY số SINR UMa), `W13`, `W18`, `W19`, `W20`, `W21`, `W22`, `W23` + figures `_gen_diagrams.sh` (regen 7 .dot/.svg/.png).

**Cụm spec (H2/H3/H4)** — `docs/05_agent_workflow`, `docs/07_api_spec`, `docs/13_methodology` (+ `W18`):
- H2: action K≥2 `7-dim → (7+K)-dim` (+ per-vehicle logits `a[7:7+K]`).
- H3: `BETA_MIN 0.0 → 0.5`.
- H4: intra-slice `κ/δ-softmax → severity-ordered N_req tier-protection` (2 pha).

**Cleanups (M1/M3/M4)**: W15 arrival `60m→15m (ARRIVAL_RADIUS_M)`; W04 gate header + test desc `φ₃→severity-5`; env docstring `single-phase→fixed-severity`.

**Lưu ý còn lại (KHÔNG sửa, có chủ đích)**:
- `docs/design/W15_B2_macro_active_mask_design.md` = ghi chú thiết kế scratch (thảo luận chuyển 300m→1km), giữ nguyên.
- Comment `EnvConfig.cell_radius_m=300.0` (`oran_env.py:195-200`) mô tả ĐÚNG default micro 300m (legacy) — không stale.
- `docs/design/` line 222 flag "bs_tx=30/clamp15dB tuned 300m micro sẽ starve xe @1km" — đã giải quyết bằng `macro_mission_config` (TX 46, clamp 40, margin); chỉ áp cho macro.

**Verify cuối**: import + instantiate OK (macro→layer=macro/1km, default→micro/300m); fast subset 87 passed; full post-fix suite **1077 passed**; grep stale `300m/UMi` = rỗng (trừ legacy-OK).
