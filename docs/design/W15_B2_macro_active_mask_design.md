# W15-B2 Design — Macro-cell 1km + active_mask cell-entry gating

**Status:** DESIGN (pre-implementation). Produced by multi-agent design workflow
(5 facet designers → synthesis → adversarial critique). 2 critics (invariants,
edge-cases) returned BLOCK with 9 concrete fixes — all folded in below. 2 critics
(thesis, rl) hit session limit; covered manually in §12. **No code changed yet.**

---

## 1. Motivation

Chia tài nguyên radio chỉ có ý nghĩa **trong lúc xe cứu thương đang vận chuyển về
viện** (để ER chuẩn bị trước), không phải ở 300 m cuối sát viện. Vì vậy: phóng cell
thành **macro 1 km**, route **bắt đầu ngoài cell** từ nhiều hướng, 3 xe hội tụ về
**tọa độ thật Bạch Mai** (`21.002966°N, 105.840780°E`). Episode bắt đầu **trước/tại
lúc xe đầu tiên vào cell** và RL phân bổ PRB **ngay khi có xe active**, ưu tiên
severity cao trong nhóm active. Episode kết thúc khi **cả 3 xe tới phòng cấp cứu**.

## 2. Decisions LOCKED

| ID | Decision |
|----|----------|
| D1-A | **Real-time single clock** (`sim_time += tti_sec`). KHÔNG nén. Delay/reliability/throughput rate-based ⇒ dt-invariant; chỉ AoI + position phụ thuộc time. |
| Macro | `BaseStation(layer="macro")` → UMa pathloss. Giữ single-cell + SNR (bảo toàn W12). Giữ map OSM. |
| Cell radius | **1000 m** — nhưng CHỈ trong macro scenario config, **KHÔNG đổi default** (xem FIX-5). |
| Destination | **Tọa độ thật Bạch Mai** = gNB anchor; edge drivable gần nhất là dest hội tụ chung. Gọi đúng "điểm cấp cứu BV", không phóng đại. |
| active_mask | Ngoài cell → 0 PRB (inactive); vào cell → active; tới viện → arrived (FIX-1). |
| Termination | **terminated khi mọi xe đã-vào đều arrived** (OD2 resolved theo yêu cầu user). |
| obs/action | **KHÔNG đổi dim** (`20+10K+F`). active_mask KHÔNG vào obs; suy ra từ sentinels. |

---

## 3. CRITIQUE FIXES (9 blocking — bắt buộc áp dụng)

### Invariants critic

- **FIX-1 (radius default):** KHÔNG bump `cell_radius_m` default 300→1000 toàn cục —
  `d_k = dist/cell_radius_m` ([oran_env.py:1064]) sẽ đổi giá trị obs của MỌI scenario
  K=1 và phá K=1 numeric preservation + layout-lock test. → Giữ `R_CELL_M`/
  `EnvConfig.cell_radius_m` default = **300.0**; set **1000.0 CHỈ trong macro
  scenario config** (`hard_mission_config(cell_radius_m=1000.0, ...)` hoặc một
  `macro_mission_config()` riêng). *(Override C5 của unified design.)*
- **FIX-2 (reset ordering):** `reset()` gọi `_update_channel`/`_update_queue_service_rates`
  ([:549-550]) → gọi `_prb_split_intra_slice` đọc `active_mask`. Phải khởi tạo
  `active_mask` (+ positions) **TRƯỚC** các call này trong `reset()`.
- **FIX-3 (B_U < Ka):** khi tổng PRB URLLC < số xe active, `b=B_U//Ka=0` → vài xe
  active nhận 0 PRB (vi phạm PRB_MIN_QOS). Xử lý tường minh: cấp 1 PRB cho `B_U` xe
  active **severity cao nhất**, 0 cho phần còn lại; `sum==B_U` vẫn giữ. Log cảnh báo
  (cell quá tải) — không nuốt im lặng.
- **FIX-4 (single mask snapshot/tick):** đọc `active_mask` **một lần đầu `_mac_tick`**,
  dùng nhất quán cả tick; cập nhật mask (crossing) ở cuối tick → áp dụng từ tick sau
  (≤0.5 ms latency). Tránh đọc 2 snapshot khác nhau trong cùng tick.
- **FIX-5 (existing floor tests):** đổi floor K→Ka làm 2 test all-k floor cũ sai. Phải
  cập nhật + flag rõ trong commit, không để fail âm thầm.

### Edge-cases critic

- **FIX-6 (arrived_mask + clamp-not-wrap):** mask write-once gây xe đã tới viện vẫn
  giữ PRB + bơm phantom violation (CHẮC CHẮN xảy ra mỗi episode khi xe tới đích, do
  trace wrap). → Thêm trạng thái **arrived** (dist ≤ ARRIVAL_RADIUS): xe arrived
  **ngừng nhận PRB**, không tính c_vec; provider **clamp** ở cuối trace (KHÔNG wrap).
  Vòng đời mask: `inactive → active → arrived`.
- **FIX-7 (AoI seed ordering):** seed AoI-on-activation phải robust, không phụ thuộc
  may rủi thứ tự call. → kiểm tra/đặt mask ở **đầu logic tick** (FIX-4) rồi mới
  `_on_activation` seed `last_delivered_gen_time = sim_time` ⇒ AoI=0 tại entry.
- **FIX-8 (no-show severity):** xe có offset > episode (KHÔNG bao giờ vào) vẫn lái
  `severity_ref/alpha_e/C3` như phantom. → `severity_ref = max severity` chỉ trên xe
  **sẽ-vào hoặc đã-vào** (loại no-show). *(Tinh chỉnh §5 unified: predictive cho xe
  sẽ-vào, nhưng loại hẳn xe không thuộc episode.)*
- **FIX-9 (offset relative to entry_times_trace):** offset cộng vào trace-t=0 KHÔNG
  điều khiển được thứ tự vào thật (vì mỗi route có `entry_time_trace` khác nhau) →
  pattern có thể đảo/sụp. → áp offset **tương đối với `entry_times_trace[k]`**:
  realized_entry[k] = entry_time_trace[k] + offset[k], và offset được sinh để
  realized_entry khớp pattern mong muốn.

---

## 4. Data model

```python
# EnvConfig (new/changed)
cell_radius_m: float = 300.0           # default UNCHANGED (FIX-5); macro config sets 1000.0
entry_pattern: str | None = None       # None ⇒ sample/episode; else forced
sumo_pool_dir: str | None = None       # pooled provider opt-in
arrival_radius_m: float = 30.0         # "tới phòng cấp cứu" threshold (FIX-6, termination)

# ORANEnv per-episode (reset, BEFORE channel/queue init — FIX-2)
self.active_mask:   np.ndarray   # (K,) bool  inactive→active
self.arrived_mask:  np.ndarray   # (K,) bool  active→arrived (FIX-6)
self._entry_sim_time:  np.ndarray  # (K,) float, -1=not entered
self._eta_edge_sec:    np.ndarray  # (K,) float, info/Manager only (NOT obs)
self.entry_pattern: str
self._will_enter:   np.ndarray   # (K,) bool — realized_entry ≤ episode_end (FIX-8)
```

Config: `R_CELL_M` stays 300 (macro config overrides); add `ENTRY_PATTERNS`,
`ENTRY_PATTERN_OFFSET_RANGES`, `ARRIVAL_RADIUS_M`. Repurpose `HANDOVER_ETA_TRIGGER`
as ETA-norm horizon for Manager.

## 5. Episode lifecycle (timeline per pattern)

Single clock `sim_time += tti_sec`. `reset()`: sim_time=0, mask all False, trace
positioned so offset-0 vehicle is just outside cell. RL allocates the instant any
`dist_k ≤ cell_radius_m`. Terminated when **all `_will_enter` vehicles arrived**
(dist ≤ arrival_radius), else truncated at `episode_duration_sec`.

```
fully_staggered (realized_entry relative to each route's edge-crossing, FIX-9):
t:   0    E0            E1(+~25)        E2(+~55)          A0   A1   A2
m:  [0,0,0]→[1,0,0]────→[1,1,0]───────→[1,1,1]──────────→arrived… → terminated
PRB: idle  → B_U(amb0) → split{0,1}  → sev-softmax{0,1,2} → drop arrived from split
```
(all_simultaneous: offsets≈[0,0,0]; pair_simultaneous: [0,0,~40]. Realized entry
computed relative to `entry_times_trace` so pattern can't invert — FIX-9.)

## 6. active_mask transition (FIX-1,2,4,6,7)

```
# top of _mac_tick — single snapshot (FIX-4)
dist = ||pos||(axis=1)
# arrival first (FIX-6): active & dist≤arrival_radius → arrived (stop PRB)
newly_arrived = active_mask & ~arrived_mask & (dist <= arrival_radius_m)
arrived_mask |= newly_arrived
# entry: eligible by realized schedule (FIX-9) & inside & not arrived
eligible = sim_time >= realized_entry_time            # = entry_time_trace + offset
newly = (~active_mask) & eligible & (dist <= cell_radius_m)
for k in newly: _on_activation(k)   # reset queue; seed AoI last_delivered=sim_time ⇒ AoI=0 (FIX-7)
active_mask |= newly
# allocation-active = active & ~arrived
```
Pooled provider owns authoritative entry schedule (`entry_times_trace + offsets`);
env merges. Legacy provider returns all-ones (`active`), no `arrived`.

## 7. PRB gating (preserves Σ==B_U, K=1; FIX-3)

`_prb_split_intra_slice`: restrict floor/remainder/softmax to **allocation-active**
set `A = active & ~arrived`. `Ka=|A|`. If `Ka==0` → zeros (eMBB absorbs, valid). If
`B_U < Ka` (FIX-3) → 1 PRB to top-`B_U` severity in A, 0 else, `Σ==B_U`. Else
`b=max(floor(κ·B_U/Ka),PRB_MIN_QOS)` (fallback `b=B_U//Ka`), `S=B_U−Ka·b`, softmax
`β·sev_norm+δ·ũ` over A. **K=1 active ⇒ result[0]=B_U** (softmax([x])=[1.0]); K=1
inactive/arrived ⇒ 0. C6/C7 untouched (live in `_apply_action`).

Queue/arrival: inactive **and arrived** k → `n_urllc=0`, `arrival_rate=0`,
`update_service_rate(0,0)`, `D_e2e=0.0` (not 2·D_max clamp).

## 8. obs/action (NO dim change)

Per-amb 10-dim block; for **inactive** k override: SINR→`clamp_min/40`, delay_norm→0,
aoi_norm→0; keep **dist (real, >1, ETA-proxy R5), speed (real), severity (real, R5),
λ (frozen warm)**. Fixed-block AoI mean/max + URLLC rho/hol/arr: **mean over active
only** (else 0). `severity_ref = max(sev) over (_will_enter | active)` (FIX-8). ETA
in info + `build_manager_state` only, NOT obs (preserves checkpoints/K=1).

## 9. Inactive/arrived semantics

| Aspect | inactive | arrived | active |
|--------|----------|---------|--------|
| PRB | 0 | 0 (FIX-6) | severity-softmax share |
| arrivals/service | 0 | 0 | real |
| D_e2e / c_vec C1,C2,C4,C5 | 0 | 0 | real |
| AoI | masked, seed-0 on entry | masked | tracked |
| Lagrangian dual | **frozen** (skip idx) | frozen | updated |
| severity_ref | counts if `_will_enter` | counts | counts |
| reward (eMBB shared) | unchanged | unchanged | unchanged |

Dual freeze is explicit (skip indices), NOT zero-subgradient — feeding `c=0` vs
`d_phi=D_max>0` would push λ down below warm-start (critic C2).

## 10. Route pool + provider

Offline `data/sumo/`: `06_generate_route_pool.py` (outer starts annulus [1000,1800]m,
stratify 6 sectors, N_POOL≈24 triplets, pairwise bearing≥60°, dist≥300m, dest=real
BM edge; depart=0), `07_run_pool_simulation.sh` (`--end 120`), `08_compute_pool_manifest.py`
(record `entry_time_trace`, `final_dist_m<arrival_radius` else drop — separate
planned-dest vs actual-arrival). Runtime `PooledSumoMobilityProvider`: seed-select
triplet, sample pattern+offsets (relative to entry_times_trace, FIX-9), hold vehicle
at trace-start until its realized schedule, expose `active_mask`/`arrived_mask`/`eta`.
Guard `episode_duration_sec ≤ trace_duration`; clamp at trace end (FIX-6).

## 11. Timebase

`tti=0.5ms`; mobility_dt=aoi_dt=tti; Worker=20·tti=10ms; Manager=10·Worker=100ms.
Delay/reliability/eMBB rate-based ⇒ dt-invariant. AoI driven by single clock,
seeded 0 on activation. ETA(info)=`max(0,dist−R_CELL)/max(radial_speed,0.1)`.
Macro episode `episode_duration_sec≈120` (OD3).

## 12. Thesis + RL review (critics that didn't run)

- **Single-cell legitimacy:** inactive xe đơn giản **chưa nằm trong vùng phủ gNB này
  / chưa stream về gNB này** — KHÔNG claim "được phục vụ bởi cell khác" (tránh ngụ ý
  multi-cell phá W12). Honest framing: "trước khi vào cell, xe chưa tiêu thụ PRB của
  cell này." SNR/I≈0 còn nguyên.
- **Gate honesty:** đích = tọa độ thật BM + edge drivable gần nhất; gọi "điểm cấp
  cứu/đích hội tụ", có test `final_dist<arrival_radius` để chứng minh xe THỰC SỰ tới.
- **RL nonstationarity:** active set đổi giữa episode là **observable** qua
  dist/speed/sentinels → policy học được; reward eMBB shared + c_vec mask đúng quy
  active; K=1 numeric preserved (FIX-1/3). Risk credit-assignment thấp vì reward
  shared, constraint per-amb mask sạch.

## 13. Test plan (nhóm)

Masking/invariants: `test_active_mask_all_false_at_reset`, `…_entry_transition_monotone`,
`test_arrived_stops_prb` (FIX-6), `test_prb_sum_invariant_partial_active`,
`test_prb_k1_active_exact`, `test_prb_k1_inactive_zero`, `test_prb_BU_lt_Ka` (FIX-3),
`test_prb_all_inactive_zero`, `test_inactive_zero_arrivals_service`,
`test_inactive_d_e2e_zero`, `test_cvec_inactive_zeroed`, `test_lagrangian_frozen_inactive`,
`test_reset_sets_mask_before_channel` (FIX-2), `test_single_mask_snapshot_per_tick` (FIX-4).
Obs: `test_obs_dim_unchanged`, `test_obs_layout_lock_passes`, `test_inactive_obs_sentinels`,
`test_k1_numeric_preservation`, `test_severity_ref_excludes_noshow` (FIX-8).
AoI: `test_on_activation_seeds_aoi_zero` (FIX-7), `test_aoi_excluded_summary_inactive`.
Lifecycle: `test_pattern_sampling_reproducible`, `test_fully_staggered_distinct_entry`,
`test_offsets_relative_to_entry_trace` (FIX-9), `test_all_arrived_terminates` (OD2),
`test_all_inactive_episode_survives`, `test_partial_tick_activation_dilution`.
Pool: `test_pool_manifest_complete`, `test_pool_bearing_diversity`, `test_pool_no_teleport`,
`test_pooled_reset_all_inactive`, `test_pattern_distribution_uniform`,
`test_episode_le_trace_guard`, `test_legacy_provider_active_mask_default_ones`.
Regression: full suite (~221) green, esp. layout-lock + K=1 preservation.

## 14. Open decisions (cần user chốt)

- **OD1 (link budget) — BLOCKER trước train:** macro 1km + UMa làm SINR/capacity đổi;
  `bs_tx_power_dbm=30`/`clamp 15dB` (tuned 300m micro) sẽ starve xe ở 1000m. Cần chốt
  TX (≈46 dBm macro?) + recalib `rrm_budget_hint`/clamp để "hard nhưng solvable".
  → **Phase 0 benchmark sẽ đo và đề xuất số cụ thể.**
- **OD2 — RESOLVED:** terminated khi mọi xe `_will_enter` arrived (theo yêu cầu user).
- **OD3:** `episode_duration_sec` macro ≈ 120s — xác nhận giá trị + ảnh hưởng PPO rollout/SMDP.
- **OD4:** offset synthetic (đã chọn, FIX-9) vs SUMO `depart=` — xác nhận chấp nhận cho
  claim mobility-fidelity (motion vẫn real SUMO, chỉ entry timing dời).
- **OD6:** URLLC summaries (rho/hol/arr) active-only mean — xác nhận.
- **OD7:** N_POOL=24 sau prune duarouter có thể giảm — chốt min pool + fallback sector.

---

**Next:** Phase 0 (de-risk) = (a) macro link-budget benchmark → đề xuất TX/clamp cho
OD1; (b) `06_generate_route_pool.py` thử + manifest → xác nhận pool khả thi. Sau Phase
0 + user chốt OD1/OD3 → implement env masking bằng TDD.
