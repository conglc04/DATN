# Chương 6. Đánh giá thực nghiệm

![Một episode](figures/07_one_episode.svg)
**Hình 6.1.** Diễn tiến một episode (Manager↔Worker↔dual). *(`\label{fig:episode}`)*

## 6.1. Kiểm chứng độc lập: oracle khả thi + oracle runtime

- **`feasibility_oracle.py`** (tĩnh, *không* RL): cài lại độc lập vật lý queue/capacity, quét toàn bộ phân chia $b\in\{0..273\}$ để xác nhận **tồn tại** phân bổ thoả C1+C3 — chứng minh *bài toán có nghiệm khả thi* (độc lập với mã đang kiểm thử).
- **`runtime_oracle.py`** / **`closure_checks.py`**: xác nhận các bất biến runtime (bảo toàn PRB, kẹp ngân sách, mask, thang thời gian) khớp ≤ $10^{-6}$.

## 6.2. Tiêu chí "giải đúng": khả thi sơ cấp có điều kiện theo severity

> **Nguyên tắc trung thực (điểm cốt lõi của đánh giá):** đường cong `ep_reward` tăng/bão hoà **không** chứng minh giải đúng CMDP — phần thưởng có thể tăng do confound (eMBB↑ hy sinh URLLC, đổi phân phối severity, episode dài hơn, $\lambda$ còn nhỏ, overfit một seed). Tiêu chí đúng là **khả thi sơ cấp**, đánh giá dưới chính sách *deterministic* trên tập held‑out, **có điều kiện theo từng mức severity**.

Công cụ `audit/feasibility_eval.py` (tái dùng `clopper_pearson_upper` của `eval_tail_bank.py`) cho mỗi mức severity:
- **C1/C4 (trung bình):** $\text{mean}\le$ ngưỡng;
- **C3 (sàn):** thiếu hụt $=\max(0,R_{\min}-R_{\text{eMBB}})=0$;
- **C2/C5 (đuôi):** chặn trên 95% (Clopper–Pearson) $\le\varepsilon$ **chỉ khi** $N\ge 3/\varepsilon$ (đủ mẫu để *phân giải* $\varepsilon$); nếu $N<3/\varepsilon$ → trạng thái **inconclusive** (KHÔNG phải chứng chỉ khả thi — tuyệt đối không tuyên bố đạt $\varepsilon$ chưa quan sát được);
- **reward/step theo severity:** *chỉ chẩn đoán*, không phải tiêu chí pass.

Một mức là *FEASIBLE* khi cả năm điều kiện đạt; chính sách *PASS* khi **mọi** mức quan sát được đều khả thi.

## 6.3. Kỷ luật thống kê

- **So sánh bội:** Holm–Bonferroni ($p<0.01$); cỡ hiệu ứng **Hedges' g**; **bootstrap 95% CI**. Không tuyên bố "thắng" nếu CI chồng lấn.
- **Sự kiện hiếm $\varepsilon$:** $10^{-5}$ không validate được bằng 10 seed; báo cáo tỉ lệ vi phạm quan sát + CI + **rule‑of‑three** $\varepsilon\le 3/N$; **không** vẽ "đạt $10^{-5}$".
- **Công bằng (không Jain toàn cục):** Jain thuần thưởng chia đều ⟹ *mâu thuẫn* ưu tiên severity. Dùng: **ordering‑compliance** ($\text{sev}_i>\text{sev}_j\wedge S>0\Rightarrow \text{PRB}_i>\text{PRB}_j$), **no‑starvation min‑share**, **Jain trong tier**, **priority‑inversion rate**.
- **λ‑saturation logging:** vẽ quỹ đạo $\lambda$ + % bước $\lambda=\Lambda_{\max}$.
- **Eval công bằng (BẮT BUỘC):** ba solver eval bằng **cùng seed** ⟹ cùng chuỗi severity/traffic/channel.

## 6.4. Thiết kế Bảng I / Bảng II

- **Bảng I** (3 solver × $K\in\{1,3\}$ = 6 ô): reward + tỉ lệ vi phạm C1–C5 + λ‑saturation.
- **Bảng II** ($K{=}3$): chỉ số severity/intra‑slice + ablation $2\times2$ (phase∈{off,on} × severity∈{off,on}).

## 6.5. Kiểm thử tự động (test suite)

Hơn một nghìn ca pytest khoá: bố cục obs, bất biến PRB, đúng dấu ràng buộc, đơn điệu severity↑⇒sàn↑, tính đúng cập nhật PPO/TD3/SAC, masked softmax, mask active/arrived, và verdict khả thi điểm‑12. Toàn bộ **xanh** ở thời điểm chốt báo cáo.

## 6.6. Kết quả kiểm chứng cấu trúc (đã có)

| Hạng mục kiểm chứng | Công cụ | Trạng thái |
|---|---|---|
| Bài toán có nghiệm khả thi (C1+C3) | `feasibility_oracle.py` | ✔ tồn tại điểm khả thi (dư C3 lớn) |
| Bất biến runtime ($\le10^{-6}$) | `runtime_oracle.py`, `closure_checks.py` | ✔ |
| Bảo đảm cấu trúc: URLLC > eMBB mọi severity; sàn đơn điệu | test + `B_RRM_FLOOR_BY_SEV` | ✔ |
| Đúng dấu: vi phạm→penalty↑, $\lambda\uparrow$; lỏng→$\lambda\downarrow$ | `lagrangian.py` + test | ✔ |
| Tính đúng thuật toán PPO/TD3/SAC | đối chiếu paper + test | ✔ |
| SMDP $\gamma_H=\gamma_L^{10}$ (không double‑discount) | chứng minh + `test_imports` | ✔ |
| Toàn bộ test suite | pytest | ✔ 0 fail |

## 6.7. Khung kết quả huấn luyện (để điền sau sweep)

> **Liêm chính khoa học:** các bảng dưới đây là *khung báo cáo*; số liệu định lượng được điền **sau khi chạy sweep W18–W23** và phải kèm 95% CI + Holm–Bonferroni. KHÔNG điền số phỏng đoán.

**Bảng 6.3 (khung). Bảng I — Hiệu năng 3 solver × $K\in\{1,3\}$.**

| Solver | K | reward (CI) | viol_C1 | viol_C2 (UB, N≥3/ε?) | C3 shortfall | viol_C4 | viol_C5 | %λ‑sat |
|---|---|---|---|---|---|---|---|---|
| PPO | 1 | ‹…› | ‹…› | ‹… / ?› | ‹…› | ‹…› | ‹…› | ‹…› |
| PPO | 3 | ‹…› | | | | | | |
| TD3 | 1/3 | ‹…› | | | | | | |
| SAC | 1/3 | ‹…› | | | | | | |

**Bảng 6.4 (khung). Verdict khả thi theo severity (điểm‑12, deterministic, held‑out, cùng seed).**

| sev | C1 mean | C4 mean | C3 shortfall | C2 tail | C5 tail | reward/step | FEASIBLE |
|---|---|---|---|---|---|---|---|
| 1..5 | ‹…› | ‹…› | ‹…› | pass/fail/inconclusive | … | ‹chẩn đoán› | ‹…› |

**Kỳ vọng nếu giải đúng:** sàn $b_{\text{rrm}}$ đơn điệu 0.65→0.85 theo severity; viol_C1/C4/C3 đạt ngưỡng; C2/C5 *pass* khi đủ mẫu (hoặc *inconclusive* trung thực khi không); $\lambda$ ổn định (không blow‑up, không pinned 0); explained‑variance chuyển âm→dương, ổn định qua seed (KHÔNG cần ≈1).

## 6.8. Thảo luận giả định và mối đe doạ tới tính hợp lệ

- **Single‑cell, nhiễu hiệu chỉnh (🔴):** biên noise‑rise −86 dBm/PRB là *declared*, không đo thực; kèm sweep độ nhạy.
- **Severity ngoại sinh cố định/episode (🔴):** trừu tượng hoá ATS; không mô hình hoá diễn tiến lâm sàng trong episode.
- **Giả định trễ/AoI (🟡/🔴):** $D_{\text{FH/BH/DET/STOCH}}$, $\text{AoI}_{\max}$, $R_{\text{REF}}$, $\eta$ có cơ sở nhưng kèm sweep.
- **Mô phỏng giải tích, không packet‑level:** M/G/1 PK giả định Poisson; burst DENM bắt bằng Monte Carlo; ns‑3 là future work.
- **Manager severity vô hướng = max (lossy):** giảm nhẹ bằng $\lambda_{\text{global}}$ trong $s_H$.
- **Citation:** tuân thủ "quy tắc thép" REFERENCE_MAP — nguồn cổ điển vắng corpus (Borkar/Altman/Boyd/Kaul/Jain…) **không** trích như đã có, thay bằng nguồn corpus tương đương.

