# Phân bổ tài nguyên vô tuyến nhận biết mức độ nguy kịch cho đa xe cứu thương trong mạng 5G O‑RAN: Tiếp cận Học tăng cường phân cấp ràng buộc (Constrained Hierarchical Reinforcement Learning)

**Báo cáo Đồ án Tốt nghiệp**

> **Ghi chú dùng cho LaTeX/Overleaf.** File này là bản thảo nội dung (Markdown). Khi port sang LaTeX:
> - `#` → `\chapter`, `##` → `\section`, `###` → `\subsection`, `####` → `\subsubsection`.
> - Công thức trong `$...$` / `$$...$$` giữ nguyên cú pháp LaTeX (dán thẳng vào môi trường `equation`/`align`).
> - Mỗi khối "**Hình x.y**" kèm đường dẫn `docs/figures/*.svg` — trong LaTeX dùng `\includegraphics` (khuyến nghị convert SVG→PDF bằng `inkscape --export-type=pdf` hoặc dùng gói `svg`), kèm `\caption{}` và `\label{}` đã gợi ý.
> - Các bảng Markdown → môi trường `tabular`/`booktabs`.
> - Trích dẫn dạng `[Tên Năm]` → khóa BibTeX (xem Chương "Tài liệu tham khảo"); các nguồn ✅ là tài liệu thật trong `documents/` hoặc chuẩn 3GPP/IEEE; nguồn 🟡 là giả định kỹ thuật có cơ sở; 🔴 là giả định khai báo (declared) kèm sweep độ nhạy.
> - **Liêm chính khoa học:** mọi con số định lượng "kết quả huấn luyện" để TRỐNG/đánh dấu `‹điền sau khi train›` — đồ án này đã hoàn tất *mô hình hoá + phát biểu bài toán + thuật toán + kiểm chứng cấu trúc*; phần kết quả hội tụ thực nghiệm là bước chạy sweep W18–W23 và phải báo cáo kèm khoảng tin cậy. KHÔNG được tuyên bố đạt ε=10⁻⁵ bằng thực nghiệm khi chưa đủ mẫu (quy tắc rule‑of‑three).

---

## Mục lục

- **Tóm tắt** (Tiếng Việt / Abstract)
- **Danh mục từ viết tắt**
- **Danh mục hình vẽ**
- **Danh mục bảng**
- **Chương 1. Giới thiệu đề tài**
  - 1.1. Bối cảnh và động lực
  - 1.2. Phát biểu vấn đề
  - 1.3. Mục tiêu và câu hỏi nghiên cứu
  - 1.4. Phạm vi và giới hạn
  - 1.5. Đóng góp của đồ án
  - 1.6. Bố cục báo cáo
- **Chương 2. Nền tảng lý thuyết**
  - 2.1. Kiến trúc O‑RAN và phân mảnh mạng (network slicing)
  - 2.2. Dịch vụ URLLC, eMBB và chỉ số tuổi thông tin (AoI)
  - 2.3. Phân loại mức độ nguy kịch y tế (ATS triage)
  - 2.4. Quá trình quyết định Markov có ràng buộc (CMDP) và đối ngẫu Lagrange
  - 2.5. Học tăng cường phân cấp hai thang thời gian (HRL)
  - 2.6. Các thuật toán RL nền: PPO, TD3, SAC
  - 2.7. Tổng quan công trình liên quan và khoảng trống nghiên cứu
- **Chương 3. Mô hình hệ thống**
  - 3.1. Hình học, di động và vòng đời xe
  - 3.2. Mô hình kênh vô tuyến và dung lượng
  - 3.3. Mô hình hàng đợi, trễ đầu‑cuối và AoI
  - 3.4. Mô hình mức độ nguy kịch (severity) ngoại sinh
  - 3.5. Phân cấp thang thời gian
- **Chương 4. Bài toán tối ưu**
  - 4.1. Khung CMDP
  - 4.2. Hàm mục tiêu (eMBB log‑utility)
  - 4.3. Năm họ ràng buộc C1–C5 và vector (4K+1) chiều
  - 4.4. Lời giải đối ngẫu: phần thưởng tăng cường (hinge) và dual ascent (signed)
  - 4.5. Ràng buộc ưu tiên cứng cho xe cứu thương
- **Chương 5. Thuật toán và Triển khai**
  - 5.1. Kiến trúc phân cấp Manager–Worker
  - 5.2. Không gian quan sát, hành động và phần thưởng
  - 5.3. Manager: chính sách liên‑lát (chỉ học `b_rrm`)
  - 5.4. Worker: phân chia nội‑lát thuần RL (masked softmax)
  - 5.5. Cập nhật đối ngẫu hai thang thời gian (SMDP)
  - 5.6. Ba solver ngang hàng PPO / TD3 / SAC
  - 5.7. Tổ chức mã nguồn (SSOT)
  - 5.8. Môi trường mô phỏng `ORANEnv`
  - 5.9. Vòng huấn luyện
  - 5.10. Tham số siêu (hyperparameters)
- **Chương 6. Đánh giá thực nghiệm**
  - 6.1. Kiểm chứng độc lập: oracle khả thi + oracle runtime
  - 6.2. Tiêu chí "giải đúng": khả thi sơ cấp có điều kiện theo severity
  - 6.3. Kỷ luật thống kê (rule‑of‑three, không Jain toàn cục)
  - 6.4. Thiết kế Bảng I / Bảng II
  - 6.5. Kiểm thử tự động (test suite)
  - 6.6. Kết quả kiểm chứng cấu trúc (đã có)
  - 6.7. Khung kết quả huấn luyện (để điền sau sweep)
  - 6.8. Thảo luận giả định và mối đe doạ tới tính hợp lệ
- **Chương 7. Kết luận và Hướng phát triển**
- **Tài liệu tham khảo**
- **Phụ lục A.** Bảng tra Công thức → Mã nguồn (Equation‑to‑Code Ledger)
- **Phụ lục B.** Bảng tham số đầy đủ
- **Phụ lục C.** Danh mục hình và mã nguồn sinh hình

---

## Tóm tắt

Đồ án nghiên cứu bài toán **phân bổ tài nguyên vô tuyến (PRB) nhận biết mức độ nguy kịch bệnh nhân** cho nhiều xe cứu thương cùng truyền dữ liệu y tế khẩn cấp (vital signs, cảnh báo DENM/CAM) qua một cell **5G O‑RAN** đô thị. Khi nhiều xe cạnh tranh tài nguyên hữu hạn trong cùng một cell, việc chia đều là không phù hợp về mặt y khoa; tài nguyên cần được ưu tiên theo **độ nguy kịch** của bệnh nhân trên xe, đồng thời vẫn bảo đảm sàn dịch vụ cho người dùng băng rộng nền (eMBB).

Đóng góp cốt lõi là **một bài toán tối ưu** được phát biểu dưới dạng **Quá trình quyết định Markov có ràng buộc (CMDP)**: cực đại hoá tiện ích log‑throughput của eMBB **với năm họ ràng buộc** về trễ trung bình, đuôi trễ (độ tin cậy), sàn thông lượng eMBB, AoI trung bình và đuôi AoI — mỗi xe chịu ràng buộc theo **chính mức nguy kịch của nó** (ánh xạ từ thang phân loại ATS 5 mức). Bài toán được giải bằng **học tăng cường phân cấp hai thang thời gian (HRL)** với tầng Manager (100 ms, điều khiển ngân sách liên‑lát URLLC↔eMBB) và tầng Worker (10 ms, phân chia nội‑lát giữa các xe), kết hợp **đối ngẫu Lagrange** thực thi ràng buộc qua dual ascent. Để bảo đảm so sánh công bằng, cùng một bài toán được giải bởi **ba solver ngang hàng**: PPO (on‑policy), TD3 và SAC (off‑policy).

Hệ thống được triển khai trên mô phỏng giải tích (M/G/1 Pollaczek–Khinchine + Monte Carlo + kênh UMa 3GPP TR 38.901) với di động từ vết SUMO/OSM khu vực Bệnh viện Bạch Mai. Tính đúng đắn của *mô hình, công thức và thuật toán* được kiểm chứng bằng **oracle độc lập** và **bộ kiểm thử tự động hơn một nghìn ca**; tiêu chí kết luận "giải đúng" được định nghĩa chặt chẽ theo **khả thi sơ cấp có điều kiện theo severity** (kèm quy tắc rule‑of‑three cho ràng buộc đuôi hiếm), thay vì chỉ dựa trên đường cong phần thưởng.

**Từ khoá:** O‑RAN, network slicing, URLLC, Age of Information, CMDP, Lagrangian, hierarchical reinforcement learning, PPO/TD3/SAC, medical triage.

---

## Danh mục từ viết tắt

| Viết tắt | Nghĩa |
|---|---|
| O‑RAN | Open Radio Access Network |
| RIC | RAN Intelligent Controller (Near‑RT) |
| O‑DU / O‑CU | O‑RAN Distributed / Centralized Unit |
| PRB | Physical Resource Block |
| URLLC | Ultra‑Reliable Low‑Latency Communication |
| eMBB | enhanced Mobile Broadband |
| AoI | Age of Information |
| CMDP | Constrained Markov Decision Process |
| SMDP | Semi‑Markov Decision Process |
| HRL | Hierarchical Reinforcement Learning |
| PPO / TD3 / SAC | Proximal Policy Optimization / Twin Delayed DDPG / Soft Actor‑Critic |
| GAE | Generalized Advantage Estimation |
| ATS | Australasian Triage Scale |
| SINR | Signal‑to‑Interference‑plus‑Noise Ratio |
| UMa | Urban Macro (3GPP channel model) |
| DENM / CAM | Decentralized Environmental Notification / Cooperative Awareness Message |

---

## Danh mục hình vẽ

| Hình | Tệp nguồn | Nội dung |
|---|---|---|
| 1.1 | `docs/figures/01_pipeline.svg` | Pipeline 3 pha: Mô hình hoá → Phát biểu tối ưu → Giải |
| 2.1 | `docs/figures/02_oran_arch.svg` | Kiến trúc O‑RAN: Near‑RT RIC ↔ Manager/Worker ↔ O‑DU |
| 3.1 | `docs/figures/03_env_internal.svg` | Nội bộ môi trường mô phỏng `ORANEnv` |
| 4.1 | `docs/figures/04_cmdp.svg` | Dòng dữ liệu CMDP–Lagrangian |
| 5.1 | `docs/figures/05_state_action.svg` | Không gian quan sát và hành động |
| 5.2 | `docs/figures/06_training_loop.svg` | Vòng huấn luyện hai thang thời gian |
| 6.1 | `docs/figures/07_one_episode.svg` | Diễn tiến một episode |

---
