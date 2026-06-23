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

# Chương 1. Giới thiệu đề tài

![Pipeline 3 pha](figures/01_pipeline.svg)
**Hình 1.1.** Pipeline phương pháp luận ba pha: (1) Mô hình hoá hệ thống vật lý, (2) Phát biểu bài toán tối ưu CMDP, (3) Giải bằng HRL với ba solver. *(LaTeX: `\caption{Pipeline phương pháp luận ba pha.}\label{fig:pipeline}`)*

## 1.1. Bối cảnh và động lực

Tại các đô thị mật độ giao thông cao như Hà Nội, nhiều xe cứu thương có thể đồng thời hoạt động trong vùng phủ của cùng một trạm gốc và cùng truyền **dữ liệu y tế khẩn cấp** — tín hiệu sinh tồn (ECG, SpO₂, nhịp tim, huyết áp) và bản tin cảnh báo giao thông (DENM/CAM) — về trung tâm điều phối cấp cứu. Loại lưu lượng này thuộc lớp **URLLC**: yêu cầu độ trễ đầu‑cuối cực thấp (cỡ mili‑giây) và độ tin cậy cực cao (tỉ lệ vi phạm cỡ 10⁻⁵). Song song, hạ tầng còn phục vụ người dùng băng rộng dân sự (**eMBB**) như một nền lưu lượng.

Mạng **5G O‑RAN** với cơ chế **phân mảnh (network slicing)** cho phép dành riêng và điều phối tài nguyên cho từng lớp dịch vụ thông qua bộ điều khiển thông minh **Near‑RT RIC** và chuẩn **RRMPolicyRatio** [3GPP TS 28.541]. Tuy nhiên, khi *nhiều xe URLLC cạnh tranh trong cùng một lát (slice)*, một câu hỏi cốt lõi nảy sinh: **chia tài nguyên trong nội bộ lát URLLC như thế nào giữa các xe?** Về mặt y khoa, chia đều là sai: một bệnh nhân nguy kịch (severity cao) cần được ưu tiên hơn một ca không khẩn cấp. Đây chính là khoảng trống "thiếu nhận biết mức nguy kịch người dùng *bên trong* lát" mà đồ án nhắm tới.

## 1.2. Phát biểu vấn đề

Cho một cell O‑RAN đơn với tổng $P_{\text{total}}=273$ PRB phục vụ đồng thời $K$ xe cứu thương (lưu lượng URLLC) và một nền eMBB. Mỗi xe $k$ mang một bệnh nhân với **mức nguy kịch** $\text{sev}_k\in\{1,\dots,5\}$ (ngoại sinh, cố định trong một episode). Cần xác định **chính sách phân bổ PRB** theo thời gian — gồm (i) tỉ lệ ngân sách liên‑lát URLLC↔eMBB và (ii) cách chia ngân sách URLLC giữa các xe — sao cho:

- **cực đại hoá** tiện ích thông lượng eMBB nền;
- **đồng thời thoả** các ràng buộc QoS theo từng mức nguy kịch của *từng xe* (trễ trung bình, đuôi trễ/độ tin cậy, độ tươi dữ liệu AoI), và một sàn thông lượng eMBB tối thiểu;
- **bảo đảm nguyên tắc y khoa**: xe cứu thương (có bệnh nhân) **luôn** được ưu tiên hơn eMBB ở mọi mức nguy kịch, và mức ưu tiên **tăng đơn điệu** theo độ nguy kịch.

## 1.3. Mục tiêu và câu hỏi nghiên cứu

**Mục tiêu tổng quát:** xây dựng và kiểm chứng một *bài toán tối ưu phân bổ tài nguyên nhận biết nguy kịch* cùng một *khung giải bằng HRL có ràng buộc*, có thể so sánh công bằng nhiều thuật toán RL.

**Câu hỏi nghiên cứu:**
- **CH1.** Làm thế nào hình thức hoá yêu cầu y khoa "ưu tiên theo nguy kịch" thành một CMDP có cơ sở chuẩn (3GPP/queueing/RL)?
- **CH2.** Kiến trúc HRL hai thang thời gian nào tách bạch đúng điều khiển liên‑lát (Manager) và nội‑lát (Worker), và thực thi ràng buộc ổn định?
- **CH3.** Tiêu chí định lượng nào kết luận một chính sách *thực sự* giải đúng bài toán (vượt qua nhược điểm "phần thưởng tăng ≠ khả thi")?

## 1.4. Phạm vi và giới hạn

**Trong phạm vi:** single‑cell UMa 1 km tại khu vực Bạch Mai; $K\in\{1,3\}$ xe; phân bổ nội‑lát nhận biết severity; pipeline ba pha (Mô hình → Tối ưu → Giải); sweep ba solver PPO/TD3/SAC.

**Ngoài phạm vi (khai báo trung thực):** chuyển giao liên cell (handover); tính toán biên/MEC offload; mô phỏng mức gói ns‑3; hiệu chỉnh số đo giao thông thực; sinh giá trị sinh tồn giả. Đồ án **không** tuyên bố sử dụng dữ liệu/hạ tầng nhà mạng thực.

## 1.5. Đóng góp của đồ án

1. **Ràng buộc nhận biết ngữ cảnh (severity‑aware)** — *đóng góp chính*: ánh xạ thang **ATS 5 mức** thành (a) tầng ngưỡng QoS `SEVERITY_QOS`, (b) trọng số phần thưởng $\alpha_e(\text{sev})$, (c) thứ tự ưu tiên nội‑lát — lấp khoảng trống "thiếu nhận biết nguy kịch *trong* lát".
2. **CMDP–Lagrangian HRL ánh xạ sang chuẩn 3GPP RRMPolicyRatio** (đã xây dựng + kiểm thử).
3. **Ràng buộc cứng AoI (C4/C5)** thay cho chỉ trễ thuần — ưu tiên *độ tươi* dữ liệu y tế.
4. **Khung kiểm chứng "giải đúng" theo khả thi sơ cấp có điều kiện theo severity** (kèm rule‑of‑three) — phương pháp đánh giá trung thực thay cho việc chỉ nhìn đường cong reward.

> *Trung thực về tính mới:* phép chiếu khả thi (feasibility projection) được trình bày như kỹ thuật an toàn theo cấu trúc, **không** tuyên bố là novel.

## 1.6. Bố cục báo cáo

Chương 2 trình bày nền tảng lý thuyết. Chương 3 mô hình hoá hệ thống. Chương 4 phát biểu bài toán tối ưu CMDP. Chương 5 thiết kế thuật toán HRL với ba solver và triển khai phần mềm. Chương 6 đánh giá thực nghiệm (kiểm chứng, kết quả, thảo luận). Chương 7 kết luận và hướng phát triển.

---

# Chương 2. Nền tảng lý thuyết

![Kiến trúc O-RAN](figures/02_oran_arch.svg)
**Hình 2.1.** Ánh xạ kiến trúc O‑RAN: tầng Manager đóng vai xApp/Near‑RT RIC điều khiển `RRMPolicyRatio`, tầng Worker điều khiển ưu tiên nội‑lát ở O‑DU. *(`\label{fig:oran}`)*

## 2.1. Kiến trúc O‑RAN và phân mảnh mạng

O‑RAN mở giao diện giữa các khối RAN và đưa trí tuệ điều khiển vào **Near‑RT RIC** (chu kỳ 10 ms–1 s) thông qua các ứng dụng *xApp*. Phân mảnh mạng dành riêng tài nguyên cho từng lớp dịch vụ; tỉ lệ tài nguyên giữa các lát được biểu diễn bằng **RRMPolicyRatio** (`RRMPolicyMinRatio`/`MaxRatio`/`DedicatedRatio`) trong mô hình thông tin quản lý [3GPP TS 28.541]. Trong đồ án, *quyết định ngân sách liên‑lát của Manager* ánh xạ trực tiếp sang đại lượng chuẩn này.

## 2.2. Dịch vụ URLLC, eMBB và chỉ số tuổi thông tin (AoI)

URLLC đặc trưng bởi cặp (trễ tối đa $D_{\max}$, xác suất vi phạm $\varepsilon$); eMBB tối ưu thông lượng. Ngoài trễ, dữ liệu y tế còn cần **độ tươi**: **Age of Information** $\Delta(t)=t-U(t)$ với $U(t)$ là thời điểm sinh của gói mới nhất đã nhận [Qi 2024; Mlika 2022] *(không trích Kaul — vắng corpus)*. AoI nắm bắt yêu cầu "dữ liệu phải mới", khác với trễ gói đơn lẻ; đồ án dùng chính sách hàng đợi **LCFS + drop‑old** để tối ưu AoI.

## 2.3. Phân loại mức độ nguy kịch y tế (ATS triage)

**Australasian Triage Scale (ATS)** chia bệnh nhân thành 5 mức từ *Non‑urgent* đến *Immediate* [ACEM]. Đồ án dùng ATS làm trục ưu tiên: mức nguy kịch là thuộc tính **ngoại sinh** của bệnh nhân trên xe (không suy ra từ tín hiệu sinh tồn giả), **cố định trong một episode** và được lấy mẫu lại giữa các episode. Quyết định mô hình hoá này (🔴 declared, kèm sweep độ nhạy) tránh việc bịa dữ liệu lâm sàng.

## 2.4. Quá trình quyết định Markov có ràng buộc (CMDP) và đối ngẫu Lagrange

Một **CMDP** mở rộng MDP $(\mathcal S,\mathcal A,P,r,\gamma)$ bằng tập ràng buộc kì vọng $J_{C_j}(\pi)\le d_j$. Bài toán

$$\max_{\pi}\;J_r(\pi)\quad\text{s.t.}\quad J_{C_j}(\pi)\le d_j,\ j=1,\dots,m$$

được giải qua **đối ngẫu Lagrange** với hàm $L(\pi,\lambda)=J_r(\pi)-\sum_j\lambda_j\big(J_{C_j}(\pi)-d_j\big)$ và **dual ascent** $\lambda\leftarrow[\lambda+\alpha_\lambda(J_{C}-d)]_+$ [Spoor 2025; Ding 2023] *(không trích Boyd/Tessler — vắng corpus)*. Khung CMDP cho mạng vô tuyến đã được dùng trong [Yongshuai Liu 2020; Wen Wu 2020; Qiang Liu 2021] *(không trích Altman)*.

## 2.5. Học tăng cường phân cấp hai thang thời gian (HRL)

HRL tách chính sách thành tầng cao (Manager, chu kỳ chậm) và tầng thấp (Worker, chu kỳ nhanh). Khi tầng cao quyết định mỗi $W$ bước tầng thấp, bài toán tầng cao là **bán‑Markov (SMDP)** với hệ số chiết khấu $\gamma_H=\gamma_L^{W}$, bảo đảm hai tầng "nhìn" cùng một chân trời thời gian thực. Sự hội tụ của lược đồ hai thang thời gian được dẫn dắt bởi **tách biệt tốc độ học** $\alpha_{\pi_H}\ll\alpha_{\pi_L}$ [Akyıldız 2024] *(Borkar 2008 vắng corpus; lưu ý: định lý chỉ biện minh THỨ TỰ tốc độ, không quy định giá trị cụ thể — các giá trị là heuristic + tinh chỉnh)*.

## 2.6. Các thuật toán RL nền: PPO, TD3, SAC

- **PPO** [Schulman 2017, `1707.06347`]: on‑policy, mục tiêu thay thế bị cắt (clipped surrogate) + **GAE** [Foundations of Deep RL].
- **TD3** [Fujimoto 2018, `fujimoto18a`]: off‑policy, *clipped double‑Q*, *target policy smoothing*, *delayed actor update*.
- **SAC** [Haarnoja 2018, `1812.05905`]: off‑policy cực đại entropy, nhiệt độ $\alpha$ tự điều chỉnh.

## 2.7. Tổng quan công trình liên quan và khoảng trống nghiên cứu

Các công trình slicing‑RL hiện có (ví dụ tối ưu eMBB log‑utility [Alsenwi 2022; Sohaib 2024]) tập trung *liên‑lát* hoặc coi URLLC là khối đồng nhất. **Khoảng trống**: chưa có cơ chế phân biệt *mức nguy kịch người dùng bên trong lát URLLC* gắn với ràng buộc QoS theo từng người dùng và AoI. Đồ án lấp khoảng trống này bằng CMDP severity‑aware + HRL nội‑lát.

---

# Chương 3. Mô hình hệ thống

![Nội bộ ORANEnv](figures/03_env_internal.svg)
**Hình 3.1.** Nội bộ môi trường `ORANEnv`: kênh → hàng đợi/AoI → phân bổ PRB → quan sát/`c_vec`. *(`\label{fig:env}`)*

> Nguồn‑sự‑thật: `baselines/env/oran_env.py` + `baselines/utils/config.py`. Kiểm chứng bởi `audit/closure_checks.py`, `audit/feasibility_oracle.py`, `audit/runtime_oracle.py`.

## 3.1. Hình học, di động và vòng đời xe

- **Một macro cell** bán kính $R_{\text{cell}}=1000$ m, gNB tại gốc cục bộ $(0,0,h{=}10\text{ m})$, kênh **3GPP UMa @ 3.5 GHz (FR1 n78)** [TS 38.101‑1].
- **Di động** từ vết FCD **SUMO/OSM** (kịch bản Bạch Mai); RWP chỉ giữ cho unit‑test.
- **Vòng đời xe** bằng các mặt nạ: `entered_mask` (chốt khi vào cell), `arrived_mask` (chốt khi tới đích), và

$$\text{active\_mask}_k=\text{entered}_k\wedge\neg\,\text{arrived}_k.$$

Xe ngoài cell vẫn di chuyển trong SUMO nhưng **không** đóng góp phần thưởng/ràng buộc (bị mask). Tách **hai khoảng cách** (`dist_to_gNB` vs `dist_to_destination`) để tránh "tới đích giả" khi tuyến chỉ đi ngang qua gNB.
- **Episode** bắt đầu khi xe đầu tiên vào cell và kết thúc khi **toàn bộ $K$ xe đã tới đích** *hoặc* hết **400 s** (timeout). Đoạn rollout PPO 1 s **không** phải là episode.

## 3.2. Mô hình kênh vô tuyến và dung lượng

- $P_{\text{total}}=273$ PRB (100 MHz, $\mu=1$), $B_{\text{PRB}}=360$ kHz [TS 38.101‑1 Bảng 5.3.2‑1; TS 38.211].
- Suy hao đường truyền `pl_uma` đơn điệu tăng theo khoảng cách [TR 38.901 §7.4]; shadowing có seed; **nhiễu = biên dự phòng noise‑rise đã hiệu chỉnh** mô hình reuse‑1 macro (🔴 declared, không phải giá trị 3GPP trực tiếp).
- Nhiễu nền $N=-174+10\log_{10}(B)+\mathrm{NF}$ dBm, $\mathrm{NF}=7$ dB; SINR kẹp $[-10,40]$ dB.
- **Dung lượng mỗi PRB:**

$$C_{\text{PRB}}=\eta\,B_{\text{PRB}}\log_2\!\big(1+\mathrm{SINR}_{\text{lin}}\big),\qquad \eta=0.75$$

với $\eta$ là hệ số hiệu suất link‑adaptation hấp thụ overhead MCS/coding (🟡 [Hyoungju Ji 2017]). Đã kiểm chứng chính xác tại $-5/0/2.7/10/20$ dB.

## 3.3. Mô hình hàng đợi, trễ đầu‑cuối và AoI

- Mỗi xe có một hàng đợi URLLC **M/G/1**; một hàng đợi eMBB gộp. Trễ hàng đợi theo **Pollaczek–Khinchine** [Kleinrock]:

$$\mathbb E[D_q]=\frac{\lambda\,\mathbb E[S^2]}{2(1-\rho)},\qquad \rho<0.9.$$

- **Trễ đầu‑cuối:**

$$D_{\text{e2e}}=D_{\text{DET}}+\tfrac{1}{\mu}+\mathbb E[D_q]+D_{\text{FH}}+D_{\text{BH}}$$

với $D_{\text{DET}}=0.07$ ms, $D_{\text{FH}}=D_{\text{BH}}=0.1$ ms (🟡 [TS 38.214; O‑RAN.WG4]), thời gian phục vụ tăng cường bởi $D_{\text{STOCH}}=0.05$ ms (🔴 declared ±50%).
- **Lưu lượng:** Poisson `ambulance_status` (một luồng hợp nhất F=1, ~50 pkt/s, 400 B) [Alsenwi 2022; Sohaib 2024]; eMBB nền từ `bystander_traffic.py`.
- **AoI:** $\Delta(t)=t-U(t)$, **LCFS + drop‑old**, chỉ reset khi cập nhật thành công; một tracker/xe, reset khi vào cell. Xe inactive **không** sinh AoI/ràng buộc giả.

## 3.4. Mô hình mức độ nguy kịch (severity) ngoại sinh

Mỗi xe nhận một mức $\text{sev}_k\in\{1..5\}$ **độc lập**, lấy mẫu mỗi episode (mặc định **đồng đều** $0.2$ mỗi mức) và **cố định** trong episode. Đại lượng **dùng chung** lấy theo

$$\text{sev}_{\text{ref}}:=\max_k \text{sev}_k$$

(xe nặng nhất lái các đại lượng chia sẻ — xem §4.5). Với $K=3$ và 5 mức, có $5^3=125$ tổ hợp có thứ tự, đồng xác suất với trọng số mặc định.

**Bảng 3.1. Tầng ngưỡng QoS theo mức nguy kịch (`SEVERITY_QOS`).** Mọi cột đơn điệu chặt dần theo severity.

| sev | Tên | $D_{\max}$ | $\varepsilon$ (đuôi trễ) | $\text{AoI}_{\max}$ | $\varepsilon_{\text{AoI}}$ | $\alpha_e$ (trọng số eMBB) |
|---|---|---|---|---|---|---|
| 1 | NON_URGENT | 20 ms | $10^{-3}$ | 1.0 s | $10^{-2}$ | 0.70 |
| 2 | SEMI_URGENT | 10 ms | $10^{-4}$ | 0.5 s | $10^{-3}$ | 0.55 |
| 3 | URGENT | 5 ms | $10^{-4}$ | 0.2 s | $10^{-3}$ | 0.40 |
| 4 | EMERGENCY | 2 ms | $10^{-5}$ | 0.1 s | $10^{-3}$ | 0.20 |
| 5 | IMMEDIATE | 1 ms | $10^{-5}$ | 0.1 s | $10^{-3}$ | 0.05 |

> Mức chi tiết của các cột là *cố ý*: $D_{\max}$ và $\alpha_e$ phân biệt đủ 5 mức; $\varepsilon$ dùng 3 bậc tin cậy chuẩn 3GPP (99.9/99.99/99.999%) để tránh bịa "nines" phi chuẩn; AoI bão hoà ở sàn tươi 0.1 s cho hai mức nguy kịch nhất [TS 22.261].

## 3.5. Phân cấp thang thời gian

**Bảng 3.2. Ba thang thời gian (Gate 2 — chính xác).**

| Tầng | Chu kỳ | Tỉ lệ | Hằng số |
|---|---|---|---|
| MAC TTI | 0.5 ms | — | `MAC_TTI_SEC` |
| Worker (xApp) | 10 ms | 20 MAC tick | `MAC_TICKS_PER_WORKER=20` |
| Manager | 100 ms | 10 Worker step | `WORKER_STEPS_PER_MANAGER=10` |
| Rollout PPO | 1 s | 100 Worker step | (chỉ để update, không reset env) |

Hệ số chiết khấu: $\gamma_L=0.99$ (Worker), $\gamma_H=\gamma_L^{10}\approx 0.9044$ (Manager).

---

# Chương 4. Bài toán tối ưu

![CMDP-Lagrangian](figures/04_cmdp.svg)
**Hình 4.1.** Dòng dữ liệu CMDP–Lagrangian: `c_vec`/`d_phi` → lệch chuẩn hoá → (đối ngẫu signed) cập nhật $\lambda$ và (sơ cấp hinge) phần thưởng tăng cường. *(`\label{fig:cmdp}`)*

## 4.1. Khung CMDP

Trạng thái $s$ mô tả tình trạng hàng đợi/kênh/AoI/severity/đối ngẫu; hành động phân cấp $(a_H,a_L)$; phần thưởng và năm tín hiệu ràng buộc $c_1,\dots,c_5$ được tính mỗi bước Worker. Đóng góp của đồ án **là bản thân bài toán** (khung chung), độc lập với thuật toán giải.

## 4.2. Hàm mục tiêu (eMBB log‑utility)

Phần thưởng **một số hạng**, chỉ theo eMBB, có trọng số theo severity tham chiếu:

$$\boxed{\;r_t=\alpha_e(\text{sev}_{\text{ref}})\cdot\log\!\Big(1+\frac{R_{\text{eMBB},t}}{R_{\text{REF}}}\Big),\quad R_{\text{REF}}=100\text{ Mbps},\ \ \text{sev}_{\text{ref}}=\max_k\text{sev}_k\;}$$

[Alsenwi 2022 Eq.13; Sohaib 2024 Eq.9]. **Vai trò của $\alpha_e$:** chỉ là trọng số *giá trị biên của eMBB*; URLLC **không** xuất hiện trong phần thưởng mà được bảo vệ hoàn toàn qua ràng buộc $\lambda$ (tránh đếm trùng). Do $\alpha_e$ giảm theo severity (0.70→0.05), khi bệnh nhân nặng hơn, hệ thống "ít coi trọng" eMBB hơn ⟹ dồn PRB cho URLLC.

**Vì sao lấy $\max$ (lựa chọn mô hình tường minh, $K\ge2$):** mục tiêu là một vô hướng duy nhất, nên trọng số $\alpha_e$ phải gộp $K$ mức về một. Lấy **max** = bệnh nhân nguy kịch nhất trong cell: ngay khi *một* xe ở mức nặng, eMBB bị hạ ưu tiên xuống mức đó (thiên về bệnh nhân — bảo thủ nhất). Đây không phải "một severity đại diện không định nghĩa": **QoS riêng từng xe vẫn được thực thi per‑vehicle** qua $\text{sev}_k$ trong C1/C2/C4/C5 và $\lambda$ per‑xe; chỉ trọng số reward chung + sàn C3 + one‑hot dùng max. Phương án mean/weighted‑sum bị loại vì làm loãng ưu tiên của bệnh nhân nguy kịch duy nhất.

## 4.3. Năm họ ràng buộc C1–C5 và vector $(4K+1)$ chiều

Mỗi xe $k$ chịu bốn ràng buộc theo **chính mức nguy kịch của nó**; C3 (sàn eMBB) là *chia sẻ*:

| ID | Loại | Ràng buộc | Ngưỡng |
|---|---|---|---|
| C1$_k$ | trung bình | $\mathbb E[D_{\text{e2e},k}]\le D_{\max}^{\text{sev}_k}$ | `SEVERITY_QOS` |
| C2$_k$ | đuôi (chance) | $\Pr[D_{\text{e2e},k}>D_{\max}^{\text{sev}_k}]\le\varepsilon^{\text{sev}_k}$ | $10^{-3}..10^{-5}$ |
| C4$_k$ | trung bình | $\mathbb E[\text{AoI}_k]\le\text{AoI}_{\max}^{\text{sev}_k}$ | `SEVERITY_QOS` |
| C5$_k$ | đuôi (chance) | $\Pr[\text{AoI}_k>\text{AoI}_{\max}^{\text{sev}_k}]\le\varepsilon_{\text{AoI}}^{\text{sev}_k}$ | $10^{-2}..10^{-3}$ |
| C3 | sàn | $R_{\text{eMBB}}\ge R_{\min}=10$ Mbps | cố định mọi severity |

Vector nhân tử/ràng buộc có $(4K+1)$ chiều, sắp xếp $[\,C1_0..C1_{K-1},\,C2_*,\,C4_*,\,C5_*,\,C3_{\text{shared}}\,]$. Tại $K=1$ đây là hoán vị $[0,1,3,4,2]$ của thứ tự cũ $[C1,C2,C3,C4,C5]$, bảo toàn chính xác hành vi. C3 cố định 10 Mbps (Gate 7): tách rời severity, đóng vai *lưới an toàn chống đói* eMBB, thoả với dư rất lớn (oracle khả thi: $R_{\text{eMBB}}\ge 97$ Mbps kể cả tải nặng cell‑edge).

## 4.4. Lời giải đối ngẫu: phần thưởng tăng cường (hinge) và dual ascent (signed)

**Sơ cấp (primal) — hinge một phía** (chỉ phạt vi phạm, không thưởng dư an toàn):

$$r^{\text{aug}}_t = r_t - \sum_{j} \lambda^{\text{local}}_j\,\max\!\Big(0,\ \frac{c_{j}-d_{j}}{\text{scale}_j}\Big).$$

**Đối ngẫu (dual) — dùng độ lệch *có dấu*** để $\lambda$ có thể *giảm* khi ràng buộc lỏng:

$$\hat g_j=\frac{1}{N}\sum\frac{c_{j}-d_{j}}{\text{scale}_j},\qquad \lambda \leftarrow \mathrm{clip}\big(\lambda+\alpha_\lambda\,\hat g,\ 0,\ \Lambda_{\max}\big),\ \ \Lambda_{\max}=10.$$

> **Điểm tinh tế cốt lõi:** primal = *hinge*, dual = *signed* — bất đối xứng **cố ý**. Hinge ngăn ràng buộc lỏng tạo "thưởng âm" làm lệch gradient; signed giữ cho dual ascent đúng nghĩa (vi phạm→$\lambda\uparrow$, dư→$\lambda\downarrow$). Lưu ý: hinge ở primal **khác** Lagrangian sách giáo khoa thuần signed‑slack — đây là biến thể có chủ đích (audit bonus‑masking).

**Ước lượng lai (hybrid) cho gradient đối ngẫu:** C1/C4/C3 (loại trung bình) dùng cửa sổ‑khoảng (N≈200, reset mỗi Manager step); **C2/C5 (loại đuôi, $\varepsilon$ tới $10^{-5}$)** dùng ước lượng *tích luỹ theo episode* — vì N≈200 không thể phân giải tỉ lệ $10^{-5}$ (cần $N\gtrsim 3/\varepsilon$).

## 4.5. Ràng buộc ưu tiên cứng cho xe cứu thương

Vì phần thưởng chỉ theo eMBB, chính sách có xu hướng tự nhiên dồn PRB cho eMBB (hiện tượng "đói URLLC"). Để **bảo đảm nguyên tắc y khoa** ("xe luôn có bệnh nhân, còi hú bật ⟹ luôn ưu tiên"), ngân sách URLLC bị chặn dưới bởi một **sàn cứng đơn điệu theo severity** (`B_RRM_FLOOR_BY_SEV`):

**Bảng 4.1. Sàn ngân sách URLLC theo severity.**

| sev | Sàn $b_{\text{rrm}}$ | PRB URLLC tối thiểu | PRB eMBB tối đa |
|---|---|---|---|
| 1 | 0.65 | 177 | 96 (~34 Mbps) |
| 2 | 0.70 | 191 | 82 |
| 3 | 0.75 | 205 | 68 |
| 4 | 0.80 | 218 | 55 |
| 5 | 0.85 | 232 | 41 (~14 Mbps) |

Sàn $>50\%$ mọi mức ⟹ URLLC **luôn** lớn hơn eMBB; tăng đơn điệu theo severity; sàn lấy theo $\text{sev}_{\text{ref}}=\max$ nên thống nhất cho cả $K=1$ và $K=3$. Trần khả thi (≈0.8645, suy từ sàn C3) > sàn cực đại 0.85 ⟹ eMBB luôn còn ≥15% (>10 Mbps), **không** xung đột giữa ưu tiên xe và C3.

---

# Chương 5. Thuật toán và Triển khai

![Không gian trạng thái-hành động](figures/05_state_action.svg)
**Hình 5.1.** Không gian quan sát $s_L$ (32 chiều, $K{=}1$) và hành động Worker. *(`\label{fig:sa}`)*

![Vòng huấn luyện](figures/06_training_loop.svg)
**Hình 5.2.** Vòng huấn luyện hai thang thời gian. *(`\label{fig:loop}`)*

## 5.1. Kiến trúc phân cấp Manager–Worker

- **Manager** (100 ms) là *bộ điều khiển duy nhất* của ngân sách liên‑lát $b_{\text{rrm}}$.
- **Worker** (10 ms) *chỉ* chia $B_{\text{URLLC}}$ giữa các xe đang active; **không** chạm $b_{\text{rrm}}$.
- Cả ba solver dùng **chung** kiến trúc HRL này; chỉ khác lõi RL (on/off‑policy).

## 5.2. Không gian quan sát, hành động và phần thưởng

**Quan sát** $s_L$ có $20+11K+F$ chiều ($K{=}1,F{=}1\to 32$; $K{=}3\to 54$):
- *Khối cố định 20 chiều* `[0:20]`: $\rho$ hàng đợi, HOL, tỉ lệ PRB, tốc độ đến, BLER, **one‑hot severity_ref `[10:15]`**, $\lambda^{C3}_{\text{shared}}$ `[15]`, anchor ngân sách, số UE nền, AoI mean/max.
- *Khối per‑xe 11 chiều* ($\times K$): $\{$SINR$_k$, $d_k$, $v_k$, delay_norm$_k$, AoI_norm$_k$, severity_norm$_k$, $\lambda^{C1}_k,\lambda^{C2}_k,\lambda^{C4}_k,\lambda^{C5}_k$, **active_mask$_k\}$**.
- *Khối luồng* $F{=}1$: AoI luồng `ambulance_status`.

> **Cờ `active_mask_k`∈{0,1}** (đóng góp tinh chỉnh): phân biệt rõ xe *inactive* (khối toàn 0) với xe *active nhưng hàng đợi rỗng* — sentinel toàn 0 đơn thuần là nhập nhằng; trước đây chỉ phân biệt *ngầm* qua severity_norm=0.

**Hành động Worker:** $K{=}1$ → 1 chiều no‑op (xe duy nhất nhận toàn bộ $B_{\text{URLLC}}$); $K\ge2$ → **$K$ chiều logit per‑xe thuần** (không còn số hạng nhiệt độ $\beta$). **Phần thưởng:** §4.2 (tăng cường hinge §4.4).

## 5.3. Manager: chính sách liên‑lát (chỉ học `b_rrm`)

Manager xuất một hành động vô hướng $a_H$, giải mã qua sigmoid + affine:

$$b_{\text{rrm}}=B_{\min}+(B_{\max}-B_{\min})\,\sigma(a_H),\quad [B_{\min},B_{\max}]=[0.65,0.85].$$

`set_rrm_budget` áp **ba tầng kẹp**: (i) biên giải mã, (ii) sàn `B_RRM_FLOOR_BY_SEV[sev_ref]` (§4.5), (iii) biên khả thi $[\text{floor},\text{cap}]$. Trạng thái Manager $s_H$ có $6+(4K+1)$ chiều: $(\rho_U,\rho_e,\text{BLER},\text{sev\_ref\_norm},\text{AoI}_{\text{mean}},\text{AoI}_{\max})$ nối với $\lambda_{\text{global}}$.

> Vì $s_H$ chứa **$\lambda_{\text{global}}$** (per‑xe theo severity), Manager phân biệt được $[5,1,1]$ với $[5,5,5]$ *qua $\lambda$* dù vô hướng severity chỉ là max — phù hợp vai trò "ngân sách chung lái bởi xe nặng nhất".

## 5.4. Worker: phân chia nội‑lát thuần RL (masked softmax)

Pipeline đúng theo sơ đồ: $\text{sev}_k\to \text{obs}_k\to \pi_L \to \ell_k \to \text{softmax} \to w_k \to \text{PRB}_k$. **Softmax có mask**: chỉ tính trên các xe *active* (`_softmax(weights[active_idx])`), nên xe inactive nhận **chính xác 0** PRB — mạnh hơn công thức $m_k e^{z_k}/\sum_j m_j e^{z_j}$ thông thường. Sau softmax là *phân bổ số nguyên dư lớn nhất* (largest‑remainder) với sàn chống đói `PRB_MIN_QOS`, bảo đảm $\sum_k \text{PRB}_k=B_{\text{URLLC}}$ chính xác. **Không** có thứ tự severity cứng: nhận biết severity là *học hoàn toàn* qua obs (severity_k, $\lambda_k$) + gradient.

## 5.5. Cập nhật đối ngẫu hai thang thời gian (SMDP)

Phần thưởng Manager là **lợi tức SMDP** trong cửa sổ:

$$R_H(t)=\sum_{i=0}^{W-1}\gamma_L^{\,i}\,r^{\text{aug}}_{t+i},\qquad \gamma_H=\gamma_L^{W}.$$

**Tính nhất quán (không double‑discount):** chiết khấu hiệu dụng của phần thưởng tại bước toàn cục $\tau$ là $\gamma_H^{\lfloor\tau/W\rfloor}\gamma_L^{\tau\bmod W}=\gamma_L^{\tau}$ — hai tầng tái tạo *chính xác* chiết khấu phẳng. Bootstrap giá trị dùng cờ `terminated` (kết thúc thật, cả K xe tới đích), **không** dùng `truncated` (timeout 400 s là trạng thái tiếp diễn hợp lệ).

## 5.6. Ba solver ngang hàng PPO / TD3 / SAC

Cùng một bài toán/khung HRL, khác lõi RL:
- **PPO:** GAE + clipped surrogate + entropy; Manager update theo lô rollout (n_eff≈100).
- **TD3:** replay + clipped double‑Q + target smoothing + delayed actor; Manager/Worker update mỗi biên.
- **SAC:** replay + soft Bellman + nhiệt độ $\alpha$ tự điều chỉnh.

Thứ tự tốc độ học khoá: $\alpha_{\pi_H}(3\!\times\!10^{-5}) < \alpha_\lambda(2\!\times\!10^{-4}) < \alpha_{\pi_L}(3\!\times\!10^{-4})$ [Akyıldız 2024].

## 5.7. Tổ chức mã nguồn (SSOT)

Toàn bộ hằng số đặt tại `baselines/utils/config.py` (SSOT); mọi consumer import, **không** hardcode. Bố cục obs khoá bằng các hằng `OBS_*_IDX`/`AMB_*_OFFSET` và test `test_obs_layout.py`.

```
baselines/
  env/        oran_env.py, channel_model.py, queue_model.py, sumo_mobility.py, ...
  agents/     manager_agent.py, worker_agent.py, ppo_core.py, td3_agent.py, sac_agent.py, lagrangian.py
  solvers/    td3.py, sac.py, _common.py, train_offpolicy.py
  utils/      config.py (SSOT), obs.py, metrics.py, logger.py
  audit/      feasibility_oracle.py, runtime_oracle.py, closure_checks.py,
              eval_tail_bank.py, feasibility_eval.py
  train.py    (vòng PPO)
  tests/      >1000 ca kiểm thử
```

## 5.8. Môi trường mô phỏng `ORANEnv`

Giao diện kiểu Gym: `reset(seed, options)` (seed cả `rng` lẫn `channel.rng` ⟹ *tất định*), `step(action)` trả `(obs, reward, terminated, truncated, info)`. `info` xuất `c_vec`/`d_phi` $(4K+1)$ chiều, `active_mask`, `prb_per_amb`, … phục vụ dual ascent + đánh giá.

## 5.9. Vòng huấn luyện

Mỗi Manager step: dựng $s_H$ → `manager.act` → `b_rrm` → `set_rrm_budget`; chạy $W{=}10$ Worker step (overlay $\lambda_{\text{local}}$ vào obs, `worker.act`, `env.step`, tăng cường hinge, tích luỹ SMDP), rồi `on_manager_step_end` (dual ascent). PPO update sau rollout 1 s **không** reset env.

## 5.10. Tham số siêu (hyperparameters)

Xem **Phụ lục B**. Các giá trị tốc độ học/clip/entropy/ngưỡng C1–C5 được **khoá** và *không* tinh chỉnh tuỳ tiện trong sweep (kỷ luật tái lập).

---

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

---

# Chương 7. Kết luận và Hướng phát triển

**Kết luận.** Đồ án đã (i) *mô hình hoá* một cell O‑RAN đa xe cứu thương với kênh UMa, hàng đợi M/G/1 và AoI; (ii) *phát biểu* bài toán phân bổ PRB nhận biết nguy kịch dưới dạng CMDP với hàm mục tiêu eMBB log‑utility và năm họ ràng buộc per‑severity; (iii) *thiết kế* lời giải HRL hai thang thời gian (Manager học $b_{\text{rrm}}$, Worker chia nội‑lát thuần RL) với đối ngẫu Lagrange (primal hinge, dual signed) và ràng buộc ưu tiên cứng cho xe cứu thương; (iv) *triển khai* trên ba solver ngang hàng PPO/TD3/SAC; và (v) thiết lập *khung kiểm chứng trung thực* dựa trên khả thi sơ cấp theo severity. Tính đúng đắn ở mức *mô hình–công thức–thuật toán* đã được xác minh bằng oracle độc lập và bộ kiểm thử.

**Hướng phát triển.** (E3) AoI LCFS vs FCFS chạy lại SUMO; (E4) stress/robustness (tải/burst/nhiễu/hỏng cảm biến); chạy đầy đủ sweep W18–W23 để điền Bảng I/II với CI; mở rộng đa cell + handover; xác thực mức gói ns‑3; bổ sung cờ trạng thái và hàm mục tiêu đa‑severity tinh vi hơn.

---

# Tài liệu tham khảo

> Khóa BibTeX gợi ý; nhãn ✅ = tài liệu thật trong `documents/` hoặc chuẩn 3GPP/IEEE; 🟡 = giả định có cơ sở; 🔴 = declared. Tham chiếu `docs/REFERENCE_MAP.md`.

**Chuẩn 3GPP / O‑RAN (✅)**
- [TS 38.101‑1] 3GPP, *UE radio transmission and reception (FR1)* — $P_{\text{total}}$, $B_{\text{PRB}}$, $f_c$, $P_{\text{tx}}$.
- [TS 38.211] 3GPP, *Physical channels and modulation* — numerology $\mu$.
- [TS 38.214] 3GPP, *Physical layer procedures for data* — MCS, $D_{\text{DET}}$.
- [TR 38.901] 3GPP, *Study on channel model for 0.5–100 GHz* — UMa path‑loss/shadowing.
- [TS 22.261] 3GPP, *Service requirements for the 5G system* — ngưỡng độ tin cậy/QoS.
- [TS 28.541] 3GPP, *Management and orchestration; 5G NRM* — RRMPolicyRatio.
- [TS 23.501] 3GPP, *System architecture for the 5G System* — 5QI (softmax priority).
- [O‑RAN.WG4] O‑RAN Alliance — ngân sách fronthaul/transport.

**Học máy / Tối ưu (✅)**
- [Schulman 2017] *Proximal Policy Optimization Algorithms*, `1707.06347`.
- [Fujimoto 2018] *Addressing Function Approximation Error in Actor‑Critic (TD3)*, `fujimoto18a`.
- [Haarnoja 2018] *Soft Actor‑Critic Algorithms and Applications*, `1812.05905`.
- [Foundations of Deep RL] — GAE (Schulman 2016 vắng corpus → dùng nguồn này).
- [Akyıldız 2024] — two‑timescale HRL (Borkar 2008 vắng corpus).
- [Spoor 2025]; [Ding 2023] — Lagrangian dual ascent (Boyd/Tessler vắng corpus).
- [Yongshuai Liu 2020]; [Wen Wu 2020]; [Qiang Liu 2021] — CMDP cho mạng (Altman vắng corpus).
- [Kleinrock] *Queueing Systems* `9780470316887` — M/G/1 Pollaczek–Khinchine.

**Slicing / AoI / Y khoa (✅/🟡)**
- [Alsenwi 2022]; [Sohaib 2024] — eMBB log‑utility, traffic Poisson.
- [Qi 2024]; [Mlika 2022] — Age of Information (Kaul 2012 vắng corpus).
- [Hyoungju Ji 2017]; [Zexian Li 2018] — $\eta$, NF.
- [Weijian Zhou] — $R_{\text{REF}}$.
- [ACEM] *Australasian Triage Scale* — thang 5 mức.

**Nguồn vắng corpus (KHÔNG trích như đã có):** Kaul 2012, Altman 1999, Boyd 2004, Tessler 2018, Borkar 2008, FeUdal 2017, Parekh&Gallager 1993, Jain 1984, Schulman 2016, Holm 1979, Hedges 1985.

---

# Phụ lục A. Bảng tra Công thức → Mã nguồn (Equation‑to‑Code Ledger)

| Công thức | Mã nguồn |
|---|---|
| Reward $\alpha_e\log(1+R/R_{\text{REF}})$ | `env/oran_env.py` (`_compute…`/`step`) |
| Capacity $\eta B\log_2(1+\text{SINR})$ | `env/channel_model.py` |
| M/G/1 PK $E[D_q]$ | `env/queue_model.py` |
| AoI $\Delta(t)=t-U(t)$ | `env/aoi_tracker.py` |
| Augmented reward (hinge) | `agents/lagrangian.py::augmented_reward` |
| Dual ascent (signed) | `agents/lagrangian.py::on_manager_step_end` |
| $b_{\text{rrm}}$ decode + 3‑tầng kẹp | `agents/manager_agent.py::decode_manager_action`, `env/oran_env.py::set_rrm_budget` |
| masked softmax intra‑slice | `env/oran_env.py::_prb_split_intra_slice` |
| SMDP $R_H$, $\gamma_H=\gamma_L^{W}$ | `train.py`, `utils/config.py` |
| Verdict khả thi điểm‑12 | `audit/feasibility_eval.py` |

> Chi tiết đầy đủ: `docs/EQUATION_TO_CODE_LEDGER.md`.

# Phụ lục B. Bảng tham số đầy đủ

| Tham số | Giá trị | Nguồn/nhãn |
|---|---|---|
| $P_{\text{total}}$, $B_{\text{PRB}}$ | 273 PRB, 360 kHz | ✅ TS 38.101‑1/38.211 |
| $f_c$ | 3.5 GHz (n78) | ✅ TS 38.101‑1 |
| $R_{\text{cell}}$ | 1000 m (UMa) | 🔴 declared (D25) |
| $\eta$, NF | 0.75, 7 dB | 🟡 Ji 2017 / Li 2018 |
| $D_{\text{DET}}/D_{\text{FH}}/D_{\text{BH}}/D_{\text{STOCH}}$ | 0.07/0.1/0.1/0.05 ms | 🟡/🔴 |
| $R_{\text{REF}}$, $R_{\min}$(C3) | 100 Mbps, 10 Mbps | 🟡 / Gate‑7 |
| $\gamma_L$, $\gamma_H$ | 0.99, $0.99^{10}\approx0.9044$ | ✅ |
| $\alpha_{\pi_H}/\alpha_\lambda/\alpha_{\pi_L}$ | $3\!\times\!10^{-5}/2\!\times\!10^{-4}/3\!\times\!10^{-4}$ | heuristic, thứ tự [Akyıldız 2024] |
| PPO clip, $\gamma$, GAE‑$\lambda$, K‑epochs, minibatch | 0.2, 0.99, 0.95, 10, 64 | ✅ |
| $\Lambda_{\max}$ | 10 | ✅ |
| $[B_{\min},B_{\max}]$, `B_RRM_FLOOR_BY_SEV` | [0.65,0.85]; {0.65,0.70,0.75,0.80,0.85} | thiết kế ưu tiên xe |
| obs dim | $20+11K+F$ (K=1→32, K=3→54) | SSOT `config.py` |

# Phụ lục C. Danh mục hình và mã nguồn sinh hình

Bảy hình ở `docs/figures/*.dot` (Graphviz) → xuất `.svg`/`.png` bằng:

```bash
cd docs/figures
for f in *.dot; do dot -Tpdf "$f" -o "${f%.dot}.pdf"; done   # khuyến nghị PDF cho LaTeX
```

| Tệp | Hình | Mục dùng |
|---|---|---|
| 01_pipeline | 1.1 | Ch.1 |
| 02_oran_arch | 2.1 | Ch.2 |
| 03_env_internal | 3.1 | Ch.3 |
| 04_cmdp | 4.1 | Ch.4 |
| 05_state_action | 5.1 | Ch.5 |
| 06_training_loop | 5.2 | Ch.5 |
| 07_one_episode | 6.1 | Ch.6 |

---

*Báo cáo này được sinh từ trạng thái mã nguồn đã kiểm chứng (baseline hiện hành). Mọi tuyên bố định lượng "kết quả huấn luyện" cần được điền sau khi chạy sweep, kèm khoảng tin cậy và kiểm định thống kê đã nêu ở Chương 6 — tuân thủ liêm chính khoa học.*
