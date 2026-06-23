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

