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

