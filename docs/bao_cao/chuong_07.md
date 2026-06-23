# Chương 7. Kết luận và Hướng phát triển

**Kết luận.** Đồ án đã (i) *mô hình hoá* một cell O‑RAN đa xe cứu thương với kênh UMa, hàng đợi M/G/1 và AoI; (ii) *phát biểu* bài toán phân bổ PRB nhận biết nguy kịch dưới dạng CMDP với hàm mục tiêu eMBB log‑utility và năm họ ràng buộc per‑severity; (iii) *thiết kế* lời giải HRL hai thang thời gian (Manager học $b_{\text{rrm}}$, Worker chia nội‑lát thuần RL) với đối ngẫu Lagrange (primal hinge, dual signed) và ràng buộc ưu tiên cứng cho xe cứu thương; (iv) *triển khai* trên ba solver ngang hàng PPO/TD3/SAC; và (v) thiết lập *khung kiểm chứng trung thực* dựa trên khả thi sơ cấp theo severity. Tính đúng đắn ở mức *mô hình–công thức–thuật toán* đã được xác minh bằng oracle độc lập và bộ kiểm thử.

**Hướng phát triển.** (E3) AoI LCFS vs FCFS chạy lại SUMO; (E4) stress/robustness (tải/burst/nhiễu/hỏng cảm biến); chạy đầy đủ sweep W18–W23 để điền Bảng I/II với CI; mở rộng đa cell + handover; xác thực mức gói ns‑3; bổ sung cờ trạng thái và hàm mục tiêu đa‑severity tinh vi hơn.

