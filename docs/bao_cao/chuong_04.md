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

