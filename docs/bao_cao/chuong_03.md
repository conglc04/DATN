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

