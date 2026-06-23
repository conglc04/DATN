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

