# AUDIT CHI TIẾT FILE `metrics(38).csv`

## Mục tiêu

Tài liệu này tổng hợp kết quả audit file `metrics(38).csv`, tập trung vào:

1. Kiểm tra tính toàn vẹn của dữ liệu.
2. Đánh giá xu hướng học của RL.
3. Kiểm tra reward, penalty, dual-ascent và PPO diagnostics.
4. Xác định các điểm bất thường có thể gây sai lệch kết quả.
5. Truy quét nguyên nhân gốc trong code.
6. Đưa ra danh sách yêu cầu sửa và unit test bắt buộc.

---

# 1. Kết luận điều hành

File gồm:

- 100 episode.
- 53 cột.
- Không có `NaN` hoặc `Inf`.
- Các bộ đếm Worker, Manager và MAC nhìn chung nhất quán.
- K=1 được xử lý đúng ở Worker actor.

Tuy nhiên, run này **chưa đủ điều kiện để tiếp tục train dài, so sánh solver hoặc sử dụng làm kết quả khoa học**.

Các vấn đề chính:

| Mức độ | Vấn đề |
|---|---|
| P0 – Critical | `penalty_total` luôn âm nhưng lại bị trừ khỏi `reward_base`, có nguy cơ biến penalty thành reward dương |
| P0 – Critical | `delivery_success_rate_amb0` chỉ khoảng 0.74–0.91, rất thấp nếu đây là reliability thực |
| P1 – High | Manager giảm mạnh `b_rrm`, khiến PRB URLLC giảm, eMBB tăng nhưng delay/violation URLLC xấu đi |
| P1 – High | Chưa có bằng chứng policy thực sự severity-aware |
| P1 – High | `manager_explained_variance` âm ở 96/100 episode |
| P1 – High | Dual-ascent và persistent multiplier có dấu hiệu phản ứng không đúng hoặc gần như đóng băng |
| P2 – Medium | Reward normalization đang trôi mạnh |
| P2 – Medium | Một số episode có lệch MAC tick ở boundary |

## Phán quyết

> Dừng ở smoke test. Không chạy 1.500 episode × nhiều seed trước khi audit và sửa reward sign, reliability metric, dual-ascent và Manager critic.

Nếu `ep_reward` hiện tại chính là reward đưa vào PPO thì checkpoint của run này phải xem là **không hợp lệ**.

---

# 2. Kiểm tra tính toàn vẹn dữ liệu

## 2.1. Schema và episode

- `step = 0...99`
- Không thiếu episode.
- Không có `NaN`.
- Không có `Inf`.
- Severity nhất quán giữa:

```text
severity_init
severity_amb0
severity_final
severity_per_amb
```

Phân bố severity:

| Severity | Số episode |
|---:|---:|
| 1 | 16 |
| 2 | 15 |
| 3 | 28 |
| 4 | 18 |
| 5 | 23 |

Phân bố chưa cân bằng tuyệt đối nhưng đủ cho smoke test.

---

## 2.2. Đồng hồ mô phỏng

Các quan hệ sau nhìn chung đúng:

\[
\text{episode\_duration\_s}
=
\frac{\text{worker\_steps}}{100}
\]

\[
\text{worker\_n\_samples}
=
\text{worker\_steps}
\]

\[
\text{manager\_n\_samples}
=
\text{manager\_n\_decisions}
\]

\[
\text{reward\_per\_step}
=
\frac{\text{ep\_reward}}{\text{worker\_steps}}
\]

Chưa thấy lỗi nghiêm trọng về lệch clock Manager–Worker.

---

## 2.3. Worker actor skip với K=1

Toàn bộ file có:

```text
worker_actor_skipped_k1 = 1
worker_actor_loss       = 0
worker_entropy          = 0
worker_clip_fraction    = 0
worker_approx_kl        = 0
```

Đây là hành vi đúng với K=1 vì Worker không có bài toán phân chia PRB giữa nhiều xe.

---

# 3. P0 – Dấu của penalty mâu thuẫn với reward

## 3.1. Bằng chứng

Trong toàn bộ 100 episode:

```text
penalty_total < 0
```

đồng thời thỏa:

\[
\boxed{
\text{ep\_reward}
=
\text{reward\_base}
-
\text{penalty\_total}
}
\]

Ví dụ:

```text
reward_base   =   6,685.94
penalty_total = -18,059.21
ep_reward     =  24,745.15
```

Tính đúng theo file:

\[
6,685.94 - (-18,059.21) = 24,745.15
\]

Như vậy một penalty âm đang bị trừ thêm lần nữa và trở thành phần thưởng dương.

---

## 3.2. Hai convention hợp lệ

### Convention A – penalty là reward contribution âm

```python
penalty_total <= 0
reward_aug = reward_base + penalty_total
```

### Convention B – penalty là cost dương

```python
penalty_cost >= 0
reward_aug = reward_base - penalty_cost
```

Hiện tại file thể hiện:

```python
penalty_total <= 0
reward_aug = reward_base - penalty_total
```

Đây là sự pha trộn sai giữa hai convention.

---

## 3.3. Ảnh hưởng theo severity

Nếu dùng convention A:

\[
r_{\text{corrected}}
=
\text{reward\_base}
+
\text{penalty\_total}
\]

| Severity | Reward đang log | Reward sau sửa dấu |
|---:|---:|---:|
| 1 | 132,920 | 132,906 |
| 2 | 91,631 | 91,346 |
| 3 | 77,347 | 73,099 |
| 4 | 54,334 | 27,119 |
| 5 | 24,767 | **−5,495** |

Toàn bộ episode severity 5 có thể chuyển từ reward dương thành reward âm nếu penalty được cộng đúng theo dấu.

---

## 3.4. Mối nguy hiểm

Nếu reward sai dấu đi vào PPO:

```text
Violation tăng
    ↓
penalty_total âm hơn
    ↓
reward_base - penalty_total
    ↓
reward tăng
```

PPO có thể học rằng vi phạm QoS tạo reward tốt hơn.

Đây là lỗi phá hỏng bản chất CMDP/Lagrangian.

---

## 3.5. Truy quét code

Truy theo chuỗi:

```text
constraint cost/slack
    ↓
dual term từng constraint
    ↓
penalty_total
    ↓
reward_aug trong env.step()
    ↓
reward trước normalize
    ↓
reward sau normalize
    ↓
rollout_buffer.add(reward=...)
    ↓
GAE / returns
    ↓
critic target
```

Tìm các mẫu:

```python
reward_base - penalty_total
reward -= penalty_total
penalty_total -= dual_term
dual_term = -lambda_value * violation
```

---

## 3.6. Unit test bắt buộc

Chọn một trong hai convention.

### Nếu penalty là cost dương

```python
assert penalty_cost >= 0
assert reward_aug == reward_base - penalty_cost
```

### Nếu penalty là contribution âm

```python
assert penalty_total <= 0
assert reward_aug == reward_base + penalty_total
```

Test monotonic:

```python
reward_low_violation = compute_reward(violation=0.0)
reward_high_violation = compute_reward(violation=1.0)

assert reward_high_violation <= reward_low_violation
```

---

# 4. P0 – Reliability có thể đang rất thấp

`delivery_success_rate_amb0`:

| Thống kê | Giá trị |
|---|---:|
| Min | 0.7386 |
| Median | 0.8598 |
| Mean | 0.8581 |
| Max | 0.9088 |

Tỷ lệ thất bại tương ứng:

\[
1 - \text{delivery success}
=
9.1\% \text{ đến } 26.1\%
\]

Nếu đây thực sự là packet delivery reliability thì mức này không phù hợp với URLLC cho xe cứu thương.

---

## 4.1. Điểm bất thường

Severity thấp có thể gần như không vi phạm delay nhưng delivery success vẫn chỉ khoảng 85–90%.

Điều này cho thấy:

- `viol_rate` có thể chỉ là delay violation.
- Reliability violation chưa được log riêng.
- Packet drop có thể không được tính vào constraint hiện tại.
- Hệ thống có thể trông như không vi phạm nhưng đang mất nhiều packet.

---

## 4.2. Nguyên nhân cần kiểm tra

1. Denominator tính sai.
2. Packet trước khi xe active vẫn bị đưa vào mẫu số.
3. Packet sau khi episode kết thúc bị tính fail.
4. Retransmission attempt được tính như packet mới.
5. Packet drop vì queue/deadline/channel không được phân loại.
6. Metric là MAC decode success chứ không phải application delivery.
7. C2 dùng metric khác với `delivery_success_rate_amb0`.

---

## 4.3. Log cần bổ sung

```text
packets_generated
packets_enqueued
packets_transmitted
packets_delivered
packets_dropped_queue
packets_dropped_channel
packets_dropped_deadline
packets_expired
reliability_value
reliability_threshold
reliability_slack
reliability_viol_rate
```

---

# 5. P1 – Manager đang kéo `b_rrm` xuống thấp

## 5.1. Ý nghĩa `b_rrm`

`b_rrm` là ngân sách/tỷ lệ tài nguyên vô tuyến Manager cấp cho lát URLLC.

Gần đúng:

\[
B_{\mathrm{URLLC}}
\approx
b_{\mathrm{rrm}}
\times
P_{\mathrm{TOTAL}}
\]

Với K=1:

\[
\text{PRB}_{\mathrm{amb0}}
\approx
B_{\mathrm{URLLC}}
\]

---

## 5.2. Xu hướng đầu–cuối run

| Metric | 10 episode đầu | 10 episode cuối | Thay đổi |
|---|---:|---:|---:|
| `manager_b_rrm_mean` | 0.399 | 0.111 | −72.1% |
| `mean_prb_amb0` | 108.5 | 29.9 | −72.5% |
| `mean_embb_mbps` | 182.4 | 243.2 | +33.4% |
| `mean_e2e_ms` | 0.315 ms | 0.435 ms | +38.2% |
| `viol_rate` | 0.000145 | 0.003755 | khoảng 25.8 lần |
| `c3_viol_rate` | 0.000689 | 0.000092 | −86.6% |

Ngay cả sau khi kiểm soát severity, xu hướng vẫn tồn tại:

- `b_rrm` giảm.
- PRB ambulance giảm.
- eMBB throughput tăng.
- URLLC delay tăng.
- URLLC violation tăng.
- C3 violation giảm.

---

## 5.3. Quan hệ vật lý

\[
\operatorname{corr}
(b_{\mathrm{rrm}}, \text{mean\_prb\_amb0})
\approx 1
\]

Mapping gần đúng:

\[
\text{PRB}_{\mathrm{amb0}}
\approx
269 \times b_{\mathrm{rrm}}
\]

Các quan hệ đáng chú ý:

| Quan hệ | Spearman |
|---|---:|
| `b_rrm` và E2E delay | −0.922 |
| `b_rrm` và eMBB throughput | −0.663 |
| `b_rrm` và C3 violation | +0.651 |
| `b_rrm` và delivery success | +0.258 |

---

## 5.4. Chuỗi nguy hiểm

```text
Manager giảm b_rrm
        ↓
URLLC nhận ít PRB
        ↓
eMBB nhận nhiều tài nguyên
        ↓
eMBB throughput tăng
C3 violation giảm
        ↓
nhưng delay/loss/violation ambulance tăng
```

Nếu penalty URLLC sai dấu hoặc quá yếu, PPO sẽ tiếp tục học theo hướng ưu tiên eMBB.

---

## 5.5. Chưa thể kết luận action chạm lower bound

`manager_b_rrm_mean` nhỏ nhất khoảng:

```text
0.071224
```

Nhưng đây là mean action cả episode.

`manager_clip_fraction` là PPO ratio clipping, không phải action bound clipping.

Cần log:

```text
action_raw
action_squashed
action_rescaled
action_after_clip
B_RRM_MIN
B_RRM_MAX
action_hit_lower_bound
action_hit_upper_bound
```

---

# 6. P1 – Chưa có bằng chứng policy severity-aware

Sau khi kiểm soát theo thời gian train, chênh lệch `b_rrm` so với severity 1:

| Severity | Chênh lệch ước lượng |
|---:|---:|
| 2 | +0.0515 |
| 3 | −0.0079 |
| 4 | −0.0170 |
| 5 | +0.0136 |

Severity 5 không nhận ngân sách cao hơn rõ ràng so với severity thấp.

Severity 2 thậm chí có hiệu ứng cao hơn severity 5.

---

## 6.1. Severity 5 đầu–cuối run

| Metric | Giai đoạn đầu | Giai đoạn cuối |
|---|---:|---:|
| `b_rrm` | 0.288 | 0.126 |
| E2E delay | 0.346 ms | 0.425 ms |
| `viol_rate` | 0.00244 | 0.01469 |
| reward/step | 3.071 | 2.738 |

Severity cao nhất:

- nhận ít tài nguyên hơn;
- delay tăng;
- violation tăng;
- reward/step giảm.

Severity 4 cũng có reward/step giảm theo thời gian.

Trong khi severity 1–3 lại có dấu hiệu tăng reward/step.

---

## 6.2. Nguyên nhân cần truy quét

### A. Severity không có trong Manager observation

Kiểm tra vector ngay trước:

```python
manager.select_action(obs_manager)
```

Xác nhận:

- severity đúng index;
- không bị ghi đè;
- không luôn cố định;
- scaling đúng;
- severity được dùng tại action time.

### B. Observation normalization làm mất tín hiệu severity

Log:

```text
severity_raw
severity_normalized
manager_obs_before_normalization
manager_obs_after_normalization
```

### C. Manager không quan sát lambda hiện tại

Nếu multiplier là trạng thái CMDP nhưng policy không thấy lambda, bài toán trở thành partially observable.

Cần đảm bảo:

```text
lambda_local_current_severity
```

được chèn đúng vào:

```text
obs
next_obs
```

trước khi lưu trajectory.

### D. Update từng episode gây catastrophic forgetting

Mỗi episode chỉ có một severity cố định.

Nếu PPO update sau từng episode:

```text
episode severity 5
→ update toàn bộ bằng severity 5

episode severity 1
→ update toàn bộ bằng severity 1
```

Critic và actor có thể dao động giữa các miền severity.

Cân nhắc:

- rollout chứa hỗn hợp severity;
- stratified batch;
- cân bằng severity;
- condition critic rõ ràng trên severity.

---

# 7. P1 – Manager critic đang thất bại

## 7.1. Explained variance

`manager_explained_variance`:

| Thống kê | Giá trị |
|---|---:|
| Episode âm | 96/100 |
| Median | −0.884 |
| Mean | −2.220 |
| Min | −21.636 |
| Max | 0.138 |

Công thức:

\[
EV
=
1
-
\frac{\operatorname{Var}(y-\hat{y})}
{\operatorname{Var}(y)}
\]

Diễn giải:

- `EV = 1`: critic tốt.
- `EV = 0`: không tốt hơn dự đoán trung bình.
- `EV < 0`: tệ hơn dự đoán trung bình.

96% episode âm là dấu hiệu critic chưa học được value function.

---

## 7.2. Critic loss nhỏ không đủ

Ví dụ có episode:

```text
manager_critic_loss        ≈ 0.089
manager_explained_variance ≈ -18.338
```

Loss nhỏ có thể do target normalized có variance nhỏ.

Critic vẫn có thể dự đoán sai cấu trúc.

PPO actor phụ thuộc vào:

\[
A_t = R_t - V(s_t)
\]

Critic sai làm advantage nhiễu.

---

## 7.3. Nguyên nhân có xác suất cao

1. Reward sai dấu.
2. Reward không stationary.
3. Critic không quan sát severity.
4. Critic không quan sát lambda.
5. Update mỗi episode chỉ có một severity.
6. GAE terminal mask sai.
7. `terminated` và `truncated` bị xử lý sai.
8. Reward normalization thay đổi liên tục.
9. Value target hoặc value clipping sai.

---

## 7.4. Unit test GAE

Tạo trajectory nhân tạo 3–5 step.

So sánh:

```text
returns_expected
advantages_expected
returns_actual
advantages_actual
```

Test riêng:

- true terminal;
- time-limit truncation;
- bootstrap từ `V(next_state)`;
- fixed decision horizon;
- episode kết thúc do xe rời cell;
- episode kết thúc do timeout.

---

# 8. P1 – Dual-ascent và persistent multiplier đáng ngờ

## 8.1. Lambda lớn ngay lần đầu severity xuất hiện

Lần xuất hiện đầu:

| Severity | C1 | C2 | C4 | C5 |
|---:|---:|---:|---:|---:|
| 1 | 0 | 0.00956 | 0 | 0 |
| 2 | 0 | 0.07995 | 0 | 0.01954 |
| 3 | 0 | 0.69996 | 0 | 0.59956 |
| 4 | 0.3711 | 1.5011 | 0.7901 | 1.5026 |
| 5 | 1.5177 | 2.2002 | 1.1711 | 2.0041 |

Có ba khả năng:

1. Lambda được khởi tạo bằng severity prior.
2. Run resume từ checkpoint cũ.
3. State bị rò từ run trước hoặc reset không sạch.

Nếu là fresh run, phải giải thích các giá trị này.

Metadata cần log:

```text
fresh_start
resume
checkpoint_path
lambda_initialization
lambda_bank_loaded
reward_rms_loaded
observation_rms_loaded
```

---

## 8.2. Một số lambda gần như đóng băng

Các multiplier như C2/C5 thay đổi rất ít theo thời gian.

Có thể do:

- dual learning rate quá nhỏ;
- slack scale quá nhỏ;
- EMA gần 0;
- log lambda trước update;
- lambda bị clip;
- multiplier là hệ số severity cố định nhưng bị đặt tên như biến học.

---

## 8.3. Violation severity 5 tăng nhưng lambda giảm

Severity 5 có:

- `b_rrm` giảm;
- delay tăng;
- `viol_rate` tăng;
- nhưng một số lambda như C1/C4 có xu hướng giảm.

Nếu C1/C4 tương ứng với constraint đang xấu đi thì đây là phản ứng ngược.

---

## 8.4. Nguyên nhân cần kiểm tra

### Sai dấu slack

Đối với constraint:

\[
c(s,a)\le d
\]

phải dùng:

```python
slack = cost - threshold
lambda_new = max(0, lambda_old + alpha * slack)
```

Không được dùng:

```python
slack = threshold - cost
lambda_new = max(0, lambda_old + alpha * slack)
```

nếu không đổi dấu update tương ứng.

### EMA làm loãng vi phạm

Episode có một số spike mạnh nhưng phần lớn tick không vi phạm.

Mean slack có thể âm dù event violation rate dương.

### Decay nhầm severity bank

Kiểm tra lambda severity 5 có bị decay trong episode severity khác hay không.

### Index severity sai

Kiểm tra:

```python
lambda_bank[severity - 1]
```

và:

```python
lambda_bank[severity]
```

có bị dùng lẫn không.

### Mapping constraint–metric sai

Cần xác nhận:

```text
C1 ↔ metric nào
C2 ↔ metric nào
C3 ↔ metric nào
C4 ↔ metric nào
C5 ↔ metric nào
```

---

## 8.5. `lambda_global_C3_shared` luôn bằng 0

Trong toàn bộ file:

```text
lambda_global_C3_shared = 0
```

nhưng:

```text
c3_viol_rate > 0
```

ở tất cả episode.

Điều này chưa đủ kết luận lỗi vì dual có thể update theo signed slack chứ không theo event rate.

Tuy nhiên, file hiện tại không đủ chứng minh C3 hoạt động đúng.

Cần log:

```text
C3_raw_value
C3_threshold
C3_slack
C3_ema_slack
lambda_C3_before
lambda_C3_after
lambda_C3_projected
```

---

# 9. P1 – Penalty phản ứng ngược với violation severity 5

Trong severity 5:

- violation tăng;
- delay tăng;
- `b_rrm` giảm;
- nhưng `penalty_total` có xu hướng ít âm hơn.

Ví dụ:

```text
-18,000 → -13,000 → -10,000
```

Tức là mức phạt giảm khi QoS xấu đi.

Tương quan trong severity 5 giữa `penalty_total` và `viol_rate` là dương đáng kể.

Vì penalty là số âm, điều này nghĩa là violation tăng nhưng penalty trở nên ít âm hơn.

---

## 9.1. Nguyên nhân có thể

1. Lambda giảm.
2. EMA slack sai dấu.
3. Penalty dùng raw slack âm.
4. Normalize penalty theo duration sai.
5. Dùng average QoS thay vì tail violation.
6. Penalty tính trước khi cập nhật packet violation.
7. Log penalty và violation lệch time index.
8. Penalty bị clip.
9. Penalty dùng constraint normalized khác với metric log.

---

# 10. P2 – Reward normalization đang trôi

`reward_norm_std`:

| Giai đoạn | Trung bình |
|---|---:|
| 10 episode đầu | 411.5 |
| 10 episode cuối | 501.1 |

Tăng khoảng:

\[
21.8\%
\]

Trong khi:

| Metric | Thay đổi đầu → cuối |
|---|---:|
| Raw `ep_reward` | +3.1% |
| `reward_per_step` | −2.4% |
| `ep_reward_normalized` | −34.9% |

Normalized reward giảm mạnh chủ yếu do denominator tăng.

Không được dùng `ep_reward_normalized` một mình để kết luận policy tốt hay xấu.

---

## 10.1. Reward distribution phụ thuộc severity

Reward/step trung bình:

| Severity | Reward/step |
|---:|---:|
| 1 | 16.02 |
| 2 | 11.56 |
| 3 | 9.27 |
| 4 | 6.26 |
| 5 | 2.86 |

Global RunningMeanStd đang trộn năm phân phối rất khác nhau.

Điều này tạo non-stationary target cho critic.

---

## 10.2. Hướng xử lý

- Log reward RMS mean/std.
- Không dùng normalized reward làm metric vật lý.
- Cân bằng severity trong rollout.
- Kiểm tra critic có severity trong observation.
- Có thể freeze RMS sau warm-up.
- Cân nhắc normalize từng reward component trước khi cộng.

---

# 11. Episode nguy hiểm nhất

Một episode severity 5 có:

```text
step                         = 50
severity                     = 5
manager_b_rrm_mean           = 0.07122
mean_prb_amb0                = 18.95
mean_e2e_ms                  = 0.60579
viol_rate                    = 0.09926
delivery_success_rate        = 0.73858
aoi_viol_rate                = 0.00713
c3_viol_rate                 = 0.000427
reward_base                  = 11,535
penalty_total                = -13,163
ep_reward                    = +24,699
manager_clip_fraction        = 0.0423
manager_approx_kl            = 0.00586
```

Chuỗi nguy hiểm:

1. Severity cao nhất.
2. URLLC budget gần thấp nhất.
3. Delay violation khoảng 9.93%.
4. Delivery success chỉ khoảng 73.86%.
5. Reward vẫn dương lớn.
6. Penalty âm làm reward tăng.

Nếu policy nhận reward tốt trong trạng thái này thì objective không bảo vệ đúng ambulance.

---

# 12. Những điểm không phải lỗi chính

## 12.1. PPO không bị numerical explosion

Manager:

- entropy giảm nhẹ;
- KL nhỏ;
- clip fraction thấp;
- actor loss không bùng nổ;
- không có NaN.

Vấn đề chính có khả năng nằm ở:

- reward objective;
- dual update;
- critic target;
- observation;
- normalization.

Không phải exploding gradient đơn thuần.

---

## 12.2. Approximate KL âm nhỏ

Một số episode có `manager_approx_kl` âm nhỏ.

Approximate KL estimator có thể âm do sampling noise.

Không phải lỗi độc lập nếu độ lớn nhỏ.

---

## 12.3. Worker critic K=1

Worker critic có EV gần 0.

Nếu Worker actor bị skip và critic không phục vụ thành phần khác, có thể skip cả Worker critic ở K=1 để giảm compute.

---

# 13. Bất thường MAC tick ở boundary

Phần lớn episode thỏa:

\[
\text{active\_mac\_ticks}
=
20
\times
\text{worker\_steps}
\]

Một số episode lệch đúng khoảng 19 tick.

Khả năng:

- Worker interval cuối chỉ có 1 MAC tick active.
- Xe vào/ra cell giữa một Worker window.
- Episode kết thúc không thẳng hàng với Worker period.

Không phải lỗi nghiêm trọng nhưng cần unit test và chú thích.

---

# 14. Sơ đồ nguyên nhân gốc có xác suất cao

```text
Penalty âm nhưng bị trừ
             │
             ├── Violation có thể làm reward tăng
             │
             ▼
URLLC penalty không chống được lợi ích eMBB
             │
             ▼
Manager giảm b_rrm
             │
       ┌─────┴────────────┐
       ▼                  ▼
PRB URLLC giảm        PRB eMBB tăng
       │                  │
       ▼                  ▼
delay/loss tăng       throughput tăng
       │                  │
       ▼                  ▼
severity 4–5 xấu      C3 violation giảm
       └─────┬────────────┘
             ▼
critic học target sai hoặc non-stationary
             ▼
explained variance âm
```

Persistent lambda/EMA có thể làm penalty tiếp tục yếu đi khi violation tăng.

---

# 15. Danh sách truy quét code theo ưu tiên

## P0.1 – Reward sign

Kiểm tra:

```text
env.step()
compute_reward()
compute_augmented_reward()
compute_constraint_penalty()
rollout_buffer.add()
reward_normalizer.update()
GAE computation
```

Log cùng một timestep:

```text
reward_base_raw
constraint_cost_C1...C5
slack_C1...C5
lambda_C1...C5
dual_term_C1...C5
penalty_cost_positive
reward_aug_raw
reward_train
```

Invariant:

\[
\frac{\partial r_{\mathrm{aug}}}
{\partial \text{violation}}
\le 0
\]

---

## P0.2 – Reliability

Xác nhận:

```python
delivery_success_rate = delivered / denominator
```

Kiểm tra denominator có tính:

- packet trước khi active;
- packet sau khi kết thúc;
- retransmission;
- expired packet;
- queue drop;
- channel drop;
- deadline drop.

---

## P1.1 – Action collapse

Log mỗi Manager decision:

```text
severity
b_rrm_raw
b_rrm_after_squash
b_rrm_after_rescale
b_rrm_after_clip
B_URLLC_PRB
B_eMBB_PRB
PRB_demand_URLLC
queue_URLLC
channel_quality
```

Tính:

```text
lower_bound_hit_rate
upper_bound_hit_rate
```

---

## P1.2 – Severity observation

Assertion:

```python
assert manager_obs[SEVERITY_IDX] == expected_severity_scaled
```

Kiểm tra cả:

```text
obs
next_obs
actor input
critic input
rollout buffer
```

---

## P1.3 – Dual bank

Test severity isolation:

```text
Update severity 5:
- bank severity 5 thay đổi;
- bank severity 1–4 giữ nguyên.
```

Test update direction:

```python
if slack > 0 and lambda_before < lambda_max:
    assert lambda_after >= lambda_before
```

Test fresh reset:

```python
fresh_env.reset_all_duals()
assert all_lambdas == configured_initial_values
```

---

## P1.4 – Critic và GAE

Log:

```text
value_pred_mean
value_pred_std
returns_mean
returns_std
advantages_mean
advantages_std
corr(value_pred, returns)
terminal_count
truncated_count
bootstrap_value
```

---

# 16. Các cột CSV cần bổ sung

```text
reward_base_raw
penalty_cost_raw
dual_term_raw
reward_aug_raw
reward_train_mean
reward_train_std
reward_norm_mean
reward_norm_std

C1_value
C1_threshold
C1_slack
C1_ema

C2_value
C2_threshold
C2_slack
C2_ema

C3_value
C3_threshold
C3_slack
C3_ema

C4_value
C4_threshold
C4_slack
C4_ema

C5_value
C5_threshold
C5_slack
C5_ema

lambda_C1_before
lambda_C1_after
lambda_C2_before
lambda_C2_after
lambda_C3_before
lambda_C3_after
lambda_C4_before
lambda_C4_after
lambda_C5_before
lambda_C5_after

action_raw_mean
action_after_clip_mean
action_lower_bound_rate
action_upper_bound_rate

reliability_viol_rate
packets_generated
packets_delivered
packets_dropped_channel
packets_dropped_queue
packets_dropped_deadline
```

---

# 17. Acceptance gate trước khi train dài

Chỉ tiếp tục training khi đạt đủ:

1. Xác nhận convention của penalty.
2. Xác nhận reward thật sự đưa vào PPO.
3. Violation tăng không thể làm reward tăng.
4. Giải thích được delivery success 0.74–0.91.
5. Có reliability violation metric riêng.
6. Manager critic EV có xu hướng tiến về 0 rồi dương.
7. Log được action lower/upper bound hit rate.
8. Log được slack và lambda before/after.
9. Fresh-run/reset multiplier được kiểm soát.
10. Severity được xác nhận có trong actor và critic observation.
11. GAE terminal/truncation unit test pass.
12. Không còn episode severity 5 vừa vi phạm nặng vừa nhận reward dương lớn do penalty sign.

---

# 18. Yêu cầu Claude thực hiện

Claude cần làm các bước sau theo thứ tự.

## Bước 1 – Audit reward end-to-end

Xác định chính xác:

```text
reward_base
penalty
dual_term
reward_aug
reward_train
reward_normalized
```

Chỉ rõ dấu và đơn vị của từng đại lượng.

Sửa theo một convention duy nhất.

---

## Bước 2 – Thêm invariant test

```python
assert reward_when_violation_high <= reward_when_violation_low
```

khi các yếu tố khác giữ nguyên.

---

## Bước 3 – Audit rollout buffer

Xác nhận giá trị đưa vào:

```python
rollout_buffer.add(reward=...)
```

là reward nào.

Không được chỉ sửa CSV logging nếu PPO vẫn nhận reward sai.

---

## Bước 4 – Audit reliability

Giải thích chính xác công thức:

```text
delivery_success_rate_amb0
```

và mapping của reliability constraint.

---

## Bước 5 – Audit dual-ascent

Với từng constraint C1–C5, log:

```text
raw_value
threshold
slack
EMA slack
lambda_before
lambda_after
projection
severity bank index
```

---

## Bước 6 – Audit Manager observation

In schema observation và index của:

```text
severity
lambda_local
queue
channel
delay
reliability
AoI
eMBB state
```

Kiểm tra obs và next_obs cùng schema.

---

## Bước 7 – Audit critic/GAE

Viết trajectory unit test thủ công.

Kiểm tra:

```text
terminated
truncated
bootstrap
GAE mask
returns
advantages
```

---

## Bước 8 – Chạy lại smoke test

Sau khi sửa:

```text
1 seed × 50–100 episode × K=1
```

Chỉ cần kiểm tra correctness trước.

Không chạy nhiều seed hoặc 1.500 episode ngay.

---

# 19. Kết luận cuối cùng

## File hiện tại có thể chứng minh

- Pipeline chạy.
- K=1 Worker actor skip đúng.
- Manager action tác động trực tiếp đến PRB.
- Giảm `b_rrm` làm tăng eMBB và làm URLLC xấu đi.
- Logging có tính nhất quán số học ở nhiều cột.

## File hiện tại không thể chứng minh

- PPO đang học đúng.
- Policy severity-aware.
- Reliability được bảo vệ.
- Dual-ascent enforce constraint.
- Policy hội tụ.
- Reward tăng nghĩa là học tốt.
- Run đủ điều kiện báo cáo khoa học.
- Có thể so sánh công bằng PPO, TD3 và SAC.

## Đánh giá cuối

```text
Smoke pipeline correctness : PASS
Learning correctness       : FAIL
Reward correctness         : FAIL / cần xác minh end-to-end
Dual-ascent correctness    : NOT PROVEN
Severity-aware behavior    : NOT PROVEN
Scientific reporting       : FAIL
Long training readiness    : FAIL
```
