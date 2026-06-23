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
