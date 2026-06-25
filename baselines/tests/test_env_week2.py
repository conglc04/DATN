"""Week 2 unit tests — env/channel_model.py, env/queue_model.py.

NOTE (cleanup 2026-06-25): bare-function formula coverage (path loss, LOS
probability, thermal noise, capacity, M/G/1 queue, tail bounds, SliceQueueManager)
was superseded by the exact-value-pinned tests/test_formulas_channel_queue_exact.py
and removed from here to avoid duplicate/weaker-assertion drift. The traffic_gen.py
generator tests were removed too — those generators (gen_urllc_*, gen_embb_image_mec,
gen_mmtc, mix_traffic, aggregate_*) are not used by the live env (ambulance_status
URLLC traffic is generated inline in oran_env.py); only Packet + gen_embb_video
remain live (via env/bystander_traffic.py) and have no dedicated unit test here.

What remains below is coverage with no equivalent elsewhere:
  - TestChannelModelClass: the BaseStation/ChannelModel class-level wrapper
    (vs. the bare pl_uma/pl_umi_los/etc. functions tested exactly elsewhere)
  - TestPhase3SanitySingleCell: standalone e2e channel+queue integration sanity
"""

from __future__ import annotations

import numpy as np


class TestChannelModelClass:
    def test_path_loss_macro_uses_uma(self):
        from env.channel_model import BaseStation, ChannelModel
        cm = ChannelModel(shadowing=False, rng=np.random.default_rng(0))
        bs = BaseStation(cell_id=0, x=0.0, y=0.0, h=25.0, layer="macro")
        pl, scenario = cm.path_loss((200.0, 0.0), bs)
        assert scenario == "UMa"

    def test_path_loss_micro_picks_los_or_nlos(self):
        from env.channel_model import BaseStation, ChannelModel
        cm = ChannelModel(shadowing=False, rng=np.random.default_rng(42))
        bs = BaseStation(cell_id=0, x=0.0, y=0.0, h=10.0, layer="micro")
        scenarios = set()
        for _ in range(50):
            _, scen = cm.path_loss((100.0, 50.0), bs)
            scenarios.add(scen)
        # Over 50 trials we should see both branches given d≈112m
        assert scenarios.issubset({"UMi_LOS", "UMi_NLOS"})

    def test_rx_power_decreases_with_distance(self):
        from env.channel_model import BaseStation, ChannelModel
        cm = ChannelModel(shadowing=False, rng=np.random.default_rng(0))
        bs = BaseStation(cell_id=0, x=0.0, y=0.0, layer="macro", tx_power_dbm=46.0)
        p_close = cm.receive_power_dbm((100.0, 0.0), bs)
        p_far = cm.receive_power_dbm((1000.0, 0.0), bs)
        assert p_close > p_far


# ============================================================
# Integration sanity: channel + queue at severity-5
# ============================================================


class TestPhase3SanitySingleCell:
    """Sanity: at severity-5 with r_min^URLLC=0.6, single ambulance, expect HOL < 1ms."""

    def test_e2e_breakdown_under_1ms(self):
        from env.channel_model import capacity_per_prb_bps
        from env.queue_model import MG1Queue
        from utils.config import D_FH, D_BH, D_DET, P_TOTAL

        # SINR=15dB typical ambulance near cell
        sinr_db_val = 15.0
        capacity = capacity_per_prb_bps(sinr_db_val)   # ~0.93 * log2(1+31.6) → ≈ 3.5 Mbps

        # PRB for URLLC at severity-5: r_min = 0.6 → 0.6 * 273 = 163 PRB
        urllc_prb = int(0.6 * P_TOTAL)

        # M/G/1 with λ=50 pkt/s (DENM steady) + 400B packets
        q = MG1Queue(name="urllc_sev5", arrival_rate=50.0, mean_packet_bits=400 * 8)
        q.update_service_rate(urllc_prb, capacity)

        assert q.is_stable
        e2e = D_DET + D_FH + D_BH + q.hol_delay()
        assert e2e < 1e-3, f"E2E at severity-5 = {e2e*1e3:.3f}ms exceeds 1ms target"
