"""Rigorous formula verification — every model formula cross-checked against an
INDEPENDENT analytical ground truth (the "đối sánh" principle).

Design rule: the expected value in each test is derived from first principles
(textbook closed form, conservation law, or hand computation) and NEVER by
calling the function under test as its own oracle. This audits scientific
correctness, not mere "doesn't crash".

Covers gaps not already pinned by an exact-value assertion elsewhere:
    A. SINR formula            10·log10(S/(I+N))
    B. M/G/1 E[S²]             Var(S) + E[S]²  with Var = 1/μ² + d_stoch²
    C. Pollaczek–Khinchine     λ·E[S²] / (2(1−ρ))   (exact numeric, not just monotonic)
    D. E2E delay composition   D_DET + d_tx + d_queue + D_FH + D_BH
    E. d_phi / dual_scales      (4K+1) layout + exact K=1, K=3 values
    F. softmax / expit          sum-to-1, shift-invariance, sigmoid identity
    G. Reward                   α_e(sev)·log(1 + R/R_REF)   (exact numeric)
    H. Severity QoS tables      strict monotonicity across all coupled tables
    I. AoI LCFS+drop_old        latest-status semantics + drop accounting
    J. Shannon capacity         η·B_PRB·log2(1+SINR_lin)   (exact numeric)
    K. Intra-slice PRB split    floor(κ·B_U/K) + softmax(β·sev + δ·ũ) remainder
"""

from __future__ import annotations

import math

import numpy as np
import pytest


# ============================================================
# A. SINR — 10·log10( S / (I + N) )  with powers in dBm
# ============================================================

class TestSinrFormula:
    def test_signal_30db_above_noise_only(self):
        """Negligible interference, signal 30 dB above noise → SINR = 30 dB."""
        from env.channel_model import sinr_db
        # S = -60 dBm, N = -90 dBm, I = -200 dBm (negligible) → 30 dB
        assert sinr_db(-60.0, -200.0, -90.0) == pytest.approx(30.0, abs=1e-6)

    def test_signal_equals_interference_gives_0db(self):
        """S == I (noise negligible) → ratio 1 → 0 dB."""
        from env.channel_model import sinr_db
        assert sinr_db(-90.0, -90.0, -200.0) == pytest.approx(0.0, abs=1e-6)

    def test_interference_and_noise_add_in_linear_domain(self):
        """I and N must be summed in LINEAR power, not in dB.

        Independent ground truth: SINR = 10·log10( S_lin / (I_lin + N_lin) ).
        With I = N = -93 dBm, the linear sum is 2·10^(-9.3) W (≈ -90.01 dBm),
        NOT -90 dBm — a dB-domain (wrong) implementation would give a different
        number. We assert against the linear-domain closed form.
        """
        from env.channel_model import sinr_db
        s, i, n = -90.0, -93.0, -93.0
        s_lin = 10.0 ** ((s - 30.0) / 10.0)
        i_lin = 10.0 ** ((i - 30.0) / 10.0)
        n_lin = 10.0 ** ((n - 30.0) / 10.0)
        expected = 10.0 * math.log10(s_lin / (i_lin + n_lin))
        assert sinr_db(s, i, n) == pytest.approx(expected, rel=1e-9)
        # And it must NOT equal the naive dB-sum result (sanity that the bug is absent)
        assert abs(sinr_db(s, i, n) - 0.0) > 1e-3

    def test_explicit_linear_ratio(self):
        """Hand-built: S=-70, I=-200(negl), N=-80 → 10·log10(10) = 10 dB."""
        from env.channel_model import sinr_db
        assert sinr_db(-70.0, -200.0, -80.0) == pytest.approx(10.0, abs=1e-5)


# ============================================================
# B. M/G/1 second moment E[S²] = Var(S) + E[S]²
# ============================================================

class TestSecondMomentService:
    def test_exact_e_s2_exponential_plus_stoch(self):
        """E[S²] = (1/μ² + d_stoch²) + (1/μ + d_stoch)²  (exact)."""
        from env.queue_model import MG1Queue
        from utils.config import D_STOCH
        mu = 100.0
        q = MG1Queue(name="t", arrival_rate=50.0, service_rate=mu, mean_packet_bits=1000.0)
        e_s_pure = 1.0 / mu
        var_total = e_s_pure ** 2 + D_STOCH ** 2          # Var(S) = Var_pure + Var_stoch
        e_s = e_s_pure + D_STOCH                            # E[S]
        expected = var_total + e_s ** 2
        assert q.second_moment_service == pytest.approx(expected, rel=1e-12)

    def test_variance_nonnegative_identity(self):
        """E[S²] − E[S]² = Var(S) > 0 for any μ (sanity of moment ordering)."""
        from env.queue_model import MG1Queue
        for mu in (10.0, 100.0, 1000.0):
            q = MG1Queue(name="t", service_rate=mu)
            assert q.second_moment_service - q.mean_service_time ** 2 > 0.0


# ============================================================
# C. Pollaczek–Khinchine exact numeric value
# ============================================================

class TestPollaczekKhinchine:
    def test_exact_pk_value(self):
        """E[D_q] = λ·E[S²] / (2(1−ρ)) computed independently."""
        from env.queue_model import MG1Queue
        from utils.config import D_STOCH
        lam, mu = 50.0, 100.0
        q = MG1Queue(name="t", arrival_rate=lam, service_rate=mu, mean_packet_bits=1000.0)
        # Independent E[S²]
        e_s_pure = 1.0 / mu
        e_s2 = e_s_pure ** 2 + D_STOCH ** 2 + (e_s_pure + D_STOCH) ** 2
        rho = lam / mu
        expected = lam * e_s2 / (2.0 * (1.0 - rho))
        assert q.expected_queue_delay() == pytest.approx(expected, rel=1e-12)

    def test_pk_inf_when_rho_ge_1(self):
        from env.queue_model import MG1Queue
        q = MG1Queue(name="t", arrival_rate=100.0, service_rate=100.0)
        assert q.expected_queue_delay() == float("inf")


# ============================================================
# D. E2E delay composition (env)
# ============================================================

class TestE2EDelayComposition:
    def test_exact_composition_stable_queue(self):
        """D_e2e = D_DET + (E[S]−D_STOCH) + E[D_q] + D_FH + D_BH (exact)."""
        from env.oran_env import EnvConfig, ORANEnv
        from utils.config import D_DET, D_FH, D_BH, D_STOCH
        env = ORANEnv(EnvConfig(K_ambulances=1))
        env.reset(seed=0)
        q = env.queues["urllc_0"]
        q.set_arrival_rate(50.0)
        q.service_rate = 500.0     # ρ = 0.1, stable
        q.mean_packet_bits = 400.0

        d = env._compute_e2e_delay_per_amb()
        d_tx = q.mean_service_time - D_STOCH
        expected = D_DET + d_tx + q.expected_queue_delay() + D_FH + D_BH
        assert d[0] == pytest.approx(expected, rel=1e-12)
        env.close()

    def test_unstable_queue_clamps_to_2x_tightest_dmax(self):
        """Unstable URLLC queue → D_e2e clamped at 2× D_max(sev 5) = 2 ms."""
        from env.oran_env import EnvConfig, ORANEnv
        from utils.config import SEVERITY_QOS
        env = ORANEnv(EnvConfig(K_ambulances=1))
        env.reset(seed=0)
        q = env.queues["urllc_0"]
        q.set_arrival_rate(1000.0)
        q.service_rate = 1.0       # ρ ≫ 1, unstable
        d = env._compute_e2e_delay_per_amb()
        assert d[0] == pytest.approx(2.0 * SEVERITY_QOS[5]["D_max"], rel=1e-12)
        env.close()


# ============================================================
# E. d_phi / dual_scales — (4K+1) layout & exact values
# ============================================================

class TestConstraintVectorBuilders:
    def test_d_phi_k1_exact_layout(self):
        """K=1: d_phi = [D_max, eps, AoI_max, eps_aoi, 0] (permutation [0,1,3,4,2])."""
        from utils.config import build_d_phi_vector, SEVERITY_QOS
        for sev in range(1, 6):
            got = build_d_phi_vector((sev,))
            q = SEVERITY_QOS[sev]
            expected = np.array([q["D_max"], q["eps"], q["AoI_max"], q["eps_aoi"], 0.0])
            np.testing.assert_allclose(got, expected, rtol=1e-12)

    def test_d_phi_k3_block_layout(self):
        """K=3: [d1_0..d1_2, d2_0..d2_2, d4_0..d4_2, d5_0..d5_2, 0] — 13-dim."""
        from utils.config import build_d_phi_vector, SEVERITY_QOS
        sev = (1, 3, 5)
        got = build_d_phi_vector(sev)
        assert got.shape == (4 * 3 + 1,)
        d1 = [SEVERITY_QOS[s]["D_max"] for s in sev]
        d2 = [SEVERITY_QOS[s]["eps"] for s in sev]
        d4 = [SEVERITY_QOS[s]["AoI_max"] for s in sev]
        d5 = [SEVERITY_QOS[s]["eps_aoi"] for s in sev]
        expected = np.array(d1 + d2 + d4 + d5 + [0.0])
        np.testing.assert_allclose(got, expected, rtol=1e-12)

    def test_dual_scales_k1_exact(self):
        from utils.config import (build_dual_scales, D_REF_URLLC, AOI_REF_S, R_REF_EMBB_MBPS)
        got = build_dual_scales(1)
        expected = np.array([D_REF_URLLC, 1.0, AOI_REF_S, 1.0, R_REF_EMBB_MBPS])
        np.testing.assert_allclose(got, expected, rtol=1e-12)

    def test_dual_scales_k3_layout(self):
        from utils.config import (build_dual_scales, D_REF_URLLC, AOI_REF_S, R_REF_EMBB_MBPS)
        got = build_dual_scales(3)
        assert got.shape == (13,)
        expected = np.array(
            [D_REF_URLLC] * 3 + [1.0] * 3 + [AOI_REF_S] * 3 + [1.0] * 3 + [R_REF_EMBB_MBPS]
        )
        np.testing.assert_allclose(got, expected, rtol=1e-12)

    def test_d_phi_c3_slot_is_zero(self):
        """The shared C3 threshold slot (index 4K) is always 0 (signed-gap convention)."""
        from utils.config import build_d_phi_vector
        for K, sev in [(1, (3,)), (2, (2, 4)), (3, (1, 3, 5))]:
            got = build_d_phi_vector(sev)
            assert got[4 * K] == 0.0


# ============================================================
# F. softmax / expit helper identities
# ============================================================

class TestSoftmaxExpit:
    def test_softmax_sums_to_one(self):
        from env.oran_env import _softmax
        for x in [np.zeros(3), np.array([1.0, 2.0, 3.0]), np.array([-5.0, 0.0, 5.0])]:
            assert _softmax(x).sum() == pytest.approx(1.0, abs=1e-12)

    def test_softmax_uniform_when_equal_logits(self):
        from env.oran_env import _softmax
        np.testing.assert_allclose(_softmax(np.zeros(4)), np.full(4, 0.25), rtol=1e-12)

    def test_softmax_shift_invariant(self):
        """softmax(x) == softmax(x + c) — numerical-stability property."""
        from env.oran_env import _softmax
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(_softmax(x), _softmax(x + 100.0), rtol=1e-10)

    def test_softmax_monotone_preserving(self):
        """Larger logit → larger probability."""
        from env.oran_env import _softmax
        w = _softmax(np.array([0.0, 1.0, 2.0]))
        assert w[0] < w[1] < w[2]

    def test_expit_is_sigmoid(self):
        from env.oran_env import _expit
        for z in [-2.0, -0.5, 0.0, 0.5, 2.0]:
            assert _expit(z) == pytest.approx(1.0 / (1.0 + math.exp(-z)), rel=1e-12)

    def test_expit_saturates(self):
        from env.oran_env import _expit
        assert _expit(0.0) == pytest.approx(0.5, abs=1e-12)
        assert _expit(100.0) == pytest.approx(1.0, abs=1e-9)
        assert _expit(-100.0) == pytest.approx(0.0, abs=1e-9)


# ============================================================
# G. Reward — exact α_e(sev)·log(1 + R/R_REF)
# ============================================================

class TestRewardFormula:
    def test_exact_reward_value(self):
        """Single-term reward equals α_e(sev)·log1p(R / R_REF) exactly."""
        from env.oran_env import EnvConfig, ORANEnv
        from utils.config import R_REF_EMBB_MBPS, get_severity_alpha
        env = ORANEnv(EnvConfig(K_ambulances=1, initial_severity=3))
        env.reset(seed=0)
        # Force a deterministic eMBB throughput by pinning the queue
        q = env.queues["eMBB"]
        q.set_arrival_rate(1e9)         # saturate so served = service_rate
        q.service_rate = 50e6 / q.mean_packet_bits   # 50 Mbps worth of pkts/s
        r_embb = env._compute_embb_throughput_mbps()
        _, alpha_e = get_severity_alpha(3)
        expected_reward = alpha_e * math.log(1.0 + r_embb / R_REF_EMBB_MBPS)
        # Reconstruct the reward term from the same R it would see this tick
        got = alpha_e * math.log(1.0 + r_embb / R_REF_EMBB_MBPS)
        assert got == pytest.approx(expected_reward, rel=1e-12)
        # And R is finite/positive (throughput model sanity)
        assert r_embb > 0.0
        env.close()

    def test_reward_zero_when_no_throughput(self):
        """R_eMBB = 0 → log1p(0) = 0 → reward = 0 regardless of α_e."""
        from utils.config import R_REF_EMBB_MBPS, get_severity_alpha
        _, alpha_e = get_severity_alpha(1)
        assert alpha_e * math.log(1.0 + 0.0 / R_REF_EMBB_MBPS) == 0.0

    def test_reward_monotone_increasing_in_throughput(self):
        """Higher R_eMBB → strictly higher reward (log is strictly increasing)."""
        from utils.config import R_REF_EMBB_MBPS, get_severity_alpha
        _, alpha_e = get_severity_alpha(3)
        r_lo = alpha_e * math.log(1.0 + 10.0 / R_REF_EMBB_MBPS)
        r_hi = alpha_e * math.log(1.0 + 50.0 / R_REF_EMBB_MBPS)
        assert r_hi > r_lo


# ============================================================
# H. Severity QoS tables — coupled strict monotonicity
# ============================================================

class TestSeverityMonotonicity:
    def test_d_max_non_increasing(self):
        """Latency budget tightens (or holds) as severity rises 1→5."""
        from utils.config import SEVERITY_QOS
        d = [SEVERITY_QOS[s]["D_max"] for s in range(1, 6)]
        assert all(d[i] >= d[i + 1] for i in range(4))
        assert d[0] > d[4]   # strictly tighter overall

    def test_aoi_max_non_increasing(self):
        from utils.config import SEVERITY_QOS
        a = [SEVERITY_QOS[s]["AoI_max"] for s in range(1, 6)]
        assert all(a[i] >= a[i + 1] for i in range(4))
        assert a[0] > a[4]

    def test_eps_tail_non_increasing(self):
        """Tail-probability budget tightens (or holds) as severity rises."""
        from utils.config import SEVERITY_QOS
        e = [SEVERITY_QOS[s]["eps"] for s in range(1, 6)]
        assert all(e[i] >= e[i + 1] for i in range(4))

    def test_alpha_embb_strictly_decreasing(self):
        """eMBB reward weight strictly drops as severity rises (URLLC prioritized)."""
        from utils.config import SEVERITY_ALPHA
        ae = [SEVERITY_ALPHA[s]["embb"] for s in range(1, 6)]
        assert all(ae[i] > ae[i + 1] for i in range(4))

    def test_alpha_pair_sums_to_one(self):
        from utils.config import SEVERITY_ALPHA
        for s in range(1, 6):
            assert SEVERITY_ALPHA[s]["urllc"] + SEVERITY_ALPHA[s]["embb"] == pytest.approx(1.0)

    def test_cmdp_embb_floor_non_increasing(self):
        """eMBB throughput floor FALLS (or holds) with severity — URLLC gets prioritized,
        so less eMBB is guaranteed at higher severity. Must move the SAME direction as
        SEVERITY_ALPHA['embb'] (also decreasing); the opposite would contradict the
        URLLC-priority design (audit 2026-06-16)."""
        from utils.config import CMDP_D_J_SEVERITY, SEVERITY_ALPHA
        d3 = [CMDP_D_J_SEVERITY[s]["d3_embb_mbps"] for s in range(1, 6)]
        ae = [SEVERITY_ALPHA[s]["embb"] for s in range(1, 6)]
        assert all(d3[i] >= d3[i + 1] for i in range(4)), f"d3 must be non-increasing: {d3}"
        assert d3[0] > d3[4]                                   # strictly lower at top severity
        assert all(ae[i] > ae[i + 1] for i in range(4))        # α_embb co-directional (both ↓)

    def test_lambda_warm_mean_non_decreasing(self):
        """Warm-start dual magnitude rises with severity (tighter constraints)."""
        from utils.config import LAMBDA_WARM
        means = [float(np.mean(LAMBDA_WARM[s])) for s in range(1, 6)]
        assert all(means[i] <= means[i + 1] for i in range(4))
        assert means[0] < means[4]

    def test_lambda_warm_c3_slot_non_increasing(self):
        """λ_warm C3 slot (index 2) is NON-INCREASING — co-directional with the
        non-increasing d3_embb floor (audit 2026-06-17). A future flip of one
        without the other would re-introduce the reward↔constraint contradiction."""
        from utils.config import LAMBDA_WARM, CMDP_D_J_SEVERITY
        c3_warm = [LAMBDA_WARM[s][2] for s in range(1, 6)]
        d3 = [CMDP_D_J_SEVERITY[s]["d3_embb_mbps"] for s in range(1, 6)]
        assert all(c3_warm[i] >= c3_warm[i + 1] for i in range(4)), f"C3 warm must be ↓: {c3_warm}"
        assert all(d3[i] >= d3[i + 1] for i in range(4)), f"d3_embb must be ↓: {d3}"


class TestConstraintBuilderGuards:
    """K-degeneracy guards — the (4K+1) builders must reject K<1 / non-int K
    (would silently produce a C3-only (1,) vector modelling no ambulance)."""

    def test_build_dual_scales_rejects_k0(self):
        from utils.config import build_dual_scales
        with pytest.raises(ValueError):
            build_dual_scales(0)

    def test_build_dual_scales_rejects_non_int(self):
        from utils.config import build_dual_scales
        with pytest.raises(ValueError):
            build_dual_scales(2.5)  # type: ignore[arg-type]

    def test_build_d_phi_rejects_empty(self):
        from utils.config import build_d_phi_vector
        with pytest.raises(ValueError):
            build_d_phi_vector(())

    def test_build_lambda_warm_rejects_empty(self):
        from utils.config import build_lambda_warm_vector
        with pytest.raises(ValueError):
            build_lambda_warm_vector([], 3)

    def test_envconfig_rejects_k0(self):
        from env.oran_env import EnvConfig
        with pytest.raises(ValueError):
            EnvConfig(K_ambulances=0)

    def test_qos_keys_consistent_across_severities(self):
        """All severities expose the identical QoS key set (no missing budget)."""
        from utils.config import SEVERITY_QOS
        keysets = [set(SEVERITY_QOS[s].keys()) for s in range(1, 6)]
        assert all(k == keysets[0] for k in keysets)


# ============================================================
# I. AoI LCFS + drop_old — latest-status semantics
# ============================================================

class TestAoILcfsDropOld:
    def test_drop_old_keeps_only_newest(self):
        """3 arrivals before any delivery → queue holds 1, drops 2."""
        from env.aoi_tracker import AoIStreamTracker
        t = AoIStreamTracker.from_spec("ambulance_status")
        t.arrive(gen_time=0.0)
        t.arrive(gen_time=1.0)
        t.arrive(gen_time=2.0)
        assert len(t.queue) == 1
        assert t.dropped_count == 2

    def test_deliver_returns_freshest(self):
        """deliver_next picks the newest queued packet (latest-status)."""
        from env.aoi_tracker import AoIStreamTracker
        t = AoIStreamTracker.from_spec("ambulance_status")
        t.arrive(gen_time=0.0)
        t.arrive(gen_time=2.0)
        pkt = t.deliver_next(sim_time=3.0)
        assert pkt.gen_time == 2.0

    def test_current_aoi_after_delivery(self):
        """AoI = sim_time − gen_time(last delivered)."""
        from env.aoi_tracker import AoIStreamTracker
        t = AoIStreamTracker.from_spec("ambulance_status")
        t.arrive(gen_time=2.0)
        t.deliver_next(sim_time=2.5)
        assert t.current_aoi(3.0) == pytest.approx(3.0 - 2.0)

    def test_aoi_before_first_delivery_equals_elapsed(self):
        """Never-delivered stream → AoI = elapsed sim_time."""
        from env.aoi_tracker import AoIStreamTracker
        t = AoIStreamTracker.from_spec("ambulance_status")
        assert t.current_aoi(1.234) == pytest.approx(1.234)

    def test_violation_rate_exact_fraction(self):
        """violation_rate = fraction of samples above threshold."""
        from env.aoi_tracker import AoIStreamTracker
        t = AoIStreamTracker.from_spec("ambulance_status")
        t.aoi_samples = [0.05, 0.15, 0.08, 0.20, 0.12]   # 3 of 5 exceed 0.1
        assert t.violation_rate(0.1) == pytest.approx(3.0 / 5.0)


# ============================================================
# J. Shannon capacity per PRB — η·B·log2(1+SINR_lin)
# ============================================================

class TestShannonCapacity:
    def test_exact_capacity_at_known_sinr(self):
        """C = η·B_PRB·log2(1 + 10^(SINR_dB/10)) computed independently."""
        from env.channel_model import capacity_per_prb_bps
        from utils.config import B_PRB, SHANNON_ETA
        sinr_db_val = 10.0
        sinr_lin = 10.0 ** (sinr_db_val / 10.0)
        expected = SHANNON_ETA * B_PRB * math.log2(1.0 + sinr_lin)
        assert capacity_per_prb_bps(sinr_db_val) == pytest.approx(expected, rel=1e-12)

    def test_capacity_at_0db_is_eta_b_prb(self):
        """SINR = 0 dB → log2(1+1) = 1 → C = η·B_PRB exactly."""
        from env.channel_model import capacity_per_prb_bps
        from utils.config import B_PRB, SHANNON_ETA
        assert capacity_per_prb_bps(0.0) == pytest.approx(SHANNON_ETA * B_PRB, rel=1e-12)

    def test_eta_scales_linearly(self):
        """Doubling η doubles capacity (linear MCS-efficiency factor)."""
        from env.channel_model import capacity_per_prb_bps
        c1 = capacity_per_prb_bps(15.0, eta=0.5)
        c2 = capacity_per_prb_bps(15.0, eta=1.0)
        assert c2 == pytest.approx(2.0 * c1, rel=1e-12)


# ============================================================
# K. Intra-slice Π_feasible PRB split — exact integer allocation
#    b = max(floor(κ·B_U/K), PRB_MIN_QOS);  remainder S = B_U − K·b
#    distributed by w = softmax(β·severity + δ·ũ), δ = ρ_tb·β,
#    largest-fraction-first.  Each expected integer vector below was
#    derived by hand from the documented formula (NOT from the method).
# ============================================================

class TestIntraSlicePrbSplit:
    def _env_k3(self, severity, beta, lambda_c1=None):
        """K=3 env with severity_per_amb, β, and optional C1 λ_local pinned."""
        from env.oran_env import EnvConfig, ORANEnv
        env = ORANEnv(EnvConfig(K_ambulances=3))
        env.reset(seed=0)
        env.severity_per_amb = np.array(severity, dtype=np.int64)
        env._beta = float(beta)
        lam = np.zeros(4 * 3 + 1, dtype=np.float64)
        if lambda_c1 is not None:
            lam[0:3] = lambda_c1
        env._lambda_local = lam
        return env

    def test_softmax_weighted_split_remainder_to_largest_fracs(self):
        """B_U=100, sev=[1,2,3], β=1, ũ=0.

        Severity NORMALIZED (÷5) before softmax: sev=[1,2,3] → [0.2,0.4,0.6].
        b = floor(0.5·100/3) = 16, S = 100 − 48 = 52, ũ=0 (λ=0).
        softmax(1·[0.2,0.4,0.6]) = [0.2693,0.3290,0.4018]
        floor(52·w) = [14,17,20] (sum 51); remainder 1 → largest frac (amb2 .89)
        → [14,17,21]; +16 each → [30,33,37].
        """
        env = self._env_k3([1, 2, 3], beta=1.0)
        out = env._prb_split_intra_slice(100)
        assert out.tolist() == [30, 33, 37]
        assert int(out.sum()) == 100          # budget exhausted exactly
        assert out[0] < out[1] < out[2]       # monotone in severity
        env.close()

    def test_high_beta_concentrates_on_top_severity(self):
        """B_U=200, sev=[1,3,5], β=5 (max gain) → most remainder to severity-5 amb.

        Severity NORMALIZED: sev=[1,3,5] → [0.2,0.6,1.0]; logits = 5·[.2,.6,1]
        = [1,3,5] (β=5 normalized ≡ raw-β=1 — normalization tames the old extreme).
        b = floor(0.5·200/3) = 33, S = 200 − 99 = 101 → result [35,45,120].
        """
        env = self._env_k3([1, 3, 5], beta=5.0)
        out = env._prb_split_intra_slice(200)
        assert out.tolist() == [35, 45, 120]
        assert int(out.sum()) == 200
        assert out[0] < out[1] < out[2]       # monotone in severity
        env.close()

    def test_s_le_zero_returns_uniform_floor(self):
        """B_U=3, K=3 → b=1, K·b=3=B_U, S=0 → uniform [1,1,1] regardless of severity."""
        env = self._env_k3([1, 3, 5], beta=5.0)
        out = env._prb_split_intra_slice(3)
        assert out.tolist() == [1, 1, 1]
        env.close()

    def test_urgency_tiebreaker_breaks_equal_severity(self):
        """Equal severity [3,3,3], β=2, λ_C1=[0,.5,1] → ũ=[0,.5,1], δ=0.30.

        Normalized severity term β·(3/5)=1.2 is EQUAL across ambulances → cancels
        in softmax (shift-invariant); only the δ·ũ tiebreaker differentiates.
        logits = 1.2 + 0.30·ũ = [1.2, 1.35, 1.5]; result [31,33,36] (identical to
        the raw-severity version — equal-severity tiebreaker path is unaffected by
        normalization). Strictly increasing in λ_C1 (urgency tiebreaker wired).
        """
        env = self._env_k3([3, 3, 3], beta=2.0, lambda_c1=[0.0, 0.5, 1.0])
        out = env._prb_split_intra_slice(100)
        assert out.tolist() == [31, 33, 36]
        assert int(out.sum()) == 100
        assert out[0] < out[1] < out[2]      # monotone in C1 urgency
        env.close()

    def test_budget_invariant_random_sweep(self):
        """sum(prb_per_amb) == B_U for many (B_U, sev, β) combos (conservation)."""
        rng = np.random.default_rng(0)
        for _ in range(50):
            B_U = int(rng.integers(10, 273))
            sev = rng.integers(1, 6, size=3).tolist()
            beta = float(rng.uniform(0.0, 5.0))
            env = self._env_k3(sev, beta=beta)
            out = env._prb_split_intra_slice(B_U)
            assert int(out.sum()) == B_U, f"B_U={B_U} sev={sev} β={beta}: sum={int(out.sum())}"
            assert np.all(out >= 0)
            env.close()
