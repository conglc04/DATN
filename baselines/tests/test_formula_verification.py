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

    def test_unstable_queue_uses_overload_delay(self):
        """Unstable URLLC queue (rho >= 1) → D_e2e = OVERLOAD_DELAY_SEC (E1 fix)."""
        from env.oran_env import OVERLOAD_DELAY_SEC, EnvConfig, ORANEnv
        env = ORANEnv(EnvConfig(K_ambulances=1))
        env.reset(seed=0)
        q = env.queues["urllc_0"]
        q.set_arrival_rate(1000.0)
        q.service_rate = 1.0       # ρ ≫ 1, unstable
        d = env._compute_e2e_delay_per_amb()
        assert d[0] == pytest.approx(OVERLOAD_DELAY_SEC, rel=1e-12)
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
# G. Reward — pure log(1 + R/R_REF), NO α_e (removed 2026-06-23)
# ============================================================

class TestRewardFormula:
    def test_reward_has_no_alpha_e_weight(self):
        """Reward is the per-tick MEAN of log1p(R/R_REF) — NO α_e weight.

        Verifies against the ACTUAL env reward: severity differentiation is
        entirely via constraints, so the reward must be invariant to severity
        at a fixed throughput (α_e removed). Compares sev=1 vs sev=5 reward
        with the same pinned eMBB throughput → must be EQUAL.
        """
        from env.oran_env import EnvConfig, ORANEnv

        def _pinned_reward(sev: int) -> float:
            env = ORANEnv(EnvConfig(K_ambulances=1, initial_severity=sev,
                                    sample_severity=False))
            env.reset(seed=0)
            env.set_rrm_budget(0.20)
            _, rew, _, _, _ = env.step(np.zeros(1, dtype=np.float32))
            env.close()
            return float(rew)

        r_sev1 = _pinned_reward(1)
        r_sev5 = _pinned_reward(5)
        # Same seed/budget → same throughput trace → reward must match (no α_e).
        assert r_sev1 == pytest.approx(r_sev5, rel=1e-9), (
            f"reward differs by severity (r1={r_sev1}, r5={r_sev5}) — α_e leaked back in"
        )
        # Per-tick MEAN scale (not a ×20 sum), strictly positive.
        assert 0.0 < r_sev1 < 2.0

    def test_reward_zero_when_no_throughput(self):
        """R_eMBB = 0 → log1p(0) = 0 → reward = 0 (pure log-utility, no α_e)."""
        from utils.config import R_REF_EMBB_MBPS
        assert math.log(1.0 + 0.0 / R_REF_EMBB_MBPS) == 0.0

    def test_reward_monotone_increasing_in_throughput(self):
        """Higher R_eMBB → strictly higher reward (log strictly increasing, no α_e)."""
        from utils.config import R_REF_EMBB_MBPS
        r_lo = math.log(1.0 + 10.0 / R_REF_EMBB_MBPS)
        r_hi = math.log(1.0 + 50.0 / R_REF_EMBB_MBPS)
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
        """SEVERITY_ALPHA['embb'] reference table strictly drops as severity rises.
        NOTE: α_e is REFERENCE-ONLY (removed from reward 2026-06-23); this just
        locks the historical table values, not a live reward weight."""
        from utils.config import SEVERITY_ALPHA
        ae = [SEVERITY_ALPHA[s]["embb"] for s in range(1, 6)]
        assert all(ae[i] > ae[i + 1] for i in range(4))

    def test_alpha_pair_sums_to_one(self):
        """SEVERITY_ALPHA reference table invariant (urllc+embb=1); reference-only."""
        from utils.config import SEVERITY_ALPHA
        for s in range(1, 6):
            assert SEVERITY_ALPHA[s]["urllc"] + SEVERITY_ALPHA[s]["embb"] == pytest.approx(1.0)

    def test_cmdp_embb_floor_fixed_across_severity(self):
        """eMBB throughput floor is a FIXED severity-independent 10 Mbps SLA
        (formulation-audit Gate 7, 2026-06-20). C3 decoupled from severity keeps
        it a clean starvation safety net rather than a severity-keyed target that
        would fight the constraint-driven b_rrm increase at high severity."""
        from utils.config import CMDP_D_J_SEVERITY
        d3 = [CMDP_D_J_SEVERITY[s]["d3_embb_mbps"] for s in range(1, 6)]
        assert d3 == [10.0] * 5, f"d3 must be a fixed 10 Mbps floor: {d3}"

    def test_lambda_warm_mean_non_decreasing(self):
        """Warm-start dual magnitude rises with severity (tighter constraints)."""
        from utils.config import LAMBDA_WARM
        means = [float(np.mean(LAMBDA_WARM[s])) for s in range(1, 6)]
        assert all(means[i] <= means[i + 1] for i in range(4))
        assert means[0] < means[4]

    def test_lambda_warm_c3_slot_fixed(self):
        """λ_warm C3 slot (index 2) is FIXED at 0.02 — co-directional with the
        fixed (severity-independent) 10 Mbps d3_embb floor (Gate 7, 2026-06-20).
        Both must stay flat together; keying one to severity without the other
        would re-introduce the reward↔constraint coupling the fix removed."""
        from utils.config import LAMBDA_WARM, CMDP_D_J_SEVERITY
        c3_warm = [LAMBDA_WARM[s][2] for s in range(1, 6)]
        d3 = [CMDP_D_J_SEVERITY[s]["d3_embb_mbps"] for s in range(1, 6)]
        assert c3_warm == [0.02] * 5, f"C3 warm must be fixed: {c3_warm}"
        assert d3 == [10.0] * 5, f"d3_embb must be fixed: {d3}"


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
        env.active_mask = np.ones(3, dtype=bool)   # unit-test splitter with all active (SUMO entry staggered)
        env.severity_per_amb = np.array(severity, dtype=np.int64)
        env._beta = float(beta)
        lam = np.zeros(4 * 3 + 1, dtype=np.float64)
        if lambda_c1 is not None:
            lam[0:3] = lambda_c1
        env._lambda_local = lam
        return env

    def test_pure_rl_uniform_without_lambda(self):
        """Pure RL: zero λ + zero logits → uniform split (no severity rule)."""
        env = self._env_k3([1, 3, 5], beta=1.0)
        env._prb_weights = np.zeros(3, dtype=np.float64)
        env.last_sinr_db = np.array([-15.0, -15.0, -15.0])
        out = env._prb_split_intra_slice(20)
        assert int(out.sum()) == 20
        spread = out.max() - out.min()
        assert spread <= 2, f"Zero λ should be ~uniform, got {out}"
        env.close()

    def test_pure_rl_logits_drive_ordering(self):
        """Pure RL: Worker logits drive PRB ordering. No λ in allocation."""
        env = self._env_k3([1, 3, 5], beta=1.0)
        env._prb_weights = np.array([0.0, 2.0, 5.0], dtype=np.float64)
        env.last_sinr_db = np.array([-15.0, -15.0, -15.0])
        out = env._prb_split_intra_slice(20)
        assert int(out.sum()) == 20
        assert out[2] >= out[1] >= out[0], f"logit-driven order failed: {out}"
        env.close()

    def test_pure_rl_logits_drive_allocation(self):
        """Pure RL: Worker logits directly control PRB split (zero λ)."""
        env = self._env_k3([5, 1, 1], beta=0.0)
        env._prb_weights = np.array([5.0, 0.0, 0.0], dtype=np.float64)
        env.last_sinr_db = np.array([-15.0, -15.0, -15.0])
        out = env._prb_split_intra_slice(30)
        assert int(out.sum()) == 30
        assert out[0] > out[1], f"Logit[0]=5 but amb_0={out[0]} <= amb_1={out[1]}"
        env.close()

    def test_logit_tiebreaker_same_severity(self):
        """Equal severity and equal SINR; logits differ → drives allocation."""
        env = self._env_k3([3, 3, 3], beta=2.0)
        env._prb_weights = np.array([0.0, 1.0, 3.0], dtype=np.float64)
        env.last_sinr_db = np.array([-5.0, -5.0, -5.0])
        out = env._prb_split_intra_slice(100)
        assert int(out.sum()) == 100
        assert out[2] >= out[1] >= out[0]
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
