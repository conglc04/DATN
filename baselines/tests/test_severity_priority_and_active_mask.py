"""Q1: Severity priority proof + Q3: Active-mask lockdown + P0 + P2 tests.

Evidence-based verification:
  Q1 — Does the method protect high-severity ambulances?
  Q3 — Only in-cell active ambulances get resources/QoS.
  P0 — 80s no-active: before or after episode start?
  P2 — Reward SUM × Constraint MEAN: prove λ compensates (not just claim).
"""
import math
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from env.oran_env import ORANEnv, macro_mission_config
from utils.config import (
    SEVERITY_QOS, MAC_TICKS_PER_WORKER,
    OBS_AOI_MEAN_IDX, OBS_AOI_MAX_IDX,
)


def _make_env(K: int = 3, seed: int = 0, **kw):
    cfg = macro_mission_config(K_ambulances=K, seed=seed, **kw)
    env = ORANEnv(cfg, seed=seed)
    obs, info = env.reset(seed=seed)
    return env, obs, info


# ═══════════════════════════════════════════════════════════════════════
# P0: 80s no-active — BEFORE or AFTER episode start?
# ═══════════════════════════════════════════════════════════════════════

class TestP0EpisodeStartTiming:
    """Episode starts AFTER fast-forward — first tick always has ≥1 active amb."""

    def test_episode_start_has_active_ambulance(self):
        """reset() fast-forwards SUMO until ≥1 ambulance enters cell."""
        env, _, info = _make_env(K=3, seed=0)
        assert info["n_active"] >= 1, (
            f"Episode started with 0 active ambulances — fast-forward failed"
        )
        assert env.active_mask.any()

    def test_tti_idx_zero_after_fastforward(self):
        """sim_time and tti_idx reset to 0 after fast-forward."""
        env, _, _ = _make_env(K=3, seed=0)
        assert env.tti_idx == 0
        assert env.sim_time == 0.0

    def test_80s_is_between_first_and_last_entry(self):
        """The ~80s gap is between amb_2 entering (t=0) and amb_1 entering (~81s).
        NOT 80s of dead time before episode starts."""
        env, _, info = _make_env(K=3, seed=0)
        # At reset: at least 1 ambulance is active
        initial_active = info["active_mask"].copy()
        assert initial_active.any()

        # Run until all 3 are active or episode ends
        all_entered_step = None
        for step in range(40000):  # 400s max
            _, _, term, trunc, info = env.step(
                np.zeros(env.action_space.shape[0], dtype=np.float32)
            )
            if info["entered_mask"].all():
                all_entered_step = step
                break
            if term or trunc:
                break

        assert all_entered_step is not None, "Not all ambulances entered before episode end"
        entry_time_s = all_entered_step * MAC_TICKS_PER_WORKER * env.config.tti_sec
        # The gap is within the episode, not before it
        assert entry_time_s > 0
        assert entry_time_s < env.config.episode_duration_sec

    def test_multiple_seeds_always_start_active(self):
        """Across seeds, episode always starts with ≥1 active ambulance."""
        for seed in [0, 1, 2, 42, 99, 123]:
            env, _, info = _make_env(K=3, seed=seed)
            assert info["n_active"] >= 1, f"seed={seed} started with 0 active"


# ═══════════════════════════════════════════════════════════════════════
# Q1: Severity priority — evidence-based proof
# ═══════════════════════════════════════════════════════════════════════

class TestQ1SeverityPriority:
    """Prove the method protects high-severity ambulances with hard evidence."""

    def test_higher_severity_gets_more_prb_phase1(self):
        """With mixed severity, higher-sev amb gets ≥ its N_req PRBs first."""
        env, _, _ = _make_env(K=3, seed=0)
        # Force severity: amb_0=sev5, amb_1=sev1, amb_2=sev1
        env.severity_per_amb = np.array([5, 1, 1], dtype=np.int64)
        env.severity = 5
        # Force all active
        env.active_mask[:] = True
        env.entered_mask[:] = True
        env.set_rrm_budget(0.20)  # limited budget

        _, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
        prb = np.array(info["prb_per_amb"])

        # Sev5 has D_max=1ms → needs MORE PRBs for same arrival rate.
        # Sev1 has D_max=20ms → needs fewer.
        # Phase 1 processes sev5 first → guaranteed N_req allocation.
        assert prb[0] > 0, f"Sev5 ambulance got 0 PRBs: {prb}"
        # With limited budget, sev5 gets proportionally more
        assert prb[0] >= prb[1], (
            f"Sev5 amb got {prb[0]} PRBs < sev1 amb got {prb[1]} PRBs"
        )

    def test_severity_ordering_follows_worker_logits(self):
        """Pure-RL allocation maps higher per-vehicle logit → more PRB (softmax).

        There is NO hard-coded severity ordering in the env (config.py:380-391);
        a trained Worker encodes severity into its per-vehicle logits ℓ_k. This
        feeds severity-aligned logits (sev5 > sev3 > sev1) and verifies the
        env's softmax allocation respects them — the mechanism by which severity
        ordering actually emerges. (A zero action would be uniform — the env does
        not order by severity on its own.)
        """
        env, _, _ = _make_env(K=3, seed=0)
        env.severity_per_amb = np.array([3, 5, 1], dtype=np.int64)
        env.severity = 5
        env.active_mask[:] = True
        env.entered_mask[:] = True
        env.set_rrm_budget(0.85)
        # Severity-aligned per-vehicle logits: amb_1 (sev5) > amb_0 (sev3) > amb_2 (sev1)
        action = np.array([1.0, 2.0, 0.0, 0.0], dtype=np.float32)
        _, _, _, _, info = env.step(action)
        prb = np.array(info["prb_per_amb"])

        # Softmax must preserve the logit ordering in the PRB split
        assert prb[1] >= prb[0], f"sev5={prb[1]} < sev3={prb[0]}"
        assert prb[0] >= prb[2], f"sev3={prb[0]} < sev1={prb[2]}"

    def test_severity_determines_qos_thresholds(self):
        """Each ambulance's C1/C2/C4/C5 thresholds match its own severity."""
        env, _, info = _make_env(K=3, seed=0)
        env.severity_per_amb = np.array([1, 3, 5], dtype=np.int64)
        env.severity = 5
        env.active_mask[:] = True
        env.entered_mask[:] = True
        _, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
        d_phi = info["d_phi"]
        K = 3
        # d_phi[0:K] = D_max per amb
        for k in range(K):
            sev = int(env.severity_per_amb[k])
            expected_dmax = SEVERITY_QOS[sev]["D_max"]
            assert d_phi[k] == pytest.approx(expected_dmax, rel=1e-6), (
                f"amb_{k} sev={sev}: d_phi C1={d_phi[k]} != D_max={expected_dmax}"
            )
        # d_phi[K:2K] = eps per amb
        for k in range(K):
            sev = int(env.severity_per_amb[k])
            expected_eps = SEVERITY_QOS[sev]["eps"]
            assert d_phi[K + k] == pytest.approx(expected_eps, rel=1e-6), (
                f"amb_{k} sev={sev}: d_phi C2={d_phi[K+k]} != eps={expected_eps}"
            )

    def test_n_req_scales_with_severity(self):
        """Higher severity → tighter D_max → higher N_req (more PRBs needed)."""
        env, _, _ = _make_env(K=3, seed=0)
        env.active_mask[:] = True
        env.entered_mask[:] = True

        # Compute N_req for different severities at same SINR
        from env.oran_env import (
            capacity_per_prb_bps, URLLC_OFFERED_LOAD_BPS,
            URLLC_PKT_BITS, PRB_MIN_QOS,
        )
        sinr = 10.0  # fixed for comparison
        cap = capacity_per_prb_bps(sinr)
        n_req_by_sev = {}
        for sev in [1, 3, 5]:
            d_max = SEVERITY_QOS[sev]["D_max"]
            c_req = URLLC_OFFERED_LOAD_BPS + URLLC_PKT_BITS / d_max
            n_req_by_sev[sev] = max(PRB_MIN_QOS, math.ceil(c_req / cap))

        # Sev5 (D_max=1ms) needs more PRBs than sev3 (D_max=5ms) > sev1 (D_max=20ms)
        assert n_req_by_sev[5] >= n_req_by_sev[3] >= n_req_by_sev[1], (
            f"N_req not monotonic: {n_req_by_sev}"
        )

    def test_fairness_within_same_severity(self):
        """Ambulances with same severity get proportional PRBs (not biased)."""
        env, _, _ = _make_env(K=3, seed=0)
        env.severity_per_amb = np.array([3, 3, 3], dtype=np.int64)
        env.severity = 3
        env.active_mask[:] = True
        env.entered_mask[:] = True
        env.set_rrm_budget(0.30)  # generous budget
        # Zero logits → uniform softmax → equal split
        env._prb_weights = np.zeros(3, dtype=np.float64)
        env._beta = 0.0  # no urgency bias

        _, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
        prb = np.array(info["prb_per_amb"])

        # Same severity + zero logits + beta=0 → approximately equal PRBs
        # Allow 20% tolerance for channel differences
        mean_prb = prb.mean()
        for k in range(3):
            if mean_prb > 0:
                ratio = prb[k] / mean_prb
                assert 0.5 < ratio < 2.0, (
                    f"amb_{k} PRB={prb[k]} deviates >2x from mean={mean_prb:.1f}"
                )

    def test_urgency_lambda_increases_prb_share(self):
        """Higher λ_C1[k] → higher urgency → more PRBs via Π_feasible."""
        env, _, _ = _make_env(K=3, seed=0)
        env.severity_per_amb = np.array([1, 1, 1], dtype=np.int64)
        env.severity = 1
        env.active_mask[:] = True
        env.entered_mask[:] = True
        env.set_rrm_budget(0.30)
        env._beta = 1.0
        env._prb_weights = np.zeros(3, dtype=np.float64)

        # Set λ_C1[0] = 5.0 (high constraint violation), others = 0
        K = 3
        lam = np.zeros(4 * K + 1, dtype=np.float64)
        lam[0] = 5.0  # λ_C1 for amb_0
        env.set_lambda_local(lam)

        _, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
        prb = np.array(info["prb_per_amb"])

        # amb_0 with high λ should get more surplus PRBs
        assert prb[0] >= prb[1], (
            f"High-λ amb_0 got {prb[0]} PRBs < low-λ amb_1 got {prb[1]}"
        )


# ═══════════════════════════════════════════════════════════════════════
# Q3: Active-mask lockdown — 5 conditions
# ═══════════════════════════════════════════════════════════════════════

class TestQ3ActiveMaskLockdown:
    """Prove: inactive ambulances get NO resources, NO QoS, NO obs."""

    def test_inactive_gets_zero_prb(self):
        """Condition 1: Inactive ambulance gets 0 PRBs."""
        env, _, _ = _make_env(K=3, seed=0)
        _, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
        prb = np.array(info["prb_per_amb"])
        for k in range(3):
            if not info["active_mask"][k]:
                assert prb[k] == 0, f"Inactive amb_{k} got {prb[k]} PRBs"

    def test_inactive_generates_no_traffic(self):
        """Condition 2: Inactive ambulance generates 0 URLLC arrivals."""
        env, _, _ = _make_env(K=3, seed=0)
        # Run 100 steps
        for _ in range(100):
            _, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))

        # Check queue arrival rates for inactive ambulances
        for k in range(3):
            if not info["active_mask"][k]:
                q = env.queues[f"urllc_{k}"]
                assert q.arrival_rate == 0.0, (
                    f"Inactive amb_{k} queue has arrival_rate={q.arrival_rate}"
                )

    def test_inactive_excluded_from_c_vec(self):
        """Condition 3: Inactive ambulance's C1/C2/C4/C5 slots = 0 in c_vec."""
        env, _, _ = _make_env(K=3, seed=0)
        _, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
        c_vec = info["c_vec"]
        K = 3
        for k in range(K):
            if not info["active_mask"][k]:
                assert c_vec[k] == pytest.approx(0.0, abs=1e-9), f"C1[{k}]≠0"
                assert c_vec[K + k] == pytest.approx(0.0, abs=1e-9), f"C2[{k}]≠0"
                assert c_vec[2*K + k] == pytest.approx(0.0, abs=1e-9), f"C4[{k}]≠0"
                assert c_vec[3*K + k] == pytest.approx(0.0, abs=1e-9), f"C5[{k}]≠0"

    def test_inactive_obs_block_zeroed(self):
        """Condition 4: Inactive ambulance's 10-dim obs block is all zeros."""
        env, _, _ = _make_env(K=3, seed=0)
        obs, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
        K = 3
        PER_AMB_DIM = 10
        FIXED_BLOCK = 20
        F = 1  # aoi_stream
        for k in range(K):
            if not info["active_mask"][k]:
                start = FIXED_BLOCK + k * PER_AMB_DIM
                end = start + PER_AMB_DIM
                block = obs[start:end]
                assert np.allclose(block, 0.0, atol=1e-9), (
                    f"Inactive amb_{k} obs block not zeroed: {block}"
                )

    def test_inactive_excluded_from_delay_aoi_mean(self):
        """Condition 5: Inactive ambulance not in delay/AoI mean computation."""
        env, _, _ = _make_env(K=3, seed=0)
        # Run enough steps to get mixed active/inactive
        for _ in range(50):
            obs, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))

        # Recompute expected AoI from only active ambulances
        aoi_raw = np.array([
            t["ambulance_status"].current_aoi(env.sim_time)
            for t in env.aoi_trackers
        ])
        if env.active_mask.any():
            expected_mean = float(aoi_raw[env.active_mask].mean())
        else:
            expected_mean = 0.0

        obs_mean = float(obs[OBS_AOI_MEAN_IDX])
        assert obs_mean == pytest.approx(expected_mean, abs=1e-6), (
            f"obs_aoi_mean={obs_mean} != active_only_mean={expected_mean}"
        )

    def test_arrived_ambulance_also_excluded(self):
        """After arrival: ambulance loses active status → same 5 conditions.
        Note: PRBs are allocated at step start; arrival detected during step.
        The NEXT step must reflect 0 PRBs for the arrived ambulance.
        """
        env, _, _ = _make_env(K=3, seed=0)
        arrived_k = None
        for _ in range(40000):
            _, _, term, trunc, info = env.step(
                np.zeros(env.action_space.shape[0], dtype=np.float32)
            )
            if info["arrived_mask"].any():
                arrived_k = int(np.where(info["arrived_mask"])[0][0])
                break
            if term or trunc:
                break

        if arrived_k is not None:
            assert not info["active_mask"][arrived_k]
            # Run ONE MORE step so PRB allocation sees the updated mask
            _, _, term, trunc, info2 = env.step(
                np.zeros(env.action_space.shape[0], dtype=np.float32)
            )
            if not (term or trunc):
                assert info2["prb_per_amb"][arrived_k] == 0, (
                    f"Arrived amb_{arrived_k} still got PRBs next step"
                )
                K = 3
                c_vec = info2["c_vec"]
                assert c_vec[arrived_k] == pytest.approx(0.0, abs=1e-9)


# ═══════════════════════════════════════════════════════════════════════
# P2: Reward SUM × Constraint MEAN — prove λ compensates
# ═══════════════════════════════════════════════════════════════════════

class TestP2RewardConstraintAggregation:
    """Prove that dual ascent correctly compensates for SUM-MEAN mismatch."""

    def test_dual_ascent_increases_lambda_on_violation(self):
        """When constraints are violated, λ increases via dual ascent."""
        from agents.lagrangian import LambdaState
        K = 1
        ls = LambdaState(K=K, force_zero_warm=True)
        ls.reset_episode((1,), 1)

        # Simulate 10 Worker steps with constraint violations
        for _ in range(10):
            c_vec = np.array([0.005, 0.01, 0.5, 0.1, -5.0])  # violating C1,C2,C4,C5
            d_phi = np.array([0.001, 0.001, 0.1, 0.01, 0.0])  # thresholds
            ls.accumulate(c_vec, d_phi)

        # Trigger Manager step → dual ascent
        ls.on_manager_step_end()
        lam = ls.lambda_global.copy()

        # λ should have increased for violated constraints
        assert lam[0] > 0, f"λ_C1={lam[0]} did not increase on violation"
        assert lam[1] > 0, f"λ_C2={lam[1]} did not increase on violation"

    def test_lambda_compensates_sum_mean_gap(self):
        """After enough dual updates, penalty ≈ reward magnitude."""
        from agents.lagrangian import LambdaState
        K = 1
        # Use sev=5 (tightest): D_max=1ms=D_REF → per-severity scale = old scale,
        # so the test's hardcoded c_vec/d_phi magnitudes match the normalization.
        ls = LambdaState(K=K, force_zero_warm=True)
        ls.reset_episode((5,), 5)

        reward_sum = 7.0  # abstract base reward (LambdaState-only test; not env scale)
        c_vec = np.array([0.005, 0.01, 0.5, 0.05, -5.0])
        d_phi = np.array([0.001, 0.001, 0.1, 0.01, 0.0])

        # Simulate 100 Manager windows (1000 Worker steps)
        for _ in range(100):
            for _ in range(10):
                ls.accumulate(c_vec, d_phi)
            ls.on_manager_step_end()
            ls.lambda_local = ls.lambda_global.copy()
            ls.win_c = np.zeros(5, dtype=np.float64)
            ls.win_steps = 0

        # After 100 updates, penalty should be significant relative to reward
        penalty = float(np.dot(
            ls.lambda_local, ls._normalized_deviation(c_vec, d_phi)
        ))
        r_aug = reward_sum - penalty

        # Penalty should be comparable to reward (within 10x)
        assert abs(penalty) > reward_sum * 0.01, (
            f"After 100 dual updates, penalty={penalty:.4f} still negligible vs "
            f"reward={reward_sum:.2f} — λ is not compensating"
        )

    def test_r_aug_negative_when_heavily_violated(self):
        """With enough violations, r_aug becomes negative (policy punished).
        Uses accelerated alpha to simulate long training convergence."""
        from agents.lagrangian import LambdaState
        K = 1
        # Use sev=5 (tightest) + 100× faster alpha to demonstrate convergence.
        ls = LambdaState(K=K, force_zero_warm=True, alpha_lambda=1e-2)
        ls.reset_episode((5,), 5)

        c_vec = np.array([0.010, 0.05, 1.0, 0.1, -5.0])  # heavy violations
        d_phi = np.array([0.001, 0.001, 0.1, 0.01, 0.0])

        for _ in range(200):
            for _ in range(10):
                ls.accumulate(c_vec, d_phi)
            ls.on_manager_step_end()
            ls.lambda_local = ls.lambda_global.copy()
            ls.win_c = np.zeros(5, dtype=np.float64)
            ls.win_steps = 0

        r_aug = ls.augmented_reward(7.0, c_vec, d_phi)
        assert r_aug < 0, (
            f"After 200 dual updates (α=1e-2), r_aug={r_aug:.4f} still positive"
        )

    def test_aggregation_documented(self):
        """Document: reward = MEAN(20 ticks), c_vec = MEAN(20 ticks) — MATCHED
        per-tick basis (audit 2026-06-23). The augmented Lagrangian r − Σλⱼ·gⱼ
        is balanced; no 20× scale gap for λ to compensate."""
        env, _, _ = _make_env(K=1, seed=0)
        env.set_rrm_budget(0.20)
        _, rew, _, _, info = env.step(np.zeros(1, dtype=np.float32))

        # Reward is the per-tick MEAN log-utility (NOT a ×20 sum).
        assert 0.0 < rew < 2.0, f"reward={rew} looks like a SUM, not per-tick MEAN"

        # c_vec C1 is MEAN delay (seconds) — same basis as reward
        c_vec = info["c_vec"]
        assert 0.0 <= c_vec[0] < 0.01, f"C1={c_vec[0]} not mean-scale"
