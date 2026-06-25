"""Solver-parity + SMDP + obs-scale + reward-sign + PRB-priority tests.

Covers every cross-solver invariant to guarantee PPO / TD3 / SAC are solving
the *same* CMDP problem with the same state convention, reward formula, and
Lagrangian update schedule.

Test classes:
  1.  TestSolverFairness             — PPO/TD3/SAC identical API, flags, lambda_state shape
  2.  TestAugmentedRewardSign        — sign, magnitude, exact formula, three-solver agreement
  3.  TestSMDPReturnAndTiming        — W=10 Manager boundary, dual ascent fires only at boundary
  4.  TestSeverityPriorityPRBSplit   — higher severity → more URLLC PRB; per-amb conservation
  5.  TestObsScaleNormalization      — every obs slot in its declared range after 50 steps
  6.  TestOverlayLambdaLocal         — slot-by-slot K=1 and K=3; pure function (no mutation)
  7.  TestMaskSeverity               — one-hot zeroed + per-amb sev_norm zeroed, K=1 and K=3
  8.  TestManagerStateDim            — manager_state_dim formula; build_manager_state content
"""

from __future__ import annotations

import numpy as np
import pytest

from agents.lagrangian import LambdaState
from agents.manager_agent import decode_manager_action, manager_state_dim
from env.oran_env import EnvConfig, ORANEnv
from solvers._common import (
    BaselineFlags,
    build_manager_state,
    mask_severity,
)
from solvers.sac import SACSolver
from solvers.td3 import TD3Solver
from utils.config import (
    AMB_ACTIVE_OFFSET,
    AMB_LAMBDA_C1_OFFSET,
    AMB_LAMBDA_C2_OFFSET,
    AMB_LAMBDA_C4_OFFSET,
    AMB_LAMBDA_C5_OFFSET,
    AMB_SEVERITY_NORM_OFFSET,
    B_RRM_MAX,
    B_RRM_MIN,
    GAMMA_MANAGER,
    GAMMA_WORKER,
    LAMBDA_C3_SHARED_OBS_INDEX,
    OBS_AOI_MAX_IDX,
    OBS_AOI_MEAN_IDX,
    OBS_BLER_IDX,
    OBS_FIXED_BLOCK_LEN,
    OBS_PER_AMB_BLOCK_LEN,
    OBS_RHO_EMBB_IDX,
    OBS_RHO_URLLC_IDX,
    OBS_SEVERITY_OH_IDX,
    OBS_SEVERITY_OH_LEN,
    P_TOTAL,
    WORKER_STEPS_PER_MANAGER,
    build_d_phi_vector,
    build_dual_scales,
    build_lambda_warm_vector,
)
from utils.obs import overlay_lambda_local


# ============================================================
# Helpers
# ============================================================

def _make_env(K: int = 1, sev: int = 3) -> ORANEnv:
    return ORANEnv(EnvConfig(K_ambulances=K, initial_severity=sev))


def _force_severity(env: ORANEnv, sev: int) -> None:
    """Pin env severity after reset (white-box unit test — project requires sample_severity=True)."""
    env.severity = sev
    env.severity_per_amb = np.full(env.config.K_ambulances, sev, dtype=int)


def _make_agents(K: int = 1, sev: int = 3):
    """Return (env, ppo_agent, td3, sac) all wired for the same environment."""
    env = _make_env(K=K, sev=sev)
    sd = env.observation_space.shape[0]
    ad = env.action_space.shape[0]
    from agents.worker_agent import WorkerAgent
    ppo = WorkerAgent(state_dim=sd, action_dim=ad)
    td3 = TD3Solver(state_dim=sd, action_dim=ad, K=K, seed=0)
    sac = SACSolver(state_dim=sd, action_dim=ad, K=K, seed=0)
    return env, ppo, td3, sac


# ============================================================
# 1. Solver Fairness
# ============================================================

class TestSolverFairness:
    """All three solvers must share the exact same state/action convention."""

    def test_flags_identical_across_solvers(self):
        td3 = TD3Solver(state_dim=31, action_dim=1, K=1)
        sac = SACSolver(state_dim=31, action_dim=1, K=1)
        for name, agent in [("td3", td3), ("sac", sac)]:
            assert agent.FLAGS.use_phase is True, f"{name}: use_phase must be True (severity visible)"
            assert agent.FLAGS.use_cmdp is True, f"{name}: use_cmdp must be True (Lagrangian active)"
            assert agent.FLAGS.use_hrl is True, f"{name}: use_hrl must be True (Manager active)"
            assert agent.FLAGS.n_constraints == 5, f"{name}: n_constraints must be 5"

    def test_ppo_worker_agent_has_no_severity_masking(self):
        """WorkerAgent (PPO) always sees the full observation including severity one-hot.
        It has no maybe_mask method (masking is a TD3/SAC solver-wrapper concern).
        """
        from agents.worker_agent import WorkerAgent
        ppo = WorkerAgent(state_dim=31, action_dim=1)
        assert not hasattr(ppo, "maybe_mask"), (
            "WorkerAgent must NOT have maybe_mask — PPO always sees full obs"
        )
        # Confirm TD3/SAC maybe_mask is identity (they ARE severity-aware)
        obs = np.ones(31, dtype=np.float32)
        td3 = TD3Solver(31, 1, K=1)
        sac = SACSolver(31, 1, K=1)
        assert np.allclose(td3.maybe_mask(obs), obs)
        assert np.allclose(sac.maybe_mask(obs), obs)

    def test_td3_sac_maybe_mask_is_identity_when_severity_aware(self):
        obs = np.ones(31, dtype=np.float32)
        for name, agent in [("td3", TD3Solver(31, 1, K=1)), ("sac", SACSolver(31, 1, K=1))]:
            assert np.allclose(agent.maybe_mask(obs), obs), f"{name} maybe_mask must be identity"

    def test_lambda_state_k1_dim(self):
        td3 = TD3Solver(31, 1, K=1)
        sac = SACSolver(31, 1, K=1)
        for name, agent in [("td3", td3), ("sac", sac)]:
            assert agent.lambda_state.n_constraints == 5
            assert agent.lambda_state.K == 1

    def test_lambda_state_k3_dim(self):
        td3 = TD3Solver(51, 4, K=3)
        sac = SACSolver(51, 4, K=3)
        for name, agent in [("td3", td3), ("sac", sac)]:
            assert agent.lambda_state.n_constraints == 13
            assert agent.lambda_state.K == 3

    def test_select_action_returns_three_tuple(self):
        """All solvers return (action, log_prob, value) — same tuple shape as PPO."""
        obs = np.zeros(31, dtype=np.float32)
        td3 = TD3Solver(31, 1, K=1)
        sac = SACSolver(31, 1, K=1)
        for name, agent in [("td3", td3), ("sac", sac)]:
            result = agent.select_action(obs)
            assert isinstance(result, tuple) and len(result) == 3, (
                f"{name}.select_action must return 3-tuple (action, log_prob, value)"
            )

    def test_action_shape_k1(self):
        obs = np.zeros(31, dtype=np.float32)
        for name, cls in [("td3", TD3Solver), ("sac", SACSolver)]:
            agent = cls(31, 1, K=1)
            action, _, _ = agent.select_action(obs)
            assert action.shape == (1,), f"{name} K=1 action must be 1-dim"

    def test_action_shape_k3(self):
        obs = np.zeros(51, dtype=np.float32)
        for name, cls in [("td3", TD3Solver), ("sac", SACSolver)]:
            agent = cls(51, 4, K=3)
            action, _, _ = agent.select_action(obs)
            assert action.shape == (4,), f"{name} K=3 action must be 4-dim"

    def test_augment_reward_api_consistent(self):
        """All solvers expose augment_reward(r, c_vec, d_phi) with identical formula."""
        sev = 2
        lam_vec = build_lambda_warm_vector([sev], sev)
        c_vec = build_d_phi_vector([sev])
        c_vec[0] += 0.001   # C1 violation
        d_phi = build_d_phi_vector([sev])
        r = 1.5
        td3 = TD3Solver(31, 1, K=1)
        sac = SACSolver(31, 1, K=1)
        td3.on_episode_start([sev], sev)
        sac.on_episode_start([sev], sev)
        aug_td3 = td3.augment_reward(r, c_vec, d_phi)
        aug_sac = sac.augment_reward(r, c_vec, d_phi)
        assert aug_td3 == pytest.approx(aug_sac, abs=1e-9), (
            f"TD3 and SAC must give identical aug reward; td3={aug_td3}, sac={aug_sac}"
        )

    def test_on_episode_start_syncs_lambda_state(self):
        """After on_episode_start, all solvers warm-start from build_lambda_warm_vector."""
        sev = 4
        expected = build_lambda_warm_vector([sev], sev)
        for name, cls in [("td3", TD3Solver), ("sac", SACSolver)]:
            agent = cls(31, 6, K=1)
            agent.on_episode_start([sev], sev)
            lam = agent.lambda_state.get_lambda_global()
            np.testing.assert_allclose(lam, expected, atol=1e-9, err_msg=f"{name} warm-start mismatch")

    def test_accumulate_constraint_increments_win_c(self):
        sev = 1
        c_vec = build_d_phi_vector([sev])
        d_phi = build_d_phi_vector([sev])
        c_vec[0] += 0.005   # C1 slightly violated
        for name, cls in [("td3", TD3Solver), ("sac", SACSolver)]:
            agent = cls(31, 6, K=1)
            agent.on_episode_start([sev], sev)
            agent.accumulate_constraint(c_vec, d_phi)
            assert agent.lambda_state.win_steps == 1, f"{name} win_steps must be 1 after one accumulate"

    def test_adapter_augment_reward_delegates_exactly(self):
        """agent.augment_reward MUST equal agent.lambda_state.augmented_reward.
        This verifies the adapter layer, not the inner raw agent (td3.td3 / sac.sac).
        The raw inner agent knows nothing about constraints.
        """
        sev = 3
        c = build_d_phi_vector([sev])
        c[2] += 0.05  # C4 slightly violated
        d = build_d_phi_vector([sev])
        r = 0.0
        for name, cls in [("td3", TD3Solver), ("sac", SACSolver)]:
            agent = cls(31, 6, K=1)
            agent.on_episode_start([sev], sev)
            aug_adapter = agent.augment_reward(r, c, d)
            aug_direct = agent.lambda_state.augmented_reward(r, c, d)
            assert aug_adapter == pytest.approx(aug_direct, abs=1e-12), (
                f"{name}: augment_reward ({aug_adapter}) must delegate exactly to "
                f"lambda_state.augmented_reward ({aug_direct})"
            )

    def test_adapter_accumulate_delegates_to_lambda_state(self):
        """agent.accumulate_constraint must update lambda_state.win_c, not a separate buffer."""
        sev = 2
        c = build_d_phi_vector([sev])
        c[0] += 0.003
        d = build_d_phi_vector([sev])
        for name, cls in [("td3", TD3Solver), ("sac", SACSolver)]:
            agent = cls(31, 6, K=1)
            agent.on_episode_start([sev], sev)
            win_c_before = agent.lambda_state.win_c.copy()
            agent.accumulate_constraint(c, d)
            win_c_after = agent.lambda_state.win_c
            assert not np.allclose(win_c_after, win_c_before), (
                f"{name}: accumulate_constraint must update lambda_state.win_c"
            )

    def test_adapter_on_manager_step_end_updates_lambda_global(self):
        """agent.on_manager_step_end must fire lambda_state.on_manager_step_end:
        lambda_global changes after a violated window.
        """
        sev = 1
        c = build_d_phi_vector([sev])
        c[0] += 0.010
        d = build_d_phi_vector([sev])
        for name, cls in [("td3", TD3Solver), ("sac", SACSolver)]:
            agent = cls(31, 6, K=1)
            agent.on_episode_start([sev], sev)
            lam_before = agent.lambda_state.get_lambda_global().copy()
            for _ in range(WORKER_STEPS_PER_MANAGER):
                agent.accumulate_constraint(c, d)
            agent.on_manager_step_end()
            lam_after = agent.lambda_state.get_lambda_global()
            assert not np.allclose(lam_after, lam_before), (
                f"{name}: on_manager_step_end must update lambda_global"
            )

    def test_on_manager_step_end_returns_dict(self):
        sev = 3
        c_vec = build_d_phi_vector([sev])
        d_phi = build_d_phi_vector([sev])
        for name, cls in [("td3", TD3Solver), ("sac", SACSolver)]:
            agent = cls(31, 6, K=1)
            agent.on_episode_start([sev], sev)
            for _ in range(WORKER_STEPS_PER_MANAGER):
                agent.accumulate_constraint(c_vec, d_phi)
            out = agent.on_manager_step_end()
            assert isinstance(out, dict), f"{name} on_manager_step_end must return dict"
            assert "subgradient_mean" in out
            assert "lambda_global_mean" in out


# ============================================================
# 2. Augmented Reward Sign
# ============================================================

class TestAugmentedRewardSign:
    """r_aug = r − Σ_j λ_j · max(0, (c_j − d_j) / scale_j)  (hinge).

    sign/magnitude verified exactly against the closed-form formula. Hinge
    fixed 2026-06-22 (bonus-masking audit): was raw signed deviation, which
    let slack constraints (c<d) BONUS the reward instead of contributing 0.
    """

    SEV = 1

    def _ls_with_warm(self) -> LambdaState:
        ls = LambdaState(K=1)
        ls.reset_episode([self.SEV], self.SEV)
        return ls

    def test_violation_reduces_reward(self):
        """c > d → penalty > 0 → r_aug < r."""
        ls = self._ls_with_warm()
        c = build_d_phi_vector([self.SEV])
        d = build_d_phi_vector([self.SEV])
        c[0] += 0.010   # C1 violated
        r = 2.0
        aug = ls.augmented_reward(r, c, d)
        assert aug < r, f"violation must reduce reward; r={r}, aug={aug}"

    def test_satisfaction_raises_reward(self):
        """c < d → hinge clips penalty to 0 → r_aug == r exactly (neutral, no bonus).

        Pre-fix (raw signed deviation) this gave aug > r — a reward BONUS for
        merely satisfying a constraint, which let slack mean-constraints
        (C1/C4) mask violated tail-constraint (C2/C5) penalties by up to 131x
        at severity 5 (bonus-masking audit, 2026-06-22).
        """
        ls = self._ls_with_warm()
        c = build_d_phi_vector([self.SEV])
        d = build_d_phi_vector([self.SEV])
        c[0] -= 0.005   # C1 satisfied below threshold
        r = 2.0
        aug = ls.augmented_reward(r, c, d)
        assert aug == pytest.approx(r, abs=1e-12), (
            f"satisfaction must be neutral (hinge), not a bonus; r={r}, aug={aug}"
        )

    def test_zero_lambda_aug_equals_raw_reward(self):
        """λ=0 everywhere → r_aug = r exactly."""
        ls = LambdaState(K=1)
        ls.lambda_local = np.zeros(5, dtype=np.float64)
        c = build_d_phi_vector([1])
        d = build_d_phi_vector([1])
        c[0] += 0.100   # large violation — irrelevant when λ=0
        r = 3.14159
        assert ls.augmented_reward(r, c, d) == pytest.approx(r, abs=1e-12)

    def test_exact_formula_single_slot(self):
        """Verify exact value: λ=[0,0,0,0,0.1], c[4]=5.0, d[4]=0.0, scale[4]=100.0.
        penalty = 0.1 * (5.0 - 0.0) / 100.0 = 0.005
        r_aug = 2.0 - 0.005 = 1.995
        """
        ls = LambdaState(K=1)
        ls.lambda_local = np.array([0.0, 0.0, 0.0, 0.0, 0.1], dtype=np.float64)
        c = np.array([0.0, 0.0, 0.0, 0.0, 5.0], dtype=np.float64)
        d = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        aug = ls.augmented_reward(2.0, c, d)
        assert aug == pytest.approx(1.995, abs=1e-9)

    def test_exact_formula_all_slots(self):
        """Full 5-dim K=1 formula cross-check at sev=3.

        scale = [1e-3, 1.0, 0.1, 1.0, 100.0]
        λ = warm_start(sev=3) = build_lambda_warm_vector([3], 3)
        c = d_phi (zero deviation) + deltas
        """
        sev = 3
        lam = build_lambda_warm_vector([sev], sev)   # shape (5,)
        # Per-severity scale (ĐX2 audit 2026-06-24): reset_episode rebuilds
        # dual_scales with severity_per_amb, so the test must use the same.
        scale = build_dual_scales(1, severity_per_amb=[sev])  # shape (5,)
        d = build_d_phi_vector([sev])
        deltas = np.array([0.002, 0.1, 0.5, 0.02, 0.003], dtype=np.float64)
        c = d + deltas
        expected_penalty = float(np.dot(lam, deltas / scale))
        r = 5.0
        expected_aug = r - expected_penalty

        ls = LambdaState(K=1)
        ls.reset_episode([sev], sev)
        aug = ls.augmented_reward(r, c, d)
        assert aug == pytest.approx(expected_aug, abs=1e-9)

    def test_three_solvers_identical_aug_reward(self):
        """TD3 and SAC produce the same aug reward as a reference LambdaState."""
        sev = 2
        c = build_d_phi_vector([sev])
        c[1] += 0.2   # C2 violated
        d = build_d_phi_vector([sev])
        r = -0.5

        # Reference
        ls_ref = LambdaState(K=1)
        ls_ref.reset_episode([sev], sev)
        aug_ref = ls_ref.augmented_reward(r, c, d)

        td3 = TD3Solver(31, 1, K=1)
        sac = SACSolver(31, 1, K=1)
        td3.on_episode_start([sev], sev)
        sac.on_episode_start([sev], sev)
        assert td3.augment_reward(r, c, d) == pytest.approx(aug_ref, abs=1e-9)
        assert sac.augment_reward(r, c, d) == pytest.approx(aug_ref, abs=1e-9)

    def test_hinge_clips_negative_deviation_to_zero(self):
        """Critical convention (fixed 2026-06-22): max(0, c-d)/scale (hinge), NOT raw signed.

        If hinge:   c < d → dev < 0 → max(0,dev) = 0 → penalty = 0 → r_aug = r (no bonus).
        If signed:  c < d → dev < 0 → penalty < 0 → r_aug > r (reward bonus — the bug).

        λ_local[0] = 1.0, c[0] = d[0] - 0.001 s, scale[0] = D_REF_URLLC = 1e-3.
        signed dev = (c[0] - d[0]) / 1e-3 = -0.001 / 1e-3 = -1.0
        r_aug = 0.0 - 1.0 * max(0, -1.0) = 0.0   ← hinge (current, correct)
        r_aug = 0.0 - 1.0 * (-1.0)       = +1.0  ← signed (OLD, bonus-masking bug)
        """
        ls = LambdaState(K=1)
        ls.lambda_local = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        d = build_d_phi_vector([1])
        c = d.copy()
        c[0] = d[0] - 0.001   # c < d by exactly 1 ms
        aug = ls.augmented_reward(0.0, c, d)
        assert aug == pytest.approx(0.0, abs=1e-9), (
            f"Hinge MUST clip slack deviation to 0 (no bonus); got {aug}. "
            "If +1.0, formula regressed to raw signed deviation (bonus-masking bug)."
        )

    def test_adapter_augment_reward_same_convention_as_lagrangian(self):
        """TD3Solver.augment_reward delegates to lambda_state.augmented_reward:
        the adapter must NOT reclip or transform the deviation — same hinge result.
        """
        sev = 1
        td3 = TD3Solver(31, 1, K=1)
        td3.on_episode_start([sev], sev)
        # Force a known lambda_local for exact check
        td3.lambda_state.lambda_local = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        d = build_d_phi_vector([sev])
        c = d.copy()
        c[0] = d[0] - 0.001
        aug = td3.augment_reward(0.0, c, d)
        assert aug == pytest.approx(0.0, abs=1e-9), (
            "TD3 adapter must use the same hinge convention as LambdaState"
        )

    def test_large_lambda_large_violation_decreases_reward_proportionally(self):
        """If λ[0] = 10.0 and C1 excess = 0.01 s, penalty = 10.0*0.01/1e-3 = 100."""
        ls = LambdaState(K=1)
        ls.lambda_local = np.array([10.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        d = build_d_phi_vector([3])  # d[0] = D_max^sev3 = 5e-3
        c = d.copy()
        c[0] = d[0] + 0.01  # excess = 0.01 s
        aug = ls.augmented_reward(0.0, c, d)
        expected = 0.0 - 10.0 * (0.01 / 1e-3)
        assert aug == pytest.approx(expected, abs=1e-8)


# ============================================================
# 3. SMDP Return and Timing
# ============================================================

class TestSMDPReturnAndTiming:
    """Manager window = W=10 Worker steps.

    The dual ascent must fire EXACTLY at step boundaries 10, 20, ... and
    NOT in between. The LambdaState win_steps counter must reset to 0 after
    on_manager_step_end().
    """

    SEV = 2

    def _ls(self) -> LambdaState:
        ls = LambdaState(K=1)
        ls.reset_episode([self.SEV], self.SEV)
        return ls

    def test_worker_steps_per_manager_equals_10(self):
        assert WORKER_STEPS_PER_MANAGER == 10

    def test_gamma_manager_equals_gamma_worker_power_10(self):
        assert GAMMA_MANAGER == pytest.approx(GAMMA_WORKER ** WORKER_STEPS_PER_MANAGER, abs=1e-9)

    def test_lambda_unchanged_before_manager_boundary(self):
        """After accumulating steps 1..9, lambda must NOT change (no dual ascent yet)."""
        ls = self._ls()
        lam_before = ls.get_lambda_global().copy()
        c = build_d_phi_vector([self.SEV])
        d = build_d_phi_vector([self.SEV])
        c[0] += 0.005   # constant C1 violation
        for _ in range(WORKER_STEPS_PER_MANAGER - 1):
            ls.accumulate(c, d)
        # Not yet fired on_manager_step_end → lambda unchanged
        np.testing.assert_array_equal(ls.get_lambda_global(), lam_before,
            err_msg="lambda must not change before on_manager_step_end()")

    def test_lambda_changes_exactly_at_manager_boundary(self):
        """After exactly W accumulate calls + on_manager_step_end, lambda MUST change."""
        ls = self._ls()
        lam_before = ls.get_lambda_global().copy()
        c = build_d_phi_vector([self.SEV])
        d = build_d_phi_vector([self.SEV])
        c[0] += 0.005
        for _ in range(WORKER_STEPS_PER_MANAGER):
            ls.accumulate(c, d)
        ls.on_manager_step_end()
        lam_after = ls.get_lambda_global()
        assert not np.allclose(lam_after, lam_before), (
            "lambda must change after W violations + on_manager_step_end()"
        )

    def test_win_c_resets_after_manager_step(self):
        """win_c and win_steps must reset to 0 after on_manager_step_end."""
        ls = self._ls()
        c = build_d_phi_vector([self.SEV])
        d = build_d_phi_vector([self.SEV])
        c[0] += 0.005
        for _ in range(WORKER_STEPS_PER_MANAGER):
            ls.accumulate(c, d)
        ls.on_manager_step_end()
        assert ls.win_steps == 0, "win_steps must reset after Manager step"
        np.testing.assert_array_equal(ls.win_c, np.zeros(5),
            err_msg="win_c must reset to zero after Manager step")

    def test_lambda_local_equals_lambda_global_after_step(self):
        """on_manager_step_end pushes lambda_global → lambda_local (Worker sync)."""
        ls = self._ls()
        c = build_d_phi_vector([self.SEV])
        d = build_d_phi_vector([self.SEV])
        c[0] += 0.003
        for _ in range(WORKER_STEPS_PER_MANAGER):
            ls.accumulate(c, d)
        ls.on_manager_step_end()
        np.testing.assert_array_equal(ls.lambda_global, ls.lambda_local,
            err_msg="lambda_local must equal lambda_global after Manager step")

    def test_two_manager_windows_accumulate_correctly(self):
        """Over 2 Manager windows, dual ascent fires twice; lambda grows monotonically
        under constant C1 violation."""
        ls = self._ls()
        c = build_d_phi_vector([self.SEV])
        d = build_d_phi_vector([self.SEV])
        c[0] += 0.010   # C1 violated by 10 ms

        lam_0 = ls.get_lambda_global()[0]
        for _ in range(WORKER_STEPS_PER_MANAGER):
            ls.accumulate(c, d)
        ls.on_manager_step_end()
        lam_1 = ls.get_lambda_global()[0]

        for _ in range(WORKER_STEPS_PER_MANAGER):
            ls.accumulate(c, d)
        ls.on_manager_step_end()
        lam_2 = ls.get_lambda_global()[0]

        assert lam_1 > lam_0, "lambda must grow after first Manager step"
        assert lam_2 > lam_1, "lambda must grow after second Manager step"

    def test_no_dual_ascent_if_no_accumulate(self):
        """on_manager_step_end with zero win_steps must be a no-op (skip update)."""
        ls = self._ls()
        lam_before = ls.get_lambda_global().copy()
        ls.on_manager_step_end()   # win_steps == 0
        np.testing.assert_array_equal(ls.get_lambda_global(), lam_before,
            err_msg="on_manager_step_end with no accumulations must not change lambda")

    def test_smdp_dual_frequency_matches_smoke_train_w(self):
        """smoke_train.py fires dual ascent at every WORKER_STEPS_PER_MANAGER boundary.
        Verify via TD3 agent that lambda changes only at multiples of W.
        """
        sev = 1
        env, _, td3, _ = _make_agents(K=1, sev=sev)
        obs, info = env.reset(seed=0)
        td3.on_episode_start([sev], sev)
        td3.on_manager_step_start([sev], sev)

        c = info["c_vec"]
        d = info["d_phi"]
        c[0] += 0.005  # constant C1 violation

        lambda_snapshots = []
        for step in range(WORKER_STEPS_PER_MANAGER * 3):
            td3.accumulate_constraint(c, d)
            if (step + 1) % WORKER_STEPS_PER_MANAGER == 0:
                td3.on_manager_step_end()
            lambda_snapshots.append(td3.lambda_state.get_lambda_global()[0])

        # Lambda snapshots: snapshot[step] = lambda AFTER step (and after ascent if boundary).
        # Boundary fires at step ∈ {9,19,29} (0-indexed) when (step+1)%W==0.
        # changes[i] = snapshot[i+1] vs snapshot[i] — a change at index i means the
        # Manager boundary fired DURING step (i+1), i.e. (i+2) % W == 0.
        changes = [abs(lambda_snapshots[i + 1] - lambda_snapshots[i]) > 1e-10
                   for i in range(len(lambda_snapshots) - 1)]
        for i, changed in enumerate(changes):
            boundary_at_this_change = (i + 2) % WORKER_STEPS_PER_MANAGER == 0
            if boundary_at_this_change:
                assert changed, f"lambda must change at changes[{i}] (Manager boundary after step {i+1})"
            else:
                assert not changed, f"lambda must not change at changes[{i}] (mid-window step {i+1})"
        env.close()


# ============================================================
# 4. Severity Priority PRB Split
# ============================================================

class TestSeverityPriorityPRBSplit:
    """Higher severity → more dedicated URLLC PRB (tighter D_max → larger r_min)."""

    def test_higher_severity_tighter_d_max_drives_higher_c1_violation(self):
        """Higher severity → smaller D_max → same PRB budget leads to higher C1 (c_j-d_j).

        PRB is NOT directly keyed by severity (it's controlled by Manager-owned
        r_min_urllc). What changes is D_max: at sev=5 D_max=1ms,
        at sev=1 D_max=20ms. For the same D_e2e, c1_sev5 = D_e2e - 1ms >> c1_sev1 = D_e2e - 20ms.
        """
        from utils.config import SEVERITY_QOS
        d_max_sev1 = SEVERITY_QOS[1]["D_max"]   # 20 ms = 0.020 s
        d_max_sev5 = SEVERITY_QOS[5]["D_max"]   # 1 ms  = 0.001 s
        assert d_max_sev5 < d_max_sev1, "D_max must be tighter at higher severity"

        # Measure actual c_vec[0] (C1 = mean D_e2e - D_max is implicit in the d_phi offset)
        # The env tracks raw D_e2e in c_vec; d_phi[0] = D_max^sev. So c_vec[0]-d_phi[0]
        # is the constraint violation. Compare across severity at the same action.
        env1 = ORANEnv(EnvConfig(K_ambulances=1))
        env1.reset(seed=0)
        _force_severity(env1, 1)   # pin severity for formula unit test
        _, _, _, _, info1 = env1.step(np.zeros(env1.action_space.shape, dtype=np.float32))
        env1.close()

        env5 = ORANEnv(EnvConfig(K_ambulances=1))
        env5.reset(seed=0)
        _force_severity(env5, 5)   # pin severity for formula unit test
        _, _, _, _, info5 = env5.step(np.zeros(env5.action_space.shape, dtype=np.float32))
        env5.close()

        # c_vec[0] = mean D_e2e_k; d_phi[0] = D_max^sev
        # (c-d)[0]_sev5 > (c-d)[0]_sev1 because d_phi[0] is much smaller at sev=5
        gap5 = info5["c_vec"][0] - info5["d_phi"][0]
        gap1 = info1["c_vec"][0] - info1["d_phi"][0]
        assert gap5 > gap1, (
            f"sev=5 C1 gap={gap5:.5f} must exceed sev=1 C1 gap={gap1:.5f} "
            "(same D_e2e, tighter D_max at sev=5)"
        )

    def test_prb_urllc_equals_int_r_min_times_p_total(self):
        """prb_urllc == int(r_min_urllc * P_TOTAL) — the B_U formula from _prb_allocation.
        This pins the formula, not just the invariant: a bug that returns a different
        allocation would still satisfy sum==sum but break the formula check.
        """
        env = ORANEnv(EnvConfig(K_ambulances=1))
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        B_U = int(env.r_min_urllc * P_TOTAL)
        assert info["prb_urllc"] == B_U, (
            f"prb_urllc={info['prb_urllc']} != int(r_min_urllc={env.r_min_urllc:.4f} * P_TOTAL={P_TOTAL}) = {B_U}"
        )
        assert sum(info["prb_per_amb"]) == B_U, (
            f"sum(prb_per_amb)={sum(info['prb_per_amb'])} != B_U={B_U}"
        )
        env.close()

    def test_prb_per_amb_sums_to_prb_urllc_k3_mixed_severity(self):
        env = ORANEnv(EnvConfig(K_ambulances=3))
        env.reset(seed=0, options={"severity_per_amb": [1, 3, 5]})
        _, _, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        B_U_k3 = int(env.r_min_urllc * P_TOTAL)
        assert info["prb_urllc"] == B_U_k3
        assert sum(info["prb_per_amb"]) == B_U_k3
        env.close()

    def test_each_amb_ge_prb_min_qos_mixed_severity(self):
        from utils.config import PRB_MIN_QOS
        env = ORANEnv(EnvConfig(K_ambulances=3))
        env.reset(seed=0, options={"severity_per_amb": [1, 2, 5]})
        for _ in range(20):
            _, _, _, _, info = env.step(env.action_space.sample())
            ac = info["active_count_per_amb"]
            for k, prb in enumerate(info["prb_per_amb"]):
                if ac[k] > 0:   # only ACTIVE ambulances are guaranteed PRB_MIN_QOS (SUMO staggers entry)
                    assert prb >= PRB_MIN_QOS, f"active amb[{k}] prb={prb} below PRB_MIN_QOS"
        env.close()

    def test_pure_rl_logits_drive_severity_allocation(self):
        """Pure RL: Worker logits directly control PRB split.
        Policy learns to output high logit for high-severity ambulance.
        """
        env = ORANEnv(EnvConfig(K_ambulances=3))
        env.reset(seed=42, options={"severity_per_amb": [1, 3, 5]})
        env.active_mask = np.ones(3, dtype=bool)
        # Simulate trained policy: high logit for sev5 (amb2)
        env._prb_weights = np.array([0.0, 1.0, 5.0], dtype=np.float64)
        prb = env._prb_split_intra_slice(27)
        assert prb[2] >= prb[1] >= prb[0], f"logit-driven order violated: {prb.tolist()}"
        assert prb[2] > prb[0], f"sev5 logit=5 must get more than sev1 logit=0"
        env.close()

    def test_r_min_anchor_determines_prb_urllc_floor(self):
        """set_rrm_budget(0.8) must yield more URLLC PRBs than set_rrm_budget(0.1)."""
        env_hi = ORANEnv(EnvConfig(K_ambulances=1))
        env_hi.reset(seed=0)
        env_hi.set_rrm_budget(0.8)
        _, _, _, _, info_hi = env_hi.step(np.zeros(env_hi.action_space.shape, dtype=np.float32))

        env_lo = ORANEnv(EnvConfig(K_ambulances=1))
        env_lo.reset(seed=0)
        env_lo.set_rrm_budget(0.1)
        _, _, _, _, info_lo = env_lo.step(np.zeros(env_lo.action_space.shape, dtype=np.float32))

        assert info_hi["prb_urllc"] > info_lo["prb_urllc"]
        env_hi.close()
        env_lo.close()

    def test_decode_manager_action_range_exhaustive(self):
        """b_rrm in [B_RRM_MIN, B_RRM_MAX] for all raw actions in [-100, 100]."""
        for raw in np.linspace(-100.0, 100.0, 50):
            b = decode_manager_action(np.array([raw]))["b_rrm"]
            assert B_RRM_MIN <= b <= B_RRM_MAX + 1e-9, (
                f"raw={raw:.2f} → b_rrm={b:.4f} out of [{B_RRM_MIN}, {B_RRM_MAX}]"
            )

    def test_decode_manager_action_midpoint(self):
        """raw=0 → b_rrm ≈ midpoint of [B_RRM_MIN, B_RRM_MAX] (sigmoid(0)=0.5)."""
        b = decode_manager_action(np.array([0.0]))["b_rrm"]
        mid = (B_RRM_MIN + B_RRM_MAX) / 2.0
        assert b == pytest.approx(mid, abs=1e-5)

    def test_total_prb_le_273_across_severities(self):
        from utils.config import P_TOTAL
        for sev in range(1, 6):
            env = ORANEnv(EnvConfig(K_ambulances=1, initial_severity=sev))
            env.reset(seed=0)
            for _ in range(10):
                _, _, _, _, info = env.step(env.action_space.sample())
                assert info["prb_urllc"] + info["prb_embb"] <= P_TOTAL
            env.close()


# ============================================================
# 5. Obs Scale Normalization
# ============================================================

class TestObsScaleNormalization:
    """Every obs slot must stay within its declared range after many steps."""

    def _collect_obs(self, K: int = 1, n_steps: int = 50) -> np.ndarray:
        env = ORANEnv(EnvConfig(K_ambulances=K))
        env.reset(seed=0)
        all_obs = []
        for _ in range(n_steps):
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            all_obs.append(obs)
            if terminated or truncated:
                env.reset(seed=0)
        env.close()
        return np.stack(all_obs)  # (n_steps, obs_dim)

    def test_rho_urllc_in_unit_interval(self):
        obs_arr = self._collect_obs()
        vals = obs_arr[:, OBS_RHO_URLLC_IDX]
        assert np.all(vals >= 0.0) and np.all(vals <= 1.0 + 1e-6), (
            f"rho_urllc out of [0,1]: min={vals.min():.4f}, max={vals.max():.4f}"
        )

    def test_rho_embb_in_unit_interval(self):
        obs_arr = self._collect_obs()
        vals = obs_arr[:, OBS_RHO_EMBB_IDX]
        assert np.all(vals >= 0.0) and np.all(vals <= 1.0 + 1e-6)

    def test_bler_in_valid_range(self):
        """BLER clipped to [1e-4, 0.5] in _sample_bler → obs slot is in [0, 0.5]."""
        obs_arr = self._collect_obs()
        vals = obs_arr[:, OBS_BLER_IDX]
        assert np.all(vals >= 0.0) and np.all(vals <= 0.5 + 1e-6), (
            f"BLER out of [0, 0.5]: min={vals.min():.4f}, max={vals.max():.4f}"
        )

    def test_severity_one_hot_sums_to_one(self):
        obs_arr = self._collect_obs()
        oh = obs_arr[:, OBS_SEVERITY_OH_IDX: OBS_SEVERITY_OH_IDX + OBS_SEVERITY_OH_LEN]
        sums = oh.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-6, err_msg="Severity OH must sum to 1")

    def test_severity_one_hot_is_binary(self):
        obs_arr = self._collect_obs()
        oh = obs_arr[:, OBS_SEVERITY_OH_IDX: OBS_SEVERITY_OH_IDX + OBS_SEVERITY_OH_LEN]
        assert np.all((oh == 0.0) | (oh == 1.0)), "Severity OH must be binary {0,1}"

    def test_per_amb_dist_norm_in_unit_interval_k1(self):
        obs_arr = self._collect_obs(K=1)
        dist_idx = OBS_FIXED_BLOCK_LEN + 1  # AMB_DIST_OFFSET=1
        vals = obs_arr[:, dist_idx]
        assert np.all(vals >= 0.0) and np.all(vals <= 1.0 + 1e-6)

    def test_per_amb_speed_norm_non_negative_k1(self):
        obs_arr = self._collect_obs(K=1)
        speed_idx = OBS_FIXED_BLOCK_LEN + 2  # AMB_SPEED_OFFSET=2
        vals = obs_arr[:, speed_idx]
        assert np.all(vals >= 0.0), f"speed_norm negative: min={vals.min()}"

    def test_per_amb_severity_k_norm_is_discrete_k1(self):
        """severity_k_norm = sev_k / 5.0, sev_k ∈ {1,2,3,4,5}.

        Verify by back-multiplying: round(v * 5) must be an integer in {1..5}.
        A range check [0.2,1.0] would pass even if the formula were wrong.
        Direct round-to-10 would fail due to float32 precision (np.float32(2/5.0)
        ≈ 0.40000000596 ≠ 0.4 in float64).
        """
        obs_arr = self._collect_obs(K=1, n_steps=100)
        sev_k_idx = OBS_FIXED_BLOCK_LEN + AMB_SEVERITY_NORM_OFFSET
        for row_i, v in enumerate(obs_arr[:, sev_k_idx]):
            sev_int = round(float(v) * 5)       # back-multiply to recover integer severity
            assert sev_int in {1, 2, 3, 4, 5}, (
                f"step {row_i}: severity_k_norm={v:.8f} → round(v*5)={sev_int} not in {{1..5}}. "
                "Formula must be sev_k/5.0 with integer sev_k ∈ {1..5}."
            )

    def test_per_amb_lambda_slots_non_negative_k1(self):
        """λ_local slots (C1/C2/C4/C5 per-ambulance + C3 shared) must be ≥ 0."""
        K = 1
        env = ORANEnv(EnvConfig(K_ambulances=K))
        obs, info = env.reset(seed=0)
        from utils.obs import overlay_lambda_local as _ov
        sev = int(info["severity"])
        ls = LambdaState(K=K)
        ls.reset_episode([sev], sev)
        for _ in range(50):
            act = env.action_space.sample()
            raw_obs, _, terminated, truncated, info = env.step(act)
            c, d = info["c_vec"], info["d_phi"]
            ls.accumulate(c, d)
            if (ls.win_steps % WORKER_STEPS_PER_MANAGER) == 0:
                ls.on_manager_step_end()
            s = _ov(raw_obs, ls.get_lambda_local(), K)
            # Check λ_C3 slot
            assert s[LAMBDA_C3_SHARED_OBS_INDEX] >= 0.0
            # Check per-amb λ slots
            base = OBS_FIXED_BLOCK_LEN
            for off in [AMB_LAMBDA_C1_OFFSET, AMB_LAMBDA_C2_OFFSET,
                        AMB_LAMBDA_C4_OFFSET, AMB_LAMBDA_C5_OFFSET]:
                assert s[base + off] >= 0.0
            if terminated or truncated:
                env.reset(seed=0)
                ls.reset_episode([sev], sev)
        env.close()

    def test_aoi_mean_non_negative_and_finite(self):
        obs_arr = self._collect_obs(K=1, n_steps=50)
        vals = obs_arr[:, OBS_AOI_MEAN_IDX]
        assert np.all(np.isfinite(vals)) and np.all(vals >= 0.0)

    def test_aoi_max_ge_aoi_mean(self):
        obs_arr = self._collect_obs(K=1, n_steps=50)
        mean_v = obs_arr[:, OBS_AOI_MEAN_IDX]
        max_v = obs_arr[:, OBS_AOI_MAX_IDX]
        assert np.all(max_v >= mean_v - 1e-6), "aoi_max must be >= aoi_mean at every step"

    def test_all_obs_finite_no_nan_k3(self):
        obs_arr = self._collect_obs(K=3, n_steps=50)
        assert not np.any(np.isnan(obs_arr)), "NaN in obs (K=3)"
        assert not np.any(np.isinf(obs_arr)), "Inf in obs (K=3)"


# ============================================================
# 6. overlay_lambda_local — slot-by-slot
# ============================================================

class TestOverlayLambdaLocal:
    """Verify exact obs indices written by overlay_lambda_local (K=1 and K=3)."""

    def test_k1_c3_shared_slot(self):
        obs = np.zeros(31, dtype=np.float32)
        lam = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)  # C1..C5 at K=1
        out = overlay_lambda_local(obs, lam, K=1)
        assert out[LAMBDA_C3_SHARED_OBS_INDEX] == pytest.approx(0.5)   # lam[4K] = lam[4]

    def test_k1_per_amb_c1_c2_c4_c5_slots(self):
        obs = np.zeros(31, dtype=np.float32)
        lam = np.array([0.11, 0.22, 0.33, 0.44, 0.55], dtype=np.float32)
        out = overlay_lambda_local(obs, lam, K=1)
        base = OBS_FIXED_BLOCK_LEN  # = 20
        assert out[base + AMB_LAMBDA_C1_OFFSET] == pytest.approx(0.11)  # C1_0 = lam[0]
        assert out[base + AMB_LAMBDA_C2_OFFSET] == pytest.approx(0.22)  # C2_0 = lam[1]
        assert out[base + AMB_LAMBDA_C4_OFFSET] == pytest.approx(0.33)  # C4_0 = lam[2]
        assert out[base + AMB_LAMBDA_C5_OFFSET] == pytest.approx(0.44)  # C5_0 = lam[3]

    def test_k1_does_not_mutate_input(self):
        obs = np.ones(31, dtype=np.float32)
        lam = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        obs_copy = obs.copy()
        overlay_lambda_local(obs, lam, K=1)
        np.testing.assert_array_equal(obs, obs_copy, err_msg="overlay_lambda_local must not mutate input")

    def test_k3_c3_shared_slot(self):
        obs_dim = OBS_FIXED_BLOCK_LEN + OBS_PER_AMB_BLOCK_LEN * 3 + 1  # 51
        obs = np.zeros(obs_dim, dtype=np.float32)
        # lam shape (13,): [C1_0,C1_1,C1_2, C2_0,C2_1,C2_2, C4_0,C4_1,C4_2, C5_0,C5_1,C5_2, C3]
        lam = np.arange(1, 14, dtype=np.float32) * 0.1   # 0.1..1.3
        out = overlay_lambda_local(obs, lam, K=3)
        assert out[LAMBDA_C3_SHARED_OBS_INDEX] == pytest.approx(lam[12])   # C3_shared = lam[12]

    def test_k3_per_amb_k0_slots(self):
        obs_dim = OBS_FIXED_BLOCK_LEN + OBS_PER_AMB_BLOCK_LEN * 3 + 1
        obs = np.zeros(obs_dim, dtype=np.float32)
        lam = np.arange(1, 14, dtype=np.float32) * 0.1
        out = overlay_lambda_local(obs, lam, K=3)
        base0 = OBS_FIXED_BLOCK_LEN + 0
        assert out[base0 + AMB_LAMBDA_C1_OFFSET] == pytest.approx(lam[0])   # C1_0
        assert out[base0 + AMB_LAMBDA_C2_OFFSET] == pytest.approx(lam[3])   # C2_0
        assert out[base0 + AMB_LAMBDA_C4_OFFSET] == pytest.approx(lam[6])   # C4_0
        assert out[base0 + AMB_LAMBDA_C5_OFFSET] == pytest.approx(lam[9])   # C5_0

    def test_k3_per_amb_k1_slots(self):
        obs_dim = OBS_FIXED_BLOCK_LEN + OBS_PER_AMB_BLOCK_LEN * 3 + 1
        obs = np.zeros(obs_dim, dtype=np.float32)
        lam = np.arange(1, 14, dtype=np.float32) * 0.1
        out = overlay_lambda_local(obs, lam, K=3)
        base1 = OBS_FIXED_BLOCK_LEN + OBS_PER_AMB_BLOCK_LEN * 1
        assert out[base1 + AMB_LAMBDA_C1_OFFSET] == pytest.approx(lam[1])   # C1_1
        assert out[base1 + AMB_LAMBDA_C2_OFFSET] == pytest.approx(lam[4])   # C2_1
        assert out[base1 + AMB_LAMBDA_C4_OFFSET] == pytest.approx(lam[7])   # C4_1
        assert out[base1 + AMB_LAMBDA_C5_OFFSET] == pytest.approx(lam[10])  # C5_1

    def test_k3_per_amb_k2_slots(self):
        obs_dim = OBS_FIXED_BLOCK_LEN + OBS_PER_AMB_BLOCK_LEN * 3 + 1
        obs = np.zeros(obs_dim, dtype=np.float32)
        lam = np.arange(1, 14, dtype=np.float32) * 0.1
        out = overlay_lambda_local(obs, lam, K=3)
        base2 = OBS_FIXED_BLOCK_LEN + OBS_PER_AMB_BLOCK_LEN * 2
        assert out[base2 + AMB_LAMBDA_C1_OFFSET] == pytest.approx(lam[2])   # C1_2
        assert out[base2 + AMB_LAMBDA_C2_OFFSET] == pytest.approx(lam[5])   # C2_2
        assert out[base2 + AMB_LAMBDA_C4_OFFSET] == pytest.approx(lam[8])   # C4_2
        assert out[base2 + AMB_LAMBDA_C5_OFFSET] == pytest.approx(lam[11])  # C5_2

    def test_wrong_lambda_shape_raises(self):
        obs = np.zeros(31, dtype=np.float32)
        with pytest.raises(ValueError):
            overlay_lambda_local(obs, np.zeros(3, dtype=np.float32), K=1)

    def test_unmodified_slots_remain_original_k1(self):
        """Slots not written by overlay must keep their original values."""
        obs = np.ones(31, dtype=np.float32) * 99.0
        lam = np.zeros(5, dtype=np.float32)
        out = overlay_lambda_local(obs, lam, K=1)
        base = OBS_FIXED_BLOCK_LEN
        written = {LAMBDA_C3_SHARED_OBS_INDEX,
                   base + AMB_LAMBDA_C1_OFFSET, base + AMB_LAMBDA_C2_OFFSET,
                   base + AMB_LAMBDA_C4_OFFSET, base + AMB_LAMBDA_C5_OFFSET}
        for i in range(31):
            if i not in written:
                assert out[i] == pytest.approx(99.0), f"slot {i} must be unchanged"


# ============================================================
# 7. mask_severity
# ============================================================

class TestMaskSeverity:
    """mask_severity must zero the severity one-hot AND each per-amb sev_k_norm slot."""

    def test_k1_one_hot_zeroed(self):
        obs = np.ones(31, dtype=np.float32)
        out = mask_severity(obs, K=1, F=1)
        oh = out[OBS_SEVERITY_OH_IDX: OBS_SEVERITY_OH_IDX + OBS_SEVERITY_OH_LEN]
        np.testing.assert_array_equal(oh, np.zeros(OBS_SEVERITY_OH_LEN))

    def test_k1_per_amb_severity_k_norm_zeroed(self):
        obs = np.ones(31, dtype=np.float32)
        out = mask_severity(obs, K=1, F=1)
        sev_k_idx = OBS_FIXED_BLOCK_LEN + AMB_SEVERITY_NORM_OFFSET
        assert out[sev_k_idx] == 0.0

    def test_k3_per_amb_severity_k_norm_all_zeroed(self):
        K, F = 3, 1
        obs_dim = OBS_FIXED_BLOCK_LEN + OBS_PER_AMB_BLOCK_LEN * K + F
        obs = np.ones(obs_dim, dtype=np.float32)
        out = mask_severity(obs, K=K, F=F)
        for k in range(K):
            idx = OBS_FIXED_BLOCK_LEN + OBS_PER_AMB_BLOCK_LEN * k + AMB_SEVERITY_NORM_OFFSET
            assert out[idx] == 0.0, f"per-amb severity_k_norm not zeroed at k={k}"

    def test_does_not_mutate_input(self):
        obs = np.ones(31, dtype=np.float32)
        obs_copy = obs.copy()
        mask_severity(obs, K=1, F=1)
        np.testing.assert_array_equal(obs, obs_copy)

    def test_non_severity_slots_unchanged_k1(self):
        obs = np.full(31, 3.14, dtype=np.float32)
        out = mask_severity(obs, K=1, F=1)
        sev_k_idx = OBS_FIXED_BLOCK_LEN + AMB_SEVERITY_NORM_OFFSET
        masked_indices = (
            set(range(OBS_SEVERITY_OH_IDX, OBS_SEVERITY_OH_IDX + OBS_SEVERITY_OH_LEN))
            | {sev_k_idx}
        )
        for i in range(31):
            if i not in masked_indices:
                assert out[i] == pytest.approx(3.14), f"slot {i} must be unchanged"


# ============================================================
# 8. Manager State Dim and Content
# ============================================================

class TestManagerStateDim:
    """manager_state_dim formula: 8 + 2*(4K+1) (audit 2026-06-23 — adds the
    current g_hat residual alongside λ_global, plus severity_mean_norm and
    n_active_norm so (5,1,1) and (5,5,5) at K=3 no longer alias under
    severity_ref=max). Layout: [0:2] rho_urllc/eMBB, [2] bler,
    [3] severity_ref_norm, [4] severity_mean_norm, [5] n_active_norm,
    [6:8] aoi_mean/max, [8:8+4K+1] lambda_global, [.:.+4K+1] g_hat."""

    def test_k1_dim(self):
        assert manager_state_dim(1) == 18   # 8 + 2*5

    def test_k3_dim(self):
        assert manager_state_dim(3) == 34   # 8 + 2*13

    def test_k1_build_manager_state_shape(self):
        obs = np.zeros(32, dtype=np.float32)
        lam = np.zeros(5, dtype=np.float32)
        g_hat = np.zeros(5, dtype=np.float32)
        s_H = build_manager_state(obs, lam, g_hat)
        assert s_H.shape == (18,)

    def test_k3_build_manager_state_shape(self):
        obs = np.zeros(OBS_FIXED_BLOCK_LEN + OBS_PER_AMB_BLOCK_LEN * 3 + 1, dtype=np.float32)
        lam = np.zeros(13, dtype=np.float32)
        g_hat = np.zeros(13, dtype=np.float32)
        s_H = build_manager_state(obs, lam, g_hat)
        assert s_H.shape == (34,)

    # ------------------------------------------------------------------
    # Sentinel layout tests: put a unique non-zero value in ONE source slot;
    # verify it appears at exactly one s_H index, all others = 0.
    # Catches: wrong index mapping, accidental slot aliasing, cross-contamination.
    # ------------------------------------------------------------------

    def _sentinel_obs(self, obs_dim: int = 32) -> np.ndarray:
        return np.zeros(obs_dim, dtype=np.float32)

    def _sentinel_lam(self, K: int = 1) -> np.ndarray:
        return np.zeros(4 * K + 1, dtype=np.float32)

    def test_sentinel_rho_urllc_lands_only_at_s_H_0(self):
        """rho_urllc → s_H[0], no other slot contaminated."""
        obs = self._sentinel_obs()
        obs[OBS_RHO_URLLC_IDX] = 0.91
        s_H = build_manager_state(obs, self._sentinel_lam(), self._sentinel_lam())
        assert s_H[0] == pytest.approx(0.91)
        # No active ambulance → severity_mean (s_H[4]) falls back to severity_ref
        # (s_H[3]=0.2), by design — excluded from the contamination check below.
        for i in (1, 2, 5):
            assert s_H[i] == pytest.approx(0.0), f"s_H[{i}] contaminated: {s_H[i]}"
        assert s_H[4] == pytest.approx(s_H[3])
        np.testing.assert_allclose(s_H[8:], 0.0, atol=1e-6, err_msg="lambda/g_hat slots contaminated")

    def test_sentinel_rho_embb_lands_only_at_s_H_1(self):
        obs = self._sentinel_obs()
        obs[OBS_RHO_EMBB_IDX] = 0.73
        s_H = build_manager_state(obs, self._sentinel_lam(), self._sentinel_lam())
        assert s_H[1] == pytest.approx(0.73)
        for i in (0, 2, 5):
            assert s_H[i] == pytest.approx(0.0), f"s_H[{i}] contaminated"
        assert s_H[4] == pytest.approx(s_H[3])  # severity_mean fallback, n_active=0

    def test_sentinel_bler_lands_only_at_s_H_2(self):
        obs = self._sentinel_obs()
        obs[OBS_BLER_IDX] = 0.18
        s_H = build_manager_state(obs, self._sentinel_lam(), self._sentinel_lam())
        assert s_H[2] == pytest.approx(0.18)
        for i in (0, 1, 5):
            assert s_H[i] == pytest.approx(0.0), f"s_H[{i}] contaminated"
        assert s_H[4] == pytest.approx(s_H[3])  # severity_mean fallback, n_active=0

    def test_sentinel_severity_oh_lands_only_at_s_H_3(self):
        """obs[OBS_SEVERITY_OH_IDX + k] = 1.0 → sev_ref_norm = (k+1)/5.0 at s_H[3].
        No active ambulance is set, so severity_mean falls back to the same
        value at s_H[4]; n_active_norm stays 0 at s_H[5]. Other scalars = 0.
        """
        for k in range(5):
            obs = self._sentinel_obs()
            obs[OBS_SEVERITY_OH_IDX + k] = 1.0
            s_H = build_manager_state(obs, self._sentinel_lam(), self._sentinel_lam())
            expected_sev_norm = (k + 1) / 5.0
            assert s_H[3] == pytest.approx(expected_sev_norm), (
                f"OH slot {k}: s_H[3]={s_H[3]:.3f} != (k+1)/5={(k+1)/5.0:.3f}"
            )
            assert s_H[4] == pytest.approx(expected_sev_norm), (
                "severity_mean must fall back to severity_ref when n_active=0"
            )
            for i in (0, 1, 2, 5):
                assert s_H[i] == pytest.approx(0.0), f"k={k}: s_H[{i}] contaminated"

    def test_sentinel_severity_mean_over_active_only(self):
        """K=3 with one active ambulance (severity_norm=0.6): severity_mean_norm
        must equal that active xe's severity, NOT be diluted by the inactive
        (zeroed) slots — and n_active_norm = 1/3."""
        K = 3
        obs = self._sentinel_obs(obs_dim=OBS_FIXED_BLOCK_LEN + OBS_PER_AMB_BLOCK_LEN * K + 1)
        per_amb_0 = OBS_FIXED_BLOCK_LEN
        obs[per_amb_0 + AMB_ACTIVE_OFFSET] = 1.0
        obs[per_amb_0 + AMB_SEVERITY_NORM_OFFSET] = 0.6
        lam = np.zeros(4 * K + 1, dtype=np.float32)
        s_H = build_manager_state(obs, lam, lam)
        assert s_H[4] == pytest.approx(0.6)
        assert s_H[5] == pytest.approx(1.0 / 3.0)

    def test_sentinel_aoi_mean_lands_only_at_s_H_6(self):
        obs = self._sentinel_obs()
        obs[OBS_AOI_MEAN_IDX] = 0.55
        s_H = build_manager_state(obs, self._sentinel_lam(), self._sentinel_lam())
        assert s_H[6] == pytest.approx(0.55)
        for i in (0, 1, 2, 7):
            assert s_H[i] == pytest.approx(0.0), f"s_H[{i}] contaminated"

    def test_sentinel_aoi_max_lands_only_at_s_H_7(self):
        obs = self._sentinel_obs()
        obs[OBS_AOI_MAX_IDX] = 0.88
        s_H = build_manager_state(obs, self._sentinel_lam(), self._sentinel_lam())
        assert s_H[7] == pytest.approx(0.88)
        for i in (0, 1, 2, 6):
            assert s_H[i] == pytest.approx(0.0), f"s_H[{i}] contaminated"

    def test_sentinel_lambda_k1_lands_at_s_H_8_to_12(self):
        """lambda_global (5-dim K=1) → s_H[8:13]. Scalar slots [0:8] must stay 0
        (severity_mean at s_H[4] falls back to severity_ref at s_H[3], n_active=0)."""
        obs = self._sentinel_obs()
        from utils.config import LAMBDA_MAX
        lam = np.array([0.11, 0.22, 0.33, 0.44, 0.55], dtype=np.float32)
        g_hat = np.zeros(5, dtype=np.float32)
        s_H = build_manager_state(obs, lam, g_hat)
        # λ normalized by LAMBDA_MAX (audit 2026-06-24); g_hat stays raw.
        np.testing.assert_allclose(s_H[8:13], lam / LAMBDA_MAX, atol=1e-6,
            err_msg="lambda_global/LAMBDA_MAX must appear at s_H[8:13]")
        for i in (0, 1, 2, 5, 6, 7):
            assert s_H[i] == pytest.approx(0.0), f"s_H[{i}] contaminated by lambda"
        assert s_H[4] == pytest.approx(s_H[3])

    def test_sentinel_g_hat_k1_lands_at_s_H_13_to_17(self):
        """g_hat (5-dim K=1) → s_H[13:18], distinct from lambda_global at [8:13]."""
        obs = self._sentinel_obs()
        lam = np.zeros(5, dtype=np.float32)
        g_hat = np.array([-0.1, 0.2, -0.3, 0.4, -0.5], dtype=np.float32)
        s_H = build_manager_state(obs, lam, g_hat)
        np.testing.assert_allclose(s_H[13:18], g_hat, atol=1e-6,
            err_msg="g_hat must appear at s_H[13:18]")
        np.testing.assert_allclose(s_H[8:13], 0.0, atol=1e-6,
            err_msg="lambda slots must not be contaminated by g_hat")

    def test_sentinel_lambda_k3_lands_at_s_H_8_to_20(self):
        """K=3: lambda (13-dim) → s_H[8:21]; g_hat (13-dim) → s_H[21:34]."""
        K = 3
        obs = self._sentinel_obs(obs_dim=OBS_FIXED_BLOCK_LEN + OBS_PER_AMB_BLOCK_LEN * K + 1)
        from utils.config import LAMBDA_MAX
        lam = np.arange(1, 14, dtype=np.float32) * 0.05
        g_hat = np.arange(1, 14, dtype=np.float32) * -0.01
        s_H = build_manager_state(obs, lam, g_hat)
        assert s_H.shape == (34,)
        # λ normalized by LAMBDA_MAX (audit 2026-06-24); g_hat raw/signed.
        np.testing.assert_allclose(s_H[8:21], lam / LAMBDA_MAX, atol=1e-6,
            err_msg="lambda_global/LAMBDA_MAX (K=3) must appear at s_H[8:21]")
        np.testing.assert_allclose(s_H[21:34], g_hat, atol=1e-6,
            err_msg="g_hat (K=3) must appear at s_H[21:34]")

    def test_build_manager_state_all_finite(self):
        env = ORANEnv(EnvConfig(K_ambulances=1))
        obs, info = env.reset(seed=0)
        sev = int(info["severity"])
        ls = LambdaState(K=1)
        ls.reset_episode([sev], sev)
        s_H = build_manager_state(obs, ls.get_lambda_global(), ls.get_deviation_hat())
        assert np.all(np.isfinite(s_H)), f"Manager state has non-finite: {s_H}"
        env.close()
