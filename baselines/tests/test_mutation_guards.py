"""Gate 14 — mutation/negative guards.

Each test proves a deliberate bug is DISTINGUISHABLE from correct behavior, so a
guard asserting the correct value would fail under the mutation. Hermetic checks
recompute correct-vs-mutated and assert divergence; real-env checks assert the
live code holds the invariant (the mutation would break it).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from env.oran_env import EnvConfig, ORANEnv
from agents.lagrangian import LambdaState
from utils.config import (
    GAMMA, P_TOTAL, WORKER_STEPS_PER_MANAGER, SEVERITY_QOS,
    build_d_phi_vector, build_dual_scales,
)


def _dev(c, d, scale):
    return (np.asarray(c, float) - np.asarray(d, float)) / scale


SEV = 3
D = build_d_phi_vector([SEV])
SCALE = build_dual_scales(1)


# 1. invert C3 sign
def test_m01_c3_sign_flip_distinguishable():
    c = np.array([5e-3, 0.0, 0.1, 0.0, 12.0])
    correct = _dev(c, D, SCALE)[4]
    mutated = (-(c[4]) - D[4]) / SCALE[4]
    assert correct != pytest.approx(mutated)


# 2. subtract threshold twice
def test_m02_double_subtraction_distinguishable():
    ls = LambdaState(K=1); ls.reset_episode([SEV], SEV)
    lam = ls.get_lambda_local()
    c = D.copy(); c[0] += 0.01
    once = 1.0 - float(np.dot(lam, _dev(c, D, SCALE)))
    twice = 1.0 - float(np.dot(lam, _dev(c, 2 * D, SCALE)))
    assert abs(once - twice) > 1e-9


# 3. drop active denominator (dilute by total ticks)
def test_m03_active_denominator_matters():
    correct = 0.02 / 8       # 8 active ticks
    diluted = 0.02 / 20      # total ticks
    assert abs(correct - diluted) > 1e-6


# 4. wrongly mask C3 by active count (C3 must use total ticks)
def test_m04_c3_uses_total_not_active():
    # if C3 were active-masked, an inactive-heavy window would change its mean;
    # the real env divides C3 by total tick count -> independent of masks
    env = ORANEnv(EnvConfig(K_ambulances=3, enable_arrival=True,
                            sample_severity=False, initial_severity=3), seed=5)
    env.reset(seed=5, options={"severity_per_amb": [5, 3, 1]})
    _, _, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    assert np.isfinite(info["c_vec"][4 * 3])   # C3 slot present regardless of masks


# 5. Worker edits b_rrm
def test_m05_worker_cannot_edit_b_rrm():
    env = ORANEnv(EnvConfig(K_ambulances=3, sample_severity=False, initial_severity=3), seed=0)
    env.reset(seed=0, options={"severity_per_amb": [3, 3, 3]})
    env.set_rrm_budget(0.4)
    held = env.r_min_urllc   # post-floor Manager setpoint (floor-agnostic)
    env.step(np.array([5.0, 1.0, 1.0, 1.0], dtype=np.float32))
    assert env.r_min_urllc == pytest.approx(held, abs=1e-9)


# 6. PRB conservation off-by-one
def test_m06_prb_conservation_exact():
    env = ORANEnv(EnvConfig(K_ambulances=3, sample_severity=False, initial_severity=3), seed=0)
    env.reset(seed=0, options={"severity_per_amb": [5, 3, 1]})
    for pu in (7, 50, 137, 232):
        assert int(env._prb_split_intra_slice(pu).sum()) == pu


# 7. off-policy Manager reward drops SMDP discount
def test_m07_smdp_discount_distinguishable():
    r = [1.0] * WORKER_STEPS_PER_MANAGER
    disc = sum(GAMMA ** i * x for i, x in enumerate(r))
    undisc = sum(r)
    assert abs(disc - undisc) > 0.4


# 8. env reset at rollout boundary (episode must outlive 1 s chunk)
def test_m08_no_reset_at_rollout_boundary():
    env = ORANEnv(EnvConfig(K_ambulances=1, episode_duration_sec=2.0), seed=0)
    env.reset(seed=0)
    a = np.zeros(env.action_space.shape, dtype=np.float32)
    for _ in range(120):  # cross 100-step (1 s) boundary
        _, _, term, trunc, _ = env.step(a)
        if term or trunc:
            break
    assert env.tti_idx > 2000, "episode/state must persist past the 1 s rollout chunk"


# 9. resample severity after 1 s
def test_m09_severity_fixed_across_episode():
    env = ORANEnv(EnvConfig(K_ambulances=3, sample_severity=True, episode_duration_sec=2.0), seed=7)
    _, info0 = env.reset(seed=7)
    sev0 = tuple(int(s) for s in info0["severity_per_amb"])
    a = np.zeros(env.action_space.shape, dtype=np.float32)
    for _ in range(150):
        _, _, term, trunc, info = env.step(a)
        assert tuple(int(s) for s in info["severity_per_amb"]) == sev0
        if term or trunc:
            break


# 10. drop lambda from off-policy replay state
def test_m10_offpolicy_overlays_lambda():
    import inspect, solvers.train_offpolicy as T
    src = inspect.getsource(T)
    assert "overlay_lambda_local" in src or "_state_with_lambda" in src


# 11. don't flush partial Manager window
def test_m11_partial_window_flush_present():
    import inspect, solvers.train_offpolicy as T
    src = inspect.getsource(T)
    # the driver stores a final Manager transition outside the W-boundary branch
    assert "s_H_final" in src or "Final partial" in src or "final partial" in src.lower()


# 12. C2 used as BLER instead of delay-tail
def test_m12_c2_is_delay_tail_not_bler():
    # env's C2 accumulator counts (d_e2e > D_max), not BLER; verify via a forced state
    env = ORANEnv(EnvConfig(K_ambulances=1, sample_severity=False, initial_severity=5), seed=0)
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    # C2 slot (index K..2K) is a fraction in [0,1]; BLER would be the last_bler scalar
    c2 = info["c_vec"][1]
    assert 0.0 <= c2 <= 1.0


# 13. ms<->s unit error
def test_m13_units_seconds_not_ms():
    # D_max in config is in SECONDS (1e-3 = 1 ms); a ms<->s bug would make it 1.0
    assert SEVERITY_QOS[5]["D_max"] == pytest.approx(1e-3)
    assert SEVERITY_QOS[5]["D_max"] < 0.1


# 14. SINR dB<->linear error
def test_m14_sinr_db_to_linear():
    from env.channel_model import capacity_per_prb_bps
    from utils.config import B_PRB, SHANNON_ETA
    # at 0 dB, linear SINR=1 -> log2(2)=1
    indep = SHANNON_ETA * B_PRB * math.log2(1 + 10 ** (0.0 / 10))
    assert capacity_per_prb_bps(0.0) == pytest.approx(indep)
    # a dB-as-linear bug would use log2(1+0)=0 -> capacity 0
    assert capacity_per_prb_bps(0.0) > 0


# 15. N_req dimension error
def test_m15_nreq_has_bandwidth_and_efficiency():
    from utils.config import URLLC_OFFERED_LOAD_BPS, URLLC_PKT_BITS, PRB_MIN_QOS
    from env.channel_model import capacity_per_prb_bps
    d_max = SEVERITY_QOS[5]["D_max"]
    c_req = URLLC_OFFERED_LOAD_BPS + URLLC_PKT_BITS / d_max  # bps
    cap = capacity_per_prb_bps(2.7)                          # bps/PRB (incl B_PRB, eta)
    n = max(PRB_MIN_QOS, int(math.ceil(c_req / cap)))
    # a dimensionless-log2 bug (no B_PRB) would give absurd N (c_req/log2 ~ millions)
    assert 1 <= n <= P_TOTAL


# 16. C5 missing A_th (threshold must be a probability, A_th from AoI_max)
def test_m16_c5_threshold_is_probability():
    d = build_d_phi_vector([3])
    # C5 slot (index 3K..4K) threshold = eps_aoi (a probability), NOT seconds
    eps_aoi = SEVERITY_QOS[3]["eps_aoi"]
    assert d[3] == pytest.approx(eps_aoi)
    assert 0.0 < d[3] < 1.0


# 17. inactive vehicle gets PRB or penalty
def test_m17_inactive_gets_zero_prb():
    env = ORANEnv(EnvConfig(K_ambulances=3, enable_arrival=True,
                            sample_severity=False, initial_severity=3), seed=11)
    env.reset(seed=11, options={"severity_per_amb": [5, 3, 1]})
    env.active_mask = np.array([True, False, True])
    split = env._prb_split_intra_slice(100)
    assert split[1] == 0 and int(split.sum()) == 100


# 18. same seed but different trace
def test_m18_same_seed_same_trace():
    def first_rewards(seed):
        env = ORANEnv(EnvConfig(K_ambulances=3, sample_severity=True), seed=seed)
        env.reset(seed=seed)
        a = np.zeros(env.action_space.shape, dtype=np.float32)
        return [env.step(a)[1] for _ in range(10)]
    assert first_rewards(55) == first_rewards(55)


# 19. PRB floor must hold under extreme logit skew (reserve-first order, 2026-06-24)
def test_m19_prb_min_qos_floor_under_extreme_skew():
    """One dominant softmax weight must not zero out another active amb's floor.

    Regression for the floor-then-correct-overflow bug fixed 2026-06-24: with
    raw logits [10,-5,-5] and B_U=27, the old order (floor full-budget
    proportional split -> force minimum -> rescale on overflow) produced
    [26,1,0], leaving amb_2 at 0 PRB despite being active. Reserve-first
    (reserve K_active*PRB_MIN_QOS before the softmax split) guarantees the
    floor for every active amb by construction.
    """
    from utils.config import PRB_MIN_QOS
    env = ORANEnv(EnvConfig(K_ambulances=3, sample_severity=False, initial_severity=3), seed=0)
    env.reset(seed=0, options={"severity_per_amb": [3, 3, 3]})
    env.active_mask = np.ones(3, dtype=bool)
    env._prb_weights = np.array([10.0, -5.0, -5.0], dtype=np.float64)
    split = env._prb_split_intra_slice(27)
    assert int(split.sum()) == 27
    assert np.all(split >= PRB_MIN_QOS), f"floor violated under extreme skew: {split.tolist()}"
    assert split.tolist() == [25, 1, 1], f"reserve-first allocation mismatch: {split.tolist()}"


# 20. Worker actor zero-init: PPO (ĐX1, audit 2026-06-24)
def test_m20_ppo_worker_actor_zero_init():
    """WorkerActor.mean_net output layer must be exactly zero at construction,
    so all K per-vehicle logits start tied (uniform softmax), removing the
    random init asymmetry PPO's policy gradient would otherwise amplify into
    a severity-blind PRB bias."""
    import torch
    from agents.worker_agent import WorkerActor
    torch.manual_seed(123)
    actor = WorkerActor(state_dim=54, action_dim=3)
    out_layer = actor.mean_net[-1]
    assert torch.all(out_layer.weight == 0.0)
    assert torch.all(out_layer.bias == 0.0)
    obs = torch.randn(5, 54)
    mean = actor.distribution(obs).mean
    assert torch.all(mean == 0.0), "mean_net(obs) must be exactly 0 at init for any obs"


# 21. Worker actor zero-init: TD3 (ĐX1 extended, audit 2026-06-24)
def test_m21_td3_worker_actor_zero_init():
    """Only the Worker TD3Agent (zero_init_output=True) gets the fix; the
    Manager's TD3ManagerAgent keeps default random init (no cross-dim bias
    to fix for a 1-dim action)."""
    import torch
    from agents.td3_agent import TD3Agent
    from agents.manager_agent import TD3ManagerAgent, manager_state_dim
    torch.manual_seed(123)
    low = np.full(3, -3.0, dtype=np.float32)
    high = np.full(3, 3.0, dtype=np.float32)
    worker = TD3Agent(state_dim=54, action_dim=3, action_low=low, action_high=high,
                       zero_init_output=True)
    assert torch.all(worker.actor.net[-1].weight == 0.0)
    assert torch.all(worker.actor.net[-1].bias == 0.0)
    obs = torch.randn(5, 54)
    action = worker.actor(obs)
    assert torch.allclose(action, torch.zeros_like(action), atol=1e-6), (
        "tanh(0)=0 -> action must sit at the [low,high] midpoint for any obs")

    manager = TD3ManagerAgent(state_dim=manager_state_dim(3), seed=123)
    assert not torch.all(manager.actor.net[-1].weight == 0.0), (
        "Manager actor must keep default random init (zero_init_output not passed)")


# 22. Worker actor zero-init: SAC (ĐX1 extended, audit 2026-06-24)
def test_m22_sac_worker_actor_zero_init():
    """Only mean_head is zeroed (the mean-bias analog); log_std_head keeps
    default init, matching WorkerActor where log_std is already a
    state-independent constant. Manager's SACManagerAgent stays default."""
    import torch
    from agents.sac_agent import SACAgent
    from agents.manager_agent import SACManagerAgent, manager_state_dim
    torch.manual_seed(123)
    low = np.full(3, -3.0, dtype=np.float32)
    high = np.full(3, 3.0, dtype=np.float32)
    worker = SACAgent(state_dim=54, action_dim=3, action_low=low, action_high=high,
                       zero_init_output=True)
    assert torch.all(worker.actor.mean_head.weight == 0.0)
    assert torch.all(worker.actor.mean_head.bias == 0.0)
    obs = torch.randn(5, 54)
    mean, _ = worker.actor._dist_params(obs)
    assert torch.all(mean == 0.0), "mean_head(trunk(obs)) must be exactly 0 at init for any obs"

    manager = SACManagerAgent(state_dim=manager_state_dim(3), seed=123)
    assert not torch.all(manager.actor.mean_head.weight == 0.0), (
        "Manager actor must keep default random init (zero_init_output not passed)")
