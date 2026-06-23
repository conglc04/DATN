"""C3 (eMBB throughput floor) constraint-signal + SIGN-CONVENTION lock.

History (context):
    W05 (2026-05-23) refactored env reward to pure Phase 2.1 multi-objective.
    W12 (2026-05-26) restructured to eMBB log-utility ONLY:
        r = alpha_e(sev_ref) * log1p(R_eMBB / R_REF_EMBB_MBPS)
    URLLC enforced via Lagrangian C1, C2 (LambdaState) — NOT in reward.

C3 (eMBB throughput floor R_eMBB >= R_min) is a CONSTRAINT signal, NOT a reward
modifier:
- Tracked as the shared C3 slot c_vec[4K] in env.step() info dict
  ((4K+1)-dim layout [C1_0..,C2_0..,C4_0..,C5_0..,C3_shared]; at K=1 -> index 4).
- Per-step threshold d_phi[4K] = 0; R_min^sev_ref is embedded in the SIGNED gap
  c_vec[4K] = R_min^sev_ref - R_eMBB.
- Penalty handled by Lagrangian lambda_C3 in agent (not embedded in env reward).

WHY THE SIGN MATTERS (the dangerous-if-wrong case):
    The dual ascent is lambda_C3 <- clip(lambda_C3 + alpha * (c - d) / scale, 0, Lmax).
    For C3 to push lambda_C3 UP when the eMBB floor is violated (R_eMBB < R_min),
    the env MUST report c_vec[4K] = R_min - R_eMBB (POSITIVE under violation).
    A flipped gap (R_eMBB - R_min) would still train without error but drive the
    dual variable the WRONG way -> the policy silently learns to STARVE eMBB.
    The tests below lock both the sign and the magnitude so a future refactor that
    flips it fails loudly.
"""

from __future__ import annotations

import numpy as np


def test_enforce_c3_field_removed():
    """EnvConfig must not have enforce_c3 / beta_c3 fields (Phase 2.1 compliance)."""
    from env.oran_env import EnvConfig
    cfg = EnvConfig()
    assert not hasattr(cfg, "enforce_c3"), (
        "enforce_c3 should be removed (W05 refactor — reward is always pure Phase 2.1)"
    )
    assert not hasattr(cfg, "beta_c3"), (
        "beta_c3 should be removed (W05 refactor)"
    )


def test_c3_tracked_as_constraint_signal():
    """C3 (signed eMBB gap, shared) must appear in info['c_vec'][4K] per the
    per-ambulance severity_k epic (4K+1)-dim layout
    [C1_0..C1_{K-1}, C2_0..C2_{K-1}, C4_0..C4_{K-1}, C5_0..C5_{K-1}, C3_shared].
    At K=1 this is index 4."""
    from env.oran_env import ORANEnv, EnvConfig
    env = ORANEnv(config=EnvConfig(initial_severity=3), seed=0)
    env.reset(seed=0)
    _, _, _, _, info = env.step(env.action_space.sample())
    assert "c_vec" in info and info["c_vec"].shape == (5,)
    assert "d_phi" in info and info["d_phi"].shape == (5,)
    # d_phi[4] = 0 because R_min^sev_ref is embedded in the signed gap c_vec[4].
    assert abs(info["d_phi"][4]) < 1e-6


def test_c3_signed_gap_sign_convention(monkeypatch):
    """LOCK the C3 sign: c_vec[4K] = R_min^sev_ref - R_eMBB.

    POSITIVE when R_eMBB < R_min (violation) so dual ascent raises lambda_C3;
    NEGATIVE when R_eMBB > R_min (surplus). A flipped sign trains silently in the
    WRONG direction (policy starves eMBB) — this guard makes that fail loudly.

    eMBB throughput is forced to a known constant via monkeypatch so the
    mean-over-MAC-ticks gap equals R_min - constant exactly (deterministic).
    """
    from env.oran_env import ORANEnv, EnvConfig
    from utils.config import CMDP_D_J_SEVERITY

    sev = 3
    r_min = float(CMDP_D_J_SEVERITY[sev]["d3_embb_mbps"])  # 10.0 Mbps fixed floor (Gate 7)
    C3 = 4  # K=1 shared slot (4K)

    # --- eMBB STARVED: R_eMBB = R_min - 10 < R_min -> violation -> gap > 0 ---
    env = ORANEnv(config=EnvConfig(initial_severity=sev), seed=0)
    env.reset(seed=0)
    monkeypatch.setattr(env, "_compute_embb_throughput_mbps", lambda: r_min - 10.0)
    _, _, _, _, info = env.step(env.action_space.sample())
    assert info["c_vec"][C3] > 0.0, (
        f"C3 gap must be POSITIVE when R_eMBB < R_min (violation); got {info['c_vec'][C3]}"
    )
    assert abs(info["c_vec"][C3] - 10.0) < 1e-4, "gap must equal R_min - R_eMBB = 10.0"

    # --- eMBB SURPLUS: R_eMBB = R_min + 30 > R_min -> satisfied -> gap < 0 ---
    env2 = ORANEnv(config=EnvConfig(initial_severity=sev), seed=0)
    env2.reset(seed=0)
    monkeypatch.setattr(env2, "_compute_embb_throughput_mbps", lambda: r_min + 30.0)
    _, _, _, _, info2 = env2.step(env2.action_space.sample())
    assert info2["c_vec"][C3] < 0.0, (
        f"C3 gap must be NEGATIVE when R_eMBB > R_min (surplus); got {info2['c_vec'][C3]}"
    )
    assert abs(info2["c_vec"][C3] - (-30.0)) < 1e-4, "gap must equal R_min - R_eMBB = -30.0"


def test_c3_violation_raises_lambda_end_to_end():
    """End-to-end: a positive C3 gap (env violation convention) must INCREASE
    lambda_C3 through LambdaState dual ascent — locks env-sign -> dual-direction.
    """
    from agents.lagrangian import LambdaState

    lam = LambdaState(K=1)
    # K=1 (4K+1)=5 layout [C1, C2, C4, C5, C3]; C3 is the last slot (index 4).
    C3 = 4
    lam.reset_episode(severity_per_amb=[3], severity_ref=3)
    lambda_c3_before = lam.get_lambda_global()[C3]

    # Simulate sustained eMBB-floor violation: positive signed gap in the C3 slot
    # (c - d, d_phi[C3] = 0). Accumulate over a Manager window, then ascend.
    c_vec = np.zeros(5, dtype=np.float64)
    d_phi = np.zeros(5, dtype=np.float64)
    c_vec[C3] = 5.0  # R_min - R_eMBB = +5 Mbps deficit
    for _ in range(lam.worker_steps_per_manager):
        lam.accumulate(c_vec, d_phi)
    lam.on_manager_step_end()

    assert lam.get_lambda_global()[C3] > lambda_c3_before, (
        "Positive C3 gap (eMBB below floor) must raise lambda_C3 via dual ascent"
    )
