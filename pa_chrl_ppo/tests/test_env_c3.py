"""DEPRECATED (W05 refactor + W12 post-critique restructure) — was: enforce_c3 / beta_c3 flag tests.

W05 (2026-05-23) refactored env reward to pure Phase 2.1 multi-objective.
W12 (2026-05-26) further restructured to eMBB log-utility ONLY:
    r = alpha_e(phi) * log1p(R_eMBB / R_REF_EMBB_MBPS)
URLLC enforced via Lagrangian C1, C2 (LambdaState) — NOT in reward.

C3 (eMBB throughput floor) is now a CONSTRAINT signal, NOT a reward modifier.
- Tracked as c_vec[2] in env.step() info dict (5 hard constraints per Phase 2.2.1)
- Per-step phase threshold d_phi[2] = 0; R_min^phi is embedded in signed gap c_vec[2]
- Penalty handled by Lagrangian lambda_3 in agent (not embedded in env reward)

This file is kept as a placeholder so pytest discovery doesn't error.
The actual Phase 2 reward + constraint tracking tests are in:
    tests/test_env_phase2.py
"""

from __future__ import annotations


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
    """C3 (signed eMBB gap) must appear in info['c_vec'][2] per Phase 2.2.1."""
    from env.oran_env import ORANEnv, EnvConfig
    env = ORANEnv(config=EnvConfig(initial_phase=3), seed=0)
    env.reset(seed=0)
    _, _, _, _, info = env.step(env.action_space.sample())
    assert "c_vec" in info and info["c_vec"].shape == (5,)
    assert "d_phi" in info and info["d_phi"].shape == (5,)
    # d_phi[2] = 0 because C3 is c3 = R_min^phi - R_eMBB <= 0.
    assert abs(info["d_phi"][2]) < 1e-6
