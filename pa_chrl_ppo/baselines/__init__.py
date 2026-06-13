"""Sibling solvers + Exp6 ablation variants.

Sibling solvers (Table I, alongside PA-CHRL-PPO):
    td3_lag              — off-policy TD3 + Lagrangian (deterministic actor)
    sac_lag              — off-policy SAC + Lagrangian (max-entropy stochastic actor)

Lower-bound + Exp6 ablation variants (NOT in Table I):
    static_slicing       — Fixed 50/50 PRB (lower bound)
    b2_hrl_ppo_soft      — HRL-PPO no CMDP, no phase
    pa_ppo_soft          — Phase-aware, no CMDP → "w/o CMDP"
    no_phase_chrl_ppo    — CHRL-PPO minus phase signaling → "w/o Phase"
    ppo_cmdp_flat        — CMDP, no HRL → "w/o HRL"
"""
