"""Sibling solvers — same CMDP, same HRL Manager+Worker, different RL core.

All 3 solvers (PPO in train.py, TD3 and SAC here) solve the SAME optimization
problem with the SAME two-timescale HRL architecture. The only difference is
the RL algorithm core (on-policy PPO vs off-policy TD3/SAC).

Sibling solvers (Table I, alongside PPO):
    td3              — HRL off-policy TD3 + (4K+1)-dim Lagrangian
    sac              — HRL off-policy SAC + (4K+1)-dim Lagrangian

Ablation variants (Table II, Exp6 — isolate contribution of each component):
    pa_ppo_soft          — Phase-aware, no CMDP → "w/o CMDP"
    ppo_cmdp_flat        — CMDP, no HRL → "w/o HRL"
"""
