"""Gate 16 — deterministic no-learning dry run (real env, fixed actions).

Rolls K=1 and K=3 ([3,3,3],[5,3,1]) with a fixed Manager+Worker action and
verifies live: PRB conservation, active masks, no NaN/Inf, severity persistence,
episode does not reset at the 1 s rollout boundary, C1-C5 finite, lambdas finite,
Manager held for W steps. Run: cd baselines && python3 -m audit.dry_run
"""
from __future__ import annotations
import numpy as np
from env.oran_env import EnvConfig, ORANEnv
from agents.lagrangian import LambdaState
from agents.manager_agent import decode_manager_action
from solvers._common import build_manager_state
from agents.manager_agent import manager_state_dim
from utils.config import P_TOTAL, WORKER_STEPS_PER_MANAGER

fails = []
def ck(name, ok, detail=""):
    print(f"  {'✅' if ok else '❌'} {name} {detail}")
    if not ok:
        fails.append(name)


def run(K, sev, label):
    print(f"\n--- dry run {label}: K={K} sev={sev} ---")
    env = ORANEnv(EnvConfig(K_ambulances=K, sample_severity=False,
                            initial_severity=max(sev), episode_duration_sec=2.0), seed=0)
    _, info = env.reset(seed=0, options={"severity_per_amb": sev})
    ls = LambdaState(K=K); ls.reset_episode(sev, max(sev))
    sev0 = tuple(int(s) for s in info["severity_per_amb"])
    a_worker = np.zeros(env.action_space.shape, dtype=np.float32)
    nan_free = True; cons_ok = True; sev_ok = True; cvec_ok = True
    manager_held = True; mgr_state_ok = True
    b_rrm = decode_manager_action(np.array([0.5]))["b_rrm"]
    env.set_rrm_budget(b_rrm)
    held_val = env.r_min_urllc
    steps = 0
    for step in range(250):
        obs, r, term, trunc, info = env.step(a_worker)
        steps += 1
        c = np.asarray(info["c_vec"])
        if not (np.all(np.isfinite(obs)) and np.isfinite(r) and np.all(np.isfinite(c))):
            nan_free = False
        pu, pe = env._prb_allocation()
        if pu + pe != P_TOTAL:
            cons_ok = False
        if int(env._prb_split_intra_slice(pu).sum()) != pu:
            cons_ok = False
        if tuple(int(s) for s in info["severity_per_amb"]) != sev0:
            sev_ok = False
        if c.shape != (4 * K + 1,):
            cvec_ok = False
        # Manager setpoint must be held within the window (we never re-set it)
        if step < WORKER_STEPS_PER_MANAGER and abs(env.r_min_urllc - held_val) > 1e-9:
            manager_held = False
        ls.accumulate(c, np.asarray(info["d_phi"]))
        if (step + 1) % WORKER_STEPS_PER_MANAGER == 0:
            ls.on_manager_step_end()
            # Manager state (incl. λ_global + current g_hat residual) at the
            # boundary must match the (10+8K)-dim contract and stay finite.
            s_H = build_manager_state(obs, ls.get_lambda_global(), ls.get_deviation_hat())
            if s_H.shape != (manager_state_dim(K),) or not np.all(np.isfinite(s_H)):
                mgr_state_ok = False
        if term or trunc:
            break
    ck(f"{label}.no_nan_inf", nan_free)
    ck(f"{label}.prb_conservation", cons_ok)
    ck(f"{label}.severity_persists", sev_ok, f"sev={sev0}")
    ck(f"{label}.cvec_dim_4K+1", cvec_ok)
    ck(f"{label}.manager_held_in_window", manager_held)
    ck(f"{label}.manager_state_dim_10+8K", mgr_state_ok, f"expect {manager_state_dim(K)}")
    ck(f"{label}.episode_past_1s_chunk", steps > 100, f"ran {steps} worker steps (>100 = past 1s)")
    ck(f"{label}.lambda_finite", bool(np.all(np.isfinite(ls.get_lambda_global()))))
    ck(f"{label}.active_mask_consistent",
       np.array_equal(env.active_mask, env.entered_mask & ~env.arrived_mask))


def main():
    run(1, [5], "K1")
    run(3, [3, 3, 3], "K3_uniform")
    run(3, [5, 3, 1], "K3_mixed")
    print("\n" + "=" * 60)
    print(f"DRY RUN — {'PASS' if not fails else 'FAIL: ' + ', '.join(fails)}")
    print("=" * 60)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
