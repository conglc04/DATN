"""Off-policy training driver for the TD3 / SAC sibling solvers.

TD3 and SAC are SIBLING SOLVERS to PPO (train.py): same env, same CMDP, same
HRL Manager+Worker, same (4K+1)-dim LambdaState, same pure-RL allocation, same episode
definition. The ONLY difference is the RL core (off-policy replay vs PPO's
on-policy GAE) — a legitimate algorithmic difference, not a problem built for
any one solver (see test_solver_parity).

Episode = ONE FULL MISSION: `while not (terminated or truncated)` runs the env
to all-arrived / episode_duration_sec timeout (identical to train.py PPO).

Usage:
    python -m solvers.train_offpolicy --baseline sac --episodes 100 --hard
    python -m solvers.train_offpolicy --baseline td3 --episodes 100 --hard
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from agents.manager_agent import decode_manager_action
from agents.ppo_agent import RolloutBuffer
from env.oran_env import EnvConfig, ORANEnv
from solvers._common import _manager_act, build_manager_state, value_bootstrap_is_terminal
from utils.checkpointing import (
    latest_ckpt_path,
    load_train_state,
    save_train_state,
    state_path,
)
from utils.config import GAMMA, REWARD_FIXED_SCALE, WORKER_STEPS_PER_MANAGER
from utils.early_stopping import EarlyStopping
from utils.logger import Logger
from utils.obs import overlay_lambda_local  # single-source λ overlay (shared with train.py)


BASELINE_REGISTRY = {
    "td3":            "solvers.td3:TD3Solver",
    "sac":            "solvers.sac:SACSolver",
}


def make_baseline(name: str, state_dim: int, action_dim: int, seed: int, device: str = "cpu", K: int = 1):
    if name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline: {name}. Choose from {list(BASELINE_REGISTRY)}")
    mod_path, cls_name = BASELINE_REGISTRY[name].split(":")
    import importlib
    mod = importlib.import_module(mod_path)
    cls = getattr(mod, cls_name)
    return cls(state_dim=state_dim, action_dim=action_dim, seed=seed, device=device, K=K)


def train(
    baseline_name: str,
    n_episodes: int,
    seed: int = 0,
    log_dir: str = "logs",
    initial_severity: int = 5,
    urllc_lambda: float = 50.0,
    M_eMBB: int = 30,
    device: str = "cpu",
    print_every: int = 10,
    use_wandb: bool = False,
    checkpoint_dir: str = "checkpoints",
    checkpoint_every: int = 500,
    hard_mission: bool = False,
    macro_mission: bool = False,
    K_ambulances: int = 1,
    enforce_c3: bool = False,  # DEPRECATED — env reward is always pure Phase 2.1
    early_stop: bool = False,
    early_stop_patience: int = 300,
    early_stop_min_delta: float = 10.0,
    early_stop_window: int = 100,
    early_stop_min_ep: int = 500,
    eval_at: int = 5000,
    resume_checkpoint: str | None = None,  # path to .pt file to resume from (explicit)
    resume_start_ep: int = 0,              # episode offset (metrics append after this ep)
    resume: bool = False,                  # auto-resume from rolling *_latest.pt
) -> dict:
    if enforce_c3:
        import warnings
        warnings.warn(
            "enforce_c3=True is deprecated. Reward is always pure Phase 2.1; "
            "C3 handled via Lagrangian λ_3.", DeprecationWarning, stacklevel=2,
        )
    if macro_mission:
        # Full SUMO mission (K=3 UMa 1km, episode = journey until all-arrived/400s).
        # Same env definition as PPO (train.py) → fair 3-solver comparison.
        from env.oran_env import macro_mission_config
        env_cfg = macro_mission_config(K_ambulances=K_ambulances)
    elif hard_mission:
        from env.oran_env import hard_mission_config
        env_cfg = hard_mission_config(K_ambulances=K_ambulances)
    else:
        env_cfg = EnvConfig(
            initial_severity=initial_severity,
            urllc_arrival_rate=urllc_lambda,
            M_eMBB=M_eMBB,
            K_ambulances=K_ambulances,
            sample_severity=True,
        )
    env = ORANEnv(config=env_cfg, seed=seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    K = env.config.K_ambulances
    agent = make_baseline(baseline_name, state_dim, action_dim, seed, device=device, K=K)

    # --- Resume: explicit path wins; else auto-resume from rolling latest ---
    ckpt_dir = Path(checkpoint_dir)
    latest = latest_ckpt_path(ckpt_dir, baseline_name, seed)
    state_file = state_path(ckpt_dir, baseline_name, seed)
    if resume_checkpoint is not None:
        agent.load(resume_checkpoint)
        print(f"[{baseline_name}] Resumed from {resume_checkpoint} (start_ep={resume_start_ep})")
    elif resume:
        st = load_train_state(state_file)
        if st is not None and latest.exists():
            if int(st.get("seed", seed)) != seed:
                raise ValueError(
                    f"Resume seed mismatch: state seed={st.get('seed')} != run seed={seed}"
                )
            agent.load(str(latest))
            resume_start_ep = int(st["last_ep"])
            lam_st = st.get("lambda_state")
            if lam_st is not None:
                agent.lambda_state.load_state_dict(lam_st)
                print(f"[{baseline_name}] RESUME from ep {resume_start_ep} (with LambdaState, seed={seed})")
            else:
                print(f"[{baseline_name}] RESUME from ep {resume_start_ep} (no LambdaState in ckpt, seed={seed})")
        else:
            print(f"[{baseline_name}] --resume requested but no latest at {state_file} — starting fresh")

    logger = Logger(
        run_name=f"{baseline_name}_seed{seed}",
        log_dir=log_dir,
        use_tensorboard=False,
        use_wandb=use_wandb,
        append_csv=(resume_checkpoint is not None or resume_start_ep > 0),
    )
    logger.log_hparams({
        "baseline": baseline_name,
        "n_episodes": n_episodes,
        "seed": seed,
        "K_ambulances": K,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "initial_severity": initial_severity,
        "sample_severity": env.config.sample_severity,
    })

    # Detect API surface
    has_lambda_state = hasattr(agent, "lambda_state")     # NEW 5-dim (TD3 + SAC siblings)
    has_old_lagrangian = hasattr(agent, "lagrangian")     # OLD 2-dim (ablation only)
    is_off_policy = hasattr(agent, "store_transition")
    has_manager = hasattr(agent, "manager")               # algorithm-matched Manager (TD3/SAC)
    buffer = None
    if hasattr(agent, "ppo") and not is_off_policy:
        buffer = RolloutBuffer(capacity=2010, state_dim=state_dim, action_dim=action_dim)

    def _state_with_lambda(raw_obs: np.ndarray) -> np.ndarray:
        """Inject the CURRENT λ_local into obs[17:22] (Markov state for the CMDP).

        Single source of truth = utils.obs.overlay_lambda_local, identical to the
        PPO path in train.py. Solvers without a 5-dim LambdaState (static / old
        2-dim ablations) keep the raw obs unchanged.
        """
        if has_lambda_state:
            return overlay_lambda_local(raw_obs, agent.lambda_state.get_lambda_local(), K)
        return raw_obs.astype(np.float32, copy=False)

    es = EarlyStopping(
        patience=early_stop_patience,
        min_delta=early_stop_min_delta,
        window=early_stop_window,
        min_ep=early_stop_min_ep,
    ) if early_stop else None

    t_start = time.time()
    episode_rewards = []
    final_stats: dict = {}

    # Fixed reward scale (replaces adaptive ReturnNormalizer, audit 2026-06-22).
    # Same constant for all 3 solvers → fair comparison preserved.
    reward_scale = float(REWARD_FIXED_SCALE)

    # Auto-resume treats n_episodes as the TARGET total → run only the remaining.
    # Manual resume_checkpoint keeps offset semantics (n_episodes = increment;
    # required by run_30runs.py). Fresh run: n_iters == n_episodes.
    n_iters = max(0, n_episodes - resume_start_ep) if (resume and resume_checkpoint is None) else n_episodes

    for ep in range(n_iters):
        global_ep = ep + resume_start_ep   # absolute episode index for logging
        obs, info = env.reset(seed=seed + global_ep)
        if buffer is not None:
            buffer.reset()

        # ---- NEW LambdaState lifecycle: sync λ_global + λ_local from λ_warm[severity] ----
        if has_lambda_state:
            severity_per_amb_init = tuple(int(s) for s in info["severity_per_amb"])
            severity_init = int(info["severity"])
            agent.on_episode_start(severity_per_amb_init, severity_init)
            agent.on_manager_step_start(severity_per_amb_init, severity_init)
        # (fixed-scale: no per-episode reset needed — scale is constant)

        ep_reward = 0.0
        ep_reward_normalized = 0.0   # fix D: sum of what the critic actually saw
        worker_step_idx = 0
        terminated = truncated = False
        # Accumulate n_active and per-ambulance active MAC-tick count for metrics.
        ep_n_active_sum = 0
        ep_steps = 0
        ep_active_count_per_amb = np.zeros(K, dtype=np.float64)
        # State carries the λ active for THIS decision (built post warm-start sync).
        s = _state_with_lambda(obs)
        # Manager initialization — set RRM budget for the first Manager window
        r_H_acc = 0.0
        intra_window_step = 0
        s_H_prev = None
        a_H_raw_prev = None
        if has_manager and has_lambda_state:
            s_H = build_manager_state(obs, agent.lambda_state.get_lambda_global())
            a_H_raw = _manager_act(agent.manager, s_H)
            b_rrm = decode_manager_action(a_H_raw)["b_rrm"]
            env.set_rrm_budget(b_rrm)
            s_H_prev, a_H_raw_prev = s_H, a_H_raw
        while not (terminated or truncated):
            action, log_prob, value = agent.select_action(s)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            ep_n_active_sum += int(info.get("n_active", K))
            ep_steps += 1
            if "active_count_per_amb" in info:
                ep_active_count_per_amb += info["active_count_per_amb"]

            # === Augmented reward + λ accumulation (use the λ embedded in `s`) ===
            if has_lambda_state:
                c_vec = info["c_vec"]
                d_phi = info["d_phi"]
                agent.accumulate_constraint(c_vec, d_phi)
                aug = agent.augment_reward(float(reward), c_vec, d_phi)
            elif has_old_lagrangian:
                # OLD 2-dim path for ablation variants
                recent_d_e2e = env.e2e_history[-20:] if env.e2e_history else []
                if hasattr(agent, "compute_constraints"):
                    agent.compute_constraints(recent_d_e2e)
                    aug = agent.augment_reward(float(reward), recent_d_e2e)
                else:
                    from solvers._common import estimate_constraints
                    cons = estimate_constraints(
                        recent_d_e2e, embb_mbps=20.0,
                        aoi_samples=None, severity=initial_severity,
                    )
                    aug = agent.augment_reward(float(reward), cons)
            else:
                aug = (
                    agent.augment_reward(float(reward), env.e2e_history[-20:])
                    if hasattr(agent, "augment_reward")
                    else float(reward)
                )

            aug_raw = float(aug)
            aug = aug_raw / reward_scale
            ep_reward_normalized += float(aug)   # what the critic saw

            if has_manager:
                # SMDP-discounted intra-window return (must match PPO path in train.py)
                r_H_acc += (GAMMA ** intra_window_step) * float(aug)
                intra_window_step += 1

            worker_step_idx += 1

            # === Manager step boundary: dual ascent + severity resync BEFORE s' ===
            # Timing convention (off-policy correctness): the λ embedded in `s`
            # is the PRE-update dual and is exactly the λ used for `aug` above.
            # The dual ascent fires here, so the next state `s_next` carries the
            # POST-update λ — the dual the next action will actually be
            # conditioned on. This keeps each replayed (s, a, r, s') tuple
            # self-consistent for the off-policy critic.
            if has_lambda_state and worker_step_idx % WORKER_STEPS_PER_MANAGER == 0:
                agent.on_manager_step_end()
                if has_manager:
                    s_H_next_mgr = build_manager_state(next_obs, agent.lambda_state.get_lambda_global())
                    if s_H_prev is not None:
                        # TD-target bootstrap mask = terminated (true terminal), NOT done:
                        # a 400s timeout truncation is non-terminal (value_bootstrap_is_terminal).
                        agent.manager.store(s_H_prev, a_H_raw_prev, r_H_acc, s_H_next_mgr,
                                            value_bootstrap_is_terminal(terminated))
                        agent.manager.update()
                    r_H_acc = 0.0
                    intra_window_step = 0
                if not done:
                    severity_per_amb_now = tuple(int(s) for s in info["severity_per_amb"])
                    severity_ref_now = int(info["severity"])
                    agent.on_manager_step_start(severity_per_amb_now, severity_ref_now)
                    if has_manager:
                        a_H_raw = _manager_act(agent.manager, s_H_next_mgr)
                        b_rrm = decode_manager_action(a_H_raw)["b_rrm"]
                        env.set_rrm_budget(b_rrm)
                        s_H_prev = s_H_next_mgr
                        a_H_raw_prev = a_H_raw

            s_next = _state_with_lambda(next_obs)

            # === Store transition / sample (PPO buffer vs TD3 off-policy) ===
            # Value-bootstrap mask = terminated (true terminal), NOT done: a 400s
            # timeout truncation is non-terminal and must bootstrap V(s')
            # (value_bootstrap_is_terminal). `done` stays only for loop/episode control.
            bootstrap_terminal = value_bootstrap_is_terminal(terminated)
            if buffer is not None:
                buffer.add(
                    agent.maybe_mask(s).astype(np.float32), action.astype(np.float32),
                    log_prob, aug, value, bootstrap_terminal,
                )
            elif is_off_policy:
                # store_transition applies maybe_mask internally; s / s_next
                # already carry λ (and the severity one-hot for severity-aware solvers).
                agent.store_transition(s, action.astype(np.float32), aug, s_next, bootstrap_terminal)
                agent.update()

            obs = next_obs
            s = s_next
            ep_reward += aug_raw   # RAW augmented reward (diagnostics); training uses normalized

            if buffer is not None and buffer.full:
                # Mid-episode buffer flush: ALWAYS bootstrap V(s') — the cut is not a
                # terminal (mirrors PPO rollout-cut handling in train.py).
                _, _, boot_v = agent.select_action(s, deterministic=True)
                buffer.compute_gae(last_value=float(boot_v))
                agent.update(buffer)
                buffer.reset()

        # End of episode — flush remaining buffer. Zero the bootstrap ONLY on a true
        # terminal; a 400s timeout truncation bootstraps V(s') (value_bootstrap_is_terminal).
        if buffer is not None and buffer.ptr > 0:
            if value_bootstrap_is_terminal(terminated):
                last_value = 0.0
            else:
                _, _, boot_v = agent.select_action(s, deterministic=True)
                last_value = float(boot_v)
            buffer.compute_gae(last_value=last_value)
            agent.update(buffer)
            buffer.reset()

        # Final partial Manager window (only fires if last episode step wasn't
        # already on a Manager boundary)
        if has_lambda_state and worker_step_idx % WORKER_STEPS_PER_MANAGER != 0:
            agent.on_manager_step_end()
            if has_manager and s_H_prev is not None:
                s_H_final = build_manager_state(obs, agent.lambda_state.get_lambda_global())
                # Final partial Manager window: bootstrap mask = terminated, NOT hardcoded
                # True — a 400s timeout truncation is non-terminal (value_bootstrap_is_terminal).
                agent.manager.store(s_H_prev, a_H_raw_prev, r_H_acc, s_H_final,
                                    value_bootstrap_is_terminal(terminated))
                agent.manager.update()
        if has_lambda_state:
            lam_for_log = agent.lambda_state.get_lambda_global()
            severity_per_amb_logged = str(list(agent.lambda_state.sev_prev))
            severity_per_amb_final = tuple(int(s) for s in info["severity_per_amb"])
            severity_ref_final = int(info["severity"])
            agent.lambda_state.on_episode_end(severity_per_amb_final, severity_ref_final)

        # OLD API: dual-ascent at end of episode (NOT per Manager step)
        if has_old_lagrangian and not has_lambda_state:
            from solvers._common import estimate_constraints
            cons_ep = estimate_constraints(
                env.e2e_history, embb_mbps=20.0,
                aoi_samples=None, severity=initial_severity,
            )
            if hasattr(agent, "compute_constraints"):
                cons_ep = agent.compute_constraints(env.e2e_history)
            agent.update_lambdas(cons_ep)

        mean_e2e_ms = env.mean_e2e_ms()
        viol = env.episode_violation_rate()
        episode_rewards.append(ep_reward)
        mean_n_active = ep_n_active_sum / ep_steps if ep_steps > 0 else 0.0
        metrics: dict = {
            "ep_reward": ep_reward,
            "ep_reward_normalized": ep_reward_normalized,   # fix D: sum of critic-seen reward
            "mean_e2e_ms": mean_e2e_ms,
            "viol_rate": viol,
            "mean_embb_mbps": env.mean_embb_mbps(),
            "c3_viol_rate": env.c3_violation_rate(),
            "mean_aoi_ms": env.mean_aoi_ms(),
            "aoi_viol_rate": env.aoi_violation_rate(),
            "mean_n_active": mean_n_active,
        }
        for k in range(K):
            metrics[f"active_mac_ticks_amb{k}"] = float(ep_active_count_per_amb[k])
        if has_lambda_state:
            lam = lam_for_log
            metrics["severity_per_amb"] = severity_per_amb_logged
            for k in range(K):
                metrics[f"lambda_global_C1_{k}"] = float(lam[k])
                metrics[f"lambda_global_C2_{k}"] = float(lam[K + k])
                metrics[f"lambda_global_C4_{k}"] = float(lam[2 * K + k])
                metrics[f"lambda_global_C5_{k}"] = float(lam[3 * K + k])
            metrics["lambda_global_C3_shared"] = float(lam[4 * K])
        elif has_old_lagrangian:
            for j, lam_j in enumerate(agent.lagrangian.lambdas):
                metrics[f"lambda_{j + 1}"] = float(lam_j)
        logger.log_dict(metrics, step=global_ep)
        final_stats = {
            "ep_reward": ep_reward,
            "mean_e2e_ms": mean_e2e_ms,
            "viol_rate": viol,
            "mean_embb_mbps": env.mean_embb_mbps(),
            "c3_viol_rate": env.c3_violation_rate(),
            "mean_aoi_ms": env.mean_aoi_ms(),
            "aoi_viol_rate": env.aoi_violation_rate(),
        }

        if checkpoint_every > 0 and ((ep + 1) % checkpoint_every == 0 or ep == n_iters - 1):
            ckpt_path = Path(checkpoint_dir) / f"{baseline_name}_seed{seed}_ep{global_ep + 1}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                agent.save(str(ckpt_path))
            except Exception as exc:
                print(f"[checkpoint] {baseline_name} save skipped: {exc}")

        # Rolling auto-save AFTER EVERY episode (overwrites; for --resume).
        try:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            agent.save(str(latest))
            save_train_state(state_file, last_ep=global_ep + 1, seed=seed,
                             extra={
                                 "baseline": baseline_name,
                                 "n_episodes_target": n_episodes,
                                 "lambda_state": agent.lambda_state.state_dict(),
                                 "reward_fixed_scale": reward_scale,
                             })
        except Exception as exc:
            print(f"[autosave] {baseline_name} latest save skipped: {exc}")

        if (ep + 1) % print_every == 0 or ep == n_iters - 1:
            elapsed = time.time() - t_start
            avg_r = float(np.mean(episode_rewards[-print_every:]))
            es_tag = f"  no_improve={es._no_improve_eps}" if es else ""
            print(
                f"[{baseline_name}] ep {global_ep + 1} (+{ep + 1}/{n_episodes})  "
                f"avg_r={avg_r:+.3f}  e2e={mean_e2e_ms:.3f}ms  "
                f"viol={viol:.4f}{es_tag}  ({elapsed:.1f}s)"
            )

        # Eval checkpoint + early stopping (use global_ep for milestone checks)
        if es is not None:
            es.maybe_save_eval(global_ep, metrics, log_dir, eval_at=eval_at,
                               run_name=f"{baseline_name}_seed{seed}")
            if es.step(global_ep, ep_reward):
                print(
                    f"[{baseline_name}] EARLY STOP at ep {global_ep + 1}  "
                    f"rolling_mean={es.rolling_mean:+.2f}  "
                    f"no_improve={es._no_improve_eps} >= patience={es.patience}"
                )
                break
        elif eval_at > 0 and (global_ep + 1) == eval_at:
            import json as _json, time as _time
            snap = {"eval_ep": global_ep + 1, "timestamp": _time.strftime("%Y-%m-%d %H:%M:%S")}
            snap.update(metrics)
            out = Path(log_dir) / f"eval_ep{eval_at}_{baseline_name}_seed{seed}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(_json.dumps(snap, indent=2, default=float), encoding="utf-8")
            print(f"[eval checkpoint] ep={ep + 1}  -> {out.name}")

    tail = max(1, min(100, len(episode_rewards)))
    summary = {
        "baseline": baseline_name,
        "seed": seed,
        "n_episodes": n_episodes,
        "final_window": tail,
        "mean_reward_tail": float(np.mean(episode_rewards[-tail:])),
        "std_reward_tail": float(np.std(episode_rewards[-tail:])),
        "final_e2e_ms": final_stats.get("mean_e2e_ms", float("nan")),
        "final_viol_rate": final_stats.get("viol_rate", float("nan")),
        "ep_reward": final_stats.get("ep_reward", float("nan")),
        "mean_e2e_ms": final_stats.get("mean_e2e_ms", float("nan")),
        "viol_rate": final_stats.get("viol_rate", float("nan")),
        "mean_embb_mbps": final_stats.get("mean_embb_mbps", float("nan")),
        "c3_viol_rate": final_stats.get("c3_viol_rate", float("nan")),
    }
    if has_lambda_state:
        summary["final_lambdas"] = agent.lambda_state.get_lambda_global().tolist()
        summary["final_lambda_warm"] = {
            str(k): v.tolist() for k, v in agent.lambda_state.get_lambda_warm_table_snapshot().items()
        }
    elif has_old_lagrangian:
        summary["final_lambdas"] = agent.lagrangian.lambdas.tolist()
    import json
    summary_path = Path(log_dir) / f"summary_{baseline_name}_seed{seed}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[{baseline_name}] DONE — summary saved to {summary_path}")
    print(f"  tail mean_reward = {summary['mean_reward_tail']:+.2f} (std {summary['std_reward_tail']:.2f})")
    print(f"  final e2e        = {summary['final_e2e_ms']:.3f} ms")
    print(f"  final viol_rate  = {summary['final_viol_rate']:.4f}")

    logger.close()
    env.close()
    return summary


def _build_offpolicy_parser(include_baseline: bool) -> argparse.ArgumentParser:
    """Shared CLI for the off-policy solvers.

    ``include_baseline=True`` keeps the legacy ``--baseline`` switch (main());
    the per-solver entry points (train_td3.py / train_sac.py) pass False and
    hard-code their baseline so TD3 and SAC are launched from SEPARATE files.
    """
    p = argparse.ArgumentParser(description="Off-policy solver training")
    if include_baseline:
        p.add_argument("--baseline", required=True, choices=list(BASELINE_REGISTRY))
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-dir", type=str, default="logs")
    p.add_argument("--severity", type=int, default=5,
                   help="Fixed patient severity 1..5 (NON_URGENT..IMMEDIATE)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--checkpoint-dir", type=str, default=None,
                   help="Checkpoint directory. Defaults to <log-dir>/checkpoints if not set.")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--hard", action="store_true", help="Use the hard-mission preset")
    p.add_argument("--macro", action="store_true",
                   help="Use W15-B2 macro mission. Defaults to K=3 unless --K is set.")
    p.add_argument("--K", dest="K_ambulances", type=int, default=None,
                   help="Number of ambulances. Defaults to 3 for --macro and 1 otherwise.")
    p.add_argument("--enforce-c3", action="store_true",
                   help="DEPRECATED — reward is always pure Phase 2.1")
    p.add_argument("--resume", action="store_true",
                   help="Auto-resume from the rolling *_latest.pt checkpoint for this seed.")
    return p


def run_cli(baseline_name: str, argv: list[str] | None = None) -> int:
    """Entry point for a SINGLE off-policy solver (used by train_td3 / train_sac)."""
    args = _build_offpolicy_parser(include_baseline=False).parse_args(argv)
    K_ambulances = args.K_ambulances if args.K_ambulances is not None else (3 if args.macro else 1)
    run_dir = str(Path(args.log_dir) / f"{baseline_name}_K{K_ambulances}_seed{args.seed}")
    if args.checkpoint_dir is None:
        args.checkpoint_dir = str(Path(run_dir) / "checkpoints")
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            print("[warn] CUDA not available, falling back to CPU")
            args.device = "cpu"
    train(
        baseline_name=baseline_name, n_episodes=args.episodes, seed=args.seed,
        log_dir=run_dir, initial_severity=args.severity, device=args.device,
        print_every=args.print_every, use_wandb=args.wandb,
        checkpoint_dir=args.checkpoint_dir, checkpoint_every=args.checkpoint_every,
        hard_mission=args.hard, macro_mission=args.macro, K_ambulances=K_ambulances,
        enforce_c3=args.enforce_c3, resume=args.resume,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Off-policy sibling solver training (TD3/SAC)")
    p.add_argument("--baseline", required=True, choices=list(BASELINE_REGISTRY))
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-dir", type=str, default="logs")
    p.add_argument("--severity", type=int, default=5,
                   help="Fixed patient severity 1..5 (NON_URGENT..IMMEDIATE)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--checkpoint-dir", type=str, default=None,
                   help="Checkpoint directory. Defaults to <log-dir>/checkpoints if not set.")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--hard", action="store_true",
                   help="Use the hard-mission preset")
    p.add_argument("--macro", action="store_true",
                   help="Use W15-B2 macro mission. Defaults to K=3 unless --K is set.")
    p.add_argument("--K", dest="K_ambulances", type=int, default=None,
                   help="Number of ambulances. Defaults to 3 for --macro and 1 otherwise.")
    p.add_argument("--enforce-c3", action="store_true",
                   help="DEPRECATED — reward is always pure Phase 2.1")
    p.add_argument("--resume", action="store_true",
                   help="Auto-resume from the rolling *_latest.pt checkpoint for this seed "
                        "and continue toward --episodes (uses the per-episode auto-save).")
    args = p.parse_args(argv)
    K_ambulances = args.K_ambulances if args.K_ambulances is not None else (3 if args.macro else 1)
    run_dir = str(Path(args.log_dir) / f"{args.baseline}_K{K_ambulances}_seed{args.seed}")
    if args.checkpoint_dir is None:
        args.checkpoint_dir = str(Path(run_dir) / "checkpoints")
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            print("[warn] CUDA not available, falling back to CPU")
            args.device = "cpu"

    train(
        baseline_name=args.baseline,
        n_episodes=args.episodes,
        seed=args.seed,
        log_dir=run_dir,
        initial_severity=args.severity,
        device=args.device,
        print_every=args.print_every,
        use_wandb=args.wandb,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        hard_mission=args.hard,
        macro_mission=args.macro,
        K_ambulances=K_ambulances,
        enforce_c3=args.enforce_c3,
        resume=args.resume,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
