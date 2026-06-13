"""Smoke training driver for solvers + Phase 3 sibling solvers.

W07 refactor: dispatches between **NEW** LambdaState API (5-dim λ, used by
TD3 + SAC as Phase 3 siblings to PPO) and the **OLD**
CMDPLagrangian API (2-dim λ, kept for ablation variants like b2_hrl_ppo_soft,
pa_ppo_soft, ppo_cmdp_flat that are NOT Phase 3 siblings — only used in Exp6).

Usage:
    python -m solvers.smoke_train --baseline static_slicing --episodes 100
    python -m solvers.smoke_train --baseline sac --episodes 100 --hard
    python -m solvers.smoke_train --baseline td3 --episodes 100 --hard
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from agents.ppo_agent import RolloutBuffer
from env.oran_env import EnvConfig, ORANEnv
from utils.config import WORKER_STEPS_PER_MANAGER
from utils.early_stopping import EarlyStopping
from utils.logger import Logger


BASELINE_REGISTRY = {
    "td3":            "solvers.td3:TD3Baseline",
    "sac":            "solvers.sac:SACBaseline",
}


def make_baseline(name: str, state_dim: int, action_dim: int, seed: int, device: str = "cpu"):
    if name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline: {name}. Choose from {list(BASELINE_REGISTRY)}")
    mod_path, cls_name = BASELINE_REGISTRY[name].split(":")
    import importlib
    mod = importlib.import_module(mod_path)
    cls = getattr(mod, cls_name)
    return cls(state_dim=state_dim, action_dim=action_dim, seed=seed, device=device)


def train(
    baseline_name: str,
    n_episodes: int,
    seed: int = 0,
    log_dir: str = "logs",
    initial_phase: int = 3,
    urllc_lambda: float = 50.0,
    M_eMBB: int = 30,
    device: str = "cpu",
    print_every: int = 10,
    use_wandb: bool = False,
    checkpoint_dir: str = "checkpoints",
    checkpoint_every: int = 500,
    hard_mission: bool = False,
    enforce_c3: bool = False,  # DEPRECATED — env reward is always pure Phase 2.1
    early_stop: bool = False,
    early_stop_patience: int = 300,
    early_stop_min_delta: float = 10.0,
    early_stop_window: int = 100,
    early_stop_min_ep: int = 500,
    eval_at: int = 5000,
    resume_checkpoint: str | None = None,  # path to .pt file to resume from
    resume_start_ep: int = 0,              # episode offset (metrics append after this ep)
) -> dict:
    if enforce_c3:
        import warnings
        warnings.warn(
            "enforce_c3=True is deprecated. Reward is always pure Phase 2.1; "
            "C3 handled via Lagrangian λ_3.", DeprecationWarning, stacklevel=2,
        )
    if hard_mission:
        from env.oran_env import hard_mission_config
        env_cfg = hard_mission_config()
    else:
        env_cfg = EnvConfig(
            initial_phase=initial_phase,
            urllc_arrival_rate=urllc_lambda,
            M_eMBB=M_eMBB,
        )
    env = ORANEnv(config=env_cfg, seed=seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = make_baseline(baseline_name, state_dim, action_dim, seed, device=device)

    # Resume from checkpoint if provided
    if resume_checkpoint is not None:
        agent.load(resume_checkpoint)
        print(f"[{baseline_name}] Resumed from {resume_checkpoint} (start_ep={resume_start_ep})")

    logger = Logger(
        run_name=f"smoke_{baseline_name}_seed{seed}",
        log_dir=log_dir,
        use_tensorboard=False,
        use_wandb=use_wandb,
        append_csv=(resume_checkpoint is not None),
    )
    logger.log_hparams({
        "baseline": baseline_name,
        "n_episodes": n_episodes,
        "seed": seed,
        "state_dim": state_dim,
        "initial_phase": initial_phase,
    })

    # Detect API surface
    has_lambda_state = hasattr(agent, "lambda_state")     # NEW 5-dim (TD3 + SAC siblings)
    has_old_lagrangian = hasattr(agent, "lagrangian")     # OLD 2-dim (ablation only)
    is_off_policy = hasattr(agent, "store_transition")
    buffer = None
    if hasattr(agent, "ppo") and not is_off_policy:
        buffer = RolloutBuffer(capacity=2010, state_dim=state_dim, action_dim=action_dim)

    es = EarlyStopping(
        patience=early_stop_patience,
        min_delta=early_stop_min_delta,
        window=early_stop_window,
        min_ep=early_stop_min_ep,
    ) if early_stop else None

    t_start = time.time()
    episode_rewards = []
    final_stats: dict = {}

    for ep in range(n_episodes):
        global_ep = ep + resume_start_ep   # absolute episode index for logging
        obs, info = env.reset(seed=seed + global_ep)
        if buffer is not None:
            buffer.reset()

        # ---- NEW LambdaState lifecycle: sync λ_global + λ_local from λ_warm[φ_init] ----
        if has_lambda_state:
            agent.on_episode_start(int(info["phase_now"]))
            agent.on_manager_step_start(int(info["phase_now"]))

        ep_reward = 0.0
        worker_step_idx = 0
        terminated = truncated = False
        while not (terminated or truncated):
            obs_masked = agent.maybe_mask(obs)
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            # === Augmented reward + λ accumulation ===
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
                        aoi_samples=None, phase=initial_phase,
                    )
                    aug = agent.augment_reward(float(reward), cons)
            else:
                aug = (
                    agent.augment_reward(float(reward), env.e2e_history[-20:])
                    if hasattr(agent, "augment_reward")
                    else float(reward)
                )

            # === Store transition / sample (PPO buffer vs TD3 off-policy) ===
            done = bool(terminated or truncated)
            if buffer is not None:
                buffer.add(
                    obs_masked.astype(np.float32), action.astype(np.float32),
                    log_prob, aug, value, done,
                )
            elif is_off_policy:
                agent.store_transition(obs, action.astype(np.float32), aug, next_obs, done)
                agent.update()

            obs = next_obs
            ep_reward += float(aug)
            worker_step_idx += 1

            # === Manager step boundary: dual ascent + phase resync ===
            if has_lambda_state and worker_step_idx % WORKER_STEPS_PER_MANAGER == 0:
                agent.on_manager_step_end()
                if not done:
                    agent.on_manager_step_start(int(info["phase_now"]))

            if buffer is not None and buffer.full:
                buffer.compute_gae(last_value=0.0)
                agent.update(buffer)
                buffer.reset()

        # End of episode — flush remaining buffer
        if buffer is not None and buffer.ptr > 0:
            buffer.compute_gae(last_value=0.0)
            agent.update(buffer)
            buffer.reset()

        # Final partial Manager window (only fires if last episode step wasn't
        # already on a Manager boundary)
        if has_lambda_state and worker_step_idx % WORKER_STEPS_PER_MANAGER != 0:
            agent.on_manager_step_end()
        if has_lambda_state:
            lam_for_log = agent.lambda_state.get_lambda_global()
            lambda_phase_for_log = int(agent.lambda_state.phi_prev)
            agent.lambda_state.on_episode_end(final_phase=int(info["phase_now"]))

        # OLD API: dual-ascent at end of episode (NOT per Manager step)
        if has_old_lagrangian and not has_lambda_state:
            from solvers._common import estimate_constraints
            cons_ep = estimate_constraints(
                env.e2e_history, embb_mbps=20.0,
                aoi_samples=None, phase=initial_phase,
            )
            if hasattr(agent, "compute_constraints"):
                cons_ep = agent.compute_constraints(env.e2e_history)
            agent.update_lambdas(cons_ep)

        mean_e2e_ms = env.mean_e2e_ms()
        viol = env.episode_violation_rate()
        episode_rewards.append(ep_reward)
        metrics: dict = {
            "ep_reward": ep_reward,
            "mean_e2e_ms": mean_e2e_ms,
            "viol_rate": viol,
            "mean_embb_mbps": env.mean_embb_mbps(),
            "c3_viol_rate": env.c3_violation_rate(),
        }
        if has_lambda_state:
            lam = lam_for_log
            metrics["lambda_phase"] = lambda_phase_for_log
            for j in range(5):
                metrics[f"lambda_global_{j + 1}"] = float(lam[j])
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
        }

        if checkpoint_every > 0 and ((ep + 1) % checkpoint_every == 0 or ep == n_episodes - 1):
            ckpt_path = Path(checkpoint_dir) / f"{baseline_name}_seed{seed}_ep{global_ep + 1}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                agent.save(str(ckpt_path))
            except Exception as exc:
                print(f"[checkpoint] {baseline_name} save skipped: {exc}")

        if (ep + 1) % print_every == 0 or ep == n_episodes - 1:
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
            int(k): v.tolist() for k, v in agent.lambda_state.get_lambda_warm_table_snapshot().items()
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Smoke training for solvers")
    p.add_argument("--baseline", required=True, choices=list(BASELINE_REGISTRY))
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-dir", type=str, default="logs")
    p.add_argument("--phase", type=int, default=3)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--hard", action="store_true",
                   help="Use the hard-mission preset")
    p.add_argument("--enforce-c3", action="store_true",
                   help="DEPRECATED — reward is always pure Phase 2.1")
    args = p.parse_args(argv)

    train(
        baseline_name=args.baseline,
        n_episodes=args.episodes,
        seed=args.seed,
        log_dir=args.log_dir,
        initial_phase=args.phase,
        device=args.device,
        print_every=args.print_every,
        use_wandb=args.wandb,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        hard_mission=args.hard,
        enforce_c3=args.enforce_c3,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
