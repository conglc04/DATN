"""Algorithm 1 main training loop for PPO (W08).

Strict pipeline (docs/13_methodology_walkthrough.md Phase 3.4.1):
    Outer FOR ep:
        s = env.reset(); LambdaState.reset_episode(info["severity"])
        Manager FOR k in [0..M-1]:
            LambdaState.on_manager_step_start(info["severity"])    # no-op (severity fixed)
            s_H = build_manager_state(...); a_H = manager.act(s_H)
            For Worker FOR t in [0..W-1]:
                s_L = env.observe() with λ_local overlay
                a_raw, log_prob_L = worker.act(s_L)
                next_s, r_t, info = env.step(a_raw)
                r_aug = LambdaState.augmented_reward(r_t, c_vec, d_phi)
                LambdaState.accumulate(c_vec, d_phi)
                buf_L.add(...)
            buf_H.add(...)                                         # Manager rollout
            LambdaState.on_manager_step_end()                      # dual ascent + reset
        PPO update Worker (γ_L = 0.99)
        PPO update Manager (γ_H ≈ 0.904 per N1)
        Log: λ_global, λ_warm, viol_rate, losses

Implementation notes (Phase 3.4.4 N1–N9):
    N1  γ_H = γ_L^W ≈ 0.904          (sourced from utils.config.GAMMA_MANAGER)
    N3  Per-step severity threshold d_j^sev lookup from env.info["d_phi"]
    N4  λ_local exposed in Worker obs at indices [15:20] (already part of env._observe)
    N5  win_c reset after each Manager step (Option b)
    N7  Dual update order: aggregate → mean → project → push → reset
    N8  PPO buffer boundary = 1 episode
    N9  Severity sync handled in LambdaState.on_manager_step_start (sync BOTH)

Usage:
    python train.py --algo ppo --episodes 5 --seed 0 --hard
    python train.py --smoke-test --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from agents.lagrangian import LambdaState
from agents.manager_agent import (
    MANAGER_ACTION_DIM_DEFAULT,
    ManagerAgent,
    decode_manager_action,
    manager_state_dim,
)
from agents.ppo_agent import RolloutBuffer
from agents.worker_agent import WorkerAgent
from env.oran_env import EnvConfig, ORANEnv, hard_mission_config
from solvers._common import build_manager_state
from utils.config import (
    GAMMA,
    SEVERITY_OH_OBS_INDEX,
    WORKER_STEPS_PER_MANAGER,
)
from utils.early_stopping import EarlyStopping
from utils.logger import Logger
from utils.obs import overlay_lambda_local  # single-source λ overlay (used by all solvers)


# ============================================================
# Episode constants (Phase 1.4 timing — 1s episode = 10 Manager × 10 Worker)
# ============================================================

MANAGER_STEPS_PER_EPISODE: int = 10
WORKER_STEPS_PER_EPISODE: int = MANAGER_STEPS_PER_EPISODE * WORKER_STEPS_PER_MANAGER  # = 100


# ============================================================
# Worker observation overlay (Phase 3.4.4 N4)
# ============================================================
# overlay_lambda_local lives in utils/obs.py — imported above as the single
# source shared by PPO (here) and TD3/SAC (solvers/smoke_train.py).
# build_manager_state imported from solvers._common (shared with smoke_train.py).


# ============================================================
# Algorithm 1 — PPO training
# ============================================================


def train_ppo(
    n_episodes: int,
    seed: int = 0,
    log_dir: str = "logs",
    initial_severity: int = 5,
    device: str = "cpu",
    print_every: int = 1,
    use_wandb: bool = False,
    checkpoint_dir: str = "checkpoints",
    checkpoint_every: int = 500,
    hard_mission: bool = False,
    worker_ent_coef: float = 0.01,
    manager_ent_coef: float = 0.01,
    disable_warm_start: bool = False,
    early_stop: bool = False,
    early_stop_patience: int = 300,
    early_stop_min_delta: float = 10.0,
    early_stop_window: int = 100,
    early_stop_min_ep: int = 500,
    eval_at: int = 5000,
) -> dict:
    """Algorithm 1 main training loop. Returns final-episode stats."""
    # --- Setup env ---
    env_cfg = hard_mission_config() if hard_mission else EnvConfig(initial_severity=initial_severity)
    env = ORANEnv(config=env_cfg, seed=seed)
    K = env.config.K_ambulances
    F = env.config.num_streams
    state_dim_l = env.observation_space.shape[0]
    action_dim_l = env.action_space.shape[0]
    assert state_dim_l == 20 + 10 * K + F, (
        f"Worker obs dim {state_dim_l} != 20+10K+F (K={K}, F={F})"
    )
    manager_state_dim_k = manager_state_dim(K)

    # --- Setup agents (PPO + LambdaState) ---
    manager = ManagerAgent(
        state_dim=manager_state_dim_k,
        action_dim=MANAGER_ACTION_DIM_DEFAULT,
        device=device,
        seed=seed,
        ent_coef=manager_ent_coef,
    )
    worker = WorkerAgent(
        state_dim=state_dim_l,
        action_dim=action_dim_l,
        device=device,
        seed=seed,
        ent_coef=worker_ent_coef,
    )
    # Exp3 phase-transition ablation (CF-2 Audit Fix 2026-05-28):
    # disable_warm_start=True overrides default LAMBDA_WARM table with all-zero,
    # forcing cold-start dual ascent at every phase entry. Used to demonstrate
    # ≥80% reduction in time-to-reconverge claim (W12 Exp3).
    lambda_state = LambdaState(K=K, force_zero_warm=disable_warm_start)

    # --- Setup logger ---
    logger = Logger(
        run_name=f"ppo_seed{seed}",
        log_dir=log_dir,
        use_tensorboard=False,
        use_wandb=use_wandb,
    )
    logger.log_hparams({
        "algo": "ppo",
        "n_episodes": n_episodes,
        "seed": seed,
        "state_dim_l": state_dim_l,
        "action_dim_l": action_dim_l,
        "manager_state_dim": manager_state_dim_k,
        "manager_action_dim": MANAGER_ACTION_DIM_DEFAULT,
        "manager_steps_per_episode": MANAGER_STEPS_PER_EPISODE,
        "worker_steps_per_manager": WORKER_STEPS_PER_MANAGER,
    })

    # Per-episode buffers (Phase 3.4.4 N8: buffer boundary = 1 episode)
    # Worker: 100 transitions; Manager: 10 transitions.
    capacity_w = WORKER_STEPS_PER_EPISODE + 10
    capacity_h = MANAGER_STEPS_PER_EPISODE + 2

    es = EarlyStopping(
        patience=early_stop_patience,
        min_delta=early_stop_min_delta,
        window=early_stop_window,
        min_ep=early_stop_min_ep,
    ) if early_stop else None

    t_start = time.time()
    final_stats: dict = {}

    for ep in range(n_episodes):
        # ---------------- Episode reset + LambdaState sync ----------------
        obs, info = env.reset(seed=seed + ep)
        severity_init = int(info["severity"])
        severity_per_amb_init = tuple(int(s) for s in info["severity_per_amb"])
        lambda_state.reset_episode(severity_per_amb_init, severity_init)

        # Fresh PPO buffers per episode (Phase 3.4.4 N8)
        buf_w = _make_storage(capacity_w, state_dim_l, action_dim_l)
        buf_h = _make_storage(capacity_h, manager_state_dim_k, MANAGER_ACTION_DIM_DEFAULT)

        ep_reward = 0.0
        worker_step_idx = 0

        # ---------------- Manager loop (10 steps per episode) ----------------
        for _k in range(MANAGER_STEPS_PER_EPISODE):
            severity_ref_now = int(info["severity"])
            severity_per_amb_now = tuple(int(s) for s in info["severity_per_amb"])
            lambda_state.on_manager_step_start(severity_per_amb_now, severity_ref_now)   # N9: sync BOTH λ_global + λ_local

            s_H = build_manager_state(obs, lambda_state.get_lambda_global())
            a_H_raw, log_prob_H, value_H = manager.act(s_H)
            b_rrm = decode_manager_action(a_H_raw)["b_rrm"]
            env.set_rrm_budget(b_rrm)
            r_H_acc = 0.0
            intra_window_step = 0
            done_in_window = False

            # ---------------- Worker loop (W=10 steps per Manager step) ----------------
            for _t in range(WORKER_STEPS_PER_MANAGER):
                # N4: expose λ_local through Worker observation
                s_L = overlay_lambda_local(obs, lambda_state.get_lambda_local(), K)
                a_raw, log_prob_L, value_L = worker.act(s_L)

                next_obs, r_t, terminated, truncated, info = env.step(np.asarray(a_raw, dtype=np.float32))
                done = bool(terminated or truncated)
                c_vec = np.asarray(info["c_vec"], dtype=np.float64)
                d_phi = np.asarray(info["d_phi"], dtype=np.float64)

                r_aug = lambda_state.augmented_reward(float(r_t), c_vec, d_phi)
                lambda_state.accumulate(c_vec, d_phi)

                _store(buf_w, s_L, a_raw, log_prob_L, r_aug, value_L, done)

                # SMDP-discounted intra-window return: Σ γ_L^i · r_aug_i
                r_H_acc += (GAMMA ** intra_window_step) * float(r_aug)
                intra_window_step += 1
                ep_reward += float(r_aug)
                worker_step_idx += 1
                obs = next_obs

                if done:
                    done_in_window = True
                    break

            # Store Manager transition (1 per Worker window)
            _store(buf_h, s_H, a_H_raw, log_prob_H, r_H_acc, value_H, done_in_window)

            # N7: dual ascent + reset win_c (Manager step boundary)
            lambda_state.on_manager_step_end()

            if done_in_window:
                break

        # ---------------- PPO updates at episode end (N8) ----------------
        worker_metrics = _ppo_update_worker(worker, buf_w)
        manager_metrics = _ppo_update_manager(manager, buf_h)

        # Capture the active dual state before flushing severity warm starts.
        lam_g = lambda_state.get_lambda_global()
        lambda_severity_per_amb = lambda_state.sev_prev

        # Flush final active severity into lambda_warm before the next episode reload.
        severity_ref_final = int(info["severity"])
        severity_per_amb_final = tuple(int(s) for s in info["severity_per_amb"])
        lambda_state.on_episode_end(severity_per_amb_final, severity_ref_final)

        # ---------------- Logging ----------------
        metrics: dict = {
            "ep_reward": ep_reward,
            "mean_e2e_ms": env.mean_e2e_ms(),
            "viol_rate": env.episode_violation_rate(),
            "mean_embb_mbps": env.mean_embb_mbps(),
            "c3_viol_rate": env.c3_violation_rate(),
            "severity_init": severity_init,
            "severity_final": severity_ref_final,
            "lambda_severity_per_amb": str(list(lambda_severity_per_amb)),
            "worker_steps": worker_step_idx,
        }
        # λ_global layout: [C1_0..C1_{K-1}, C2_0..C2_{K-1}, C4_0..C4_{K-1}, C5_0..C5_{K-1}, C3_shared]
        for k in range(K):
            metrics[f"lambda_global_C1_{k}"] = float(lam_g[k])
            metrics[f"lambda_global_C2_{k}"] = float(lam_g[K + k])
            metrics[f"lambda_global_C4_{k}"] = float(lam_g[2 * K + k])
            metrics[f"lambda_global_C5_{k}"] = float(lam_g[3 * K + k])
        metrics["lambda_global_C3_shared"] = float(lam_g[4 * K])
        for k, v in worker_metrics.items():
            metrics[k] = v
        for k, v in manager_metrics.items():
            metrics[k] = v
        logger.log_dict(metrics, step=ep)
        final_stats = metrics

        # Periodic checkpoint
        if checkpoint_every > 0 and ((ep + 1) % checkpoint_every == 0 or ep == n_episodes - 1):
            ckpt_dir = Path(checkpoint_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            manager.save(str(ckpt_dir / f"manager_seed{seed}_ep{ep + 1}.pt"))
            worker.save(str(ckpt_dir / f"worker_seed{seed}_ep{ep + 1}.pt"))

        if (ep + 1) % print_every == 0 or ep == n_episodes - 1:
            elapsed = time.time() - t_start
            es_tag = f"  no_improve={es._no_improve_eps}" if es else ""
            print(
                f"[ppo] ep {ep + 1}/{n_episodes}  "
                f"R={ep_reward:+.2f}  e2e={metrics['mean_e2e_ms']:.3f}ms  "
                f"viol={metrics['viol_rate']:.4f}  "
                f"lambda_mean={float(np.mean(lam_g)):.3f}"
                f"{es_tag}  ({elapsed:.1f}s)"
            )

        # Eval checkpoint + early stopping
        if es is not None:
            es.maybe_save_eval(ep, metrics, log_dir, eval_at=eval_at,
                               run_name=f"ppo_seed{seed}")
            if es.step(ep, ep_reward):
                print(
                    f"[ppo] EARLY STOP at ep {ep + 1}  "
                    f"rolling_mean={es.rolling_mean:+.2f}  "
                    f"no_improve={es._no_improve_eps} >= patience={es.patience}"
                )
                break
        elif eval_at > 0 and (ep + 1) == eval_at:
            _save_eval_snapshot(ep, metrics, log_dir, f"ppo_seed{seed}", eval_at)

    # Save summary
    summary = dict(final_stats)
    summary["algo"] = "ppo"
    summary["seed"] = seed
    summary["n_episodes"] = n_episodes
    summary["final_lambda_warm"] = {
        str(k): v.tolist() for k, v in lambda_state.get_lambda_warm_table_snapshot().items()
    }
    summary_path = Path(log_dir) / f"summary_ppo_seed{seed}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(f"[ppo] DONE — summary saved to {summary_path}")

    logger.close()
    env.close()
    return summary


# ============================================================
# Eval snapshot helper (no early stopping — milestone only)
# ============================================================


def _save_eval_snapshot(
    ep: int,
    metrics: dict,
    log_dir: str,
    run_name: str,
    eval_at: int,
) -> None:
    import time as _time
    snap = {"eval_ep": ep + 1, "timestamp": _time.strftime("%Y-%m-%d %H:%M:%S")}
    snap.update(metrics)
    out = Path(log_dir) / f"eval_ep{eval_at}_{run_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(__import__("json").dumps(snap, indent=2, default=float), encoding="utf-8")
    print(f"[eval checkpoint] ep={ep + 1}  -> {out.name}")


# ============================================================
# Storage helpers (numpy-backed mini-buffer)
# ============================================================


def _make_storage(capacity: int, state_dim: int, action_dim: int) -> dict:
    return {
        "obs": np.zeros((capacity, state_dim), dtype=np.float32),
        "actions": np.zeros((capacity, action_dim), dtype=np.float32),
        "log_probs": np.zeros(capacity, dtype=np.float32),
        "rewards": np.zeros(capacity, dtype=np.float32),
        "values": np.zeros(capacity, dtype=np.float32),
        "dones": np.zeros(capacity, dtype=np.float32),
        "ptr": 0,
        "capacity": capacity,
    }


def _store(buf, obs, action, log_prob, reward, value, done) -> None:
    i = buf["ptr"]
    if i >= buf["capacity"]:
        return
    buf["obs"][i] = obs
    buf["actions"][i] = action
    buf["log_probs"][i] = log_prob
    buf["rewards"][i] = reward
    buf["values"][i] = value
    buf["dones"][i] = float(done)
    buf["ptr"] = i + 1


def _slice(buf) -> dict:
    n = buf["ptr"]
    return {k: v[:n] for k, v in buf.items() if isinstance(v, np.ndarray)}


def _ppo_update_worker(worker, buf) -> dict[str, float]:
    if buf["ptr"] == 0:
        return {"worker_n_samples": 0}
    sl = _slice(buf)
    return worker.update(
        obs=sl["obs"],
        actions_raw=sl["actions"],
        old_log_probs=sl["log_probs"],
        rewards=sl["rewards"],
        values=sl["values"],
        dones=sl["dones"],
        last_value=0.0,
    )


def _ppo_update_manager(manager, buf) -> dict[str, float]:
    if buf["ptr"] == 0:
        return {"manager_n_samples": 0}
    sl = _slice(buf)
    return manager.update(
        obs=sl["obs"],
        actions=sl["actions"],
        old_log_probs=sl["log_probs"],
        rewards=sl["rewards"],
        values=sl["values"],
        dones=sl["dones"],
        last_value=0.0,
    )


# ============================================================
# Smoke test (Week 1 stub — preserved)
# ============================================================


def smoke_test(args: argparse.Namespace) -> int:
    print(f"[smoke-test] algo={args.algo} seed={args.seed}")
    try:
        from utils.config import P_TOTAL, SEVERITY_QOS
    except ImportError as exc:
        print(f"[smoke-test] FAILED — utils import error: {exc}", file=sys.stderr)
        return 1
    assert P_TOTAL == 273, f"P_TOTAL mismatch: {P_TOTAL} != 273"
    assert SEVERITY_QOS[5]["D_max"] == 1e-3, "SEVERITY_QOS[5].D_max mismatch"
    logger = Logger(
        run_name=f"smoke_{args.algo}_seed{args.seed}",
        log_dir=str(args.log_dir),
        use_tensorboard=False,
        use_wandb=False,
    )
    for step in range(10):
        logger.log_scalar("smoke/step", float(step), step)
    logger.close()
    print("[smoke-test] Stub: 10 steps OK")
    return 0


# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO training entry point")
    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        choices=["ppo", "td3", "sac", "static_slicing",
                 "b2_hrl_ppo_soft", "pa_ppo_soft", "no_phase_ppo", "ppo_cmdp_flat"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--resume-checkpoint", type=str, default=None,
                        help="Path to .pt checkpoint to resume from (solvers only)")
    parser.add_argument("--resume-start-ep", type=int, default=0,
                        help="Episode offset when resuming (metrics appended after this ep)")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--hard", action="store_true", help="Use hard-mission preset")
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--severity", type=int, default=5,
                        help="Fixed patient severity 1..5 (NON_URGENT..IMMEDIATE) for the run")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--worker-ent-coef", type=float, default=0.01,
        help="Worker actor entropy bonus coefficient (default 0.01; "
             "sweep ∈ {0.01, 0.03, 0.05, 0.1} per W10 Track A)",
    )
    parser.add_argument(
        "--manager-ent-coef", type=float, default=0.01,
        help="Manager actor entropy bonus coefficient",
    )
    parser.add_argument(
        "--no-warm-start", action="store_true",
        help="Disable LAMBDA_WARM table (cold-start dual ascent at every severity).",
    )
    # Early stopping
    parser.add_argument("--early-stop", action="store_true",
                        help="Enable early stopping when reward plateaus.")
    parser.add_argument("--early-stop-patience", type=int, default=300,
                        help="Episodes without improvement before stopping (default: 300).")
    parser.add_argument("--early-stop-min-delta", type=float, default=10.0,
                        help="Minimum reward improvement to reset patience (default: 10.0).")
    parser.add_argument("--early-stop-window", type=int, default=100,
                        help="Rolling window size for mean reward (default: 100).")
    parser.add_argument("--early-stop-min-ep", type=int, default=500,
                        help="Minimum episodes before early stopping can fire (default: 500).")
    parser.add_argument("--eval-at", type=int, default=5000,
                        help="Save eval snapshot at this episode milestone (default: 5000).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.smoke_test:
        return smoke_test(args)

    if args.algo == "ppo":
        train_ppo(
            n_episodes=args.episodes,
            seed=args.seed,
            log_dir=str(args.log_dir),
            initial_severity=args.severity,
            device=args.device,
            print_every=args.print_every,
            use_wandb=args.wandb,
            checkpoint_dir=str(args.checkpoint_dir),
            checkpoint_every=args.checkpoint_every,
            hard_mission=args.hard,
            worker_ent_coef=args.worker_ent_coef,
            manager_ent_coef=args.manager_ent_coef,
            disable_warm_start=args.no_warm_start,
            early_stop=args.early_stop,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_delta=args.early_stop_min_delta,
            early_stop_window=args.early_stop_window,
            early_stop_min_ep=args.early_stop_min_ep,
            eval_at=args.eval_at,
        )
        return 0

    # Baselines / ablations delegate to solvers.smoke_train
    from solvers.smoke_train import train as bl_train
    bl_train(
        baseline_name=args.algo,
        n_episodes=args.episodes,
        seed=args.seed,
        log_dir=str(args.log_dir),
        initial_severity=args.severity,
        device=args.device,
        print_every=args.print_every,
        use_wandb=args.wandb,
        checkpoint_dir=str(args.checkpoint_dir),
        checkpoint_every=args.checkpoint_every,
        hard_mission=args.hard,
        early_stop=args.early_stop,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        early_stop_window=args.early_stop_window,
        early_stop_min_ep=args.early_stop_min_ep,
        eval_at=args.eval_at,
        resume_checkpoint=args.resume_checkpoint,
        resume_start_ep=args.resume_start_ep,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
