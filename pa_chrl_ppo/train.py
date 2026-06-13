"""Algorithm 1 main training loop for PA-CHRL-PPO (W08).

Strict pipeline (docs/13_methodology_walkthrough.md Phase 3.4.1):
    Outer FOR ep:
        s = env.reset(); LambdaState.reset_episode(env.phase)
        Manager FOR k in [0..M-1]:
            LambdaState.on_manager_step_start(env.phase)          # Fix Error 1
            s_H = build_manager_state(...); a_H = manager.act(s_H)
            For Worker FOR t in [0..W-1]:
                s_L = env.observe() with λ_local overlay
                a_raw, log_prob_L = worker.act(s_L)
                a_safe = nsf.forward(s_L, a_raw)
                next_s, r_t, info = env.step(a_safe)
                r_aug = LambdaState.augmented_reward(r_t, c_vec, d_phi)
                LambdaState.accumulate(c_vec, d_phi)
                buf_L.add(...)
            buf_H.add(...)                                         # Manager rollout
            LambdaState.on_manager_step_end()                      # dual ascent + reset
        PPO update Worker (γ_L = 0.99) + β_qp distillation
        PPO update Manager (γ_H ≈ 0.904 per N1)
        Log: λ_global, λ_warm, viol_rate, losses

Implementation notes (Phase 3.4.4 N1–N9):
    N1  γ_H = γ_L^W ≈ 0.904          (sourced from utils.config.GAMMA_MANAGER)
    N2  a_safe.detach() inside Worker update qp distillation
    N3  Per-step phase threshold d_j^φ_t lookup from env.info["d_phi"]
    N4  λ_local exposed in Worker obs at indices [17:22] (already part of env._observe)
    N5  win_c reset after each Manager step (Option b)
    N6  β_qp anneal per-episode (not per-step)
    N7  Dual update order: aggregate → mean → project → push → reset
    N8  PPO buffer boundary = 1 episode
    N9  Phase transition handled in LambdaState.on_manager_step_start (sync BOTH)

Usage:
    python train.py --algo pa_chrl_ppo --episodes 5 --seed 0 --hard
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
    MANAGER_STATE_DIM_DEFAULT,
    ManagerAgent,
)
from agents.nsf import IdentityNSF
from agents.ppo_agent import RolloutBuffer
from agents.worker_agent import WORKER_STATE_DIM_DEFAULT, WorkerAgent
from env.oran_env import EnvConfig, ORANEnv, hard_mission_config
from utils.config import (
    BETA_QP_FINAL,
    BETA_QP_FLOOR,
    BETA_QP_INIT,
    BETA_QP_T_ANNEAL,
    LAMBDA_LOCAL_OBS_INDEX,
    PHASE_OH_OBS_INDEX,
    WORKER_STEPS_PER_MANAGER,
)
from utils.early_stopping import EarlyStopping
from utils.logger import Logger


# ============================================================
# Episode constants (Phase 1.4 timing — 1s episode = 10 Manager × 10 Worker)
# ============================================================

MANAGER_STEPS_PER_EPISODE: int = 10
WORKER_STEPS_PER_EPISODE: int = MANAGER_STEPS_PER_EPISODE * WORKER_STEPS_PER_MANAGER  # = 100


# ============================================================
# β_qp anneal (Phase 3.2.2)
# ============================================================


def anneal_beta_qp(
    episode: int,
    beta_init: float = BETA_QP_INIT,
    beta_final: float = BETA_QP_FINAL,
    t_anneal: int = BETA_QP_T_ANNEAL,
) -> float:
    """Linear anneal β_qp from β_init → β_final over t_anneal episodes.

    Reviewer Mn1 (Gemini W08, 2026-05-27): clamp at BETA_QP_FLOOR to prevent
    catastrophic forgetting of NSF/QP safety boundaries at end-of-training.
    Empirical: β_qp → 0 would let PPO drift away from QP imitation → violations
    under out-of-distribution conditions (Exp11 sensor failure scenarios).
    """
    if t_anneal <= 0:
        return max(BETA_QP_FLOOR, beta_final)
    frac = min(float(episode) / float(t_anneal), 1.0)
    raw = float(beta_init + (beta_final - beta_init) * frac)
    return max(BETA_QP_FLOOR, raw)


# ============================================================
# Manager state construction
# ============================================================


def build_manager_state(
    worker_obs: np.ndarray,
    lambda_global: np.ndarray,
) -> np.ndarray:
    """Construct 11-dim Manager state s_H from current Worker obs + λ_global.

    Layout (Phase 3.3.1 W08 placeholder — finalize multi-cell aggregation in W11+):
        [0:2]   ρ_urllc, ρ_eMBB              (from worker [0:2])
        [2]     mean BLER                    (from worker [9])
        [3]     phase index (normalized)     (argmax worker [10:15] / 5)
        [4:6]   aoi mean, aoi max            (from worker [25:27])
        [6:11]  λ_global per-constraint (5)  (from LambdaState)
    """
    rho_urllc = float(worker_obs[0])
    rho_emBB = float(worker_obs[1])
    bler = float(worker_obs[9])
    phase_oh = worker_obs[PHASE_OH_OBS_INDEX : PHASE_OH_OBS_INDEX + 5]
    phase_idx = float((np.argmax(phase_oh) + 1) / 5.0)
    aoi_mean = float(worker_obs[25])
    aoi_max = float(worker_obs[26])
    s_H = np.concatenate(
        [
            np.array([rho_urllc, rho_emBB, bler, phase_idx, aoi_mean, aoi_max], dtype=np.float32),
            np.asarray(lambda_global, dtype=np.float32),
        ]
    )
    return s_H.astype(np.float32)


# ============================================================
# Worker observation overlay (Phase 3.4.4 N4)
# ============================================================


def overlay_lambda_local(obs: np.ndarray, lambda_local: np.ndarray) -> np.ndarray:
    """Overwrite worker obs[17:22] with current λ_local (5-dim)."""
    out = obs.astype(np.float32, copy=True)
    out[LAMBDA_LOCAL_OBS_INDEX : LAMBDA_LOCAL_OBS_INDEX + 5] = lambda_local.astype(np.float32)
    return out


# ============================================================
# Algorithm 1 — PA-CHRL-PPO training
# ============================================================


def train_pa_chrl_ppo(
    n_episodes: int,
    seed: int = 0,
    log_dir: str = "logs",
    initial_phase: int = 3,
    device: str = "cpu",
    print_every: int = 1,
    use_wandb: bool = False,
    checkpoint_dir: str = "checkpoints",
    checkpoint_every: int = 500,
    hard_mission: bool = False,
    beta_qp_init: float = BETA_QP_INIT,
    beta_qp_final: float = BETA_QP_FINAL,
    beta_qp_t_anneal: int = BETA_QP_T_ANNEAL,
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
    env_cfg = hard_mission_config() if hard_mission else EnvConfig(initial_phase=initial_phase)
    env = ORANEnv(config=env_cfg, seed=seed)
    state_dim_l = env.observation_space.shape[0]
    action_dim_l = env.action_space.shape[0]
    assert state_dim_l == WORKER_STATE_DIM_DEFAULT, (
        f"Worker obs dim {state_dim_l} != {WORKER_STATE_DIM_DEFAULT}"
    )

    # --- Setup agents (PA-CHRL-PPO + IdentityNSF + LambdaState) ---
    manager = ManagerAgent(
        state_dim=MANAGER_STATE_DIM_DEFAULT,
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
    nsf = IdentityNSF()
    # Exp3 phase-transition ablation (CF-2 Audit Fix 2026-05-28):
    # disable_warm_start=True overrides default LAMBDA_WARM table with all-zero,
    # forcing cold-start dual ascent at every phase entry. Used to demonstrate
    # ≥80% reduction in time-to-reconverge claim (W12 Exp3).
    if disable_warm_start:
        zero_warm = {phi: np.zeros(5, dtype=np.float64) for phi in range(1, 6)}
        lambda_state = LambdaState(lambda_warm=zero_warm)
    else:
        lambda_state = LambdaState()

    # --- Setup logger ---
    logger = Logger(
        run_name=f"pa_chrl_ppo_seed{seed}",
        log_dir=log_dir,
        use_tensorboard=False,
        use_wandb=use_wandb,
    )
    logger.log_hparams({
        "algo": "pa_chrl_ppo",
        "n_episodes": n_episodes,
        "seed": seed,
        "state_dim_l": state_dim_l,
        "action_dim_l": action_dim_l,
        "manager_state_dim": MANAGER_STATE_DIM_DEFAULT,
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
        phase_init = int(info["phase_now"])
        lambda_state.reset_episode(phase_init)

        # Fresh PPO buffers per episode (Phase 3.4.4 N8)
        buf_w = _make_storage(capacity_w, state_dim_l, action_dim_l)
        buf_h = _make_storage(capacity_h, MANAGER_STATE_DIM_DEFAULT, MANAGER_ACTION_DIM_DEFAULT)
        beta_qp = anneal_beta_qp(ep, beta_qp_init, beta_qp_final, beta_qp_t_anneal)

        ep_reward = 0.0
        worker_step_idx = 0

        # ---------------- Manager loop (10 steps per episode) ----------------
        for _k in range(MANAGER_STEPS_PER_EPISODE):
            phi_now = int(info["phase_now"])
            lambda_state.on_manager_step_start(phi_now)   # N9: sync BOTH λ_global + λ_local

            s_H = build_manager_state(obs, lambda_state.get_lambda_global())
            a_H_raw, log_prob_H, value_H = manager.act(s_H)
            r_H_acc = 0.0
            done_in_window = False

            # ---------------- Worker loop (W=10 steps per Manager step) ----------------
            for _t in range(WORKER_STEPS_PER_MANAGER):
                # N4: expose λ_local through Worker observation
                s_L = overlay_lambda_local(obs, lambda_state.get_lambda_local())
                a_raw, log_prob_L, value_L = worker.act(s_L)
                a_safe = nsf.forward(s_L, a_raw)

                next_obs, r_t, terminated, truncated, info = env.step(np.asarray(a_safe, dtype=np.float32))
                done = bool(terminated or truncated)
                c_vec = np.asarray(info["c_vec"], dtype=np.float64)
                d_phi = np.asarray(info["d_phi"], dtype=np.float64)

                r_aug = lambda_state.augmented_reward(float(r_t), c_vec, d_phi)
                lambda_state.accumulate(c_vec, d_phi)

                _store(buf_w, s_L, a_raw, log_prob_L, r_aug, value_L, done)

                r_H_acc += float(r_t)
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
        worker_metrics = _ppo_update_worker(worker, buf_w, beta_qp=beta_qp)
        manager_metrics = _ppo_update_manager(manager, buf_h)

        # Capture the active dual state before flushing final-phase warm starts.
        lam_g = lambda_state.get_lambda_global()
        lambda_phase = int(lambda_state.phi_prev)

        # Flush final active phase into lambda_warm before the next episode reload.
        lambda_state.on_episode_end(final_phase=int(info["phase_now"]))

        # ---------------- Logging ----------------
        metrics: dict = {
            "ep_reward": ep_reward,
            "mean_e2e_ms": env.mean_e2e_ms(),
            "viol_rate": env.episode_violation_rate(),
            "mean_embb_mbps": env.mean_embb_mbps(),
            "c3_viol_rate": env.c3_violation_rate(),
            "beta_qp": beta_qp,
            "phase_init": phase_init,
            "phase_final": int(info["phase_now"]),
            "lambda_phase": lambda_phase,
            "worker_steps": worker_step_idx,
        }
        for j in range(5):
            metrics[f"lambda_global_{j + 1}"] = float(lam_g[j])
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
                f"[pa_chrl_ppo] ep {ep + 1}/{n_episodes}  "
                f"R={ep_reward:+.2f}  e2e={metrics['mean_e2e_ms']:.3f}ms  "
                f"viol={metrics['viol_rate']:.4f}  "
                f"lambda_mean={float(np.mean(lam_g)):.3f}  beta_qp={beta_qp:.4f}"
                f"{es_tag}  ({elapsed:.1f}s)"
            )

        # Eval checkpoint + early stopping
        if es is not None:
            es.maybe_save_eval(ep, metrics, log_dir, eval_at=eval_at,
                               run_name=f"pa_chrl_ppo_seed{seed}")
            if es.step(ep, ep_reward):
                print(
                    f"[pa_chrl_ppo] EARLY STOP at ep {ep + 1}  "
                    f"rolling_mean={es.rolling_mean:+.2f}  "
                    f"no_improve={es._no_improve_eps} >= patience={es.patience}"
                )
                break
        elif eval_at > 0 and (ep + 1) == eval_at:
            _save_eval_snapshot(ep, metrics, log_dir, f"pa_chrl_ppo_seed{seed}", eval_at)

    # Save summary
    summary = dict(final_stats)
    summary["algo"] = "pa_chrl_ppo"
    summary["seed"] = seed
    summary["n_episodes"] = n_episodes
    summary["final_lambda_warm"] = {
        int(k): v.tolist() for k, v in lambda_state.get_lambda_warm_table_snapshot().items()
    }
    summary_path = Path(log_dir) / f"summary_pa_chrl_ppo_seed{seed}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(f"[pa_chrl_ppo] DONE — summary saved to {summary_path}")

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


def _ppo_update_worker(worker, buf, beta_qp: float) -> dict[str, float]:
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
        actions_safe=sl["actions"],   # IdentityNSF: a_safe == a_raw
        beta_qp=beta_qp,
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
        from utils.config import P_TOTAL, PHASE_QOS
    except ImportError as exc:
        print(f"[smoke-test] FAILED — utils import error: {exc}", file=sys.stderr)
        return 1
    assert P_TOTAL == 273, f"P_TOTAL mismatch: {P_TOTAL} != 273"
    assert PHASE_QOS[3]["D_max"] == 1e-3, "PHASE_QOS[3].D_max mismatch"
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
    parser = argparse.ArgumentParser(description="PA-CHRL-PPO training entry point")
    parser.add_argument(
        "--algo",
        type=str,
        default="pa_chrl_ppo",
        choices=["pa_chrl_ppo", "td3_lag", "sac_lag", "static_slicing",
                 "b2_hrl_ppo_soft", "pa_ppo_soft", "no_phase_chrl_ppo", "ppo_cmdp_flat"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--resume-checkpoint", type=str, default=None,
                        help="Path to .pt checkpoint to resume from (baselines only)")
    parser.add_argument("--resume-start-ep", type=int, default=0,
                        help="Episode offset when resuming (metrics appended after this ep)")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--hard", action="store_true", help="Use hard-mission preset")
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--phase", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--beta-qp-init", type=float, default=BETA_QP_INIT,
        help="β_qp anneal start value (Phase 3.2.2; use 0 for IdentityNSF diagnostic)",
    )
    parser.add_argument(
        "--beta-qp-final", type=float, default=BETA_QP_FINAL,
        help="β_qp anneal endpoint value",
    )
    parser.add_argument(
        "--beta-qp-t-anneal", type=int, default=BETA_QP_T_ANNEAL,
        help="β_qp linear anneal horizon (episodes)",
    )
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
        help="Disable LAMBDA_WARM table (cold-start dual ascent at every phase). "
             "Used for Exp3 phase-transition ablation (CF-2 audit fix 2026-05-28).",
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

    if args.algo == "pa_chrl_ppo":
        train_pa_chrl_ppo(
            n_episodes=args.episodes,
            seed=args.seed,
            log_dir=str(args.log_dir),
            initial_phase=args.phase,
            device=args.device,
            print_every=args.print_every,
            use_wandb=args.wandb,
            checkpoint_dir=str(args.checkpoint_dir),
            checkpoint_every=args.checkpoint_every,
            hard_mission=args.hard,
            beta_qp_init=args.beta_qp_init,
            beta_qp_final=args.beta_qp_final,
            beta_qp_t_anneal=args.beta_qp_t_anneal,
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

    # Baselines / ablations delegate to baselines.smoke_train
    from baselines.smoke_train import train as bl_train
    bl_train(
        baseline_name=args.algo,
        n_episodes=args.episodes,
        seed=args.seed,
        log_dir=str(args.log_dir),
        initial_phase=args.phase,
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
