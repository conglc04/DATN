"""Unified training entry for the 3 SIBLING solvers (PPO / TD3 / SAC).

`main()` dispatches by --algo: PPO (on-policy) runs here in `train_ppo`; TD3/SAC
(off-policy) run in `solvers/train_offpolicy.py`. All three solve the SAME problem
with the SAME HRL framework — the only difference is the RL core. Shared (solver-
agnostic) pieces, so the comparison is fair (no problem built for any one algorithm):
    - Env / CMDP: reward `α_e·log(1+R_eMBB/R_REF)`, constraints c_vec/d_phi, and the
      pure-RL softmax PRB split all live in `env/oran_env.py` (identical for every solver).
    - HRL Manager+Worker: PPO→ManagerAgent, TD3→TD3ManagerAgent, SAC→SACManagerAgent.
    - `LambdaState` (4K+1)-dim dual ascent, `overlay_lambda_local` (utils/obs.py),
      `build_manager_state`/`decode_manager_action`/`env.set_rrm_budget` (shared).
    - Episode = ONE FULL MISSION: env runs until terminated (all arrived) or
      truncated (episode_duration_sec). Identical for all 3 (see test_solver_parity).

PPO pipeline (on-policy, docs/13 §3.2):
    FOR ep:                                  # ep = 1 full mission
        env.reset(); LambdaState.reset_episode(severity_per_amb)   # severity fixed/episode
        WHILE not done:                      # env PERSISTS across rollouts
            collect 1 rollout = MANAGER_STEPS_PER_ROLLOUT windows (= 100 Worker steps):
                Manager: s_H=build_manager_state; a_H=manager.act; env.set_rrm_budget
                Worker ×W: s_L=overlay λ_local; a=worker.act; env.step; r_aug; accumulate
                on_manager_step_end()                         # dual ascent
            GAE bootstrap V(s) if not terminal, else 0
            PPO update Worker (γ_L=0.99) + Manager (γ_H≈0.904)  # per rollout
        on_episode_end()                     # flush λ_warm

Implementation notes (Phase 3.4.4 N1–N9):
    N1  γ_H = γ_L^W ≈ 0.904          (sourced from utils.config.GAMMA_MANAGER)
    N3  Per-step severity threshold d_j^sev lookup from env.info["d_phi"]
    N4  λ_local exposed in Worker obs at indices [15:20] (already part of env._observe)
    N5  win_c reset after each Manager step (Option b)
    N7  Dual update order: aggregate → mean → project → push → reset
    N8  PPO buffer boundary = 1 rollout (100 Worker steps); episode = mission until done
    N9  Severity sync handled in LambdaState.on_manager_step_start (sync BOTH)

Usage:
    python train.py --algo ppo --episodes 5 --seed 0 --hard
    python train.py --algo td3 --macro --episodes 1000    # off-policy sibling, full SUMO mission
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
from env.oran_env import EnvConfig, ORANEnv, hard_mission_config, macro_mission_config
from solvers._common import build_manager_state, value_bootstrap_is_terminal
from utils.config import (
    GAMMA,
    MINIBATCH_SIZE,
    OBS_FIXED_BLOCK_LEN,
    OBS_PER_AMB_BLOCK_LEN,
    REWARD_FIXED_SCALE,
    SEVERITY_OH_OBS_INDEX,
    SEVERITY_QOS,
    WORKER_STEPS_PER_MANAGER,
)
from utils.checkpointing import (
    latest_ckpt_path,
    load_train_state,
    save_train_state,
    state_path,
)
from utils.early_stopping import EarlyStopping
from utils.logger import Logger
from utils.obs import overlay_lambda_local  # single-source λ overlay (used by all solvers)


def _git_commit_short() -> str:
    """Return short git commit hash, or 'unknown' if not in a git repo."""
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _config_hash() -> str:
    """Deterministic hash of key training config constants."""
    import hashlib
    from utils.config import (
        ALPHA_LAMBDA_DUAL, GAMMA_MANAGER, GAMMA_WORKER,
        LR_PI_H, LR_PI_L, LR_V_H, LR_V_L,
        MAC_TICKS_PER_WORKER, PPO_CLIP_EPS, PPO_K_EPOCHS,
    )
    blob = (
        f"GAMMA_W={GAMMA_WORKER},GAMMA_M={GAMMA_MANAGER},"
        f"LR_PI_L={LR_PI_L},LR_V_L={LR_V_L},"
        f"LR_PI_H={LR_PI_H},LR_V_H={LR_V_H},"
        f"CLIP={PPO_CLIP_EPS},K_EPOCHS={PPO_K_EPOCHS},"
        f"MAC={MAC_TICKS_PER_WORKER},ALPHA_LAM={ALPHA_LAMBDA_DUAL}"
    )
    return hashlib.md5(blob.encode()).hexdigest()[:8]



# ============================================================
# Episode constants (Phase 1.4 timing — 1s episode = 10 Manager × 10 Worker)
# ============================================================

MANAGER_STEPS_PER_ROLLOUT: int = 10
WORKER_STEPS_PER_ROLLOUT: int = MANAGER_STEPS_PER_ROLLOUT * WORKER_STEPS_PER_MANAGER  # = 100

# Audit 2026-06-22 (Manager critic EV never converges, unlike Worker's — see
# docs audit): Manager's PPO batch was n=MANAGER_STEPS_PER_ROLLOUT=10 per
# update, 10x smaller than Worker's n=100. Decouple Manager's update cadence
# from Worker's: accumulate buf_h across N rollouts (n_eff = 10*N = 100,
# matching Worker) before calling manager.update(); Worker keeps updating
# every single rollout, unchanged.
MANAGER_UPDATE_EVERY_N_ROLLOUTS: int = 10


# ============================================================
# Worker observation overlay (Phase 3.4.4 N4)
# ============================================================
# overlay_lambda_local lives in utils/obs.py — imported above as the single
# source shared by PPO (here) and TD3/SAC (solvers/train_offpolicy.py).
# build_manager_state imported from solvers._common (shared with train_offpolicy.py).


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
    macro_mission: bool = False,
    K_ambulances: int = 1,
    worker_ent_coef: float = 0.01,
    manager_ent_coef: float = 0.01,
    disable_warm_start: bool = False,
    early_stop: bool = False,
    early_stop_patience: int = 300,
    early_stop_min_delta: float = 10.0,
    early_stop_window: int = 100,
    early_stop_min_ep: int = 500,
    eval_at: int = 5000,
    resume: bool = False,
) -> dict:
    """Algorithm 1 main training loop. Returns final-episode stats.

    Auto-saves a rolling ``*_latest.pt`` (manager + worker) + ``ppo_seed{seed}_state.json``
    after EVERY episode. Pass ``resume=True`` to continue from the last saved episode.
    """
    # --- Setup env ---
    # macro_mission = full SUMO mission (K=3 UMa 1km, episode runs until all-arrived
    # / 400s timeout); the W18–W23 sweep config. hard_mission / default = 300m micro
    # scenarios (legacy / unit test). All paths share the same loop (mission until done).
    if macro_mission:
        env_cfg = macro_mission_config(K_ambulances=K_ambulances)
    elif hard_mission:
        env_cfg = hard_mission_config(K_ambulances=K_ambulances)
    else:
        env_cfg = EnvConfig(
            initial_severity=initial_severity,
            K_ambulances=K_ambulances,
            sample_severity=True,
        )
    env = ORANEnv(config=env_cfg, seed=seed)
    K = env.config.K_ambulances
    F = env.config.num_streams
    state_dim_l = env.observation_space.shape[0]
    action_dim_l = env.action_space.shape[0]
    assert state_dim_l == OBS_FIXED_BLOCK_LEN + OBS_PER_AMB_BLOCK_LEN * K + F, (
        f"Worker obs dim {state_dim_l} != "
        f"{OBS_FIXED_BLOCK_LEN}+{OBS_PER_AMB_BLOCK_LEN}K+F (K={K}, F={F})"
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

    # --- Resume from rolling latest checkpoint (auto-save mechanism) ---
    start_ep = 0
    ckpt_dir = Path(checkpoint_dir)
    state_file = state_path(ckpt_dir, "ppo", seed)
    mgr_latest = latest_ckpt_path(ckpt_dir, "manager", seed)
    wkr_latest = latest_ckpt_path(ckpt_dir, "worker", seed)
    if resume:
        st = load_train_state(state_file)
        if st is not None and mgr_latest.exists() and wkr_latest.exists():
            if int(st.get("seed", seed)) != seed:
                raise ValueError(
                    f"Resume seed mismatch: state seed={st.get('seed')} != run seed={seed}"
                )
            manager.load(str(mgr_latest))
            worker.load(str(wkr_latest))
            start_ep = int(st["last_ep"])
            lam_st = st.get("lambda_state")
            if lam_st is not None:
                lambda_state.load_state_dict(lam_st)
                print(f"[ppo] RESUME from ep {start_ep} (with LambdaState, seed={seed})")
            else:
                print(f"[ppo] RESUME from ep {start_ep} (no LambdaState in ckpt, seed={seed})")
            if start_ep >= n_episodes:
                print(f"[ppo] already at/past target {n_episodes} episodes — nothing to do")
        else:
            print(f"[ppo] --resume requested but no latest checkpoint at {state_file} — starting fresh")

    # --- Setup logger (append CSV when resuming so history is preserved) ---
    logger = Logger(
        run_name=f"ppo_seed{seed}",
        log_dir=log_dir,
        use_tensorboard=False,
        use_wandb=use_wandb,
        append_csv=(start_ep > 0),
    )
    logger.log_hparams({
        "algo": "ppo",
        "n_episodes": n_episodes,
        "seed": seed,
        "K_ambulances": K,
        "state_dim_l": state_dim_l,
        "action_dim_l": action_dim_l,
        "manager_state_dim": manager_state_dim_k,
        "manager_action_dim": MANAGER_ACTION_DIM_DEFAULT,
        "manager_steps_per_episode": MANAGER_STEPS_PER_ROLLOUT,
        "worker_steps_per_manager": WORKER_STEPS_PER_MANAGER,
        "sample_severity": env.config.sample_severity,
        "git_commit": _git_commit_short(),
        "config_hash": _config_hash(),
    })

    # Per-episode buffers (Phase 3.4.4 N8: buffer boundary = 1 episode)
    # Worker: 100 transitions per update (every rollout). Manager: accumulates
    # MANAGER_UPDATE_EVERY_N_ROLLOUTS rollouts (n_eff = 100) before updating —
    # see MANAGER_UPDATE_EVERY_N_ROLLOUTS above.
    capacity_w = WORKER_STEPS_PER_ROLLOUT + 10
    capacity_h = MANAGER_STEPS_PER_ROLLOUT * MANAGER_UPDATE_EVERY_N_ROLLOUTS + 2

    es = EarlyStopping(
        patience=early_stop_patience,
        min_delta=early_stop_min_delta,
        window=early_stop_window,
        min_ep=early_stop_min_ep,
    ) if early_stop else None

    t_start = time.time()
    final_stats: dict = {}

    # Fixed reward scale (audit 2026-06-22, replaces adaptive ReturnNormalizer).
    # Divides r_aug by a constant so the critic target stays in a learnable range
    # WITHOUT cross-severity contamination (the adaptive σ was dominated by sev=1,
    # crushing sev=5 penalty signal to invisibility).
    reward_scale = float(REWARD_FIXED_SCALE)

    for ep in range(start_ep, n_episodes):
        # ---------------- Episode reset + LambdaState sync ----------------
        # 1 EPISODE = ONE FULL MISSION (parity với TD3/SAC trong solvers/train_offpolicy.py):
        # env chạy tới khi terminated (cả K xe arrived) hoặc truncated
        # (episode_duration_sec timeout). Severity mỗi xe sample MỘT LẦN ở đây và
        # cố định suốt hành trình. Env PERSIST qua biên PPO update — ta thu thập 1
        # rollout cố định (MANAGER_STEPS_PER_ROLLOUT Manager window = 100 Worker step
        # = 1 s), chạy 1 PPO update, rồi TIẾP TỤC cùng env/scenario tới khi mission
        # done. (N8 revised: PPO buffer boundary = 1 rollout; episode = mission done.)
        # 3 solver dùng CÙNG định nghĩa episode → so sánh công bằng (không xây bài
        # toán riêng cho thuật toán nào).
        obs, info = env.reset(seed=seed + ep)
        severity_init = int(info["severity"])
        severity_per_amb_init = tuple(int(s) for s in info["severity_per_amb"])
        lambda_state.reset_episode(severity_per_amb_init, severity_init)
        # (fixed-scale: no per-episode reset needed — scale is constant)

        ep_reward = 0.0
        ep_reward_base = 0.0
        ep_penalty_total = 0.0
        # Per-constraint signed penalty accumulator (4K+1,): Σ_t λ·(c−d)/scale per
        # constraint. Reveals bonus-masking (slack mean-constraints C1/C4 paying a
        # negative/bonus term that hides violated tail-constraints C2/C5).
        ep_penalty_per_constraint = np.zeros(lambda_state.n_constraints, dtype=np.float64)
        ep_reward_normalized = 0.0   # fix D: sum of what the critic actually saw
        worker_step_idx = 0
        ep_n_active_sum = 0
        ep_steps = 0
        ep_active_count_per_amb = np.zeros(K_ambulances, dtype=np.float64)
        ep_delay_sum_per_amb = np.zeros(K_ambulances, dtype=np.float64)
        ep_delay_viol_per_amb = np.zeros(K_ambulances, dtype=np.float64)
        ep_aoi_sum_per_amb = np.zeros(K_ambulances, dtype=np.float64)
        ep_aoi_viol_per_amb = np.zeros(K_ambulances, dtype=np.float64)
        ep_prb_sum_per_amb = np.zeros(K_ambulances, dtype=np.float64)
        ep_worker_steps_per_amb = np.zeros(K_ambulances, dtype=np.float64)
        done = False
        terminated = False   # last-step true-terminal flag; gates rollout value bootstrap
        ep_b_rrm_sum = 0.0
        ep_b_rrm_count = 0

        w_loss_acc = 0.0
        w_critic_acc = 0.0
        w_entropy_acc = 0.0
        w_clip_acc = 0.0
        w_kl_acc = 0.0
        w_ev_acc = 0.0
        w_n_total = 0
        w_n_updates = 0
        w_n_skipped = 0
        m_loss_acc = 0.0
        m_critic_acc = 0.0
        m_entropy_acc = 0.0
        m_clip_acc = 0.0
        m_kl_acc = 0.0
        m_ev_acc = 0.0
        m_n_total = 0
        m_n_updates = 0
        m_n_skipped = 0

        # Manager buffer persists across MANAGER_UPDATE_EVERY_N_ROLLOUTS rollouts
        # (n_eff=100, matching Worker) — created once per episode, reset only when
        # manager.update() actually fires (every Nth rollout, or at episode end).
        buf_h = _make_storage(capacity_h, manager_state_dim_k, MANAGER_ACTION_DIM_DEFAULT)
        rollouts_since_manager_update = 0

        # ---------------- Mission loop: rollout → PPO update → repeat tới done ------
        while not done:
            # Fresh Worker PPO buffer MỖI ROLLOUT (on-policy: bỏ sau mỗi update).
            # buf_h is NOT recreated here — it accumulates across rollouts (see above).
            buf_w = _make_storage(capacity_w, state_dim_l, action_dim_l)

            # ------------ Manager loop (MANAGER_STEPS_PER_ROLLOUT window = 1 rollout) -
            for _k in range(MANAGER_STEPS_PER_ROLLOUT):
                severity_ref_now = int(info["severity"])
                severity_per_amb_now = tuple(int(s) for s in info["severity_per_amb"])
                lambda_state.on_manager_step_start(severity_per_amb_now, severity_ref_now)   # N9: sync BOTH λ_global + λ_local

                s_H = build_manager_state(obs, lambda_state.get_lambda_global())
                a_H_raw, log_prob_H, value_H = manager.act(s_H)
                b_rrm = decode_manager_action(a_H_raw)["b_rrm"]
                env.set_rrm_budget(b_rrm)
                ep_b_rrm_sum += b_rrm
                ep_b_rrm_count += 1
                r_H_acc = 0.0
                intra_window_step = 0
                done_in_window = False
                terminated_in_window = False   # true MDP terminal (NOT timeout) — bootstrap mask

                # ---------------- Worker loop (W=10 steps per Manager step) ----------
                for _t in range(WORKER_STEPS_PER_MANAGER):
                    # N4: expose λ_local through Worker observation
                    s_L = overlay_lambda_local(obs, lambda_state.get_lambda_local(), K)
                    a_raw, log_prob_L, value_L = worker.act(s_L)

                    next_obs, r_t, terminated, truncated, info = env.step(np.asarray(a_raw, dtype=np.float32))
                    done = bool(terminated or truncated)
                    ep_n_active_sum += int(info.get("n_active", K_ambulances))
                    ep_steps += 1
                    if "active_count_per_amb" in info:
                        ep_active_count_per_amb += info["active_count_per_amb"]
                    _amask = info.get("active_mask", np.ones(K_ambulances, dtype=bool))
                    _am = _amask.astype(np.float64)
                    ep_delay_sum_per_amb += np.asarray(info["delay_norm_per_amb"], dtype=np.float64) * _am
                    ep_aoi_sum_per_amb += np.asarray(info["aoi_norm_per_amb"], dtype=np.float64) * _am
                    ep_prb_sum_per_amb += np.asarray(info["prb_per_amb"], dtype=np.float64)
                    ep_worker_steps_per_amb += _am
                    c_vec = np.asarray(info["c_vec"], dtype=np.float64)
                    ep_delay_viol_per_amb += c_vec[K:2*K] * _am
                    ep_aoi_viol_per_amb += c_vec[3*K:4*K] * _am
                    d_phi = np.asarray(info["d_phi"], dtype=np.float64)

                    r_base = float(r_t)
                    r_aug = lambda_state.augmented_reward(r_base, c_vec, d_phi)
                    ep_reward_base += r_base
                    ep_penalty_total += r_base - float(r_aug)   # RAW penalty (diagnostics)
                    ep_penalty_per_constraint += lambda_state.penalty_breakdown(c_vec, d_phi)
                    lambda_state.accumulate(c_vec, d_phi)

                    r_aug_norm = r_aug / reward_scale
                    ep_reward_normalized += float(r_aug_norm)   # what the critic saw

                    # GAE bootstrap mask = `terminated` (true terminal), NOT `done`:
                    # a 400s timeout truncation is non-terminal (see value_bootstrap_is_terminal).
                    _store(buf_w, s_L, a_raw, log_prob_L, r_aug_norm, value_L, terminated)

                    # SMDP-discounted intra-window return: Σ γ_L^i · r_aug_norm,i
                    r_H_acc += (GAMMA ** intra_window_step) * float(r_aug_norm)
                    intra_window_step += 1
                    ep_reward += float(r_aug)   # RAW augmented reward (diagnostics)
                    worker_step_idx += 1
                    obs = next_obs

                    if done:
                        done_in_window = True
                        terminated_in_window = terminated   # truncation (timeout) ⇒ False ⇒ bootstrap
                        break

                # Store Manager transition (1 per Worker window). Bootstrap mask =
                # terminated_in_window (NOT done_in_window): timeout truncation bootstraps.
                _store(buf_h, s_H, a_H_raw, log_prob_H, r_H_acc, value_H, terminated_in_window)

                # N7: dual ascent + reset win_c (Manager step boundary)
                lambda_state.on_manager_step_end()

                if done_in_window:
                    break

            # ------------ GAE bootstrap tại biên rollout ------------
            # Chỉ zero last_value khi TERMINATED thật (cả K xe arrived). Khi cắt
            # rollout giữa-mission HOẶC truncated (timeout 400s — KHÔNG phải terminal)
            # thì bootstrap V(s) của state env tiếp tục từ đó (Pardo 2018; xem
            # value_bootstrap_is_terminal). done=terminated|truncated chỉ dùng để
            # kết thúc vòng lặp, KHÔNG quyết định bootstrap.
            if value_bootstrap_is_terminal(terminated):
                last_value_w = 0.0
                last_value_h = 0.0
            else:
                s_L_boot = overlay_lambda_local(obs, lambda_state.get_lambda_local(), K)
                _, _, last_value_w = worker.act(s_L_boot, deterministic=True)
                s_H_boot = build_manager_state(obs, lambda_state.get_lambda_global())
                _, _, last_value_h = manager.act(s_H_boot, deterministic=True)

            # ------------ PPO updates (MỖI ROLLOUT, env persist) ------------
            # K=1: Worker action is a true no-op — softmax([ℓ_0])=[1.0] always,
            # so PRB_0=B_U regardless of the action value. WorkerAgent.update()
            # already guards this internally (P1 fix: skip_actor = action_dim==1)
            # — actor gradient is skipped, but the critic still trains (V(s) is
            # still a meaningful target for GAE/Manager bootstrap regardless of
            # the Worker's degenerate action space). Call unconditionally here;
            # do NOT also skip the call at this level — that would additionally
            # (and unnecessarily) starve the critic, which is not what was asked.
            wm = _ppo_update_worker(worker, buf_w, last_value=last_value_w)
            wn = wm.get("worker_n_samples", 0)
            w_n_total += wn
            if wm.get("worker_skipped_partial"):
                w_n_skipped += 1
            elif wn > 0:
                w_loss_acc += wm.get("worker_actor_loss", 0.0) * wn
                w_critic_acc += wm.get("worker_critic_loss", 0.0) * wn
                w_entropy_acc += wm.get("worker_entropy", 0.0) * wn
                w_clip_acc += wm.get("worker_clip_fraction", 0.0) * wn
                w_kl_acc += wm.get("worker_approx_kl", 0.0) * wn
                w_ev_acc += wm.get("worker_explained_variance", 0.0)
                w_n_updates += 1

            # Manager updates every MANAGER_UPDATE_EVERY_N_ROLLOUTS rollouts (n_eff=100,
            # matching Worker) — or immediately at mission end so no transitions leak
            # across the episode boundary (lambda_state resets per-episode;
            # reward_scale is constant). buf_h keeps accumulating on skipped rollouts.
            rollouts_since_manager_update += 1
            if rollouts_since_manager_update >= MANAGER_UPDATE_EVERY_N_ROLLOUTS or done:
                mm = _ppo_update_manager(manager, buf_h, last_value=last_value_h)
                mn = mm.get("manager_n_samples", 0)
                m_n_total += mn
                if mm.get("manager_skipped_partial"):
                    m_n_skipped += 1
                elif mn > 0:
                    m_loss_acc += mm.get("manager_actor_loss", 0.0) * mn
                    m_critic_acc += mm.get("manager_critic_loss", 0.0) * mn
                    m_entropy_acc += mm.get("manager_entropy", 0.0) * mn
                    m_kl_acc += mm.get("manager_approx_kl", 0.0) * mn
                    m_ev_acc += mm.get("manager_explained_variance", 0.0)
                    m_clip_acc += mm.get("manager_clip_fraction", 0.0) * mn
                    m_n_updates += 1
                buf_h = _make_storage(capacity_h, manager_state_dim_k, MANAGER_ACTION_DIM_DEFAULT)
                rollouts_since_manager_update = 0

        # ---------------- Episode (mission) end: flush warm-start ----------------
        # Capture the active dual state before flushing severity warm starts.
        lam_g = lambda_state.get_lambda_global()
        severity_per_amb_logged = lambda_state.sev_prev

        # Flush final active severity into lambda_warm before the next episode reload.
        severity_ref_final = int(info["severity"])
        severity_per_amb_final = tuple(int(s) for s in info["severity_per_amb"])
        lambda_state.on_episode_end(severity_per_amb_final, severity_ref_final)

        # ---------------- Logging ----------------
        mean_n_active = ep_n_active_sum / ep_steps if ep_steps > 0 else 0.0
        metrics: dict = {
            "ep_reward": ep_reward,
            "mean_e2e_ms": env.mean_e2e_ms(),
            "viol_rate": env.episode_violation_rate(),
            "mean_embb_mbps": env.mean_embb_mbps(),
            "c3_viol_rate": env.c3_violation_rate(),
            "mean_aoi_ms": env.mean_aoi_ms(),
            "aoi_viol_rate": env.aoi_violation_rate(),
            "mean_n_active": mean_n_active,
            "severity_init": severity_init,
        }
        for k in range(K_ambulances):
            metrics[f"active_mac_ticks_amb{k}"] = float(ep_active_count_per_amb[k])
            ws_k = ep_worker_steps_per_amb[k]
            d_max_k = float(SEVERITY_QOS[int(severity_per_amb_logged[k])]["D_max"])
            aoi_max_k = float(SEVERITY_QOS[int(severity_per_amb_logged[k])]["AoI_max"])
            metrics[f"severity_amb{k}"] = int(severity_per_amb_logged[k])
            metrics[f"mean_delay_ms_amb{k}"] = float(ep_delay_sum_per_amb[k] / max(ws_k, 1) * d_max_k * 1e3)
            metrics[f"delay_viol_rate_amb{k}"] = float(ep_delay_viol_per_amb[k] / max(ws_k, 1))
            metrics[f"mean_aoi_ms_amb{k}"] = float(ep_aoi_sum_per_amb[k] / max(ws_k, 1) * aoi_max_k * 1e3)
            metrics[f"aoi_viol_rate_amb{k}"] = float(ep_aoi_viol_per_amb[k] / max(ws_k, 1))
            metrics[f"mean_prb_amb{k}"] = float(ep_prb_sum_per_amb[k] / max(ws_k, 1))
            pkt_del = int(info.get("aoi_pkt_delivered", np.zeros(K_ambulances))[k])
            pkt_total = pkt_del + int(info.get("aoi_pkt_failed_bler", np.zeros(K_ambulances))[k]) \
                + int(info.get("aoi_pkt_failed_no_prb", np.zeros(K_ambulances))[k]) \
                + int(info.get("aoi_pkt_failed_no_capacity", np.zeros(K_ambulances))[k])
            metrics[f"delivery_success_rate_amb{k}"] = pkt_del / max(pkt_total, 1)
        metrics["severity_final"] = severity_ref_final
        metrics.update({
            "severity_per_amb": str(list(severity_per_amb_logged)),
            "worker_steps": worker_step_idx,
            "reward_per_step": ep_reward / worker_step_idx if worker_step_idx > 0 else 0.0,
            "reward_base": ep_reward_base,
            "penalty_total": ep_penalty_total,
            "ep_reward_normalized": ep_reward_normalized,   # fix D: sum of critic-seen reward
            "episode_duration_s": worker_step_idx * 0.01,
        })
        # λ_global layout: [C1_0..C1_{K-1}, C2_0..C2_{K-1}, C4_0..C4_{K-1}, C5_0..C5_{K-1}, C3_shared]
        ppc = ep_penalty_per_constraint   # same (4K+1,) layout as λ_global
        for k in range(K):
            metrics[f"lambda_global_C1_{k}"] = float(lam_g[k])
            metrics[f"lambda_global_C2_{k}"] = float(lam_g[K + k])
            metrics[f"lambda_global_C4_{k}"] = float(lam_g[2 * K + k])
            metrics[f"lambda_global_C5_{k}"] = float(lam_g[3 * K + k])
            # Per-constraint signed penalty (negative = slack/bonus, positive = violated).
            metrics[f"penalty_C1_{k}"] = float(ppc[k])
            metrics[f"penalty_C2_{k}"] = float(ppc[K + k])
            metrics[f"penalty_C4_{k}"] = float(ppc[2 * K + k])
            metrics[f"penalty_C5_{k}"] = float(ppc[3 * K + k])
        metrics["lambda_global_C3_shared"] = float(lam_g[4 * K])
        metrics["penalty_C3_shared"] = float(ppc[4 * K])

        w_updated_n = w_n_total - (worker_step_idx % WORKER_STEPS_PER_ROLLOUT if w_n_skipped else 0)
        w_d = max(w_updated_n, 1)
        metrics.update({
            "worker_actor_loss": w_loss_acc / w_d if w_n_updates > 0 else 0.0,
            "worker_critic_loss": w_critic_acc / w_d if w_n_updates > 0 else 0.0,
            "worker_entropy": w_entropy_acc / w_d if w_n_updates > 0 else 0.0,
            "worker_clip_fraction": w_clip_acc / w_d if w_n_updates > 0 else 0.0,
            "worker_approx_kl": w_kl_acc / w_d if w_n_updates > 0 else 0.0,
            "worker_explained_variance": w_ev_acc / max(w_n_updates, 1),
            "worker_n_samples": w_n_total,
            "worker_n_updates": w_n_updates,
            "worker_n_skipped": w_n_skipped,
            "worker_actor_skipped_k1": int(action_dim_l == 1),
        })
        m_d = max(m_n_total, 1)
        metrics.update({
            "manager_actor_loss": m_loss_acc / m_d if m_n_updates > 0 else 0.0,
            "manager_critic_loss": m_critic_acc / m_d if m_n_updates > 0 else 0.0,
            "manager_entropy": m_entropy_acc / m_d if m_n_updates > 0 else 0.0,
            "manager_clip_fraction": m_clip_acc / m_d if m_n_updates > 0 else 0.0,
            "manager_approx_kl": m_kl_acc / m_d if m_n_updates > 0 else 0.0,
            "manager_explained_variance": m_ev_acc / max(m_n_updates, 1),
            "manager_b_rrm_mean": ep_b_rrm_sum / max(ep_b_rrm_count, 1),
            "manager_n_decisions": ep_b_rrm_count,
            "manager_n_samples": m_n_total,
            "manager_n_updates": m_n_updates,
            "manager_n_skipped": m_n_skipped,
            "reward_fixed_scale": reward_scale,
        })
        logger.log_dict(metrics, step=ep)
        final_stats = metrics

        # Milestone checkpoint (archival, kept)
        if checkpoint_every > 0 and ((ep + 1) % checkpoint_every == 0 or ep == n_episodes - 1):
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            manager.save(str(ckpt_dir / f"manager_seed{seed}_ep{ep + 1}.pt"))
            worker.save(str(ckpt_dir / f"worker_seed{seed}_ep{ep + 1}.pt"))

        # Rolling auto-save AFTER EVERY episode (overwrites; for --resume).
        # State sidecar written last + atomically so it only points at a fully
        # saved (manager, worker) pair.
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        manager.save(str(mgr_latest))
        worker.save(str(wkr_latest))
        save_train_state(state_file, last_ep=ep + 1, seed=seed,
                         extra={
                             "algo": "ppo",
                             "n_episodes_target": n_episodes,
                             "lambda_state": lambda_state.state_dict(),
                             "reward_fixed_scale": reward_scale,
                         })

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


def _ppo_update_worker(worker, buf, last_value: float = 0.0) -> dict[str, float]:
    n = buf["ptr"]
    if n == 0:
        return {"worker_n_samples": 0}
    if n < MINIBATCH_SIZE:
        return {"worker_n_samples": n, "worker_skipped_partial": 1}
    sl = _slice(buf)
    return worker.update(
        obs=sl["obs"],
        actions_raw=sl["actions"],
        old_log_probs=sl["log_probs"],
        rewards=sl["rewards"],
        values=sl["values"],
        dones=sl["dones"],
        last_value=last_value,
    )


def _ppo_update_manager(manager, buf, last_value: float = 0.0) -> dict[str, float]:
    n = buf["ptr"]
    if n == 0:
        return {"manager_n_samples": 0}
    if n < 2:
        return {"manager_n_samples": n, "manager_skipped_partial": 1}
    sl = _slice(buf)
    return manager.update(
        obs=sl["obs"],
        actions=sl["actions"],
        old_log_probs=sl["log_probs"],
        rewards=sl["rewards"],
        values=sl["values"],
        dones=sl["dones"],
        last_value=last_value,
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
        choices=["ppo", "td3", "sac",
                 "pa_ppo_soft", "ppo_cmdp_flat"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, default=None,
                        help="Checkpoint directory. Defaults to <log-dir>/checkpoints if not set.")
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--resume-checkpoint", type=str, default=None,
                        help="Path to .pt checkpoint to resume from (solvers only)")
    parser.add_argument("--resume-start-ep", type=int, default=0,
                        help="Episode offset when resuming (metrics appended after this ep)")
    parser.add_argument("--resume", action="store_true",
                        help="Auto-resume from the rolling *_latest.pt checkpoint for this "
                             "seed and continue toward --episodes (uses the per-episode auto-save).")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--hard", action="store_true", help="Use hard-mission preset (300m micro, 1s)")
    parser.add_argument("--macro", action="store_true",
                        help="Use W15-B2 macro mission (K=3 UMa 1km SUMO, episode = full journey until all-arrived/400s). The W18-W23 sweep config.")
    parser.add_argument("--K", dest="K_ambulances", type=int, default=None,
                        help="Number of ambulances. Defaults to 3 for --macro and 1 otherwise.")
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--severity", type=int, default=5,
                        help="Fixed patient severity 1..5 (NON_URGENT..IMMEDIATE) for the run")
    parser.add_argument("--device", type=str, default="cuda")
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
    K_ambulances = args.K_ambulances if args.K_ambulances is not None else (3 if args.macro else 1)
    run_dir = args.log_dir / f"{args.algo}_K{K_ambulances}_seed{args.seed}"
    if args.checkpoint_dir is None:
        args.checkpoint_dir = run_dir / "checkpoints"
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            print("[warn] CUDA not available, falling back to CPU")
            args.device = "cpu"

    if args.algo == "ppo":
        train_ppo(
            n_episodes=args.episodes,
            seed=args.seed,
            log_dir=str(run_dir),
            initial_severity=args.severity,
            device=args.device,
            print_every=args.print_every,
            use_wandb=args.wandb,
            checkpoint_dir=str(args.checkpoint_dir),
            checkpoint_every=args.checkpoint_every,
            hard_mission=args.hard,
            macro_mission=args.macro,
            K_ambulances=K_ambulances,
            worker_ent_coef=args.worker_ent_coef,
            manager_ent_coef=args.manager_ent_coef,
            disable_warm_start=args.no_warm_start,
            early_stop=args.early_stop,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_delta=args.early_stop_min_delta,
            early_stop_window=args.early_stop_window,
            early_stop_min_ep=args.early_stop_min_ep,
            eval_at=args.eval_at,
            resume=args.resume,
        )
        return 0

    # Sibling solvers (TD3/SAC) + ablations delegate to solvers.train_offpolicy
    from solvers.train_offpolicy import train as bl_train
    bl_train(
        baseline_name=args.algo,
        n_episodes=args.episodes,
        seed=args.seed,
        log_dir=str(run_dir),
        initial_severity=args.severity,
        device=args.device,
        print_every=args.print_every,
        use_wandb=args.wandb,
        checkpoint_dir=str(args.checkpoint_dir),
        checkpoint_every=args.checkpoint_every,
        hard_mission=args.hard,
        macro_mission=args.macro,
        K_ambulances=K_ambulances,
        early_stop=args.early_stop,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        early_stop_window=args.early_stop_window,
        early_stop_min_ep=args.early_stop_min_ep,
        eval_at=args.eval_at,
        resume_checkpoint=args.resume_checkpoint,
        resume_start_ep=args.resume_start_ep,
        resume=args.resume,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
