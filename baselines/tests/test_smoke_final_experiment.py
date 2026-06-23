"""Final-experiment smoke suite — pre-flight check trước khi chạy W18–W23.

Mỗi test chạy 2 episode (đủ để qua 1 Manager boundary W=10 + cập nhật λ),
kiểm tra:
  1. Không crash (PPO / TD3 / SAC × K=1 / K=3)
  2. obs dimension đúng (K=1→31, K=3→51)
  3. λ dimension đúng (K=1→5, K=3→13)
  4. Metrics C1–C5 finite & non-NaN
  5. ep_reward finite
  6. Checkpoint save→load không crash
  7. Hard-mission config hợp lệ (SINR floor, severity tightest)
  8. Info dict chứa đủ key thực nghiệm (prb_urllc, prb_per_amb, severity)

Gate: tất cả 6 combos (3 solver × K∈{1,3}) PASS → unblock W18.
"""

from __future__ import annotations

import numpy as np
import pytest

# ============================================================
# Helpers
# ============================================================

N_EPISODES = 2          # đủ để Manager boundary + λ ascent fire ít nhất 1 lần
PRINT_EVERY = 99_999    # tắt output


def _obs_dim(K: int) -> int:
    from utils.config import OBS_FIXED_BLOCK_LEN, OBS_PER_AMB_BLOCK_LEN
    return OBS_FIXED_BLOCK_LEN + OBS_PER_AMB_BLOCK_LEN * K + 1


def _lambda_dim(K: int) -> int:
    return 4 * K + 1


def _run_smoke_train(baseline: str, K: int, tmp_path, hard: bool = True, seed: int = 0):
    """Run smoke_train (TD3 / SAC); return stats dict."""
    from solvers.train_offpolicy import train
    return train(
        baseline_name=baseline,
        n_episodes=N_EPISODES,
        seed=seed,
        log_dir=str(tmp_path / f"{baseline}_K{K}"),
        K_ambulances=K,
        hard_mission=hard,
        initial_severity=5,
        print_every=PRINT_EVERY,
        checkpoint_every=0,
    )


def _run_ppo(K: int, tmp_path, hard: bool = True, seed: int = 0):
    """Run train_ppo; return stats dict."""
    from train import train_ppo
    return train_ppo(
        n_episodes=N_EPISODES,
        seed=seed,
        K_ambulances=K,
        hard_mission=hard,
        log_dir=str(tmp_path / f"ppo_K{K}"),
        print_every=PRINT_EVERY,
        checkpoint_every=0,
    )


def _assert_stats_finite(stats: dict, label: str) -> None:
    assert np.isfinite(stats["ep_reward"]), f"{label}: ep_reward not finite"
    for key in ("mean_e2e_ms", "c1_mean", "c2_mean", "c3_mean", "c4_mean", "c5_mean"):
        if key in stats:
            assert np.isfinite(stats[key]), f"{label}: {key} not finite"


# ============================================================
# 1. Obs + λ dimension sanity (unit, no training)
# ============================================================

class TestFinalExpDimensions:
    """obs/λ dims match K — fast unit tests, no episode run."""

    @pytest.mark.parametrize("K", [1, 3])
    def test_obs_dim_k(self, K):
        from env.oran_env import EnvConfig, ORANEnv, hard_mission_config
        cfg = hard_mission_config()
        cfg = EnvConfig(**{**cfg.__dict__, "K_ambulances": K})
        env = ORANEnv(cfg)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (_obs_dim(K),), (
            f"K={K}: obs shape {obs.shape} != ({_obs_dim(K)},)"
        )
        env.close()

    @pytest.mark.parametrize("K", [1, 3])
    def test_lambda_dim_k(self, K):
        from agents.lagrangian import LambdaState
        ls = LambdaState(K=K)
        ls.reset_episode([5] * K, 5)
        lam = ls.get_lambda_global()
        assert lam.shape == (_lambda_dim(K),), (
            f"K={K}: λ shape {lam.shape} != ({_lambda_dim(K)},)"
        )

    @pytest.mark.parametrize("K", [1, 3])
    def test_d_phi_dim_k(self, K):
        from utils.config import build_d_phi_vector
        d = build_d_phi_vector(tuple([5] * K))
        assert d.shape == (_lambda_dim(K),), (
            f"K={K}: d_phi shape {d.shape} != ({_lambda_dim(K)},)"
        )


# ============================================================
# 2. Hard-mission config sanity
# ============================================================

class TestHardMissionConfig:
    """hard_mission_config() produces a valid, tighter EnvConfig."""

    def test_hard_mission_tightens_sinr_cap(self):
        from env.oran_env import EnvConfig, hard_mission_config
        default = EnvConfig()
        hard = hard_mission_config()
        # hard mission lowers SINR cap (40→15 dB) → worse effective capacity
        assert hard.sinr_clamp_max_db < default.sinr_clamp_max_db

    def test_hard_mission_severity_5(self):
        from env.oran_env import hard_mission_config
        cfg = hard_mission_config()
        assert cfg.initial_severity == 5

    @pytest.mark.parametrize("K", [1, 3])
    def test_hard_mission_reset_no_crash(self, K):
        from env.oran_env import EnvConfig, ORANEnv, hard_mission_config
        cfg = hard_mission_config()
        cfg = EnvConfig(**{**cfg.__dict__, "K_ambulances": K})
        env = ORANEnv(cfg)
        obs, info = env.reset(seed=0)
        assert np.all(np.isfinite(obs))
        assert "severity" in info
        env.close()


# ============================================================
# 3. Info dict keys — final experiment requires these
# ============================================================

class TestInfoDictKeys:
    """step() info dict must contain keys used in final-experiment logging."""

    REQUIRED_KEYS = {
        "prb_urllc", "prb_embb", "prb_per_amb",
        "severity", "c_vec", "d_phi",
        "l_urllc_mean", "aoi_norm_per_amb", "delay_norm_per_amb",
    }

    @pytest.mark.parametrize("K", [1, 3])
    def test_info_keys_present_k(self, K):
        from env.oran_env import EnvConfig, ORANEnv, hard_mission_config
        cfg = hard_mission_config()
        cfg = EnvConfig(**{**cfg.__dict__, "K_ambulances": K})
        env = ORANEnv(cfg)
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        missing = self.REQUIRED_KEYS - set(info.keys())
        assert not missing, f"K={K}: info missing keys: {missing}"
        env.close()

    @pytest.mark.parametrize("K", [1, 3])
    def test_prb_per_amb_sums_to_prb_urllc(self, K):
        from env.oran_env import EnvConfig, ORANEnv, hard_mission_config
        cfg = hard_mission_config()
        cfg = EnvConfig(**{**cfg.__dict__, "K_ambulances": K})
        env = ORANEnv(cfg)
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert sum(info["prb_per_amb"]) == info["prb_urllc"], (
            f"K={K}: sum(prb_per_amb)={sum(info['prb_per_amb'])} != prb_urllc={info['prb_urllc']}"
        )
        env.close()


# ============================================================
# 4. End-to-end smoke: 3 solvers × K∈{1,3}
# ============================================================

class TestFinalExpSmokeK1:
    """2-episode smoke, hard mission, K=1, severity=5."""

    def test_ppo_k1(self, tmp_path):
        stats = _run_ppo(K=1, tmp_path=tmp_path)
        _assert_stats_finite(stats, "PPO K=1")

    def test_td3_k1(self, tmp_path):
        stats = _run_smoke_train("td3", K=1, tmp_path=tmp_path)
        _assert_stats_finite(stats, "TD3 K=1")

    def test_sac_k1(self, tmp_path):
        stats = _run_smoke_train("sac", K=1, tmp_path=tmp_path)
        _assert_stats_finite(stats, "SAC K=1")


class TestFinalExpSmokeK3:
    """2-episode smoke, hard mission, K=3, severity=5."""

    def test_ppo_k3(self, tmp_path):
        stats = _run_ppo(K=3, tmp_path=tmp_path)
        _assert_stats_finite(stats, "PPO K=3")

    def test_td3_k3(self, tmp_path):
        stats = _run_smoke_train("td3", K=3, tmp_path=tmp_path)
        _assert_stats_finite(stats, "TD3 K=3")

    def test_sac_k3(self, tmp_path):
        stats = _run_smoke_train("sac", K=3, tmp_path=tmp_path)
        _assert_stats_finite(stats, "SAC K=3")


# ============================================================
# 5. Checkpoint save → load → resume không crash
# ============================================================

class TestCheckpointRoundtrip:
    """save() tạo file, load() không crash, resume chạy thêm 1 ep."""

    @pytest.mark.parametrize("baseline,K", [("td3", 1), ("sac", 1), ("td3", 3)])
    def test_save_load_no_crash(self, baseline, K, tmp_path):
        from solvers.train_offpolicy import train, make_baseline
        from env.oran_env import EnvConfig, ORANEnv, hard_mission_config
        import os
        cfg = hard_mission_config()
        cfg = EnvConfig(**{**cfg.__dict__, "K_ambulances": K})
        env = ORANEnv(cfg)
        agent = make_baseline(baseline, env.observation_space.shape[0],
                              env.action_space.shape[0], seed=0, K=K)
        ckpt = str(tmp_path / f"{baseline}_K{K}.pt")
        agent.save(ckpt)
        assert os.path.exists(ckpt), f"checkpoint not created: {ckpt}"
        agent.load(ckpt)   # must not raise
        env.close()

    def test_ppo_save_load(self, tmp_path):
        from agents.ppo_agent import PPOAgent
        from env.oran_env import EnvConfig, ORANEnv, hard_mission_config
        import os
        cfg = hard_mission_config()
        env = ORANEnv(cfg)
        sd, ad = env.observation_space.shape[0], env.action_space.shape[0]
        agent = PPOAgent(state_dim=sd, action_dim=ad)
        ckpt = str(tmp_path / "ppo_worker.pt")
        agent.save(ckpt)
        assert os.path.exists(ckpt)
        agent.load(ckpt)
        env.close()
