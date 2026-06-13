"""Week 1 sanity: verify all packages + utility modules import without error,
and that key constants in utils.config match the docs single-source-of-truth.
"""

from __future__ import annotations

import importlib


def test_package_imports():
    """All scaffolded packages must be importable."""
    for pkg in ("env", "agents", "baselines", "experiments", "utils"):
        importlib.import_module(pkg)


def test_utils_imports():
    """Utility submodules must import."""
    from utils import config, logger, metrics  # noqa: F401


def test_config_constants_match_docs():
    """Constants must match docs single-source-of-truth values."""
    from utils import config as cfg

    # Hardware
    assert cfg.P_TOTAL == 273, f"P_TOTAL mismatch: {cfg.P_TOTAL}"
    assert cfg.B_PRB == 360e3
    assert cfg.TTI_SEC == 0.5e-3
    assert cfg.F_CARRIER == 3.5e9
    assert cfg.F_MEC == 10e9

    # Delay components
    assert cfg.D_FH == 0.1e-3
    assert cfg.D_BH == 0.1e-3
    assert cfg.D_DET == 0.07e-3


def test_phase_qos_table():
    """Phase QoS table must have all 5 phases with the right tightest budgets."""
    from utils import config as cfg

    assert set(cfg.PHASE_QOS.keys()) == {1, 2, 3, 4, 5}
    # φ₃ SCENE is tightest
    assert cfg.PHASE_QOS[3]["D_max"] == 1e-3
    assert cfg.PHASE_QOS[3]["eps"] == 1e-5
    assert cfg.PHASE_QOS[3]["AoI_max_HR"] == 0.1
    # φ₁/φ₅ relaxed
    assert cfg.PHASE_QOS[1]["D_max"] == 20e-3
    assert cfg.PHASE_QOS[5]["D_max"] == 20e-3


def test_cmdp_thresholds():
    """CMDP d_j^φ values must cover all 5 constraints per phase."""
    from utils import config as cfg

    required_keys = {"d1_lat_mean", "d2_lat_tail", "d3_embb_mbps", "d4_aoi_mean", "d5_aoi_tail"}
    for phase, thresholds in cfg.CMDP_D_J_PHI.items():
        assert required_keys.issubset(thresholds.keys()), (
            f"Phase {phase} missing keys: {required_keys - set(thresholds.keys())}"
        )


def test_lambda_warm():
    """λ_warm post-training values for φ₃ must match docs/05_agent_workflow.md:178."""
    from utils import config as cfg

    expected_phi3 = [1.80, 2.20, 0.10, 1.50, 2.00]
    assert cfg.LAMBDA_WARM[3] == expected_phi3, f"λ_warm[φ₃] mismatch: {cfg.LAMBDA_WARM[3]}"


def test_rl_hyperparams():
    """RL hyperparams match docs/09_execution_plan.md reference table."""
    from utils import config as cfg

    assert cfg.PPO_CLIP_EPS == 0.2
    assert cfg.GAMMA == 0.99
    assert cfg.GAE_LAMBDA == 0.95
    assert cfg.PPO_K_EPOCHS == 10
    assert cfg.MINIBATCH_SIZE == 64
    # Borkar 2008 two-timescale: α_πH ≪ α_πL (ratio 0.01, 2 orders apart).
    # Old test asserted 1e-4 / 3e-4 (ratio 0.33) — corrected per
    # docs/13 Phase 1.4 Borkar review (2026-05-20).
    assert cfg.LR_PI_H == 1e-5
    assert cfg.LR_PI_L == 1e-3
    assert cfg.LR_PI_H / cfg.LR_PI_L == 1e-2


def test_phase2_constants_w05():
    """W05 Phase 2.1 reward normalization + Phase 3.2.4 distinct discount notation."""
    from utils import config as cfg

    # Phase 2.1 reward normalization (docs/13 Phase 2.1)
    assert cfg.D_REF_URLLC == 1e-3, "D_REF_URLLC should be 1 ms (tightest D_max budget)"
    assert cfg.R_REF_EMBB_MBPS == 100.0, "R_REF_EMBB_MBPS should be 100 Mbps"

    # Phase 3.2.4 Worker/Manager distinct discount (γ_H = γ_L^W)
    assert cfg.GAMMA_WORKER == 0.99
    assert abs(cfg.GAMMA_MANAGER - 0.99 ** cfg.WORKER_STEPS_PER_MANAGER) < 1e-9
    assert abs(cfg.GAMMA_MANAGER - 0.9043820750088045) < 1e-9

    # Three-rate hierarchy locked (Phase 2.3.4)
    assert cfg.LR_PI_H < cfg.ALPHA_LAMBDA_DUAL < cfg.LR_PI_L


def test_master_table_helper():
    """get_phase_thresholds() returns 5-key dict matching Master Table."""
    from utils.config import get_phase_thresholds, get_phase_alpha

    # Phase 3 SCENE — tightest constraints
    th = get_phase_thresholds(3)
    assert set(th.keys()) == {"d1", "d2", "d3", "d4", "d5"}
    assert th["d1"] == 1e-3       # D_max
    assert th["d2"] == 1e-5       # eps tail
    assert th["d3"] == 0.0        # C3 threshold; R_min eMBB is in CMDP_D_J_PHI
    assert th["d4"] == 0.1        # AoI_max HR
    assert th["d5"] == 1e-3       # eps AoI tail

    # Phase 1 STANDBY — relaxed
    th1 = get_phase_thresholds(1)
    assert th1["d1"] == 20e-3
    assert th1["d3"] == 0.0

    # alpha helper
    au, ae = get_phase_alpha(3)
    assert au == 0.95 and ae == 0.05
    assert abs(au + ae - 1.0) < 1e-9


def test_worker_steps_per_manager():
    """O-RAN 3-level timing hierarchy (post 2026-05-20 correction).

    MAC tick    0.5 ms (T_TTI_SEC)
    Worker xApp 10 ms = 20 MAC ticks (MAC_TICKS_PER_WORKER)
    Manager rApp 100 ms sim = 10 Worker steps (WORKER_STEPS_PER_MANAGER)
    """
    from utils import config as cfg

    assert cfg.T_TTI_SEC == 0.5e-3
    assert cfg.T_L_SEC == 10e-3
    assert cfg.T_H_SEC == 100e-3
    assert cfg.MAC_TICKS_PER_WORKER == 20
    assert cfg.WORKER_STEPS_PER_MANAGER == 10


def test_handover_eta_trigger():
    """Trigger threshold = 10s per Bug 2 fix (not 5s)."""
    from utils import config as cfg

    assert cfg.HANDOVER_ETA_TRIGGER == 10.0


def test_logger_csv_mode():
    """Logger should work in CSV-only mode without TB / WandB."""
    import tempfile
    from utils.logger import Logger

    with tempfile.TemporaryDirectory() as tmp:
        with Logger(
            run_name="test_csv",
            log_dir=tmp,
            use_tensorboard=False,
            use_wandb=False,
        ) as lg:
            lg.log_scalar("foo", 1.0, step=0)
            lg.log_scalar("foo", 2.0, step=1)
            lg.log_dict({"bar": 3.0, "baz": 4.0}, step=2)


def test_metrics_basic():
    """Basic metrics behave sanely."""
    from utils.metrics import (
        violation_rate,
        jain_fairness,
        aoi_violation_rate,
        embb_throughput_mbps,
        queue_stability_check,
        hoeffding_sample_size,
        assert_prb_budget,
    )

    assert violation_rate([0.5e-3, 0.8e-3, 2.0e-3], 1e-3) == 1.0 / 3.0
    assert jain_fairness([1.0, 1.0, 1.0]) == 1.0
    assert aoi_violation_rate([], 1.0) == 0.0
    assert embb_throughput_mbps(1e6, 1.0) == 1.0
    assert queue_stability_check(0.5, 1.0) is True
    assert queue_stability_check(1.0, 1.0) is False
    n = hoeffding_sample_size(target_eps=1e-4, observed_eps=1e-5, confidence=0.99)
    assert n > 1_000_000  # Hoeffding for rare events is huge (per Bug 2 fix)
    assert_prb_budget(100, 100)  # OK
