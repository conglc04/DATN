"""TD3/SAC --resume must restore LambdaState, not just network weights.

Gap found during the 2026-06-21 cross-solver sync audit: solvers/train_offpolicy.py
saved/restored only agent.save()/load() (network weights) on --resume — unlike
train.py (PPO), which explicitly round-trips lambda_state.state_dict() through the
JSON state sidecar. This meant a resumed TD3/SAC run silently reset λ_global/
λ_local/λ_warm/win_c/cum_c to zero/warm-start, discarding all dual-ascent progress
from before the interruption. Fixed in solvers/train_offpolicy.py (save: add
"lambda_state" to the `extra` dict; resume: load it back via
agent.lambda_state.load_state_dict()).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from solvers.train_offpolicy import train as train_offpolicy


@pytest.mark.parametrize("baseline_name", ["td3", "sac"])
def test_resume_restores_lambda_state_not_just_weights(tmp_path, baseline_name):
    ckpt_dir = tmp_path / "checkpoints"
    log_dir = tmp_path / "logs"

    out1 = train_offpolicy(
        baseline_name=baseline_name,
        n_episodes=2,
        seed=0,
        log_dir=str(log_dir / "run1"),
        checkpoint_dir=str(ckpt_dir),
        checkpoint_every=0,
        print_every=10_000,
        hard_mission=True,
        K_ambulances=1,
    )
    lam_after_run1 = out1["final_lambdas"]
    # Sanity: dual ascent must have moved λ away from a trivial all-zero state
    # for this test to be meaningful (otherwise "restored" vs "reset" look identical).
    assert any(v != 0.0 for v in lam_after_run1), (
        f"λ_global all zero after run1 — test cannot distinguish restore vs reset: {lam_after_run1}"
    )

    out2 = train_offpolicy(
        baseline_name=baseline_name,
        n_episodes=2,  # n_episodes = TARGET total when auto-resuming
        seed=0,
        log_dir=str(log_dir / "run2"),
        checkpoint_dir=str(ckpt_dir),
        checkpoint_every=0,
        print_every=10_000,
        hard_mission=True,
        K_ambulances=1,
        resume=True,
    )
    # n_episodes(2) == resume_start_ep(2) -> train() runs 0 additional iterations;
    # out2 should reflect the run1 LambdaState, untouched, confirming the resume
    # path actually restored it rather than starting from zero/warm-start.
    lam_after_resume = out2["final_lambdas"]
    np.testing.assert_allclose(
        lam_after_resume, lam_after_run1,
        err_msg=f"{baseline_name} resume lost λ state: before={lam_after_run1} after={lam_after_resume}",
    )


@pytest.mark.parametrize("baseline_name", ["td3", "sac"])
def test_state_sidecar_contains_lambda_state(tmp_path, baseline_name):
    """Direct check: the JSON state sidecar must carry a non-trivial lambda_state."""
    from utils.checkpointing import load_train_state, state_path

    ckpt_dir = tmp_path / "checkpoints"
    train_offpolicy(
        baseline_name=baseline_name,
        n_episodes=2,
        seed=1,
        log_dir=str(tmp_path / "logs"),
        checkpoint_dir=str(ckpt_dir),
        checkpoint_every=0,
        print_every=10_000,
        hard_mission=True,
        K_ambulances=1,
    )
    st = load_train_state(state_path(ckpt_dir, baseline_name, 1))
    assert st is not None
    assert "lambda_state" in st, f"{baseline_name} state sidecar missing lambda_state key"
    assert "lambda_global" in st["lambda_state"]
