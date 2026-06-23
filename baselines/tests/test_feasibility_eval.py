"""Unit tests for the point-12 severity-conditioned feasibility verdict.

Drives the PURE `feasibility_verdict` with synthetic pooled accumulators (no env)
so the scientific criterion is locked: feasibility-first, three-valued tails
(pass/fail/inconclusive), reward/step diagnostic-only.
"""
from __future__ import annotations

from audit.feasibility_eval import feasibility_verdict
from utils.config import SEVERITY_QOS


def _amb(delay_ms, aoi_s, ticks, delay_exc=0, aoi_exc=0):
    """Build a per-amb accumulator with given mean delay (ms) / mean AoI (s)."""
    return {
        "delay_sum": delay_ms * 1e-3 * ticks, "delay_ticks": ticks, "delay_exc": delay_exc,
        "aoi_sum": aoi_s * ticks, "aoi_ticks": ticks, "aoi_exc": aoi_exc,
    }


def _ref(gap_mbps, steps=100, reward=5.0):
    return {"embb_gap_sum": gap_mbps * steps, "embb_steps": steps,
            "reward_sum": reward * steps, "reward_steps": steps}


def test_all_feasible_sev1():
    # sev1: D_max=20ms, AoI_max=1.0s, eps=1e-3 (need N>=3000), eps_aoi=1e-2 (N>=300)
    n = 5000  # resolvable for both tails
    per_amb = {1: _amb(delay_ms=5.0, aoi_s=0.5, ticks=n, delay_exc=0, aoi_exc=0)}
    per_ref = {1: _ref(gap_mbps=-50.0)}   # eMBB surplus → no shortfall
    v = feasibility_verdict(per_amb, per_ref)
    row = v["per_severity"][1]
    assert row["C1"] is True
    assert row["C4"] is True
    assert row["C3"] is True
    assert row["C2"] == "pass"
    assert row["C5"] == "pass"
    assert row["feasible"] is True
    assert v["overall_pass"] is True


def test_mean_delay_over_budget_fails_c1():
    n = 5000
    per_amb = {1: _amb(delay_ms=30.0, aoi_s=0.5, ticks=n)}   # 30ms > 20ms
    per_ref = {1: _ref(gap_mbps=-50.0)}
    v = feasibility_verdict(per_amb, per_ref)
    assert v["per_severity"][1]["C1"] is False
    assert v["per_severity"][1]["feasible"] is False
    assert v["overall_pass"] is False


def test_tail_inconclusive_when_undersampled():
    # sev1 eps=1e-3 needs N>=3000; give 100 → C2 inconclusive (cannot certify)
    per_amb = {1: _amb(delay_ms=5.0, aoi_s=0.5, ticks=100, delay_exc=0, aoi_exc=0)}
    per_ref = {1: _ref(gap_mbps=-50.0)}
    v = feasibility_verdict(per_amb, per_ref)
    row = v["per_severity"][1]
    assert row["C2"] == "inconclusive"   # NOT a feasibility certificate
    assert row["feasible"] is False
    assert v["overall_pass"] is False


def test_tail_fail_when_resolvable_but_over_eps():
    n = 5000
    # 50 delay exceedances in 5000 → obs 1e-2 >> eps=1e-3 → C2 fail
    per_amb = {1: _amb(delay_ms=5.0, aoi_s=0.5, ticks=n, delay_exc=50, aoi_exc=0)}
    per_ref = {1: _ref(gap_mbps=-50.0)}
    v = feasibility_verdict(per_amb, per_ref)
    assert v["per_severity"][1]["C2"] == "fail"
    assert v["per_severity"][1]["feasible"] is False


def test_c3_shortfall_fails():
    n = 5000
    per_amb = {1: _amb(delay_ms=5.0, aoi_s=0.5, ticks=n)}
    per_ref = {1: _ref(gap_mbps=+5.0)}   # mean gap +5 Mbps → eMBB below floor
    v = feasibility_verdict(per_amb, per_ref)
    row = v["per_severity"][1]
    assert row["C3"] is False
    assert row["embb_shortfall_mbps"] == 5.0
    assert row["feasible"] is False


def test_reward_is_diagnostic_not_a_gate():
    # Low reward but fully feasible → still feasible (reward must NOT gate).
    n = 5000
    per_amb = {1: _amb(delay_ms=5.0, aoi_s=0.5, ticks=n)}
    per_ref = {1: _ref(gap_mbps=-50.0, reward=0.001)}
    v = feasibility_verdict(per_amb, per_ref)
    row = v["per_severity"][1]
    assert row["feasible"] is True          # feasibility independent of reward
    assert row["reward_per_step"] < 0.01    # reward still reported
    assert v["overall_pass"] is True


def test_empty_accumulators_do_not_pass():
    # No observed tier → cannot declare success.
    v = feasibility_verdict({}, {})
    assert v["per_severity"] == {}
    assert v["overall_pass"] is False


def test_eps_resolution_threshold_matches_rule_of_three():
    # sev5 eps=1e-5 needs N>=3e5; a short eval can NEVER certify it → inconclusive.
    assert SEVERITY_QOS[5]["eps"] == 1e-5
    per_amb = {5: _amb(delay_ms=0.5, aoi_s=0.05, ticks=10_000, delay_exc=0, aoi_exc=0)}
    per_ref = {5: _ref(gap_mbps=-50.0)}
    v = feasibility_verdict(per_amb, per_ref)
    assert v["per_severity"][5]["C2"] == "inconclusive"   # 10k < 3e5
    assert v["overall_pass"] is False
