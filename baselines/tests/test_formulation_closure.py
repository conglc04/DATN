"""Gate 15 — formulation closure: run the consolidated independent checks.

Wraps audit/closure_checks.py (independent re-derivations of channel, queue,
timescale, N_req, constraint sign, determinism, conservation) as a pytest gate.
"""
from __future__ import annotations


def test_all_independent_closure_checks_pass():
    from audit.closure_checks import main
    assert main() == 0, "one or more independent closure checks FAILED"
