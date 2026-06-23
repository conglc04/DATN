"""SAC training entry point (separate from TD3).

Dedicated runnable file for the SAC sibling solver so SAC and TD3 are launched
from SEPARATE files (not one shared switch). The shared off-policy training
loop lives in `solvers.train_offpolicy` (a library); this file only fixes
the solver to SAC and exposes the CLI.

Usage:
    python -m solvers.train_sac --macro --K 1 --episodes 10000 --seed 0
    python -m solvers.train_sac --macro --K 3 --episodes 10000 --seed 0
"""
from __future__ import annotations

from solvers.train_offpolicy import run_cli

BASELINE = "sac"


def main(argv: list[str] | None = None) -> int:
    return run_cli(BASELINE, argv)


if __name__ == "__main__":
    raise SystemExit(main())
