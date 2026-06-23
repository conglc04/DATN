"""TD3 training entry point (separate from SAC).

Dedicated runnable file for the TD3 sibling solver so TD3 and SAC are launched
from SEPARATE files (not one shared switch). The shared off-policy training
loop lives in `solvers.train_offpolicy` (a library); this file only fixes
the solver to TD3 and exposes the CLI.

Usage:
    python -m solvers.train_td3 --macro --K 1 --episodes 10000 --seed 0
    python -m solvers.train_td3 --macro --K 3 --episodes 10000 --seed 0
"""
from __future__ import annotations

from solvers.train_offpolicy import run_cli

BASELINE = "td3"


def main(argv: list[str] | None = None) -> int:
    return run_cli(BASELINE, argv)


if __name__ == "__main__":
    raise SystemExit(main())
