"""Per-episode rolling checkpoint + resume state for training continuation.

Two layers of checkpoints:
  - **Milestone** (``--checkpoint-every``): archival, named ``*_ep{N}.pt`` (kept).
  - **Rolling latest** (every episode): overwrites a fixed ``*_latest.pt`` + a
    ``*_state.json`` sidecar with the last completed episode. Lets training be
    interrupted at any episode and resumed with ``--resume`` from the next one.

The state JSON is written atomically (tmp + ``os.replace``) so an interruption
mid-write never corrupts a resumable checkpoint.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def latest_ckpt_path(checkpoint_dir: str | Path, prefix: str, seed: int) -> Path:
    """Rolling weight-checkpoint path, e.g. ``checkpoints/worker_seed0_latest.pt``."""
    return Path(checkpoint_dir) / f"{prefix}_seed{seed}_latest.pt"


def state_path(checkpoint_dir: str | Path, run_tag: str, seed: int) -> Path:
    """Resume-state sidecar path, e.g. ``checkpoints/ppo_seed0_state.json``."""
    return Path(checkpoint_dir) / f"{run_tag}_seed{seed}_state.json"


def save_train_state(
    path: str | Path,
    *,
    last_ep: int,
    seed: int,
    extra: dict[str, Any] | None = None,
) -> None:
    """Atomically write the resume sidecar.

    Args:
        last_ep: number of COMPLETED episodes (resume starts here).
        seed: run seed (guards against resuming a mismatched run).
        extra: optional extra fields (e.g. best metric) merged into the payload.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"last_ep": int(last_ep), "seed": int(seed)}
    if extra:
        payload.update(extra)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)   # atomic on POSIX — never leaves a half-written file


def load_train_state(path: str | Path) -> dict[str, Any] | None:
    """Read the resume sidecar; returns None if absent or unreadable."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
