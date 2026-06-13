"""Early stopping for RL training loops.

Stops training when rolling-mean reward plateaus for `patience` episodes
AND at least `min_ep` episodes have elapsed.

Evaluation checkpoints are saved at fixed milestones (`eval_at`) regardless
of whether early stopping fires.

Usage in training loop::

    es = EarlyStopping(patience=300, min_delta=10.0, window=100, min_ep=500)
    for ep in range(n_episodes):
        ...
        should_stop = es.step(ep, ep_reward)
        if should_stop:
            break

    # Save eval snapshot at milestone
    es.maybe_save_eval(ep, metrics, log_dir, eval_at=5000)
"""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np


class EarlyStopping:
    """Rolling-mean plateau detector for RL training.

    Parameters
    ----------
    patience:   Episodes without improvement before stopping. Checked every
                `check_every` episodes; counter incremented by `check_every`
                each time no improvement is detected.
    min_delta:  Minimum reward improvement (absolute) to reset patience.
    window:     Rolling window size for mean reward (episodes).
    min_ep:     Minimum episodes before early stopping can fire.
    check_every: Interval (episodes) between plateau checks.
    """

    def __init__(
        self,
        patience: int = 300,
        min_delta: float = 10.0,
        window: int = 100,
        min_ep: int = 500,
        check_every: int = 50,
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.window = window
        self.min_ep = min_ep
        self.check_every = check_every

        self._buffer: deque[float] = deque(maxlen=window)
        self._best_mean: float = -float("inf")
        self._no_improve_eps: int = 0
        self._triggered: bool = False
        self._trigger_ep: int | None = None
        self._eval_saved: set[int] = set()

    # ------------------------------------------------------------------
    # Core step — call once per episode
    # ------------------------------------------------------------------

    def step(self, ep: int, ep_reward: float) -> bool:
        """Record episode reward and check plateau criterion.

        Returns True when training should stop.
        """
        self._buffer.append(ep_reward)

        if ep < self.min_ep or len(self._buffer) < self.window:
            return False

        if (ep + 1) % self.check_every != 0:
            return False

        mean_r = float(np.mean(self._buffer))

        if mean_r > self._best_mean + self.min_delta:
            self._best_mean = mean_r
            self._no_improve_eps = 0
        else:
            self._no_improve_eps += self.check_every

        if self._no_improve_eps >= self.patience:
            self._triggered = True
            self._trigger_ep = ep + 1
            return True

        return False

    # ------------------------------------------------------------------
    # Eval snapshot — call after every episode to catch milestones
    # ------------------------------------------------------------------

    def maybe_save_eval(
        self,
        ep: int,
        metrics: dict[str, Any],
        log_dir: str | Path,
        eval_at: int = 5000,
        run_name: str = "run",
    ) -> bool:
        """Save eval_ep{N}.json at `eval_at` milestone or when ES triggers.

        Returns True if snapshot was saved this call.
        """
        current_ep = ep + 1  # 1-indexed
        should_save = (
            current_ep == eval_at and eval_at not in self._eval_saved
        ) or (
            self._triggered
            and self._trigger_ep is not None
            and self._trigger_ep not in self._eval_saved
        )
        if not should_save:
            return False

        snap_ep = self._trigger_ep if self._triggered else current_ep
        self._eval_saved.add(snap_ep)

        rolling = list(self._buffer)
        snapshot = {
            "eval_ep": snap_ep,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "early_stopped": self._triggered,
            "rolling_window": len(rolling),
            "rolling_mean_reward": float(np.mean(rolling)) if rolling else float("nan"),
            "rolling_std_reward": float(np.std(rolling)) if rolling else float("nan"),
            "best_mean_reward": self._best_mean,
            "no_improve_eps": self._no_improve_eps,
        }
        snapshot.update(metrics)

        out = Path(log_dir) / f"eval_ep{snap_ep}_{run_name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(snapshot, indent=2, default=float), encoding="utf-8")
        label = "[EARLY-STOP eval]" if self._triggered else "[eval checkpoint]"
        print(
            f"{label} ep={snap_ep}  "
            f"rolling_mean={snapshot['rolling_mean_reward']:+.2f}  "
            f"viol={metrics.get('viol_rate', float('nan')):.4f}  "
            f"-> {out.name}"
        )
        return True

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def triggered(self) -> bool:
        return self._triggered

    @property
    def trigger_ep(self) -> int | None:
        return self._trigger_ep

    @property
    def rolling_mean(self) -> float:
        if not self._buffer:
            return float("nan")
        return float(np.mean(self._buffer))

    def state_dict(self) -> dict:
        return {
            "best_mean": self._best_mean,
            "no_improve_eps": self._no_improve_eps,
            "triggered": self._triggered,
            "trigger_ep": self._trigger_ep,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "window": self.window,
            "min_ep": self.min_ep,
        }
