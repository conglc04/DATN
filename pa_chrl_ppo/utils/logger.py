"""Logger wrapper: TensorBoard + WandB + CSV.

Each backend is optional. CSV always writes (cheap, no dependency).
Use:
    logger = Logger("run_42", log_dir="logs")
    logger.log_scalar("reward", 1.0, step=0)
    logger.log_dict({"viol_rate": 1e-4, "eMBB_tput": 25.0}, step=0)
    logger.close()
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class Logger:
    """Lightweight wrapper around TensorBoard + WandB + CSV.

    Lazy-imports backends so the logger works even when wandb/tensorboard
    not installed (degrades to CSV-only).
    """

    def __init__(
        self,
        run_name: str,
        log_dir: str | Path = "logs",
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: str = "pa-chrl-ppo",
        wandb_config: dict[str, Any] | None = None,
        append_csv: bool = False,  # If True, load existing metrics.csv and append
    ) -> None:
        self.run_name = run_name
        self.log_dir = Path(log_dir) / run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # CSV writer: buffer rows in memory so we can rewrite the header
        # if new metric keys appear partway through a run.
        self._csv_path = self.log_dir / "metrics.csv"
        self._csv_fieldnames: list[str] = []
        self._csv_rows: list[dict[str, Any]] = []

        # Resume: load existing CSV rows so new rows are appended on close
        if append_csv and self._csv_path.exists():
            import csv as _csv
            with self._csv_path.open("r", newline="", encoding="utf-8") as f:
                reader = _csv.DictReader(f)
                self._csv_fieldnames = list(reader.fieldnames or [])
                for row in reader:
                    self._csv_rows.append({k: v for k, v in row.items()})

        # TensorBoard
        self._tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb_writer = SummaryWriter(log_dir=str(self.log_dir / "tb"))
            except ImportError:
                print(f"[Logger] tensorboard not available, skipping TB for run={run_name}")

        # WandB
        self._wandb = None
        if use_wandb:
            try:
                import wandb
                self._wandb = wandb.init(
                    project=wandb_project,
                    name=run_name,
                    dir=str(self.log_dir),
                    config=wandb_config or {},
                    reinit=True,
                )
            except ImportError:
                print(f"[Logger] wandb not available, skipping WandB for run={run_name}")
            except Exception as exc:
                print(f"[Logger] WandB init failed ({exc}); continuing without WandB")

    def log_scalar(self, key: str, value: float, step: int) -> None:
        self._csv_write({"step": step, key: value})
        if self._tb_writer is not None:
            self._tb_writer.add_scalar(key, value, step)
        if self._wandb is not None:
            self._wandb.log({key: value}, step=step)

    def log_dict(self, metrics: dict[str, float], step: int) -> None:
        row = {"step": step, **metrics}
        self._csv_write(row)
        if self._tb_writer is not None:
            for k, v in metrics.items():
                self._tb_writer.add_scalar(k, v, step)
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def log_hparams(self, hparams: dict[str, Any]) -> None:
        """Save hyperparameters as JSON next to CSV."""
        path = self.log_dir / "hparams.json"
        path.write_text(json.dumps(hparams, indent=2, default=str), encoding="utf-8")

    def _csv_write(self, row: dict[str, Any]) -> None:
        """Buffer a row, expanding fieldnames if new keys appear.

        Actual file write happens in `_csv_flush()` (called by `close()`).
        We could flush on every row for crash safety, but the rewrite cost is
        O(n) per row if schema is unstable; in practice flushing every N rows
        is a better tradeoff. For Week 1 we just flush on close.
        """
        for key in row.keys():
            if key not in self._csv_fieldnames:
                self._csv_fieldnames.append(key)
        self._csv_rows.append(row)

    def _csv_flush(self) -> None:
        """Write all buffered rows to disk with the union of fieldnames."""
        with self._csv_path.open("w", newline="", encoding="utf-8") as f:
            if not self._csv_fieldnames:
                return
            writer = csv.DictWriter(f, fieldnames=self._csv_fieldnames)
            writer.writeheader()
            for row in self._csv_rows:
                writer.writerow(row)

    def close(self) -> None:
        if self._tb_writer is not None:
            self._tb_writer.flush()
            self._tb_writer.close()
            self._tb_writer = None
        if self._wandb is not None:
            self._wandb.finish()
            self._wandb = None
        self._csv_flush()

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
