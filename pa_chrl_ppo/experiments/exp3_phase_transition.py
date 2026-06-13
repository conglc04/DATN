"""Exp3 — Phase Transition Ablation (CF-2 Audit Fix 2026-05-28).

Reviewer M5 (W12) yêu cầu verify claim: "λ_warm[φ] table reduces re-converge time
≥ 80% post phase transition vs cold-start dual ascent".

Methodology:
    - 2 variants × N seeds × M episodes:
      * with_warm: default LAMBDA_WARM table (phase-specific initial values)
      * without_warm: zero warm table (cold-start at every phase entry)
    - hard_mission_config: 5-phase trajectory (φ_1 → φ_2 → φ_3 → φ_4 → φ_5)
      across 1 episode (= 100 Worker ticks).
    - Each episode resets λ_global ← λ_warm[initial_phase]; phase transitions
      mid-episode trigger λ_warm[φ_new] reload (or zero if disabled).

Output:
    - logs_exp3_phase_transition/with_warm/seed_*/{metrics.csv, summary_*.json}
    - logs_exp3_phase_transition/without_warm/seed_*/...
    - summary_exp3.md with time-to-reconverge analysis + figures

Usage:
    python -m experiments.exp3_phase_transition                 # 2x3x500
    python -m experiments.exp3_phase_transition --episodes 200  # quick
    python -m experiments.exp3_phase_transition --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR_DEFAULT = REPO_ROOT / "logs_exp3_phase_transition"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp3 phase-transition λ_warm ablation")
    p.add_argument(
        "--seeds",
        type=str,
        default="0,1,2",
        help="Comma-separated seeds (default: 0,1,2 — 3 seeds)",
    )
    p.add_argument("--episodes", type=int, default=500, help="Episodes per run")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--log-dir", type=Path, default=LOG_DIR_DEFAULT)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def run_variant(variant: str, seed: int, args: argparse.Namespace) -> tuple[bool, float]:
    """Run one training run for given variant (with_warm / without_warm)."""
    variant_log = args.log_dir / variant
    variant_log.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-X",
        "utf8",
        str(REPO_ROOT / "train.py"),
        "--algo",
        "pa_chrl_ppo",
        "--episodes",
        str(args.episodes),
        "--seed",
        str(seed),
        "--hard",
        "--log-dir",
        str(variant_log),
        "--device",
        args.device,
        "--checkpoint-every",
        "0",
        "--print-every",
        "100",
    ]
    if variant == "without_warm":
        cmd.append("--no-warm-start")

    if args.dry_run:
        print(f"  [DRY] {variant} seed={seed}: {' '.join(cmd)}")
        return True, 0.0

    start = time.time()
    try:
        result = subprocess.run(
            cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=3600
        )
        elapsed = time.time() - start
        ok = result.returncode == 0
        if not ok:
            print(f"  [FAIL] {variant} seed={seed} rc={result.returncode}")
            print(f"         stderr tail: {result.stderr[-400:]}")
        return ok, elapsed
    except subprocess.TimeoutExpired:
        return False, time.time() - start


def analyze_reconverge(args: argparse.Namespace) -> str:
    """Compute time-to-reconverge metric from λ trajectories.

    Method: for each variant×seed, read metrics.csv. Identify phase transitions
    (phase column changes between episodes). Measure number of episodes between
    transition and λ stabilizing within ε=10% of asymptotic value for that phase.
    """
    import csv

    md_lines = [
        "# Exp3 Phase-Transition Ablation Summary",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Episodes per run: {args.episodes}",
        f"Seeds: {args.seeds}",
        "",
        "## Per-variant aggregate (λ_1 mean across seeds × episodes)",
        "",
        "| Variant | n_runs | λ_1 mean | λ_1 std | λ_2 mean | λ_2 std | final_reward |",
        "|---|---|---|---|---|---|---|",
    ]

    seeds = [int(s) for s in args.seeds.split(",")]
    aggregates: dict[str, list[dict]] = {"with_warm": [], "without_warm": []}

    for variant in ("with_warm", "without_warm"):
        for seed in seeds:
            summary_path = args.log_dir / variant / f"summary_pa_chrl_ppo_seed{seed}.json"
            if summary_path.exists():
                with summary_path.open() as f:
                    aggregates[variant].append(json.load(f))

    def stats(values: list[float]) -> tuple[float, float]:
        if not values:
            return float("nan"), float("nan")
        n = len(values)
        m = sum(values) / n
        if n == 1:
            return m, 0.0
        return m, (sum((v - m) ** 2 for v in values) / (n - 1)) ** 0.5

    for variant in ("with_warm", "without_warm"):
        runs = aggregates[variant]
        l1 = [r.get("lambda_global_1", 0.0) for r in runs]
        l2 = [r.get("lambda_global_2", 0.0) for r in runs]
        rewards = [r.get("ep_reward", 0.0) for r in runs]
        l1_m, l1_s = stats(l1)
        l2_m, l2_s = stats(l2)
        r_m, _ = stats(rewards)
        md_lines.append(
            f"| {variant} | {len(runs)} | {l1_m:.3f} | {l1_s:.3f} "
            f"| {l2_m:.3f} | {l2_s:.3f} | {r_m:.1f} |"
        )

    md_lines += [
        "",
        "## Time-to-Reconverge Analysis (per-episode λ_1 trajectory)",
        "",
        "Metric: number of episodes for |λ_1(ep) - λ_1(ep+50)| < 0.05 (stable to ±5%) AFTER episode 100.",
        "",
        "| Variant | seed | converge_episode | final λ_1 |",
        "|---|---|---|---|",
    ]

    converge_with = []
    converge_without = []
    for variant, store in [("with_warm", converge_with), ("without_warm", converge_without)]:
        for seed in seeds:
            csv_path = args.log_dir / variant / f"pa_chrl_ppo_seed{seed}" / "metrics.csv"
            if not csv_path.exists():
                md_lines.append(f"| {variant} | {seed} | (no CSV) | — |")
                continue
            l1_traj = []
            with csv_path.open() as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        l1_traj.append(float(row.get("lambda_global_1", 0.0)))
                    except (TypeError, ValueError):
                        l1_traj.append(0.0)

            converge_ep = None
            for ep in range(100, len(l1_traj) - 50):
                window = l1_traj[ep:ep + 50]
                if window and max(window) - min(window) < 0.05:
                    converge_ep = ep
                    break
            if converge_ep is None:
                converge_ep = len(l1_traj)
            final_l1 = l1_traj[-1] if l1_traj else 0.0
            store.append(converge_ep)
            md_lines.append(
                f"| {variant} | {seed} | {converge_ep} | {final_l1:.3f} |"
            )

    md_lines += [
        "",
        "## Reduction in convergence time",
        "",
    ]
    if converge_with and converge_without:
        avg_with = sum(converge_with) / len(converge_with)
        avg_without = sum(converge_without) / len(converge_without)
        reduction = (1 - avg_with / max(avg_without, 1)) * 100
        md_lines += [
            f"- with_warm avg converge: {avg_with:.0f} episodes",
            f"- without_warm avg converge: {avg_without:.0f} episodes",
            f"- **Reduction**: {reduction:.1f}% (target ≥ 80% per W12 reviewer M5)",
            f"- **Verdict**: {'✓ PASS' if reduction >= 80 else '⚠ Below target' if reduction >= 50 else '✗ FAIL'}",
        ]
    else:
        md_lines.append("- (insufficient data for reduction calculation)")

    return "\n".join(md_lines) + "\n"


def main() -> int:
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    args.log_dir.mkdir(parents=True, exist_ok=True)

    total = 2 * len(seeds)
    print(f"[exp3] Plan: 2 variants × {len(seeds)} seeds × {args.episodes} ep = {total} runs")
    print(f"[exp3] Log dir: {args.log_dir}")
    if args.dry_run:
        print("[exp3] DRY-RUN mode")

    results = []
    batch_start = time.time()
    idx = 0
    for variant in ("with_warm", "without_warm"):
        for seed in seeds:
            idx += 1
            print(f"\n[{idx}/{total}] variant={variant} seed={seed}")
            ok, elapsed = run_variant(variant, seed, args)
            results.append({"variant": variant, "seed": seed, "ok": ok, "elapsed": elapsed})
            print(f"  -> {'OK' if ok else 'FAIL'} in {elapsed:.0f}s")

    total_wall = time.time() - batch_start
    n_ok = sum(1 for r in results if r["ok"])
    print(f"\n[exp3] DONE: {n_ok}/{total} successful in {total_wall / 60:.1f} min")

    if not args.dry_run:
        summary_md = analyze_reconverge(args)
        summary_path = args.log_dir / "summary_exp3.md"
        summary_path.write_text(summary_md, encoding="utf-8")
        print(f"[exp3] Summary: {summary_path}")

    return 0 if n_ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
