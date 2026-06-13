"""Production batch runner: 3 methods × 10 seeds × N episodes.

Default: 5000 episodes with early stopping enabled (eval checkpoint at ep 5000).
Early stopping fires when rolling-mean reward plateaus for 300 episodes after
at least 500 episodes have elapsed — saves compute for fast-converging seeds.

Output: <log-dir>/<algo>_seed<S>/ + summary_30runs.md + eval_ep5000_*.json

Usage:
    python run_30runs.py                          # 30 jobs × 5000 ep, early stop on
    python run_30runs.py --episodes 1000          # shorter run
    python run_30runs.py --no-early-stop          # disable early stopping
    python run_30runs.py --methods pa_chrl_ppo    # single method (10 runs)
    python run_30runs.py --seeds 0,1,2            # subset seeds
    python run_30runs.py --dry-run                # print commands only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
LOG_DIR_DEFAULT = REPO_ROOT / "logs_review_30runs"

DEFAULT_METHODS = ["pa_chrl_ppo", "td3_lag", "sac_lag"]
DEFAULT_SEEDS = list(range(10))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="3 methods × 10 seeds × 500 ep batch runner")
    p.add_argument(
        "--methods",
        type=str,
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated algo names (default: pa_chrl_ppo,td3_lag,sac_lag)",
    )
    p.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="Comma-separated seeds (default: 0..9)",
    )
    p.add_argument("--episodes", type=int, default=5000, help="Episodes per run (default: 5000)")
    p.add_argument("--device", type=str, default="cuda", help="cuda | cpu")
    p.add_argument(
        "--log-dir",
        type=Path,
        default=LOG_DIR_DEFAULT,
        help=f"Output directory (default: {LOG_DIR_DEFAULT})",
    )
    p.add_argument("--dry-run", action="store_true", help="Print commands only, do not execute")
    p.add_argument("--no-summary", action="store_true",
                   help="Skip summary_30runs.md generation after run")
    # Early stopping flags (passed through to train.py)
    p.add_argument("--no-early-stop", action="store_true",
                   help="Disable early stopping (default: early stop ON)")
    p.add_argument("--early-stop-patience", type=int, default=300)
    p.add_argument("--early-stop-min-delta", type=float, default=10.0)
    p.add_argument("--early-stop-window", type=int, default=100)
    p.add_argument("--early-stop-min-ep", type=int, default=500)
    p.add_argument("--eval-at", type=int, default=5000,
                   help="Eval checkpoint milestone episode (default: 5000)")
    # Resume flags
    p.add_argument("--resume-from", type=Path, default=None,
                   help="Log dir of previous run to resume from (load checkpoints + append CSV)")
    p.add_argument("--resume-start-ep", type=int, default=0,
                   help="Episode offset for resume (set to previous n_episodes)")
    return p.parse_args()


def run_single(method: str, seed: int, args: argparse.Namespace) -> tuple[bool, float]:
    """Execute one training run via subprocess. Returns (success, elapsed_sec)."""
    use_early_stop = not args.no_early_stop
    # Resolve checkpoint path for resume
    resume_ckpt = None
    if args.resume_from is not None:
        prefix_map = {
            "pa_chrl_ppo": "pa_chrl_ppo",
            "td3_lag": "td3_lag",
            "sac_lag": "sac_lag",
        }
        ckpt_name = f"{prefix_map.get(method, method)}_seed{seed}_ep{args.resume_start_ep}.pt"
        resume_ckpt = args.resume_from / "checkpoints" / ckpt_name
        if not resume_ckpt.exists():
            # Try checkpoint_dir at root
            resume_ckpt = REPO_ROOT / "checkpoints" / ckpt_name
        if not resume_ckpt.exists():
            print(f"  [WARN] resume checkpoint not found: {ckpt_name}, starting fresh")
            resume_ckpt = None

    cmd = [
        sys.executable, "-X", "utf8",
        str(REPO_ROOT / "train.py"),
        "--algo", method,
        "--episodes", str(args.episodes),
        "--seed", str(seed),
        "--hard",
        "--log-dir", str(args.log_dir),
        "--device", args.device,
        "--checkpoint-every", str(args.resume_start_ep + args.episodes),  # save at final ep
        "--print-every", "100",
        "--eval-at", str(args.eval_at),
        "--early-stop-patience", str(args.early_stop_patience),
        "--early-stop-min-delta", str(args.early_stop_min_delta),
        "--early-stop-window", str(args.early_stop_window),
        "--early-stop-min-ep", str(args.early_stop_min_ep),
        "--resume-start-ep", str(args.resume_start_ep),
    ]
    if resume_ckpt is not None:
        cmd += ["--resume-checkpoint", str(resume_ckpt)]
    if use_early_stop:
        cmd.append("--early-stop")
    if args.dry_run:
        print(f"  [DRY] {' '.join(cmd)}")
        return True, 0.0

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=7200,  # 2-hour cap per run (TD3-Lag at 5000ep needs ~5500s without ES)
        )
        elapsed = time.time() - start
        ok = result.returncode == 0
        if not ok:
            print(f"  [FAIL] {method} seed={seed} returncode={result.returncode}")
            print(f"         stderr tail: {result.stderr[-500:]}")
        return ok, elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"  [TIMEOUT] {method} seed={seed} after {elapsed:.0f}s")
        return False, elapsed
    except Exception as e:
        print(f"  [ERROR] {method} seed={seed}: {e}")
        return False, time.time() - start


def aggregate_summary(args: argparse.Namespace, results: list[dict]) -> str:
    """Generate summary_30runs.md from logs_review_30runs/summary_*.json files."""
    log_dir = args.log_dir
    methods = args.methods.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    per_method: dict[str, list[dict]] = {m: [] for m in methods}
    for method in methods:
        for seed in seeds:
            summary_path = log_dir / f"summary_{method}_seed{seed}.json"
            if summary_path.exists():
                with summary_path.open("r", encoding="utf-8") as f:
                    per_method[method].append(json.load(f))
            else:
                per_method[method].append({})

    use_early_stop = not args.no_early_stop
    md_lines = [
        f"# Training Run Summary (3 × {len(seeds)} × {args.episodes}ep"
        f"{' + EarlyStop' if use_early_stop else ''})",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total runs: {len(methods) * len(seeds)}",
        f"Successful: {sum(1 for r in results if r['ok'])}",
        f"Failed: {sum(1 for r in results if not r['ok'])}",
        f"Total wall-clock: {sum(r['elapsed'] for r in results) / 3600:.2f} hours",
        "",
        "## Per-method aggregate (mean ± std across seeds)",
        "",
        "| Method | ep_reward | mean_e2e_ms | viol_rate | mean_eMBB_mbps | c3_viol_rate | λ_global_mean |",
        "|---|---|---|---|---|---|---|",
    ]

    def stats(values: list[float]) -> str:
        if not values:
            return "—"
        n = len(values)
        mean = sum(values) / n
        if n == 1:
            return f"{mean:.2f}"
        var = sum((x - mean) ** 2 for x in values) / (n - 1)
        std = var ** 0.5
        return f"{mean:.2f} ± {std:.2f}"

    for method in methods:
        runs = [r for r in per_method[method] if r]  # non-empty
        if not runs:
            md_lines.append(f"| {method} | (no data) | | | | | |")
            continue
        rewards = [r.get("ep_reward", 0.0) for r in runs]
        e2es = [r.get("mean_e2e_ms", 0.0) for r in runs]
        viols = [r.get("viol_rate", 0.0) for r in runs]
        embbs = [r.get("mean_embb_mbps", 0.0) for r in runs]
        c3s = [r.get("c3_viol_rate", 0.0) for r in runs]
        lams = [
            (r.get("lambda_global_1", 0.0) + r.get("lambda_global_2", 0.0)
             + r.get("lambda_global_3", 0.0) + r.get("lambda_global_4", 0.0)
             + r.get("lambda_global_5", 0.0)) / 5.0
            for r in runs
        ]
        md_lines.append(
            f"| {method} | {stats(rewards)} | {stats(e2es)} | {stats(viols)} "
            f"| {stats(embbs)} | {stats(c3s)} | {stats(lams)} |"
        )

    md_lines += [
        "",
        "## Verification flags",
        "",
    ]

    # Check no-NaN across all runs
    all_finite = all(
        all(
            isinstance(r.get(k), (int, float))
            and r.get(k) == r.get(k)  # not NaN
            for k in ("ep_reward", "mean_e2e_ms", "viol_rate")
        )
        for runs in per_method.values()
        for r in runs
        if r
    )
    md_lines.append(f"- All runs finite (no NaN): {'✓' if all_finite else '✗'}")

    # Check λ saturation
    from utils.config import LAMBDA_MAX  # noqa: E402
    all_lambda_ok = True
    for method, runs in per_method.items():
        for r in runs:
            if not r:
                continue
            for j in range(1, 6):
                if r.get(f"lambda_global_{j}", 0.0) >= LAMBDA_MAX - 0.01:
                    all_lambda_ok = False
                    md_lines.append(
                        f"  ⚠ {method} λ_{j} = {r[f'lambda_global_{j}']:.3f} "
                        f"(saturating at LAMBDA_MAX={LAMBDA_MAX})"
                    )
    md_lines.append(f"- No λ saturation at LAMBDA_MAX={LAMBDA_MAX}: {'✓' if all_lambda_ok else '✗'}")

    md_lines += [
        "",
        "## Decision (manual review required)",
        "",
        f"- [ ] Results at {args.eval_at}-ep eval checkpoint reviewed",
        "- [ ] Investigate any early-stopped or failed runs",
        "",
        "## Raw run log",
        "",
        "| Idx | Method | Seed | Status | Elapsed (s) |",
        "|---|---|---|---|---|",
    ]
    for i, r in enumerate(results, 1):
        md_lines.append(
            f"| {i:2d} | {r['method']} | {r['seed']} | "
            f"{'OK' if r['ok'] else 'FAIL'} | {r['elapsed']:.0f} |"
        )

    return "\n".join(md_lines) + "\n"


def main() -> int:
    args = parse_args()
    methods = args.methods.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    args.log_dir.mkdir(parents=True, exist_ok=True)

    total = len(methods) * len(seeds)
    print(f"[run_30runs] Plan: {len(methods)} methods × {len(seeds)} seeds × {args.episodes} ep "
          f"= {total} runs to {args.log_dir}")
    print(f"[run_30runs] Methods: {methods}")
    print(f"[run_30runs] Seeds:   {seeds}")
    print(f"[run_30runs] Device:  {args.device}")
    if args.dry_run:
        print("[run_30runs] DRY-RUN mode — printing commands only.\n")

    results: list[dict] = []
    batch_start = time.time()
    idx = 0
    for method in methods:
        for seed in seeds:
            idx += 1
            elapsed_total = time.time() - batch_start
            eta_remaining = (
                (elapsed_total / max(idx - 1, 1)) * (total - idx + 1) if idx > 1 else 0.0
            )
            print(f"\n[{idx}/{total}] {method} seed={seed}  "
                  f"(elapsed {elapsed_total / 60:.1f}m, ETA {eta_remaining / 60:.1f}m)")
            ok, elapsed = run_single(method, seed, args)
            results.append({"method": method, "seed": seed, "ok": ok, "elapsed": elapsed})
            print(f"  -> {'OK' if ok else 'FAIL'} in {elapsed:.0f}s")

    total_wall = time.time() - batch_start
    n_ok = sum(1 for r in results if r["ok"])
    print(f"\n[run_30runs] DONE: {n_ok}/{total} successful in {total_wall / 60:.1f} min")

    if not args.dry_run and not args.no_summary:
        summary_md = aggregate_summary(args, results)
        summary_path = args.log_dir / "summary_30runs.md"
        summary_path.write_text(summary_md, encoding="utf-8")
        print(f"[run_30runs] Summary written: {summary_path}")

    return 0 if n_ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
