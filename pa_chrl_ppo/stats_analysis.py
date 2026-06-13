"""Statistical analysis for the solver sweep: PA-CHRL-PPO vs TD3-Lag vs SAC-Lag.

Tests:
- Mann-Whitney U (non-parametric; appropriate for small n=10, distribution-agnostic)
- Holm-Bonferroni correction across all pairwise-metric comparisons
- Cohen's d (pooled std) + Hedges' g correction for small n
- BH-FDR (secondary)

Primary comparisons: PA vs TD3, PA vs SAC, TD3 vs SAC
Primary metrics: ep_reward, mean_e2e_ms, viol_rate, mean_embb_mbps, c3_viol_rate

Usage:
    python stats_analysis.py
    python stats_analysis.py --log-dir logs_review_30runs --out stats_w11.md
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_CSV_PREFIXES = {
    "pa_chrl_ppo": "pa_chrl_ppo",
    "td3_lag": "smoke_td3_lag",
    "sac_lag": "smoke_sac_lag",
}


def load_metric(log_dir: Path, method: str, metric: str, n_seeds: int = 10) -> list[float]:
    """Load metric from metrics.csv tail-100 (ground truth, consistent for all methods).

    F-A fix (2026-05-29): baseline summary JSONs lacked mean_embb_mbps / c3_viol_rate,
    causing those fields to default to 0. Reading from metrics.csv avoids this entirely.
    """
    import pandas as pd  # local import so module is importable without pandas

    prefix = _CSV_PREFIXES.get(method, method)
    vals = []
    for s in range(n_seeds):
        csv_path = log_dir / f"{prefix}_seed{s}" / "metrics.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if metric in df.columns:
                vals.append(float(df[metric].tail(100).mean()))
            else:
                vals.append(float("nan"))
        else:
            vals.append(float("nan"))
    return vals


METRICS = [
    ("ep_reward",       "Episode reward",           "higher"),
    ("mean_e2e_ms",     "Mean E2E latency (ms)",    "lower"),
    ("viol_rate",       "Violation rate",            "lower"),
    ("mean_embb_mbps",  "Mean eMBB (Mbps)",         "higher"),
    ("c3_viol_rate",    "C3 violation rate",         "lower"),
]

COMPARISONS = [
    ("pa_chrl_ppo", "td3_lag",  "PA vs TD3"),
    ("pa_chrl_ppo", "sac_lag",  "PA vs SAC"),
    ("td3_lag",     "sac_lag",  "TD3 vs SAC"),
]


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def cohens_d(a: list[float], b: list[float]) -> float:
    """Pooled Cohen's d."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = math.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / sp if sp > 0 else float("nan")


def hedges_g(d: float, n1: int, n2: int) -> float:
    """Hedges' g bias correction for small samples."""
    df = n1 + n2 - 2
    if df <= 0:
        return float("nan")
    j = 1 - 3 / (4 * df - 1)
    return d * j


def effect_label(g: float) -> str:
    ag = abs(g)
    if ag >= 2.0:
        return "massive"
    if ag >= 1.2:
        return "very large"
    if ag >= 0.8:
        return "large"
    if ag >= 0.5:
        return "medium"
    if ag >= 0.2:
        return "small"
    return "negligible"


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down correction. Returns adjusted p-values."""
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = p_values[idx] * (n - rank)
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def bh_fdr(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [0.0] * n
    running_min = 1.0
    for rank in range(n - 1, -1, -1):
        idx = order[rank]
        adj = p_values[idx] * n / (rank + 1)
        running_min = min(running_min, adj)
        adjusted[idx] = running_min
    return adjusted


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyse(log_dir: Path, n_seeds: int = 10) -> str:
    # Load all data
    data: dict[str, dict[str, list[float]]] = {}
    for method in set(m for c in COMPARISONS for m in c[:2]):
        data[method] = {}
        for metric, _, _ in METRICS:
            data[method][metric] = load_metric(log_dir, method, metric, n_seeds)

    # Build raw test table
    rows = []
    for cmp_a, cmp_b, cmp_label in COMPARISONS:
        for metric, metric_label, direction in METRICS:
            a = [v for v in data[cmp_a][metric] if not math.isnan(v)]
            b = [v for v in data[cmp_b][metric] if not math.isnan(v)]
            if len(a) < 2 or len(b) < 2:
                rows.append({
                    "comparison": cmp_label, "metric": metric_label,
                    "direction": direction,
                    "mean_a": float("nan"), "mean_b": float("nan"),
                    "U": float("nan"), "p_raw": float("nan"),
                    "d": float("nan"), "g": float("nan"),
                })
                continue
            u_stat, p_raw = sp_stats.mannwhitneyu(a, b, alternative="two-sided")
            d = cohens_d(a, b)
            g = hedges_g(d, len(a), len(b))
            rows.append({
                "comparison": cmp_label, "metric": metric_label,
                "direction": direction,
                "mean_a": float(np.mean(a)), "std_a": float(np.std(a, ddof=1)),
                "mean_b": float(np.mean(b)), "std_b": float(np.std(b, ddof=1)),
                "U": float(u_stat), "p_raw": float(p_raw),
                "d": d, "g": g,
            })

    # Corrections
    p_raws = [r["p_raw"] for r in rows]
    p_holm = holm_bonferroni(p_raws)
    p_bh   = bh_fdr(p_raws)
    for i, r in enumerate(rows):
        r["p_holm"] = p_holm[i]
        r["p_bh"] = p_bh[i]

    # ---------------------------------------------------------------------------
    # Format markdown report
    # ---------------------------------------------------------------------------
    alpha = 0.05
    lines = [
        "# W11 Exp1 — Statistical Analysis",
        "",
        "## Methods",
        "- **Test**: Mann-Whitney U (two-sided, non-parametric)",
        "  - Rationale: n=10 per method; distribution-agnostic (Shapiro-Wilk may fail at small n)",
        "- **Primary correction**: Holm-Bonferroni (α=0.05)",
        "- **Secondary correction**: Benjamini-Hochberg FDR",
        "- **Effect size**: Hedges' g (bias-corrected Cohen's d for small n)",
        "- **Thresholds**: |g| ≥ 0.2 small, ≥ 0.5 medium, ≥ 0.8 large, ≥ 1.2 very large",
        "",
        "## Descriptive Statistics",
        "",
        "| Method | ep_reward | e2e_ms | viol_rate | eMBB_Mbps | c3_viol |",
        "|---|---|---|---|---|---|",
    ]
    for method in ["pa_chrl_ppo", "td3_lag", "sac_lag"]:
        row_vals = []
        for metric, _, _ in METRICS:
            v = [x for x in data[method][metric] if not math.isnan(x)]
            row_vals.append(f"{np.mean(v):.2f} ± {np.std(v, ddof=1):.2f}" if v else "N/A")
        lines.append(f"| {method} | " + " | ".join(row_vals) + " |")

    # Normality check (Shapiro-Wilk)
    lines += ["", "## Normality Check (Shapiro-Wilk, α=0.05)", ""]
    lines.append("| Method | Metric | W | p | Normal? |")
    lines.append("|---|---|---|---|---|")
    for method in ["pa_chrl_ppo", "td3_lag", "sac_lag"]:
        for metric, mlabel, _ in METRICS:
            v = [x for x in data[method][metric] if not math.isnan(x)]
            if len(v) >= 3:
                W, p = sp_stats.shapiro(v)
                lines.append(f"| {method} | {mlabel} | {W:.4f} | {p:.4f} | {'YES' if p > 0.05 else '**NO**'} |")

    # Main results table
    lines += [
        "",
        "## Pairwise Tests — All Results",
        "",
        "| Comparison | Metric | Mean A | Mean B | U | p_raw | p_Holm | p_BH | Hedges g | Effect | Sig? |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if math.isnan(r["p_raw"]):
            continue
        sig = "**YES**" if r["p_holm"] < alpha else "no"
        g_str = f"{r['g']:.3f}" if not math.isnan(r["g"]) else "N/A"
        lines.append(
            f"| {r['comparison']} | {r['metric']} "
            f"| {r['mean_a']:.3f} | {r['mean_b']:.3f} "
            f"| {r['U']:.0f} | {r['p_raw']:.4f} | {r['p_holm']:.4f} | {r['p_bh']:.4f} "
            f"| {g_str} | {effect_label(r['g'])} | {sig} |"
        )

    # Summary of significant results
    sig_rows = [r for r in rows if not math.isnan(r.get("p_holm", float("nan"))) and r["p_holm"] < alpha]
    lines += [
        "",
        f"## Significant Results (Holm-Bonferroni, α={alpha})",
        f"",
        f"{len(sig_rows)} out of {len(rows)} tests are significant after correction.",
        "",
    ]
    if sig_rows:
        lines.append("| Comparison | Metric | p_Holm | Hedges g | Direction |")
        lines.append("|---|---|---|---|---|")
        for r in sig_rows:
            direction = r["direction"]
            a_better = (r["mean_a"] > r["mean_b"]) if direction == "higher" else (r["mean_a"] < r["mean_b"])
            winner = r["comparison"].split(" vs ")[0] if a_better else r["comparison"].split(" vs ")[1]
            lines.append(
                f"| {r['comparison']} | {r['metric']} | {r['p_holm']:.4f} "
                f"| {r['g']:.3f} | {winner} better ({effect_label(r['g'])}) |"
            )

    # PA primary metrics summary
    pa_wins = []
    for r in rows:
        if "PA vs" not in r["comparison"]:
            continue
        if math.isnan(r.get("p_holm", float("nan"))):
            continue
        direction = r["direction"]
        a_better = (r["mean_a"] > r["mean_b"]) if direction == "higher" else (r["mean_a"] < r["mean_b"])
        if r["p_holm"] < alpha and a_better:
            pa_wins.append(r)

    lines += [
        "",
        "## PA-CHRL-PPO Advantage Summary",
        "",
        f"PA-CHRL-PPO significantly outperforms baselines on **{len(pa_wins)}** metric-comparison pairs (Holm, α=0.05):",
        "",
    ]
    for r in pa_wins:
        lines.append(f"- **{r['metric']}** ({r['comparison']}): Hedges g = {r['g']:.3f} ({effect_label(r['g'])}), p_Holm = {r['p_holm']:.4f}")

    lines += [
        "",
        "## Interpretation Notes",
        "",
        "- Report computed per run; do NOT hardcode conclusions. λ-saturation + ordering/",
        "  no-starvation metrics (Table II) interpret severity behaviour for K=3.",
        "- KHÔNG claim 'thắng' nếu CI chồng lấn (06_validation.md); pre-register cặp so sánh.",
    ]

    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", type=Path, default=Path("logs_review_30runs"))
    p.add_argument("--out", type=Path, default=Path("stats_w11.md"))
    p.add_argument("--seeds", type=int, default=10)
    args = p.parse_args()

    report = analyse(args.log_dir, args.seeds)
    args.out.write_text(report, encoding="utf-8")
    print(f"Report written to {args.out}")
    print()
    # Also print to stdout
    print(report)


if __name__ == "__main__":
    main()
