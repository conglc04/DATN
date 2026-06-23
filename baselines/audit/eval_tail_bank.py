"""Held-out tail-constraint estimator (C2 delay-tail, C5 AoI-tail).

Pools per-ambulance exceedance counts across MANY episodes, keyed by severity, so
the empirical tail probabilities have enough samples to be meaningful — separate
from the short per-episode window used for the dual update (user decision
2026-06-20, Gate-7 Q1). No env/methodology change: the per-tick exceedance counts
are recovered from the info dict (c_vec[K+k]*active_count[k] = exceeding ticks for
ambulance k over its active ticks this Worker step; severity_per_amb[k] is the tier).

Reports, per severity tier: N samples, observed tail rate, rule-of-three 95%
upper bound (Clopper-Pearson when violations>0), and whether N is large enough to
RESOLVE the target epsilon (need N >~ 3/eps for a zero-violation certificate).

Run (random policy = pre-training harness validation):
  cd baselines && python3 -m audit.eval_tail_bank --episodes 50 --K 3
Post-training: load a checkpoint policy instead of random (--policy ckpt path).
"""
from __future__ import annotations
import argparse
import math
import numpy as np

from env.oran_env import EnvConfig, ORANEnv
from utils.config import SEVERITY_QOS


def clopper_pearson_upper(k, n, alpha=0.05):
    """95% upper bound on a binomial rate given k successes in n trials."""
    if n == 0:
        return 1.0
    if k == 0:
        return 1.0 - (alpha) ** (1.0 / n)          # = 1-(0.05)^(1/n) ~ rule-of-three/n
    # Beta inverse via simple bisection (no scipy dependency)
    from math import lgamma
    def betainc_cdf(p, a, b):
        # regularized incomplete beta via continued fraction (Lentz) — adequate here
        if p <= 0: return 0.0
        if p >= 1: return 1.0
        lbeta = lgamma(a) + lgamma(b) - lgamma(a + b)
        front = math.exp(a * math.log(p) + b * math.log(1 - p) - lbeta) / a
        f, c, d = 1.0, 1.0, 0.0
        for i in range(0, 200):
            m = i // 2
            if i == 0: num = 1.0
            elif i % 2 == 0: num = (m * (b - m) * p) / ((a + 2*m - 1) * (a + 2*m))
            else: num = -((a + m) * (a + b + m) * p) / ((a + 2*m) * (a + 2*m + 1))
            d = 1.0 + num * d
            if abs(d) < 1e-30: d = 1e-30
            d = 1.0 / d
            c = 1.0 + num / c
            if abs(c) < 1e-30: c = 1e-30
            f *= d * c
            if abs(1.0 - d * c) < 1e-10: break
        return front * (f - 1.0)
    lo, hi = k / n, 1.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if 1.0 - betainc_cdf(mid, k, n - k + 1) < alpha:   # P(X<=k-1) ... upper tail
            hi = mid
        else:
            lo = mid
    return hi


def run(args):
    K = args.K
    rng = np.random.default_rng(args.seed)
    # per-severity accumulators: [delay_exceed, delay_n, aoi_exceed, aoi_n]
    acc = {s: np.zeros(4) for s in range(1, 6)}
    cfg = EnvConfig(K_ambulances=K, sample_severity=(args.severity is None),
                    initial_severity=args.severity or 5,
                    episode_duration_sec=args.episode_sec)
    for ep in range(args.episodes):
        env = ORANEnv(cfg, seed=args.seed + ep)
        opts = None if args.severity is None else {"severity_per_amb": [args.severity] * K}
        _, info = env.reset(seed=args.seed + ep, options=opts)
        sev = [int(s) for s in info["severity_per_amb"]]
        adim = env.action_space.shape[0]
        done = False
        while not done:
            a = rng.normal(size=adim).astype(np.float32)   # random policy (harness validation)
            _, _, term, trunc, info = env.step(a)
            done = term or trunc
            c = np.asarray(info["c_vec"]); ac = np.asarray(info["active_count_per_amb"])
            for k in range(K):
                n = float(ac[k])
                if n <= 0:
                    continue
                s = sev[k]
                acc[s][0] += round(float(c[K + k]) * n)        # delay exceedances
                acc[s][1] += n
                acc[s][2] += round(float(c[3 * K + k]) * n)    # AoI exceedances
                acc[s][3] += n

    print("=" * 100)
    print(f"HELD-OUT TAIL BANK — {args.episodes} episodes, K={K}, "
          f"{'random severity' if args.severity is None else 'sev='+str(args.severity)}, "
          f"policy=random(harness-validation)")
    print("=" * 100)
    print(f"{'sev':<4}{'constraint':<8}{'N samples':<12}{'#viol':<8}{'obs tail':<12}"
          f"{'95% upper':<12}{'target eps':<12}{'N>=3/eps?':<10}")
    print("-" * 100)
    for s in range(1, 6):
        de, dn, ae, an = acc[s]
        if dn == 0:
            continue
        eps_d = SEVERITY_QOS[s]["eps"]; eps_a = SEVERITY_QOS[s]["eps_aoi"]
        for label, exc, nn, eps in (("C2 delay", de, dn, eps_d), ("C5 AoI", ae, an, eps_a)):
            obs = exc / nn if nn else 0.0
            ub = clopper_pearson_upper(int(exc), int(nn))
            suff = "YES" if nn >= 3.0 / eps else f"no(need {int(3/eps)})"
            print(f"{s:<4}{label:<8}{int(nn):<12}{int(exc):<8}{obs:<12.2e}{ub:<12.2e}"
                  f"{eps:<12.0e}{suff:<10}")
    print("=" * 100)
    print("Note: rates above use a RANDOM policy (validates the bank harness). Post-training,"
          " re-run with a trained checkpoint; a converged policy should show obs tail << target.")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--severity", type=int, default=None, help="fix severity (else random/stratified)")
    p.add_argument("--episode-sec", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--policy", type=str, default="random")
    raise SystemExit(run(p.parse_args()))
