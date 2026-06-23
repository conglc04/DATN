"""Point 12 — severity-conditioned PRIMAL-FEASIBILITY verdict (audit 2026-06-23).

`ep_reward` rising/saturating does NOT prove the CMDP is solved: reward can rise
from confounds (eMBB↑ while sacrificing URLLC, severity-mix shift, longer
episodes, reward-scale change, λ still too small, single-seed overfit). The
correct verdict is PRIMAL FEASIBILITY per constraint, CONDITIONED ON severity,
under the DETERMINISTIC policy on held-out episodes — reward is a diagnostic,
NOT a pass criterion (feasibility first, reward second).

Per severity tier:
  C1 mean delay  : mean(D_e2e)  <= D_max^s                       (mean)
  C4 mean AoI    : mean(AoI)    <= AoI_max^s                     (mean)
  C3 eMBB floor  : mean(R_eMBB) >= R_min  ⇔ mean signed gap <= 0 (shortfall)
  C2 delay tail  : Pr[D>D_max^s]   <= eps^s     (Clopper-Pearson 95% upper)
  C5 AoI  tail   : Pr[AoI>AoI_max] <= eps_aoi^s  (Clopper-Pearson 95% upper)

Tail status is THREE-valued: a tail is `pass` only if the 95% upper bound <= eps
AND N >= 3/eps (enough samples to RESOLVE eps — rule of three); if N < 3/eps the
tail is `inconclusive` (we must NOT claim a 1e-5 rate we cannot observe), which
does NOT count as feasible. A tier is FEASIBLE iff C1,C4,C3 hold and C2,C5 are
`pass`; the policy PASSES iff every OBSERVED tier is feasible.

C1/C2/C4/C5 bucket by each ambulance's OWN severity (severity_per_amb[k]); C3 and
the reward/step diagnostic bucket by severity_ref = max(severity_per_amb). Reuses
`clopper_pearson_upper` from eval_tail_bank (same tail statistics, no drift).

Run (random policy = harness validation; a converged policy should show
obs tail << target and all means under budget):
  cd baselines && python3 -m audit.feasibility_eval --episodes 200 --K 3
"""
from __future__ import annotations

import argparse

import numpy as np

from audit.eval_tail_bank import clopper_pearson_upper
from env.oran_env import EnvConfig, ORANEnv
from utils.config import CMDP_D_J_SEVERITY, SEVERITY_QOS


def _tail_status(exc: int, n: int, eps: float) -> tuple[str, float, bool]:
    """Three-valued tail verdict: ('pass'|'fail'|'inconclusive', upper95, resolvable).

    resolvable ⇔ n >= 3/eps (rule of three: a zero-violation certificate at level
    eps needs ~3/eps samples). When not resolvable we return 'inconclusive' — we
    refuse to certify a rate we cannot observe (do NOT claim eps below 3/n).
    """
    resolvable = n >= (3.0 / eps)
    ub = clopper_pearson_upper(int(exc), int(n)) if n > 0 else 1.0
    if not resolvable:
        return "inconclusive", ub, False
    return ("pass" if ub <= eps else "fail"), ub, True


def feasibility_verdict(per_amb: dict, per_ref: dict) -> dict:
    """Pure verdict from pooled accumulators (no env — fully unit-testable).

    per_amb[s] (keyed by per-ambulance severity s) keys:
        delay_sum, delay_ticks, delay_exc, aoi_sum, aoi_ticks, aoi_exc
    per_ref[s] (keyed by severity_ref s) keys:
        embb_gap_sum, embb_steps, reward_sum, reward_steps

    Returns {"per_severity": {s: row}, "overall_pass": bool}. A tier is feasible
    iff C1,C4,C3 are True and C2,C5 status == 'pass'. overall_pass iff every
    OBSERVED tier is feasible (and at least one tier was observed).
    """
    out: dict = {"per_severity": {}, "overall_pass": True}
    for s in range(1, 6):
        row: dict = {}
        pa = per_amb.get(s)
        if pa and pa.get("delay_ticks", 0) > 0:
            d_max = SEVERITY_QOS[s]["D_max"]
            eps = SEVERITY_QOS[s]["eps"]
            aoi_max = SEVERITY_QOS[s]["AoI_max"]
            eps_a = SEVERITY_QOS[s]["eps_aoi"]
            mean_delay = pa["delay_sum"] / pa["delay_ticks"]
            mean_aoi = pa["aoi_sum"] / max(pa["aoi_ticks"], 1)
            row["mean_delay"] = mean_delay
            row["d_max"] = d_max
            row["C1"] = bool(mean_delay <= d_max)
            row["mean_aoi"] = mean_aoi
            row["aoi_max"] = aoi_max
            row["C4"] = bool(mean_aoi <= aoi_max)
            st_d, ub_d, res_d = _tail_status(pa["delay_exc"], pa["delay_ticks"], eps)
            st_a, ub_a, res_a = _tail_status(pa["aoi_exc"], pa["aoi_ticks"], eps_a)
            row.update(eps=eps, tail_delay_ub=ub_d, C2=st_d,
                       eps_aoi=eps_a, tail_aoi_ub=ub_a, C5=st_a)
        pr = per_ref.get(s)
        if pr and pr.get("embb_steps", 0) > 0:
            mean_gap = pr["embb_gap_sum"] / pr["embb_steps"]
            shortfall = max(0.0, mean_gap)
            row["embb_shortfall_mbps"] = shortfall
            row["C3"] = bool(shortfall <= 0.0)
            row["reward_per_step"] = (
                pr["reward_sum"] / pr["reward_steps"] if pr["reward_steps"] else float("nan")
            )
        if not row:
            continue
        mean_ok = all(row.get(c, True) for c in ("C1", "C4", "C3"))
        tail_ok = all(row.get(c, "pass") == "pass" for c in ("C2", "C5") if c in row)
        # A tier with no measured constraints cannot be declared feasible.
        measured_any = any(c in row for c in ("C1", "C4", "C3", "C2", "C5"))
        row["feasible"] = bool(mean_ok and tail_ok and measured_any)
        out["per_severity"][s] = row
        if not row["feasible"]:
            out["overall_pass"] = False
    if not out["per_severity"]:
        out["overall_pass"] = False
    return out


def collect(env_factory, act_fn, *, episodes: int, K: int, seed: int,
            severity: int | None) -> tuple[dict, dict]:
    """Run held-out deterministic episodes, pool samples by severity.

    env_factory(ep_seed) -> fresh ORANEnv; act_fn(obs, env) -> action (deterministic
    for a real eval; random for harness validation). Returns (per_amb, per_ref).
    """
    per_amb = {s: {"delay_sum": 0.0, "delay_ticks": 0.0, "delay_exc": 0.0,
                   "aoi_sum": 0.0, "aoi_ticks": 0.0, "aoi_exc": 0.0} for s in range(1, 6)}
    per_ref = {s: {"embb_gap_sum": 0.0, "embb_steps": 0.0,
                   "reward_sum": 0.0, "reward_steps": 0.0} for s in range(1, 6)}
    for ep in range(episodes):
        env = env_factory(seed + ep)
        opts = None if severity is None else {"severity_per_amb": [severity] * K}
        obs, info = env.reset(seed=seed + ep, options=opts)
        sev = [int(s) for s in info["severity_per_amb"]]
        sev_ref = int(info["severity"])
        done = False
        while not done:
            action = act_fn(obs, env)
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            c = np.asarray(info["c_vec"]); ac = np.asarray(info["active_count_per_amb"])
            for k in range(K):
                n = float(ac[k])
                if n <= 0:
                    continue
                s = sev[k]
                per_amb[s]["delay_sum"] += float(c[k]) * n           # c[k]=mean delay → ×ticks≈sum
                per_amb[s]["delay_ticks"] += n
                per_amb[s]["delay_exc"] += round(float(c[K + k]) * n)
                per_amb[s]["aoi_sum"] += float(c[2 * K + k]) * n
                per_amb[s]["aoi_ticks"] += n
                per_amb[s]["aoi_exc"] += round(float(c[3 * K + k]) * n)
            per_ref[sev_ref]["embb_gap_sum"] += float(c[4 * K])      # signed eMBB gap (Mbps)
            per_ref[sev_ref]["embb_steps"] += 1.0
            per_ref[sev_ref]["reward_sum"] += float(reward)
            per_ref[sev_ref]["reward_steps"] += 1.0
    return per_amb, per_ref


def _print_report(verdict: dict, episodes: int, K: int, policy: str) -> None:
    print("=" * 108)
    print(f"PRIMAL-FEASIBILITY VERDICT (point 12) — {episodes} episodes, K={K}, policy={policy}")
    print("=" * 108)
    hdr = (f"{'sev':<4}{'C1 mean-delay':<22}{'C4 mean-AoI':<22}"
           f"{'C3 eMBB':<14}{'C2 tail':<14}{'C5 tail':<14}{'r/step':<9}{'FEASIBLE':<9}")
    print(hdr); print("-" * 108)
    for s in range(1, 6):
        r = verdict["per_severity"].get(s)
        if not r:
            continue
        c1 = f"{r['mean_delay']*1e3:.2f}/{r['d_max']*1e3:.0f}ms {'OK' if r['C1'] else 'X'}" if "C1" in r else "-"
        c4 = f"{r['mean_aoi']*1e3:.0f}/{r['aoi_max']*1e3:.0f}ms {'OK' if r['C4'] else 'X'}" if "C4" in r else "-"
        c3 = (f"{r['embb_shortfall_mbps']:.1f}Mbps {'OK' if r['C3'] else 'X'}") if "C3" in r else "-"
        c2 = r.get("C2", "-"); c5 = r.get("C5", "-")
        rp = f"{r['reward_per_step']:.3f}" if "reward_per_step" in r else "-"
        feas = "YES" if r.get("feasible") else "NO"
        print(f"{s:<4}{c1:<22}{c4:<22}{c3:<14}{c2:<14}{c5:<14}{rp:<9}{feas:<9}")
    print("=" * 108)
    print(f"OVERALL: {'PASS' if verdict['overall_pass'] else 'FAIL'}  "
          "(feasibility first; reward/step is diagnostic only). "
          "'inconclusive' tail = N<3/eps, NOT a certificate.")


def run(args) -> int:
    K = args.K
    rng = np.random.default_rng(args.seed)
    cfg = EnvConfig(K_ambulances=K, sample_severity=(args.severity is None),
                    initial_severity=args.severity or 5,
                    episode_duration_sec=args.episode_sec)

    def env_factory(ep_seed: int) -> ORANEnv:
        return ORANEnv(cfg, seed=ep_seed)

    def act_random(obs, env):
        return rng.normal(size=env.action_space.shape[0]).astype(np.float32)

    per_amb, per_ref = collect(env_factory, act_random, episodes=args.episodes,
                               K=K, seed=args.seed, severity=args.severity)
    verdict = feasibility_verdict(per_amb, per_ref)
    _print_report(verdict, args.episodes, K, args.policy)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--severity", type=int, default=None, help="fix severity (else stratified random)")
    p.add_argument("--episode-sec", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--policy", type=str, default="random")
    raise SystemExit(run(p.parse_args()))
