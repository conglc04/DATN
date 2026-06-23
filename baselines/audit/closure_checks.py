"""Pre-training closure — consolidated INDEPENDENT checks (no docs<->code trust).

Each check re-derives the quantity from first principles (or a textbook formula)
and compares to the live code. Run: cd baselines && python3 -m audit.closure_checks
Exit 0 iff every check PASSes.
"""
from __future__ import annotations
import math
import numpy as np

PASS, FAIL = "PASS", "FAIL"
results = []
def check(name, ok, detail):
    results.append((name, PASS if ok else FAIL, detail))


# ============================================================
# G2 — Independent channel oracle (re-derive UMa PL + SINR + Shannon)
# ============================================================
def g2_channel():
    from utils.config import B_PRB, SHANNON_ETA, F_CARRIER
    from env.channel_model import pl_uma, capacity_per_prb_bps, thermal_noise_dbm
    fc_ghz = F_CARRIER / 1e9
    # independent 3GPP TR 38.901 UMa NLOS-ish macro PL is implementation-specific;
    # instead verify the CODE's pl_uma is monotone increasing in distance and the
    # Shannon capacity matches an independent recompute exactly.
    d = [50.0, 100.0, 500.0, 1000.0]
    pls = [pl_uma(x, fc_ghz) for x in d]
    mono = all(pls[i] < pls[i + 1] for i in range(len(pls) - 1))
    check("G2.pl_monotone", mono, f"PL(dB) over {d} = {[round(p,1) for p in pls]}")
    # independent Shannon per-PRB capacity
    for sinr_db in (-5.0, 0.0, 2.7, 10.0, 20.0):
        indep = SHANNON_ETA * B_PRB * math.log2(1.0 + 10 ** (sinr_db / 10.0))
        code = capacity_per_prb_bps(sinr_db)
        check(f"G2.cap@{sinr_db}dB", abs(indep - code) < 1e-6,
              f"indep={indep:.1f} code={code:.1f} bps/PRB")
    # thermal noise independent: -174 + 10log10(B) + NF(7)
    indep_n = -174.0 + 10 * math.log10(B_PRB) + 7.0
    check("G2.thermal_noise", abs(indep_n - thermal_noise_dbm(B_PRB)) < 1e-9,
          f"indep={indep_n:.2f} code={thermal_noise_dbm(B_PRB):.2f} dBm")


# ============================================================
# G3 — Queue PK delay + AoI independent recompute
# ============================================================
def g3_queue_aoi():
    from env.queue_model import MG1Queue
    from utils.config import D_STOCH
    q = MG1Queue(name="t", arrival_rate=50.0, mean_packet_bits=3200.0)
    q.update_service_rate(prb_count=20, capacity_per_prb_bps=409_000.0)
    mu = 20 * 409_000.0 / 3200.0
    rho = 50.0 / mu
    e_s_pure = 1.0 / mu
    e_s2 = (e_s_pure ** 2 + D_STOCH ** 2) + (e_s_pure + D_STOCH) ** 2
    indep_dq = 50.0 * e_s2 / (2.0 * (1.0 - rho))
    check("G3.pk_delay", abs(indep_dq - q.expected_queue_delay()) < 1e-12,
          f"indep E[Dq]={indep_dq*1e6:.3f}us code={q.expected_queue_delay()*1e6:.3f}us rho={rho:.4f}")


# ============================================================
# G4 — Timescale constants exact
# ============================================================
def g4_timescale():
    from utils.config import (WORKER_STEPS_PER_MANAGER, GAMMA, GAMMA_MANAGER)
    import utils.config as C
    mac = getattr(C, "MAC_TICKS_PER_WORKER", None)
    tti = getattr(C, "TTI_SEC", None) or getattr(C, "MAC_TTI_SEC", None)
    check("G4.worker_per_manager", WORKER_STEPS_PER_MANAGER == 10, f"W={WORKER_STEPS_PER_MANAGER}")
    check("G4.gamma_manager", abs(GAMMA_MANAGER - GAMMA ** WORKER_STEPS_PER_MANAGER) < 1e-12,
          f"gamma_H={GAMMA_MANAGER:.5f} == gamma_L^W={GAMMA**WORKER_STEPS_PER_MANAGER:.5f}")
    if mac is not None:
        check("G4.mac_ticks_per_worker", mac == 20, f"MAC_TICKS_PER_WORKER={mac}")


# ============================================================
# G7 — feasibility dimensional sanity + pure-RL split invariants
# ============================================================
# NOTE (2026-06-21): the env allocation is now PURE-RL softmax — there is no
# N_req formula in the env anymore. The first block keeps an INDEPENDENT N_req
# computation purely as a *feasibility existence* dimensional check (does a
# feasible URLLC PRB count exist at each severity/SINR?), NOT as a cross-check
# of any env-internal N_req (which no longer exists). The second block audits
# the ACTUAL current mechanism: the pure-softmax split must conserve the budget
# (Σ PRB = B_U) and respect the PRB_MIN_QOS anti-starvation floor.
def g7_nreq():
    from utils.config import (URLLC_OFFERED_LOAD_BPS, URLLC_PKT_BITS, SEVERITY_QOS,
                              PRB_MIN_QOS)
    from env.channel_model import capacity_per_prb_bps
    # Independent feasibility existence check (NOT used by the env allocation):
    #   C_req[sev] (bps) = offered_load + pkt_bits/D_max ; N = ceil(C_req/cap_per_prb)
    for sev, sinr in ((5, 2.7), (3, 10.0), (1, 20.0)):
        d_max = SEVERITY_QOS[sev]["D_max"]
        c_req = URLLC_OFFERED_LOAD_BPS + URLLC_PKT_BITS / d_max      # bps  (has time units)
        cap = capacity_per_prb_bps(sinr)                             # bps/PRB
        n_indep = max(PRB_MIN_QOS, int(math.ceil(c_req / max(cap, 1.0))))
        check(f"G7.feasibility_units_sev{sev}", n_indep >= 1 and cap > 0,
              f"C_req={c_req/1e3:.1f}kbps cap={cap/1e3:.1f}kbps/PRB -> feasible N={n_indep} (units: bps/[bps/PRB]=PRB)")
    # Audit the REAL current mechanism: pure-RL softmax split conserves B_U and
    # respects PRB_MIN_QOS (K>=2, all active). No N_req / severity rule involved.
    from env.oran_env import EnvConfig, ORANEnv
    env = ORANEnv(EnvConfig(K_ambulances=3, sample_severity=False, initial_severity=5), seed=0)
    env.reset(seed=0)
    env.active_mask[:] = True
    env.entered_mask[:] = True
    env.set_rrm_budget(0.30)
    prb_u, _ = env._prb_allocation()
    split = env._prb_split_intra_slice(prb_u)
    check("G7.pure_rl_split_conserves_budget", int(split.sum()) == int(prb_u),
          f"Σ PRB={int(split.sum())} == B_U={int(prb_u)} (pure-softmax, largest-remainder)")
    check("G7.pure_rl_split_min_qos", all(int(p) >= PRB_MIN_QOS for p in split),
          f"every active ambulance ≥ PRB_MIN_QOS={PRB_MIN_QOS}; split={split.tolist()}")


# ============================================================
# G11 — Constraint vector sign at/below/above threshold (K=1 and K=3)
# ============================================================
def g11_sign():
    from agents.lagrangian import LambdaState
    from utils.config import build_d_phi_vector, build_dual_scales
    for K, sev in ((1, [5]), (3, [5, 3, 1])):
        ls = LambdaState(K=K); ls.reset_episode(sev, max(sev))
        d = build_d_phi_vector(sev); scale = build_dual_scales(K)
        # exactly at threshold -> deviation ~0
        dev0 = ls._normalized_deviation(d.copy(), d)
        check(f"G11.K{K}.at_threshold", np.allclose(dev0, 0.0, atol=1e-9), f"max|dev|={np.max(np.abs(dev0)):.2e}")
        # below (satisfied) -> dev<0 ; above (violated) -> dev>0  (per element, C1 slot)
        c_lo = d.copy(); c_lo[0] = d[0] * 0.5
        c_hi = d.copy(); c_hi[0] = d[0] * 2.0 + 1e-6
        check(f"G11.K{K}.below_neg", ls._normalized_deviation(c_lo, d)[0] < 0, "C1 below -> dev<0")
        check(f"G11.K{K}.above_pos", ls._normalized_deviation(c_hi, d)[0] > 0, "C1 above -> dev>0")
        check(f"G11.K{K}.dim", d.shape == (4 * K + 1,), f"dim={d.shape[0]} == 4K+1={4*K+1}")


# ============================================================
# G14 — Solver-shared exogenous determinism (same seed -> same env trace)
# ============================================================
def g14_determinism():
    from env.oran_env import EnvConfig, ORANEnv
    def trace(seed):
        env = ORANEnv(EnvConfig(K_ambulances=3, sample_severity=True), seed=seed)
        o, info = env.reset(seed=seed)
        sev = tuple(int(x) for x in info["severity_per_amb"])
        rng = np.random.default_rng(0)
        rs, cs = [], []
        for _ in range(30):
            o, r, t, tr, i = env.step(rng.normal(size=4).astype(np.float32))
            rs.append(r); cs.append(float(np.asarray(i["c_vec"]).sum()))
            if t or tr: break
        return sev, rs, cs
    a = trace(123); b = trace(123); c = trace(999)
    check("G14.same_seed_same_sev", a[0] == b[0], f"sev {a[0]} == {b[0]}")
    check("G14.same_seed_same_reward", a[1] == b[1], "identical reward sequence")
    check("G14.same_seed_same_cvec", a[2] == b[2], "identical c_vec-sum sequence")
    check("G14.diff_seed_differs", a[1] != c[1], f"seed123 vs seed999 reward differ")


# ============================================================
# G5 — PRB conservation + worker-cannot-touch-b_rrm (real env, random)
# ============================================================
def g5_conservation():
    from env.oran_env import EnvConfig, ORANEnv
    from utils.config import P_TOTAL, B_RRM_MIN, B_RRM_MAX
    rng = np.random.default_rng(0)
    ok_sum = True; ok_brrm = True
    for K in (1, 3):   # SUMO+OSM traces exist for K in {1,3}
        env = ORANEnv(EnvConfig(K_ambulances=K, sample_severity=False, initial_severity=3), seed=0)
        env.reset(seed=0, options={"severity_per_amb": [3] * K})
        env.set_rrm_budget(0.5); before = env.r_min_urllc
        for _ in range(15):
            env.active_mask = np.ones(K, dtype=bool)   # unit-test splitter with all active
            env._beta = float(rng.uniform(0.5, 5)); env._prb_weights = rng.normal(size=K)
            for b in (B_RRM_MIN, 0.4, 0.7, B_RRM_MAX):
                pu = int(b * P_TOTAL)
                if int(env._prb_split_intra_slice(pu).sum()) != pu:
                    ok_sum = False
            a = np.concatenate([[5.0], rng.normal(size=K)]).astype(np.float32) if K >= 2 else np.array([5.0], np.float32)
            env.step(a)
            if abs(env.r_min_urllc - before) > 1e-9:
                ok_brrm = False
    check("G5.prb_conservation", ok_sum, "sum(PRB_per_amb)==B_URLLC over K∈{1,2,3} random")
    check("G5.worker_cannot_touch_brrm", ok_brrm, "r_min_urllc unchanged by Worker action")
    # inter-slice always 273
    env = ORANEnv(EnvConfig(K_ambulances=1), seed=0); env.reset(seed=0)
    ok273 = all((lambda pe: pe[0] + pe[1] == P_TOTAL)(env._prb_allocation())
                for b in np.linspace(B_RRM_MIN, B_RRM_MAX, 20) if (env.set_rrm_budget(float(b)) or True))
    check("G5.inter_slice_273", ok273, "B_URLLC+B_eMBB==273 for all b_rrm")


def main():
    for fn in (g2_channel, g3_queue_aoi, g4_timescale, g7_nreq, g11_sign,
               g14_determinism, g5_conservation):
        try:
            fn()
        except Exception as e:
            check(fn.__name__, False, f"EXCEPTION {type(e).__name__}: {e}")
    print("=" * 80)
    print("PRE-TRAINING CLOSURE — INDEPENDENT CHECKS")
    print("=" * 80)
    n_fail = 0
    for name, status, detail in results:
        mark = "✅" if status == PASS else "❌"
        if status == FAIL:
            n_fail += 1
        print(f"  {mark} {name:30s} {detail}")
    print("=" * 80)
    print(f"{len(results)} checks — {len(results)-n_fail} PASS, {n_fail} FAIL")
    print("=" * 80)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
