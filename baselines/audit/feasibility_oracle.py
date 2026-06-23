"""Gate 8 — Feasibility oracle (NO RL, independent physics).

Question answered: for the CMDP as DEFINED (thresholds, 273 PRB, M/G/1 + Shannon),
does there EXIST a PRB allocation that simultaneously satisfies the binding
resource constraints C1 (URLLC mean delay) and C3 (eMBB throughput floor)?

Independence: this script re-implements the queueing/capacity physics from the
textbook formulas. It pulls ONLY the *problem statement* (per-severity D_max,
eMBB floor, P_TOTAL, packet sizes, arrival rates) from utils.config — those
constants ARE the problem definition, not the code under audit. It never calls
ORANEnv / _prb_split_intra_slice / queue_model, so a bug in those cannot mask an
infeasible problem.

Search: brute-force the inter-slice split b in {prb_urllc=0..273}. For each b,
give each active URLLC vehicle its severity-required PRBs first (waterfilling by
descending severity), then check delay; eMBB gets the remainder. Report the best
feasible point and the per-constraint slack.

Run:  python -m audit.feasibility_oracle
"""

from __future__ import annotations

import itertools
import math

from utils.config import (
    B_PRB,
    CMDP_D_J_SEVERITY,
    D_BH,
    D_DET,
    D_FH,
    D_STOCH,
    P_TOTAL,
    SEVERITY_QOS,
    SHANNON_ETA,
)

# ---- Problem constants (from the problem statement) ----
URLLC_ARRIVAL = 50.0            # pkt/s per ambulance
URLLC_BITS = 400 * 8           # 3200 bits
EMBB_BITS = 1500 * 8          # 12000 bits
EMBB_RATE_PER_UE = 1000.0      # pkt/s per bystander UE
RHO_STABLE = 0.9               # engineering stability margin


# ---- Independent physics (textbook, not imported from env) ----
def cap_per_prb_bps(sinr_db: float) -> float:
    """Shannon w/ efficiency: eta * B * log2(1 + 10^(SINR/10))."""
    return SHANNON_ETA * B_PRB * math.log2(1.0 + 10.0 ** (sinr_db / 10.0))


def urllc_mean_delay(prb: int, sinr_db: float) -> float:
    """End-to-end mean delay for one URLLC vehicle (M/G/1 PK + fixed terms).

    Mirrors the DEFINITION (D_e2e = D_det + 1/mu + E[Dq] + D_fh + D_bh), with
    E[Dq] from Pollaczek-Khinchine, recomputed independently here.
    """
    if prb <= 0:
        return math.inf
    cap = cap_per_prb_bps(sinr_db)
    mu = prb * cap / URLLC_BITS
    if mu <= 0:
        return math.inf
    rho = URLLC_ARRIVAL / mu
    if rho >= RHO_STABLE:
        return math.inf  # unstable / over-margin -> treat as infeasible
    e_s_pure = 1.0 / mu
    e_s = e_s_pure + D_STOCH
    var_total = e_s_pure ** 2 + D_STOCH ** 2
    e_s2 = var_total + e_s ** 2
    e_dq = URLLC_ARRIVAL * e_s2 / (2.0 * (1.0 - rho))
    return D_DET + e_s_pure + e_dq + D_FH + D_BH


def embb_throughput_mbps(prb: int, sinr_db: float, n_ue: int) -> float:
    """eMBB total slice throughput (Mbps) = min(arrival, service)*bits/1e6."""
    if prb <= 0:
        return 0.0
    cap = cap_per_prb_bps(sinr_db)
    mu = prb * cap / EMBB_BITS
    lam = EMBB_RATE_PER_UE * n_ue
    return min(lam, mu) * EMBB_BITS / 1e6


def n_req_for(sev: int, sinr_db: float) -> int:
    """Independent minimum PRB so a vehicle's M/G/1 delay <= D_max[sev]."""
    d_max = SEVERITY_QOS[sev]["D_max"]
    for prb in range(1, P_TOTAL + 1):
        if urllc_mean_delay(prb, sinr_db) <= d_max:
            return prb
    return P_TOTAL + 1  # infeasible even with all PRB


def feasibility(severities, sinr_per_veh, n_embb_ue, embb_sinr_db):
    """Search inter-slice split for a point satisfying C1 (all veh) and C3.

    Returns dict with feasible flag, best allocation, and per-constraint slack.
    """
    K = len(severities)
    sev_ref = max(severities)
    embb_floor = CMDP_D_J_SEVERITY[sev_ref]["d3_embb_mbps"]

    best = None
    # prb_urllc: give protected vehicles their N_req (descending severity), rest eMBB
    order = sorted(range(K), key=lambda k: -severities[k])
    n_req = [n_req_for(severities[k], sinr_per_veh[k]) for k in range(K)]
    total_urllc_need = sum(n_req)

    for prb_urllc in range(0, P_TOTAL + 1):
        prb_embb = P_TOTAL - prb_urllc
        # Allocate URLLC PRBs by protection order
        alloc = [0] * K
        budget = prb_urllc
        for k in order:
            take = min(n_req[k], budget)
            alloc[k] = take
            budget -= take
        # C1 check: every vehicle meets its delay target
        delays = [urllc_mean_delay(alloc[k], sinr_per_veh[k]) for k in range(K)]
        c1_ok = all(delays[k] <= SEVERITY_QOS[severities[k]]["D_max"] for k in range(K))
        # C3 check: eMBB floor met
        r_embb = embb_throughput_mbps(prb_embb, embb_sinr_db, n_embb_ue)
        c3_ok = r_embb >= embb_floor
        if c1_ok and c3_ok:
            # slack: how much margin (positive = satisfied)
            c1_slack = min(SEVERITY_QOS[severities[k]]["D_max"] - delays[k] for k in range(K))
            c3_slack = r_embb - embb_floor
            cand = {
                "feasible": True, "prb_urllc": prb_urllc, "prb_embb": prb_embb,
                "alloc": alloc, "delays_ms": [d * 1e3 for d in delays],
                "r_embb": r_embb, "embb_floor": embb_floor,
                "c1_slack_ms": c1_slack * 1e3, "c3_slack_mbps": c3_slack,
            }
            if best is None or c3_slack > best["c3_slack_mbps"]:
                best = cand
    if best is not None:
        return best
    return {
        "feasible": False, "total_urllc_need": total_urllc_need,
        "embb_floor": embb_floor, "n_req": n_req,
        "note": "no inter-slice split satisfies C1(all)+C3 with 273 PRB",
    }


SINR = {"center": 20.0, "mid": 10.0, "edge": 2.7}
LOAD = {"light": 15, "medium": 30, "heavy": 60}
SEV_CASES = {
    "[3,3,3]": [3, 3, 3],
    "[5,3,1]": [5, 3, 1],
    "[5,5,5]": [5, 5, 5],
}


def main():
    print("=" * 100)
    print("GATE 8 — FEASIBILITY ORACLE (independent M/G/1 + Shannon, no RL, no env)")
    print("=" * 100)
    print(f"P_TOTAL={P_TOTAL}  URLLC: {URLLC_ARRIVAL}pkt/s x {URLLC_BITS}b   "
          f"eMBB: {EMBB_RATE_PER_UE}pkt/s/UE x {EMBB_BITS}b")
    print(f"{'SINR per PRB → N_req(sev5)':<40}: "
          f"center={n_req_for(5,20.0)}  mid={n_req_for(5,10.0)}  edge={n_req_for(5,2.7)}")
    print("-" * 100)
    header = (f"{'K_act':<6}{'severity':<10}{'pos':<8}{'load':<8}"
              f"{'feas':<6}{'prbU':<6}{'prbE':<6}{'C1 slack(ms)':<14}"
              f"{'R_eMBB':<10}{'floor':<7}{'C3 slack':<10}")
    print(header)
    print("-" * 100)

    total, feasible = 0, 0
    infeasible_cases = []
    for sev_name, sev in SEV_CASES.items():
        for K_active in (1, 2, 3):
            sub = sev[:K_active]
            for pos, sinr_db in SINR.items():
                for load, n_ue in LOAD.items():
                    sinr_per_veh = [sinr_db] * K_active
                    res = feasibility(sub, sinr_per_veh, n_ue, sinr_db)
                    total += 1
                    if res["feasible"]:
                        feasible += 1
                        print(f"{K_active:<6}{str(sub):<10}{pos:<8}{load:<8}"
                              f"{'YES':<6}{res['prb_urllc']:<6}{res['prb_embb']:<6}"
                              f"{res['c1_slack_ms']:<14.3f}{res['r_embb']:<10.1f}"
                              f"{res['embb_floor']:<7.0f}{res['c3_slack_mbps']:<10.1f}")
                    else:
                        infeasible_cases.append((K_active, sub, pos, load, res))
                        print(f"{K_active:<6}{str(sub):<10}{pos:<8}{load:<8}"
                              f"{'NO':<6}{'-':<6}{'-':<6}{'need=' + str(res['total_urllc_need']):<14}"
                              f"{'-':<10}{res['embb_floor']:<7.0f}{'-':<10}")
    print("-" * 100)
    print(f"FEASIBLE: {feasible}/{total} scenarios "
          f"({100.0*feasible/total:.1f}%)")
    if infeasible_cases:
        print(f"\nINFEASIBLE CASES ({len(infeasible_cases)}):")
        for K_active, sub, pos, load, res in infeasible_cases:
            print(f"  K={K_active} sev={sub} pos={pos} load={load}: "
                  f"URLLC need={res['total_urllc_need']} PRB (have {P_TOTAL}), "
                  f"N_req={res['n_req']}, eMBB floor={res['embb_floor']}")
    print("=" * 100)
    # Main-experiment domain = the sweep working points (edge SINR 2.7 dB, the
    # documented cell-edge point) at all severity/load. Report that subset.
    main_feasible = True
    for sev_name, sev in SEV_CASES.items():
        for K_active in (1, 2, 3):
            res = feasibility(sev[:K_active], [SINR["edge"]] * K_active,
                              LOAD["medium"], SINR["edge"])
            if not res["feasible"]:
                main_feasible = False
    print(f"MAIN-EXPERIMENT DOMAIN (edge 2.7dB, medium load): "
          f"{'ALL FEASIBLE' if main_feasible else 'HAS INFEASIBLE CASES'}")
    print("=" * 100)
    return 0 if main_feasible else 1


if __name__ == "__main__":
    raise SystemExit(main())
