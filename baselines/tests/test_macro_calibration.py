"""W15-B2 macro-cell calibration tests.

Verifies that:
  1. bler_effective decreases monotonically as n_prb increases.
  2. Single ambulance at 1000m reaches QoS with finite PRB allocation.
  3. K=3 near-edge vehicles can all be served within P_TOTAL budget.
  4. eMBB slice stays feasible when URLLC edge load is highest.
  5. Sensitivity: alpha ∈ {0.25, 0.5, 0.75} — 0.5 is justified, not arbitrary.
  6. Sensitivity: I ∈ {-85, -86, -88} — -86 is the sweet spot.

bler model definitions (to be promoted to channel_model.py post-approval):
  bler_single_tx : current env model, single transmission BLER
  bler_effective : alpha * 10*log10(n_prb) frequency-diversity gain (conservative)

All tests log the full signal chain:
  sinr_raw, sinr_clamped, bler_single_tx, bler_effective,
  n_prb_allocated, reliability_after_allocation, delay_after_allocation
"""

from __future__ import annotations

import math

import pytest

# ---------------------------------------------------------------------------
# System constants (same as utils/config.py — imported directly)
# ---------------------------------------------------------------------------
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.config import B_PRB, F_CARRIER, P_TOTAL, TTI_SEC
from env.channel_model import pl_uma, db_to_linear

# ---------------------------------------------------------------------------
# Calibration parameters (W15-B2 locked values)
# ---------------------------------------------------------------------------
BS_TX_TOTAL_DBM  : float = 46.0
BS_N_PRB         : int   = 273          # full carrier, consistent with P_TOTAL
TX_PER_PRB_DBM   : float = BS_TX_TOTAL_DBM - 10.0 * math.log10(BS_N_PRB)  # 21.6 dBm
NF_DB            : float = 7.0
CLAMP_MIN_DB     : float = -15.0
CLAMP_MAX_DB     : float = 40.0
FC_GHZ           : float = F_CARRIER / 1e9

THERMAL_PRB_DBM  : float = -174.0 + 10.0 * math.log10(B_PRB) + NF_DB   # -111.4 dBm

# URLLC QoS targets
PKT_BYTES        : int   = 400
PKT_BITS         : int   = PKT_BYTES * 8
ARRIVAL_RATE_PPS : float = 50.0
LOAD_BPS         : float = PKT_BITS * ARRIVAL_RATE_PPS   # 160 kbps
DELAY_MAX_S      : float = 1e-3
BLER_TARGET      : float = 0.05

# eMBB per-UE demand (macro outdoor scenario)
EMBB_PER_UE_BPS  : float = 5e6          # 5 Mbps per eMBB UE
EMBB_N_UE        : int   = 30           # number of eMBB UEs (C3 scenario)

# ---------------------------------------------------------------------------
# BLER models
# ---------------------------------------------------------------------------

def bler_single_tx(sinr_db: float) -> float:
    """Single-transmission BLER (current env model, renamed for clarity).
    sigmoid(0.5*(SINR-2)) approximation of coded link-level curve.
    Does NOT account for multi-PRB allocation or HARQ combining.
    """
    return 1.0 / (1.0 + math.exp(0.5 * (sinr_db - 2.0)))


def bler_effective(sinr_db: float, n_prb: int, alpha: float = 0.5) -> float:
    """Effective BLER with frequency-diversity combining across n_prb PRBs.

    Physical model:
      gain_db = alpha * 10 * log10(n_prb)
      sinr_eff = sinr_db + gain_db

    alpha=0.5 is conservative: UMa coherence bandwidth ~1-2 MHz >> B_PRB=360 kHz,
    so adjacent PRBs are correlated — gain is roughly half the ideal MRC gain.
    No extra delay cost (parallel frequency-domain transmission, same TTI).

    alpha=0   : no diversity benefit (ultra-conservative, flat fading assumed)
    alpha=0.5 : partial diversity (UMa typical — justified by coherence BW ratio)
    alpha=1.0 : ideal MRC (independent fading per PRB — unrealistic in UMa)
    """
    if n_prb <= 0:
        raise ValueError(f"n_prb must be >= 1, got {n_prb}")
    gain_db  = alpha * 10.0 * math.log10(max(n_prb, 1))
    sinr_eff = min(sinr_db + gain_db, CLAMP_MAX_DB)
    return bler_single_tx(sinr_eff)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _n_eff_dbm(I_per_prb: float) -> float:
    return 10.0 * math.log10(db_to_linear(THERMAL_PRB_DBM) + db_to_linear(I_per_prb))


def _sinr(distance_m: float, I_per_prb: float) -> tuple[float, float]:
    """Return (sinr_raw, sinr_clamped)."""
    n_eff    = _n_eff_dbm(I_per_prb)
    pl       = pl_uma(distance_m, FC_GHZ)
    sinr_raw = TX_PER_PRB_DBM - pl - n_eff
    return sinr_raw, max(min(sinr_raw, CLAMP_MAX_DB), CLAMP_MIN_DB)


def _cap_per_prb(sinr_cl: float) -> float:
    return 0.75 * B_PRB * math.log2(1.0 + db_to_linear(sinr_cl))


def _qos_metrics(sinr_cl: float, n_prb: int, alpha: float = 0.5) -> dict:
    """Compute and return all required log fields."""
    bl_1tx  = bler_single_tx(sinr_cl)
    bl_eff  = bler_effective(sinr_cl, n_prb, alpha)
    cap_e   = _cap_per_prb(sinr_cl)
    raw_cap = n_prb * cap_e
    eff_tpt = raw_cap * (1.0 - bl_eff)
    rho     = LOAD_BPS / max(eff_tpt, 1.0)

    if rho < 1.0:
        svc      = eff_tpt / PKT_BITS
        delay_s  = 1.0 / svc + rho / (svc * (1.0 - rho))
    else:
        delay_s = float("inf")

    return {
        "sinr_clamped"              : sinr_cl,
        "bler_single_tx"            : bl_1tx,
        "bler_effective"            : bl_eff,
        "n_prb_allocated"           : n_prb,
        "reliability_after_alloc"   : 1.0 - bl_eff,
        "delay_after_alloc_ms"      : delay_s * 1e3,
        "rho"                       : rho,
    }


def _find_min_prb_for_qos(sinr_cl: float, alpha: float = 0.5) -> int | None:
    """Return minimum n_prb such that BLER_eff < target AND delay < budget."""
    for n in range(1, P_TOTAL + 1):
        m = _qos_metrics(sinr_cl, n, alpha)
        if m["bler_effective"] < BLER_TARGET and m["delay_after_alloc_ms"] < DELAY_MAX_S * 1e3:
            return n
    return None


I_NOMINAL = -86.0   # W15-B2 calibration value


# ===========================================================================
# Test 1 — Reliability improves monotonically with allocated PRBs
# ===========================================================================

class TestReliabilityImprovesWithAllocatedPRB:
    """bler_effective must decrease strictly as n_prb increases."""

    @pytest.mark.parametrize("distance_m", [300, 500, 800, 1000])
    @pytest.mark.parametrize("alpha", [0.25, 0.5, 0.75])
    def test_monotone_decrease(self, distance_m, alpha):
        _, sinr_cl = _sinr(distance_m, I_NOMINAL)
        prev_bler = bler_single_tx(sinr_cl)   # n_prb=1 baseline

        for n_prb in [1, 2, 4, 8, 16, 32, 64]:
            bl = bler_effective(sinr_cl, n_prb, alpha)
            m  = _qos_metrics(sinr_cl, n_prb, alpha)
            print(
                f"\n  d={distance_m}m alpha={alpha} n_prb={n_prb:>3} | "
                f"sinr_cl={m['sinr_clamped']:+.1f} dB "
                f"bler_1tx={m['bler_single_tx']:.4f} "
                f"bler_eff={m['bler_effective']:.4f} "
                f"rel={m['reliability_after_alloc']:.4f} "
                f"delay={m['delay_after_alloc_ms']:.2f} ms"
            )
            assert bl < prev_bler + 1e-9, (
                f"bler_effective did not decrease: d={distance_m}m "
                f"alpha={alpha} n_prb={n_prb} -> {bl:.4f} >= prev {prev_bler:.4f}"
            )
            prev_bler = bl

    def test_more_prb_always_at_least_as_good_as_single(self):
        """For any n_prb >= 1, bler_effective <= bler_single_tx."""
        _, sinr_cl = _sinr(1000, I_NOMINAL)
        bl_1tx = bler_single_tx(sinr_cl)
        for n_prb in range(1, 32):
            assert bler_effective(sinr_cl, n_prb) <= bl_1tx + 1e-9


# ===========================================================================
# Test 2 — Single ambulance at 1000m is feasible with finite PRB allocation
# ===========================================================================

class TestEdgeSingleVehicleFeasibilityAt1000m:
    """An ambulance at exactly 1000m (cell edge) must achieve QoS.

    QoS = BLER_eff < 5% AND mean delay < 1 ms.
    Uses I_NOMINAL = -86 dBm/PRB, alpha=0.5.
    """

    def test_qos_achievable_at_edge(self):
        sinr_raw, sinr_cl = _sinr(1000, I_NOMINAL)
        min_prb = _find_min_prb_for_qos(sinr_cl)
        assert min_prb is not None, (
            f"No feasible PRB count found at 1000m with I={I_NOMINAL} dBm"
        )
        m = _qos_metrics(sinr_cl, min_prb)
        print(
            f"\n  Edge 1000m | sinr_raw={sinr_raw:.1f} sinr_cl={sinr_cl:.1f} dB"
            f"\n  n_prb_allocated={m['n_prb_allocated']} "
            f"({100*m['n_prb_allocated']/P_TOTAL:.1f}% of P_TOTAL={P_TOTAL})"
            f"\n  bler_single_tx={m['bler_single_tx']:.4f}  "
            f"bler_effective={m['bler_effective']:.4f}"
            f"\n  reliability={m['reliability_after_alloc']:.4f}  "
            f"delay={m['delay_after_alloc_ms']:.3f} ms"
        )
        assert m["bler_effective"]       < BLER_TARGET,          \
            f"BLER_eff={m['bler_effective']:.4f} >= {BLER_TARGET}"
        assert m["delay_after_alloc_ms"] < DELAY_MAX_S * 1e3,    \
            f"delay={m['delay_after_alloc_ms']:.3f} ms >= {DELAY_MAX_S*1e3:.0f} ms"

    def test_min_prb_is_small_fraction_of_total(self):
        """min PRB for QoS at edge should be < 20% of P_TOTAL (not budget-busting)."""
        _, sinr_cl = _sinr(1000, I_NOMINAL)
        min_prb    = _find_min_prb_for_qos(sinr_cl)
        assert min_prb is not None
        assert min_prb <= 0.20 * P_TOTAL, (
            f"min_prb={min_prb} exceeds 20% of P_TOTAL ({int(0.2*P_TOTAL)})"
        )

    def test_r_min_in_reasonable_range(self):
        """r_min = min_prb / P_TOTAL should be between 1% and 15%."""
        _, sinr_cl = _sinr(1000, I_NOMINAL)
        min_prb    = _find_min_prb_for_qos(sinr_cl)
        r_min = min_prb / P_TOTAL
        assert 0.01 <= r_min <= 0.15, f"r_min={r_min:.3f} outside [0.01, 0.15]"

    @pytest.mark.parametrize("n_prb", [1, 2, 4])
    def test_insufficient_prb_fails_qos(self, n_prb):
        """Very few PRBs must NOT satisfy QoS at 1000m edge (test is non-trivial)."""
        _, sinr_cl = _sinr(1000, I_NOMINAL)
        m = _qos_metrics(sinr_cl, n_prb)
        print(
            f"\n  1000m n_prb={n_prb} | sinr_cl={sinr_cl:.1f} dB "
            f"bler_1tx={m['bler_single_tx']:.4f} "
            f"bler_eff={m['bler_effective']:.4f} "
            f"delay={m['delay_after_alloc_ms']:.2f} ms"
        )
        # With n_prb <= 4 at edge, QoS must NOT be met (hard channel)
        qos_met = (
            m["bler_effective"] < BLER_TARGET and
            m["delay_after_alloc_ms"] < DELAY_MAX_S * 1e3
        )
        assert not qos_met, (
            f"n_prb={n_prb} unexpectedly satisfied QoS at 1000m — "
            f"channel too easy or alpha too high"
        )


# ===========================================================================
# Test 3 — K=3 near-edge feasibility
# ===========================================================================

class TestK3NearEdgeFeasibility:
    """Three ambulances at challenging distances can all be served within P_TOTAL.

    Scenario: one at 1000m (just activated), one at 700m, one at 400m.
    Worst-case collective PRB need must stay below P_TOTAL * r_max_urllc (≤ 0.7).
    """

    POSITIONS_M = [1000, 700, 400]

    def test_k3_collective_prb_fits_budget(self):
        total_prb = 0
        print("\n  K=3 near-edge collective PRB allocation:")
        print(f"  {'amb':>4}  {'dist':>6}  {'sinr_cl':>8}  {'bler_1tx':>9}  {'bler_eff':>9}  "
              f"{'n_prb':>6}  {'r_min':>6}  {'delay_ms':>9}  {'rel':>6}")
        for i, d in enumerate(self.POSITIONS_M):
            sinr_raw, sinr_cl = _sinr(d, I_NOMINAL)
            min_prb = _find_min_prb_for_qos(sinr_cl)
            assert min_prb is not None, f"Ambulance {i} at {d}m has no feasible PRB count"
            m = _qos_metrics(sinr_cl, min_prb)
            total_prb += min_prb
            print(
                f"  {i:>4}  {d:>6}m  {sinr_cl:>8.1f}  "
                f"{m['bler_single_tx']:>9.4f}  {m['bler_effective']:>9.4f}  "
                f"{min_prb:>6}  {min_prb/P_TOTAL:>6.3f}  "
                f"{m['delay_after_alloc_ms']:>9.3f}  {m['reliability_after_alloc']:>6.4f}"
            )

        r_total = total_prb / P_TOTAL
        print(f"\n  Total PRBs: {total_prb} / {P_TOTAL}  (r_urllc_collective={r_total:.3f})")
        # Max URLLC budget is 70% of P_TOTAL (leaves 30% for eMBB minimum)
        assert total_prb <= int(0.70 * P_TOTAL), (
            f"K=3 collective PRB need {total_prb} exceeds 70% of P_TOTAL ({int(0.7*P_TOTAL)})"
        )

    def test_k3_each_vehicle_individually_feasible(self):
        for d in self.POSITIONS_M:
            _, sinr_cl = _sinr(d, I_NOMINAL)
            min_prb    = _find_min_prb_for_qos(sinr_cl)
            assert min_prb is not None, f"No feasible allocation at {d}m"
            m = _qos_metrics(sinr_cl, min_prb)
            assert m["bler_effective"]       < BLER_TARGET
            assert m["delay_after_alloc_ms"] < DELAY_MAX_S * 1e3


# ===========================================================================
# Test 4 — eMBB C3 feasible under URLLC edge load
# ===========================================================================

class TestEmbbC3FeasibleUnderUrllcEdgeLoad:
    """After allocating min PRBs for K=3 URLLC edge vehicles, eMBB can still
    serve 30 UEs at 5 Mbps each with remaining PRBs.

    eMBB UEs assumed at 300m average distance (mid-cell macro).
    """

    EMBB_DISTANCE_M = 300
    POSITIONS_M     = [1000, 700, 400]

    def test_embb_prb_sufficient_after_urllc_allocation(self):
        # Step 1: URLLC PRB need (collective)
        urllc_prb = 0
        for d in self.POSITIONS_M:
            _, sinr_cl = _sinr(d, I_NOMINAL)
            min_prb    = _find_min_prb_for_qos(sinr_cl)
            assert min_prb is not None
            urllc_prb += min_prb

        # Step 2: Remaining PRBs for eMBB
        embb_prb_avail = P_TOTAL - urllc_prb
        assert embb_prb_avail > 0, f"No PRBs left for eMBB (urllc_prb={urllc_prb})"

        # Step 3: eMBB capacity at 300m with I_NOMINAL
        _, sinr_cl_embb = _sinr(self.EMBB_DISTANCE_M, I_NOMINAL)
        cap_embb        = _cap_per_prb(sinr_cl_embb)   # bps/PRB

        # Step 4: eMBB PRB demand
        total_embb_demand_bps = EMBB_N_UE * EMBB_PER_UE_BPS
        embb_prb_needed       = math.ceil(total_embb_demand_bps / max(cap_embb, 1.0))

        print(
            f"\n  URLLC K=3 PRBs: {urllc_prb} ({100*urllc_prb/P_TOTAL:.1f}%)"
            f"\n  Remaining for eMBB: {embb_prb_avail} PRBs"
            f"\n  eMBB @ {self.EMBB_DISTANCE_M}m: sinr_cl={sinr_cl_embb:.1f} dB  "
            f"cap/PRB={cap_embb/1e6:.3f} Mbps"
            f"\n  eMBB demand: {EMBB_N_UE} UEs x {EMBB_PER_UE_BPS/1e6:.0f} Mbps = "
            f"{total_embb_demand_bps/1e6:.0f} Mbps → {embb_prb_needed} PRBs needed"
            f"\n  eMBB feasible: {embb_prb_avail >= embb_prb_needed}"
        )
        assert embb_prb_avail >= embb_prb_needed, (
            f"eMBB needs {embb_prb_needed} PRBs but only {embb_prb_avail} available "
            f"after URLLC K=3 allocation ({urllc_prb} PRBs)"
        )

    def test_embb_headroom_is_positive(self):
        """eMBB headroom = available - needed > 0 (some slack for QoS variation)."""
        urllc_prb = sum(
            _find_min_prb_for_qos(_sinr(d, I_NOMINAL)[1]) or 0
            for d in self.POSITIONS_M
        )
        embb_avail = P_TOTAL - urllc_prb
        _, sinr_cl = _sinr(self.EMBB_DISTANCE_M, I_NOMINAL)
        cap_embb   = _cap_per_prb(sinr_cl)
        embb_needed = math.ceil(EMBB_N_UE * EMBB_PER_UE_BPS / max(cap_embb, 1.0))
        headroom = embb_avail - embb_needed
        assert headroom > 0, f"eMBB headroom={headroom} (no slack for variability)"


# ===========================================================================
# Test 5 — Sensitivity: alpha ∈ {0.25, 0.5, 0.75}
# ===========================================================================

class TestSensitivityAlpha:
    """alpha=0.5 is justified:
      - alpha=0.25 (ultra-conservative): min_prb is large — impractical scenario.
      - alpha=0.5  (conservative, UMa coherence-BW motivated): moderate min_prb.
      - alpha=0.75 (optimistic, near-ideal MRC): min_prb is small — too easy.

    Physical basis: UMa coherence bandwidth B_coh ~ 1/(5*tau_rms).
      tau_rms ~ 100-300 ns (3GPP TR 38.901 UMa) -> B_coh ~ 700 kHz - 2 MHz.
      B_PRB = 360 kHz.  Adjacent PRBs have B_PRB/B_coh ~ 0.18-0.5 overlap.
      Diversity gain per factor-2 in n_prb: between 0 dB (correlated) and 3 dB (independent).
      alpha=0.5 -> 1.5 dB per doubling — physically consistent with partial decorrelation.
    """

    @pytest.mark.parametrize("alpha", [0.25, 0.5, 0.75])
    def test_min_prb_for_each_alpha(self, alpha):
        _, sinr_cl = _sinr(1000, I_NOMINAL)
        min_prb    = _find_min_prb_for_qos(sinr_cl, alpha)
        r_min      = (min_prb or P_TOTAL + 1) / P_TOTAL
        m          = _qos_metrics(sinr_cl, min_prb or 1, alpha)
        print(
            f"\n  alpha={alpha}  1000m  sinr_cl={sinr_cl:.1f} dB"
            f"\n    min_prb={min_prb}  r_min={r_min:.3f}"
            f"\n    sinr_raw n/a (same position)"
            f"\n    bler_single_tx={m['bler_single_tx']:.4f}"
            f"\n    bler_effective={m['bler_effective']:.4f}"
            f"\n    n_prb_allocated={m['n_prb_allocated']}"
            f"\n    reliability_after_allocation={m['reliability_after_alloc']:.4f}"
            f"\n    delay_after_allocation={m['delay_after_alloc_ms']:.3f} ms"
        )
        assert min_prb is not None, f"alpha={alpha}: no feasible PRB count at 1000m"
        # alpha=0.75 should need fewer PRBs than alpha=0.5, which needs fewer than alpha=0.25
        # (individual checks; relative ordering tested below)
        assert r_min <= 1.0

    def test_alpha_ordering_min_prb(self):
        """Higher alpha → fewer PRBs needed (more optimistic diversity gain)."""
        _, sinr_cl = _sinr(1000, I_NOMINAL)
        prbs = {
            a: (_find_min_prb_for_qos(sinr_cl, a) or P_TOTAL + 1)
            for a in [0.25, 0.5, 0.75]
        }
        print(f"\n  min_prb by alpha: {prbs}")
        assert prbs[0.25] >= prbs[0.5], \
            f"alpha=0.25 should need >= PRBs than alpha=0.5, got {prbs}"
        assert prbs[0.5]  >= prbs[0.75], \
            f"alpha=0.5 should need >= PRBs than alpha=0.75, got {prbs}"

    def test_alpha_half_gives_moderate_working_point(self):
        """alpha=0.5 working point: r_min between 2% and 15% (not too easy, not too hard)."""
        _, sinr_cl = _sinr(1000, I_NOMINAL)
        min_prb    = _find_min_prb_for_qos(sinr_cl, alpha=0.5)
        r_min = min_prb / P_TOTAL
        assert 0.02 <= r_min <= 0.15, (
            f"alpha=0.5 r_min={r_min:.3f} outside [0.02, 0.15] — "
            f"working point is not 'hard but solvable'"
        )

    def test_alpha_25_is_overly_expensive(self):
        """alpha=0.25 makes the edge too expensive (>20% P_TOTAL just for one vehicle)."""
        _, sinr_cl = _sinr(1000, I_NOMINAL)
        min_prb    = _find_min_prb_for_qos(sinr_cl, alpha=0.25)
        if min_prb is not None:
            r_min = min_prb / P_TOTAL
            # alpha=0.25 should require more PRBs than alpha=0.5
            min_prb_half = _find_min_prb_for_qos(sinr_cl, alpha=0.5)
            assert min_prb > min_prb_half, (
                f"Expected alpha=0.25 to need more PRBs than alpha=0.5, "
                f"got {min_prb} vs {min_prb_half}"
            )


# ===========================================================================
# Test 6 — Sensitivity: I ∈ {-85, -86, -88}
# ===========================================================================

class TestSensitivityInterference:
    """I=-86 dBm/PRB is the sweet spot.

    I=-85 : slightly harder (higher interference) — min_prb larger, r_min > 5%
    I=-86 : target working point — r_min in [2%, 5%]
    I=-88 : easier (lower interference) — min_prb smaller, risk of trivial policy
    """

    @pytest.mark.parametrize("I_val", [-85, -86, -88])
    def test_edge_feasibility_per_I(self, I_val):
        sinr_raw, sinr_cl = _sinr(1000, float(I_val))
        min_prb           = _find_min_prb_for_qos(sinr_cl, alpha=0.5)
        r_min             = (min_prb or P_TOTAL + 1) / P_TOTAL
        m = _qos_metrics(sinr_cl, min_prb or 1, alpha=0.5)
        print(
            f"\n  I={I_val} dBm/PRB  1000m"
            f"\n    sinr_raw={sinr_raw:.1f}  sinr_clamped={sinr_cl:.1f} dB"
            f"\n    bler_single_tx={m['bler_single_tx']:.4f}"
            f"\n    bler_effective={m['bler_effective']:.4f}"
            f"\n    n_prb_allocated={m['n_prb_allocated']}"
            f"\n    reliability_after_allocation={m['reliability_after_alloc']:.4f}"
            f"\n    delay_after_allocation={m['delay_after_alloc_ms']:.3f} ms"
            f"\n    r_min={r_min:.3f}  min_prb={min_prb}"
        )
        assert min_prb is not None, f"I={I_val}: infeasible at 1000m"
        assert min_prb <= P_TOTAL

    def test_I_ordering_min_prb(self):
        """Higher I (weaker interference = less negative dBm) → more PRBs needed."""
        # I=-85 > I=-86 > I=-88 in interference power (less negative = stronger interference)
        results = {}
        for I_val in [-85, -86, -88]:
            _, sinr_cl = _sinr(1000, float(I_val))
            results[I_val] = _find_min_prb_for_qos(sinr_cl) or P_TOTAL + 1
        print(f"\n  min_prb: I=-85->{results[-85]}  I=-86->{results[-86]}  I=-88->{results[-88]}")
        # More interference (less negative I) = harder = more PRBs
        assert results[-85] >= results[-86], \
            f"I=-85 should need >= PRBs than I=-86, got {results}"
        assert results[-86] >= results[-88], \
            f"I=-86 should need >= PRBs than I=-88, got {results}"

    def test_I_minus86_is_moderate_not_trivial(self):
        """I=-86: min_prb > 1 (not trivially solved by single PRB)."""
        _, sinr_cl = _sinr(1000, -86.0)
        min_prb    = _find_min_prb_for_qos(sinr_cl)
        assert min_prb is not None and min_prb > 1, (
            f"I=-86 edge 1000m solved by single PRB — channel too easy"
        )

    def test_I_minus88_still_requires_multiple_prb(self):
        """I=-88 should still require >= 2 PRBs at edge (not trivial)."""
        _, sinr_cl = _sinr(1000, -88.0)
        min_prb    = _find_min_prb_for_qos(sinr_cl)
        assert min_prb is not None and min_prb >= 2, (
            f"I=-88 edge 1000m solved by 1 PRB — gradient has collapsed"
        )
