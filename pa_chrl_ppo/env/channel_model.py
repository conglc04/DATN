"""3GPP channel models for O-RAN simulator (Hanoi hybrid).

Implements:
    - UMa  (Urban Macro)   — vành đai, trục đường lớn (3GPP TR 38.901)
    - UMi  (Urban Micro Street Canyon) — phố cổ Hà Nội (LOS + NLOS branches)
    - SINR computation + Shannon capacity per PRB

Reference:
    - docs/03_architecture.md#channel-model
    - 3GPP TR 38.901 v17 Section 7.4 (path loss models)
    - 3GPP TR 38.901 Table 7.4.2-1 (LOS probability)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from utils.config import B_PRB, F_CARRIER, SHANNON_ETA, TX_POWER_UE_DBM

# ============================================================
# Path loss models (3GPP TR 38.901)
# ============================================================


def pl_uma(distance_m: float, f_c_ghz: float = F_CARRIER / 1e9) -> float:
    """UMa path loss in dB.

    Reference: docs/03_architecture.md:116
        PL_UMa(dB) = 28 + 22·log10(d) + 20·log10(f_c)

    Note: This is a simplified UMa formula (3GPP TR 38.901 uses 3D distance
    and has LOS/NLOS branches; the docs choose the LOS-equivalent compact form).

    Args:
        distance_m: Horizontal distance UE-BS (meters), must be > 0.
        f_c_ghz:    Carrier frequency in GHz (default = system carrier).
    """
    if distance_m <= 0:
        raise ValueError(f"distance must be positive, got {distance_m}")
    return 28.0 + 22.0 * math.log10(distance_m) + 20.0 * math.log10(f_c_ghz)


def pl_umi_los(distance_m: float, f_c_ghz: float = F_CARRIER / 1e9) -> float:
    """UMi Street Canyon — LOS path loss in dB.

    Reference: docs/03_architecture.md:123
        PL_UMi_LOS(dB) = 32.4 + 21·log10(d_3D) + 20·log10(f_c)
        Valid: d ≤ 5 km
    """
    if distance_m <= 0:
        raise ValueError(f"distance must be positive, got {distance_m}")
    if distance_m > 5000:
        # Outside spec range — caller should switch to UMa for distances > 5km
        pass
    return 32.4 + 21.0 * math.log10(distance_m) + 20.0 * math.log10(f_c_ghz)


def pl_umi_nlos(
    distance_m: float, h_ue_m: float = 1.5, f_c_ghz: float = F_CARRIER / 1e9
) -> float:
    """UMi Street Canyon — NLOS path loss in dB.

    Reference: docs/03_architecture.md:124
        PL_UMi_NLOS(dB) = 22.4 + 35.3·log10(d_3D) + 21.3·log10(f_c) - 0.3·(h_UE - 1.5)
    """
    if distance_m <= 0:
        raise ValueError(f"distance must be positive, got {distance_m}")
    return (
        22.4
        + 35.3 * math.log10(distance_m)
        + 21.3 * math.log10(f_c_ghz)
        - 0.3 * (h_ue_m - 1.5)
    )


def los_probability_umi(distance_2d_m: float) -> float:
    """UMi LOS probability per 3GPP TR 38.901 Table 7.4.2-1.

    P_LOS(d_2D) = min(18/d, 1) · (1 - exp(-d/36)) + exp(-d/36)
    """
    if distance_2d_m <= 0:
        return 1.0
    return min(18.0 / distance_2d_m, 1.0) * (
        1.0 - math.exp(-distance_2d_m / 36.0)
    ) + math.exp(-distance_2d_m / 36.0)


# ============================================================
# Shadow fading (log-normal)
# ============================================================

SHADOW_SIGMA_UMA_DB: float = 4.0          # 3GPP TR 38.901 UMa LOS
SHADOW_SIGMA_UMI_LOS_DB: float = 4.0      # UMi LOS
SHADOW_SIGMA_UMI_NLOS_DB: float = 7.82    # UMi NLOS


def sample_shadow_fading(scenario: Literal["UMa", "UMi_LOS", "UMi_NLOS"], rng: np.random.Generator) -> float:
    """Sample log-normal shadow fading offset in dB (zero-mean Gaussian)."""
    sigma = {
        "UMa": SHADOW_SIGMA_UMA_DB,
        "UMi_LOS": SHADOW_SIGMA_UMI_LOS_DB,
        "UMi_NLOS": SHADOW_SIGMA_UMI_NLOS_DB,
    }[scenario]
    return float(rng.normal(0.0, sigma))


# ============================================================
# Composite channel model
# ============================================================


@dataclass(slots=True)
class BaseStation:
    """O-RAN cell / O-RU placement."""

    cell_id: int
    x: float
    y: float
    h: float = 25.0                           # antenna height (m)
    layer: Literal["macro", "micro"] = "micro"
    tx_power_dbm: float = 46.0                # gNB TX power (DL)


@dataclass(slots=True)
class ChannelModel:
    """Compute path loss + SINR for UE-BS link.

    Strategy:
        - BS in macro layer → UMa
        - BS in micro layer → UMi-Street Canyon with LOS/NLOS sampling
        - Optional log-normal shadow fading
    """

    f_c_ghz: float = F_CARRIER / 1e9
    h_ue_m: float = 1.5
    shadowing: bool = True
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def path_loss(self, ue_pos: tuple[float, float], bs: BaseStation) -> tuple[float, str]:
        """Return (PL_dB, scenario_label)."""
        d = math.hypot(ue_pos[0] - bs.x, ue_pos[1] - bs.y)
        d = max(d, 1.0)  # guard against d=0

        if bs.layer == "macro":
            pl = pl_uma(d, self.f_c_ghz)
            scenario = "UMa"
        else:
            p_los = los_probability_umi(d)
            if self.rng.random() < p_los:
                pl = pl_umi_los(d, self.f_c_ghz)
                scenario = "UMi_LOS"
            else:
                pl = pl_umi_nlos(d, self.h_ue_m, self.f_c_ghz)
                scenario = "UMi_NLOS"

        if self.shadowing:
            pl += sample_shadow_fading(scenario, self.rng)  # type: ignore[arg-type]

        return pl, scenario

    def receive_power_dbm(
        self,
        ue_pos: tuple[float, float],
        bs: BaseStation,
        tx_power_dbm: float | None = None,
    ) -> float:
        """RX power in dBm (linear-in-dB)."""
        pl, _ = self.path_loss(ue_pos, bs)
        p_tx = bs.tx_power_dbm if tx_power_dbm is None else tx_power_dbm
        return p_tx - pl


# ============================================================
# Noise + SINR + Shannon capacity
# ============================================================


def thermal_noise_dbm(bandwidth_hz: float, noise_figure_db: float = 7.0) -> float:
    """Thermal noise: −174 dBm/Hz + 10·log10(B) + NF.

    Default NF = 7 dB: typical gNB receiver per 3GPP TR 38.101-4 Table A.2.2-1
    (base station NF = 7 dB for FR1). See REFERENCE_MAP §2.
    """
    return -174.0 + 10.0 * math.log10(bandwidth_hz) + noise_figure_db


def db_to_linear(db: float) -> float:
    return 10.0 ** (db / 10.0)


def sinr_db(rx_power_dbm: float, interference_dbm: float, noise_dbm: float) -> float:
    """SINR in dB given signal (dBm) and interference + noise (dBm)."""
    signal_lin = db_to_linear(rx_power_dbm - 30.0)              # to Watts (or just any consistent unit)
    interference_lin = db_to_linear(interference_dbm - 30.0)
    noise_lin = db_to_linear(noise_dbm - 30.0)
    snr_lin = signal_lin / max(interference_lin + noise_lin, 1e-30)
    return 10.0 * math.log10(snr_lin)


def capacity_per_prb_bps(sinr_db_val: float, eta: float = SHANNON_ETA) -> float:
    """Shannon capacity per PRB in bps.

    Reference: docs/03_architecture.md:175
        C_k = B_PRB · log2(1 + SINR)   với B_PRB = 360 kHz @ μ=1
    eta is realistic MCS efficiency (≈ 0.75) vs theoretical Shannon.
    """
    sinr_lin = db_to_linear(sinr_db_val)
    return eta * B_PRB * math.log2(1.0 + sinr_lin)


def aggregate_capacity_bps(prb_count: int, sinr_db_val: float, eta: float = SHANNON_ETA) -> float:
    """Total throughput across n PRBs (Mbps if /1e6)."""
    return prb_count * capacity_per_prb_bps(sinr_db_val, eta)
