"""Tier-A density sweep pipeline verification tests.

These tests verify the FCD output of 07_generate_density_sweep.py.
All tests are SKIPPED if the density FCD files have not been generated yet
(so CI stays green before the pipeline is run).

Run the pipeline first:
    cd data/sumo && python3 07_generate_density_sweep.py

Then run:
    pytest baselines/tests/test_density_sweep_pipeline.py -v

Simulation specs verified:
  SIM_DURATION = 300s, STEP_LENGTH = 0.1s
  FCD only contains ambulance vehicles (device.fcd.explicit)
  TLS: SUMO-generated via --tls.guess (heuristic, not field-measured)
  Background densities: light=50, medium=200, heavy=500 bg vehicles / 300s
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import pytest

from env.sumo_mobility import (
    SumoMobilityProvider,
    default_fcd_path,
    density_fcd_path,
    default_route_pool,
)

# ---------------------------------------------------------------------------
# Density directory path
# ---------------------------------------------------------------------------

_DATA_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "sumo")
)
_DENSITY_DIR = os.path.join(_DATA_DIR, "density")

# ---------------------------------------------------------------------------
# Skip helpers — skip entire module if Tier-A files are absent
# ---------------------------------------------------------------------------

_TIER_A_DENSITIES = ["light", "medium", "heavy"]
_SMOKE_DENSITIES = ["light_smoke", "medium_smoke", "heavy_smoke"]
_ALL_DENSITIES = _TIER_A_DENSITIES + _SMOKE_DENSITIES
_K_VALUES = [1, 3]


def _fcd_exists(K: int, density: str) -> bool:
    path = os.path.join(_DENSITY_DIR, f"bachmaiHN_mci_k{K}_{density}.fcd.xml")
    return os.path.exists(path)


def _skip_if_missing(K: int, density: str):
    if not _fcd_exists(K, density):
        pytest.skip(
            f"density FCD not found: bachmaiHN_mci_k{K}_{density}.fcd.xml\n"
            "Run: cd data/sumo && python3 07_generate_density_sweep.py"
        )


def _all_tier_a_exist() -> bool:
    return all(_fcd_exists(K, d) for K in _K_VALUES for d in _TIER_A_DENSITIES)


# ---------------------------------------------------------------------------
# 1. File existence
# ---------------------------------------------------------------------------

class TestFCDFilesExist:
    @pytest.mark.parametrize("K", _K_VALUES)
    @pytest.mark.parametrize("density", _TIER_A_DENSITIES)
    def test_tier_a_file_exists(self, K, density):
        _skip_if_missing(K, density)
        path = os.path.join(_DENSITY_DIR, f"bachmaiHN_mci_k{K}_{density}.fcd.xml")
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    @pytest.mark.parametrize("K", _K_VALUES)
    @pytest.mark.parametrize("density", _SMOKE_DENSITIES)
    def test_smoke_file_exists(self, K, density):
        _skip_if_missing(K, density)
        path = os.path.join(_DENSITY_DIR, f"bachmaiHN_mci_k{K}_{density}.fcd.xml")
        assert os.path.exists(path)


# ---------------------------------------------------------------------------
# 2. FCD structure — only ambulance vehicles, no background vehicles
# ---------------------------------------------------------------------------

class TestFCDContainsOnlyAmbulances:
    @pytest.mark.parametrize("K,expected_ids", [
        (1, {"amb_0"}),
        (3, {"amb_0", "amb_1", "amb_2"}),
    ])
    @pytest.mark.parametrize("density", _TIER_A_DENSITIES)
    def test_only_ambulances_in_fcd(self, K, expected_ids, density):
        _skip_if_missing(K, density)
        path = os.path.join(_DENSITY_DIR, f"bachmaiHN_mci_k{K}_{density}.fcd.xml")
        found_ids: set[str] = set()
        for _, elem in ET.iterparse(path, events=["start"]):
            if elem.tag == "vehicle":
                vid = elem.get("id", "")
                if vid.startswith("bg_"):
                    pytest.fail(
                        f"Background vehicle {vid!r} found in FCD — "
                        "--device.fcd.explicit filter may not have worked"
                    )
                found_ids.add(vid)
            elem.clear()
        assert found_ids == expected_ids, (
            f"Expected ambulance IDs {expected_ids}, got {found_ids}"
        )

    @pytest.mark.parametrize("K", _K_VALUES)
    @pytest.mark.parametrize("density", _TIER_A_DENSITIES)
    def test_minimum_timesteps_per_ambulance(self, K, density):
        """Each ambulance must appear in at least 50 timesteps (5s at 0.1s step)."""
        _skip_if_missing(K, density)
        path = os.path.join(_DENSITY_DIR, f"bachmaiHN_mci_k{K}_{density}.fcd.xml")
        amb_ids = ["amb_0"] if K == 1 else ["amb_0", "amb_1", "amb_2"]
        counts: dict[str, int] = {aid: 0 for aid in amb_ids}
        for _, elem in ET.iterparse(path, events=["start"]):
            if elem.tag == "vehicle":
                vid = elem.get("id", "")
                if vid in counts:
                    counts[vid] += 1
            elem.clear()
        for aid, count in counts.items():
            assert count >= 50, (
                f"{aid} has only {count} FCD timesteps in k{K}_{density} "
                "— ambulance may not have reached destination"
            )


# ---------------------------------------------------------------------------
# 3. FCD loads correctly via SumoMobilityProvider
# ---------------------------------------------------------------------------

class TestDensityFCDLoadsViaProvider:
    @pytest.mark.parametrize("K", _K_VALUES)
    @pytest.mark.parametrize("density", _TIER_A_DENSITIES)
    def test_provider_reset_and_step(self, K, density):
        _skip_if_missing(K, density)
        path = density_fcd_path(K, density)
        provider = SumoMobilityProvider(path, K=K, tti_sec=0.0005)
        pos = provider.reset()
        assert pos.shape == (K, 2)
        pos2, vel = provider.step()
        assert pos2.shape == (K, 2)
        assert vel.shape == (K, 2)

    @pytest.mark.parametrize("K", _K_VALUES)
    @pytest.mark.parametrize("density", _TIER_A_DENSITIES)
    def test_positions_within_cell_radius(self, K, density):
        """Ambulance positions must be within 2×R_CELL of gNB."""
        import math
        from utils.config import R_CELL_M
        _skip_if_missing(K, density)
        path = density_fcd_path(K, density)
        provider = SumoMobilityProvider(path, K=K, tti_sec=0.0005)
        provider.reset()
        for _ in range(100):
            pos, _ = provider.step()
            for k in range(K):
                dist = math.sqrt(pos[k, 0] ** 2 + pos[k, 1] ** 2)
                assert dist < 2 * R_CELL_M, (
                    f"k={K} {density}: amb_{k} at distance {dist:.0f}m > 2×{R_CELL_M}m"
                )


# ---------------------------------------------------------------------------
# 4. density_fcd_path() API
# ---------------------------------------------------------------------------

class TestDensityFcdPathAPI:
    def test_invalid_density_raises_valueerror(self):
        with pytest.raises(ValueError, match="density must be one of"):
            density_fcd_path(1, "ultra")

    def test_missing_file_raises_filenotfounderror(self, tmp_path, monkeypatch):
        """density_fcd_path raises FileNotFoundError if file absent."""
        import env.sumo_mobility as sm
        monkeypatch.setattr(sm, "_DATA_DIR", str(tmp_path))
        with pytest.raises(FileNotFoundError, match="07_generate_density_sweep"):
            density_fcd_path(1, "light")


# ---------------------------------------------------------------------------
# 5. default_route_pool() returns density pool when Tier-A files exist
# ---------------------------------------------------------------------------

class TestDefaultRoutePoolUpgrade:
    def test_pool_has_at_least_tier_a_when_complete(self):
        """Pool contains at least the 3 Tier-A base files; may include variants."""
        if not _all_tier_a_exist():
            pytest.skip("Tier-A FCD files not yet generated")
        pool = default_route_pool(1)
        assert len(pool) >= 3, f"Expected ≥3-path pool (base Tier-A), got {len(pool)}"

    def test_pool_falls_back_to_single_when_tier_a_absent(self, tmp_path, monkeypatch):
        """When Tier-A files missing, default_route_pool returns single-path pool."""
        import env.sumo_mobility as sm
        monkeypatch.setattr(sm, "_DATA_DIR", str(tmp_path))
        # Create a fake original FCD file so single-path fallback doesn't error
        fake_fcd = tmp_path / "bachmaiHN_mci_k1.fcd.xml"
        fake_fcd.write_text("<fcd-export/>")
        pool = default_route_pool(1)
        assert len(pool) == 1

    def test_pool_paths_all_exist_when_tier_a_complete(self):
        if not _all_tier_a_exist():
            pytest.skip("Tier-A FCD files not yet generated")
        for K in _K_VALUES:
            pool = default_route_pool(K)
            for p in pool:
                assert os.path.exists(p), f"Pool path not found: {p}"

    def test_variant_files_included_when_present(self, tmp_path, monkeypatch):
        """Pool grows when variant FCD files are present in density/variants/."""
        import env.sumo_mobility as sm

        # Set up a fake data directory with base Tier-A files + 2 variant files
        density_dir = tmp_path / "density"
        variants_dir = density_dir / "variants"
        variants_dir.mkdir(parents=True)

        # Create fake base files (K=1, 3 densities)
        for d in ["light", "medium", "heavy"]:
            (tmp_path / f"bachmaiHN_mci_k1_{d}.fcd.xml").write_text("<fcd/>")
            (density_dir / f"bachmaiHN_mci_k1_{d}.fcd.xml").write_text("<fcd/>")

        # Create fake original single-trace file (fallback path)
        (tmp_path / "bachmaiHN_mci_k1.fcd.xml").write_text("<fcd/>")

        monkeypatch.setattr(sm, "_DATA_DIR", str(tmp_path))
        pool_base_only = sm.default_route_pool(1)
        assert len(pool_base_only) == 3, "Expected 3 base files before variants"

        # Now add 2 variant files
        (variants_dir / "bachmaiHN_mci_k1_v00_light.fcd.xml").write_text("<fcd/>")
        (variants_dir / "bachmaiHN_mci_k1_v00_heavy.fcd.xml").write_text("<fcd/>")

        pool_with_variants = sm.default_route_pool(1)
        assert len(pool_with_variants) == 5, (
            f"Expected 3 base + 2 variants = 5, got {len(pool_with_variants)}"
        )


# ---------------------------------------------------------------------------
# 6. Density effect on kinematics (heavy should not be faster than light)
# ---------------------------------------------------------------------------

class TestDensityKinematicOrdering:
    """Heavy congestion slows ambulances vs light traffic.

    Verified by comparing mean speed magnitude across all timesteps.
    We use a one-sided test: speed_heavy <= speed_light + tolerance.
    Not guaranteed for short traces (stochastic routing can vary), so
    we use a generous tolerance of 2 m/s.
    """
    def test_heavy_not_faster_than_light_k3(self):
        if not (_fcd_exists(3, "light") and _fcd_exists(3, "heavy")):
            pytest.skip("light or heavy FCD for K=3 not yet generated")
        import numpy as np

        def mean_speed(density):
            provider = SumoMobilityProvider(density_fcd_path(3, density), K=3, tti_sec=0.0005)
            provider.reset()
            speeds = []
            for _ in range(500):
                _, vel = provider.step()
                speeds.append(float(np.linalg.norm(vel)))
            return float(np.mean(speeds))

        speed_light = mean_speed("light")
        speed_heavy = mean_speed("heavy")
        assert speed_heavy <= speed_light + 2.0, (
            f"Heavy ({speed_heavy:.2f} m/s) unexpectedly faster than light "
            f"({speed_light:.2f} m/s) by more than 2 m/s tolerance"
        )
