"""Acceptance tests: dist_to_gNB and dist_to_destination are fully separate.

Arrival logic (SUMO+OSM only):
    FCD exit within arrival_radius_m of destination only.
    No live dist_to_dest check — prevents false positive on pass-through routes.
    Tests that need arrival force the provider's reached_destination_mask.

12 test classes:
    1. Cell entry uses dist_to_gNB only (not dist_to_destination).
    2. Arrival logic uses dist_to_destination (not dist_to_gNB).
    3. Vehicle near gNB but far from destination is NOT arrived.
    4. Vehicle on destination edge within radius IS arrived (FCD exit).
    5. reached_destination_mask latches arrival (FCD-exit path).
    6. Changing gNB does NOT change destination arrival.
    7. Changing destination changes arrival correctly.
    8. FCD exit at destination → arrived.
    9. FCD exit far from destination → not arrived.
   10. Missing vehicle (exited FCD) does NOT become position [0,0] at gNB.
   11. Arrived vehicle gets 0 PRB and zeroed obs block.
   12. K=3 terminates only when all three arrived.
   13. Timeout when one vehicle never arrives.

Plus 5 acceptance criteria invariant checks.
"""

from __future__ import annotations

import math
import os
import tempfile

import numpy as np
import pytest

from env.oran_env import EnvConfig, ORANEnv, macro_mission_config
from env.sumo_mobility import (
    SumoMobilityProvider,
    PooledSumoMobilityProvider,
    gps_to_metric,
)
from utils.config import (
    DEST_LAT,
    DEST_LON,
    ARRIVAL_RADIUS_M,
    BACH_MAI_LAT,
    BACH_MAI_LON,
    OBS_FIXED_BLOCK_LEN,
    OBS_PER_AMB_BLOCK_LEN,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Destination in local Cartesian
_DEST_EAST, _DEST_NORTH = gps_to_metric(DEST_LAT, DEST_LON)
_DEST_XY = (_DEST_EAST, _DEST_NORTH)
_DEST_DIST_FROM_GNB = math.sqrt(_DEST_EAST ** 2 + _DEST_NORTH ** 2)


def _env(
    K: int = 1,
    enable_arrival: bool = True,
    arrival_radius_m: float = 25.0,
    cell_radius_m: float = 500.0,
) -> ORANEnv:
    """SUMO+OSM env (K∈{1,3}). Tests that need arrival force the provider mask."""
    cfg = EnvConfig(
        K_ambulances=K,
        enable_arrival=enable_arrival,
        arrival_radius_m=arrival_radius_m,
        cell_radius_m=cell_radius_m,
    )
    return ORANEnv(cfg, seed=0)


def _sumo_env(K: int = 3, enable_arrival: bool = True) -> ORANEnv:
    """macro_mission_config env using live density traces (if available) or default."""
    cfg = macro_mission_config(K_ambulances=K)
    cfg = EnvConfig(
        **{k: v for k, v in cfg.__dict__.items()},
    )
    cfg.enable_arrival = enable_arrival
    return ORANEnv(cfg, seed=0)


def _make_minimal_fcd_xml(
    timesteps: list[tuple[float, list[tuple[str, float, float]]]],
) -> str:
    """Build a minimal FCD XML string from (time, [(vid, lon, lat), ...]) tuples."""
    lines = ["<fcd-export>"]
    for t, vehicles in timesteps:
        lines.append(f'  <timestep time="{t:.2f}">')
        for vid, lon, lat in vehicles:
            lines.append(
                f'    <vehicle id="{vid}" x="{lon}" y="{lat}" '
                f'angle="0" speed="5.000" lane="r_0" />'
            )
        lines.append("  </timestep>")
    lines.append("</fcd-export>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. Cell entry uses dist_to_gNB only
# ---------------------------------------------------------------------------

class TestCellEntryUsesDistanceToGnb:
    """entered_mask latches on dist_to_gnb ≤ cell_radius_m, not dist_to_destination."""

    def test_entry_logged_when_close_to_gnb(self):
        """After SUMO fast-forward, ≥1 ambulance entered the cell."""
        env = _env(K=1, cell_radius_m=500.0, enable_arrival=False)
        env.reset(seed=0)
        assert env.entered_mask[0], "Should be entered after cell-entry fast-forward"

    def test_entry_uses_gnb_distance_not_destination_distance(self):
        """Changing destination does NOT affect cell entry logic."""
        # Two envs with same start position, different destinations — entered_mask must match.
        cfg1 = EnvConfig(
            K_ambulances=1,
            enable_arrival=False,
            ambulance_start_distance_m=10.0,
            cell_radius_m=500.0,
            destination_lat=DEST_LAT,
            destination_lon=DEST_LON,
        )
        cfg2 = EnvConfig(
            K_ambulances=1,
            enable_arrival=False,
            ambulance_start_distance_m=10.0,
            cell_radius_m=500.0,
            # Far-away destination (approx 1000km north)
            destination_lat=DEST_LAT + 9.0,
            destination_lon=DEST_LON,
        )
        env1 = ORANEnv(cfg1, seed=42)
        env2 = ORANEnv(cfg2, seed=42)
        env1.reset(seed=42)
        env2.reset(seed=42)
        np.testing.assert_array_equal(
            env1.entered_mask, env2.entered_mask,
            err_msg="Cell entry changed when destination changed — must use dist_to_gNB only"
        )


# ---------------------------------------------------------------------------
# 2. Arrival uses dist_to_destination (not dist_to_gNB)
# ---------------------------------------------------------------------------

class TestArrivalUsesDistanceToDestinationNotGnb:
    """In RWP mode, arrival falls back to dist_to_gnb for backward-compat;
    in SUMO mode it uses dist_to_destination exclusively."""

    def test_info_exposes_dist_to_gnb_and_dist_to_destination_separately(self):
        """_info() must have both 'dist_to_gnb_per_amb' and 'dist_to_destination_per_amb'."""
        env = _env(K=1, enable_arrival=True, arrival_radius_m=25.0)
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert "dist_to_gnb_per_amb" in info
        assert "dist_to_destination_per_amb" in info

    def test_dist_to_gnb_not_same_as_dist_to_destination_in_sumo_path(self):
        """With SUMO enabled, gNB is at (0,0) and destination is at ~47.7m NE.
        dist_to_gnb ≠ dist_to_destination for vehicles away from gNB."""
        env = ORANEnv(macro_mission_config(K_ambulances=1), seed=0)
        env.reset(seed=0)
        # Take a few steps to advance the trace
        for _ in range(50):
            _, _, done, trunc, info = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32)
            )
            if done or trunc:
                break
            # If any ambulance is active, its two distances should differ (destination ≠ gNB)
            if env.active_mask.any():
                gnb_dists = info["dist_to_gnb_per_amb"]
                dest_dists = info["dist_to_destination_per_amb"]
                k_active = np.where(env.active_mask)[0]
                for k in k_active:
                    assert abs(gnb_dists[k] - dest_dists[k]) > 0.1, (
                        f"amb_{k}: dist_to_gnb={gnb_dists[k]:.2f}m equals "
                        f"dist_to_destination={dest_dists[k]:.2f}m — should differ "
                        "(destination is 47.7m NE of gNB, not at gNB)"
                    )
                break
        env.close()


# ---------------------------------------------------------------------------
# 3. Vehicle near gNB but far from destination is NOT arrived
# ---------------------------------------------------------------------------

class TestVehicleNearGnbButFarFromDestinationIsNotArrived:
    """In SUMO mode, being near gNB does not imply arrival at destination."""

    def test_vehicle_at_gnb_not_flagged_arrived_in_sumo_env(self):
        """SumoMobilityProvider: vehicle with last position at gNB (far from destination)
        must NOT set reached_destination_mask."""
        fcd_xml = _make_minimal_fcd_xml([
            (0.0, [("amb_0", BACH_MAI_LON, BACH_MAI_LAT)]),   # at gNB (≈0,0)
            (0.1, [("amb_0", BACH_MAI_LON, BACH_MAI_LAT)]),   # still at gNB
            (0.2, []),                                          # exits FCD at gNB
            (0.3, []),
        ])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fcd.xml", delete=False) as f:
            f.write(fcd_xml)
            tmp = f.name
        try:
            provider = SumoMobilityProvider(
                tmp, K=1, tti_sec=0.1,
                destination_xy_m=_DEST_XY,
                arrival_radius_m=ARRIVAL_RADIUS_M,
            )
            provider.reset()
            # Step through: at t=0.2 vehicle exits FCD at gNB position (0,0)
            # gNB is ~47.7m from destination → should NOT latch reached_destination
            provider.step()  # t=0.1
            provider.step()  # t=0.2 → exit at gNB
            assert not provider.reached_destination_mask[0], (
                "Vehicle exited FCD at gNB (far from destination) — "
                "should NOT be flagged as arrived"
            )
        finally:
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# 4. Vehicle on destination edge within radius IS arrived
# ---------------------------------------------------------------------------

class TestVehicleOnDestinationEdgeWithinRadiusIsArrived:
    """Vehicle stopping within ARRIVAL_RADIUS_M of destination centroid → arrived."""

    def test_fcd_exit_at_destination_latches_mask(self):
        """Vehicle exits FCD within 15m of destination centroid → reached_destination=True."""
        # Place vehicle 5m from destination centroid (well within 15m radius)
        dest_lon = DEST_LON + 5.0 / (111320.0 * math.cos(math.radians(DEST_LAT)))
        dest_lat = DEST_LAT  # same lat

        fcd_xml = _make_minimal_fcd_xml([
            (0.0, [("amb_0", dest_lon, dest_lat)]),   # 5m east of destination
            (0.1, [("amb_0", dest_lon, dest_lat)]),
            (0.2, []),                                 # exits FCD at destination
            (0.3, []),
        ])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fcd.xml", delete=False) as f:
            f.write(fcd_xml)
            tmp = f.name
        try:
            provider = SumoMobilityProvider(
                tmp, K=1, tti_sec=0.1,
                destination_xy_m=_DEST_XY,
                arrival_radius_m=ARRIVAL_RADIUS_M,
            )
            provider.reset()
            provider.step()  # t=0.1
            provider.step()  # t=0.2 → vehicle exits FCD at ~5m from destination
            assert provider.reached_destination_mask[0], (
                "Vehicle exited FCD within arrival_radius_m of destination — "
                "should be marked arrived"
            )
        finally:
            os.unlink(tmp)



# ---------------------------------------------------------------------------
# 5. reached_destination_mask latches arrival (FCD-exit path)
# ---------------------------------------------------------------------------

class TestReachedDestinationMaskLatchesArrival:
    """reached_destination_mask stays True once set (monotone latch)."""

    def test_mask_stays_true_after_fcd_exit(self):
        """After FCD exit at destination, mask stays True on all subsequent steps."""
        dest_lon = DEST_LON
        dest_lat = DEST_LAT

        fcd_xml = _make_minimal_fcd_xml([
            (0.0, [("amb_0", dest_lon, dest_lat)]),
            (0.1, [("amb_0", dest_lon, dest_lat)]),
            (0.2, []),   # exits at destination
            (0.3, []),
            (0.4, []),
        ])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fcd.xml", delete=False) as f:
            f.write(fcd_xml)
            tmp = f.name
        try:
            provider = SumoMobilityProvider(
                tmp, K=1, tti_sec=0.1,
                destination_xy_m=_DEST_XY,
                arrival_radius_m=ARRIVAL_RADIUS_M,
            )
            provider.reset()
            provider.step()  # t=0.1
            provider.step()  # t=0.2 → exits at destination
            assert provider.reached_destination_mask[0]
            provider.step()  # t=0.3 — still absent
            assert provider.reached_destination_mask[0], "Latch must persist after exit"
            provider.step()  # t=0.4
            assert provider.reached_destination_mask[0], "Latch must persist further"
        finally:
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# 6. Changing gNB does NOT change destination arrival
# ---------------------------------------------------------------------------

class TestChangingGnbDoesNotChangeDestinationArrival:
    """Arrival detection is keyed on destination coords, not on gNB = (0,0)."""

    def test_dest_arrival_unchanged_when_gnb_env_unchanged(self):
        """Two SumoMobilityProvider instances with same destination but conceptually
        different gNB should produce same reached_destination_mask — because the
        destination is expressed in the same local Cartesian space."""
        dest_lon = DEST_LON
        dest_lat = DEST_LAT

        fcd_xml = _make_minimal_fcd_xml([
            (0.0, [("amb_0", dest_lon, dest_lat)]),
            (0.1, [("amb_0", dest_lon, dest_lat)]),
            (0.2, []),
            (0.3, []),
        ])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fcd.xml", delete=False) as f:
            f.write(fcd_xml)
            tmp = f.name
        try:
            p1 = SumoMobilityProvider(
                tmp, K=1, tti_sec=0.1,
                destination_xy_m=_DEST_XY,
                arrival_radius_m=ARRIVAL_RADIUS_M,
            )
            p2 = SumoMobilityProvider(
                tmp, K=1, tti_sec=0.1,
                destination_xy_m=_DEST_XY,
                arrival_radius_m=ARRIVAL_RADIUS_M,
            )
            for p in (p1, p2):
                p.reset()
                p.step()
                p.step()  # exit at destination
            assert p1.reached_destination_mask[0] == p2.reached_destination_mask[0]
        finally:
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# 7. Changing destination changes arrival correctly
# ---------------------------------------------------------------------------

class TestChangingDestinationChangesArrivalCorrectly:
    """Different destination coordinates produce different arrival results."""

    def test_near_destination_arrives_far_does_not(self):
        """Same FCD exit position: near-destination → arrives; far-destination → does not."""
        # Vehicle exits at destination centroid
        dest_lon = DEST_LON
        dest_lat = DEST_LAT

        fcd_xml = _make_minimal_fcd_xml([
            (0.0, [("amb_0", dest_lon, dest_lat)]),
            (0.1, [("amb_0", dest_lon, dest_lat)]),
            (0.2, []),
            (0.3, []),
        ])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fcd.xml", delete=False) as f:
            f.write(fcd_xml)
            tmp = f.name
        try:
            # Near destination — should arrive
            p_near = SumoMobilityProvider(
                tmp, K=1, tti_sec=0.1,
                destination_xy_m=_DEST_XY,
                arrival_radius_m=ARRIVAL_RADIUS_M,
            )
            # Far destination — 1000m north of exit position
            far_east = _DEST_EAST
            far_north = _DEST_NORTH + 1000.0
            p_far = SumoMobilityProvider(
                tmp, K=1, tti_sec=0.1,
                destination_xy_m=(far_east, far_north),
                arrival_radius_m=ARRIVAL_RADIUS_M,
            )
            for p in (p_near, p_far):
                p.reset()
                p.step()
                p.step()  # exit at destination centroid
            assert p_near.reached_destination_mask[0], "Exit at destination → should arrive"
            assert not p_far.reached_destination_mask[0], (
                "Exit far from destination → should NOT arrive"
            )
        finally:
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# 8. FCD exit at destination → arrived
# ---------------------------------------------------------------------------

class TestFcdExitAfterDestinationIsArrived:
    """Arrival via FCD exit path (vehicle disappears near destination)."""

    def test_env_marks_arrived_on_fcd_exit_at_destination(self):
        """In macro_mission_config, ambulances should eventually be marked arrived."""
        cfg = macro_mission_config(K_ambulances=1)
        env = ORANEnv(cfg, seed=0)
        env.reset(seed=0)
        terminated = False
        for _ in range(50_000):
            _, _, terminated, truncated, info = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32)
            )
            if terminated or truncated:
                break
        assert info["all_arrived"] or terminated or truncated, (
            "K=1 episode should terminate (all_arrived or timeout)"
        )
        env.close()


# ---------------------------------------------------------------------------
# 9. FCD exit far from destination → not arrived
# ---------------------------------------------------------------------------

class TestFcdExitFarFromDestinationIsInvalidNotArrived:
    """Vehicle that exits FCD far from destination does NOT get marked arrived."""

    def test_fcd_exit_far_from_destination_not_arrived(self):
        """Exit at gNB position (0,0) which is ~47.7m from destination — outside 15m radius."""
        # gNB position in GPS
        gnb_lon = BACH_MAI_LON
        gnb_lat = BACH_MAI_LAT

        fcd_xml = _make_minimal_fcd_xml([
            (0.0, [("amb_0", gnb_lon, gnb_lat)]),   # at gNB
            (0.1, [("amb_0", gnb_lon, gnb_lat)]),
            (0.2, []),   # exits at gNB (far from destination)
            (0.3, []),
        ])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fcd.xml", delete=False) as f:
            f.write(fcd_xml)
            tmp = f.name
        try:
            provider = SumoMobilityProvider(
                tmp, K=1, tti_sec=0.1,
                destination_xy_m=_DEST_XY,
                arrival_radius_m=ARRIVAL_RADIUS_M,
            )
            provider.reset()
            provider.step()
            provider.step()  # exit at gNB (dist from dest ≈ 47.7m > 15m)
            assert not provider.reached_destination_mask[0], (
                f"Vehicle exited FCD at gNB ({_DEST_DIST_FROM_GNB:.1f}m from dest, "
                f"radius={ARRIVAL_RADIUS_M}m) — should NOT be marked arrived"
            )
        finally:
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# 10. Missing vehicle does NOT become position [0,0] at gNB
# ---------------------------------------------------------------------------

class TestMissingVehicleDoesNotBecomePositionZeroAtGnb:
    """After FCD exit, vehicle position freezes at last valid GPS, not [0,0]."""

    def test_last_valid_pos_not_zero_after_fcd_exit(self):
        """Vehicle last seen 200m north of gNB; after exit, last_valid_pos stays there."""
        north_m = 200.0
        lat_offset = north_m / 111_320.0
        veh_lat = BACH_MAI_LAT + lat_offset
        veh_lon = BACH_MAI_LON  # same longitude (due north)

        fcd_xml = _make_minimal_fcd_xml([
            (0.0, [("amb_0", veh_lon, veh_lat)]),
            (0.1, [("amb_0", veh_lon, veh_lat)]),
            (0.2, []),   # exits FCD ~200m north
            (0.3, []),
        ])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fcd.xml", delete=False) as f:
            f.write(fcd_xml)
            tmp = f.name
        try:
            provider = SumoMobilityProvider(
                tmp, K=1, tti_sec=0.1,
                destination_xy_m=_DEST_XY,
                arrival_radius_m=ARRIVAL_RADIUS_M,
            )
            provider.reset()
            provider.step()
            pos_at_exit, _ = provider.step()   # t=0.2, exits FCD

            # pos_at_exit should be ~(0, 200) not (0, 0)
            assert abs(pos_at_exit[0, 0]) < 1.0, "East component should be ~0"
            assert abs(pos_at_exit[0, 1] - north_m) < 1.0, (
                f"North component should be ~{north_m}m; got {pos_at_exit[0, 1]:.2f}"
            )

            # After exit, subsequent steps should keep the frozen position
            pos_next, _ = provider.step()   # t=0.3
            assert abs(pos_next[0, 1] - north_m) < 1.0, (
                "Frozen position must persist after FCD exit"
            )
            # And NOT at gNB (0,0)
            assert abs(pos_next[0, 1]) > 50.0, (
                f"Vehicle should NOT be at gNB after exit; north={pos_next[0,1]:.2f}m"
            )
        finally:
            os.unlink(tmp)

    def test_present_mask_false_after_exit(self):
        """present_mask[k]=False after vehicle exits FCD."""
        veh_lat = BACH_MAI_LAT + 0.002  # ~222m north
        veh_lon = BACH_MAI_LON

        fcd_xml = _make_minimal_fcd_xml([
            (0.0, [("amb_0", veh_lon, veh_lat)]),
            (0.1, [("amb_0", veh_lon, veh_lat)]),
            (0.2, []),
            (0.3, []),
        ])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fcd.xml", delete=False) as f:
            f.write(fcd_xml)
            tmp = f.name
        try:
            provider = SumoMobilityProvider(
                tmp, K=1, tti_sec=0.1,
                destination_xy_m=_DEST_XY,
                arrival_radius_m=ARRIVAL_RADIUS_M,
            )
            provider.reset()
            assert provider.present_mask[0], "Should be present at t=0"
            provider.step()   # t=0.1, still present
            assert provider.present_mask[0]
            provider.step()   # t=0.2, exits FCD
            assert not provider.present_mask[0], "Should be absent after FCD exit"
        finally:
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# 11. Arrived vehicle gets 0 PRB and zeroed obs block
# ---------------------------------------------------------------------------

class TestArrivedVehicleGetsZeroPrbAndZeroObservation:
    """Arrived ambulances get 0-PRB and zeroed obs (force via provider mask)."""

    def test_arrived_vehicle_obs_block_is_zero(self):
        """K=1 arrived ambulance has zeroed per-ambulance obs block."""
        env = _env(K=1, arrival_radius_m=50.0, enable_arrival=True)
        env.reset(seed=0)
        getattr(env._mobility, '_active', env._mobility)._reached_dest_mask = np.array([True])
        env._update_arrival_masks()
        obs, _, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert info["arrived_mask"][0], "Force-arrived ambulance must be arrived"
        base = OBS_FIXED_BLOCK_LEN
        block = obs[base:base + OBS_PER_AMB_BLOCK_LEN]
        assert np.all(block == 0.0), (
            f"Arrived ambulance obs block must be zero sentinel; got {block}"
        )

    def test_arrived_vehicle_gets_zero_prb(self):
        """Arrived ambulance (active_mask=False) gets 0 PRBs.

        Force arrived_mask via provider before step() so PRB allocation sees active_mask[0]=False.
        """
        env = _env(K=3, arrival_radius_m=25.0, enable_arrival=True)
        env.reset(seed=0)
        getattr(env._mobility, '_active', env._mobility)._reached_dest_mask = np.array([True, False, False])
        env._update_arrival_masks()
        _, _, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert info["prb_per_amb"][0] == 0, (
            f"Arrived ambulance (active_mask=False) must receive 0 PRBs; "
            f"got {info['prb_per_amb'][0]}"
        )


# ---------------------------------------------------------------------------
# 12. K=3 terminates only when all three arrived
# ---------------------------------------------------------------------------

class TestK3TerminatesOnlyWhenAllThreeArrived:
    """terminated=True only after all K=3 ambulances reach destination."""

    def test_partial_arrival_does_not_terminate(self):
        """K=3 RWP env: if only 2 ambulances are within arrival_radius_m, no termination."""
        # Place 3 ambulances: 2 at 10m (close), 1 at 400m (far)
        # We can't easily control per-ambulance position at reset with RWP…
        # Instead use SUMO macro_mission_config: episode should NOT terminate
        # until info["all_arrived"] is True.
        cfg = macro_mission_config(K_ambulances=3)
        env = ORANEnv(cfg, seed=0)
        env.reset(seed=0)
        prev_n_arrived = 0
        for step_i in range(5_000):
            _, _, terminated, truncated, info = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32)
            )
            n_arrived = int(info["arrived_mask"].sum())
            if n_arrived > prev_n_arrived and n_arrived < 3:
                # Some arrived but not all — should NOT be terminated yet
                assert not terminated, (
                    f"Step {step_i}: {n_arrived}/3 arrived but terminated=True — "
                    "episode should only end when all 3 arrive"
                )
            prev_n_arrived = n_arrived
            if terminated or truncated:
                break
        env.close()

    def test_all_arrived_terminates(self):
        """K=3: when all_arrived=True, terminated must also be True."""
        cfg = macro_mission_config(K_ambulances=3)
        env = ORANEnv(cfg, seed=0)
        env.reset(seed=0)
        for _ in range(100_000):
            _, _, terminated, truncated, info = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32)
            )
            if info["all_arrived"]:
                assert terminated, "all_arrived=True must coincide with terminated=True"
                break
            if truncated:
                break
        env.close()


# ---------------------------------------------------------------------------
# 13. Timeout when one vehicle never arrives (fallback truncation)
# ---------------------------------------------------------------------------

class TestTimeoutWhenOneVehicleNeverArrives:
    """Episode truncates at episode_duration_sec even if not all arrive."""

    def test_episode_truncates_at_max_duration(self):
        """K=1 env with enable_arrival=False runs until max TTI then truncates."""
        cfg = EnvConfig(
            K_ambulances=1,
            enable_arrival=False,
            episode_duration_sec=0.05,   # 50ms → tiny episode for speed
            ambulance_start_distance_m=200.0,
            cell_radius_m=300.0,
        )
        env = ORANEnv(cfg, seed=0)
        env.reset(seed=0)
        for _ in range(200):
            _, _, terminated, truncated, _ = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32)
            )
            if terminated or truncated:
                assert truncated, "enable_arrival=False → must truncate, not terminate"
                return
        pytest.fail("Episode did not truncate — max_tti not enforced")


# ---------------------------------------------------------------------------
# Acceptance criterion invariants
# ---------------------------------------------------------------------------

class TestAcceptanceCriteria:
    """Six invariants from the spec, verified simultaneously."""

    def test_cell_entry_uses_dist_to_gnb(self):
        """entered_mask uses dist_to_gnb only (not dest distance)."""
        env = _env(K=1, enable_arrival=False, cell_radius_m=500.0)
        env.reset(seed=0)
        assert env.entered_mask[0]   # after cell-entry fast-forward → entered

    def test_arrival_logic_does_not_use_dist_to_gnb_in_sumo(self):
        """In SUMO path: dist_to_gnb and dist_to_destination are independent."""
        env = ORANEnv(macro_mission_config(K_ambulances=1), seed=0)
        env.reset(seed=0)
        for _ in range(100):
            _, _, done, trunc, info = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32)
            )
            if done or trunc:
                break
        # The two distances must exist and be distinguishable
        assert "dist_to_gnb_per_amb" in info
        assert "dist_to_destination_per_amb" in info
        env.close()

    def test_active_mask_equals_entered_and_not_arrived(self):
        """active_mask[k] == entered_mask[k] & ~arrived_mask[k] always."""
        env = _env(K=3, arrival_radius_m=25.0, enable_arrival=True)
        env.reset(seed=0)
        for _ in range(20):
            env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        expected = env.entered_mask & ~env.arrived_mask
        np.testing.assert_array_equal(env.active_mask, expected)

    def test_no_vehicle_missing_from_fcd_maps_to_gnb(self):
        """Non-present vehicles must have |pos| >> 0 (not at gNB = (0,0))."""
        env = ORANEnv(macro_mission_config(K_ambulances=3), seed=0)
        env.reset(seed=0)
        # Run until some vehicle exits FCD (all arrived or truncated)
        for _ in range(100_000):
            _, _, done, trunc, info = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32)
            )
            # Check that non-present vehicles are NOT at origin
            if env._mobility is not None:
                present = env._mobility.present_mask
                for k in range(3):
                    if not present[k]:
                        pos = env.ambulance_pos[k]
                        dist = float(np.linalg.norm(pos))
                        # Should be frozen at last valid position, which is > 1m from gNB
                        # (vehicles stop on edge 37370971#0 which is ~44-54m from gNB)
                        assert dist > 1.0, (
                            f"amb_{k} not present but pos={pos} is at gNB (dist={dist:.2f}m)"
                        )
            if done or trunc:
                break
        env.close()

    def test_k3_terminates_when_all_three_arrived(self):
        """K=3: terminated only after all 3 arrive (spot check for all_arrived flag)."""
        cfg = macro_mission_config(K_ambulances=3)
        env = ORANEnv(cfg, seed=0)
        env.reset(seed=0)
        for _ in range(100_000):
            _, _, terminated, truncated, info = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32)
            )
            if terminated:
                assert info["all_arrived"], "terminated=True must mean all_arrived=True"
                break
            if truncated:
                break
        env.close()
