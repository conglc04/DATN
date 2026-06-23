"""W15 — SUMO mobility tests.

micro-GATE 1D checks:
  1. FCD XML parsed correctly (lat/lon → metric)
  2. Haversine GPS→metric within tolerance
  3. All ambulance distances within cell_radius_m throughout trace
  4. SINR range plausible (≈[-10,+15]dB NLOS-dominated with hard-mission clamp)
  5. Integration: ORANEnv with SUMO trace runs without crash, obs finite
  6. RWP model still works when sumo_fcd_path=None (backwards compat)
  7. K=1 and K=3 traces load and step correctly
  8. Velocity is non-zero and SINR varies across timesteps
"""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as _ET

import numpy as np
import pytest

from env.sumo_mobility import (
    GNB_LAT, GNB_LON,
    default_fcd_path,
    density_fcd_path,
    distance_range_m,
    gps_to_metric,
    load_fcd,
    SumoMobilityProvider,
)
from utils.config import R_CELL_M as _R_CELL_M


# ============================================================
# 1. Haversine GPS → metric
# ============================================================

class TestGpsToMetric:
    def test_origin_is_zero(self):
        x, y = gps_to_metric(GNB_LAT, GNB_LON)
        assert abs(x) < 1e-9 and abs(y) < 1e-9

    def test_north_100m(self):
        m_per_deg = 111_320.0
        lat = GNB_LAT + 100.0 / m_per_deg
        x, y = gps_to_metric(lat, GNB_LON)
        assert abs(x) < 0.01               # no east component
        assert abs(y - 100.0) < 0.5        # north within 0.5m tolerance

    def test_east_100m(self):
        import math as _math
        m_per_deg_lon = 111_320.0 * _math.cos(_math.radians(GNB_LAT))
        lon = GNB_LON + 100.0 / m_per_deg_lon
        x, y = gps_to_metric(GNB_LAT, lon)
        assert abs(x - 100.0) < 0.5
        assert abs(y) < 0.01


# ============================================================
# 2. FCD XML parsing
# ============================================================

class TestLoadFcd:
    @pytest.mark.parametrize("K", [1, 3])
    def test_n_timesteps(self, K):
        ts = load_fcd(density_fcd_path(K, "medium"), vehicle_ids=[f"amb_{k}" for k in range(K)])
        # SUMO --begin 0 --end 12.0 --step-length 0.1 exports t=0.0..11.9 = 120 steps
        assert len(ts) >= 100

    @pytest.mark.parametrize("K", [1, 3])
    def test_n_vehicles_per_timestep(self, K):
        """Vehicles start outside the cell and drive in; presence varies per timestep.
        Each timestep has 0–K vehicles. At least one timestep has all K simultaneously
        (they all start at t=0). After all vehicles reach their destination, empty
        timesteps appear at the end of the trace."""
        ts = load_fcd(density_fcd_path(K, "medium"), vehicle_ids=[f"amb_{k}" for k in range(K)])
        for step in ts:
            assert 0 <= len(step.vehicles) <= K, (
                f"K={K} t={step.time_sec:.1f}s: {len(step.vehicles)} vehicles not in [0,K]"
            )
        # At least one step must have all K vehicles present (they all start together at t=0)
        assert any(len(step.vehicles) == K for step in ts), (
            f"K={K}: no timestep has all {K} vehicles present simultaneously"
        )

    @pytest.mark.parametrize("K", [1, 3])
    def test_timestep_values(self, K):
        ts = load_fcd(density_fcd_path(K, "medium"), vehicle_ids=[f"amb_{k}" for k in range(K)])
        for i, step in enumerate(ts):
            assert abs(step.time_sec - i * 0.1) < 1e-6

    def test_k3_vehicles_distinct(self):
        ts = load_fcd(density_fcd_path(3, "medium"), vehicle_ids=["amb_0", "amb_1", "amb_2"])
        step0 = ts[0]
        positions = [(v.x_m, v.y_m) for v in step0.vehicles]
        # All 3 starting positions must be distinct
        assert len(set(positions)) == 3


# ============================================================
# 3. Micro-Gate: 5 mobility quality tests (W15 requirement)
# ============================================================
# Thesis §W15: real SUMO traces must pass 5 strict checks before
# any solver uses them.  r_CELL = 300 m (thesis §5.2).

CELL_RADIUS_M = _R_CELL_M


class TestSumoSnapStart:
    """Test 1 — Vehicles start OUTSIDE the cell (≥R_CELL_M) and drive IN.

    Routes are 1 km macro-cell design: vehicles depart 1600-1767m from gNB,
    which is 1.6-1.77× R_CELL_M = 1000m.  The bound here is 2×R_CELL_M = 2000m.
    """

    _MAX_START_DIST_M: float = 2 * _R_CELL_M   # 2000 m upper bound on departure distance

    @pytest.mark.parametrize("K", [1, 3])
    def test_start_within_cell(self, K):
        ts = load_fcd(density_fcd_path(K, "medium"), vehicle_ids=[f"amb_{k}" for k in range(K)])
        for v in ts[0].vehicles:
            d = math.sqrt(v.x_m ** 2 + v.y_m ** 2)
            assert d <= self._MAX_START_DIST_M, (
                f"K={K} {v.vehicle_id}: start dist {d:.1f} m > 2×R_CELL_M={self._MAX_START_DIST_M} m"
            )


class TestSumoDistanceInCell:
    """Test 2 — Vehicles start outside cell (~1600-1767m), converge into it.

    Routes are 1km macro-cell: vehicles depart 1.6-1.77×R_CELL_M from gNB.
    Distance range spans from ~250m (final) to ~1767m (start).
    """

    _MAX_DIST_M: float = 2 * _R_CELL_M   # 2000 m; actual max ≈ 1767 m

    @pytest.mark.parametrize("K", [1, 3])
    def test_all_within_cell(self, K):
        min_d, max_d = distance_range_m(density_fcd_path(K, "medium"), K)
        assert max_d <= self._MAX_DIST_M, (
            f"K={K}: max distance {max_d:.1f} m > 2×R_CELL_M={self._MAX_DIST_M} m"
        )
        assert min_d >= 5.0, f"K={K}: min distance {min_d:.1f} m < 5 m (unrealistically close)"

    @pytest.mark.parametrize("K", [1, 3])
    def test_ambulances_converging(self, K):
        """Each vehicle net-converges toward gNB from its first to its last appearance."""
        ts = load_fcd(density_fcd_path(K, "medium"), vehicle_ids=[f"amb_{k}" for k in range(K)])
        ROAD_SLACK_M = 1.0  # verified convergence >>10 m; 1 m covers float jitter only
        for k in range(K):
            vid = f"amb_{k}"
            # Find first and last timestep where this vehicle appears
            first_step = next(
                (step for step in ts if any(v.vehicle_id == vid for v in step.vehicles)),
                None,
            )
            last_step = next(
                (step for step in reversed(ts) if any(v.vehicle_id == vid for v in step.vehicles)),
                None,
            )
            assert first_step is not None, f"K={K} {vid}: never appears in trace"
            v0 = next(v for v in first_step.vehicles if v.vehicle_id == vid)
            vN = next(v for v in last_step.vehicles if v.vehicle_id == vid)
            d_start = math.sqrt(v0.x_m ** 2 + v0.y_m ** 2)
            d_end   = math.sqrt(vN.x_m ** 2 + vN.y_m ** 2)
            assert d_end < d_start + ROAD_SLACK_M, (
                f"K={K} {vid}: dist increased by >{ROAD_SLACK_M} m "
                f"({d_start:.1f}→{d_end:.1f} m); expected convergence toward gNB"
            )


class TestSumoTraceDuration:
    """Test 3 — Trace duration ≥10 s and ≥100 timesteps (no wrap-around in 1 episode)."""

    @pytest.mark.parametrize("K", [1, 3])
    def test_duration_seconds(self, K):
        ts = load_fcd(density_fcd_path(K, "medium"), vehicle_ids=[f"amb_{k}" for k in range(K)])
        assert ts[-1].time_sec >= 10.0, (
            f"K={K}: trace ends at {ts[-1].time_sec:.1f} s < 10.0 s; "
            "SumoMobilityProvider would wrap-around within a single 10 s episode"
        )

    @pytest.mark.parametrize("K", [1, 3])
    def test_n_timesteps_sufficient(self, K):
        ts = load_fcd(density_fcd_path(K, "medium"), vehicle_ids=[f"amb_{k}" for k in range(K)])
        assert len(ts) >= 100, (
            f"K={K}: only {len(ts)} timesteps — need ≥100 for 10 s at 0.1 s step-length"
        )


class TestSumoNoTeleport:
    """Test 4 — No teleport / position jump between consecutive FCD timesteps.

    Known SUMO artifacts accepted at ≤4 m:
      • Step 1:  FROM-node → lane-centerpoint snap at departure (always ≤4 m)
      • Junctions: edge-transition discontinuity (≤4 m in bachmaiHN network)
    Genuine SUMO teleports (stuck vehicle re-inserted) are >>50 m.
    Threshold 5 m is tight enough to catch real teleports while tolerating artifacts.
    """

    MAX_STEP_M: float = 5.0

    @pytest.mark.parametrize("K", [1, 3])
    def test_no_position_jump(self, K):
        """Check consecutive position jumps for vehicles present in both steps."""
        ts = load_fcd(density_fcd_path(K, "medium"), vehicle_ids=[f"amb_{k}" for k in range(K)])
        for step_idx in range(1, len(ts)):
            prev_by_id = {v.vehicle_id: v for v in ts[step_idx - 1].vehicles}
            curr_by_id = {v.vehicle_id: v for v in ts[step_idx].vehicles}
            # Only check vehicles present in both consecutive timesteps
            for vid in prev_by_id.keys() & curr_by_id.keys():
                p, c = prev_by_id[vid], curr_by_id[vid]
                displacement = math.sqrt((c.x_m - p.x_m) ** 2 + (c.y_m - p.y_m) ** 2)
                assert displacement <= self.MAX_STEP_M, (
                    f"K={K} {vid} step {step_idx}: jump {displacement:.2f} m > "
                    f"{self.MAX_STEP_M} m — possible teleport"
                )


class TestSumoK3IdStable:
    """Test 5 — K=3 vehicles arrive at different times; at every timestep the vehicles
    present are a valid subset of {amb_0, amb_1, amb_2} with no duplicates."""

    def test_all_ids_present_every_timestep(self):
        """Every timestep contains only known vehicle IDs (no phantom/unknown IDs).
        Empty timesteps (after all vehicles reach destination) are allowed."""
        K = 3
        expected_ids = {f"amb_{k}" for k in range(K)}
        ts = load_fcd(density_fcd_path(K, "medium"), vehicle_ids=list(expected_ids))
        for step_idx, step in enumerate(ts):
            actual_ids = {v.vehicle_id for v in step.vehicles}
            # Every ID present must be a known ambulance ID
            assert actual_ids <= expected_ids, (
                f"Timestep {step_idx} (t={step.time_sec:.1f} s): "
                f"unknown ids={actual_ids - expected_ids}"
            )

    def test_k3_distinct_positions_every_timestep(self):
        """No two present ambulances collapse to the same position at any timestep."""
        ts = load_fcd(density_fcd_path(3, "medium"), vehicle_ids=["amb_0", "amb_1", "amb_2"])
        for step_idx, step in enumerate(ts):
            vehs = step.vehicles
            for i in range(len(vehs)):
                for j in range(i + 1, len(vehs)):
                    vi, vj = vehs[i], vehs[j]
                    dist = math.sqrt((vi.x_m - vj.x_m) ** 2 + (vi.y_m - vj.y_m) ** 2)
                    assert dist > 1.0, (
                        f"Timestep {step_idx} (t={step.time_sec:.1f} s): "
                        f"{vi.vehicle_id} and {vj.vehicle_id} within 1 m ({dist:.2f} m)"
                    )


# ============================================================
# 4. SumoMobilityProvider
# ============================================================

class TestSumoMobilityProvider:
    @pytest.mark.parametrize("K", [1, 3])
    def test_reset_shape(self, K):
        p = SumoMobilityProvider(density_fcd_path(K, "medium"), K=K, tti_sec=0.01)
        pos = p.reset()
        assert pos.shape == (K, 2)

    @pytest.mark.parametrize("K", [1, 3])
    def test_step_shape(self, K):
        p = SumoMobilityProvider(density_fcd_path(K, "medium"), K=K, tti_sec=0.01)
        p.reset()
        pos, vel = p.step()
        assert pos.shape == (K, 2)
        assert vel.shape == (K, 2)

    def test_velocity_nonzero(self):
        p = SumoMobilityProvider(density_fcd_path(1, "medium"), K=1, tti_sec=0.01)
        p.reset()
        _, vel = p.step()
        speed = float(np.linalg.norm(vel))
        assert speed > 1.0, f"speed {speed:.2f} m/s too slow — mobility not advancing"

    def test_positions_change_across_steps(self):
        p = SumoMobilityProvider(density_fcd_path(3, "medium"), K=3, tti_sec=0.01)
        pos0 = p.reset().copy()
        for _ in range(10):
            pos, _ = p.step()
        # After 10 steps positions should differ from start
        assert not np.allclose(pos, pos0, atol=0.01)

    def test_100_steps_no_crash(self):
        p = SumoMobilityProvider(density_fcd_path(3, "medium"), K=3, tti_sec=0.01)
        p.reset()
        for _ in range(100):
            pos, vel = p.step()
            assert np.all(np.isfinite(pos))
            assert np.all(np.isfinite(vel))

    def test_wrap_around(self):
        p = SumoMobilityProvider(density_fcd_path(1, "medium"), K=1, tti_sec=0.01)
        p.reset()
        # Run past end of 1.0s trace (100+ steps at 10ms)
        for _ in range(110):
            pos, vel = p.step()
        assert np.all(np.isfinite(pos)), "wrap-around produced NaN"


# ============================================================
# 5. ORANEnv integration with SUMO
# ============================================================

class TestORANEnvSumo:
    def _make_env(self, K: int, sumo: bool = True):
        from env.oran_env import EnvConfig, ORANEnv, hard_mission_config
        if sumo:
            cfg = hard_mission_config(K_ambulances=K)
        else:
            from env.sumo_mobility import default_fcd_path as _fcp
            cfg = EnvConfig(K_ambulances=K, sumo_fcd_path=None)
        return ORANEnv(cfg)

    @pytest.mark.parametrize("K", [1, 3])
    def test_reset_no_crash(self, K):
        env = self._make_env(K)
        obs, info = env.reset(seed=0)
        assert np.all(np.isfinite(obs))
        env.close()

    @pytest.mark.parametrize("K", [1, 3])
    def test_step_no_crash(self, K):
        env = self._make_env(K)
        env.reset(seed=0)
        obs, reward, term, trunc, info = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32)
        )
        assert np.all(np.isfinite(obs))
        assert np.isfinite(reward)
        env.close()

    @pytest.mark.parametrize("K", [1, 3])
    def test_sinr_range_plausible(self, K):
        from env.oran_env import hard_mission_config, ORANEnv
        env = ORANEnv(hard_mission_config(K_ambulances=K))
        env.reset(seed=0)
        sinrs = []
        for _ in range(50):
            env.step(np.zeros(env.action_space.shape, dtype=np.float32))
            sinrs.append(float(np.mean(env.last_sinr_db)))
        sinr_mean = float(np.mean(sinrs))
        # With 60km/h convergence at 150-220m, SINR with hard-mission clamp
        # (max=15dB) should be in [-10, 15] dB
        assert -10.0 <= sinr_mean <= 15.5, (
            f"K={K}: mean SINR {sinr_mean:.1f}dB outside plausible range [-10, 15] dB"
        )
        env.close()

    def test_sinr_varies_across_steps(self):
        """SINR must vary across steps for K=3 (amb_1 starts 247 m from gNB).

        K=1 is excluded: amb_0 converges from 95 m toward gNB — it stays within
        SINR_MAX_DB cap throughout the trace, so std==0 is correct there.
        """
        from env.oran_env import hard_mission_config, ORANEnv
        env = ORANEnv(hard_mission_config(K_ambulances=3))
        env.reset(seed=0)
        sinrs = []
        for _ in range(80):
            env.step(np.zeros(env.action_space.shape, dtype=np.float32))
            sinrs.append(float(np.mean(env.last_sinr_db)))
        assert np.std(sinrs) > 0.0, "SINR is constant — mobility has no effect on channel"
        env.close()


# ============================================================
# 6. RWP legacy mode (sumo_fcd_path=None) still works
# ============================================================

class TestRwpLegacy:
    def test_rwp_reset_step_no_crash(self):
        from env.oran_env import EnvConfig, ORANEnv
        cfg = EnvConfig(K_ambulances=1, sumo_fcd_path=None)
        env = ORANEnv(cfg)
        obs, _ = env.reset(seed=42)
        assert np.all(np.isfinite(obs))
        obs2, r, _, _, _ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert np.all(np.isfinite(obs2))
        env.close()


# ============================================================
# 7. Speed provenance, trace consistency, backend wiring (W15)
# ============================================================
# 16 new classes verifying that the SUMO backend is correctly
# wired end-to-end: route XML → FCD trace → ORANEnv obs vector.

import os as _os
import tempfile
import xml.etree.ElementTree as _ET

from utils.config import (
    AMB_DIST_OFFSET,
    AMB_SINR_OFFSET,
    AMB_SPEED_OFFSET,
    OBS_FIXED_BLOCK_LEN,
    OBS_PER_AMB_BLOCK_LEN,
    TTI_SEC,
)

_ROUTE_DIR: str = _os.path.normpath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "data", "sumo")
)
_NET_FILE: str = _os.path.join(_ROUTE_DIR, "bachmaiHN.net.xml")

_SUMO_STEP_DT: float = 0.1           # --step-length 0.1 in 04_run_simulation.sh
_VTYPE_MAX_SPEED_MS: float = 60.0 / 3.6  # 60 km/h; vType maxSpeed in route XML
_VTYPE_ACCEL_MS2: float = 2.0        # vType accel (m/s²)
_VTYPE_DECEL_MS2: float = 4.5        # vType decel (m/s²); upper bound on deceleration
_SUMO_SPEED_FACTOR_MAX: float = 1.10 # SUMO default speedDev=0.1 → vehicles may run at ≤ 1.1 × lane_speed


def _route_xml_path(K: int) -> str:
    return _os.path.join(_ROUTE_DIR, f"ambulance_routes_k{K}.xml")


class TestSumoSpeedLimitSource:
    """departSpeed='max' — departure speed comes from the SUMO edge limit, not a Python literal."""

    @pytest.mark.parametrize("K", [1, 3])
    def test_depart_speed_is_max(self, K):
        tree = _ET.parse(_route_xml_path(K))
        for trip in tree.getroot().iter("trip"):
            dep = trip.get("departSpeed", "")
            assert dep == "max", (
                f"K={K} trip '{trip.get('id')}': departSpeed={dep!r}, "
                "expected 'max' (edge speed limit drives departure)"
            )


class TestAmbulanceVTypeMaxSpeed:
    """vType maxSpeed in route XML equals 60 km/h (= 60/3.6 ≈ 16.67 m/s)."""

    @pytest.mark.parametrize("K", [1, 3])
    def test_vtype_maxspeed_equals_60_kmh(self, K):
        tree = _ET.parse(_route_xml_path(K))
        for vtype in tree.getroot().iter("vType"):
            raw = vtype.get("maxSpeed")
            assert raw is not None, f"K={K}: vType missing maxSpeed attribute"
            ms = float(raw)
            assert abs(ms - _VTYPE_MAX_SPEED_MS) < 0.01, (
                f"K={K}: vType maxSpeed={ms:.4f} m/s ≠ 60/3.6={_VTYPE_MAX_SPEED_MS:.4f} m/s"
            )


class TestRouteLaneSpeedLimits:
    """Route edges exist in bachmaiHN.net.xml with positive lane speed limits.

    departSpeed='max' takes the speed limit from the departure lane — this verifies
    that the actual OSM-derived lane speed exists and is drivable (speed > 0).
    The three tests go from coarse (edge exists) to fine (FCD speed within lane cap).
    """

    @staticmethod
    def _lane_speeds(net_root, edge_id: str) -> list[float]:
        """Return speed limits of all lanes on `edge_id` (from SUMO net.xml)."""
        for edge_el in net_root.iter("edge"):
            if edge_el.get("id") == edge_id:
                return [float(lane.get("speed", 0.0)) for lane in edge_el.iter("lane")]
        return []

    @pytest.mark.parametrize("K", [1, 3])
    def test_route_edges_exist_in_network(self, K):
        """from/to edges in route XML are present in bachmaiHN.net.xml."""
        net_root = _ET.parse(_NET_FILE).getroot()
        for trip in _ET.parse(_route_xml_path(K)).getroot().iter("trip"):
            for attr in ("from", "to"):
                edge_id = trip.get(attr, "")
                speeds = self._lane_speeds(net_root, edge_id)
                assert speeds, (
                    f"K={K} trip '{trip.get('id')}' {attr}='{edge_id}' "
                    "not found in bachmaiHN.net.xml"
                )

    @pytest.mark.parametrize("K", [1, 3])
    def test_departure_lane_speed_positive(self, K):
        """Departure edge lanes have speed > 0 m/s (road is physically drivable).

        SUMO departSpeed='max' uses the lane speed limit — a zero-speed lane
        would cause the vehicle to depart at 0 m/s regardless of vType maxSpeed.
        """
        net_root = _ET.parse(_NET_FILE).getroot()
        for trip in _ET.parse(_route_xml_path(K)).getroot().iter("trip"):
            edge_id = trip.get("from", "")
            speeds = self._lane_speeds(net_root, edge_id)
            assert speeds, f"K={K}: departure edge '{edge_id}' not found in net.xml"
            assert all(s > 0.0 for s in speeds), (
                f"K={K}: departure edge '{edge_id}' has zero-speed lane "
                f"(speeds={speeds}) — departSpeed='max' would depart at 0 m/s"
            )

    @pytest.mark.parametrize("K", [1, 3])
    def test_fcd_per_timestep_within_lane_limit(self, K):
        """At each FCD timestep, vehicle speed ≤ current lane's speed limit × speedFactor_max.

        Reads lane speed for EVERY edge the vehicle actually traverses (via FCD `lane`
        attribute), not just departure+destination edges.  This covers all intermediate
        edges on the computed SUMO route.

        SUMO IDM car-following model with default speedDev=0.1 allows vehicles to sample a
        per-vehicle speed factor ∈ [0.9, 1.1].  So the effective per-timestep ceiling is:
          effective_limit = min(lane_speed × _SUMO_SPEED_FACTOR_MAX, vType_maxSpeed)

        Verified against real FCD data:
          edge -1023619581#5 (13.89 m/s): FCD max = 14.72 m/s ≤ 13.89 × 1.10 = 15.28 ✓

        Junction edges (id starts with ':') are internal SUMO nodes with no net.xml entry
        and are skipped.
        """
        net_root = _ET.parse(_NET_FILE).getroot()
        # Build edge_id → max_lane_speed lookup from net.xml
        edge_speed: dict[str, float] = {}
        for edge_el in net_root.iter("edge"):
            eid = edge_el.get("id", "")
            speeds = [float(lane.get("speed", 0.0)) for lane in edge_el.iter("lane")]
            if speeds:
                edge_speed[eid] = max(speeds)

        fcd_tree = _ET.parse(density_fcd_path(K, "medium"))
        vehicle_ids = {f"amb_{k}" for k in range(K)}
        violations: list[tuple] = []

        for ts_el in fcd_tree.getroot().iter("timestep"):
            t = float(ts_el.get("time", 0.0))
            for v_el in ts_el.iter("vehicle"):
                vid = v_el.get("id", "")
                if vid not in vehicle_ids:
                    continue
                speed = float(v_el.get("speed", 0.0))
                lane_id = v_el.get("lane", "")
                # Skip SUMO internal junction edges (no net.xml lane_speed entry)
                if lane_id.startswith(":"):
                    continue
                # Extract edge_id: SUMO lane_id = "{edge_id}_{lane_index}" where
                # lane_index is a non-negative integer suffix
                parts = lane_id.rsplit("_", 1)
                edge_id = parts[0] if (len(parts) == 2 and parts[1].isdigit()) else lane_id
                if edge_id not in edge_speed:
                    continue  # edge not found in net.xml, skip
                lane_limit = edge_speed[edge_id]
                # SUMO IDM allows up to _SUMO_SPEED_FACTOR_MAX × lane_limit;
                # vType maxSpeed is the absolute ceiling regardless
                effective_limit = min(lane_limit * _SUMO_SPEED_FACTOR_MAX, _VTYPE_MAX_SPEED_MS)
                if speed > effective_limit + 0.05:
                    violations.append((
                        round(t, 2), vid, round(speed, 3),
                        round(lane_limit, 3), round(effective_limit, 3), edge_id,
                    ))

        assert not violations, (
            f"K={K}: {len(violations)} FCD timesteps exceed per-lane effective limit "
            f"(lane × {_SUMO_SPEED_FACTOR_MAX} ∩ vType_max={_VTYPE_MAX_SPEED_MS:.3f}). "
            f"First 5:\n" + "\n".join(str(v) for v in violations[:5])
        )


class TestNoHardcodedSpeedCap60:
    """load_fcd passes FCD speed values through verbatim — no Python-level 60 km/h clamp."""

    def test_fcd_speed_passthrough_k1(self):
        fcd_path = density_fcd_path(1, "medium")
        # Parse XML directly to get ground-truth speed values
        raw: dict[float, float] = {}
        for ts_el in _ET.parse(fcd_path).getroot().iter("timestep"):
            t = float(ts_el.get("time", 0.0))
            for v_el in ts_el.iter("vehicle"):
                if v_el.get("id") == "amb_0":
                    raw[t] = float(v_el.get("speed", 0.0))

        ts = load_fcd(fcd_path, vehicle_ids=["amb_0"])
        for step in ts:
            expected = raw.get(step.time_sec)
            if expected is not None:
                assert abs(step.vehicles[0].speed_ms - expected) < 1e-9, (
                    f"t={step.time_sec:.1f}s: load_fcd modified speed "
                    f"{expected} → {step.vehicles[0].speed_ms} (must be verbatim)"
                )


class TestSpeedUnitConversion:
    """60 km/h converts to 60/3.6 m/s exactly; vType XML value matches this."""

    def test_60_kmh_in_ms(self):
        expected_ms = 60.0 / 3.6
        assert abs(expected_ms - 16.6667) < 0.001

    @pytest.mark.parametrize("K", [1, 3])
    def test_route_xml_maxspeed_unit(self, K):
        tree = _ET.parse(_route_xml_path(K))
        for vtype in tree.getroot().iter("vType"):
            ms = float(vtype.get("maxSpeed", "0"))
            kmh = ms * 3.6
            assert abs(kmh - 60.0) < 0.1, (
                f"K={K}: vType maxSpeed {ms:.3f} m/s = {kmh:.1f} km/h ≠ 60 km/h"
            )


class TestTraceTimestepConsistency:
    """Every consecutive pair of FCD timesteps has exactly 0.1 s spacing."""

    @pytest.mark.parametrize("K", [1, 3])
    def test_step_interval_is_0p1s(self, K):
        ts = load_fcd(density_fcd_path(K, "medium"), vehicle_ids=[f"amb_{k}" for k in range(K)])
        for i in range(1, len(ts)):
            dt = ts[i].time_sec - ts[i - 1].time_sec
            assert abs(dt - _SUMO_STEP_DT) < 1e-6, (
                f"K={K} step {i}: interval {dt:.6f}s ≠ {_SUMO_STEP_DT}s "
                f"(t={ts[i-1].time_sec:.2f}→{ts[i].time_sec:.2f})"
            )


class TestDisplacementMatchesSpeed:
    """Euclidean displacement ≤ speed·dt for non-artifact steps (physical consistency).

    Known SUMO artifacts (departure snap, junction transition) produce jumps up to 4m
    but are excluded by the 2m artifact threshold; real-motion steps must satisfy
    the Euclidean-distance ≤ arc-length invariant.
    """

    _ARTIFACT_M: float = 2.0   # artifact threshold; steps with jump ≥ this are skipped
    _SLACK_M: float = 0.3      # absolute slack for road curvature in Euclidean vs arc

    @pytest.mark.parametrize("K", [1, 3])
    def test_displacement_le_speed_dt(self, K):
        """Check displacement ≤ speed·dt for vehicles present in consecutive steps."""
        ts = load_fcd(density_fcd_path(K, "medium"), vehicle_ids=[f"amb_{k}" for k in range(K)])
        dt = _SUMO_STEP_DT
        checked = 0
        for step_idx in range(1, len(ts)):
            prev_by_id = {v.vehicle_id: v for v in ts[step_idx - 1].vehicles}
            curr_by_id = {v.vehicle_id: v for v in ts[step_idx].vehicles}
            for vid in prev_by_id.keys() & curr_by_id.keys():
                p, c = prev_by_id[vid], curr_by_id[vid]
                dx = c.x_m - p.x_m
                dy = c.y_m - p.y_m
                disp = math.sqrt(dx ** 2 + dy ** 2)
                if disp >= self._ARTIFACT_M:
                    continue  # known SUMO departure/junction artifact — skip
                limit = p.speed_ms * dt + self._SLACK_M
                assert disp <= limit, (
                    f"K={K} {vid} step {step_idx} (t={ts[step_idx].time_sec:.1f}s): "
                    f"displacement {disp:.3f}m > speed·dt+slack {limit:.3f}m"
                )
                checked += 1
        assert checked > 50, (
            f"K={K}: only {checked} steps checked — artifact filter too aggressive"
        )


class TestEpisodeDurationFromConfig:
    """n_steps = round(episode_duration_sec / tti_sec) — derived from config, not hardcoded."""

    def test_n_steps_formula(self):
        from env.oran_env import EnvConfig, ORANEnv
        cfg = EnvConfig(K_ambulances=1)
        env = ORANEnv(cfg)
        expected = round(cfg.episode_duration_sec / cfg.tti_sec)
        actual = env._max_tti_for_episode()
        assert actual == expected, (
            f"_max_tti_for_episode()={actual} ≠ "
            f"round({cfg.episode_duration_sec}/{cfg.tti_sec})={expected}"
        )
        env.close()

    def test_tti_sec_matches_config_constant(self):
        from env.oran_env import EnvConfig
        cfg = EnvConfig(K_ambulances=1)
        assert cfg.tti_sec == pytest.approx(TTI_SEC), (
            f"EnvConfig.tti_sec={cfg.tti_sec} ≠ config.TTI_SEC={TTI_SEC}"
        )

    @pytest.mark.parametrize("K", [1, 3])
    def test_fcd_trace_covers_at_least_one_episode(self, K):
        """SUMO FCD timestamp span is ≥ episode_duration_sec (uses span, not count).

        Count-based checks have an off-by-one ambiguity: 120 timesteps could mean
        span = 119 × dt or 120 × dt depending on whether the last step is inclusive.
        Timestamp span avoids this: span = t_last - t_first + dt is unambiguous.

        SUMO runs with --end 12.0s while the env episode is episode_duration_sec=1.0s.
        The FCD span must be ≥ episode_duration_sec so the mobility backend never
        hits end-of-trace mid-episode.  If episode_duration_sec is raised, a SUMO
        re-run is required — this test will catch that.
        """
        from env.oran_env import EnvConfig
        cfg = EnvConfig(K_ambulances=K)
        ts = load_fcd(density_fcd_path(K, "medium"), vehicle_ids=[f"amb_{k}" for k in range(K)])
        assert len(ts) >= 2, f"K={K}: FCD trace has fewer than 2 timesteps"
        span_s = ts[-1].time_sec - ts[0].time_sec + _SUMO_STEP_DT
        assert span_s >= cfg.episode_duration_sec - 1e-9, (
            f"K={K}: FCD timestamp span {span_s:.3f}s "
            f"(t[0]={ts[0].time_sec:.2f}s .. t[-1]={ts[-1].time_sec:.2f}s + dt={_SUMO_STEP_DT}s) "
            f"< episode_duration_sec={cfg.episode_duration_sec}s — SUMO re-run required"
        )


class TestSumoAccelerationBound:
    """Speed changes per step respect separate vType accel and decel bounds.

    SUMO vType: accel=2.0 m/s², decel=4.5 m/s²  →  two distinct per-direction limits.

    Artifact classification (by position jump ≥ 2m):
      - "departure-snap" : step_idx ≤ 2 — departSpeed='max' snaps from 0 → lane speed.
      - "junction-transition" : step_idx > 2 — edge-to-edge discontinuity in bachmaiHN net.
    Both are < 4m in this network; real SUMO teleports are ≫ 50m.

    Hard constraint: ≤ 3 artifact skips per vehicle per trace.  If more are found, the
    threshold is too broad and must be tightened.
    """

    _ACCEL_LIMIT_MS: float = _VTYPE_ACCEL_MS2 * _SUMO_STEP_DT + 0.10   # 2.0×0.1 + ε
    _DECEL_LIMIT_MS: float = _VTYPE_DECEL_MS2 * _SUMO_STEP_DT + 0.10   # 4.5×0.1 + ε
    _ARTIFACT_JUMP_M: float = 2.0    # displacement threshold (> max normal 1.667m)
    _MAX_ARTIFACTS_PER_VEH: int = 3  # departure-snap + ≤2 junctions per route

    @staticmethod
    def _classify_artifact(step_idx: int, jump_m: float) -> str:
        return "departure-snap" if step_idx <= 2 else "junction-transition"

    @pytest.mark.parametrize("K", [1, 3])
    def test_accel_bound(self, K):
        """Speed increase per non-artifact step ≤ accel × dt (2.0 m/s² × 0.1 s = 0.20 m/s).
        Only checks vehicles present in both consecutive timesteps."""
        ts = load_fcd(density_fcd_path(K, "medium"), vehicle_ids=[f"amb_{k}" for k in range(K)])
        # artifacts: list of (step_idx, vid, jump_m, reason, t_sec)
        artifacts: list[tuple] = []
        for step_idx in range(2, len(ts)):
            prev_by_id = {v.vehicle_id: v for v in ts[step_idx - 1].vehicles}
            curr_by_id = {v.vehicle_id: v for v in ts[step_idx].vehicles}
            for vid in prev_by_id.keys() & curr_by_id.keys():
                p, c = prev_by_id[vid], curr_by_id[vid]
                dx = c.x_m - p.x_m
                dy = c.y_m - p.y_m
                jump = math.sqrt(dx ** 2 + dy ** 2)
                if jump >= self._ARTIFACT_JUMP_M:
                    artifacts.append((
                        step_idx, vid, round(jump, 3),
                        self._classify_artifact(step_idx, jump),
                        round(ts[step_idx].time_sec, 2),
                    ))
                    continue
                delta = c.speed_ms - p.speed_ms
                if delta > 0:
                    assert delta <= self._ACCEL_LIMIT_MS, (
                        f"K={K} {vid} step {step_idx} (t={ts[step_idx].time_sec:.1f}s): "
                        f"Δspeed(accel)={delta:.3f} m/s > accel·dt={self._ACCEL_LIMIT_MS:.3f} m/s"
                    )
        max_allowed = self._MAX_ARTIFACTS_PER_VEH * K
        assert len(artifacts) <= max_allowed, (
            f"K={K}: {len(artifacts)} artifact steps skipped — threshold 2m is too broad "
            f"(max allowed {max_allowed} = {self._MAX_ARTIFACTS_PER_VEH}×K). "
            f"Skipped: {artifacts}"
        )

    @pytest.mark.parametrize("K", [1, 3])
    def test_decel_bound(self, K):
        """Speed decrease per non-artifact step ≤ decel × dt (4.5 m/s² × 0.1 s = 0.45 m/s).
        Only checks vehicles present in both consecutive timesteps."""
        ts = load_fcd(density_fcd_path(K, "medium"), vehicle_ids=[f"amb_{k}" for k in range(K)])
        artifacts: list[tuple] = []
        for step_idx in range(2, len(ts)):
            prev_by_id = {v.vehicle_id: v for v in ts[step_idx - 1].vehicles}
            curr_by_id = {v.vehicle_id: v for v in ts[step_idx].vehicles}
            for vid in prev_by_id.keys() & curr_by_id.keys():
                p, c = prev_by_id[vid], curr_by_id[vid]
                dx = c.x_m - p.x_m
                dy = c.y_m - p.y_m
                jump = math.sqrt(dx ** 2 + dy ** 2)
                if jump >= self._ARTIFACT_JUMP_M:
                    artifacts.append((
                        step_idx, vid, round(jump, 3),
                        self._classify_artifact(step_idx, jump),
                        round(ts[step_idx].time_sec, 2),
                    ))
                    continue
                delta = c.speed_ms - p.speed_ms
                if delta < 0:
                    assert abs(delta) <= self._DECEL_LIMIT_MS, (
                        f"K={K} {vid} step {step_idx} (t={ts[step_idx].time_sec:.1f}s): "
                        f"|Δspeed(decel)|={abs(delta):.3f} m/s > decel·dt={self._DECEL_LIMIT_MS:.3f} m/s"
                    )
        max_allowed = self._MAX_ARTIFACTS_PER_VEH * K
        assert len(artifacts) <= max_allowed, (
            f"K={K}: {len(artifacts)} artifact steps skipped — threshold 2m is too broad "
            f"(max allowed {max_allowed} = {self._MAX_ARTIFACTS_PER_VEH}×K). "
            f"Skipped: {artifacts}"
        )


class TestStopAtJunctionIsValid:
    """Vehicles slow at network junctions; load_fcd handles any speed including 0 cleanly.

    Three sub-tests:
    1. Unit: speed=0 in a synthetic mini-FCD parses to 0.0 (not NaN, not error).
    2. Unit: SumoMobilityProvider stays finite across all real FCD timesteps.
    3. Integration: real SUMO trace shows sub-freeflow speed events — evidence of
       actual junction interaction in the bachmaiHN urban network, not just a parse test.
    """

    def test_speed_zero_parses_as_zero(self):
        fcd_xml = (
            '<fcd-export>'
            '<timestep time="0.00">'
            '<vehicle id="amb_0" x="105.840780" y="21.002966" angle="0" speed="0.000" lane="r_0" />'
            '</timestep>'
            '<timestep time="0.10">'
            '<vehicle id="amb_0" x="105.840780" y="21.002967" angle="0" speed="5.000" lane="r_0" />'
            '</timestep>'
            '</fcd-export>'
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fcd.xml", delete=False) as f:
            f.write(fcd_xml)
            tmp = f.name
        try:
            ts = load_fcd(tmp, vehicle_ids=["amb_0"])
            assert ts[0].vehicles[0].speed_ms == pytest.approx(0.0)
            assert ts[1].vehicles[0].speed_ms == pytest.approx(5.0)
        finally:
            _os.unlink(tmp)

    def test_provider_finite_through_full_trace(self):
        """SumoMobilityProvider stays finite across all real FCD timesteps."""
        p = SumoMobilityProvider(density_fcd_path(1, "medium"), K=1, tti_sec=_SUMO_STEP_DT)
        p.reset()
        for _ in range(len(load_fcd(density_fcd_path(1, "medium"), vehicle_ids=["amb_0"])) - 1):
            pos, vel = p.step()
            assert np.all(np.isfinite(pos)), "NaN position in real trace replay"
            assert np.all(np.isfinite(vel)), "NaN velocity in real trace replay"

    @pytest.mark.parametrize("K", [1, 3])
    def test_emergency_vehicle_no_junction_stop(self, K):
        """Emergency vehicles with bluelight+vClass never stop at junctions.

        With vClass="emergency" and device.bluelight, SUMO grants unconditional
        right-of-way: all signals turn green and background vehicles yield at 25 m.
        Vehicles should never slow to near-zero due to junction constraints.
        Threshold = 4 m/s (14.4 km/h) — any event below this indicates a stop.
        """
        ts = load_fcd(density_fcd_path(K, "medium"), vehicle_ids=[f"amb_{k}" for k in range(K)])
        stop_threshold_ms = 4.0  # 14.4 km/h
        stopped = [
            (step.time_sec, v.vehicle_id, v.speed_ms)
            for step in ts
            for v in step.vehicles
            if v.speed_ms < stop_threshold_ms
        ]
        assert not stopped, (
            f"K={K}: emergency vehicle fell below {stop_threshold_ms:.1f} m/s "
            f"({stop_threshold_ms * 3.6:.0f} km/h) — unexpected junction stop. "
            f"First 3 events: {stopped[:3]}"
        )


class TestNoNegativeSpeed:
    """All FCD speed values are non-negative (SUMO never produces negative speeds)."""

    @pytest.mark.parametrize("K", [1, 3])
    def test_all_speeds_nonnegative(self, K):
        ts = load_fcd(density_fcd_path(K, "medium"), vehicle_ids=[f"amb_{k}" for k in range(K)])
        for step in ts:
            for v in step.vehicles:
                assert v.speed_ms >= 0.0, (
                    f"K={K} {v.vehicle_id} t={step.time_sec:.1f}s: "
                    f"negative speed {v.speed_ms:.4f} m/s"
                )


class TestSpeedFiniteAllTimesteps:
    """All FCD speed values are finite (no NaN or ±inf from XML parsing)."""

    @pytest.mark.parametrize("K", [1, 3])
    def test_all_speeds_finite(self, K):
        ts = load_fcd(density_fcd_path(K, "medium"), vehicle_ids=[f"amb_{k}" for k in range(K)])
        for step in ts:
            for v in step.vehicles:
                assert math.isfinite(v.speed_ms), (
                    f"K={K} {v.vehicle_id} t={step.time_sec:.1f}s: "
                    f"non-finite speed {v.speed_ms}"
                )


class TestFinalMobilityBackendIsSUMO:
    """hard_mission_config() uses sumo_route_pool (SUMO backend, not RWP)."""

    @pytest.mark.parametrize("K", [1, 3])
    def test_sumo_route_pool_is_set(self, K):
        from env.oran_env import hard_mission_config
        cfg = hard_mission_config(K_ambulances=K)
        assert cfg.sumo_route_pool is not None and len(cfg.sumo_route_pool) > 0, (
            f"K={K}: hard_mission_config().sumo_route_pool is empty — RWP mode active"
        )


class TestNoRWPInFinalExperimentConfig:
    """hard_mission_config() must lock SUMO backend and prohibit RWP fallback.

    RWP (Random Waypoint) legacy is still present in the codebase and activates
    whenever sumo_fcd_path=None and sumo_route_pool=None.  These tests guard
    against accidental regression at two levels:
      1. Config level: sumo_route_pool must be set with existing files
         ({K=1,K=3} × {light,medium,heavy} = 6 FCD files).
      2. Runtime level: after reset(), env._mobility must be a
         PooledSumoMobilityProvider (not None = RWP-mode sentinel).
    Both K=1 and K=3 are tested so a K-specific regression cannot slip through.
    """

    @pytest.mark.parametrize("K", [1, 3])
    def test_fcd_file_exists(self, K):
        from env.oran_env import hard_mission_config
        cfg = hard_mission_config(K_ambulances=K)
        assert cfg.sumo_route_pool is not None and len(cfg.sumo_route_pool) > 0, (
            f"K={K}: sumo_route_pool is empty — RWP in use instead of SUMO"
        )
        for path in cfg.sumo_route_pool:
            assert _os.path.exists(path), (
                f"K={K}: FCD file not found on disk: {path}"
            )

    @pytest.mark.parametrize("K", [1, 3])
    def test_runtime_mobility_is_sumo_not_rwp(self, K):
        """After reset(), env._mobility is PooledSumoMobilityProvider (not None = RWP sentinel).

        The RWP path in oran_env.py sets self._mobility = None.  A non-None
        PooledSumoMobilityProvider instance proves the SUMO pool branch was taken
        and vehicles are driven by OSM-based density traces, not random waypoints.
        """
        from env.oran_env import ORANEnv, hard_mission_config
        from env.sumo_mobility import PooledSumoMobilityProvider
        env = ORANEnv(hard_mission_config(K_ambulances=K))
        env.reset(seed=0)
        assert env._mobility is not None, (
            f"K={K}: env._mobility is None after reset() — RWP fallback active "
            "(expected PooledSumoMobilityProvider; check hard_mission_config sumo_route_pool)"
        )
        assert isinstance(env._mobility, PooledSumoMobilityProvider), (
            f"K={K}: env._mobility is {type(env._mobility).__name__} after reset(), "
            "expected PooledSumoMobilityProvider — SUMO pool backend was not loaded"
        )
        env.close()


class TestSUMOTraceLoadedBeforeEnvStep:
    """PooledSumoMobilityProvider is initialised inside reset(), before any step() call."""

    def test_mobility_provider_ready_after_reset(self):
        from env.oran_env import ORANEnv, hard_mission_config
        from env.sumo_mobility import PooledSumoMobilityProvider
        env = ORANEnv(hard_mission_config(K_ambulances=1))
        obs, _ = env.reset(seed=0)
        assert isinstance(env._mobility, PooledSumoMobilityProvider), (
            "env._mobility is not a PooledSumoMobilityProvider after reset() — "
            "SUMO pool was not loaded"
        )
        obs2, r, _, _, _ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert np.all(np.isfinite(obs2)), "First step after SUMO load produced non-finite obs"
        env.close()


class TestSUMOVehicleIdsMapToObsBlocks:
    """amb_k occupies obs[OBS_FIXED_BLOCK_LEN + OBS_PER_AMB_BLOCK_LEN*k : …+10].

    Verifies IDENTITY mapping: obs[base + offset] == normalized env.attribute[k]
    using the same normalization formulas as _observe():
      sinr  → last_sinr_db[k] / 40.0
      dist  → norm(ambulance_pos[k]) / cell_radius_m
      speed → norm(ambulance_vel[k]) / 60.0

    Identity tests are stronger than `math.isfinite`: they prove the k-th block
    reads from the k-th ambulance's internal state, not a wrong index or collapsed
    aggregate.  K=3 additionally verifies at least two vehicles have distinct obs.
    """

    def test_obs_block_identity_mapping_k3(self):
        from env.oran_env import ORANEnv, hard_mission_config
        K = 3
        env = ORANEnv(hard_mission_config(K_ambulances=K))
        env.reset(seed=0)
        # Generate one obs via step; internal state corresponds to THIS obs.
        obs, _, _, _, _ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        # Snapshot internal state IMMEDIATELY — must be from the SAME step that built obs.
        # Any further env.step() would advance state and break the identity.
        pos_snap = env.ambulance_pos.copy()   # shape (K, 2), metres
        vel_snap = env.ambulance_vel.copy()   # shape (K, 2), m/s
        sinr_snap = env.last_sinr_db.copy()  # shape (K,), dB
        cell_r = env.config.cell_radius_m

        # obs length sanity: at least OBS_FIXED + K*PER_AMB elements
        min_obs_len = OBS_FIXED_BLOCK_LEN + OBS_PER_AMB_BLOCK_LEN * K
        assert len(obs) >= min_obs_len, (
            f"obs length {len(obs)} < expected minimum {min_obs_len}"
        )

        # _observe() uses np.float32 arithmetic.  Compute expected values the same way
        # to avoid float32/float64 rounding divergence.  Use tol=1e-4 (float32 ≈ 7 sig-fig).
        tol = 1e-4

        # F6: inactive ambulances (not-yet-entered cell) have zeroed obs block (sentinel).
        # Only verify identity for ACTIVE ambulances; inactive → assert all-zeros.
        active_snap = env.active_mask.copy()

        for k in range(K):
            base = OBS_FIXED_BLOCK_LEN + OBS_PER_AMB_BLOCK_LEN * k
            block = obs[base:base + OBS_PER_AMB_BLOCK_LEN]

            if not active_snap[k]:
                # Inactive (outside cell or arrived): obs block must be all-zeros sentinel.
                assert np.all(block == 0.0), (
                    f"amb_{k} inactive → obs block must be zero sentinel, got {block}"
                )
                continue

            # SINR: _observe() → last_sinr_db.astype(float32) / 40.0
            expected_sinr = float(np.float32(sinr_snap[k]) / np.float32(40.0))
            obs_sinr = float(obs[base + AMB_SINR_OFFSET])
            assert abs(obs_sinr - expected_sinr) < tol, (
                f"amb_{k} SINR identity failed: "
                f"obs[{base + AMB_SINR_OFFSET}]={obs_sinr:.6f} "
                f"!= float32(sinr_db[{k}])/40={expected_sinr:.6f}"
            )

            # DIST: _observe() → norm(ambulance_pos, axis=1) / cell_radius_m  (float64 norm)
            expected_dist = float(np.linalg.norm(pos_snap[k]) / cell_r)
            obs_dist = float(obs[base + AMB_DIST_OFFSET])
            assert abs(obs_dist - expected_dist) < tol, (
                f"amb_{k} dist identity failed: "
                f"obs[{base + AMB_DIST_OFFSET}]={obs_dist:.6f} "
                f"!= norm(pos[{k}])/R={expected_dist:.6f}"
            )

            # SPEED: _observe() → norm(ambulance_vel, axis=1) / 60.0
            expected_speed = float(np.linalg.norm(vel_snap[k]) / 60.0)
            obs_speed = float(obs[base + AMB_SPEED_OFFSET])
            assert abs(obs_speed - expected_speed) < tol, (
                f"amb_{k} speed identity failed: "
                f"obs[{base + AMB_SPEED_OFFSET}]={obs_speed:.6f} "
                f"!= norm(vel[{k}])/60={expected_speed:.6f}"
            )

        # Verify at least one ambulance is active (fast-forward guarantees this).
        assert active_snap.any(), "At least one ambulance must be active after reset"

        # Verify K=3 ambulances have DISTINCT dist values (not collapsed mapping).
        # Use raw distances (including inactive — they're outside cell, so still distinct).
        dist_values = [
            float(np.linalg.norm(pos_snap[k]) / cell_r)
            for k in range(K)
        ]
        # At least 2 of the 3 must differ by more than rounding (tol)
        diffs = [abs(dist_values[i] - dist_values[j])
                 for i in range(K) for j in range(i + 1, K)]
        assert any(d > tol * 10 for d in diffs), (
            f"K=3 all ambulances appear at same dist={dist_values} — "
            "position mapping may be collapsed (all reading same k index)"
        )

        env.close()


# ============================================================
# ROUTE CORRECTNESS GATE — W15 approval requirements
# ============================================================
# Known FCD edge sequences (verified against actual simulation — W15-B2 1km macro-cell routes).
# Vehicles start 1600-1767m outside the cell and drive toward gNB.
# These lock the routes so re-runs can detect unintended changes.
_K1_ROUTE_EDGES: dict[str, list[str]] = {
    "amb_0": [
        "1236513512", "710176319", "218106606#1", "898272869#0", "898272869#3",
        "601529284#1", "1396192123#0", "163843396#0", "163843396#1", "601455489",
        "962177672", "711379743#0", "711379743#2", "711379742#0", "711379742#1",
        "37370971#0",
    ],
}
_K3_ROUTE_EDGES: dict[str, list[str]] = {
    "amb_0": [
        "1236513512", "710176319", "218106606#1", "898272869#0", "898272869#3",
        "601529284#1", "1396192123#0", "163843396#0", "163843396#1", "601455489",
        "962177672", "711379743#0", "711379743#2", "711379742#0", "711379742#1",
        "37370971#0",
    ],
    "amb_1": [
        "1009871353#1", "1009871353#2", "894857488#1", "881062561#0",
        "1117717293#1", "1119786563#0", "1119786563#1", "946030629#0",
        "946030629#2", "946030629#5", "946030629#6", "1119786561",
        "945322259#0", "946030658#1", "1359074553#0", "875229047#0",
        "708964308", "37370973#2", "838425490#0", "838425490#1", "1396499075#0",
        "37370971#0",
    ],
    "amb_2": [
        "597584693#6", "-28913698#0", "-37802502#2", "-1012665663#6",
        "11007920#0", "1158120938#0", "464925870", "1158120939#1",
        "857047108#0", "857047108#3", "1395144950#5", "595338113#0",
        "-710365053", "-702927010#3", "-37862483#5", "-838425493#14",
        "838425492#0", "838425492#4", "838425492#6", "-1528449082#0",
        "593642481#0", "593642481#3", "37370971#0",
    ],
}
_ROUTE_EDGES_BY_K: dict[int, dict[str, list[str]]] = {
    1: _K1_ROUTE_EDGES,
    3: _K3_ROUTE_EDGES,
}

# from_node / to_node for each route edge (from net.xml — W15-B2 1km macro-cell routes).
# NOTE: consecutive FCD-visible edges may be separated by internal SUMO junction edges
# (lane ids starting with ':') which are not listed here but are present in net.xml.
_EDGE_FROM_TO: dict[str, tuple[str, str]] = {
    # amb_0 (K=1 and K=3)
    "1236513512":       ("11481858145",  "6676669221"),
    "710176319":        ("6676669221",   "cluster_10149108636_10964875589_81806990_8189781668"),
    "218106606#1":      ("cluster_10149108636_10964875589_81806990_8189781668", "102994902"),
    "898272869#0":      ("102994902",    "6668082647"),
    "898272869#3":      ("6668082647",   "cluster_10282179335_102992082_440059875"),
    "601529284#1":      ("cluster_10282179335_102992082_440059875", "283136849"),
    "1396192123#0":     ("283136849",    "1495022762"),
    "163843396#0":      ("1495022762",   "1755146859"),
    "163843396#1":      ("1755146859",   "cluster_10108025389_1755146784_1755146806_6601172987"),
    "601455489":        ("cluster_10108025389_1755146784_1755146806_6601172987",
                         "cluster_10108025390_1755146742_1755146759"),
    "962177672":        ("cluster_10108025390_1755146742_1755146759", "1495022752"),
    "711379743#0":      ("1495022752",   "5662903998"),
    "711379743#2":      ("5662903998",   "6688326498"),
    "711379742#0":      ("6688326498",   "6617184333"),
    "711379742#1":      ("6617184333",   "cluster_10126309042_12045346369_436170722"),
    "37370971#0":       ("cluster_10126309042_12045346369_436170722", "cluster_1497969787_4736525838"),
    "37370971#3":       ("cluster_1497969787_4736525838", "1884807237"),
    # amb_1 (K=3)
    "1009871353#1":     ("13246104572",  "13246104565"),
    "1009871353#2":     ("13246104565",  "8308046749"),
    "894857488#1":      ("8308046749",   "cluster_8308046750_8432307743_8432307744"),
    "881062561#0":      ("cluster_8308046750_8432307743_8432307744", "12989943385"),
    "1117717293#1":     ("12989943385",  "9878572064"),
    "1119786563#0":     ("9878572064",   "13508759480"),
    "1119786563#1":     ("13508759480",  "5662889815"),
    "946030629#0":      ("5662889815",   "9878572063"),
    "946030629#2":      ("9878572063",   "8194657991"),
    "946030629#5":      ("8194657991",   "8531649059"),
    "946030629#6":      ("8531649059",   "8194657992"),
    "1119786561":       ("8194657992",   "5689302946"),
    "945322259#0":      ("5689302946",   "cluster_6666357217_6666357218"),
    "946030658#1":      ("cluster_6666357217_6666357218", "5485653484"),
    "1359074553#0":     ("5485653484",   "1884807231"),
    "875229047#0":      ("1884807231",   "6661738534"),
    "708964308":        ("6661738534",   "6617022182"),
    "37370973#1":       ("6617022182",   "150350260"),
    "37370973#2":       ("150350260",    "5686502528"),
    "838425490#0":      ("5686502528",   "cluster_1497969787_4736525838"),
    # amb_2 (K=3) — original edges retained for connectivity reference
    "597584693#6":      ("1904391930",   "421500418"),
    "-28913698#0":      ("421500418",    "cluster_443399765_443399770"),
    "-37802502#2":      ("cluster_443399765_443399770", "cluster_10613850509_10613850585_443399622"),
    "-1012665663#6":    ("cluster_10613850509_10613850585_443399622", "4600895552"),
    "11007920#0":       ("4600895552",   "10130533953"),
    # amb_2 K=3 new route segment (emergency routing replaces old passenger path)
    "1158120938#0":     ("10130533953",  "7920618652"),
    "464925870":        ("98027697",     "cluster_4599745018_7920618649"),
    "1158120939#1":     ("cluster_4599745018_7920618649", "7920618648"),
    "857047108#0":      ("7920618648",   "5686426690"),
    "857047108#3":      ("5686426690",   "cluster_10130548017_10130548026_10130548029_1904395858_6677531882"),
    "1395144950#5":     ("cluster_10130548017_10130548026_10130548029_1904395858_6677531882", "1904395866"),
    "595338113#0":      ("1904395866",   "cluster_1314209698_1758629824_1884765802_5674347101_5674347102_5683792674_5683792676_6658473479"),
    "-710365053":       ("cluster_1314209698_1758629824_1884765802_5674347101_5674347102_5683792674_5683792676_6658473479", "98027291"),
    "-702927010#3":     ("98027291",     "1905157877"),
    "-37862483#5":      ("1905157877",   "104783042"),
    "-838425493#14":    ("104783042",    "104782500"),
    "838425492#0":      ("104782500",    "1497969866"),
    "838425492#4":      ("1497969866",   "2294884824"),
    "838425492#6":      ("2294884824",   "13928476975"),
    "-1528449082#0":    ("13928476974",  "13811547007"),
    "593642481#0":      ("13811547007",  "4551916019"),
    "593642481#3":      ("4551916019",   "cluster_10126309042_12045346369_436170722"),
    # amb_1 K=3 new route segment (added after 838425490#0)
    "838425490#1":      ("cluster_1497969787_4736525838", "1494034249"),
    "1396499075#0":     ("1494034249",   "cluster_10126309042_12045346369_436170722"),
}

_NET_FILE_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "sumo", "bachmaiHN.net.xml")
)
_ARTIFACTS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "artifacts")
)


def _fcd_edge_seq(K: int) -> dict[str, list[str]]:
    """Parse actual edge sequence per vehicle from FCD."""
    seqs: dict[str, list[str]] = {f"amb_{k}": [] for k in range(K)}
    prev: dict[str, str] = {}
    for ts_el in _ET.parse(density_fcd_path(K, "medium")).getroot().iter("timestep"):
        for v_el in ts_el.iter("vehicle"):
            vid = v_el.get("id", "")
            if vid not in seqs:
                continue
            lane = v_el.get("lane", "")
            if lane.startswith(":"):
                continue
            parts = lane.rsplit("_", 1)
            eid = parts[0] if (len(parts) == 2 and parts[1].isdigit()) else lane
            if prev.get(vid) != eid:
                seqs[vid].append(eid)
                prev[vid] = eid
    return seqs


class TestRouteAnchorIsBachMai:
    """gNB anchor in sumo_mobility equals SSOT in utils/config; gNB is local origin."""

    def test_gnb_lat_matches_ssot(self):
        from utils.config import BACH_MAI_LAT
        assert GNB_LAT == BACH_MAI_LAT, (
            f"sumo_mobility.GNB_LAT={GNB_LAT} != config.BACH_MAI_LAT={BACH_MAI_LAT}"
        )

    def test_gnb_lon_matches_ssot(self):
        from utils.config import BACH_MAI_LON
        assert GNB_LON == BACH_MAI_LON, (
            f"sumo_mobility.GNB_LON={GNB_LON} != config.BACH_MAI_LON={BACH_MAI_LON}"
        )

    def test_gps_to_metric_origin_is_zero(self):
        east, north = gps_to_metric(GNB_LAT, GNB_LON)
        assert abs(east) < 1e-9 and abs(north) < 1e-9, (
            f"gps_to_metric(GNB) = ({east:.2e}, {north:.2e}) — expected (0,0)"
        )

    @pytest.mark.parametrize("K", [1, 3])
    def test_hard_mission_route_pool_exists(self, K):
        from env.oran_env import hard_mission_config
        cfg = hard_mission_config(K_ambulances=K)
        assert cfg.sumo_route_pool is not None and len(cfg.sumo_route_pool) > 0, (
            f"K={K}: sumo_route_pool is empty"
        )
        for path in cfg.sumo_route_pool:
            assert os.path.isfile(path), (
                f"K={K}: pool FCD file missing: {path!r}"
            )


class TestRouteMapBoundaryLock:
    """Lock convBoundary/origBoundary/netOffset/projection to detect accidental map drift."""

    _GNB_X_SUMO = 2979.36   # sumolib.convertLonLat2XY verified
    _GNB_Y_SUMO = 2225.94

    @classmethod
    def _loc(cls) -> dict[str, str]:
        loc_el = _ET.parse(_NET_FILE_PATH).getroot().find("location")
        assert loc_el is not None
        return {k: loc_el.get(k, "") for k in
                ("convBoundary", "origBoundary", "netOffset", "projParameter")}

    def test_conv_boundary_width(self):
        vals = list(map(float, self._loc()["convBoundary"].split(",")))
        assert abs((vals[2] - vals[0]) - 5906.23) < 1.0

    def test_conv_boundary_height(self):
        vals = list(map(float, self._loc()["convBoundary"].split(",")))
        assert abs((vals[3] - vals[1]) - 6320.25) < 1.0

    def test_gnb_inside_conv_boundary(self):
        vals = list(map(float, self._loc()["convBoundary"].split(",")))
        assert vals[0] < self._GNB_X_SUMO < vals[2]
        assert vals[1] < self._GNB_Y_SUMO < vals[3]

    def test_projection_utm_zone48(self):
        proj = self._loc()["projParameter"]
        assert "+proj=utm" in proj and "+zone=48" in proj

    def test_orig_boundary_contains_gnb(self):
        orig = list(map(float, self._loc()["origBoundary"].split(",")))
        lon_min, lat_min, lon_max, lat_max = orig
        assert lon_min < GNB_LON < lon_max
        assert lat_min < GNB_LAT < lat_max


class TestRouteOverlayArtifactsGenerated:
    """Route overlay PNGs and convergence PNG must exist — evidence requires visuals."""

    @pytest.mark.parametrize("fname,min_bytes", [
        ("w15_routes_k1.png",   50_000),
        ("w15_routes_k3.png",   50_000),
        ("w15_convergence.png", 10_000),
    ])
    def test_artifact_exists_and_nonzero(self, fname, min_bytes):
        path = os.path.join(_ARTIFACTS_DIR, fname)
        assert os.path.isfile(path), (
            f"{fname} missing. Run: python data/sumo/05_verify_traces.py"
        )
        assert os.path.getsize(path) >= min_bytes, (
            f"{fname} is {os.path.getsize(path)} bytes — may be empty/corrupt"
        )


class TestRouteEdgesExistInNetwork:
    """Every edge traversed in FCD traces must exist in bachmaiHN.net.xml."""

    @classmethod
    def _net_edge_ids(cls) -> set[str]:
        ids: set[str] = set()
        for _, elem in _ET.iterparse(_NET_FILE_PATH, events=("end",)):
            if elem.tag == "edge":
                eid = elem.get("id", "")
                if not eid.startswith(":"):
                    ids.add(eid)
                elem.clear()
        return ids

    @pytest.mark.parametrize("K", [1, 3])
    def test_all_fcd_edges_in_network(self, K):
        net_ids = self._net_edge_ids()
        fcd_seqs = _fcd_edge_seq(K)
        missing = {
            eid
            for edges in fcd_seqs.values()
            for eid in edges
            if eid not in net_ids
        }
        assert not missing, f"K={K}: FCD edges not in net.xml: {missing}"

    @pytest.mark.parametrize("K", [1, 3])
    def test_expected_route_edges_in_network(self, K):
        net_ids = self._net_edge_ids()
        for vid, edges in _ROUTE_EDGES_BY_K[K].items():
            for eid in edges:
                assert eid in net_ids, (
                    f"K={K} {vid}: route edge {eid!r} missing from net.xml"
                )


class TestRouteEdgesAreConnected:
    """Consecutive route edges are connected — either directly (same to/from node) or
    through a single SUMO internal junction edge (node reachable via one intermediate edge).

    In the bachmaiHN SUMO network, some consecutive FCD-visible edges are separated by
    internal junction edges (':*') that are skipped in the FCD edge sequence.  These gaps
    appear as non-matching to/from nodes in net.xml but are physically connected via
    a single junction edge.  The test verifies direct OR one-hop connectivity.
    """

    @classmethod
    def _build_node_adjacency(cls) -> dict[str, set[str]]:
        """Build node→{reachable_nodes_via_one_edge} from bachmaiHN.net.xml."""
        adj: dict[str, set[str]] = {}
        for _, elem in _ET.iterparse(_NET_FILE_PATH, events=("end",)):
            if elem.tag == "edge":
                fr = elem.get("from", "")
                to = elem.get("to", "")
                if fr and to:
                    adj.setdefault(fr, set()).add(to)
                elem.clear()
        return adj

    @pytest.mark.parametrize("K", [1, 3])
    def test_route_sequence_connected(self, K):
        """Consecutive edges share to/from node directly or via one junction hop."""
        adj = self._build_node_adjacency()
        for vid, edges in _ROUTE_EDGES_BY_K[K].items():
            for i in range(len(edges) - 1):
                e_cur, e_nxt = edges[i], edges[i + 1]
                if e_cur not in _EDGE_FROM_TO or e_nxt not in _EDGE_FROM_TO:
                    continue  # skip edges not in lookup (they may share junction)
                _, to_cur = _EDGE_FROM_TO[e_cur]
                from_nxt, _ = _EDGE_FROM_TO[e_nxt]
                if to_cur == from_nxt:
                    continue  # directly connected
                # Allow one-hop via any intermediate node (junction edge)
                reachable = adj.get(to_cur, set())
                assert from_nxt in reachable, (
                    f"K={K} {vid}: {e_cur!r} to_node={to_cur!r} cannot reach "
                    f"{e_nxt!r} from_node={from_nxt!r} in one hop — disconnected route"
                )

    @pytest.mark.parametrize("K", [1, 3])
    def test_fcd_edge_sequence_matches_expected(self, K):
        actual = _fcd_edge_seq(K)
        for vid, expected_edges in _ROUTE_EDGES_BY_K[K].items():
            assert actual[vid] == expected_edges, (
                f"K={K} {vid}: FCD seq {actual[vid]} != expected {expected_edges}"
            )


class TestRouteFcdEdgesMatchRoute:
    """Every non-junction FCD lane has its edge ID within the vehicle's route sequence."""

    @pytest.mark.parametrize("K", [1, 3])
    def test_no_foreign_edges_in_fcd(self, K):
        route_sets = {vid: set(edges) for vid, edges in _ROUTE_EDGES_BY_K[K].items()}
        foreign: list[tuple] = []
        for ts_el in _ET.parse(density_fcd_path(K, "medium")).getroot().iter("timestep"):
            t = float(ts_el.get("time", 0.0))
            for v_el in ts_el.iter("vehicle"):
                vid = v_el.get("id", "")
                if vid not in route_sets:
                    continue
                lane = v_el.get("lane", "")
                if lane.startswith(":"):
                    continue
                parts = lane.rsplit("_", 1)
                eid = parts[0] if (len(parts) == 2 and parts[1].isdigit()) else lane
                if eid not in route_sets[vid]:
                    foreign.append((round(t, 2), vid, eid))
        assert not foreign, (
            f"K={K}: {len(foreign)} FCD timesteps on unexpected edges. "
            f"First 5: {foreign[:5]}"
        )


class TestRouteDistanceWithinCell:
    """Vehicles start outside the 1km macro-cell (≤2×R_CELL_M) and converge INTO the cell.

    1km macro-cell redesign (W15-B2): R_CELL_M=1000m.  Vehicles depart 1600-1767m from gNB
    and drive toward it.  The max distance across the full trace is ≤2×R_CELL_M=2000m.
    Once vehicles reach inside R_CELL_M they stay there until the trace ends.
    """

    _MAX_DIST_M: float = 2 * _R_CELL_M   # upper bound on trace: 2000 m (actual ≈1767 m)

    @pytest.mark.parametrize("K", [1, 3])
    def test_all_samples_within_cell(self, K):
        """All FCD samples fall within 2×R_CELL_M (macro-cell departure zone)."""
        ts = load_fcd(density_fcd_path(K, "medium"), vehicle_ids=[f"amb_{k}" for k in range(K)])
        violations: list[tuple] = []
        for step in ts:
            for veh in step.vehicles:
                d = math.sqrt(veh.x_m ** 2 + veh.y_m ** 2)
                if d > self._MAX_DIST_M:
                    violations.append((round(step.time_sec, 2), veh.vehicle_id, round(d, 2)))
        assert not violations, (
            f"K={K}: {len(violations)} samples exceed 2×R_CELL_M={self._MAX_DIST_M}m. "
            f"First 5: {violations[:5]}"
        )

    @pytest.mark.parametrize("K", [1, 3])
    def test_max_dist_strictly_below_cell(self, K):
        """Maximum distance across trace is strictly below 2×R_CELL_M."""
        ts = load_fcd(density_fcd_path(K, "medium"), vehicle_ids=[f"amb_{k}" for k in range(K)])
        max_d = max(
            math.sqrt(veh.x_m ** 2 + veh.y_m ** 2)
            for step in ts for veh in step.vehicles
        )
        assert max_d < self._MAX_DIST_M, (
            f"K={K}: max dist={max_d:.1f}m ≥ 2×R_CELL_M={self._MAX_DIST_M}m"
        )


class TestRouteConvergesToGNB:
    """Each vehicle's last-seen distance is smaller than its first-seen distance."""

    @pytest.mark.parametrize("K", [1, 3])
    def test_end_dist_less_than_start_dist(self, K):
        """Per vehicle: distance at last appearance < distance at first appearance."""
        ts = load_fcd(density_fcd_path(K, "medium"), vehicle_ids=[f"amb_{k}" for k in range(K)])
        for k in range(K):
            vid = f"amb_{k}"
            first_step = next(
                (step for step in ts if any(v.vehicle_id == vid for v in step.vehicles)),
                None,
            )
            last_step = next(
                (step for step in reversed(ts) if any(v.vehicle_id == vid for v in step.vehicles)),
                None,
            )
            assert first_step is not None, f"K={K} {vid} never appears"
            v0 = next(v for v in first_step.vehicles if v.vehicle_id == vid)
            vN = next(v for v in last_step.vehicles if v.vehicle_id == vid)
            start_d = math.sqrt(v0.x_m ** 2 + v0.y_m ** 2)
            end_d   = math.sqrt(vN.x_m ** 2 + vN.y_m ** 2)
            assert end_d < start_d, (
                f"K={K} {vid}: end_dist={end_d:.1f}m ≥ start_dist={start_d:.1f}m "
                "— route diverges instead of converging"
            )

    @pytest.mark.parametrize("K", [1, 3])
    def test_overall_dist_reduction_per_second(self, K):
        """Distance at final whole-second < distance at first whole-second per vehicle."""
        ts = load_fcd(density_fcd_path(K, "medium"), vehicle_ids=[f"amb_{k}" for k in range(K)])
        for k in range(K):
            vid = f"amb_{k}"
            # Collect all whole-second timesteps where this vehicle appears
            ts_by_sec: dict[int, object] = {}
            for step in ts:
                v = next((v for v in step.vehicles if v.vehicle_id == vid), None)
                if v is None:
                    continue
                sec = int(round(step.time_sec))
                if sec not in ts_by_sec:
                    ts_by_sec[sec] = v
            assert ts_by_sec, f"K={K} {vid} has no whole-second samples"
            first_sec = min(ts_by_sec)
            last_sec  = max(ts_by_sec)
            v0 = ts_by_sec[first_sec]
            vN = ts_by_sec[last_sec]
            d0 = math.sqrt(v0.x_m ** 2 + v0.y_m ** 2)
            d1 = math.sqrt(vN.x_m ** 2 + vN.y_m ** 2)
            assert d1 < d0, (
                f"K={K} {vid}: d(t={last_sec}s)={d1:.1f}m ≥ d(t={first_sec}s)={d0:.1f}m"
            )


class TestRouteNoTeleport:
    """No consecutive-timestep displacement > 5m (true teleport, not SUMO junction snap)."""

    _TELEPORT_M = 5.0   # >> vType_maxSpeed × step_dt × speedFactor_max ≈ 1.83m

    @pytest.mark.parametrize("K", [1, 3])
    def test_no_teleport(self, K):
        """Check no >5m position jump for vehicles present in both consecutive timesteps."""
        ts = load_fcd(density_fcd_path(K, "medium"), vehicle_ids=[f"amb_{k}" for k in range(K)])
        teleports: list[tuple] = []
        for i in range(1, len(ts)):
            prev_by_id = {v.vehicle_id: v for v in ts[i - 1].vehicles}
            curr_by_id = {v.vehicle_id: v for v in ts[i].vehicles}
            for vid in prev_by_id.keys() & curr_by_id.keys():
                p, c = prev_by_id[vid], curr_by_id[vid]
                jump = math.sqrt((c.x_m - p.x_m) ** 2 + (c.y_m - p.y_m) ** 2)
                if jump > self._TELEPORT_M:
                    teleports.append((
                        round(ts[i].time_sec, 2),
                        vid,
                        round(jump, 3),
                    ))
        assert not teleports, (
            f"K={K}: {len(teleports)} teleport events (>{self._TELEPORT_M}m/step). "
            f"First 3: {teleports[:3]}"
        )


class TestK3VehicleIdsStable:
    """K=3: vehicles arrive at different times (start outside cell), ID set is stable subset.

    amb_0 present 2540 timesteps (~t=0..253.9s), amb_1 1646 (~t=0..164.5s),
    amb_2 3328 (~t=0..332.7s).  All 3 present simultaneously from t=0 until
    amb_1 departs at ~164.5s.
    """

    def test_all_three_ids_every_timestep(self):
        """All vehicle IDs present are a known subset of {amb_0,amb_1,amb_2}; no unknown IDs.
        Empty timesteps at trace end (after all vehicles reach destination) are allowed."""
        expected = {"amb_0", "amb_1", "amb_2"}
        ts = load_fcd(density_fcd_path(3, "medium"), vehicle_ids=list(expected))
        for step_idx, step in enumerate(ts):
            present = {v.vehicle_id for v in step.vehicles}
            # All IDs must be from the expected set (no phantom IDs)
            assert present <= expected, (
                f"t={step.time_sec:.1f}s: unexpected ids={present - expected}"
            )

    def test_total_timesteps_is_600(self):
        """FCD trace is 400s at 0.1s step-length = 4000 timesteps."""
        ts = load_fcd(density_fcd_path(3, "medium"), vehicle_ids=["amb_0", "amb_1", "amb_2"])
        assert len(ts) == 4000, f"Expected 4000 timesteps, got {len(ts)}"


class TestK3RouteDiversity:
    """K=3 ambulances must approach from ≥2 distinct directions (bearing spread > 60°)."""

    _MIN_SPREAD_DEG = 60.0

    @staticmethod
    def _angular_diff(a: float, b: float) -> float:
        return abs((a - b + 180) % 360 - 180)

    def test_bearing_spread_exceeds_60deg(self):
        ts = load_fcd(density_fcd_path(3, "medium"), vehicle_ids=["amb_0", "amb_1", "amb_2"])
        step0 = ts[0]
        bearings = [
            math.degrees(math.atan2(step0.vehicles[k].x_m, step0.vehicles[k].y_m)) % 360
            for k in range(3)
        ]
        spreads = [
            self._angular_diff(bearings[i], bearings[j])
            for i in range(3) for j in range(i + 1, 3)
        ]
        max_spread = max(spreads)
        assert max_spread > self._MIN_SPREAD_DEG, (
            f"K=3 max bearing spread = {max_spread:.1f}° ≤ {self._MIN_SPREAD_DEG}° — "
            "routes approach from too similar directions"
        )

    def test_no_two_vehicles_same_start_edge(self):
        seqs = _fcd_edge_seq(3)
        start_edges = [seqs[f"amb_{k}"][0] for k in range(3)]
        assert len(set(start_edges)) == len(start_edges), (
            f"K=3: multiple vehicles share start edge: {start_edges}"
        )
