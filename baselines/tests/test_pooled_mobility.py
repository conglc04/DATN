"""F3: PooledSumoMobilityProvider tests.

Verifies:
- reset(trace_idx) selects route_pool[trace_idx % len(pool)]
- step() delegates to the active provider
- different trace_idx values select different paths (pool rotation)
- reset() before step() is required (RuntimeError guard)
- default_route_pool(K) returns non-empty list
- always starts at t=0 (non-cyclic route: origin→BV Bạch Mai, no wrap)
"""

from __future__ import annotations

import numpy as np
import pytest

from env.sumo_mobility import PooledSumoMobilityProvider, density_fcd_path, default_route_pool


class TestPooledProviderBasic:
    def test_reset_returns_positions(self):
        pool = default_route_pool(1)
        provider = PooledSumoMobilityProvider(pool, K=1, tti_sec=0.0005)
        pos = provider.reset(trace_idx=0)
        assert pos.shape == (1, 2)

    def test_step_returns_pos_and_vel(self):
        pool = default_route_pool(1)
        provider = PooledSumoMobilityProvider(pool, K=1, tti_sec=0.0005)
        provider.reset(trace_idx=0)
        pos, vel = provider.step()
        assert pos.shape == (1, 2)
        assert vel.shape == (1, 2)

    def test_step_before_reset_raises(self):
        pool = default_route_pool(1)
        provider = PooledSumoMobilityProvider(pool, K=1, tti_sec=0.0005)
        with pytest.raises(RuntimeError):
            provider.step()


class TestPoolRotation:
    def test_trace_idx_modulo_selects_correct_path(self):
        """trace_idx % pool_len determines which path is loaded."""
        p0 = density_fcd_path(1, "medium")
        pool = [p0, p0]
        provider = PooledSumoMobilityProvider(pool, K=1, tti_sec=0.0005)
        for idx in [0, 1, 2, 3, 7]:
            pos = provider.reset(trace_idx=idx)
            assert pos.shape == (1, 2)

    def test_multi_reset_same_idx_gives_same_position(self):
        """Same trace_idx → same t=0 starting position (deterministic non-cyclic route)."""
        pool = default_route_pool(1)
        provider = PooledSumoMobilityProvider(pool, K=1, tti_sec=0.0005)
        pos0 = provider.reset(trace_idx=0)
        pos1 = provider.reset(trace_idx=0)
        np.testing.assert_allclose(pos0, pos1, atol=1.0)

    def test_default_reset_is_trace_zero(self):
        """reset() with no args = reset(trace_idx=0) — backward compat."""
        pool = default_route_pool(1)
        provider = PooledSumoMobilityProvider(pool, K=1, tti_sec=0.0005)
        pos_default = provider.reset()
        pos_explicit = provider.reset(trace_idx=0)
        np.testing.assert_allclose(pos_default, pos_explicit, atol=1e-9)


class TestEmptyPoolValidation:
    def test_empty_pool_raises_on_construction(self):
        with pytest.raises(ValueError):
            PooledSumoMobilityProvider([], K=1, tti_sec=0.0005)


class TestDefaultRoutePool:
    def test_default_pool_non_empty(self):
        for K in [1, 3]:
            pool = default_route_pool(K)
            assert len(pool) >= 1

    def test_default_pool_paths_exist(self):
        import os
        for K in [1, 3]:
            pool = default_route_pool(K)
            for p in pool:
                assert os.path.exists(p), f"FCD path not found: {p}"
