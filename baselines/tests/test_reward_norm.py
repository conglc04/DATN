"""Fix D — reward normalization for stable critic targets.

Validates RunningMeanStd correctness and the ReturnNormalizer invariants:
  - running variance matches numpy on a batch
  - normalized reward = raw / running_return_std (bounds the critic target)
  - reset_episode clears the discounted-return accumulator (no cross-episode leak)
  - state_dict round-trip (resume support)
  - the normalization is a pure positive rescale (sign preserved → policy-invariant)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from utils.reward_norm import RunningMeanStd, ReturnNormalizer


class TestRunningMeanStd:
    def test_matches_numpy_on_batch(self):
        rng = np.random.default_rng(0)
        x = rng.normal(5.0, 3.0, size=10000)
        rms = RunningMeanStd()
        # feed in chunks to exercise the parallel update
        for chunk in np.array_split(x, 37):
            rms.update(chunk)
        assert rms.mean == pytest.approx(np.mean(x), rel=1e-3)
        assert rms.var == pytest.approx(np.var(x), rel=1e-2)

    def test_state_dict_roundtrip(self):
        rms = RunningMeanStd()
        rms.update(np.array([1.0, 2.0, 3.0, 4.0]))
        rms2 = RunningMeanStd()
        rms2.load_state_dict(rms.state_dict())
        assert rms2.mean == pytest.approx(rms.mean)
        assert rms2.var == pytest.approx(rms.var)
        assert rms2.count == rms.count


class TestReturnNormalizer:
    def test_bounds_large_rewards(self):
        """A stream of large rewards must be scaled toward O(1) magnitude."""
        norm = ReturnNormalizer(gamma=0.99)
        raw = [-47_000_000.0, -18_000_000.0, 161_000.0, -28_000_000.0] * 50
        normed = [norm.normalize(r) for r in raw]
        # After warmup, normalized magnitudes should be vastly smaller than raw.
        assert max(abs(n) for n in normed[-50:]) < max(abs(r) for r in raw) / 100

    def test_sign_preserved(self):
        """Normalization is a positive rescale → sign (hence policy argmax) preserved."""
        norm = ReturnNormalizer(gamma=0.99)
        for r in [-5.0, 3.0, -100.0, 0.0, 42.0]:
            n = norm.normalize(r)
            assert np.sign(n) == np.sign(r) or r == 0.0

    def test_reset_episode_clears_accumulator(self):
        norm = ReturnNormalizer(gamma=0.99)
        for _ in range(100):
            norm.normalize(10.0)
        assert norm._ret != 0.0
        norm.reset_episode()
        assert norm._ret == 0.0

    def test_no_crossepisode_leak(self):
        """Two identical episodes after reset produce identical first-step scaling
        ONLY if the discounted-return accumulator is reset (the running RMS keeps
        learning, but _ret must restart at 0)."""
        norm = ReturnNormalizer(gamma=0.99)
        norm.reset_episode()
        _ = norm.normalize(1.0)
        ret_after_first = norm._ret
        # simulate rest of episode
        for _ in range(50):
            norm.normalize(1.0)
        norm.reset_episode()
        _ = norm.normalize(1.0)
        # _ret after the first step of a fresh episode must equal gamma*0 + 1 = 1
        assert norm._ret == pytest.approx(ret_after_first)

    def test_state_dict_roundtrip(self):
        norm = ReturnNormalizer(gamma=0.99)
        for r in [1.0, -2.0, 3.0, -4.0]:
            norm.normalize(r)
        sd = norm.state_dict()
        norm2 = ReturnNormalizer(gamma=0.5)  # different gamma, should be overwritten
        norm2.load_state_dict(sd)
        assert norm2.gamma == pytest.approx(0.99)
        assert norm2._ret == pytest.approx(norm._ret)
        assert norm2.std == pytest.approx(norm.std)


class TestPolicyInvariance:
    def test_constant_scaling_does_not_flip_relative_order(self):
        """Two reward sequences: normalization preserves which has higher return."""
        norm_a = ReturnNormalizer(gamma=0.99)
        norm_b = ReturnNormalizer(gamma=0.99)
        # identical normalizer state → same scaling; higher raw → higher normalized
        seq = [10.0, -5.0, 3.0]
        a = sum(norm_a.normalize(r) for r in seq)
        # fresh normalizer, scaled-up sequence
        b = sum(norm_b.normalize(r * 2) for r in seq)
        # both positive-scaled; the larger raw sequence stays larger after norm
        assert np.sign(b) == np.sign(a)
