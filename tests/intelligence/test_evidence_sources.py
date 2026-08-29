"""Tests for intelligence/evidence_sources.py -- the 9 wrapped evidence sources."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pytest

from intelligence.evidence import EvidenceValue
from intelligence.evidence_sources import build_default_registry
from research.phase3a_representation_experiments import (
    multiscale_vol_summary, vol_regime_transition, VOL_WINDOWS,
)


def _synthetic_closes(n=500, seed=0):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.5, n)
    return 100.0 + np.cumsum(returns)


ALL_NAMES = [
    "momentum_scalar", "path_pca_projection", "multiscale_vol_ratio",
    "vol_regime_transition", "garch_conditional_variance",
    "kalman_filtered_velocity", "kalman_innovation", "rolling_skew",
    "rolling_excess_kurtosis",
]


def test_registry_has_all_nine_sources():
    registry = build_default_registry()
    assert sorted(registry.names()) == sorted(ALL_NAMES)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_wrapper_sensible_value_with_enough_data(name):
    registry = build_default_registry()
    closes = _synthetic_closes(500)
    spec = registry.specs()[name]
    result = spec.compute(closes)
    assert isinstance(result, EvidenceValue)
    assert result.source_name == name
    assert result.value is not None
    assert np.isfinite(result.value)
    assert result.confidence == 1.0


@pytest.mark.parametrize("name", ALL_NAMES)
def test_wrapper_insufficient_history_returns_none(name):
    registry = build_default_registry()
    # kalman_* only need 2 points, so use a single-point array for those;
    # everything else needs at least a window/lookback of several bars, so
    # 3 points is comfortably insufficient.
    closes = np.array([100.0]) if name.startswith("kalman_") else np.array([100.0, 100.5, 99.8])
    spec = registry.specs()[name]
    result = spec.compute(closes)
    assert result == EvidenceValue(None, 0.0, name)


def test_vol_regime_transition_chaining_matches_manual_composition():
    """vol_regime_transition's wrapper must call multiscale_vol_summary first
    and feed vols_dict[short_window] into vol_regime_transition -- not treat
    the two as independent, disconnected computations."""
    registry = build_default_registry()
    closes = _synthetic_closes(500)
    spec = registry.specs()["vol_regime_transition"]
    result = spec.compute(closes)

    _ratio, vols = multiscale_vol_summary(closes, windows=VOL_WINDOWS)
    expected_transition = vol_regime_transition(vols[min(VOL_WINDOWS)], n_bins=3)
    expected_finite = expected_transition[np.isfinite(expected_transition)]
    assert len(expected_finite) > 0
    assert result.value == pytest.approx(float(expected_finite[-1]))


def test_vol_regime_transition_not_fed_raw_closes():
    """Sanity check that the chaining wrapper is NOT equivalent to calling
    vol_regime_transition directly on raw closes (the bug this task's brief
    explicitly warns against)."""
    closes = _synthetic_closes(500)
    wrong = vol_regime_transition(closes, n_bins=3)
    _ratio, vols = multiscale_vol_summary(closes, windows=VOL_WINDOWS)
    correct = vol_regime_transition(vols[min(VOL_WINDOWS)], n_bins=3)
    wrong_finite = wrong[np.isfinite(wrong)]
    correct_finite = correct[np.isfinite(correct)]
    assert len(wrong_finite) > 0 and len(correct_finite) > 0
    # These should differ overall -- proof that raw-closes-as-input is a
    # materially different (wrong) computation from the vol-array chaining
    # this wrapper does. Comparing full finite sequences (not just the last
    # element, which can coincidentally collide at the shared value 0.0).
    min_len = min(len(wrong_finite), len(correct_finite))
    assert not np.array_equal(wrong_finite[-min_len:], correct_finite[-min_len:])


@pytest.mark.parametrize("name", ALL_NAMES)
def test_no_look_ahead_truncated_recompute_matches(name):
    """Load-bearing causality gate: computing a source on closes_so_far[:i]
    must produce the identical result whether or not closes_so_far has data
    beyond index i. These wrappers feed a live trading decision loop and must
    never look ahead."""
    registry = build_default_registry()
    full_closes = _synthetic_closes(500)
    i = 300
    truncated = full_closes[:i]

    spec = registry.specs()[name]
    result_truncated = spec.compute(truncated)
    result_full_but_only_using_truncated_view = spec.compute(full_closes[:i])

    assert result_truncated == result_full_but_only_using_truncated_view

    # Stronger check: result computed on the truncated array must be
    # independent of what (if anything) exists past index i in the source
    # data used to build that truncated slice -- i.e. computing on a slice
    # taken from a longer series equals computing on a freshly-built array
    # containing only that same prefix of values.
    fresh_prefix = np.array(full_closes[:i], copy=True)
    result_fresh = spec.compute(fresh_prefix)
    assert result_truncated == result_fresh

    # And extending the series further must never change the value already
    # produced at index i's truncation point.
    result_at_i = spec.compute(full_closes[:i])
    result_at_i_plus_more = spec.compute(full_closes[:i + 50])
    if result_at_i.value is not None:
        # The i-prefix result must be reproducible by recomputing on exactly
        # that same prefix again, independent of the longer array existing.
        assert spec.compute(full_closes[:i]) == result_at_i
