"""Tests for intelligence/applicability.py -- the mechanical hard-floor gates."""
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pytest

from contracts.market_state import DataQuality, FeedHealthState, MarketState
from intelligence.applicability import (
    MIN_HISTORY_REQUIRED,
    apply_applicability,
    default_checks_for,
    history_length_check,
    market_state_check,
)
from intelligence.evidence import EvidenceValue
from intelligence.evidence_sources import build_default_registry

ALL_NAMES = [
    "momentum_scalar", "path_pca_projection", "multiscale_vol_ratio",
    "vol_regime_transition", "garch_conditional_variance",
    "kalman_filtered_velocity", "kalman_innovation", "rolling_skew",
    "rolling_excess_kurtosis",
]


def _synthetic_closes(n=500, seed=0):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.5, n)
    return 100.0 + np.cumsum(returns)


def _market_state(*, data_quality=DataQuality.VALID, market_closed=False):
    now = datetime.now(timezone.utc)
    return MarketState(
        symbol="XAUUSD",
        source="synthetic_replay",
        sequence=1,
        market_timestamp=now,
        ingestion_timestamp=now,
        processing_timestamp=now,
        bid=2000.0,
        ask=2000.2,
        mid=2000.1,
        spread=0.2,
        data_quality=data_quality,
        tick_count_60s=10,
        tick_count_300s=50,
        tick_rate_per_sec=1.0,
        market_closed=market_closed,
        feed_health=FeedHealthState.CONNECTED,
        last_tick_age_sec=0.5,
    )


# -- Consistency between this task's history gate and Task 3's internal gate --

@pytest.mark.parametrize("name", ALL_NAMES)
def test_min_history_consistent_with_wrappers(name):
    """This task's MIN_HISTORY_REQUIRED must agree exactly with
    evidence_sources.py's own internal insufficient-history gate: just below
    the threshold, the wrapper itself already returns confidence 0.0; at/above
    it, the wrapper returns a real (confidence=1.0) value. If these two gates
    disagreed, either this module would zero out a value Task 3 considered
    good, or (worse) let through a value Task 3 itself doesn't trust."""
    registry = build_default_registry()
    spec = registry.specs()[name]
    required = MIN_HISTORY_REQUIRED[name]
    closes_full = _synthetic_closes(500)

    below = closes_full[: required - 1]
    result_below = spec.compute(below)
    assert result_below == EvidenceValue(None, 0.0, name), (
        f"{name}: wrapper did not self-gate at required-1 length; "
        f"MIN_HISTORY_REQUIRED is inconsistent with the wrapper's own threshold"
    )

    at = closes_full[:required]
    result_at = spec.compute(at)
    assert result_at.confidence == 1.0, (
        f"{name}: wrapper still gates at the required length; "
        f"MIN_HISTORY_REQUIRED is too low"
    )

    # And applicability's own check agrees on both sides.
    check = history_length_check(name)
    assert check.check(below, None) is False
    assert check.check(at, None) is True


# -- history_length_check in isolation --

def test_history_length_check_unknown_source_passes():
    check = history_length_check("not_a_real_source")
    assert check.check(np.array([1.0]), None) is True


# -- market_state_check in isolation --

def test_market_state_check_none_passes():
    check = market_state_check()
    assert check.check(np.array([1.0, 2.0]), None) is True


def test_market_state_check_valid_open_passes():
    check = market_state_check()
    ms = _market_state()
    assert check.check(np.array([1.0, 2.0]), ms) is True


def test_market_state_check_invalid_data_quality_fails():
    check = market_state_check()
    ms = _market_state(data_quality=DataQuality.INVALID)
    assert check.check(np.array([1.0, 2.0]), ms) is False


def test_market_state_check_market_closed_fails():
    check = market_state_check()
    ms = _market_state(market_closed=True)
    assert check.check(np.array([1.0, 2.0]), ms) is False


# -- apply_applicability: the actual hard floor --

def test_apply_applicability_insufficient_history_forces_zero_confidence():
    """A source with insufficient history has confidence forced to 0.0 --
    consistent with (not contradicting) Task 3's own internal gate, which
    already produced EvidenceValue(None, 0.0, ...) here."""
    name = "momentum_scalar"
    required = MIN_HISTORY_REQUIRED[name]
    closes = _synthetic_closes(500)[: required - 1]
    registry = build_default_registry()
    raw = registry.specs()[name].compute(closes)

    gated = apply_applicability(name, raw, closes, market_state=None)
    assert gated.confidence == 0.0


def test_apply_applicability_market_closed_forces_zero_confidence_despite_full_confidence_value():
    """The case Task 3's own wrappers CANNOT catch: ample history, a
    confidence=1.0 raw value, but a MarketState flagged market_closed=True.
    Without this module's independent gate, the raw value would slip through
    to the downstream posterior unmodified."""
    name = "momentum_scalar"
    closes = _synthetic_closes(500)
    registry = build_default_registry()
    raw = registry.specs()[name].compute(closes)
    assert raw.value is not None
    assert raw.confidence == 1.0

    ms = _market_state(market_closed=True)
    gated = apply_applicability(name, raw, closes, market_state=ms)
    assert gated.confidence == 0.0
    assert gated.value == raw.value  # value preserved, only confidence zeroed
    assert gated.source_name == raw.source_name


def test_apply_applicability_data_quality_invalid_forces_zero_confidence_despite_full_confidence_value():
    name = "garch_conditional_variance"
    closes = _synthetic_closes(500)
    registry = build_default_registry()
    raw = registry.specs()[name].compute(closes)
    assert raw.confidence == 1.0

    ms = _market_state(data_quality=DataQuality.INVALID)
    gated = apply_applicability(name, raw, closes, market_state=ms)
    assert gated.confidence == 0.0


def test_apply_applicability_passes_through_when_all_checks_pass():
    name = "kalman_filtered_velocity"
    closes = _synthetic_closes(500)
    registry = build_default_registry()
    raw = registry.specs()[name].compute(closes)

    ms = _market_state()
    gated = apply_applicability(name, raw, closes, market_state=ms)
    assert gated == raw


@pytest.mark.parametrize("name", ALL_NAMES)
def test_default_checks_for_includes_history_and_market_state(name):
    checks = default_checks_for(name)
    check_names = [c.name for c in checks]
    assert f"{name}_min_history" in check_names
    assert "market_state_valid" in check_names
