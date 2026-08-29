"""Conditional applicability gates for the GOLDEX intelligence layer
(mandate Section 12, item 3).

Task 2's `EvidenceSourceSpec.known_failure_conditions` is qualitative prose.
This module makes a small slice of that machine-checkable: a set of boolean
`ApplicabilityCheck`s an `EvidenceValue` passes through BEFORE it's trusted
by any downstream consumer (Task 5's posterior update). This is a hard
floor, not a learned applicability model -- if any check fails, confidence
is forced to 0.0 regardless of what the source itself computed.

Two check families, deliberately minimal:

1. Minimum history length. Task 3's wrappers (`intelligence/evidence_sources.py`)
   already gate on insufficient history internally, returning
   `EvidenceValue(None, 0.0, name)` before doing real work -- this module does
   NOT re-derive a second, possibly-inconsistent threshold. `MIN_HISTORY_REQUIRED`
   below mirrors, bar for bar, the exact length checks each wrapper performs
   (same lookback/window constants, same wrapper-specific cushions such as
   vol_regime_transition's n_bins*5). It exists as a second, independent line
   of defense -- e.g. for a future evidence source whose own wrapper doesn't
   self-gate -- not as the primary enforcement mechanism for the 9 current
   sources. See test_applicability.py::test_min_history_consistent_with_wrappers
   for the consistency proof, and NOTE below for why no source can slip
   through on history alone.

2. MarketState.data_quality / market_closed. Task 3's wrappers only ever see
   `closes_so_far` -- they have no visibility into feed health or session
   status. This check is the genuinely new gate: a value computed from
   perfectly sufficient history is still not trustworthy if the MarketState
   it's being used against is flagged `data_quality=INVALID` or
   `market_closed=True`.

NOTE on "slip-through": because MIN_HISTORY_REQUIRED mirrors Task 3's own
gates exactly, no source can slip through the history check that Task 3
didn't already block -- confirmed by test, not assumed. The MarketState
check is where independent gating actually changes the outcome: a source
can compute a full-confidence value from ample history while the
MarketState it's paired with is simultaneously invalid.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from contracts.market_state import DataQuality, MarketState
from intelligence.evidence import EvidenceValue
from research.phase3a_representation_experiments import (
    MOMENTUM_LOOKBACK,
    PATH_WINDOW,
    VOL_WINDOWS,
)
from research.phase4_distributional_mechanism import WINDOW as DIST_WINDOW

# Mirrors the exact minimum-length checks in intelligence/evidence_sources.py's
# wrapper closures, one entry per registered source name. `vol_regime_transition`
# additionally needs n_bins*5 finite vol points beyond the raw window minimum,
# with n_bins=3 as hardcoded in that wrapper.
MIN_HISTORY_REQUIRED: dict[str, int] = {
    "momentum_scalar": MOMENTUM_LOOKBACK + 1,
    "path_pca_projection": PATH_WINDOW + 1,
    "multiscale_vol_ratio": max(VOL_WINDOWS) + 1,
    "vol_regime_transition": max(VOL_WINDOWS) + 3 * 5,
    "garch_conditional_variance": 31,
    "kalman_filtered_velocity": 2,
    "kalman_innovation": 2,
    "rolling_skew": DIST_WINDOW + 1,
    "rolling_excess_kurtosis": DIST_WINDOW + 1,
}


@dataclass
class ApplicabilityCheck:
    """One mechanical, boolean gate. `check` returns True if the source is
    applicable (passes), False if it should be zeroed out."""

    name: str
    check: Callable[[np.ndarray, Optional[MarketState]], bool]


def history_length_check(source_name: str) -> ApplicabilityCheck:
    """Gate on `len(closes_so_far) >= MIN_HISTORY_REQUIRED[source_name]`.
    Unknown source names pass unconditionally (no requirement on record)."""

    required = MIN_HISTORY_REQUIRED.get(source_name)

    def check(closes_so_far: np.ndarray, market_state: Optional[MarketState] = None) -> bool:
        if required is None:
            return True
        return len(closes_so_far) >= required

    return ApplicabilityCheck(name=f"{source_name}_min_history", check=check)


def market_state_check() -> ApplicabilityCheck:
    """Gate on MarketState.data_quality/market_closed. Passes (True) if no
    MarketState is supplied -- the check is opt-in per call site."""

    def check(closes_so_far: np.ndarray, market_state: Optional[MarketState] = None) -> bool:
        if market_state is None:
            return True
        if market_state.market_closed:
            return False
        if market_state.data_quality == DataQuality.INVALID:
            return False
        return True

    return ApplicabilityCheck(name="market_state_valid", check=check)


def default_checks_for(source_name: str) -> list[ApplicabilityCheck]:
    """The standard gate set applied to every source: its own min-history
    check plus the shared MarketState check."""

    return [history_length_check(source_name), market_state_check()]


def apply_applicability(
    source_name: str,
    evidence_value: EvidenceValue,
    closes_so_far: np.ndarray,
    market_state: Optional[MarketState] = None,
    checks: Optional[list[ApplicabilityCheck]] = None,
) -> EvidenceValue:
    """Runs `checks` (default: `default_checks_for(source_name)`) against the
    given inputs. If any check fails, returns a new EvidenceValue with the
    same value/source_name but confidence forced to 0.0 -- the hard floor
    beneath Task 5's learned trust layer. If all checks pass, returns
    `evidence_value` unchanged."""

    if checks is None:
        checks = default_checks_for(source_name)

    for c in checks:
        if not c.check(closes_so_far, market_state):
            return EvidenceValue(evidence_value.value, 0.0, evidence_value.source_name)

    return evidence_value
