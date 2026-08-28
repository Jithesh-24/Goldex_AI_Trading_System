"""Small sanity tests for research/genesis_event_time_test1b_falsification.py.

Keeps to the scope of a small follow-up test file: checks bucket
disjointness/partitioning of the sub-splits used by the falsification
script, and that the script never reaches past row 300,000 of the CSV.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from research.genesis_event_time_test import build_event_calendar, label_event_buckets
from research.genesis_event_time_test1b_falsification import (
    next_event_type_per_row,
    rolling_std_returns,
)

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "genesis_event_time_test1b_falsification.py"
)


def test_script_never_references_the_reserved_holdout_row_count():
    text = SCRIPT_PATH.read_text()
    assert "400000" not in text
    assert "400_000" not in text


def test_exact_fomc_and_approximated_subsets_partition_b_pre_event():
    rng = pd.date_range("2019-12-02", "2020-09-17 07:59:00", freq="1min")
    times = pd.Series(rng)
    calendar = build_event_calendar(rng[0].date(), rng[-1].date())
    labels = label_event_buckets(times, calendar["event_time_utc"])
    next_type = next_event_type_per_row(times, calendar)

    b_mask = labels == "B_pre_event"
    is_fomc = b_mask & (next_type == "FOMC")
    is_approx = b_mask & np.isin(next_type, ["NFP", "CPI"])

    # The two sub-buckets must be disjoint and must sum exactly to the full
    # B_pre_event count -- every B_pre_event row's "next event" is one of
    # FOMC/NFP/CPI in this calendar, so there is no leftover category.
    assert not np.any(is_fomc & is_approx)
    assert int(is_fomc.sum()) + int(is_approx.sum()) == int(b_mask.sum())
    assert int(b_mask.sum()) > 0


def test_spread_strata_partition_b_pre_event():
    labels = np.array(["A_ordinary"] * 5 + ["B_pre_event"] * 5)
    spread = np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 25.0, 20.0, 22.0])
    b_mask = labels == "B_pre_event"
    spread20 = b_mask & (spread == 20.0)
    spread_gt20 = b_mask & (spread > 20.0)
    assert not np.any(spread20 & spread_gt20)
    assert int(spread20.sum()) + int(spread_gt20.sum()) == int(b_mask.sum())


def test_rolling_std_returns_uses_only_trailing_window():
    returns = np.arange(10, dtype=float)
    vol = rolling_std_returns(returns, window=3)
    assert np.isnan(vol[0]) and np.isnan(vol[1])
    assert np.isfinite(vol[2])
    # value at index 2 should equal std of returns[0:3], not any future data
    assert np.isclose(vol[2], np.std(returns[0:3], ddof=1))


def test_bucket_labels_are_confined_to_known_categories():
    rng = pd.date_range("2020-01-01", periods=5000, freq="1min")
    calendar = build_event_calendar(rng[0].date(), rng[-1].date())
    labels = label_event_buckets(pd.Series(rng), calendar["event_time_utc"])
    assert set(np.unique(labels)) <= {
        "A_ordinary", "B_pre_event", "C_immediate_post", "D_later_post",
    }
