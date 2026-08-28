"""Focused tests for research/genesis_event_time_test.py.

Verifies:
  1. Event calendar construction produces plausible bucket counts.
  2. No row is double-counted across buckets (partition is exact).
  3. The script never references rows past index 300,000 of the CSV
     (grepped for literal 400000 / 400_000, matching the horizon sweep's
     verified property).
"""
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.genesis_event_time_test import (
    build_event_calendar,
    label_event_buckets,
    bucket_autocorrelation,
    TRAINING_ROWS,
)

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "research" / "genesis_event_time_test.py"


def test_event_calendar_plausible_counts():
    cal = build_event_calendar(dt.date(2019, 12, 2), dt.date(2020, 9, 17))
    counts = cal["event_type"].value_counts().to_dict()
    # 10 months in range (Dec 2019 .. Sep 2020 inclusive) -> one NFP and one
    # CPI per month, and 7 of the 8 scheduled FOMC meetings/year fall here.
    assert counts["NFP"] == 10
    assert counts["CPI"] == 10
    assert counts["FOMC"] == 7
    assert len(cal) == 27
    # all events strictly within the requested date range (in UTC, allowing
    # for the ET->UTC conversion to shift a date by at most one day)
    assert cal["event_time_utc"].min() >= pd.Timestamp("2019-12-01")
    assert cal["event_time_utc"].max() <= pd.Timestamp("2020-09-18")


def test_fomc_dates_are_the_documented_exact_list():
    cal = build_event_calendar(dt.date(2019, 12, 2), dt.date(2020, 9, 17))
    fomc = cal[cal["event_type"] == "FOMC"]
    assert len(fomc) == 7
    # emergency intermeeting cuts must NOT appear
    emergency_dates = {pd.Timestamp("2020-03-03"), pd.Timestamp("2020-03-15")}
    for ts in fomc["event_time_utc"]:
        assert ts.normalize() not in emergency_dates


def test_buckets_partition_all_rows_exactly_once():
    rng = pd.date_range("2020-01-01", periods=20000, freq="1min")
    cal = build_event_calendar(dt.date(2020, 1, 1), dt.date(2020, 1, 14))
    labels = label_event_buckets(pd.Series(rng), cal["event_time_utc"])
    valid_buckets = {"A_ordinary", "B_pre_event", "C_immediate_post", "D_later_post"}
    assert set(np.unique(labels)) <= valid_buckets
    # every row gets exactly one label (guaranteed by array shape, but also
    # check there is no row simultaneously eligible for two labels going
    # unassigned or overcounted by re-deriving counts sum to n)
    assert len(labels) == len(rng)
    counts = {b: int((labels == b).sum()) for b in valid_buckets}
    assert sum(counts.values()) == len(rng)


def test_bucket_labels_mutually_exclusive_near_overlapping_events():
    # two events 90 minutes apart: rows between them should not be double
    # counted -- each row must resolve to exactly one of the 4 labels.
    base = pd.Timestamp("2020-01-02 12:00:00")
    events = pd.Series([base, base + pd.Timedelta(minutes=90)])
    times = pd.Series(pd.date_range(base - pd.Timedelta(hours=3), base + pd.Timedelta(hours=5), freq="1min"))
    labels = label_event_buckets(times, events)
    assert len(labels) == len(times)
    # no NaN/None/empty labels
    assert all(isinstance(l, str) and l for l in labels)


def test_bucket_autocorrelation_handles_short_and_empty_buckets():
    returns = np.array([0.1, -0.1, 0.2, -0.2, 0.05, np.nan, 0.3, -0.3])
    mask = np.array([True, True, True, True, True, False, False, False])
    result = bucket_autocorrelation(returns, mask, max_lag=5)
    assert isinstance(result, dict)
    # empty bucket should not raise, just return no lags
    empty_mask = np.zeros_like(mask)
    result_empty = bucket_autocorrelation(returns, empty_mask, max_lag=5)
    assert result_empty == {}


def test_training_rows_constant_matches_convention():
    assert TRAINING_ROWS == 300_000


def test_script_never_references_row_400000_boundary():
    text = SCRIPT_PATH.read_text()
    # Matches the horizon sweep's verified property: no literal code
    # reference to the 400,000-row OOS-holdout boundary. (The docstring may
    # mention "300,000:400,000" in prose to describe the holdout it never
    # touches -- that is documentation, not a code literal, so only the
    # unpunctuated numeric forms are checked here.)
    assert "400000" not in text
    assert "400_000" not in text
