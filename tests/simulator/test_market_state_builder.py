"""tests/simulator/test_market_state_builder.py"""
import os
import sys
from datetime import timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import math

import pandas as pd

from contracts.market_state import DataQuality
from simulator.market_state_builder import build_snapshot


def _make_df():
    times = pd.date_range("2020-01-06 10:00:00", periods=5, freq="1min")
    return pd.DataFrame({
        "time": times,
        "open": [1500.0, 1501.0, 1502.0, 1503.0, 1504.0],
        "high": [1500.5, 1501.5, 1502.5, 1503.5, 1504.5],
        "low": [1499.5, 1500.5, 1501.5, 1502.5, 1503.5],
        "close": [1501.0, 1502.0, 1503.0, 1504.0, 1505.0],
        "tick_volume": [10, 12, 11, 9, 13],
        "spread": [20.0, 20.0, 21.0, 19.0, 20.0],
    })


def pytest_approx(x, tol=1e-9):
    return x


def test_snapshot_mid_uses_current_bar_open_only():
    df = _make_df()
    snap = build_snapshot(df, 2)
    assert snap.mid == 1502.0  # row 2's open, not high/low/close
    assert snap.spread == pytest_approx(21.0 * 0.01)


def test_snapshot_never_reads_current_row_high_low_close():
    df = _make_df()
    poisoned = df.copy()
    poisoned.loc[2, ["high", "low", "close"]] = [999999.0, -999999.0, 999999.0]
    snap_clean = build_snapshot(df, 2)
    snap_poisoned = build_snapshot(poisoned, 2)
    assert snap_clean.mid == snap_poisoned.mid
    assert snap_clean.current_m1.high == snap_poisoned.current_m1.high
    assert snap_clean.current_m1.low == snap_poisoned.current_m1.low
    assert snap_clean.current_m1.close == snap_poisoned.current_m1.close


def test_snapshot_never_reads_future_rows():
    df = _make_df()
    poisoned = df.copy()
    poisoned.loc[3:, ["open", "high", "low", "close", "spread"]] = -1.0
    snap_clean = build_snapshot(df, 2)
    snap_poisoned = build_snapshot(poisoned, 2)
    assert snap_clean.mid == snap_poisoned.mid
    assert snap_clean.spread == snap_poisoned.spread
    assert snap_clean.completed_m1.close == snap_poisoned.completed_m1.close


def test_snapshot_completed_m1_uses_previous_row():
    df = _make_df()
    snap = build_snapshot(df, 2)
    assert snap.completed_m1.close == 1502.0  # row 1's close
    assert snap.completed_m1.complete is True
    assert snap.current_m1.complete is False


def test_snapshot_first_row_has_no_completed_bar():
    df = _make_df()
    snap = build_snapshot(df, 0)
    assert snap.completed_m1 is None


def test_snapshot_flags_valid_row_as_valid():
    df = _make_df()
    snap = build_snapshot(df, 2)
    assert snap.data_quality == DataQuality.VALID


def test_snapshot_flags_nan_price_as_invalid():
    df = _make_df()
    poisoned = df.copy()
    poisoned.loc[2, "open"] = float("nan")
    snap = build_snapshot(poisoned, 2)
    assert snap.data_quality == DataQuality.INVALID
    # bid/ask must still satisfy the contract's gt=0, and fall back to the
    # nearest prior valid mid (row 1's open == 1501.0) rather than crashing.
    assert snap.bid > 0
    assert snap.ask > 0
    assert not math.isnan(snap.mid)
    assert snap.mid == 1501.0


def test_snapshot_flags_negative_price_as_invalid():
    df = _make_df()
    poisoned = df.copy()
    poisoned.loc[2, "open"] = -5.0
    snap = build_snapshot(poisoned, 2)
    assert snap.data_quality == DataQuality.INVALID
    assert snap.bid > 0
    assert snap.ask > 0


def test_snapshot_flags_zero_price_as_invalid():
    df = _make_df()
    poisoned = df.copy()
    poisoned.loc[2, "open"] = 0.0
    snap = build_snapshot(poisoned, 2)
    assert snap.data_quality == DataQuality.INVALID


def test_snapshot_flags_anomalous_spread_as_invalid():
    df = _make_df()
    poisoned = df.copy()
    # Normal spread column is ~20 points (0.20 price); make row 4's spread
    # 100x that so it stands out clearly against the trailing 60s window
    # built from rows 0..3.
    poisoned.loc[4, "spread"] = 2000.0
    snap = build_snapshot(poisoned, 4)
    assert snap.data_quality == DataQuality.INVALID
    # Price itself is fine -- only the spread is anomalous.
    assert snap.mid == 1504.0


def test_snapshot_normal_spread_not_flagged():
    df = _make_df()
    snap = build_snapshot(df, 4)
    assert snap.data_quality == DataQuality.VALID


if __name__ == "__main__":
    test_snapshot_mid_uses_current_bar_open_only()
    test_snapshot_never_reads_current_row_high_low_close()
    test_snapshot_never_reads_future_rows()
    test_snapshot_completed_m1_uses_previous_row()
    test_snapshot_first_row_has_no_completed_bar()
    test_snapshot_flags_valid_row_as_valid()
    test_snapshot_flags_nan_price_as_invalid()
    test_snapshot_flags_negative_price_as_invalid()
    test_snapshot_flags_zero_price_as_invalid()
    test_snapshot_flags_anomalous_spread_as_invalid()
    test_snapshot_normal_spread_not_flagged()
    print("tests/simulator/test_market_state_builder.py: OK")
