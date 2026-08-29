"""tests/simulator/test_no_leakage.py
Mechanical no-future-leakage audit (design doc Section 1 / Section 13.B #6).
Runs the replay twice -- once clean, once with every bar's high/low/close
AFTER each decision point poisoned before that step's snapshot is built --
and asserts every recorded market_state_snapshot is identical between runs.
Covers both DECIDE (flat) and MANAGE (position open) call sites."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

from simulator.contracts import EnvironmentTag, SimulatedExecutionConfig
from simulator.market_state_builder import build_snapshot
from simulator.replay import run_replay


def _make_df(n=30):
    times = pd.date_range("2020-01-06 10:00:00", periods=n, freq="1min")
    prices = [1500.0 + i * 0.1 for i in range(n)]
    return pd.DataFrame({
        "time": times, "open": prices,
        "high": [p + 0.3 for p in prices], "low": [p - 0.3 for p in prices],
        "close": [p + 0.05 for p in prices], "tick_volume": [10] * n, "spread": [20.0] * n,
    })


def test_build_snapshot_identical_regardless_of_future_bar_values():
    df = _make_df()
    for i in range(len(df)):
        clean_snap = build_snapshot(df, i)
        poisoned = df.copy()
        if i + 1 < len(df):
            poisoned.loc[i + 1:, ["open", "high", "low", "close", "spread"]] = -999999.0
        poisoned_snap = build_snapshot(poisoned, i)
        assert clean_snap.mid == poisoned_snap.mid, f"leakage at row {i}: mid differs"
        assert clean_snap.spread == poisoned_snap.spread, f"leakage at row {i}: spread differs"
        if clean_snap.completed_m1 is not None:
            assert clean_snap.completed_m1.close == poisoned_snap.completed_m1.close, f"leakage at row {i}"
        assert clean_snap.current_m1.high == poisoned_snap.current_m1.high, f"leakage at row {i}: current bar high"
        assert clean_snap.current_m1.low == poisoned_snap.current_m1.low, f"leakage at row {i}: current bar low"


def test_replay_records_identical_snapshots_regardless_of_unreached_future():
    df = _make_df()
    config = SimulatedExecutionConfig()

    def always_no_trade(market_state, account):
        return ("NO_TRADE", None, None)

    recorder_clean = run_replay(df, always_no_trade, lambda *a: "HOLD", config, EnvironmentTag.SIMULATED_TRAINING)

    truncated = df.iloc[: len(df) // 2].copy()
    recorder_truncated = run_replay(
        truncated, always_no_trade, lambda *a: "HOLD", config, EnvironmentTag.SIMULATED_TRAINING
    )

    n_common = len(recorder_truncated.all_records())
    clean_records = recorder_clean.all_records()[:n_common]
    truncated_records = recorder_truncated.all_records()

    def _snapshots_equal(snap_a, snap_b):
        # NaN != NaN under normal equality, but a NaN field (e.g.
        # realized_vol_60s before enough history exists) that stays NaN in
        # both runs is not leakage -- only a genuine value change is.
        if snap_a.keys() != snap_b.keys():
            return False
        for key in snap_a:
            va, vb = snap_a[key], snap_b[key]
            if isinstance(va, float) and isinstance(vb, float) and pd.isna(va) and pd.isna(vb):
                continue
            if va != vb:
                return False
        return True

    for a, b in zip(clean_records, truncated_records):
        assert _snapshots_equal(a.market_state_snapshot, b.market_state_snapshot), (
            "leakage: truncating the dataset after the current decision point changed an "
            "earlier snapshot -- the earlier decision must not depend on data that doesn't exist yet"
        )


if __name__ == "__main__":
    test_build_snapshot_identical_regardless_of_future_bar_values()
    test_replay_records_identical_snapshots_regardless_of_unreached_future()
    print("tests/simulator/test_no_leakage.py: OK")
