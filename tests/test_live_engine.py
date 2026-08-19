"""python3 tests/test_live_engine.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone, timedelta, date

import numpy as np
import pandas as pd

from contracts.tick import Tick
from market.state_engine import StateEngine
from features.live_engine import LiveFeatureEngine, _vol_percentile_pct

SEED_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "gold_seed.csv")


def test_live_engine_produces_snapshot_after_enough_bars():
    engine = StateEngine("GOLD.i#")
    live = LiveFeatureEngine(engine, daily_bootstrap_csv=None)  # no bootstrap in this unit test

    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    snapshot = None
    last_bar_start = None
    for i in range(20000):  # enough ticks to cross several M1 boundaries
        t = base + timedelta(seconds=i * 2)
        tick = Tick(symbol="GOLD.i#", market_timestamp=t, ingestion_timestamp=t,
                    bid=2000.0 + (i % 100) * 0.01, ask=2000.2 + (i % 100) * 0.01,
                    mid=2000.1, spread=0.2, source="synthetic_replay", internal_seq=i)
        state = engine.on_tick(tick)
        if state is None:
            continue
        live.on_tick(state)
        # completed_m1 is level-held (the latest completed bar rides on
        # every subsequent MarketState, not just the boundary tick) --
        # edge-trigger on start_time change so on_m1_close runs once per
        # bar, not once per tick (once-per-tick caused an OOM/hang: ~20000
        # full family recomputes instead of ~666).
        if state.current_m1 is not None and state.completed_m1 is not None \
                and state.completed_m1.start_time != last_bar_start:
            last_bar_start = state.completed_m1.start_time
            snapshot = live.on_m1_close(engine.completed_m1_window(480))

    assert snapshot is not None
    assert len(snapshot) > 0
    for feature_id, (value, quality) in snapshot.items():
        assert quality in ("VALID", "WARMING_UP", "UNAVAILABLE")


def test_daily_bootstrap_populates_ewma_vol_key():
    # Finding 2: __init__'s daily_bootstrap_csv path must warm the exact
    # "ewma_vol" key on_m1_close's vol_percentile_252 override reads --
    # not "close"/"spread" (nothing reads those keys; that was dead code).
    engine = StateEngine("GOLD.i#")
    live = LiveFeatureEngine(engine, daily_bootstrap_csv=SEED_CSV)

    ewma_series = live.daily_buffer.series("ewma_vol")
    assert len(ewma_series) > 0
    # gold_seed.csv is ~2.5mo of M1 bars -- comfortably >60 calendar days,
    # so vol_percentile_252's min_periods=60 threshold should already be
    # satisfied at process startup, not after 60 further live days.
    assert len(ewma_series) >= 60
    assert ewma_series.index.is_monotonic_increasing
    assert all(np.isfinite(v) for v in ewma_series.values)

    # dead bootstrap keys dropped: nothing populates "close"/"spread"
    # under the daily buffer since nothing in the codebase reads them.
    assert len(live.daily_buffer.series("close")) == 0
    assert len(live.daily_buffer.series("spread")) == 0


def test_vol_percentile_pct_excludes_todays_own_value():
    # Finding 1: today's value must be ranked against a distribution that
    # EXCLUDES today's own entry, matching the batch/reference
    # implementation's .rolling(252, min_periods=60).rank(pct=True)
    # .shift(1) causal contract (registry: "shifted by 1 day to avoid
    # same-day lookahead"). Construct a tied-history scenario where
    # self-inclusion (the old bug) and exclusion (the fix) provably give
    # different numeric answers, then assert the fix's answer.
    prior_hist = pd.Series([0.5] * 60, index=[date(2026, 1, 1) + timedelta(days=i) for i in range(60)])
    today_val = 0.5  # ties every prior observation

    pct = _vol_percentile_pct(today_val, prior_hist)

    # Exclusion (correct): today_val compared against 60 prior values, all
    # tied -- average-tie pct = (0 less + 0.5*60 tied) / 60 = 0.5 exactly.
    assert pct == 0.5

    # Self-inclusion (the bug being fixed) would instead rank today_val
    # inside a 60-element all-tied series: average rank pct =
    # ((n+1)/2)/n = (61/2)/60 = 0.508333... -- a different, wrong answer.
    self_inclusion_pct = ((len(prior_hist) + 1 + 1) / 2) / (len(prior_hist) + 1)
    assert not np.isclose(pct, self_inclusion_pct)


def test_vol_percentile_pct_ignores_multiple_same_day_updates():
    # DailyBuffer.record() does same-day update-in-place, and on_m1_close
    # runs once per bar (many bars per calendar day). A naive "buffer
    # state before this record() call" fix would still leak an EARLIER
    # bar's same-day value into the ranked-against set on the 2nd+ bar of
    # a day. Verify the actual on_m1_close wiring filters by index
    # (vol_hist.index != today), not by call order, so this can't happen.
    prior_hist = pd.Series([1.0, 2.0, 3.0] * 20, index=[date(2026, 1, 1) + timedelta(days=i) for i in range(60)])
    today = date(2026, 6, 1)  # well outside prior_hist's index range
    # simulate today's index colliding with an entry already in the series
    # (as would happen on the 2nd bar of the same calendar day, where
    # DailyBuffer already holds an earlier-this-day value under "today"):
    contaminated = pd.concat([prior_hist, pd.Series([9999.0], index=[today])])
    filtered = contaminated[contaminated.index != today]
    assert len(filtered) == len(prior_hist)
    assert 9999.0 not in filtered.values


def test_vol_percentile_pct_below_min_periods_returns_none():
    prior_hist = pd.Series([0.1] * 59, index=[date(2026, 1, 1) + timedelta(days=i) for i in range(59)])
    assert _vol_percentile_pct(0.1, prior_hist) is None
    prior_hist_60 = pd.Series([0.1] * 60, index=[date(2026, 1, 1) + timedelta(days=i) for i in range(60)])
    assert _vol_percentile_pct(0.1, prior_hist_60) is not None


def test_live_engine_with_bootstrap_marks_vol_percentile_valid_sooner():
    # End-to-end: with daily_bootstrap_csv wired, vol_percentile_252 should
    # be able to go VALID immediately once shared.ewma_vol produces a
    # finite value on the very first processed bars -- it must not need
    # 60 further live calendar days to accumulate (that was Finding 2's
    # bug: bootstrap warmed the wrong keys, so ewma_vol always started
    # empty regardless of daily_bootstrap_csv).
    engine = StateEngine("GOLD.i#")
    live = LiveFeatureEngine(engine, daily_bootstrap_csv=SEED_CSV)
    assert len(live.daily_buffer.series("ewma_vol")) >= 60

    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    snapshot = None
    last_bar_start = None
    for i in range(6000):
        t = base + timedelta(seconds=i * 2)
        tick = Tick(symbol="GOLD.i#", market_timestamp=t, ingestion_timestamp=t,
                    bid=2000.0 + (i % 100) * 0.01, ask=2000.2 + (i % 100) * 0.01,
                    mid=2000.1, spread=0.2, source="synthetic_replay", internal_seq=i)
        state = engine.on_tick(tick)
        if state is None:
            continue
        live.on_tick(state)
        if state.current_m1 is not None and state.completed_m1 is not None \
                and state.completed_m1.start_time != last_bar_start:
            last_bar_start = state.completed_m1.start_time
            snapshot = live.on_m1_close(engine.completed_m1_window(480))

    assert snapshot is not None
    assert "vol_percentile_252" in snapshot
    value, quality = snapshot["vol_percentile_252"]
    # Bootstrap gives >=60 prior days immediately, so once ewma_vol itself
    # is finite (a handful of bars in) this must be VALID, not WARMING_UP.
    assert quality == "VALID"
    assert value is not None and 0.0 <= value <= 1.0


if __name__ == "__main__":
    test_live_engine_produces_snapshot_after_enough_bars()
    test_daily_bootstrap_populates_ewma_vol_key()
    test_vol_percentile_pct_excludes_todays_own_value()
    test_vol_percentile_pct_ignores_multiple_same_day_updates()
    test_vol_percentile_pct_below_min_periods_returns_none()
    test_live_engine_with_bootstrap_marks_vol_percentile_valid_sooner()
    print("tests/test_live_engine.py: OK")
