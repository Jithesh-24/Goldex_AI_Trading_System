"""python3 tests/test_feature_warmup_missing.py

Verifies LiveFeatureEngine.on_m1_close's NaN-check-to-quality-flag mapping
never lets a short-history or live-incompatible feature report a
plausible-looking VALID number: it must report WARMING_UP (insufficient
bars, computed as NaN by the underlying rolling/window math) or
UNAVAILABLE (live_compatible=False in the registry, e.g. spread-history
features with no per-bar spread tracking live) instead.

completed_m1 is level-held (market/state_engine.py:155 -- the latest
completed bar rides on every MarketState tick after the boundary, not
just the boundary tick itself), so both loops below edge-trigger on
state.completed_m1.start_time changing, mirroring the already-reviewed
pattern in tests/test_live_engine.py (calling on_m1_close once per bar,
not once per tick, matters for correctness here too: on_m1_close's
DailyBuffer.record() call in the vol_percentile_252 override is an
update-in-place per calendar day, but over-calling it burns cycles
needlessly and diverges from how the real engine is driven)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone, timedelta

from contracts.tick import Tick
from market.state_engine import StateEngine
from features.live_engine import LiveFeatureEngine


def _run_ticks(engine, live, n_ticks, seconds_per_tick=1, price_drift=0.0):
    """Feed n_ticks synthetic ticks through engine/live, calling
    on_m1_close once per completed bar (edge-triggered on start_time
    change, not level-held completed_m1). Returns the final snapshot."""
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    snapshot = None
    last_bar_start = None
    for i in range(n_ticks):
        t = base + timedelta(seconds=i * seconds_per_tick)
        tick = Tick(symbol="GOLD.i#", market_timestamp=t, ingestion_timestamp=t,
                    bid=2000.0 + i * price_drift, ask=2000.2 + i * price_drift,
                    mid=2000.1, spread=0.2, source="synthetic_replay", internal_seq=i)
        state = engine.on_tick(tick)
        if state is None:
            continue
        live.on_tick(state)
        if state.current_m1 is not None and state.completed_m1 is not None \
                and state.completed_m1.start_time != last_bar_start:
            last_bar_start = state.completed_m1.start_time
            snapshot = live.on_m1_close(engine.completed_m1_window(480))
    return snapshot


def test_short_history_reports_warming_up_not_plausible_numbers():
    engine = StateEngine("GOLD.i#")
    live = LiveFeatureEngine(engine, daily_bootstrap_csv=None)

    snapshot = _run_ticks(engine, live, n_ticks=200)  # only ~3 M1 bars --
    # nowhere near the 240/252-bar windows

    assert snapshot is not None
    long_window_features = ["ret_240", "return_skew_240", "hurst_240", "changepoint_intensity_240"]
    for fid in long_window_features:
        if fid in snapshot:
            value, quality = snapshot[fid]
            assert quality in ("WARMING_UP", "UNAVAILABLE"), \
                f"{fid} claimed {quality} with only 3 bars of history"
            assert value is None, f"{fid} reported a non-None value ({value}) while {quality}"


def test_spread_history_features_marked_unavailable_live():
    engine = StateEngine("GOLD.i#")
    live = LiveFeatureEngine(engine, daily_bootstrap_csv=None)

    snapshot = _run_ticks(engine, live, n_ticks=2000, price_drift=0.001)

    assert snapshot is not None
    for fid in ("spread_change_1", "spread_percentile_252", "spread_volatility_60"):
        if fid in snapshot:
            value, quality = snapshot[fid]
            assert quality == "UNAVAILABLE", \
                f"{fid} should be UNAVAILABLE live (no per-bar spread history), got {quality}"
            assert value is None


def test_all_registry_features_never_go_valid_before_their_own_warmup_bars():
    """Broader than the two tests above (which only name a handful of
    features): every feature in the registry declares its own
    warmup_bars (contracts/feature_schema.py: required int field). Rather
    than trust that live_engine.py's blanket NaN-check-to-WARMING_UP
    mapping happens to cover the handful of features the brief names by
    hand, walk every descriptor and check the invariant directly against
    however many bars actually got fed into on_m1_close: if a feature's
    own declared warmup_bars exceeds that count, it must NOT report
    VALID -- regardless of family, regardless of whether it happens to be
    on the brief's named list. This is what would have caught a
    live_engine.py NaN-check bug for a feature the brief's sketch didn't
    happen to enumerate (Task 17's established pattern: review catches
    under-scoped test coverage)."""
    engine = StateEngine("GOLD.i#")
    live = LiveFeatureEngine(engine, daily_bootstrap_csv=None)

    snapshot = _run_ticks(engine, live, n_ticks=200)  # ~3 completed M1 bars
    assert snapshot is not None

    bars_fed = engine.completed_m1_window(480)
    num_bars = len(bars_fed)
    assert 0 < num_bars < 10, f"expected only a handful of bars, got {num_bars}"

    checked = 0
    for feature_id, descriptor in live._descriptors.items():
        if feature_id not in snapshot:
            continue  # not computed by on_m1_close (e.g. baseline_v1 family,
                       # out of live_engine.py's scope -- see module docstring)
        value, quality = snapshot[feature_id]
        if descriptor.warmup_bars > num_bars:
            checked += 1
            assert quality != "VALID", (
                f"{feature_id} (warmup_bars={descriptor.warmup_bars}) reported VALID "
                f"with only {num_bars} bars of history -- a plausible-looking number "
                f"before real data exists"
            )
            assert quality in ("WARMING_UP", "UNAVAILABLE"), \
                f"{feature_id} reported unexpected quality flag {quality!r}"
            assert value is None, \
                f"{feature_id} reported a non-None value ({value}) while {quality}"

    # Sanity: this loop must actually have exercised the long-warmup
    # features (ret_240 etc. all have warmup_bars far above num_bars) --
    # otherwise the assertions above would be vacuously true.
    assert checked >= 10, f"only checked {checked} short-history features -- test is too weak"


if __name__ == "__main__":
    test_short_history_reports_warming_up_not_plausible_numbers()
    test_spread_history_features_marked_unavailable_live()
    test_all_registry_features_never_go_valid_before_their_own_warmup_bars()
    print("tests/test_feature_warmup_missing.py: OK")
