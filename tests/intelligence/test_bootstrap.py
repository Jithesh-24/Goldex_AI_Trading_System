"""tests/intelligence/test_bootstrap.py -- Task 11: analytical SL/TP/sizing
bootstrap. Confirms (a) SL/TP distance scales with a synthetic volatility
input per the documented formula (SL_VOL_MULTIPLIER/TP_VOL_MULTIPLIER *
realized_vol_60s * mid), and (b) the sizing bootstrap stays a normal caller
of Phase 1's EXISTING seam -- `simulator.engine.open_position` still enforces
INSUFFICIENT_MARGIN on whatever size this bootstrap proposes; nothing here
silently bypasses that check."""
import os
import sys
from datetime import datetime, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from contracts.market_state import DataQuality, FeedHealthState, MarketState
from intelligence.bootstrap import (
    SL_VOL_MULTIPLIER,
    TP_VOL_MULTIPLIER,
    analytical_sizing_bootstrap,
    analytical_sltp_bootstrap,
)
from intelligence.fast_tier import Hypothesis
from simulator.contracts import AccountState, Side, SimulatedExecutionConfig
from simulator.engine import open_position


def _market_state(mid=2000.0, spread=0.2, realized_vol_60s=0.0005):
    now = datetime.now(timezone.utc)
    return MarketState(
        symbol="XAUUSD", source="synthetic_replay", sequence=1,
        market_timestamp=now, ingestion_timestamp=now, processing_timestamp=now,
        bid=mid - spread / 2, ask=mid + spread / 2, mid=mid, spread=spread,
        data_quality=DataQuality.VALID, tick_count_60s=10, tick_count_300s=50,
        tick_rate_per_sec=1.0, market_closed=False, feed_health=FeedHealthState.CONNECTED,
        last_tick_age_sec=0.5, realized_vol_60s=realized_vol_60s,
    )


# --- analytical_sltp_bootstrap ---------------------------------------------

def test_sltp_long_shaped_when_belief_nonnegative():
    ms = _market_state(mid=2000.0, realized_vol_60s=0.001)
    hyp = Hypothesis(0.5, 0.1, [])
    sl, tp = analytical_sltp_bootstrap(hyp, ms)
    assert sl < ms.mid < tp


def test_sltp_short_shaped_when_belief_negative():
    ms = _market_state(mid=2000.0, realized_vol_60s=0.001)
    hyp = Hypothesis(-0.5, 0.1, [])
    sl, tp = analytical_sltp_bootstrap(hyp, ms)
    assert tp < ms.mid < sl


def test_sltp_matches_documented_formula():
    mid = 2000.0
    vol = 0.001
    ms = _market_state(mid=mid, realized_vol_60s=vol)
    hyp = Hypothesis(0.5, 0.1, [])
    sl, tp = analytical_sltp_bootstrap(hyp, ms)
    expected_sl_distance = SL_VOL_MULTIPLIER * vol * mid
    expected_tp_distance = TP_VOL_MULTIPLIER * vol * mid
    assert sl == mid - expected_sl_distance
    assert tp == mid + expected_tp_distance


def test_sltp_distance_scales_with_volatility():
    mid = 2000.0
    hyp = Hypothesis(0.5, 0.1, [])

    ms_low = _market_state(mid=mid, realized_vol_60s=0.0005)
    sl_low, tp_low = analytical_sltp_bootstrap(hyp, ms_low)
    sl_dist_low = mid - sl_low
    tp_dist_low = tp_low - mid

    ms_high = _market_state(mid=mid, realized_vol_60s=0.0010)  # doubled vol
    sl_high, tp_high = analytical_sltp_bootstrap(hyp, ms_high)
    sl_dist_high = mid - sl_high
    tp_dist_high = tp_high - mid

    # Doubling vol_estimate should roughly (exactly, since the formula is
    # linear in vol) double the SL/TP distance.
    assert sl_dist_high == pytest.approx(2.0 * sl_dist_low)
    assert tp_dist_high == pytest.approx(2.0 * tp_dist_low)


def test_sltp_returns_none_when_vol_missing():
    ms = _market_state(realized_vol_60s=None)
    hyp = Hypothesis(0.5, 0.1, [])
    sl, tp = analytical_sltp_bootstrap(hyp, ms)
    assert sl is None and tp is None


def test_sltp_returns_none_when_vol_nonpositive():
    ms = _market_state(realized_vol_60s=0.0)
    hyp = Hypothesis(0.5, 0.1, [])
    sl, tp = analytical_sltp_bootstrap(hyp, ms)
    assert sl is None and tp is None


# --- analytical_sizing_bootstrap -------------------------------------------

def test_sizing_bootstrap_reproduces_open_position_default_formula():
    """Confirms this is a THIN wrapper: the size it proposes equals exactly
    what simulator.engine.open_position computes on its own (size=None
    default) for the same config/account -- see simulator/engine.py:34-35."""
    config = SimulatedExecutionConfig(starting_balance=10000.0, risk_fraction_of_equity=0.02)
    now = datetime.now(timezone.utc)
    account = AccountState.initial(config, now)
    sizing_bootstrap = analytical_sizing_bootstrap(config)
    hyp = Hypothesis(0.5, 0.1, [])

    proposed_size = sizing_bootstrap(hyp, account)
    expected_size = account.equity * config.risk_fraction_of_equity
    assert proposed_size == expected_size


def test_sizing_bootstrap_size_accepted_by_open_position_when_margin_sufficient():
    """The proposed size, when actually passed through Phase 1's existing
    open_position seam, succeeds normally under an ordinary account."""
    config = SimulatedExecutionConfig(starting_balance=10000.0, risk_fraction_of_equity=0.01)
    ms = _market_state(mid=2000.0, realized_vol_60s=0.001)
    account = AccountState.initial(config, ms.market_timestamp)
    sizing_bootstrap = analytical_sizing_bootstrap(config)
    hyp = Hypothesis(0.5, 0.1, [])
    size = sizing_bootstrap(hyp, account)

    sl_price, tp_price = analytical_sltp_bootstrap(hyp, ms)
    position, new_account, rejection_reason = open_position(
        ms, account, Side.LONG, sl_price, tp_price, config, size=size,
    )
    assert rejection_reason is None
    assert position is not None
    assert position.size == size


def test_sizing_bootstrap_does_not_bypass_insufficient_margin_check():
    """Confirms this bootstrap doesn't silently bypass Phase 1's existing
    INSUFFICIENT_MARGIN check -- a small starting balance with the default
    risk fraction still proposes a size open_position correctly rejects
    once leverage/margin math makes it unaffordable relative to margin_free,
    demonstrated here by driving margin_free to (near) zero via a tiny
    account balance and requesting the position on top of an already-used
    margin. The bootstrap is a normal caller of the existing seam: it
    proposes a size, and Phase 1's own open_position is still the sole
    enforcer of the margin check."""
    config = SimulatedExecutionConfig(starting_balance=10000.0, risk_fraction_of_equity=0.01,
                                       leverage=1.0)
    ms = _market_state(mid=2000.0, realized_vol_60s=0.001)
    account = AccountState.initial(config, ms.market_timestamp)
    # Simulate margin already exhausted by a prior position (margin_free=0).
    exhausted_account = AccountState(
        balance=account.balance, equity=account.equity,
        margin_used=account.equity, margin_free=0.0,
        exposure=0.0, open_position_id="existing-position",
        simulation_timestamp=account.simulation_timestamp,
        realized_pnl_total=0.0, peak_equity=account.peak_equity, drawdown=0.0,
        currency=account.currency,
    )
    sizing_bootstrap = analytical_sizing_bootstrap(config)
    hyp = Hypothesis(0.5, 0.1, [])
    size = sizing_bootstrap(hyp, exhausted_account)
    assert size > 0  # bootstrap still proposes a normal, non-zero size

    sl_price, tp_price = analytical_sltp_bootstrap(hyp, ms)
    position, new_account, rejection_reason = open_position(
        ms, exhausted_account, Side.LONG, sl_price, tp_price, config, size=size,
    )
    assert position is None
    assert rejection_reason == "INSUFFICIENT_MARGIN"
    assert new_account is exhausted_account  # unchanged on rejection


if __name__ == "__main__":
    test_sltp_long_shaped_when_belief_nonnegative()
    test_sltp_short_shaped_when_belief_negative()
    test_sltp_matches_documented_formula()
    test_sltp_distance_scales_with_volatility()
    test_sltp_returns_none_when_vol_missing()
    test_sltp_returns_none_when_vol_nonpositive()
    test_sizing_bootstrap_reproduces_open_position_default_formula()
    test_sizing_bootstrap_size_accepted_by_open_position_when_margin_sufficient()
    test_sizing_bootstrap_does_not_bypass_insufficient_margin_check()
    print("tests/intelligence/test_bootstrap.py: OK")
