"""tests/simulator/test_engine.py"""
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simulator.contracts import Side, AccountState, SimulatedExecutionConfig, Position, PositionOutcome
from simulator.engine import (
    open_position, close_position, check_liquidation, to_position_view, mark_to_market,
)


class _FakeMarketState:
    def __init__(self, mid, spread, market_timestamp, realized_vol_60s=0.001):
        self.mid = mid
        self.spread = spread
        self.market_timestamp = market_timestamp
        self.realized_vol_60s = realized_vol_60s


def test_open_position_reduces_margin_free():
    config = SimulatedExecutionConfig(starting_balance=10000.0, leverage=100.0, risk_fraction_of_equity=0.01)
    ts = datetime(2020, 1, 1, tzinfo=timezone.utc)
    account = AccountState.initial(config, ts)
    ms = _FakeMarketState(mid=1500.0, spread=0.2, market_timestamp=ts)
    position, new_account = open_position(ms, account, Side.LONG, sl_price=1495.0, tp_price=None, config=config)
    assert position.side == Side.LONG
    assert position.entry_price > 1500.0  # crossed the spread
    assert new_account.margin_used > 0.0
    assert new_account.margin_free < account.margin_free
    assert new_account.open_position_id == position.position_id


def test_close_position_updates_balance_with_realized_pnl_minus_cost():
    config = SimulatedExecutionConfig(starting_balance=10000.0, leverage=100.0, risk_fraction_of_equity=0.01)
    ts0 = datetime(2020, 1, 1, tzinfo=timezone.utc)
    account = AccountState.initial(config, ts0)
    ms0 = _FakeMarketState(mid=1500.0, spread=0.2, market_timestamp=ts0)
    position, account = open_position(ms0, account, Side.LONG, sl_price=1495.0, tp_price=None, config=config)
    ts1 = datetime(2020, 1, 1, 0, 10, tzinfo=timezone.utc)
    ms1 = _FakeMarketState(mid=1510.0, spread=0.2, market_timestamp=ts1)
    exit_price = 1509.5
    net_pnl, cost_amount, cost_r, new_account = close_position(ms1, account, position, exit_price, config)
    assert new_account.open_position_id is None
    assert new_account.margin_used == 0.0
    assert new_account.balance == account.balance + net_pnl
    # Regression: realized PnL is fill-based (spread/slippage already embedded);
    # cost_amount REPORTS that embedded round-trip cost and is not deducted twice.
    assert net_pnl == (exit_price - position.entry_price) * position.size
    expected_round_trip = 2 * (0.2 / 2.0 + 0.2 * config.slippage_fraction_of_spread) * position.size
    assert abs(cost_amount - expected_round_trip) < 1e-9
    assert cost_amount > 0.0
    # Regression: round_trip_cost_r must actually produce a value in replay
    # (the live 5s staleness default previously made it None on every bar).
    assert cost_r is not None and cost_r > 0.0
    # Regression: sl_distance must be passed as an R-multiple, not a return
    # fraction. cost_R = spread*2 / (R * vol * mid) = spread*2 / (abs price dist).
    sl_return_fraction = abs(position.entry_price - position.sl_price) / position.entry_price
    expected_cost_r = (0.2 * 2) / (sl_return_fraction * ms1.mid)
    assert abs(cost_r - expected_cost_r) / expected_cost_r < 1e-9


def test_check_liquidation_true_when_equity_below_threshold_ratio():
    config = SimulatedExecutionConfig(liquidation_threshold=0.2)
    ts = datetime(2020, 1, 1, tzinfo=timezone.utc)
    account = AccountState(balance=100.0, equity=15.0, margin_used=100.0, margin_free=-85.0,
                            exposure=1000.0, open_position_id="p1", simulation_timestamp=ts)
    assert check_liquidation(account, config) is True


def test_check_liquidation_false_when_flat():
    config = SimulatedExecutionConfig(liquidation_threshold=0.2)
    ts = datetime(2020, 1, 1, tzinfo=timezone.utc)
    account = AccountState.initial(config, ts)
    assert check_liquidation(account, config) is False


def test_mark_to_market_moves_equity_and_enables_liquidation():
    """Regression: equity used to stay frozen at its open-time value while a
    position was open, so check_liquidation() could never fire."""
    config = SimulatedExecutionConfig(starting_balance=10000.0, leverage=100.0,
                                       risk_fraction_of_equity=0.01, liquidation_threshold=0.2)
    ts = datetime(2020, 1, 1, tzinfo=timezone.utc)
    account = AccountState.initial(config, ts)
    ms = _FakeMarketState(mid=1500.0, spread=0.2, market_timestamp=ts)
    position, account = open_position(ms, account, Side.LONG, sl_price=None, tp_price=None, config=config)
    assert check_liquidation(account, config) is False
    # A catastrophic adverse move must now be visible in equity and must trip
    # the liquidation safety net even with no SL set.
    crashed = mark_to_market(account, position, current_mid=1400.0)
    assert crashed.equity < account.equity
    assert crashed.equity == account.balance + position.unrealized_pnl(1400.0)
    wiped = mark_to_market(account, position, current_mid=1500.0 - 9990.0 / position.size)
    assert check_liquidation(wiped, config) is True


def test_realized_pnl_and_drawdown_track_across_sequence_of_trades():
    """Open/close a sequence of trades with known winning and losing PnL and
    assert realized_pnl_total accumulates, peak_equity only rises or holds,
    and drawdown reflects the actual max observed drop from the running peak."""
    config = SimulatedExecutionConfig(starting_balance=10000.0, leverage=100.0, risk_fraction_of_equity=0.01)
    ts = datetime(2020, 1, 1, tzinfo=timezone.utc)
    account = AccountState.initial(config, ts)
    assert account.currency == "USD"
    assert account.peak_equity == 10000.0
    assert account.drawdown == 0.0

    # Trade 1: winner.
    ms0 = _FakeMarketState(mid=1500.0, spread=0.2, market_timestamp=ts)
    position, account = open_position(ms0, account, Side.LONG, sl_price=1495.0, tp_price=None, config=config)
    ts1 = datetime(2020, 1, 1, 0, 10, tzinfo=timezone.utc)
    ms1 = _FakeMarketState(mid=1510.0, spread=0.2, market_timestamp=ts1)
    net_pnl_1, _, _, account = close_position(ms1, account, position, exit_price=1509.5, config=config)
    assert net_pnl_1 > 0.0
    assert account.realized_pnl_total == net_pnl_1
    assert account.peak_equity == account.equity  # new high
    assert account.drawdown == 0.0

    peak_after_trade_1 = account.peak_equity

    # Trade 2: loser, large enough to pull equity below the running peak.
    ms2 = _FakeMarketState(mid=1510.0, spread=0.2, market_timestamp=ts1)
    position, account = open_position(ms2, account, Side.LONG, sl_price=1495.0, tp_price=None, config=config)
    ts2 = datetime(2020, 1, 1, 0, 20, tzinfo=timezone.utc)
    ms3 = _FakeMarketState(mid=1490.0, spread=0.2, market_timestamp=ts2)
    net_pnl_2, _, _, account = close_position(ms3, account, position, exit_price=1490.5, config=config)
    assert net_pnl_2 < 0.0
    assert account.realized_pnl_total == net_pnl_1 + net_pnl_2
    # Peak must not have moved (this trade was a loss).
    assert account.peak_equity == peak_after_trade_1
    assert account.drawdown > 0.0
    expected_drawdown = (peak_after_trade_1 - account.equity) / peak_after_trade_1
    assert abs(account.drawdown - expected_drawdown) < 1e-9

    equity_after_trade_2 = account.equity
    drawdown_after_trade_2 = account.drawdown

    # Trade 3: winner, but not large enough to exceed the trade-1 peak yet.
    ms4 = _FakeMarketState(mid=1490.0, spread=0.2, market_timestamp=ts2)
    position, account = open_position(ms4, account, Side.LONG, sl_price=1485.0, tp_price=None, config=config)
    ts3 = datetime(2020, 1, 1, 0, 30, tzinfo=timezone.utc)
    ms5 = _FakeMarketState(mid=1492.0, spread=0.2, market_timestamp=ts3)
    net_pnl_3, _, _, account = close_position(ms5, account, position, exit_price=1491.5, config=config)
    assert account.realized_pnl_total == net_pnl_1 + net_pnl_2 + net_pnl_3
    assert account.equity > equity_after_trade_2
    # Peak still hasn't moved past trade-1's peak (equity remains below it).
    assert account.peak_equity == peak_after_trade_1
    assert account.drawdown < drawdown_after_trade_2  # drawdown shrank as equity recovered

    # Mark-to-market on an open position must also feed peak/drawdown tracking.
    ms6 = _FakeMarketState(mid=1492.0, spread=0.2, market_timestamp=ts3)
    position, account = open_position(ms6, account, Side.LONG, sl_price=None, tp_price=None, config=config)
    marked = mark_to_market(account, position, current_mid=1600.0)
    assert marked.equity > peak_after_trade_1
    assert marked.peak_equity == marked.equity  # new all-time high via mark-to-market alone
    assert marked.drawdown == 0.0
    assert marked.realized_pnl_total == account.realized_pnl_total  # unrealized move, not realized


def test_position_state_completeness_across_open_hold_close():
    """Task 6: current_price updates on every monitor step, execution_cost_total
    accumulates entry AND exit cost by close time, exit_reason is None while
    OPEN and set correctly (and only) at close."""
    config = SimulatedExecutionConfig(starting_balance=10000.0, leverage=100.0, risk_fraction_of_equity=0.01)
    ts0 = datetime(2020, 1, 1, tzinfo=timezone.utc)
    account = AccountState.initial(config, ts0)
    ms0 = _FakeMarketState(mid=1500.0, spread=0.2, market_timestamp=ts0)
    position, account = open_position(ms0, account, Side.LONG, sl_price=1495.0, tp_price=None, config=config)

    # OPEN: exit_reason must be absent (nothing has exited yet), current_price
    # is known from entry, execution_cost_total is entry cost only so far.
    assert position.exit_reason is None
    assert position.current_price == position.entry_price
    assert position.execution_cost_total == position.entry_cost_amount
    entry_cost = position.entry_cost_amount
    assert entry_cost > 0.0

    # Hold across several bars: current_price must track each monitor step's
    # mid, not stay frozen at entry.
    mids = [1502.0, 1505.0, 1503.0, 1508.0]
    for mid in mids:
        account = mark_to_market(account, position, current_mid=mid)
        view = to_position_view(position, current_mid=mid, bars_held=1)
        assert position.current_price == mid
        assert view.current_price == mid
        # Still just entry cost while open -- no exit fill has happened yet.
        assert position.execution_cost_total == entry_cost

    # Close: exit_reason set, execution_cost_total reflects BOTH entry and
    # exit cost with a concrete, checkable number.
    ts1 = datetime(2020, 1, 1, 0, 40, tzinfo=timezone.utc)
    ms1 = _FakeMarketState(mid=1509.5, spread=0.2, market_timestamp=ts1)
    exit_price = 1509.5
    net_pnl, cost_amount, cost_r, account = close_position(
        ms1, account, position, exit_price, config, exit_reason=PositionOutcome.POLICY_EXIT
    )

    expected_exit_cost = (ms1.spread / 2.0 + ms1.spread * config.slippage_fraction_of_spread) * position.size
    expected_total_cost = entry_cost + expected_exit_cost
    assert abs(cost_amount - expected_total_cost) < 1e-9
    assert position.execution_cost_total == cost_amount
    assert abs(position.execution_cost_total - expected_total_cost) < 1e-9
    # Genuinely BOTH components are present, not just entry or just exit alone.
    assert position.execution_cost_total > entry_cost
    assert position.execution_cost_total > expected_exit_cost

    assert position.exit_reason == PositionOutcome.POLICY_EXIT
    assert position.current_price == exit_price

    close_view = to_position_view(position, current_mid=exit_price, bars_held=4)
    # PositionView deliberately never carries exit_reason (leakage guard --
    # see test_leakage_extended.py) even after close; Position is the source
    # of truth for it.
    assert not hasattr(close_view, "exit_reason")
    assert close_view.execution_cost_total == position.execution_cost_total
    assert close_view.current_price == exit_price


def test_to_position_view_tracks_bars_held():
    ts = datetime(2020, 1, 1, tzinfo=timezone.utc)
    position = Position(position_id="p1", side=Side.LONG, entry_time=ts, entry_price=1500.0,
                         size=1.0, sl_price=1495.0, tp_price=None, margin_used=15.0)
    view = to_position_view(position, current_mid=1510.0, bars_held=7)
    assert view.bars_held == 7
    assert view.unrealized_pnl == 10.0


if __name__ == "__main__":
    test_open_position_reduces_margin_free()
    test_close_position_updates_balance_with_realized_pnl_minus_cost()
    test_check_liquidation_true_when_equity_below_threshold_ratio()
    test_check_liquidation_false_when_flat()
    test_mark_to_market_moves_equity_and_enables_liquidation()
    test_realized_pnl_and_drawdown_track_across_sequence_of_trades()
    test_position_state_completeness_across_open_hold_close()
    test_to_position_view_tracks_bars_held()
    print("tests/simulator/test_engine.py: OK")
