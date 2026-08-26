"""tests/simulator/test_engine.py"""
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simulator.contracts import Side, AccountState, SimulatedExecutionConfig, Position
from simulator.engine import open_position, close_position, check_liquidation, to_position_view


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
    net_pnl, cost_amount, new_account = close_position(ms1, account, position, exit_price, config)
    assert new_account.open_position_id is None
    assert new_account.margin_used == 0.0
    assert new_account.balance == account.balance + net_pnl
    assert cost_amount >= 0.0


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
    test_to_position_view_tracks_bars_held()
    print("tests/simulator/test_engine.py: OK")
