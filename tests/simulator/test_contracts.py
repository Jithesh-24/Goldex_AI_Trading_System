"""tests/simulator/test_contracts.py"""
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simulator.contracts import (
    EnvironmentTag, PositionOutcome, Side, SimulatedExecutionConfig, AccountState, Position, PositionView,
)


def test_environment_tag_values():
    assert set(EnvironmentTag) == {
        EnvironmentTag.SIMULATED_TRAINING, EnvironmentTag.SIMULATED_VALIDATION,
        EnvironmentTag.SIMULATED_OOS_TEST, EnvironmentTag.LIVE_DEMO, EnvironmentTag.LIVE_REAL,
    }


def test_position_outcome_values():
    assert set(PositionOutcome) == {
        PositionOutcome.POLICY_EXIT, PositionOutcome.SL_HIT, PositionOutcome.TP_HIT,
        PositionOutcome.LIQUIDATION, PositionOutcome.END_OF_REPLAY_FORCED_CLOSE,
    }


def test_account_state_initial():
    config = SimulatedExecutionConfig(starting_balance=5000.0)
    ts = datetime(2020, 1, 1, tzinfo=timezone.utc)
    acct = AccountState.initial(config, ts)
    assert acct.balance == 5000.0
    assert acct.equity == 5000.0
    assert acct.margin_used == 0.0
    assert acct.margin_free == 5000.0
    assert acct.open_position_id is None
    assert acct.simulation_timestamp == ts


def test_position_unrealized_pnl_long_and_short():
    ts = datetime(2020, 1, 1, tzinfo=timezone.utc)
    long_pos = Position(position_id="p1", side=Side.LONG, entry_time=ts, entry_price=100.0,
                         size=2.0, sl_price=None, tp_price=None, margin_used=10.0)
    assert long_pos.unrealized_pnl(110.0) == 20.0
    short_pos = Position(position_id="p2", side=Side.SHORT, entry_time=ts, entry_price=100.0,
                          size=2.0, sl_price=None, tp_price=None, margin_used=10.0)
    assert short_pos.unrealized_pnl(90.0) == 20.0


def test_position_view_construction():
    ts = datetime(2020, 1, 1, tzinfo=timezone.utc)
    view = PositionView(position_id="p1", side=Side.LONG, entry_time=ts, entry_price=100.0,
                         size=1.0, sl_price=None, tp_price=None, unrealized_pnl=5.0, bars_held=3)
    assert view.bars_held == 3
    assert view.sl_price is None and view.tp_price is None  # neither is ever mandatory


if __name__ == "__main__":
    test_environment_tag_values()
    test_position_outcome_values()
    test_account_state_initial()
    test_position_unrealized_pnl_long_and_short()
    test_position_view_construction()
    print("tests/simulator/test_contracts.py: OK")
