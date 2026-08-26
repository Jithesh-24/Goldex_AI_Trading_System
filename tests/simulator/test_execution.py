"""tests/simulator/test_execution.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simulator.contracts import Side, SimulatedExecutionConfig
from simulator.execution import entry_fill_price, exit_fill_price, resolve_same_bar_ambiguity


def test_entry_fill_long_pays_ask_plus_slippage():
    config = SimulatedExecutionConfig(slippage_fraction_of_spread=0.5)
    price = entry_fill_price(Side.LONG, mid=100.0, spread=1.0, config=config)
    assert price == 100.0 + 0.5 + 0.5  # half-spread + slippage(0.5*spread)


def test_entry_fill_short_receives_bid_minus_slippage():
    config = SimulatedExecutionConfig(slippage_fraction_of_spread=0.5)
    price = entry_fill_price(Side.SHORT, mid=100.0, spread=1.0, config=config)
    assert price == 100.0 - 0.5 - 0.5


def test_exit_fill_long_receives_bid_minus_slippage():
    config = SimulatedExecutionConfig(slippage_fraction_of_spread=0.5)
    price = exit_fill_price(Side.LONG, mid=100.0, spread=1.0, config=config)
    assert price == 100.0 - 0.5 - 0.5


def test_same_bar_ambiguity_both_touched_charges_adverse_side():
    result = resolve_same_bar_ambiguity(Side.LONG, bar_high=110.0, bar_low=90.0, sl_price=95.0, tp_price=105.0)
    assert result == "SL_HIT"


def test_same_bar_ambiguity_only_tp_touched():
    result = resolve_same_bar_ambiguity(Side.LONG, bar_high=110.0, bar_low=99.0, sl_price=95.0, tp_price=105.0)
    assert result == "TP_HIT"


def test_same_bar_ambiguity_neither_touched():
    result = resolve_same_bar_ambiguity(Side.LONG, bar_high=101.0, bar_low=99.0, sl_price=95.0, tp_price=105.0)
    assert result is None


def test_same_bar_ambiguity_no_sl_or_tp_set():
    result = resolve_same_bar_ambiguity(Side.LONG, bar_high=110.0, bar_low=90.0, sl_price=None, tp_price=None)
    assert result is None


if __name__ == "__main__":
    test_entry_fill_long_pays_ask_plus_slippage()
    test_entry_fill_short_receives_bid_minus_slippage()
    test_exit_fill_long_receives_bid_minus_slippage()
    test_same_bar_ambiguity_both_touched_charges_adverse_side()
    test_same_bar_ambiguity_only_tp_touched()
    test_same_bar_ambiguity_neither_touched()
    test_same_bar_ambiguity_no_sl_or_tp_set()
    print("tests/simulator/test_execution.py: OK")
