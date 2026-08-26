"""simulator/execution.py
Spread-crossing entry/exit fills plus a configurable slippage fraction of
spread (slippage is unmodeled anywhere else in this codebase -- see
decision/ev_cost.py, which handles spread cost only, not slippage). Cost
still goes through decision.ev_cost.round_trip_cost_r unmodified.
resolve_same_bar_ambiguity reuses features/labeling.py's existing
_triple_barrier_core convention: if both SL and TP are touched within one
bar, the adverse side wins."""
from typing import Optional

from decision.ev_cost import round_trip_cost_r
from simulator.contracts import Side, SimulatedExecutionConfig


def entry_fill_price(side: Side, mid: float, spread: float, config: SimulatedExecutionConfig) -> float:
    half_spread = spread / 2.0
    slippage = spread * config.slippage_fraction_of_spread
    if side == Side.LONG:
        return mid + half_spread + slippage
    return mid - half_spread - slippage


def exit_fill_price(side: Side, mid: float, spread: float, config: SimulatedExecutionConfig) -> float:
    half_spread = spread / 2.0
    slippage = spread * config.slippage_fraction_of_spread
    if side == Side.LONG:
        return mid - half_spread - slippage
    return mid + half_spread + slippage


def compute_cost_r(market_state, sl_distance_r: Optional[float], config: SimulatedExecutionConfig) -> Optional[float]:
    if sl_distance_r is None or sl_distance_r <= 0:
        return None
    return round_trip_cost_r(market_state, sl_distance_r, config.max_staleness_seconds)


def resolve_same_bar_ambiguity(side: Side, bar_high: float, bar_low: float,
                                sl_price: Optional[float], tp_price: Optional[float]) -> Optional[str]:
    sl_touched = sl_price is not None and (
        (side == Side.LONG and bar_low <= sl_price) or (side == Side.SHORT and bar_high >= sl_price)
    )
    tp_touched = tp_price is not None and (
        (side == Side.LONG and bar_high >= tp_price) or (side == Side.SHORT and bar_low <= tp_price)
    )
    if sl_touched:
        return "SL_HIT"
    if tp_touched:
        return "TP_HIT"
    return None
