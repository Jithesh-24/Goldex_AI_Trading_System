"""simulator/engine.py
Position open/close mechanics and account-state bookkeeping. No decision
logic lives here -- side/sl_price/tp_price/exit timing are all supplied by
the caller (a policy in Phase 2+, or a test stub here in Phase 1)."""
import uuid
from typing import Optional

from simulator.contracts import (
    AccountState, Position, PositionOutcome, PositionView, Side, SimulatedExecutionConfig,
)
from simulator.execution import entry_fill_price, compute_cost_r


def to_position_view(position: Position, current_mid: float, bars_held: int) -> PositionView:
    return PositionView(
        position_id=position.position_id, side=position.side, entry_time=position.entry_time,
        entry_price=position.entry_price, size=position.size, sl_price=position.sl_price,
        tp_price=position.tp_price, unrealized_pnl=position.unrealized_pnl(current_mid), bars_held=bars_held,
        current_price=position.current_price, execution_cost_total=position.execution_cost_total,
    )


def open_position(market_state, account: AccountState, side: Side, sl_price: Optional[float],
                   tp_price: Optional[float], config: SimulatedExecutionConfig,
                   size: Optional[float] = None):
    entry_price = entry_fill_price(side, market_state.mid, market_state.spread, config)
    if size is None:
        size = (account.equity * config.risk_fraction_of_equity)
    margin_used = (size * entry_price) / config.leverage
    entry_cost_amount = (market_state.spread / 2.0
                         + market_state.spread * config.slippage_fraction_of_spread) * size
    position = Position(position_id=str(uuid.uuid4()), side=side, entry_time=market_state.market_timestamp,
                         entry_price=entry_price, size=size, sl_price=sl_price, tp_price=tp_price,
                         margin_used=margin_used, entry_cost_amount=entry_cost_amount,
                         current_price=entry_price, execution_cost_total=entry_cost_amount)
    new_account = AccountState(
        balance=account.balance, equity=account.equity,
        margin_used=account.margin_used + margin_used,
        margin_free=account.equity - (account.margin_used + margin_used),
        exposure=account.exposure + size * entry_price,
        open_position_id=position.position_id, simulation_timestamp=market_state.market_timestamp,
        realized_pnl_total=account.realized_pnl_total,
        peak_equity=account.peak_equity, drawdown=account.drawdown, currency=account.currency,
    )
    return position, new_account


def close_position(market_state, account: AccountState, position: Position, exit_price: float,
                    config: SimulatedExecutionConfig, exit_reason: Optional[PositionOutcome] = None):
    direction = 1.0 if position.side == Side.LONG else -1.0
    realized_pnl = direction * (exit_price - position.entry_price) * position.size

    # BUGFIX (whole-branch review): spread + slippage are ALREADY embedded in
    # entry_fill_price()/exit_fill_price(), so realized_pnl is already net of
    # round-trip execution cost. The previous code subtracted a second,
    # separately-computed round-trip spread cost on top of that -- double
    # charging, and asymmetrically so (only when an SL happened to be set).
    # cost_amount now REPORTS the cost actually embedded in the fills.
    exit_cost_amount = (market_state.spread / 2.0
                        + market_state.spread * config.slippage_fraction_of_spread) * position.size
    cost_amount = position.entry_cost_amount + exit_cost_amount

    # Position-side bookkeeping: mutate the closing position in place so the
    # object reflects its final state (round-trip cost, last known price, and
    # why it closed) at the moment close_position returns.
    position.current_price = exit_price
    position.execution_cost_total = cost_amount
    position.exit_reason = exit_reason

    # BUGFIX (whole-branch review): decision.ev_cost.round_trip_cost_r requires
    # candidate_sl_distance in R-MULTIPLES (already volatility-normalized), not
    # a raw price-return fraction. The previous code passed |entry-sl|/entry (a
    # return fraction), causing a second division by vol inside ev_cost and a
    # cost_R inflated by ~1/vol. Convert to R here: R = return_fraction / vol.
    # cost_r is a recorded raw ingredient (R-space diagnostic), never a second
    # deduction from PnL -- no reward formula lives in the simulator.
    cost_r = None
    if position.sl_price is not None and getattr(market_state, "realized_vol_60s", None):
        sl_return_fraction = abs(position.entry_price - position.sl_price) / position.entry_price
        cost_r = compute_cost_r(market_state, sl_return_fraction / market_state.realized_vol_60s, config)

    net_pnl = realized_pnl
    new_balance = account.balance + net_pnl
    peak_equity, drawdown = AccountState.compute_drawdown(account.peak_equity, new_balance)
    new_account = AccountState(
        balance=new_balance, equity=new_balance, margin_used=0.0, margin_free=new_balance,
        exposure=0.0, open_position_id=None, simulation_timestamp=market_state.market_timestamp,
        realized_pnl_total=account.realized_pnl_total + net_pnl,
        peak_equity=peak_equity, drawdown=drawdown, currency=account.currency,
    )
    return net_pnl, cost_amount, cost_r, new_account


def mark_to_market(account: AccountState, position: Position, current_mid: float) -> AccountState:
    """BUGFIX (whole-branch review): equity was never revalued while a position
    was open, so check_liquidation()'s equity/margin_used ratio was frozen at
    its open-time value and LIQUIDATION could never fire -- the safety net the
    design relies on for policies that run with no SL at all was dead code."""
    equity = account.balance + position.unrealized_pnl(current_mid)
    position.current_price = current_mid
    peak_equity, drawdown = AccountState.compute_drawdown(account.peak_equity, equity)
    return AccountState(
        balance=account.balance, equity=equity, margin_used=account.margin_used,
        margin_free=equity - account.margin_used, exposure=account.exposure,
        open_position_id=account.open_position_id, simulation_timestamp=account.simulation_timestamp,
        realized_pnl_total=account.realized_pnl_total,
        peak_equity=peak_equity, drawdown=drawdown, currency=account.currency,
    )


def check_liquidation(account: AccountState, config: SimulatedExecutionConfig) -> bool:
    if account.margin_used <= 0:
        return False
    return (account.equity / account.margin_used) < config.liquidation_threshold
