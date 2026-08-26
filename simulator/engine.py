"""simulator/engine.py
Position open/close mechanics and account-state bookkeeping. No decision
logic lives here -- side/sl_price/tp_price/exit timing are all supplied by
the caller (a policy in Phase 2+, or a test stub here in Phase 1)."""
import uuid
from typing import Optional

from simulator.contracts import AccountState, Position, PositionView, Side, SimulatedExecutionConfig
from simulator.execution import entry_fill_price, compute_cost_r


def to_position_view(position: Position, current_mid: float, bars_held: int) -> PositionView:
    return PositionView(
        position_id=position.position_id, side=position.side, entry_time=position.entry_time,
        entry_price=position.entry_price, size=position.size, sl_price=position.sl_price,
        tp_price=position.tp_price, unrealized_pnl=position.unrealized_pnl(current_mid), bars_held=bars_held,
    )


def open_position(market_state, account: AccountState, side: Side, sl_price: Optional[float],
                   tp_price: Optional[float], config: SimulatedExecutionConfig):
    entry_price = entry_fill_price(side, market_state.mid, market_state.spread, config)
    size = (account.equity * config.risk_fraction_of_equity)
    margin_used = (size * entry_price) / config.leverage
    position = Position(position_id=str(uuid.uuid4()), side=side, entry_time=market_state.market_timestamp,
                         entry_price=entry_price, size=size, sl_price=sl_price, tp_price=tp_price,
                         margin_used=margin_used)
    new_account = AccountState(
        balance=account.balance, equity=account.equity,
        margin_used=account.margin_used + margin_used,
        margin_free=account.equity - (account.margin_used + margin_used),
        exposure=account.exposure + size * entry_price,
        open_position_id=position.position_id, simulation_timestamp=market_state.market_timestamp,
    )
    return position, new_account


def close_position(market_state, account: AccountState, position: Position, exit_price: float,
                    config: SimulatedExecutionConfig):
    direction = 1.0 if position.side == Side.LONG else -1.0
    realized_pnl = direction * (exit_price - position.entry_price) * position.size
    sl_distance_r = (abs(position.entry_price - position.sl_price) / position.entry_price
                      if position.sl_price is not None else None)
    cost_r = compute_cost_r(market_state, sl_distance_r, config) if sl_distance_r is not None else None
    cost_amount = (cost_r * sl_distance_r * position.entry_price * position.size) if cost_r is not None else 0.0
    net_pnl = realized_pnl - cost_amount
    new_balance = account.balance + net_pnl
    new_account = AccountState(
        balance=new_balance, equity=new_balance, margin_used=0.0, margin_free=new_balance,
        exposure=0.0, open_position_id=None, simulation_timestamp=market_state.market_timestamp,
    )
    return net_pnl, cost_amount, new_account


def check_liquidation(account: AccountState, config: SimulatedExecutionConfig) -> bool:
    if account.margin_used <= 0:
        return False
    return (account.equity / account.margin_used) < config.liquidation_threshold
