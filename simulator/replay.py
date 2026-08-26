"""simulator/replay.py
Chronological single-pass orchestrator. Calls decide_fn once per bar while
flat, manage_fn once per bar while a position is open, and enforces SL/TP/
liquidation as safety-net checks every bar regardless of what manage_fn
returns. No inter-trade cooldown: the bar immediately after a close is
eligible for a fresh decide_fn call. decide_fn/manage_fn are policy
callables -- this module contains no trading logic of its own."""
from typing import Callable, Optional

from simulator.contracts import AccountState, EnvironmentTag, PositionOutcome, Side, SimulatedExecutionConfig
from simulator.engine import (
    open_position, close_position, check_liquidation, to_position_view, mark_to_market,
)
from simulator.execution import exit_fill_price, resolve_same_bar_ambiguity
from simulator.experience import ExperienceRecord, ExperienceRecorder, write_tag_guard
from simulator.market_state_builder import build_snapshot

DecideFn = Callable[[object, AccountState], tuple]
ManageFn = Callable[[object, object, AccountState], str]


def _account_dict(account: AccountState) -> dict:
    return {
        "balance": account.balance, "equity": account.equity, "margin_used": account.margin_used,
        "margin_free": account.margin_free, "exposure": account.exposure,
        "open_position_id": account.open_position_id,
    }


def run_replay(df, decide_fn: DecideFn, manage_fn: ManageFn, config: SimulatedExecutionConfig,
               environment_tag: EnvironmentTag) -> ExperienceRecorder:
    recorder = ExperienceRecorder()
    n = len(df)
    if n == 0:
        return recorder
    first_ts = df.iloc[0]["time"].to_pydatetime()
    account = AccountState.initial(config, first_ts)
    position = None
    bars_held = 0

    for i in range(n):
        market_state = build_snapshot(df, i)
        row = df.iloc[i]

        if position is None:
            action, sl_price, tp_price = decide_fn(market_state, account)
            record = ExperienceRecord(
                environment_tag=environment_tag, timestamp=market_state.market_timestamp, event_type="DECIDE",
                market_state_snapshot={"mid": market_state.mid, "spread": market_state.spread},
                position_view=None, action=action, account_state=_account_dict(account),
                realized_pnl=None, cost_amount=None, outcome=None,
            )
            write_tag_guard(environment_tag, record)
            recorder.record(record)
            if action in ("LONG", "SHORT"):
                side = Side.LONG if action == "LONG" else Side.SHORT
                position, account = open_position(market_state, account, side, sl_price, tp_price, config)
                bars_held = 0
            continue

        bars_held += 1
        # BUGFIX (whole-branch review): revalue equity against the current mid
        # BEFORE the liquidation safety-net check, otherwise the check compares
        # a frozen open-time equity and can never trigger.
        account = mark_to_market(account, position, market_state.mid)
        position_view = to_position_view(position, market_state.mid, bars_held)
        outcome = None
        exit_price = None

        if i == n - 1:
            outcome = PositionOutcome.END_OF_REPLAY_FORCED_CLOSE
            exit_price = exit_fill_price(position.side, market_state.mid, market_state.spread, config)
        else:
            same_bar_result = resolve_same_bar_ambiguity(
                position.side, float(row["high"]), float(row["low"]), position.sl_price, position.tp_price
            )
            if same_bar_result == "SL_HIT":
                outcome = PositionOutcome.SL_HIT
                exit_price = position.sl_price
            elif same_bar_result == "TP_HIT":
                outcome = PositionOutcome.TP_HIT
                exit_price = position.tp_price
            elif check_liquidation(account, config):
                outcome = PositionOutcome.LIQUIDATION
                exit_price = exit_fill_price(position.side, market_state.mid, market_state.spread, config)
            else:
                manage_decision = manage_fn(market_state, position_view, account)
                record = ExperienceRecord(
                    environment_tag=environment_tag, timestamp=market_state.market_timestamp, event_type="MANAGE",
                    market_state_snapshot={"mid": market_state.mid, "spread": market_state.spread},
                    position_view=position_view.__dict__, action=manage_decision,
                    account_state=_account_dict(account), realized_pnl=None, cost_amount=None, outcome=None,
                )
                write_tag_guard(environment_tag, record)
                recorder.record(record)
                if manage_decision == "EXIT":
                    outcome = PositionOutcome.POLICY_EXIT
                    exit_price = exit_fill_price(position.side, market_state.mid, market_state.spread, config)

        if outcome is not None:
            net_pnl, cost_amount, cost_r, account = close_position(
                market_state, account, position, exit_price, config
            )
            close_record = ExperienceRecord(
                environment_tag=environment_tag, timestamp=market_state.market_timestamp, event_type="POSITION_CLOSED",
                market_state_snapshot={"mid": market_state.mid, "spread": market_state.spread},
                position_view=position_view.__dict__, action=None, account_state=_account_dict(account),
                realized_pnl=net_pnl, cost_amount=cost_amount, outcome=outcome, cost_r=cost_r,
            )
            write_tag_guard(environment_tag, close_record)
            recorder.record(close_record)
            position = None
            bars_held = 0

    # BUGFIX (whole-branch review): the in-loop END_OF_REPLAY_FORCED_CLOSE branch
    # only fires for a position that was ALREADY open when bar n-1 was reached.
    # A position opened by decide_fn ON bar n-1 hit `continue` and the loop then
    # ended, leaving it open forever -- no close record, and an account left with
    # a stale open_position_id/margin_used/exposure. Close it here so the
    # "flat at end of replay" invariant is structural, not data-dependent.
    if position is not None:
        market_state = build_snapshot(df, n - 1)
        account = mark_to_market(account, position, market_state.mid)
        position_view = to_position_view(position, market_state.mid, bars_held)
        exit_price = exit_fill_price(position.side, market_state.mid, market_state.spread, config)
        net_pnl, cost_amount, cost_r, account = close_position(
            market_state, account, position, exit_price, config
        )
        close_record = ExperienceRecord(
            environment_tag=environment_tag, timestamp=market_state.market_timestamp, event_type="POSITION_CLOSED",
            market_state_snapshot={"mid": market_state.mid, "spread": market_state.spread},
            position_view=position_view.__dict__, action=None, account_state=_account_dict(account),
            realized_pnl=net_pnl, cost_amount=cost_amount,
            outcome=PositionOutcome.END_OF_REPLAY_FORCED_CLOSE, cost_r=cost_r,
        )
        write_tag_guard(environment_tag, close_record)
        recorder.record(close_record)
        position = None
        bars_held = 0

    return recorder
