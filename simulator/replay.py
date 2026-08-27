"""simulator/replay.py
Chronological single-pass orchestrator. Calls decide_fn once per bar while
flat, manage_fn once per bar while a position is open, and enforces SL/TP/
liquidation as safety-net checks every bar regardless of what manage_fn
returns. No inter-trade cooldown: the bar immediately after a close is
eligible for a fresh decide_fn call. decide_fn/manage_fn are policy
callables -- this module contains no trading logic of its own."""
from typing import Callable, Optional

from simulator.closure import classify_gap
from simulator.contracts import AccountState, EnvironmentTag, PositionOutcome, Side, SimulatedExecutionConfig
from simulator.engine import (
    open_position, close_position, check_liquidation, to_position_view, mark_to_market,
)
from simulator.execution import exit_fill_price, resolve_same_bar_ambiguity
from simulator.experience import ExperienceRecord, ExperienceRecorder, write_tag_guard
from simulator.market_state_builder import build_snapshot

DecideFn = Callable[[object, AccountState], tuple]
ManageFn = Callable[[object, object, AccountState], str]


def _extract_observation_features(fn: Callable) -> Optional[dict]:
    """Phase 3A addition: optional, opt-in way for a candidate to expose the
    feature/observation dict it used for its most recent decide()/manage()
    call, without changing the decide_fn/manage_fn contract that
    phase2_tournament.py and every existing candidate already rely on.

    Convention: if `fn` is a bound method, and its instance (`fn.__self__`)
    has an attribute `last_decision_features` set to a dict, that dict is
    recorded and then cleared so a stale value from a previous bar can never
    leak forward into a bar where the candidate didn't set anything. Nothing
    here inspects or requires specific feature names -- purely a generic
    passthrough. Candidates that don't set this attribute are unaffected;
    this always returns None for them."""
    instance = getattr(fn, "__self__", None)
    if instance is None:
        return None
    features = getattr(instance, "last_decision_features", None)
    if features is None:
        return None
    if not isinstance(features, dict):
        return None
    # Clear immediately so it can't be silently reused on the next bar if a
    # candidate forgets to set it every call.
    instance.last_decision_features = None
    return features


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
    prev_timestamp = None

    for i in range(n):
        market_state = build_snapshot(df, i)
        row = df.iloc[i]
        current_timestamp = market_state.market_timestamp

        # Classify the gap between the previous bar and this bar
        gap_type = "NORMAL"
        if i > 0:
            gap_type = classify_gap(prev_timestamp, current_timestamp)

        if position is None:
            action, sl_price, tp_price = decide_fn(market_state, account)
            observation_features = _extract_observation_features(decide_fn)
            # Refuse to open new positions if there's a DATA_GAP
            if action in ("LONG", "SHORT") and gap_type == "DATA_GAP":
                action = "NO_TRADE"
            record = ExperienceRecord(
                environment_tag=environment_tag, timestamp=market_state.market_timestamp, event_type="DECIDE",
                market_state_snapshot={"mid": market_state.mid, "spread": market_state.spread},
                position_view=None, action=action, account_state=_account_dict(account),
                realized_pnl=None, cost_amount=None, outcome=None, gap_type=gap_type,
                observation_features=observation_features,
            )
            write_tag_guard(environment_tag, record)
            recorder.record(record)
            if action in ("LONG", "SHORT"):
                side = Side.LONG if action == "LONG" else Side.SHORT
                position, account = open_position(market_state, account, side, sl_price, tp_price, config)
                bars_held = 0
            prev_timestamp = current_timestamp
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
                manage_observation_features = _extract_observation_features(manage_fn)
                record = ExperienceRecord(
                    environment_tag=environment_tag, timestamp=market_state.market_timestamp, event_type="MANAGE",
                    market_state_snapshot={"mid": market_state.mid, "spread": market_state.spread},
                    position_view=position_view.__dict__, action=manage_decision,
                    account_state=_account_dict(account), realized_pnl=None, cost_amount=None, outcome=None,
                    gap_type=gap_type, observation_features=manage_observation_features,
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
                gap_type=gap_type,
            )
            write_tag_guard(environment_tag, close_record)
            recorder.record(close_record)
            position = None
            bars_held = 0

        prev_timestamp = current_timestamp

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
            # This record is for bar n-1, the same bar the loop just classified.
            # Reuse that classification rather than hardcoding "NORMAL", which
            # would mislabel a position opened on a WEEKEND_CLOSURE bar and make
            # the DECIDE and POSITION_CLOSED records for the same bar disagree.
            gap_type=gap_type,
        )
        write_tag_guard(environment_tag, close_record)
        recorder.record(close_record)
        position = None
        bars_held = 0

    return recorder
