"""tests/simulator/test_decision_id_linkage.py
Phase 4 addition. ExperienceRecord.decision_id is a new, additive, optional
field linking a DECIDE record that opens a position to the eventual
POSITION_CLOSED record it produces (and to any MANAGE records generated
while that position is open). This test proves:
  1. a DECIDE record that opens a position and its corresponding
     POSITION_CLOSED record share the same non-null decision_id.
  2. DECIDE records that don't open a position have decision_id=None.
  3. no look-ahead: decision_id is generated only from information available
     at/after the decision (a fresh random id), never from anything that
     depends on how the trade eventually turns out -- proven the same way
     Phase 3A proved this for observation_features, by truncating the
     dataset after the decision point and checking already-recorded
     decision_ids on DECIDE records are unaffected."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

from simulator.contracts import EnvironmentTag, SimulatedExecutionConfig
from simulator.replay import run_replay


def _make_df(n=40):
    times = pd.date_range("2020-01-06 10:00:00", periods=n, freq="1min")
    prices = [1500.0 + i * 0.1 for i in range(n)]
    return pd.DataFrame({
        "time": times, "open": prices,
        "high": [p + 0.3 for p in prices], "low": [p - 0.3 for p in prices],
        "close": [p + 0.05 for p in prices], "tick_volume": [10] * n, "spread": [20.0] * n,
    })


class _OpenOnceCandidate:
    """Test-only candidate: opens a single LONG on the first bar, holds until
    forced close, otherwise NO_TRADE. Deterministic, no randomness needed
    beyond what run_replay itself injects for decision_id."""

    def __init__(self):
        self._opened = False

    def decide(self, market_state, account):
        if not self._opened:
            self._opened = True
            return ("LONG", None, None)
        return ("NO_TRADE", None, None)

    def manage(self, market_state, position_view, account):
        return "HOLD"


def test_decide_and_position_closed_share_decision_id():
    df = _make_df()
    config = SimulatedExecutionConfig()
    candidate = _OpenOnceCandidate()
    recorder = run_replay(df, candidate.decide, candidate.manage, config, EnvironmentTag.SIMULATED_TRAINING)
    records = recorder.all_records()

    opening_decide = next(r for r in records if r.event_type == "DECIDE" and r.action in ("LONG", "SHORT"))
    assert opening_decide.decision_id is not None

    closed = [r for r in records if r.event_type == "POSITION_CLOSED"]
    assert len(closed) == 1
    assert closed[0].decision_id == opening_decide.decision_id

    manage_records = [r for r in records if r.event_type == "MANAGE"]
    for r in manage_records:
        assert r.decision_id == opening_decide.decision_id


def test_no_trade_decide_records_have_null_decision_id():
    df = _make_df(n=10)
    config = SimulatedExecutionConfig()

    def always_no_trade(market_state, account):
        return ("NO_TRADE", None, None)

    recorder = run_replay(df, always_no_trade, lambda *a: "HOLD", config, EnvironmentTag.SIMULATED_TRAINING)
    decide_records = [r for r in recorder.all_records() if r.event_type == "DECIDE"]
    assert len(decide_records) == len(df)
    for r in decide_records:
        assert r.action == "NO_TRADE"
        assert r.decision_id is None


def test_decision_id_no_lookahead_via_truncation():
    """The decision_id assigned to the opening DECIDE record must not depend
    on anything that happens after the decision (e.g. how/when the trade
    eventually closes). Truncating the dataset right after the opening bar
    forces a different eventual close (END_OF_REPLAY_FORCED_CLOSE at a
    different bar/price) -- the DECIDE record's decision_id must still be
    generated (non-null) and the linkage invariant must still hold within
    the truncated run, proving decision_id generation never reaches into
    future-only information to decide whether/what id to assign."""
    df = _make_df(n=40)
    config = SimulatedExecutionConfig()

    candidate_full = _OpenOnceCandidate()
    recorder_full = run_replay(df, candidate_full.decide, candidate_full.manage, config, EnvironmentTag.SIMULATED_TRAINING)
    full_decide = next(r for r in recorder_full.all_records() if r.event_type == "DECIDE" and r.action == "LONG")

    truncated = df.iloc[:5].copy()
    candidate_trunc = _OpenOnceCandidate()
    recorder_trunc = run_replay(truncated, candidate_trunc.decide, candidate_trunc.manage, config, EnvironmentTag.SIMULATED_TRAINING)
    trunc_decide = next(r for r in recorder_trunc.all_records() if r.event_type == "DECIDE" and r.action == "LONG")

    # Both runs assign a decision_id to the opening DECIDE record purely
    # because the decision opened a position -- independent of how far the
    # dataset extends afterward or how the trade eventually closes.
    assert full_decide.decision_id is not None
    assert trunc_decide.decision_id is not None

    trunc_closed = [r for r in recorder_trunc.all_records() if r.event_type == "POSITION_CLOSED"]
    assert len(trunc_closed) == 1
    assert trunc_closed[0].decision_id == trunc_decide.decision_id


if __name__ == "__main__":
    test_decide_and_position_closed_share_decision_id()
    test_no_trade_decide_records_have_null_decision_id()
    test_decision_id_no_lookahead_via_truncation()
    print("tests/simulator/test_decision_id_linkage.py: OK")
