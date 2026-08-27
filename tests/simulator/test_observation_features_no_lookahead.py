"""tests/simulator/test_observation_features_no_lookahead.py
Phase 3A addition. ExperienceRecord.observation_features is a new, additive,
opt-in field carrying the full feature vector a candidate used at decision
time. This test proves the same invariant Phase 1 proved for
market_state_snapshot in test_no_leakage.py: truncating the dataset after
the current decision point must not change any already-recorded
observation_features, because a real feature vector may only ever be
derived from bars [0..i-1] relative to the decision bar."""
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


class _CausalFeatureCandidate:
    """Test-only candidate that exposes an observation_features dict derived
    strictly from bars seen strictly before the current decision bar (the
    candidate's own running history), via the opt-in
    `last_decision_features` convention read by simulator.replay."""

    def __init__(self):
        self.last_decision_features = None
        self._seen_mids = []  # only ever appended to AFTER using it for features

    def decide(self, market_state, account):
        # Feature derived only from mids observed on strictly earlier bars.
        prior_mids = list(self._seen_mids)
        mean_prior_mid = sum(prior_mids) / len(prior_mids) if prior_mids else None
        self.last_decision_features = {
            "n_prior_bars_seen": len(prior_mids),
            "mean_prior_mid": mean_prior_mid,
        }
        self._seen_mids.append(market_state.mid)
        return ("NO_TRADE", None, None)

    def manage(self, market_state, position_view, account):
        return "HOLD"


def test_observation_features_identical_regardless_of_unreached_future():
    df = _make_df()
    config = SimulatedExecutionConfig()

    candidate_clean = _CausalFeatureCandidate()
    recorder_clean = run_replay(
        df, candidate_clean.decide, candidate_clean.manage, config, EnvironmentTag.SIMULATED_TRAINING
    )

    truncated = df.iloc[: len(df) // 2].copy()
    candidate_trunc = _CausalFeatureCandidate()
    recorder_trunc = run_replay(
        truncated, candidate_trunc.decide, candidate_trunc.manage, config, EnvironmentTag.SIMULATED_TRAINING
    )

    n_common = len(recorder_trunc.all_records())
    clean_records = recorder_clean.all_records()[:n_common]
    trunc_records = recorder_trunc.all_records()
    assert n_common > 0
    for a, b in zip(clean_records, trunc_records):
        assert a.observation_features == b.observation_features, (
            "leakage: truncating the dataset after the current decision point changed an "
            "earlier record's observation_features"
        )


def test_observation_features_populated_and_causal():
    df = _make_df()
    config = SimulatedExecutionConfig()
    candidate = _CausalFeatureCandidate()
    recorder = run_replay(df, candidate.decide, candidate.manage, config, EnvironmentTag.SIMULATED_TRAINING)

    decide_records = [r for r in recorder.all_records() if r.event_type == "DECIDE"]
    assert len(decide_records) == len(df)
    for i, record in enumerate(decide_records):
        assert record.observation_features is not None
        assert record.observation_features["n_prior_bars_seen"] == i


def test_observation_features_defaults_to_none_for_candidates_that_dont_opt_in():
    df = _make_df(n=5)
    config = SimulatedExecutionConfig()

    def always_no_trade(market_state, account):
        return ("NO_TRADE", None, None)

    recorder = run_replay(df, always_no_trade, lambda *a: "HOLD", config, EnvironmentTag.SIMULATED_TRAINING)
    for record in recorder.all_records():
        assert record.observation_features is None


if __name__ == "__main__":
    test_observation_features_identical_regardless_of_unreached_future()
    test_observation_features_populated_and_causal()
    test_observation_features_defaults_to_none_for_candidates_that_dont_opt_in()
    print("tests/simulator/test_observation_features_no_lookahead.py: OK")
