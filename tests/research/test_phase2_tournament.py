"""tests/research/test_phase2_tournament.py"""
import os
import sys
import shutil
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

from candidates.controls import RandomCandidate, NoTradeCandidate
from candidates.statistical_null import MomentumMeanReversionCandidate
from simulator.contracts import SimulatedExecutionConfig
from research.phase2_experience_store import ExperienceStore
from research.phase2_tournament import run_tournament


def _make_df(n=200):
    times = pd.date_range("2020-01-06 10:00:00", periods=n, freq="1min")
    prices = [1500.0 + (i % 20) * 0.05 for i in range(n)]
    return pd.DataFrame({
        "time": times, "open": prices,
        "high": [p + 0.3 for p in prices], "low": [p - 0.3 for p in prices],
        "close": [p + 0.05 for p in prices], "tick_volume": [10] * n, "spread": [20.0] * n,
    })


def test_tournament_runs_roster_and_produces_verdicts():
    tmp_dir = tempfile.mkdtemp()
    try:
        df_train = _make_df(200)
        df_val = _make_df(100)
        roster = [NoTradeCandidate(), MomentumMeanReversionCandidate(lookback_bars=10, z_threshold=1.0)]
        config = SimulatedExecutionConfig()
        store = ExperienceStore(base_dir=tmp_dir)
        result = run_tournament(df_train, df_val, roster, config, store, run_id="test_run_001")
        assert "control_gate" in result
        assert "candidates" in result
        assert "control_no_trade" in result["candidates"]
        assert "statistical_null_mean_reversion" in result["candidates"]
        assert result["candidates"]["control_no_trade"]["verdict"] == "CONTROL"
    finally:
        shutil.rmtree(tmp_dir)


def test_tournament_halts_if_random_control_is_persistently_profitable():
    tmp_dir = tempfile.mkdtemp()
    try:
        df_train = _make_df(50)
        df_val = _make_df(30)

        class _AlwaysWinningRandomStandIn:
            metadata = RandomCandidate(seed=0).metadata

            def decide(self, market_state, account):
                return ("LONG", None, None)

            def manage(self, market_state, position_view, account):
                return "EXIT"

        roster = [_AlwaysWinningRandomStandIn()]
        config = SimulatedExecutionConfig()
        store = ExperienceStore(base_dir=tmp_dir)
        result = run_tournament(df_train, df_val, roster, config, store, run_id="test_run_002")
        assert isinstance(result["control_gate"]["passed"], bool)
    finally:
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    test_tournament_runs_roster_and_produces_verdicts()
    test_tournament_halts_if_random_control_is_persistently_profitable()
    print("tests/research/test_phase2_tournament.py: OK")
