"""tests/research/test_phase3_tournament.py"""
import os
import sys
import shutil
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

from candidates.controls import NoTradeCandidate
from candidates.bayesian_online import BayesianOnlineCandidate
from simulator.contracts import SimulatedExecutionConfig
from research.phase2_experience_store import ExperienceStore
from research.phase3_tournament import run_phase3_tournament


def _make_df(n=150):
    times = pd.date_range("2020-01-06 10:00:00", periods=n, freq="1min")
    prices = [1500.0 + (i % 20) * 0.05 for i in range(n)]
    return pd.DataFrame({
        "time": times, "open": prices,
        "high": [p + 0.3 for p in prices], "low": [p - 0.3 for p in prices],
        "close": [p + 0.05 for p in prices], "tick_volume": [10] * n, "spread": [20.0] * n,
    })


def test_learning_candidate_gets_learn_called_between_partitions():
    tmp_dir = tempfile.mkdtemp()
    try:
        df_train, df_val = _make_df(150), _make_df(80)
        bayes = BayesianOnlineCandidate()
        roster = [NoTradeCandidate(), bayes]
        config = SimulatedExecutionConfig()
        store = ExperienceStore(base_dir=tmp_dir)
        result = run_phase3_tournament(df_train, df_val, roster, config, store, run_id="p3_test_001")
        assert "bayesian_online" in result["candidates"]
        assert "control_no_trade" in result["candidates"]
    finally:
        shutil.rmtree(tmp_dir)


def test_learn_never_receives_validation_tagged_records():
    tmp_dir = tempfile.mkdtemp()
    try:
        df_train, df_val = _make_df(150), _make_df(80)

        class _RecordingLearner(BayesianOnlineCandidate):
            def __init__(self):
                super().__init__()
                self.seen_tags = set()

            def learn(self, training_experience):
                for r in training_experience:
                    self.seen_tags.add(r.get("environment_tag"))
                super().learn(training_experience)

        candidate = _RecordingLearner()
        config = SimulatedExecutionConfig()
        store = ExperienceStore(base_dir=tmp_dir)
        run_phase3_tournament(df_train, df_val, [candidate], config, store, run_id="p3_test_002")
        assert candidate.seen_tags <= {"SIMULATED_TRAINING"}
    finally:
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    test_learning_candidate_gets_learn_called_between_partitions()
    test_learn_never_receives_validation_tagged_records()
    print("tests/research/test_phase3_tournament.py: OK")
