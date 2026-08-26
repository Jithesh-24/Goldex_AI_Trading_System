"""tests/research/test_phase2_experience_store.py"""
import os
import sys
import shutil
import tempfile
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simulator.contracts import EnvironmentTag
from simulator.experience import ExperienceRecord
from research.phase2_experience_store import ExperienceStore


def _make_record():
    return ExperienceRecord(
        environment_tag=EnvironmentTag.SIMULATED_TRAINING, timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
        event_type="DECIDE", market_state_snapshot={"mid": 1500.0}, position_view=None, action="NO_TRADE",
        account_state={"balance": 10000.0}, realized_pnl=None, cost_amount=None, outcome=None, gap_type="NORMAL",
    )


def test_write_and_read_round_trip():
    tmp_dir = tempfile.mkdtemp()
    try:
        store = ExperienceStore(base_dir=tmp_dir)
        records = [_make_record(), _make_record()]
        path = store.write_run("cand_a", "v1", "run_001", EnvironmentTag.SIMULATED_TRAINING, records)
        assert os.path.exists(path)
        read_back = store.read_run("cand_a", "v1", "run_001", EnvironmentTag.SIMULATED_TRAINING)
        assert len(read_back) == 2
        assert read_back[0]["action"] == "NO_TRADE"
        assert read_back[0]["environment_tag"] == "SIMULATED_TRAINING"
    finally:
        shutil.rmtree(tmp_dir)


def test_list_runs_filters_by_candidate_id():
    tmp_dir = tempfile.mkdtemp()
    try:
        store = ExperienceStore(base_dir=tmp_dir)
        store.write_run("cand_a", "v1", "run_001", EnvironmentTag.SIMULATED_TRAINING, [_make_record()])
        store.write_run("cand_b", "v1", "run_001", EnvironmentTag.SIMULATED_TRAINING, [_make_record()])
        all_runs = store.list_runs()
        assert len(all_runs) == 2
        only_a = store.list_runs(candidate_id="cand_a")
        assert len(only_a) == 1
        assert only_a[0]["candidate_id"] == "cand_a"
    finally:
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    test_write_and_read_round_trip()
    test_list_runs_filters_by_candidate_id()
    print("tests/research/test_phase2_experience_store.py: OK")
