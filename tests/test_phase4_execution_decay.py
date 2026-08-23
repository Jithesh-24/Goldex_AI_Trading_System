"""python3 tests/test_phase4_execution_decay.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase4_execution_decay import run_execution_decay_proxy


def test_run_execution_decay_proxy_reports_data_limited():
    result = run_execution_decay_proxy(rows=20000)
    assert result["data_limited"] is True
    assert result["n_events"] > 0
    for delay in ("30s", "60s", "120s"):
        assert delay in result["drift_by_delay"]


if __name__ == "__main__":
    test_run_execution_decay_proxy_reports_data_limited()
    print("tests/test_phase4_execution_decay.py: OK")
