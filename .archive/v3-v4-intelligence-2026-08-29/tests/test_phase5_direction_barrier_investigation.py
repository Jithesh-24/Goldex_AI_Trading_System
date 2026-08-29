"""tests/test_phase5_direction_barrier_investigation.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase5_direction_barrier_investigation import investigate_direction_barrier_relationship


def test_investigation_returns_decile_table():
    result = investigate_direction_barrier_relationship(max_holding=15, rows=100000)
    assert "decile_table" in result
    assert isinstance(result["correction_needed"], bool)
    assert isinstance(result["correction_note"], str) and len(result["correction_note"]) > 0


if __name__ == "__main__":
    test_investigation_returns_decile_table()
    print("tests/test_phase5_direction_barrier_investigation.py: OK")
