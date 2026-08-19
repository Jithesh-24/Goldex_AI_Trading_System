"""python3 tests/test_historical_coverage.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.historical_coverage import measure_coverage

def test_measure_coverage_real_data():
    result = measure_coverage("data/gold_seed_merged_full6yr.csv")
    assert 0.0 <= result["real_volume_nonzero_frac"] < 0.5, result["real_volume_nonzero_frac"]
    assert result["tick_volume_nonzero_frac"] < 1.0
    assert result["tick_volume_degrades_after"] is not None
    assert result["spread_constant_frac"] > 0.9
    assert len(result["spread_unique_values"]) > 0  # Spread has multiple unique values

if __name__ == "__main__":
    test_measure_coverage_real_data()
    print("tests/test_historical_coverage.py: OK")
