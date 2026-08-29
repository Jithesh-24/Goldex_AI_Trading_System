"""python3 tests/test_feature_diagnostics.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from features.registry.diagnostics import correlation_redundancy, distribution_stability


def test_correlation_redundancy_detects_duplicate_column():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 500)
    df = pd.DataFrame({"a": a, "b": a * 2 + 0.0001, "c": rng.normal(0, 1, 500)})
    pairs = correlation_redundancy(df, threshold=0.95)
    names = {(p[0], p[1]) for p in pairs}
    assert ("a", "b") in names or ("b", "a") in names


def test_distribution_stability_flags_shift():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0, 1, 500))
    b = pd.Series(rng.normal(5, 1, 500))  # large mean shift
    result = distribution_stability(a, b)
    assert result["mean_shift"] > 4.0


if __name__ == "__main__":
    test_correlation_redundancy_detects_duplicate_column()
    test_distribution_stability_flags_shift()
    print("tests/test_feature_diagnostics.py: OK")
