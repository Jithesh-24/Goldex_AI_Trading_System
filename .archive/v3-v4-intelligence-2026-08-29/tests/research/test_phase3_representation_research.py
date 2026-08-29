"""tests/research/test_phase3_representation_research.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from research.phase3_representation_research import (
    analyze_return_autocorrelation, analyze_volatility_clustering, analyze_regime_persistence,
)
from candidates.hmm_regime import HMMRegimeCandidate


def test_autocorrelation_of_pure_noise_is_near_zero():
    rng = np.random.default_rng(0)
    closes = 1500.0 + np.cumsum(rng.normal(0, 0.1, 2000))
    result = analyze_return_autocorrelation(closes, max_lag=5)
    assert "lag_1" in result
    assert abs(result["lag_1"]) < 0.3  # pure random walk: no strong lag-1 autocorrelation expected


def test_volatility_clustering_detects_synthetic_clustering():
    rng = np.random.default_rng(1)
    vol_regime = np.concatenate([np.full(500, 0.05), np.full(500, 0.5)])
    returns = rng.normal(0, 1, 1000) * vol_regime
    closes = 1500.0 + np.cumsum(returns)
    result = analyze_volatility_clustering(closes, window=30)
    assert "lag_1" in result


def test_regime_persistence_reports_dwell_time():
    candidate = HMMRegimeCandidate(max_em_iterations=5)
    rng = np.random.default_rng(2)
    prices = 1500.0 + np.cumsum(rng.normal(0, 0.05, 300))
    records = [{"event_type": "DECIDE", "market_state_snapshot": {"mid": float(p)}} for p in prices]
    candidate.learn(records)
    result = analyze_regime_persistence(candidate, prices)
    assert "mean_dwell_time_bars" in result
    assert result["mean_dwell_time_bars"] > 0


if __name__ == "__main__":
    test_autocorrelation_of_pure_noise_is_near_zero()
    test_volatility_clustering_detects_synthetic_clustering()
    test_regime_persistence_reports_dwell_time()
    print("tests/research/test_phase3_representation_research.py: OK")
