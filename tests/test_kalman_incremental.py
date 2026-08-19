"""python3 tests/test_kalman_incremental.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from features.kalman import kalman_local_level, StatefulKalman


def test_stateful_kalman_matches_batch():
    rng = np.random.default_rng(0)
    prices = 2000 + np.cumsum(rng.normal(0, 1, 200))
    q, r = 1e-5, 1.0

    batch_level, batch_velocity, batch_residual = kalman_local_level(prices, q, r)

    kf = StatefulKalman(q=q, r=r)
    live_level, live_velocity, live_residual = [], [], []
    for p in prices:
        level, velocity, residual = kf.update(p)
        live_level.append(level); live_velocity.append(velocity); live_residual.append(residual)

    assert np.allclose(batch_level, live_level, rtol=1e-9, atol=1e-12)
    assert np.allclose(batch_velocity, live_velocity, rtol=1e-9, atol=1e-12)
    assert np.allclose(batch_residual, live_residual, rtol=1e-9, atol=1e-12)


if __name__ == "__main__":
    test_stateful_kalman_matches_batch()
    print("tests/test_kalman_incremental.py: OK")
