"""Quick correctness smoke test for features/volatility.py. Run directly:
python3 tests/test_volatility.py
Split from core/test_smoke.py during the Phase 1 V3 features/ relocation."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from features.volatility import ewma_vol, garman_klass, rogers_satchell, yang_zhang


def test_vol_estimators_run_and_are_finite_in_steady_state():
    rng = np.random.default_rng(1)
    n = 2000
    close = 100 + np.cumsum(rng.normal(0, 0.3, n))
    open_ = close + rng.normal(0, 0.05, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.1, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.1, n))
    returns = np.diff(np.log(close), prepend=np.log(close[0]))

    ev = ewma_vol(returns, span=50)
    gk = garman_klass(open_, high, low, close, window=20)
    rs = rogers_satchell(open_, high, low, close, window=20)
    yz = yang_zhang(open_, high, low, close, window=20)

    for name, arr in [("ewma_vol", ev), ("garman_klass", gk), ("rogers_satchell", rs),
                       ("yang_zhang", yz)]:
        tail = arr[200:]
        assert np.isfinite(tail).mean() > 0.95, f"{name} produced too many NaNs in steady state"
    print("OK  volatility estimators run and are finite in steady state")


if __name__ == "__main__":
    test_vol_estimators_run_and_are_finite_in_steady_state()
    print("tests/test_volatility.py: OK")
