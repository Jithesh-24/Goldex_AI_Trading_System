"""research/phase3_representation_research.py
Design doc Section 4: investigates whether exploitable temporal/state
information exists in the real data BEFORE candidates are finalized against
it. Produces findings, makes no KEEP/REJECT decision -- that judgment is
made by a human reading the report, same discipline as Batch 1/2's
diagnostics."""
import numpy as np

from candidates.hmm_regime import HMMRegimeCandidate


def analyze_return_autocorrelation(closes, max_lag: int = 20) -> dict:
    closes = np.asarray(closes, dtype=np.float64)
    returns = np.diff(closes)
    result = {}
    for lag in range(1, max_lag + 1):
        if len(returns) <= lag:
            continue
        a, b = returns[:-lag], returns[lag:]
        corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else 0.0
        result[f"lag_{lag}"] = corr
    return result


def analyze_volatility_clustering(closes, window: int = 60, max_lag: int = 20) -> dict:
    closes = np.asarray(closes, dtype=np.float64)
    returns = np.diff(closes)
    rolling_vol = np.array([
        np.std(returns[max(0, i - window):i]) for i in range(window, len(returns))
    ])
    result = {}
    for lag in range(1, max_lag + 1):
        if len(rolling_vol) <= lag:
            continue
        a, b = rolling_vol[:-lag], rolling_vol[lag:]
        corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else 0.0
        result[f"lag_{lag}"] = corr
    return result


def analyze_regime_persistence(hmm_candidate: HMMRegimeCandidate, closes) -> dict:
    closes = np.asarray(closes, dtype=np.float64)

    class _MinimalBar:
        def __init__(self, close):
            self.close = close

    class _MinimalMarketState:
        def __init__(self, close):
            self.completed_m1 = _MinimalBar(close)
            self.mid = close
            self.realized_vol_60s = None

    regimes = []
    for price in closes:
        result = hmm_candidate._update_belief(_MinimalMarketState(float(price)))
        regimes.append(result[0] if result is not None else None)

    dwell_times = []
    current_run = 0
    current_regime = None
    for r in regimes:
        if r is None:
            continue
        if r == current_regime:
            current_run += 1
        else:
            if current_run > 0:
                dwell_times.append(current_run)
            current_regime = r
            current_run = 1
    if current_run > 0:
        dwell_times.append(current_run)

    mean_dwell = float(np.mean(dwell_times)) if dwell_times else 0.0
    return {"mean_dwell_time_bars": mean_dwell, "n_regime_switches": len(dwell_times)}
