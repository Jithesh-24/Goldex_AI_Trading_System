"""Family I -- discretized regime/state variables from already-computed
continuous features (NOT an HMM). Moved from research/features_v3.py
lines 655-677. Consumes family C's bars_since array directly instead of
recomputing CUSUM state (see Task 7's internal-key handoff), and family
D's shannon_entropy_returns_60 / family H's tick_volume_zscore_60 via a
merged `upstream` dict rather than recomputing them (would duplicate
family D/H)."""
import numpy as np
import pandas as pd

from features._shared import SharedInputs


def causal_tercile(x, window):
    """Rolling (trailing, shift(1)) tercile bucket -- unlike pd.cut(x, 3),
    which fixes bin edges from the WHOLE series (past+future, a leakage
    bug the original research caught during smoke-testing), this only
    ever uses thresholds computed from data strictly before the current
    row."""
    s = pd.Series(x)
    lo = s.rolling(window, min_periods=window // 4).quantile(0.333).shift(1)
    hi = s.rolling(window, min_periods=window // 4).quantile(0.667).shift(1)
    return np.where(s <= lo, 0.0, np.where(s >= hi, 2.0, 1.0))


def compute_regime_state(shared: SharedInputs, upstream: dict) -> dict:
    """upstream must contain: _bars_since_last_changepoint_internal (family
    C's internal key, from compute_jump_detection), shannon_entropy_returns_60
    (family D, from compute_distribution_info), tick_volume_zscore_60
    (family H, from compute_microstructure_history)."""
    ewma_vol, times = shared.ewma_vol, shared.times
    bars_since = upstream["_bars_since_last_changepoint_internal"]
    f = {}
    ev_daily2 = pd.Series(ewma_vol, index=times).resample("1D").last()
    tercile = ev_daily2.rolling(252, min_periods=60).apply(
        lambda w: np.searchsorted(np.percentile(w, [33.3, 66.7]), w[-1]), raw=True).shift(1)
    vol_state_tercile = tercile.reindex(times, method="ffill").to_numpy()
    f["vol_state_tercile"] = vol_state_tercile
    f["jump_state"] = np.where(bars_since <= 5, 2.0, np.where(bars_since <= 20, 1.0, 0.0))
    persistence_score = shared.hurst_120 - 0.5
    f["persistence_state"] = causal_tercile(persistence_score, 500)
    f["entropy_state"] = causal_tercile(upstream["shannon_entropy_returns_60"], 500)
    f["activity_state"] = causal_tercile(upstream["tick_volume_zscore_60"], 500)
    f["changepoint_state"] = np.where(bars_since <= 10, 0.0, np.where(bars_since <= 60, 1.0, 2.0))
    f["composite_state_id"] = np.nan_to_num(vol_state_tercile, nan=0) * 3 + np.nan_to_num(f["persistence_state"], nan=0)
    return f
