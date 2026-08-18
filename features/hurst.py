"""
Rolling Hurst exponent (rescaled-range method) — trend-persistence (H>0.5)
vs mean-reversion (H<0.5) regime signal. Causal: value at i uses only the
trailing `window` bars ending at i.
"""
import numba
import numpy as np


@numba.njit(cache=True)
def _hurst_rs(x: np.ndarray) -> float:
    n = len(x)
    if n < 20:
        return np.nan
    mean = x.mean()
    dev = x - mean
    cum = np.cumsum(dev)
    r = cum.max() - cum.min()
    s = np.std(x)
    if s < 1e-12 or r < 1e-12:
        return np.nan
    return np.log(r / s) / np.log(n)


@numba.njit(cache=True)
def rolling_hurst(returns: np.ndarray, window: int) -> np.ndarray:
    n = len(returns)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(window, n):
        out[i] = _hurst_rs(returns[i - window:i])
    return out
