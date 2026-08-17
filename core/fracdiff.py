"""
Fixed-width window fractional differentiation (de Prado, ch. 5).
Raw close is non-stationary (unit root); plain returns are stationary but
throw away almost all memory. Frac-diff finds the minimal differencing
order d in (0,1] that makes the series stationary while keeping as much
memory (long-range dependence) as possible -> a better model input than
either raw price or first-difference returns.
"""
import numba
import numpy as np


def ffd_weights(d: float, thresh: float = 1e-5, max_size: int = 5000) -> np.ndarray:
    """Weights for fixed-width-window frac-diff, truncated once |w_k| < thresh."""
    w = [1.0]
    k = 1
    while k < max_size:
        w_k = -w[-1] / k * (d - k + 1)
        if abs(w_k) < thresh:
            break
        w.append(w_k)
        k += 1
    return np.array(w[::-1], dtype=np.float64)  # oldest-first, so it lines up with a sliding window


@numba.njit(cache=True)
def _causal_conv(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    n = len(x)
    width = len(w)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(width - 1, n):
        acc = 0.0
        for k in range(width):
            acc += w[k] * x[i - width + 1 + k]
        out[i] = acc
    return out


def frac_diff_ffd(series: np.ndarray, d: float, thresh: float = 1e-5) -> np.ndarray:
    """Causal fixed-width frac-diff: output[i] uses series[i-window+1 : i+1]
    only (never the future). First `window-1` entries are NaN (insufficient
    history). `d` around 0.3-0.5 is typical for FX/metals log-price."""
    x = np.asarray(series, dtype=np.float64)
    w = ffd_weights(d, thresh)
    if len(w) > len(x):
        return np.full(len(x), np.nan, dtype=np.float64)
    return _causal_conv(x, w)
