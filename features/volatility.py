"""
Volatility estimators for barrier scaling and features.
All functions are causal: value at index i uses only data <= i (no look-ahead).
Every function takes plain numpy arrays (open/high/low/close) so it can run
on any bar timeframe (M1, M5, ...) without assumptions baked in.
"""
import numpy as np
import pandas as pd


def ewma_vol(returns: np.ndarray, span: int = 100) -> np.ndarray:
    """Causal EWMA volatility of returns (de Prado's daily-vol scaling, applied
    at whatever bar frequency `returns` is sampled). span in bars, not days.
    Vectorized via pandas .ewm() (causal: each row only uses <= that row)."""
    s = pd.Series(np.asarray(returns, dtype=np.float64))
    var = s.ewm(span=span, min_periods=1, adjust=False).var(bias=False)
    return np.sqrt(var.clip(lower=0.0)).to_numpy()


def garman_klass(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                  close: np.ndarray, window: int = 20) -> np.ndarray:
    """Garman-Klass per-bar variance proxy, rolling-averaged over `window` bars.
    Uses full OHLC range instead of just close-to-close returns -> more
    efficient than ATR/close-based vol for the same window. Assumes ~zero drift."""
    o, h, l, c = map(lambda a: np.asarray(a, dtype=np.float64), (open_, high, low, close))
    with np.errstate(divide="ignore", invalid="ignore"):
        log_hl = np.log(h / l)
        log_co = np.log(c / o)
    gk = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
    gk = np.where(np.isfinite(gk), gk, np.nan)
    return _causal_rolling_mean(gk, window)


def rogers_satchell(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                     close: np.ndarray, window: int = 20) -> np.ndarray:
    """Rogers-Satchell per-bar variance proxy — drift-independent (better than
    Garman-Klass for a trending instrument like gold), rolling-averaged."""
    o, h, l, c = map(lambda a: np.asarray(a, dtype=np.float64), (open_, high, low, close))
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = (np.log(h / c) * np.log(h / o)) + (np.log(l / c) * np.log(l / o))
    rs = np.where(np.isfinite(rs), rs, np.nan)
    return _causal_rolling_mean(rs, window)


def yang_zhang(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
               close: np.ndarray, window: int = 20) -> np.ndarray:
    """Yang-Zhang variance estimator: combines overnight (close->open) jump
    variance with a drift-independent intrabar (Rogers-Satchell) component.
    Best single all-around OHLC vol estimator for gapping instruments."""
    o, h, l, c = map(lambda a: np.asarray(a, dtype=np.float64), (open_, high, low, close))
    n = len(c)
    prev_c = np.roll(c, 1)
    prev_c[0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        log_oc = np.log(o / prev_c)          # overnight/gap return
        log_co = np.log(c / o)               # open-to-close return
    overnight_var = _causal_rolling_var(log_oc, window)
    open_var = _causal_rolling_var(log_co, window)
    rs = rogers_satchell(o, h, l, c, window=1)  # per-bar RS, not yet averaged
    rs_var = _causal_rolling_mean(rs, window)
    k = 0.34 / (1.34 + (window + 1.0) / max(window - 1.0, 1.0))
    yz = overnight_var + k * open_var + (1 - k) * rs_var
    return np.sqrt(np.clip(yz, 0.0, None))


def bipower_variation(sub_returns: np.ndarray, window: int) -> np.ndarray:
    """Barndorff-Nielsen & Shephard bipower variation over a rolling window of
    fine-grained (e.g. tick or M1-within-M5) sub-returns. Robust to jumps —
    the gap between realized variance and bipower variation isolates the jump
    component. `sub_returns` must be causal (each row's returns known by
    that row's timestamp)."""
    r = np.asarray(sub_returns, dtype=np.float64)
    abs_r = np.abs(r)
    prod = abs_r[1:] * abs_r[:-1]
    prod = np.concatenate([[np.nan], prod])
    mu1_sq_inv = np.pi / 2.0  # (E|Z|)^-2 for standard normal, BNS scaling constant
    bv = _causal_rolling_mean(prod, window) * mu1_sq_inv
    return bv


def realized_variance(sub_returns: np.ndarray, window: int) -> np.ndarray:
    r = np.asarray(sub_returns, dtype=np.float64)
    return _causal_rolling_mean(r ** 2, window)


def jump_component(sub_returns: np.ndarray, window: int) -> np.ndarray:
    """RV - BV, floored at 0. Large positive value = a jump occurred in the
    window (useful around scheduled events the system already flags by
    proximity, this quantifies actual jump magnitude)."""
    rv = realized_variance(sub_returns, window)
    bv = bipower_variation(sub_returns, window)
    return np.clip(rv - bv, 0.0, None)


def _causal_rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Vectorized (pandas) causal rolling mean — window looks strictly
    backward from each index, min_periods=1 so it's defined near the start."""
    if window <= 1:
        return np.asarray(x, dtype=np.float64).copy()
    s = pd.Series(np.asarray(x, dtype=np.float64))
    return s.rolling(window, min_periods=1).mean().to_numpy()


def _causal_rolling_var(x: np.ndarray, window: int) -> np.ndarray:
    """Vectorized (pandas) causal rolling sample variance (ddof=1)."""
    s = pd.Series(np.asarray(x, dtype=np.float64))
    return s.rolling(window, min_periods=2).var(ddof=1).to_numpy()
