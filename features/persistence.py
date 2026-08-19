"""Family F -- mean reversion / persistence. Moved from
research/features_v3.py lines 595-610. Reuses rolling_autocorr_lag1 from
features.returns_dynamics (family A) -- the one kernel genuinely shared
across two families, not duplicated."""
import numba
import numpy as np
import pandas as pd

from features._shared import SharedInputs
from features.hurst import rolling_hurst
from features.returns_dynamics import rolling_autocorr_lag1


@numba.njit(cache=True)
def mean_reversion_speed(close, window):
    """OLS slope of delta_close[t] ~ (close[t-1] - rolling_mean[t-1]) over
    the trailing window -- a direct empirical Ornstein-Uhlenbeck-style
    mean-reversion-speed coefficient, distinct from the Hurst exponent
    (a scaling-law estimate) already in the base feature set."""
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(window + 1, n):
        seg = close[i - window - 1:i]
        m = seg[:-1].mean()
        x = seg[:-1] - m           # deviation from local mean
        y = seg[1:] - seg[:-1]     # next-step change
        xm = x.mean()
        ym = y.mean()
        denom = ((x - xm) ** 2).sum()
        out[i] = ((x - xm) * (y - ym)).sum() / denom if denom > 1e-15 else 0.0
    return out


@numba.njit(cache=True)
def autocorr_decay_rate(ret, window, lags):
    """Fits ACF(lag) ~ exp(-k*lag) via a simple log-linear regression across
    a handful of lags -- the decay RATE k, distinct from any single-lag ACF
    value already computed elsewhere."""
    n = len(ret)
    n_lags = len(lags)
    out = np.full(n, np.nan)
    for i in range(window + lags[-1], n):
        seg = ret[i - window:i]
        m = seg.mean()
        d = seg - m
        r0 = (d * d).sum()
        if r0 < 1e-15:
            out[i] = 0.0
            continue
        log_acfs = np.empty(n_lags)
        valid = 0
        xs = np.empty(n_lags)
        for li in range(n_lags):
            lag = lags[li]
            acf = (d[lag:] * d[:-lag]).sum() / r0
            if acf > 1e-6:
                log_acfs[valid] = np.log(acf)
                xs[valid] = lag
                valid += 1
        if valid < 2:
            out[i] = 0.0
            continue
        xm = xs[:valid].mean()
        ym = log_acfs[:valid].mean()
        denom = ((xs[:valid] - xm) ** 2).sum()
        slope = ((xs[:valid] - xm) * (log_acfs[:valid] - ym)).sum() / denom if denom > 1e-12 else 0.0
        out[i] = -slope  # positive decay rate
    return out


def compute_persistence(shared: SharedInputs) -> dict:
    ret1, c = shared.ret1, shared.c
    kalman_resid, hurst_120, base_feat = shared.kalman_resid, shared.hurst_120, shared.base_feat
    f = {}
    f["hurst_240"] = rolling_hurst(ret1, window=240)
    f["mean_reversion_speed_60"] = mean_reversion_speed(c, 60)
    speed = f["mean_reversion_speed_60"]
    with np.errstate(invalid="ignore", divide="ignore"):
        f["half_life_60"] = np.where(speed < 0, -np.log(2) / np.log(1 + speed), np.nan)
    f["autocorr_decay_rate_60"] = autocorr_decay_rate(ret1, 240, np.array([1, 2, 3, 5, 10], dtype=np.int64))
    f["persistence_score"] = hurst_120 - 0.5
    f["residual_mean_reversion_60"] = rolling_autocorr_lag1(np.nan_to_num(kalman_resid), 60)
    fracdiff = base_feat["fracdiff_log_price"].to_numpy()
    fd_s = pd.Series(fracdiff)
    x_idx = pd.Series(np.arange(len(fd_s), dtype=np.float64))
    cov = fd_s.rolling(60).cov(x_idx)
    var = x_idx.rolling(60).var()
    f["fracdiff_slope_60"] = (cov / var).to_numpy()
    return f
