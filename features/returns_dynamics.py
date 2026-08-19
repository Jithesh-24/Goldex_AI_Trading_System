"""Family A -- return dynamics. Moved from research/features_v3.py,
math unchanged (spec section 4). Causal by construction: only
np.roll/pd.Series.rolling/.shift and backward-scanning numba loops."""
import numba
import numpy as np
import pandas as pd

from features._shared import SharedInputs


@numba.njit(cache=True)
def run_length_signed(sign):
    n = len(sign)
    out = np.zeros(n, dtype=np.float64)
    cur = 0.0
    prev = 0.0
    for i in range(n):
        s = sign[i]
        if s == 0.0:
            cur = 0.0
        elif s == prev:
            cur += s
        else:
            cur = s
        out[i] = cur
        prev = s
    return out


@numba.njit(cache=True)
def rolling_autocorr_lag1(x, window):
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(window, n):
        seg = x[i - window:i]
        m = seg.mean()
        a = seg[:-1] - m
        b = seg[1:] - m
        denom = np.sqrt((a * a).sum() * (b * b).sum())
        out[i] = (a * b).sum() / denom if denom > 1e-12 else 0.0
    return out


@numba.njit(cache=True)
def rolling_pacf1_ar2(x, window):
    """Partial autocorrelation at lag 1 via a 2-lag Yule-Walker solve
    (removes the lag-2 dependency's indirect contribution to lag-1's raw
    ACF -- the textbook definition of PACF(1) beyond trivial ACF(1))."""
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(window, n):
        seg = x[i - window:i]
        m = seg.mean()
        d = seg - m
        r0 = (d * d).sum()
        if r0 < 1e-18:
            out[i] = 0.0
            continue
        r1 = (d[1:] * d[:-1]).sum() / r0
        r2 = (d[2:] * d[:-2]).sum() / r0 if window > 2 else 0.0
        denom = 1 - r1 * r1
        out[i] = r1 if abs(denom) < 1e-9 else (r2 - r1 * r1) / denom
    return out


@numba.njit(cache=True)
def sign_flip_rate(sign, window):
    n = len(sign)
    out = np.full(n, np.nan)
    for i in range(window, n):
        seg = sign[i - window:i]
        flips = 0
        for j in range(1, window):
            if seg[j] != 0.0 and seg[j - 1] != 0.0 and seg[j] != seg[j - 1]:
                flips += 1
        out[i] = flips / (window - 1)
    return out


@numba.njit(cache=True)
def directional_entropy(sign, window):
    """Shannon entropy (base 2) of the {up, down, flat} proportions over the
    trailing window -- distinct from D's magnitude-distribution entropy,
    this is purely about the DIRECTION sequence."""
    n = len(sign)
    out = np.full(n, np.nan)
    for i in range(window, n):
        seg = sign[i - window:i]
        up = 0
        down = 0
        flat = 0
        for j in range(window):
            if seg[j] > 0:
                up += 1
            elif seg[j] < 0:
                down += 1
            else:
                flat += 1
        h = 0.0
        for c in (up, down, flat):
            if c > 0:
                p = c / window
                h -= p * np.log2(p)
        out[i] = h
    return out


def compute_returns_dynamics(shared: SharedInputs) -> dict:
    ret1, sign1 = shared.ret1, shared.sign1
    ret1_s = pd.Series(ret1)
    log_c = shared.log_c
    base_feat = shared.base_feat
    f = {}
    f["ret_240"] = log_c - np.roll(log_c, 240)
    f["ret_240"][:240] = np.nan
    f["sign_ret_240"] = np.sign(f["ret_240"])
    f["ret_accel_5_15"] = base_feat["ret_5"].to_numpy() - base_feat["ret_15"].to_numpy()
    f["ret_decel_15_60"] = base_feat["ret_15"].to_numpy() - base_feat["ret_60"].to_numpy()
    f["run_length_signed"] = run_length_signed(sign1)
    f["return_autocorr_20"] = rolling_autocorr_lag1(ret1, 20)
    f["return_autocorr_60"] = rolling_autocorr_lag1(ret1, 60)
    f["return_pacf1_60"] = rolling_pacf1_ar2(ret1, 60)
    f["sign_flip_rate_20"] = sign_flip_rate(sign1, 20)
    f["rolling_mean_ret_20"] = ret1_s.rolling(20).mean().to_numpy()
    f["rolling_median_ret_20"] = ret1_s.rolling(20).median().to_numpy()
    f["return_dispersion_20"] = ret1_s.rolling(20).std().to_numpy()
    up = np.where(ret1 > 0, ret1, np.nan)
    down = np.where(ret1 < 0, -ret1, np.nan)
    up_mean_60 = pd.Series(up).rolling(60, min_periods=5).mean()
    down_mean_60 = pd.Series(down).rolling(60, min_periods=5).mean()
    f["upside_downside_asymmetry_60"] = (up_mean_60 / down_mean_60).to_numpy()
    f["return_skew_60"] = ret1_s.rolling(60).skew().to_numpy()
    f["return_kurt_60"] = ret1_s.rolling(60).kurt().to_numpy()
    f["return_skew_240"] = ret1_s.rolling(240).skew().to_numpy()
    f["return_percentile_rank_60"] = ret1_s.rolling(60).rank(pct=True).to_numpy()
    ret15 = base_feat["ret_15"].to_numpy()
    f["return_quantile_pos_240"] = pd.Series(ret15).rolling(240).rank(pct=True).to_numpy()
    f["directional_entropy_60"] = directional_entropy(sign1, 60)
    return f
