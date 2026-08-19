"""Family E -- market-state / price geometry. Moved from
research/features_v3.py lines 325-394 (kernels) and 565-593 (assembly),
math unchanged."""
import numba
import numpy as np
import pandas as pd

from features._shared import SharedInputs


@numba.njit(cache=True)
def breakout_failure_magnitude(close, high, low, window, lookback):
    """If the high broke the trailing `window`-bar high within the last
    `lookback` bars, how far has price since retraced back below that
    broken level (0 if no breakout or breakout still holding)."""
    n = len(close)
    out = np.zeros(n, dtype=np.float64)
    for i in range(window + lookback, n):
        broke_level = -1.0
        for j in range(i - lookback, i):
            prior_high = high[j - window:j].max()
            if high[j] > prior_high:
                broke_level = prior_high
        if broke_level > 0 and close[i] < broke_level:
            out[i] = (broke_level - close[i]) / close[i]
    return out


@numba.njit(cache=True)
def avg_run_length(sign, window):
    """Trailing-window AVERAGE run length (distinct from the CURRENT signed
    run length already computed) -- how persistent has directionality been
    lately, on average, not just right now."""
    n = len(sign)
    out = np.full(n, np.nan)
    for i in range(window, n):
        seg = sign[i - window:i]
        total_len = 0
        n_runs = 0
        cur_len = 0
        prev = 0.0
        for j in range(window):
            s = seg[j]
            if s == 0.0:
                if cur_len > 0:
                    total_len += cur_len
                    n_runs += 1
                cur_len = 0
                prev = 0.0
            elif s == prev:
                cur_len += 1
            else:
                if cur_len > 0:
                    total_len += cur_len
                    n_runs += 1
                cur_len = 1
                prev = s
        if cur_len > 0:
            total_len += cur_len
            n_runs += 1
        out[i] = total_len / n_runs if n_runs > 0 else np.nan
    return out


@numba.njit(cache=True)
def high_low_density(high, low, window):
    """Fraction of the trailing `window` bars whose [low,high] range
    overlaps the CURRENT bar's [low,high] range -- a purely statistical
    'how congested is price right now vs recent history' proxy, not a
    hardcoded support/resistance rule."""
    n = len(high)
    out = np.full(n, np.nan)
    for i in range(window, n):
        h0, l0 = high[i], low[i]
        cnt = 0
        for j in range(i - window, i):
            if high[j] >= l0 and low[j] <= h0:
                cnt += 1
        out[i] = cnt / window
    return out


def compute_market_geometry(shared: SharedInputs) -> dict:
    c, h, l, sign1 = shared.c, shared.h, shared.l, shared.sign1
    close_s = pd.Series(c)
    f = {}
    roll_max_h_20 = pd.Series(h).rolling(20).max()
    roll_min_l_20 = pd.Series(l).rolling(20).min()
    roll_max_h_60 = pd.Series(h).rolling(60).max()
    roll_min_l_60 = pd.Series(l).rolling(60).min()
    f["dist_from_high_20"] = ((close_s - roll_max_h_20) / close_s).to_numpy()
    f["dist_from_low_20"] = ((close_s - roll_min_l_20) / close_s).to_numpy()
    rng20 = (roll_max_h_20 - roll_min_l_20)
    f["range_position_20"] = ((close_s - roll_min_l_20) / rng20).to_numpy()
    rng60 = (roll_max_h_60 - roll_min_l_60)
    f["range_position_60"] = ((close_s - roll_min_l_60) / rng60).to_numpy()
    f["range_width_20"] = (rng20 / close_s).to_numpy()
    range_width_60 = (rng60 / close_s).to_numpy()
    f["range_width_ratio_20_60"] = np.where(range_width_60 > 1e-12, f["range_width_20"] / range_width_60, np.nan)
    roll_mean_c_60 = close_s.rolling(60).mean()
    roll_std_c_60 = close_s.rolling(60).std()
    f["displacement_from_equilibrium_60"] = ((close_s - roll_mean_c_60) / roll_std_c_60).to_numpy()
    prior_high_20 = pd.Series(h).rolling(20).max().shift(1)
    f["breakout_magnitude_20"] = (np.maximum(0, close_s - prior_high_20) / close_s).to_numpy()
    f["breakout_failure_magnitude_20"] = breakout_failure_magnitude(c, h, l, 20, 5)
    roll_median_c_60 = close_s.rolling(60).median()
    above_median = (close_s > roll_median_c_60).astype(np.float64)
    crossings = above_median.diff().abs()
    f["reversal_frequency_60"] = crossings.rolling(60).sum().to_numpy()
    f["avg_run_length_60"] = avg_run_length(sign1, 60)
    excursion_std_20 = close_s.rolling(20).std()
    f["excursion_from_recent_distribution_20"] = np.where(
        excursion_std_20 > 1e-9, (close_s - close_s.shift(20)) / excursion_std_20, np.nan)
    f["high_low_density_60"] = high_low_density(h, l, 60)
    return f
