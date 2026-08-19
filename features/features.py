"""
Pure price/volatility feature matrix, single-instrument (XAUUSD), no
cross-asset macro dependency (macro_daily.csv was a stale, slow-moving daily
join — dropped 2026-08-17 in favor of a purely self-contained model: the
system learns entirely from gold's own price action). Deliberately excludes
classic lagging indicators (SMA/EMA crossovers, RSI, MACD, Bollinger) — those
are smoothed transforms of price that react after the move. Everything here
is a statistical estimator of the CURRENT state (volatility, regime
persistence, trend level via Kalman which adapts its own lag).

Every column is causal: value at row i depends only on data at or before i.
Leading NaNs from warmup windows are expected — drop them before training,
don't fill them.
"""
import numpy as np
import pandas as pd

from features.volatility import (bipower_variation, ewma_vol, garman_klass,
                                  jump_component, rogers_satchell, yang_zhang)
from features.kalman import kalman_local_level
from features.hurst import rolling_hurst
from features.fracdiff import frac_diff_ffd

VOL_WINDOWS = (20, 60, 240)
RET_HORIZONS = (1, 5, 15, 60)


def build_tier1_features(df: pd.DataFrame) -> pd.DataFrame:
    """df must have columns: time, open, high, low, close (M1 or M5, any
    timeframe — everything here is expressed in bars of whatever df is)."""
    o = df["open"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)
    log_c = np.log(c)
    ret1 = np.diff(log_c, prepend=log_c[0])

    feat = {}

    for hz in RET_HORIZONS:
        feat[f"ret_{hz}"] = log_c - np.roll(log_c, hz)
        feat[f"ret_{hz}"][:hz] = np.nan
        # explicit sign as its own feature: GBDT histogram binning can't
        # reproduce an exact zero-threshold split as precisely as sign()
        # does, and the short-horizon reversal effect this system trades
        # lives almost entirely in the sign, not the magnitude, of recent
        # returns -- confirmed via a direct in-sample test where a
        # continuous-only single-feature model underperformed a trivial
        # sign-flip rule by ~1.5pp purely from quantization error.
        feat[f"sign_ret_{hz}"] = np.sign(feat[f"ret_{hz}"])

    ev = ewma_vol(ret1, span=100)
    feat["ewma_vol"] = ev

    for w in VOL_WINDOWS:
        feat[f"gk_vol_{w}"] = garman_klass(o, h, l, c, window=w)
        feat[f"rs_vol_{w}"] = rogers_satchell(o, h, l, c, window=w)
        feat[f"yz_vol_{w}"] = yang_zhang(o, h, l, c, window=w)

    feat["bipower_var_60"] = bipower_variation(ret1, window=60)
    feat["jump_component_60"] = jump_component(ret1, window=60)

    level, velocity, residual = kalman_local_level(c, q=1e-5, r=1.0)
    feat["kalman_level_dist"] = (c - level) / c  # normalized distance from adaptive trend
    feat["kalman_velocity"] = velocity / c  # normalized trend slope
    feat["kalman_residual_z"] = residual / np.where(ev > 1e-8, ev * c, np.nan)

    feat["hurst_120"] = rolling_hurst(ret1, window=120)
    feat["hurst_480"] = rolling_hurst(ret1, window=480)

    feat["fracdiff_log_price"] = frac_diff_ffd(log_c, d=0.4)

    feat["spread"] = df["spread"].to_numpy(dtype=np.float64) if "spread" in df.columns else np.nan
    feat["tick_volume"] = df["tick_volume"].to_numpy(dtype=np.float64) if "tick_volume" in df.columns else np.nan

    out = pd.DataFrame(feat, index=df.index)
    out.insert(0, "time", df["time"].to_numpy())
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature matrix, aligned to df's row order."""
    return build_tier1_features(df)


