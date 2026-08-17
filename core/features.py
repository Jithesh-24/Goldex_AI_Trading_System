"""
Tier-1 (price/vol, single-instrument) + Tier-2 (cross-asset macro) feature
matrix. Deliberately excludes classic lagging indicators (SMA/EMA crossovers,
RSI, MACD, Bollinger) — those are smoothed transforms of price that react
after the move. Everything here is either a statistical estimator of the
CURRENT state (volatility, regime persistence, trend level via Kalman which
adapts its own lag) or genuinely different information (cross-asset macro).

Every column is causal: value at row i depends only on data at or before i.
Leading NaNs from warmup windows are expected — drop them before training,
don't fill them.
"""
import os

import numpy as np
import pandas as pd

from core.volatility import (bipower_variation, ewma_vol, garman_klass,
                              jump_component, rogers_satchell, yang_zhang)
from core.kalman import kalman_local_level
from core.hurst import rolling_hurst
from core.fracdiff import frac_diff_ffd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MACRO_CSV = os.path.join(BASE, "macro_daily.csv")

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


def load_macro_daily(path: str = MACRO_CSV) -> pd.DataFrame:
    m = pd.read_csv(path, parse_dates=["date"])
    m["date"] = m["date"].dt.tz_localize(None).dt.normalize()
    m = m.sort_values("date").reset_index(drop=True)
    return m


def merge_tier2_macro(df: pd.DataFrame, macro: pd.DataFrame = None) -> pd.DataFrame:
    """As-of merge of daily cross-asset macro features onto the bar series.
    Shifted by 1 calendar day (row for date D reflects D's close, which
    isn't known until D ends) so no same-day lookahead leaks into intraday
    bars trading on day D."""
    if macro is None:
        macro = load_macro_daily()
    m = macro.copy()
    m["date"] = m["date"] + pd.Timedelta(days=1)  # available starting next day
    macro_cols = [c for c in m.columns if c != "date"]

    # df["time"] is chronological -> bar_date is non-decreasing, satisfies
    # merge_asof's sorted-key requirement without needing an explicit sort
    # (which would silently desync row order from df).
    left = pd.DataFrame({"date": pd.to_datetime(df["time"]).dt.normalize()})
    merged = pd.merge_asof(left, m.sort_values("date"), on="date", direction="backward")

    result = df.reset_index(drop=True).copy()
    for c in macro_cols:
        result[f"macro_{c}"] = merged[c].to_numpy()
    return result


def build_features(df: pd.DataFrame, macro: pd.DataFrame = None) -> pd.DataFrame:
    """Full Tier-1 + Tier-2 feature matrix, aligned to df's row order."""
    tier1 = build_tier1_features(df)
    full = merge_tier2_macro(tier1, macro=macro)
    return full


if __name__ == "__main__":
    import time
    from core.data import load_raw_m1, to_m5

    df = load_raw_m1()
    df5 = to_m5(df)
    print(f"building features on {len(df5):,} M5 bars...")
    t0 = time.time()
    feats = build_features(df5)
    print(f"done in {time.time() - t0:.1f}s, shape={feats.shape}")
    print(feats.tail(3).T)
    nan_frac = feats.drop(columns=["time"]).isna().mean().sort_values(ascending=False)
    print("\ntop NaN fractions (warmup expected at top of file, should be near-0 elsewhere):")
    print(nan_frac.head(10))
