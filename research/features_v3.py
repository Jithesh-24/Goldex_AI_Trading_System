"""
Phase 3B Part 1 -- candidate quantitative feature library, RESEARCH ONLY.
Not wired into core/features.py, core/train.py's default path, or the live
engine. Every column here is causal (value at row i depends only on data at
or before i) -- enforced both by construction (only backward pandas
.rolling()/.shift()/.expanding() or backward-scanning numba loops are used)
and verified by research/v3_causality_check.py's truncation test.

~92 NEW candidates across 10 families (families A-J below), on top of the
existing 26 (core/features.py). Deliberately NOT 92 window-variations of the
same handful of stats -- each family caps extra window-variants of an
already-present statistic at 0-2 and instead adds genuinely different
mathematical objects (autocorrelation vs raw returns, entropy vs skew,
mean-reversion SPEED vs Hurst EXPONENT, etc). Family H (microstructure) is
honestly scoped down to what XM/MT5's M1 history actually contains --
tick_volume (per-bar tick COUNT) and spread -- no tick-by-tick arrival
times, no bid/ask stream, no order book exist in this dataset, so true
tick-arrival-intensity/inter-arrival-variance/order-flow-imbalance are NOT
implemented (would be fabricated). Family I builds simple discretized
STATE variables from already-computed continuous features (no HMM) so the
research pipeline can empirically test "do explicit states help" before
any Markov-switching model is considered.

None of this is the production feature set. research/v3_feature_selection.py
decides, from OOS evidence, which (if any) candidates survive.
"""
import numpy as np
import numba
import pandas as pd

# ---------------------------------------------------------------------------
# numba kernels -- only used where no vectorized pandas equivalent exists
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def _run_length_signed(sign):
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
def _rolling_autocorr_lag1(x, window):
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
def _rolling_pacf1_ar2(x, window):
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
def _sign_flip_rate(sign, window):
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
def _directional_entropy(sign, window):
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


@numba.njit(cache=True)
def _shannon_entropy_returns(ret, window, nbins):
    n = len(ret)
    out = np.full(n, np.nan)
    for i in range(window, n):
        seg = ret[i - window:i]
        lo = seg.min()
        hi = seg.max()
        rng = hi - lo
        if rng < 1e-15:
            out[i] = 0.0
            continue
        counts = np.zeros(nbins, dtype=np.float64)
        for j in range(window):
            b = int((seg[j] - lo) / rng * nbins)
            if b >= nbins:
                b = nbins - 1
            counts[b] += 1
        h = 0.0
        for c in counts:
            if c > 0:
                p = c / window
                h -= p * np.log2(p)
        out[i] = h
    return out


@numba.njit(cache=True)
def _permutation_entropy(ret, window, order):
    """Bandt-Pompe permutation entropy, order=3 (6 possible orderings of 3
    consecutive values) -- ordinal-pattern complexity, distinct from the
    value-distribution entropy above (this is invariant to monotonic
    rescaling, sensitive only to up/down ORDER patterns)."""
    n = len(ret)
    out = np.full(n, np.nan)
    n_patterns = 1
    for k in range(2, order + 1):
        n_patterns *= k  # order! possible orderings
    for i in range(window, n):
        seg = ret[i - window:i]
        counts = np.zeros(n_patterns, dtype=np.float64)
        n_windows = window - order + 1
        for j in range(n_windows):
            a, b, c = seg[j], seg[j + 1], seg[j + 2]
            # rank the 3 values into one of 6 patterns (order=3 fixed here)
            if a < b:
                if b < c:
                    p = 0
                elif a < c:
                    p = 1
                else:
                    p = 2
            else:
                if a < c:
                    p = 3
                elif b < c:
                    p = 4
                else:
                    p = 5
            counts[p] += 1
        h = 0.0
        for cnt in counts:
            if cnt > 0:
                p = cnt / n_windows
                h -= p * np.log2(p)
        out[i] = h / np.log2(n_patterns)  # normalized to [0,1]
    return out


@numba.njit(cache=True)
def _sample_entropy(ret, window, m, r_mult):
    """Sample entropy, deliberately small window (20, not 60+) -- SampEn is
    an O(window^2) estimator and unstable on short samples; a 60-bar window
    would be both slow at 2.4M-row scale and a noisy estimate, so this uses
    the smaller window the estimator is actually reliable at (matches the
    library's own 'avoid unstable estimators when sample size is
    inadequate' instruction)."""
    n = len(ret)
    out = np.full(n, np.nan)
    for i in range(window, n):
        seg = ret[i - window:i]
        r = r_mult * seg.std()
        if r < 1e-15:
            out[i] = 0.0
            continue
        nm = window - m
        nm1 = window - m - 1
        count_m = 0
        count_m1 = 0
        for a in range(nm):
            for b in range(a + 1, nm):
                match_m = True
                for k in range(m):
                    if abs(seg[a + k] - seg[b + k]) > r:
                        match_m = False
                        break
                if match_m:
                    count_m += 1
                    if a < nm1 and b < nm1 and abs(seg[a + m] - seg[b + m]) <= r:
                        count_m1 += 1
        if count_m == 0 or count_m1 == 0:
            out[i] = np.nan
        else:
            out[i] = -np.log(count_m1 / count_m)
    return out


@numba.njit(cache=True)
def _mi_proxy_sign_lag(sign, lag, window):
    """Cheap causal mutual-information proxy: plug-in MI estimate between
    sign(ret_1) and its own value `lag` bars earlier, over the trailing
    window, via a 3x3 (up/down/flat x up/down/flat) contingency table --
    avoids an expensive kernel-density MI estimator while still answering
    'is there information in the lagged sign beyond linear correlation'."""
    n = len(sign)
    out = np.full(n, np.nan)
    for i in range(window + lag, n):
        joint = np.zeros((3, 3))
        for j in range(i - window, i):
            a = sign[j]
            b = sign[j - lag]
            ai = 0 if a > 0 else (1 if a < 0 else 2)
            bi = 0 if b > 0 else (1 if b < 0 else 2)
            joint[ai, bi] += 1.0
        joint /= window
        pa = joint.sum(axis=1)
        pb = joint.sum(axis=0)
        mi = 0.0
        for x in range(3):
            for y in range(3):
                if joint[x, y] > 1e-12 and pa[x] > 1e-12 and pb[y] > 1e-12:
                    mi += joint[x, y] * np.log2(joint[x, y] / (pa[x] * pb[y]))
        out[i] = mi
    return out


@numba.njit(cache=True)
def _mean_reversion_speed(close, window):
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
def _autocorr_decay_rate(ret, window, lags):
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


@numba.njit(cache=True)
def _breakout_failure_magnitude(close, high, low, window, lookback):
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
def _avg_run_length(sign, window):
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
def _high_low_density(high, low, window):
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


@numba.njit(cache=True)
def _first_passage_stats(ret, close, window, sub_horizon, r_threshold_frac):
    """For each of the trailing `window` bars (all fully resolved before
    the current bar -- causal), retrospectively check: did price move
    >= r_threshold_frac (a fraction of that bar's local close) within the
    next sub_horizon bars, and how long did it take? Aggregates into a
    LOCAL empirical first-passage probability/time/frequency -- distinct
    from any global constant, and distinct from the barrier-touch labels
    themselves (this only ever looks at fully-resolved past outcomes)."""
    n = len(close)
    p_reach = np.full(n, np.nan)
    time_to = np.full(n, np.nan)
    hit_freq = np.full(n, np.nan)
    fav_adv_ratio = np.full(n, np.nan)
    for i in range(window + sub_horizon, n):
        n_hits = 0
        n_checked = 0
        total_time = 0.0
        n_timed = 0
        fav_sum = 0.0
        adv_sum = 0.0
        for t0 in range(i - window - sub_horizon, i - sub_horizon):
            p0 = close[t0]
            thr = r_threshold_frac * p0
            hit_t = -1
            best_fav = 0.0
            best_adv = 0.0
            for j in range(t0 + 1, t0 + sub_horizon + 1):
                d = close[j] - p0
                if d > best_fav:
                    best_fav = d
                if -d > best_adv:
                    best_adv = -d
                if hit_t < 0 and (d >= thr or -d >= thr):
                    hit_t = j - t0
            n_checked += 1
            fav_sum += best_fav
            adv_sum += best_adv
            if hit_t > 0:
                n_hits += 1
                total_time += hit_t
                n_timed += 1
        p_reach[i] = n_hits / n_checked if n_checked > 0 else np.nan
        time_to[i] = total_time / n_timed if n_timed > 0 else sub_horizon
        hit_freq[i] = n_hits / (n_checked * sub_horizon) if n_checked > 0 else np.nan
        fav_adv_ratio[i] = fav_sum / adv_sum if adv_sum > 1e-12 else np.nan
    return p_reach, time_to, hit_freq, fav_adv_ratio


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

def build_candidate_features(df: pd.DataFrame, base_feat: pd.DataFrame) -> pd.DataFrame:
    """df: raw OHLCV (+tick_volume/spread). base_feat: output of
    core.features.build_features(df) (the existing 26 columns, reused here
    instead of recomputed -- e.g. ewma_vol, hurst_120, kalman_residual_z).
    Returns ONLY the new candidate columns, aligned to df's row order."""
    o = df["open"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)
    log_c = np.log(c)
    ret1 = np.diff(log_c, prepend=log_c[0])
    ret1[0] = np.nan
    sign1 = np.sign(ret1)
    sign1 = np.nan_to_num(sign1, nan=0.0)
    times = pd.to_datetime(df["time"].to_numpy())
    ret1_s = pd.Series(ret1)
    close_s = pd.Series(c)
    ewma_vol = base_feat["ewma_vol"].to_numpy(dtype=np.float64)
    kalman_resid = base_feat["kalman_residual_z"].to_numpy(dtype=np.float64)
    hurst_120 = base_feat["hurst_120"].to_numpy(dtype=np.float64)
    tick_vol = df["tick_volume"].to_numpy(dtype=np.float64) if "tick_volume" in df.columns else np.full(len(df), np.nan)
    spread = df["spread"].to_numpy(dtype=np.float64) if "spread" in df.columns else np.full(len(df), np.nan)

    f = {}

    # ---- A. return dynamics ----
    f["ret_240"] = log_c - np.roll(log_c, 240); f["ret_240"][:240] = np.nan
    f["sign_ret_240"] = np.sign(f["ret_240"])
    f["ret_accel_5_15"] = base_feat["ret_5"].to_numpy() - base_feat["ret_15"].to_numpy()
    f["ret_decel_15_60"] = base_feat["ret_15"].to_numpy() - base_feat["ret_60"].to_numpy()
    f["run_length_signed"] = _run_length_signed(sign1)
    f["return_autocorr_20"] = _rolling_autocorr_lag1(ret1, 20)
    f["return_autocorr_60"] = _rolling_autocorr_lag1(ret1, 60)
    f["return_pacf1_60"] = _rolling_pacf1_ar2(ret1, 60)
    f["sign_flip_rate_20"] = _sign_flip_rate(sign1, 20)
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
    f["directional_entropy_60"] = _directional_entropy(sign1, 60)

    # ---- B. volatility ----
    f["realized_variance_20"] = (ret1_s ** 2).rolling(20).sum().to_numpy()
    ret1_up = pd.Series(np.where(ret1 > 0, ret1, 0.0))
    ret1_down = pd.Series(np.where(ret1 < 0, ret1, 0.0))
    f["realized_semivar_upside_20"] = (ret1_up ** 2).rolling(20).sum().to_numpy()
    f["realized_semivar_downside_20"] = (ret1_down ** 2).rolling(20).sum().to_numpy()
    log_hl = np.log(h / l)
    f["parkinson_vol_60"] = np.sqrt(pd.Series(log_hl ** 2).rolling(60).mean().to_numpy() / (4 * np.log(2)))
    f["vol_acceleration_30"] = ewma_vol - pd.Series(ewma_vol).shift(30).to_numpy()
    f["vol_of_vol_60"] = pd.Series(ewma_vol).rolling(60).std().to_numpy()
    ev_daily = pd.Series(ewma_vol, index=times).resample("1D").last()
    vol_pctile_252 = ev_daily.rolling(252, min_periods=60).rank(pct=True).shift(1)
    f["vol_percentile_252"] = vol_pctile_252.reindex(times, method="ffill").to_numpy()
    ev_roll_mean_60 = pd.Series(ewma_vol).rolling(60).mean()
    ev_roll_std_60 = pd.Series(ewma_vol).rolling(60).std()
    f["vol_zscore_60"] = ((pd.Series(ewma_vol) - ev_roll_mean_60) / ev_roll_std_60).to_numpy()
    gk20 = base_feat["gk_vol_20"].to_numpy()
    gk240 = base_feat["gk_vol_240"].to_numpy()
    f["vol_compression_ratio"] = np.where(gk240 > 1e-12, gk20 / gk240, np.nan)

    # ---- C. jump / change detection ----
    from learning.train import CUSUM_K
    threshold = np.clip(CUSUM_K * np.nan_to_num(ewma_vol, nan=np.nanmedian(ewma_vol)) * c, 1e-6, None)
    cusum_pos = np.zeros(len(c)); cusum_neg = np.zeros(len(c))
    sp, sn = 0.0, 0.0
    for i in range(1, len(c)):
        diff = c[i] - c[i - 1]
        sp = max(0.0, sp + diff); sn = min(0.0, sn + diff)
        if sp > threshold[i] or sn < -threshold[i]:
            sp, sn = 0.0, 0.0
        cusum_pos[i], cusum_neg[i] = sp, sn
    f["cusum_distance_to_threshold"] = np.maximum(cusum_pos, -cusum_neg) / np.clip(threshold, 1e-9, None)
    local_vol_price = ewma_vol * c
    is_jump = np.abs(ret1 * c) > 3 * np.clip(local_vol_price, 1e-9, None)
    is_jump_s = pd.Series(is_jump.astype(np.float64))
    f["jump_intensity_60"] = is_jump_s.rolling(60).sum().to_numpy()
    jump_mag = pd.Series(np.where(is_jump, np.abs(ret1), np.nan))
    f["jump_magnitude_mean_60"] = jump_mag.rolling(60, min_periods=1).mean().to_numpy()
    jump_dir = pd.Series(np.where(is_jump, sign1, np.nan))
    f["jump_direction_bias_60"] = jump_dir.rolling(60, min_periods=1).mean().to_numpy()
    changepoint = (cusum_pos == 0) & (cusum_neg == 0)
    changepoint[0] = False
    cp_idx = np.where(changepoint)[0]
    bars_since = np.full(len(c), np.nan)
    last_cp = -1
    for i in range(len(c)):
        if changepoint[i]:
            last_cp = i
        bars_since[i] = i - last_cp if last_cp >= 0 else np.nan
    f["bars_since_last_changepoint"] = bars_since
    f["changepoint_intensity_240"] = pd.Series(changepoint.astype(np.float64)).rolling(240).sum().to_numpy()
    shock = np.abs(ret1) / np.clip(ewma_vol, 1e-9, None)
    f["vol_shock_zscore"] = (pd.Series(shock) - pd.Series(shock).rolling(60).mean()).to_numpy()

    # ---- D. distribution / information theory ----
    ret_std_60 = ret1_s.rolling(60).std()
    f["tail_probability_60"] = (np.abs(ret1_s) > 2 * ret_std_60).rolling(60).mean().to_numpy()
    f["shannon_entropy_returns_60"] = _shannon_entropy_returns(ret1, 60, 8)
    f["permutation_entropy_60"] = _permutation_entropy(ret1, 60, 3)
    f["sample_entropy_20"] = _sample_entropy(ret1, 20, 2, 0.2)
    r2 = ret1_s ** 2
    f["return_concentration_60"] = ((r2 ** 2).rolling(60).sum() / (r2.rolling(60).sum() ** 2)).to_numpy()
    f["mi_proxy_sign_lag5_240"] = _mi_proxy_sign_lag(sign1, 5, 240)

    # ---- E. market-state / price geometry ----
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
    f["breakout_failure_magnitude_20"] = _breakout_failure_magnitude(c, h, l, 20, 5)
    roll_median_c_60 = close_s.rolling(60).median()
    above_median = (close_s > roll_median_c_60).astype(np.float64)
    crossings = above_median.diff().abs()
    f["reversal_frequency_60"] = crossings.rolling(60).sum().to_numpy()
    f["avg_run_length_60"] = _avg_run_length(sign1, 60)
    excursion_std_20 = close_s.rolling(20).std()
    f["excursion_from_recent_distribution_20"] = np.where(
        excursion_std_20 > 1e-9, (close_s - close_s.shift(20)) / excursion_std_20, np.nan)
    f["high_low_density_60"] = _high_low_density(h, l, 60)

    # ---- F. mean reversion / persistence ----
    from features.hurst import rolling_hurst
    f["hurst_240"] = rolling_hurst(ret1, window=240)
    f["mean_reversion_speed_60"] = _mean_reversion_speed(c, 60)
    speed = f["mean_reversion_speed_60"]
    with np.errstate(invalid="ignore", divide="ignore"):
        f["half_life_60"] = np.where(speed < 0, -np.log(2) / np.log(1 + speed), np.nan)
    f["autocorr_decay_rate_60"] = _autocorr_decay_rate(ret1, 240, np.array([1, 2, 3, 5, 10], dtype=np.int64))
    f["persistence_score"] = hurst_120 - 0.5
    f["residual_mean_reversion_60"] = _rolling_autocorr_lag1(np.nan_to_num(kalman_resid), 60)
    fracdiff = base_feat["fracdiff_log_price"].to_numpy()
    fd_s = pd.Series(fracdiff)
    x_idx = pd.Series(np.arange(len(fd_s), dtype=np.float64))
    cov = (fd_s.rolling(60).cov(x_idx))
    var = x_idx.rolling(60).var()
    f["fracdiff_slope_60"] = (cov / var).to_numpy()

    # ---- G. time / session (UTC; MT5 server time offset not modeled here -- relative encodings only) ----
    hour = times.hour.to_numpy(dtype=np.float64)
    minute = times.minute.to_numpy(dtype=np.float64)
    dow = times.dayofweek.to_numpy(dtype=np.float64)
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    f["minute_sin"] = np.sin(2 * np.pi * minute / 60)
    f["minute_cos"] = np.cos(2 * np.pi * minute / 60)
    f["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    f["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    f["session_asian"] = ((hour >= 0) & (hour < 8)).astype(np.float64)
    f["session_london"] = ((hour >= 8) & (hour < 16)).astype(np.float64)
    f["session_ny"] = ((hour >= 13) & (hour < 21)).astype(np.float64)
    f["session_london_ny_overlap"] = ((hour >= 13) & (hour < 16)).astype(np.float64)
    session_id = np.select(
        [f["session_london_ny_overlap"] > 0, f["session_london"] > 0, f["session_ny"] > 0, f["session_asian"] > 0],
        [3, 1, 2, 0], default=0)
    session_change = pd.Series(session_id).diff().abs() > 0
    f["session_transition_flag"] = session_change.astype(np.float64).to_numpy()
    hour_key = pd.Series(hour, index=times)
    ev_by_hour_ref = pd.Series(ewma_vol, index=times).groupby(hour_key.to_numpy()).transform(
        lambda s: s.expanding(min_periods=20).mean().shift(1))
    f["vol_conditional_on_session"] = (pd.Series(ewma_vol) - ev_by_hour_ref.to_numpy())
    ret5 = base_feat["ret_5"].to_numpy()
    ret5_by_hour_ref = pd.Series(ret5, index=times).groupby(hour_key.to_numpy()).transform(
        lambda s: s.expanding(min_periods=20).mean().shift(1))
    f["ret_conditional_on_session"] = (pd.Series(ret5) - ret5_by_hour_ref.to_numpy())
    tv_by_hour_ref = pd.Series(tick_vol, index=times).groupby(hour_key.to_numpy()).transform(
        lambda s: s.expanding(min_periods=20).mean().shift(1))
    f["activity_conditional_on_session"] = (pd.Series(tick_vol) - tv_by_hour_ref.to_numpy())

    # ---- H. microstructure (honestly scoped: tick_volume + spread only, no tick stream) ----
    tv_s = pd.Series(tick_vol)
    f["tick_volume_zscore_60"] = ((tv_s - tv_s.rolling(60).mean()) / tv_s.rolling(60).std()).to_numpy()
    f["tick_volume_accel_20"] = (tv_s.rolling(20).mean() - tv_s.rolling(20).mean().shift(20)).to_numpy()
    sp_s = pd.Series(spread)
    f["spread_change_1"] = sp_s.diff().to_numpy()
    sp_daily = sp_s.copy(); sp_daily.index = times
    spread_pctile = sp_daily.resample("1D").last().rolling(252, min_periods=60).rank(pct=True).shift(1)
    f["spread_percentile_252"] = spread_pctile.reindex(times, method="ffill").to_numpy()
    f["spread_volatility_60"] = sp_s.rolling(60).std().to_numpy()
    f["tick_volume_spread_ratio"] = (tv_s / (sp_s + 1.0)).to_numpy()

    # ---- I. regime/state variables (discretized from continuous features above, NOT an HMM) ----
    ev_daily2 = pd.Series(ewma_vol, index=times).resample("1D").last()
    tercile = ev_daily2.rolling(252, min_periods=60).apply(
        lambda w: np.searchsorted(np.percentile(w, [33.3, 66.7]), w[-1]), raw=True).shift(1)
    vol_state_tercile = tercile.reindex(times, method="ffill").to_numpy()
    f["vol_state_tercile"] = vol_state_tercile
    f["jump_state"] = np.where(bars_since <= 5, 2.0, np.where(bars_since <= 20, 1.0, 0.0))

    def causal_tercile(x, window):
        """Rolling (trailing, shift(1)) tercile bucket -- unlike pd.cut(x, 3),
        which fixes bin edges from the WHOLE series (past+future, a leakage
        bug caught during smoke-testing), this only ever uses thresholds
        computed from data strictly before the current row."""
        s = pd.Series(x)
        lo = s.rolling(window, min_periods=window // 4).quantile(0.333).shift(1)
        hi = s.rolling(window, min_periods=window // 4).quantile(0.667).shift(1)
        return np.where(s <= lo, 0.0, np.where(s >= hi, 2.0, 1.0))

    f["persistence_state"] = causal_tercile(f["persistence_score"], 500)
    f["entropy_state"] = causal_tercile(f["shannon_entropy_returns_60"], 500)
    f["activity_state"] = causal_tercile(f["tick_volume_zscore_60"], 500)
    f["changepoint_state"] = np.where(bars_since <= 10, 0.0, np.where(bars_since <= 60, 1.0, 2.0))
    f["composite_state_id"] = np.nan_to_num(vol_state_tercile, nan=0) * 3 + np.nan_to_num(f["persistence_state"], nan=0)

    # ---- J. first-passage / path information (fully-resolved-past-only, causal) ----
    p_reach, time_to, hit_freq, fav_adv = _first_passage_stats(ret1, c, 60, 10, 0.001)
    f["hist_p_reach_10bps_10b_60"] = p_reach
    f["hist_time_to_10bps_60"] = time_to
    f["hist_barrier_hit_freq_60"] = hit_freq
    f["hist_path_asymmetry_60"] = fav_adv

    out = pd.DataFrame(f, index=df.index)
    out.insert(0, "time", df["time"].to_numpy())
    return out
