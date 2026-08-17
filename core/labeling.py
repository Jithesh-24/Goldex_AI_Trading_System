"""
Triple-barrier labeling + CUSUM event sampling (de Prado, "Advances in
Financial Machine Learning"), adapted for a bar series (works on M1 or M5,
whatever `close/high/low` you pass in).

Root-cause fix for the old system's fixed-horizon labeling: "did price move
X in N bars" ignores path — a row can be labeled a winner even though price
would have hit the stop first in a real trade. Triple-barrier labeling asks
the honest question: which of (take-profit, stop-loss, max-holding-time) is
touched FIRST, using each bar's high/low (not just close) so the label
matches what a real trade would have experienced.

No hardcoded thresholds: pt/sl width and horizon are all parameters, driven
by a volatility estimate you supply (see core/volatility.py) — nothing here
assumes a fixed dollar/pip amount.
"""
from dataclasses import dataclass

import numba
import numpy as np
import pandas as pd


@numba.njit(cache=True)
def _cusum_filter_core(x: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    n = len(x)
    out = np.zeros(n, dtype=np.bool_)
    s_pos = 0.0
    s_neg = 0.0
    for i in range(1, n):
        diff = x[i] - x[i - 1]
        s_pos = max(0.0, s_pos + diff)
        s_neg = min(0.0, s_neg + diff)
        h = threshold[i]
        if s_pos > h:
            s_pos = 0.0
            out[i] = True
        elif s_neg < -h:
            s_neg = 0.0
            out[i] = True
    return out


def cusum_filter(price: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    """Symmetric CUSUM filter (de Prado 2.5.2.1): emits an event index only
    when cumulative signed price change since the last event exceeds a
    (time-varying, e.g. volatility-scaled) threshold. Use this to decide
    WHICH bars become candidate rows at all, instead of every single bar —
    concentrates the model on bars where something informative happened
    rather than training on 30-40M mostly-noise rows.

    `threshold` must be the same length as `price` and causal (each entry
    known at that bar) — pass e.g. k * ewma_vol(returns) from volatility.py.
    Returns a boolean mask, same length as price, True = event.
    """
    price = np.asarray(price, dtype=np.float64)
    threshold = np.asarray(threshold, dtype=np.float64)
    threshold = np.where(np.isfinite(threshold) & (threshold > 0), threshold, np.inf)
    return _cusum_filter_core(price, threshold)


@numba.njit(cache=True)
def _triple_barrier_core(close, high, low, t0_idx, upper_w, lower_w, max_hold, use_side, side):
    n_events = len(t0_idx)
    n_bars = len(close)
    t1 = np.empty(n_events, dtype=np.int64)
    touch = np.zeros(n_events, dtype=np.int8)  # 1 = upper touched, -1 = lower, 0 = vertical
    ret = np.empty(n_events, dtype=np.float64)

    for e in range(n_events):
        t0 = t0_idx[e]
        p0 = close[t0]
        s = side[e] if use_side else 1.0  # if no side given, scan for symmetric up/down
        upper = p0 * (1.0 + upper_w[e])
        lower = p0 * (1.0 - lower_w[e])
        end = t0 + max_hold[e]
        if end >= n_bars:
            end = n_bars - 1

        hit_t = end
        hit_type = 0
        for j in range(t0 + 1, end + 1):
            h = high[j]
            l = low[j]
            up_hit = h >= upper
            dn_hit = l <= lower
            if up_hit and dn_hit:
                # both touched intrabar and we can't see which came first from
                # OHLC alone — conservative: charge the adverse side for `s`
                hit_t = j
                hit_type = -1 if s >= 0 else 1
                break
            elif up_hit:
                hit_t = j
                hit_type = 1
                break
            elif dn_hit:
                hit_t = j
                hit_type = -1
                break

        t1[e] = hit_t
        touch[e] = hit_type
        ret[e] = (close[hit_t] - p0) / p0
    return t1, touch, ret


@dataclass
class TripleBarrierConfig:
    pt_mult: float = 1.0     # take-profit width = pt_mult * vol[t0] (fraction of price)
    sl_mult: float = 1.0     # stop-loss width = sl_mult * vol[t0]
    max_holding: int = 60    # vertical barrier, in bars
    min_vol: float = 1e-6    # floor to avoid zero-width barriers on dead vol readings


def triple_barrier_labels(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                           t0_idx: np.ndarray, vol: np.ndarray,
                           cfg: TripleBarrierConfig,
                           side: np.ndarray = None) -> pd.DataFrame:
    """
    close/high/low: full bar arrays (float64), same length, aligned.
    t0_idx: int array of event indices to label (e.g. from cusum_filter, or
        np.arange(len(close)) for "label every bar").
    vol: per-bar causal volatility estimate, same length as close, in RETURN
        units (e.g. ewma_vol of log returns) — barrier widths are
        pt_mult*vol[t0] and sl_mult*vol[t0], so barriers adapt to current
        volatility instead of a fixed dollar/pip amount.
    side: optional array (len == t0_idx), +1 (long) / -1 (short) per event.
        If given, upper/lower barriers are interpreted as THIS side's TP/SL
        (asymmetric pt_mult/sl_mult meaningful), and `label` is binary:
        1 = TP hit before SL (win), 0 = SL hit or vertical (not a clean win).
        This is the mode used to build the meta-labeling target: "given the
        primary model's proposed side, was it a precise entry?"
        If side is None, barriers are symmetric (pt_mult should == sl_mult)
        and `label` is the primary direction target: 1 = up touched first,
        -1 = down touched first, 0 = neither (vertical timeout).

    Returns a DataFrame indexed by t0 with columns: t1, touch, ret, label,
    holding_bars. No look-ahead beyond what's structurally required to know
    the outcome (t1 is always > t0).
    """
    close = np.asarray(close, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    t0_idx = np.asarray(t0_idx, dtype=np.int64)
    vol_at_t0 = np.asarray(vol, dtype=np.float64)[t0_idx]
    vol_at_t0 = np.where(np.isfinite(vol_at_t0) & (vol_at_t0 > cfg.min_vol),
                          vol_at_t0, cfg.min_vol)

    upper_w = np.full(len(t0_idx), cfg.pt_mult, dtype=np.float64) * vol_at_t0
    lower_w = np.full(len(t0_idx), cfg.sl_mult, dtype=np.float64) * vol_at_t0
    max_hold = np.full(len(t0_idx), cfg.max_holding, dtype=np.int64)

    use_side = side is not None
    side_arr = np.asarray(side, dtype=np.float64) if use_side else np.ones(len(t0_idx))

    t1, touch, ret = _triple_barrier_core(close, high, low, t0_idx, upper_w, lower_w,
                                           max_hold, use_side, side_arr)

    if use_side:
        # meta-label: did the proposed side's TP get hit first?  win iff
        # touch matches the side's favorable direction.
        favorable = np.where(side_arr >= 0, 1, -1)
        label = (touch == favorable).astype(np.int8)
    else:
        label = touch.astype(np.int8)  # -1 / 0 / 1

    out = pd.DataFrame({
        "t0": t0_idx,
        "t1": t1,
        "touch": touch,
        "ret": ret,
        "label": label,
        "holding_bars": t1 - t0_idx,
    })
    return out.set_index("t0")
