"""Family J -- first-passage / path information, fully-resolved-past-only
(causal, verified by direct code read during Phase 3 design: every inner
loop index in first_passage_stats is strictly < i, no lookahead -- see
spec section 2). Moved from research/features_v3.py lines 397-443, 679-684,
math unchanged."""
import numba
import numpy as np

from features._shared import SharedInputs


@numba.njit(cache=True)
def first_passage_stats(ret, close, window, sub_horizon, r_threshold_frac):
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


def compute_first_passage(shared: SharedInputs) -> dict:
    ret1, c = shared.ret1, shared.c
    p_reach, time_to, hit_freq, fav_adv = first_passage_stats(ret1, c, 60, 10, 0.001)
    return {
        "hist_p_reach_10bps_10b_60": p_reach,
        "hist_time_to_10bps_60": time_to,
        "hist_barrier_hit_freq_60": hit_freq,
        "hist_path_asymmetry_60": fav_adv,
    }
