"""research/genesis_event_time_test.py
Genesis reset -- Test 1: event-time information conditioning.

All 27 prior hypotheses (Phase 3, 3A, 4, the horizon sweep) pooled every row
in the training partition together when computing MI between a
representation and forward returns. This script asks a narrower question:
is there information specifically *around scheduled macro announcements*
(NFP, CPI, FOMC) that gets averaged away when pooling all rows together?
It buckets rows by proximity to an approximate scheduled-event calendar and
re-runs the exact same validated MI-vs-shuffled-null estimator
(binned_mutual_information / mi_with_shuffle_control, imported unchanged
from research/phase3a_representation_experiments.py) inside each bucket.

DATA USED: data/gold_seed_merged_full6yr.csv rows 0:300,000 only (same
TRAINING_ROWS convention as every prior phase). The reserved OOS holdout,
rows 300,000:400,000, is never read by this script. The training window's
calendar date range is 2019-12-02 00:00 through 2020-09-17 07:59 (checked
directly off the CSV) -- the event calendar below only needs to cover that
span.

TIMEZONE: the `time` column is timestamp-naive. Existing repo convention
(research/phase5_ev_dataset.py, comment at line ~105-108) tz-localizes this
column to UTC when timezone-aware comparisons are needed elsewhere in the
codebase. This script follows that same convention: the data's `time`
column is treated as UTC. All macro-event release times are specified in
US Eastern local time (as officially published) and converted to UTC via
zoneinfo's America/New_York, which correctly handles the EST/EDT transition
in March 2020 that falls inside the training window.

EVENT CALENDAR -- HONESTY ABOUT WHICH DATES ARE EXACT VS. APPROXIMATED:
  * FOMC rate-decision dates: EXACT. These are fixed historical public
    record (announcement always at 2:00pm ET on the second day of a
    two-day meeting). Hardcoded from public FOMC meeting-date history for
    the meetings that fall inside the training window (Dec 2019 - Sep
    2020). Note: the emergency intermeeting rate cuts of Mar 3, 2020 and
    Mar 15, 2020 are NOT included -- they were unscheduled, ad hoc actions,
    not part of a pre-known recurring calendar, so including them would
    violate the "pick the calendar without looking at outcomes" spirit of
    a *scheduled*-event test. Only the 8 regularly scheduled FOMC decision
    dates are used.
  * NFP release dates: APPROXIMATED as "first Friday of the month, 8:30am
    ET". This is the standard rule of thumb but is not exact: BLS
    occasionally shifts the release by a few days around holidays (e.g.
    the real July-2020 NFP report was released Aug 7 which the first-
    Friday rule gets right, but the actual January 2020 report -- covering
    December 2019 data -- was released Jan 10, one week after the
    mechanical "first Friday" of Jan 3). This script deliberately uses the
    simple mechanical rule as instructed, not hand-corrected historical
    lookup dates, to keep the calendar construction principled and
    reproducible rather than cherry-picked. Expect single-week timing
    error in some months.
  * CPI release dates: APPROXIMATED, and this is the weakest part of the
    calendar. True BLS CPI release dates float within roughly the 10th-15th
    of the month with no fixed weekday rule. This script approximates each
    month's CPI release as the *second Wednesday of the month, 8:30am ET*.
    Spot-checking against known actual CPI dates in this window (e.g. real
    Jan-2020 CPI was released Jan 14, a Tuesday; real Mar-2020 CPI was
    released Mar 11, a Wednesday) shows this approximation is sometimes
    off by 1-3 calendar days and occasionally the wrong weekday. Any CPI
    bucket result should be read as noisy-calendar, not exact-event,
    conditioning.

BUCKET DEFINITIONS (fixed BEFORE looking at any MI result, based on generic
macro-announcement volatility-decay reasoning -- FX/rates volatility after
a scheduled macro release typically spikes immediately and decays over
1-4 hours, per standard market-microstructure literature on macro
announcement effects):
  A. ordinary       -- more than 4 hours after the last event AND more than
                        2 hours before the next event.
  B. pre-event       -- within 2 hours before the next event (and not
                        already inside another event's post-window).
  C. immediate post  -- within 0-60 minutes after the last event.
  D. later post      -- within 60-240 minutes (1-4 hours) after the last
                        event.
Buckets are mutually exclusive and collectively exhaustive by construction
(see `label_event_buckets`): priority order C > D > B > A per row, so a row
can never be counted in more than one bucket even when two events' windows
overlap.

REPRESENTATION: momentum_scalar and volatility_regime_transition, both
imported unchanged from research/phase3a_representation_experiments.py.
volatility_regime_transition is the trend-invariant confound-diagnostic
reference used throughout the Genesis reset (Phase 3A/4, horizon sweep).

TARGET: forward_return(closes, horizon) for horizon in (5, 15), identical
construction to the horizon sweep (research/genesis_horizon_sweep.py,
forward_return reused unchanged from there).

MODEL/METHOD: none. Marginal MI only (binned_mutual_information with a
20-permutation shuffled-label null), computed separately inside each of
the 4 buckets and once more pooled over all rows as a baseline check that
this script's own pipeline reproduces the known pooled null result. Also
computes raw-return autocorrelation (lags 1-5) within each bucket using the
bucket's own sub-sequence of returns (not the full contiguous series) --
see `bucket_autocorrelation` docstring for the exact construction and its
one caveat.

TRAIN/EXPLORATION PERIOD: rows 0:300,000 of
data/gold_seed_merged_full6yr.csv only. This script never reads or
references rows 300,000 onward.

LIMITATIONS (see also the report at
docs/superpowers/reports/2026-08-28-goldex-event-time-findings.md):
  - CPI calendar is a rough approximation (see above); NFP calendar is a
    mechanical approximation with known few-day slippage in some months.
  - Buckets B (pre-event) and C (immediate post) are, by construction, far
    smaller than bucket A (ordinary) -- roughly 2h and 1h wide respectively
    per event, against thousands of events-worth of "ordinary" minutes.
    MI and autocorrelation estimates in B/C therefore have much wider
    sampling variance than in A; do not treat a marginal B/C result with
    the same confidence as a large-N pooled result.
  - Single fixed-seed 20-permutation shuffle null per cell, same limitation
    already disclosed in every prior phase and the horizon sweep.
  - No OOS predictive check -- this is a marginal-MI information-content
    test only, exactly like the horizon sweep. No trading rule is built or
    implied.

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 research/genesis_event_time_test.py
"""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from research.phase3a_representation_experiments import (
    TRAINING_ROWS, DATA_PATH, N_BINS, RNG_SEED,
    MOMENTUM_LOOKBACK, VOL_WINDOWS,
    binned_mutual_information, mi_with_shuffle_control,
    momentum_scalar, multiscale_vol_summary, vol_regime_transition,
)
from research.genesis_horizon_sweep import forward_return

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

HORIZONS = (5, 15)
CONFOUND_CLEAR_SIGMAS = 3.0

# Bucket window widths, fixed before looking at any result (see module
# docstring for the volatility-decay reasoning behind these numbers).
PRE_EVENT_WINDOW = dt.timedelta(hours=2)
IMMEDIATE_POST_WINDOW = dt.timedelta(minutes=60)
LATER_POST_WINDOW = dt.timedelta(hours=4)

# EXACT: publicly documented scheduled FOMC rate-decision dates (2nd day of
# a 2-day meeting, announcement 2:00pm ET) that fall inside the training
# window's date range (2019-12-02 .. 2020-09-17). Excludes the unscheduled
# emergency intermeeting cuts of 2020-03-03 and 2020-03-15.
FOMC_DECISION_DATES_ET = [
    dt.date(2019, 12, 11),
    dt.date(2020, 1, 29),
    dt.date(2020, 3, 18),
    dt.date(2020, 4, 29),
    dt.date(2020, 6, 10),
    dt.date(2020, 7, 29),
    dt.date(2020, 9, 16),
]
FOMC_TIME_ET = dt.time(14, 0)

NFP_TIME_ET = dt.time(8, 30)
CPI_TIME_ET = dt.time(8, 30)


def _first_friday(year: int, month: int) -> dt.date:
    d = dt.date(year, month, 1)
    offset = (4 - d.weekday()) % 7  # weekday(): Mon=0 .. Fri=4
    return d + dt.timedelta(days=offset)


def _second_wednesday(year: int, month: int) -> dt.date:
    d = dt.date(year, month, 1)
    first_wed = d + dt.timedelta(days=(2 - d.weekday()) % 7)  # Wed=2
    return first_wed + dt.timedelta(days=7)


def _months_between(start: dt.date, end: dt.date):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


def _to_utc(date_: dt.date, time_: dt.time) -> pd.Timestamp:
    local = dt.datetime.combine(date_, time_, tzinfo=ET)
    return pd.Timestamp(local.astimezone(UTC)).tz_localize(None)


def build_event_calendar(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Builds the approximate scheduled-macro-event calendar, in UTC
    (tz-naive, to match the data's `time` column), for [start, end]
    inclusive by month. See module docstring for exact-vs-approximated
    disclosure per event type."""
    rows = []
    for label, date_ in [("FOMC", d) for d in FOMC_DECISION_DATES_ET]:
        if start <= date_ <= end:
            rows.append((_to_utc(date_, FOMC_TIME_ET), "FOMC"))

    for y, m in _months_between(start, end):
        nfp_date = _first_friday(y, m)
        if start <= nfp_date <= end:
            rows.append((_to_utc(nfp_date, NFP_TIME_ET), "NFP"))
        cpi_date = _second_wednesday(y, m)
        if start <= cpi_date <= end:
            rows.append((_to_utc(cpi_date, CPI_TIME_ET), "CPI"))

    cal = pd.DataFrame(rows, columns=["event_time_utc", "event_type"])
    cal = cal.sort_values("event_time_utc").reset_index(drop=True)
    return cal


def label_event_buckets(times: pd.Series, event_times: pd.Series) -> np.ndarray:
    """Labels each timestamp in `times` into exactly one of
    {'A_ordinary', 'B_pre_event', 'C_immediate_post', 'D_later_post'}.

    Priority order per row is C > D > B > A: if a row is within the
    immediate-post window of the most recent past event, it is C even if
    it also happens to fall within the pre-event window of the next event
    (this can only happen with a same-day back-to-back event pair, e.g.
    NFP and CPI landing very close together, which does not occur in this
    calendar but the priority rule keeps the buckets exhaustive and
    mutually exclusive in general). This guarantees no row is ever counted
    twice.
    """
    times = pd.to_datetime(times).to_numpy()
    ev = np.sort(pd.to_datetime(event_times).to_numpy())
    n = len(times)
    labels = np.full(n, "A_ordinary", dtype=object)

    # index of the most recent event at or before each row, and the next
    # event at or after each row (searchsorted on sorted event array).
    idx_next = np.searchsorted(ev, times, side="left")
    idx_prev = idx_next - 1

    has_prev = idx_prev >= 0
    has_next = idx_next < len(ev)

    time_since_prev = np.full(n, np.timedelta64("NaT"), dtype="timedelta64[ns]")
    time_since_prev[has_prev] = times[has_prev] - ev[idx_prev[has_prev]]

    time_to_next = np.full(n, np.timedelta64("NaT"), dtype="timedelta64[ns]")
    time_to_next[has_next] = ev[idx_next[has_next]] - times[has_next]

    immediate_post_ns = np.timedelta64(IMMEDIATE_POST_WINDOW)
    later_post_ns = np.timedelta64(LATER_POST_WINDOW)
    pre_event_ns = np.timedelta64(PRE_EVENT_WINDOW)

    is_c = has_prev & (time_since_prev >= np.timedelta64(0, "ns")) & (time_since_prev < immediate_post_ns)
    is_d = has_prev & ~is_c & (time_since_prev >= immediate_post_ns) & (time_since_prev < later_post_ns)
    is_b = has_next & ~is_c & ~is_d & (time_to_next > np.timedelta64(0, "ns")) & (time_to_next <= pre_event_ns)

    labels[is_d] = "D_later_post"
    labels[is_b] = "B_pre_event"
    labels[is_c] = "C_immediate_post"  # highest priority, applied last
    return labels


def bucket_autocorrelation(returns: np.ndarray, bucket_mask: np.ndarray, max_lag: int = 5) -> dict:
    """Lag-1..max_lag autocorrelation of the *sub-sequence* of raw returns
    at bucket_mask==True positions (i.e. treats the bucket's own rows,
    read off in original chronological order but skipping non-bucket rows,
    as a compact series and lag-correlates that). This is a deliberate
    simplification: within a bucket like C (immediate post-event, a 60min
    window per event), consecutive bucket rows are almost always also
    consecutive calendar minutes, so this closely approximates true
    within-window autocorrelation. For bucket A (ordinary), which is not
    a single contiguous span, consecutive bucket entries can jump across
    an excluded event window; lag-1 in that case measures correlation
    between the closest two "ordinary" minutes, not necessarily calendar-
    adjacent ones. This caveat is disclosed in the report.
    """
    sub = returns[bucket_mask]
    sub = sub[np.isfinite(sub)]
    result = {}
    for lag in range(1, max_lag + 1):
        if len(sub) <= lag:
            continue
        a, b = sub[:-lag], sub[lag:]
        corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else 0.0
        result[f"lag_{lag}"] = corr
    return result


def load_training_frame() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["time"])
    df_training = df.iloc[:TRAINING_ROWS].reset_index(drop=True)
    return df_training


def classify_cell(real_mi, null_mean, null_std):
    clears_null = real_mi > (null_mean + CONFOUND_CLEAR_SIGMAS * null_std)
    return "clears_null" if clears_null else "null_consistent"


def main():
    df = load_training_frame()
    closes = df["close"].to_numpy(dtype=np.float64)
    times = df["time"]
    raw_returns = np.full(len(closes), np.nan)
    raw_returns[1:] = closes[1:] - closes[:-1]

    start_date = times.iloc[0].date()
    end_date = times.iloc[-1].date()
    print(f"Training window date range: {start_date} .. {end_date} ({len(df)} rows)")

    calendar = build_event_calendar(start_date, end_date)
    print(f"\nEvent calendar ({len(calendar)} events): "
          f"{calendar['event_type'].value_counts().to_dict()}")

    labels = label_event_buckets(times, calendar["event_time_utc"])
    bucket_names = ["A_ordinary", "B_pre_event", "C_immediate_post", "D_later_post"]
    counts = {b: int((labels == b).sum()) for b in bucket_names}
    print(f"Bucket row counts: {counts}  (sum={sum(counts.values())}, total_rows={len(labels)})")
    assert sum(counts.values()) == len(labels), "buckets must partition all rows exactly once"

    representations = {
        "momentum_scalar": momentum_scalar(closes, MOMENTUM_LOOKBACK),
    }
    vol_ratio, vols = multiscale_vol_summary(closes, VOL_WINDOWS)
    representations["volatility_regime_transition"] = vol_regime_transition(vols[min(VOL_WINDOWS)])

    all_results = []
    for horizon in HORIZONS:
        fwd = forward_return(closes, horizon)
        print(f"\n{'='*100}\nHORIZON = {horizon} bars\n{'='*100}")

        for bucket in ["ALL_POOLED"] + bucket_names:
            mask = np.ones(len(labels), dtype=bool) if bucket == "ALL_POOLED" else (labels == bucket)
            print(f"\n-- bucket: {bucket}  (n_rows={int(mask.sum())}) --")
            print(f"{'representation':32s} {'real_MI':>10s} {'null_mean':>10s} {'null_std':>10s} {'class':>16s}")

            autocorr = bucket_autocorrelation(raw_returns, mask, max_lag=5)
            autocorr_near_zero = all(abs(v) < 0.05 for v in autocorr.values()) if autocorr else True
            print(f"  autocorrelation (lags 1-5): {autocorr}  near_zero={autocorr_near_zero}")

            for name, repr_series in representations.items():
                r = np.asarray(repr_series)[mask]
                y = np.asarray(fwd)[mask]
                valid = np.isfinite(r) & np.isfinite(y)
                n_valid = int(valid.sum())
                if n_valid < 50:
                    print(f"{name:32s} {'skip (n_valid=' + str(n_valid) + ')':>60s}")
                    continue
                stats = mi_with_shuffle_control(r[valid], y[valid])
                cls = classify_cell(stats["real_mi_nats"], stats["null_mi_mean"], stats["null_mi_std"])
                print(f"{name:32s} {stats['real_mi_nats']:10.6f} {stats['null_mi_mean']:10.6f} "
                      f"{stats['null_mi_std']:10.6f} {cls:>16s}")
                all_results.append({
                    "horizon": horizon, "bucket": bucket, "representation": name,
                    "n_valid": n_valid, "real_mi_nats": stats["real_mi_nats"],
                    "null_mi_mean": stats["null_mi_mean"], "null_mi_std": stats["null_mi_std"],
                    "null_mi_max": stats["null_mi_max"], "classification": cls,
                    "autocorr_near_zero": autocorr_near_zero,
                })

    print("\nDone. See docs/superpowers/reports/2026-08-28-goldex-event-time-findings.md "
          "for full interpretation.")
    return all_results


if __name__ == "__main__":
    main()
