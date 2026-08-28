"""research/genesis_event_time_test1b_falsification.py
Genesis reset -- Test 1b: falsification follow-up on the B_pre_event anomaly.

Test 1 (research/genesis_event_time_test.py, see
docs/superpowers/reports/2026-08-28-goldex-event-time-findings.md) found that
of the 4 event-time buckets built over rows 0:300,000 of
data/gold_seed_merged_full6yr.csv, only B_pre_event (within 2h before the
next scheduled macro event, n=3,180, ~1.06% of rows) cleared the shuffled-
label null with momentum_scalar MI ~36-44x the trend-invariant reference
(volatility_regime_transition), AND had return autocorrelation not near
zero (lag-1 ~= -0.079, ~4.4 SE from zero). That result was explicitly NOT
accepted as genuine -- it was flagged as possibly (a) an artifact of the
approximated NFP/CPI event dates rather than the exact FOMC dates, (b) a
bid/ask spread-widening microstructure artifact never checked against the
CSV's `spread` column, or (c) thin-sample noise.

This script is a narrow, strictly-scoped falsification follow-up. It does
NOT build any new architecture, model, or trading rule. It reuses the exact
same event-calendar-construction and MI/null functions as Test 1, unchanged,
and only adds three additional stratifications of B_pre_event:

  1. EXACT-FOMC vs APPROXIMATED-EVENT split: within B_pre_event, split rows
     by whether the *next* event driving their B_pre_event label is an exact
     FOMC date (7 known dates) vs an approximated NFP/CPI date (first-Friday /
     second-Wednesday heuristic). Rerun the same MI/null/autocorrelation
     methodology on each sub-bucket separately.

  2. SPREAD CONTROL: split B_pre_event rows by the CSV's `spread` column
     into spread==20 (the modal/"normal" value) vs spread>20 ("widened").
     Rerun MI/null/autocorrelation on each stratum. If the anomaly is only
     present when spread>20, that supports the spread-widening-artifact
     explanation.

  3. MATCHED-CONTROL comparison: draw a comparison sample from A_ordinary
     matched on `spread` value (and, where the match set is large enough,
     roughly on local realized volatility -- a rolling std of raw 1-bar
     returns over the preceding 30 bars) to have the same distribution as
     B_pre_event, then rerun MI/null/autocorrelation on this matched-ordinary
     sample. This isolates the "does this look like ordinary rows with the
     same spread/volatility profile" question from the "is this literally
     time-near-an-event" question.

DATA USED: data/gold_seed_merged_full6yr.csv rows 0:300,000 only, identical
to Test 1. This script never reads rows 300,000 onward, and no reference to
the reserved-holdout row-count literal appears anywhere in this file --
verified in the accompanying test, matching the discipline of every prior
phase.

IMPORTANT PRE-REGISTERED OBSERVATION (found before writing any MI code,
purely by inspecting the `spread` column): within rows 0:300,000, `spread`
is EXACTLY 20.0 for every single row (checked directly against the CSV --
this training window predates essentially all of the spread variation that
exists later in the full 6-year file, where spread==20 in only ~98.9% of
rows overall). This means the "spread>20" stratum of B_pre_event is EMPTY
in this training window, and the spread-matched comparison sample from
A_ordinary is--trivially--matched on spread (since every row has the same
value). The MI/null/autocorrelation split described in item 2 above is run
exactly as specified, but its result is degenerate by construction: there
is no widened-spread stratum to compare against inside this window. This is
reported honestly rather than silently skipped, because it is itself
informative -- it rules out "logged spread widening, as recorded by this
broker feed, in this window" as an explanation, without ruling out
unlogged/real bid-ask-widening that this particular feed's spread field
does not capture.

REPRESENTATION / TARGET / METHOD: identical to Test 1 -- momentum_scalar and
volatility_regime_transition (imported unchanged from
research/phase3a_representation_experiments.py), forward_return for horizon
in (5, 15) (imported unchanged from research/genesis_horizon_sweep.py),
binned_mutual_information / mi_with_shuffle_control (imported unchanged from
research/phase3a_representation_experiments.py, 10-bin quantile MI with a
20-permutation shuffled-label null), and bucket_autocorrelation (imported
unchanged from research/genesis_event_time_test.py).

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 research/genesis_event_time_test1b_falsification.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.phase3a_representation_experiments import (
    TRAINING_ROWS, DATA_PATH,
    binned_mutual_information, mi_with_shuffle_control,
    momentum_scalar, multiscale_vol_summary, vol_regime_transition,
    MOMENTUM_LOOKBACK, VOL_WINDOWS,
)
from research.genesis_horizon_sweep import forward_return
from research.genesis_event_time_test import (
    build_event_calendar,
    label_event_buckets,
    bucket_autocorrelation,
    classify_cell,
    load_training_frame,
    CONFOUND_CLEAR_SIGMAS,
)

HORIZONS = (5, 15)
VOL_MATCH_WINDOW = 30  # bars, rolling std of raw returns for local-vol matching


def next_event_type_per_row(times: pd.Series, calendar: pd.DataFrame) -> np.ndarray:
    """For each row, returns the event_type ('FOMC', 'NFP', or 'CPI') of the
    *next* upcoming event in the sorted calendar, or '' if there is no next
    event (row after the last calendar event). This mirrors the idx_next
    computation inside label_event_buckets exactly, so that "which event is
    driving this row's B_pre_event label" can be recovered without
    duplicating the bucket-assignment logic itself."""
    times_arr = pd.to_datetime(times).to_numpy()
    cal_sorted = calendar.sort_values("event_time_utc").reset_index(drop=True)
    ev_times = pd.to_datetime(cal_sorted["event_time_utc"]).to_numpy()
    ev_types = cal_sorted["event_type"].to_numpy()
    idx_next = np.searchsorted(ev_times, times_arr, side="left")
    n = len(times_arr)
    out = np.full(n, "", dtype=object)
    has_next = idx_next < len(ev_times)
    out[has_next] = ev_types[idx_next[has_next]]
    return out


def rolling_std_returns(returns: np.ndarray, window: int) -> np.ndarray:
    """Simple trailing rolling std of raw returns (uses only past data:
    row i's value is std of returns[i-window+1 : i+1])."""
    s = pd.Series(returns)
    return s.rolling(window=window, min_periods=window).std().to_numpy()


def matched_ordinary_sample(
    spread_all: np.ndarray,
    local_vol_all: np.ndarray,
    ordinary_mask: np.ndarray,
    target_spread: np.ndarray,
    target_local_vol: np.ndarray,
    n_target: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Builds a boolean mask over the full row range selecting n_target rows
    from A_ordinary, matched on spread value exactly (simple equality filter,
    trivial here since spread is constant in this window -- see module
    docstring) and, within that, matched on local realized volatility by
    simple decile-binning: bin both the target's and A_ordinary's local-vol
    values into 10 quantile bins (computed off the pooled A_ordinary
    distribution), then sample from each bin in proportion to how many
    target rows fall in that bin. No nearest-neighbor algorithm -- deliberately
    simple per task scope."""
    candidate_idx = np.where(ordinary_mask)[0]
    cand_spread = spread_all[candidate_idx]
    cand_vol = local_vol_all[candidate_idx]

    # Spread match: keep only candidates whose spread value appears in the
    # target's spread distribution (exact-value match; degenerate to "all"
    # here since every value is 20.0, but written generally).
    target_spread_values = set(np.unique(target_spread[np.isfinite(target_spread)]))
    spread_ok = np.array([v in target_spread_values for v in cand_spread])

    # Local-vol match: quantile-bin both target and eligible candidates using
    # bin edges derived from the eligible-candidate pool, then sample
    # candidates from each bin proportional to the target's occupancy of
    # that bin.
    valid_cand = spread_ok & np.isfinite(cand_vol)
    valid_target = np.isfinite(target_local_vol)
    if valid_cand.sum() < 50 or valid_target.sum() < 50:
        # not enough data to vol-match; fall back to spread-only random sample
        pool = candidate_idx[spread_ok]
        chosen = rng.choice(pool, size=min(n_target, len(pool)), replace=False)
        out = np.zeros(len(spread_all), dtype=bool)
        out[chosen] = True
        return out

    edges = np.unique(np.quantile(cand_vol[valid_cand], np.linspace(0, 1, 11)))
    target_bins = np.clip(np.digitize(target_local_vol[valid_target], edges[1:-1]), 0, len(edges) - 2)
    cand_bins = np.clip(np.digitize(cand_vol[valid_cand], edges[1:-1]), 0, len(edges) - 2)

    cand_pool_idx = candidate_idx[valid_cand]
    chosen_all = []
    n_bins = len(edges) - 1
    for b in range(n_bins):
        n_needed = int((target_bins == b).sum())
        if n_needed == 0:
            continue
        bin_pool = cand_pool_idx[cand_bins == b]
        if len(bin_pool) == 0:
            continue
        n_take = min(n_needed, len(bin_pool))
        chosen_all.append(rng.choice(bin_pool, size=n_take, replace=False))
    chosen = np.concatenate(chosen_all) if chosen_all else np.array([], dtype=int)
    out = np.zeros(len(spread_all), dtype=bool)
    out[chosen] = True
    return out


def run_cell(name, mask, representations, fwd, raw_returns, results, horizon):
    n = int(mask.sum())
    print(f"\n-- {name}  (n_rows={n}) --")
    if n < 50:
        print("  skipped: n_rows < 50")
        results.append({"horizon": horizon, "cell": name, "n_valid": n, "skipped": True})
        return
    autocorr = bucket_autocorrelation(raw_returns, mask, max_lag=5)
    autocorr_near_zero = all(abs(v) < 0.05 for v in autocorr.values()) if autocorr else True
    print(f"  autocorrelation (lags 1-5): {autocorr}  near_zero={autocorr_near_zero}")
    print(f"  {'representation':32s} {'real_MI':>10s} {'null_mean':>10s} {'null_std':>10s} {'class':>16s}")
    for rep_name, rep_series in representations.items():
        r = np.asarray(rep_series)[mask]
        y = np.asarray(fwd)[mask]
        valid = np.isfinite(r) & np.isfinite(y)
        n_valid = int(valid.sum())
        if n_valid < 50:
            print(f"  {rep_name:32s} {'skip (n_valid=' + str(n_valid) + ')':>60s}")
            continue
        stats = mi_with_shuffle_control(r[valid], y[valid])
        cls = classify_cell(stats["real_mi_nats"], stats["null_mi_mean"], stats["null_mi_std"])
        print(f"  {rep_name:32s} {stats['real_mi_nats']:10.6f} {stats['null_mi_mean']:10.6f} "
              f"{stats['null_mi_std']:10.6f} {cls:>16s}")
        results.append({
            "horizon": horizon, "cell": name, "representation": rep_name,
            "n_valid": n_valid, "real_mi_nats": stats["real_mi_nats"],
            "null_mi_mean": stats["null_mi_mean"], "null_mi_std": stats["null_mi_std"],
            "null_mi_max": stats["null_mi_max"], "classification": cls,
            "autocorr": autocorr, "autocorr_near_zero": autocorr_near_zero,
        })


def main():
    df = load_training_frame()
    closes = df["close"].to_numpy(dtype=np.float64)
    times = df["time"]
    spread = df["spread"].to_numpy(dtype=np.float64)
    raw_returns = np.full(len(closes), np.nan)
    raw_returns[1:] = closes[1:] - closes[:-1]
    local_vol = rolling_std_returns(raw_returns, VOL_MATCH_WINDOW)

    start_date = times.iloc[0].date()
    end_date = times.iloc[-1].date()
    print(f"Training window date range: {start_date} .. {end_date} ({len(df)} rows)")

    print(f"\nspread column check (training window only): "
          f"unique values = {sorted(np.unique(spread).tolist())}, "
          f"frac==20.0 = {(spread == 20.0).mean():.6f}")

    calendar = build_event_calendar(start_date, end_date)
    labels = label_event_buckets(times, calendar["event_time_utc"])
    next_type = next_event_type_per_row(times, calendar)

    b_mask = labels == "B_pre_event"
    a_mask = labels == "A_ordinary"
    n_b_total = int(b_mask.sum())
    print(f"\nB_pre_event total: n={n_b_total}")

    is_fomc = b_mask & (next_type == "FOMC")
    is_approx = b_mask & np.isin(next_type, ["NFP", "CPI"])
    print(f"  B_pre_event driven by exact FOMC event: n={int(is_fomc.sum())}")
    print(f"  B_pre_event driven by approximated NFP/CPI event: n={int(is_approx.sum())}")
    assert int(is_fomc.sum()) + int(is_approx.sum()) == n_b_total, \
        "every B_pre_event row must be attributed to exactly one next-event type"

    spread20 = b_mask & (spread == 20.0)
    spread_gt20 = b_mask & (spread > 20.0)
    print(f"  B_pre_event with spread==20: n={int(spread20.sum())}")
    print(f"  B_pre_event with spread>20: n={int(spread_gt20.sum())}")
    assert int(spread20.sum()) + int(spread_gt20.sum()) == n_b_total

    representations = {
        "momentum_scalar": momentum_scalar(closes, MOMENTUM_LOOKBACK),
    }
    vol_ratio, vols = multiscale_vol_summary(closes, VOL_WINDOWS)
    representations["volatility_regime_transition"] = vol_regime_transition(vols[min(VOL_WINDOWS)])

    rng = np.random.default_rng(42)
    matched_mask = matched_ordinary_sample(
        spread, local_vol, a_mask,
        target_spread=spread[b_mask], target_local_vol=local_vol[b_mask],
        n_target=n_b_total, rng=rng,
    )
    print(f"\nMatched-control sample drawn from A_ordinary: n={int(matched_mask.sum())} "
          f"(target n from B_pre_event = {n_b_total})")

    all_results = []
    for horizon in HORIZONS:
        fwd = forward_return(closes, horizon)
        print(f"\n{'='*100}\nHORIZON = {horizon} bars\n{'='*100}")

        run_cell("B_pre_event (full, reproduces Test 1)", b_mask, representations, fwd, raw_returns, all_results, horizon)
        run_cell("B_pre_event x exact_FOMC", is_fomc, representations, fwd, raw_returns, all_results, horizon)
        run_cell("B_pre_event x approximated_NFP_CPI", is_approx, representations, fwd, raw_returns, all_results, horizon)
        run_cell("B_pre_event x spread==20", spread20, representations, fwd, raw_returns, all_results, horizon)
        run_cell("B_pre_event x spread>20", spread_gt20, representations, fwd, raw_returns, all_results, horizon)
        run_cell("A_ordinary matched-control (spread+local-vol matched to B_pre_event)", matched_mask, representations, fwd, raw_returns, all_results, horizon)

    print("\nDone. See docs/superpowers/reports/2026-08-28-goldex-event-time-test1b-falsification.md "
          "for full interpretation and A/B/C classification.")
    return all_results


if __name__ == "__main__":
    main()
