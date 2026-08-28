# GOLDEX Genesis Reset — Test 1: Event-Time Information Conditioning

Date: 2026-08-28
Script: `research/genesis_event_time_test.py`
Test file: `tests/research/test_genesis_event_time_test.py`

## QUESTION

All 27 prior hypotheses (Phase 3, 3A, 4, the horizon sweep) pooled every
row in the training partition when computing MI between a representation
and forward returns. Does conditioning on proximity to scheduled macro
announcements (NFP, CPI, FOMC) reveal information that pooling averages
away?

## DATA USED

`data/gold_seed_merged_full6yr.csv`, rows **0:300,000 only** (`TRAINING_ROWS`
convention, identical to every prior phase). The reserved OOS holdout, rows
300,000:400,000, was never read. The training window's actual date range,
checked directly off the CSV, is **2019-12-02 00:00:00 through 2020-09-17
07:59:00** — the event calendar only needed to cover that span.

## TIMEZONE

The `time` column is timestamp-naive. Existing repo convention
(`research/phase5_ev_dataset.py`, comment at lines ~105–108) treats this
column as UTC. This script follows the same convention: all macro-event
release times are specified in US Eastern local time (as officially
published) and converted to UTC via `zoneinfo`'s `America/New_York`, which
correctly handles the one EST→EDT transition (March 2020) inside the
training window.

## EVENT CALENDAR — EXACT VS. APPROXIMATED (full honesty)

- **FOMC rate decisions — EXACT.** Fixed historical public record,
  announcement always 2:00pm ET on the second day of a scheduled two-day
  meeting. 7 meetings fall in the window: 2019-12-11, 2020-01-29,
  2020-03-18, 2020-04-29, 2020-06-10, 2020-07-29, 2020-09-16. The
  unscheduled emergency intermeeting cuts of 2020-03-03 and 2020-03-15 are
  **excluded** — they were not part of a pre-known recurring calendar and
  including them would break the "pick a scheduled calendar without
  looking at outcomes" discipline this test needs.
- **NFP — APPROXIMATED** as "first Friday of the month, 8:30am ET" (10
  events in-window). This is the standard rule of thumb but is not exact:
  BLS occasionally shifts releases around holidays. Example divergence:
  the real January-2020 NFP report (December data) was released Jan 10,
  one week after the mechanical "first Friday" of Jan 3. Some months in
  this calendar are off by up to a week.
- **CPI — APPROXIMATED, weakest part of the calendar.** True BLS CPI
  dates float within roughly the 10th–15th of the month with no fixed
  weekday rule. Approximated here as "second Wednesday of the month,
  8:30am ET" (10 events in-window). Spot-checks against real historical
  CPI dates in this window (e.g. real Jan-2020 CPI was Jan 14, a Tuesday;
  real Mar-2020 CPI was Mar 11, a Wednesday) show 1–3 day and occasional
  weekday misses. Any CPI-driven signal in these results should be read
  as noisy-calendar conditioning, not exact-event conditioning.

Total calendar: 27 events (10 NFP + 10 CPI + 7 FOMC) in-window.

## BUCKET DEFINITIONS (fixed before looking at any result)

Based on generic macro-announcement volatility-decay reasoning (FX/rates
volatility after a scheduled release typically spikes immediately and
decays over 1–4 hours — standard market-microstructure pattern, not tuned
to this dataset):

| Bucket | Definition |
|---|---|
| A. ordinary | more than 4h after the last event AND more than 2h before the next |
| B. pre-event | within 2h before the next event (and not already inside another event's post-window) |
| C. immediate post | 0–60 min after the last event |
| D. later post | 60–240 min (1–4h) after the last event |

Priority order per row: **C > D > B > A**, guaranteeing every row lands in
exactly one bucket (verified in code and in the test file — bucket counts
sum exactly to 300,000).

## REPRESENTATION

`momentum_scalar` and `volatility_regime_transition`, both imported
**unchanged** from `research/phase3a_representation_experiments.py`.
`volatility_regime_transition` is the trend-invariant confound-diagnostic
reference used throughout the Genesis reset.

## TARGET

`forward_return(closes, horizon)` for horizon ∈ {5, 15} bars, reusing
`forward_return` unchanged from `research/genesis_horizon_sweep.py`.

## MODEL/METHOD

None — marginal MI only. `binned_mutual_information` +
`mi_with_shuffle_control` (10-bin quantile MI, 20-permutation shuffled-
label null), imported unchanged from `research/phase3a_representation_experiments.py`,
computed separately inside each bucket and once pooled over all rows as a
sanity check that this script reproduces the known pooled result. Raw-
return autocorrelation (lags 1–5) computed per bucket on the bucket's own
return sub-sequence (see script docstring for the exact construction and
its one caveat re: bucket A's non-contiguity).

## TRAIN/EXPLORATION PERIOD

**Rows 0:300,000 only.** Verified: the script contains no literal
`400000`/`400_000` occurrence (checked by the accompanying test).

## RESULTS

Bucket row counts (sum to 300,000 exactly):

| Bucket | n rows | % of total |
|---|---|---|
| A_ordinary | 290,367 | 96.8% |
| B_pre_event | 3,180 | 1.06% |
| C_immediate_post | 1,593 | 0.53% |
| D_later_post | 4,860 | 1.62% |

Full MI results (real MI clears null if it exceeds `null_mean + 3·null_std`):

**Horizon = 5 bars**

| Bucket | n | Representation | real MI (nats) | null mean | null std | clears null? | MI / regime-MI margin |
|---|---|---|---|---|---|---|---|
| ALL_POOLED | 300,000 | momentum_scalar | 0.117292 | 0.000144 | 0.000020 | yes | 149.6x |
| ALL_POOLED | 300,000 | volatility_regime_transition | 0.000784 | 0.000014 | 0.000008 | yes | — |
| A_ordinary | 290,367 | momentum_scalar | 0.118245 | 0.000140 | 0.000022 | yes | 149.1x |
| A_ordinary | 290,367 | volatility_regime_transition | 0.000793 | 0.000015 | 0.000008 | yes | — |
| B_pre_event | 3,180 | momentum_scalar | 0.058740 | 0.012482 | 0.002036 | yes | 36.3x |
| B_pre_event | 3,180 | volatility_regime_transition | 0.001617 | 0.001408 | 0.000645 | **no** | — |
| C_immediate_post | 1,593 | momentum_scalar | 0.084100 | 0.026300 | 0.004263 | yes | 12.5x |
| C_immediate_post | 1,593 | volatility_regime_transition | 0.006720 | 0.002823 | 0.001116 | yes | — |
| D_later_post | 4,860 | momentum_scalar | 0.123015 | 0.008221 | 0.001577 | yes | 57.8x |
| D_later_post | 4,860 | volatility_regime_transition | 0.002127 | 0.000977 | 0.000418 | **no** | — |

**Horizon = 15 bars**

| Bucket | n | Representation | real MI (nats) | null mean | null std | clears null? | MI / regime-MI margin |
|---|---|---|---|---|---|---|---|
| ALL_POOLED | 300,000 | momentum_scalar | 0.097740 | 0.000130 | 0.000018 | yes | 152.2x |
| ALL_POOLED | 300,000 | volatility_regime_transition | 0.000642 | 0.000015 | 0.000007 | yes | — |
| A_ordinary | 290,367 | momentum_scalar | 0.098515 | 0.000134 | 0.000023 | yes | 155.4x |
| A_ordinary | 290,367 | volatility_regime_transition | 0.000634 | 0.000015 | 0.000006 | yes | — |
| B_pre_event | 3,180 | momentum_scalar | 0.062164 | 0.012281 | 0.001820 | yes | 44.1x |
| B_pre_event | 3,180 | volatility_regime_transition | 0.001409 | 0.001415 | 0.000787 | **no** | — |
| C_immediate_post | 1,593 | momentum_scalar | 0.074150 | 0.025857 | 0.003316 | yes | 13.8x |
| C_immediate_post | 1,593 | volatility_regime_transition | 0.005368 | 0.003006 | 0.001130 | **no** | — |
| D_later_post | 4,860 | momentum_scalar | 0.103395 | 0.008427 | 0.001219 | yes | 55.0x |
| D_later_post | 4,860 | volatility_regime_transition | 0.001880 | 0.000938 | 0.000377 | **no** | — |

Raw-return autocorrelation per bucket (identical for both horizons — computed
on raw 1-bar returns, independent of the forward-return horizon):

| Bucket | lag1 | lag2 | lag3 | lag4 | lag5 | "near zero" (all \|r\|<0.05)? |
|---|---|---|---|---|---|---|
| ALL_POOLED | -0.0390 | -0.0204 | -0.0055 | -0.0082 | -0.0104 | yes |
| A_ordinary | -0.0395 | -0.0222 | -0.0041 | -0.0094 | -0.0129 | yes |
| B_pre_event | **-0.0794** | -0.0083 | -0.0135 | -0.0209 | 0.0283 | **no** |
| C_immediate_post | -0.0152 | -0.0095 | -0.0329 | -0.0045 | 0.0166 | yes |
| D_later_post | -0.0178 | 0.0080 | -0.0236 | 0.0056 | 0.0067 | yes |

## INTERPRETATION AGAINST THE THREE-CRITERIA RULE

Following the horizon sweep's established classification (a cell counts as
genuine structure only if **all three** hold: clears null, MI is not
explained by the trend confound i.e. beats the regime-transition reference
by a wide margin, AND raw-return autocorrelation in that bucket is not
near zero):

- **A_ordinary, ALL_POOLED**: MI clears null with ~150x margin over the
  regime reference, but autocorrelation is near-zero (as in every prior
  phase) — classic **trend-confounded**, matches the entire prior 27-hypothesis
  record exactly. Confirms this script's pipeline reproduces the known
  pooled null.
- **C_immediate_post, D_later_post**: MI clears null with double-digit
  margins over the regime reference at both horizons, but autocorrelation
  is near-zero in both buckets at both horizons — **trend-confounded**,
  same pattern as the pooled baseline, no new information.
- **B_pre_event**: MI clears null with a 36–44x margin over the regime
  reference (which itself does *not* clear its own null in this bucket),
  **and** autocorrelation is not near-zero (lag-1 = -0.079, roughly 4.4
  standard errors from zero given n=3,180). This is the **only** cell that
  satisfies all three criteria mechanically.

## LIMITATIONS (read before trusting the B_pre_event result)

1. **Sample size.** B_pre_event has only 3,180 rows versus 290,367 in
   A_ordinary — roughly 1% of the data. MI and autocorrelation estimates
   here carry far more sampling variance than the pooled/A_ordinary
   numbers; a single-lag autocorrelation crossing ~4 SE with n=3,180 and
   5 lags tested is suggestive but not strong evidence on its own
   (informal multiple-comparisons exposure).
2. **A benign, non-informational explanation is available and was not
   ruled out.** Bid/ask spread widening in the ~2 hours ahead of known
   high-impact scheduled releases is well-documented market
   microstructure behavior (liquidity providers pull in before
   uncertainty resolves). Spread widening mechanically produces exactly
   this signature — elevated momentum-scalar MI plus a negative lag-1
   return autocorrelation from bid/ask bounce — without requiring any
   forward-looking predictive information. This script cannot distinguish
   "genuine predictive information" from "known, benign spread-widening
   microstructure effect," because it never modeled spread at all (the
   `spread` column in the CSV was not used here).
3. **Calendar approximation quality.** CPI dates (10 of the 27 events,
   contributing to the pre-event window) are the weakest link — up to
   several days off from actual BLS release dates in some months. Some
   rows currently labeled B_pre_event for a CPI event may not actually be
   near a real CPI release. NFP dates carry smaller but nonzero slippage.
   FOMC dates are exact.
4. This is a marginal-MI information-content test only, exactly like the
   horizon sweep — no OOS predictive check was run, no trading rule was
   built or implied, and bucket boundaries were fixed before any result
   was examined (as instructed).

## CONCLUSION

**Mostly NULL, with one low-confidence candidate that does not survive
scrutiny as "genuine structure" by this test alone.**

- A_ordinary, C_immediate_post, and D_later_post reproduce the exact same
  trend-confounded pattern seen in all 27 prior hypotheses: large MI vs.
  the naive momentum representation, near-zero autocorrelation, and MI
  fully explained by the pre-existing trend confound (regime-transition
  reference sits at the same order of magnitude once you account for
  bucket size). **No new information in these three buckets.**
- B_pre_event is the one cell that mechanically clears all three
  criteria from the horizon-sweep classification rule. However, given (a)
  the ~40x smaller sample size than the pooled baseline, (b) a fully
  plausible non-informational explanation (bid/ask spread widening ahead
  of scheduled releases) that this script did not rule out, and (c) CPI
  calendar noise contributing to this bucket's composition, **this result
  should not be reported as confirmed genuine predictive structure.** It
  is a weak, mechanistically-explainable candidate that would need a
  dedicated follow-up (e.g. controlling for the `spread` column, using
  the exact FOMC-only subset where the calendar is exact rather than
  approximated, and/or an OOS check) before it could be taken seriously —
  none of which was in scope here. Absent that follow-up, this test's
  honest overall verdict is: **event-time conditioning does not reveal
  confirmed genuine information; it reveals one microstructure-plausible,
  statistically fragile signal (B_pre_event) worth flagging but not
  worth acting on.**
