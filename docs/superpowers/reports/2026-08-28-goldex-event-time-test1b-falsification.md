# GOLDEX Genesis Reset — Test 1b: Falsification Follow-Up on B_pre_event

Date: 2026-08-28
Script: `research/genesis_event_time_test1b_falsification.py`
Test file: `tests/research/test_genesis_event_time_test1b.py`
Follows: `research/genesis_event_time_test.py` /
`docs/superpowers/reports/2026-08-28-goldex-event-time-findings.md`

## QUESTION

Test 1 found that B_pre_event (rows within 2h of a scheduled macro event,
n=3,180, ~1.06% of rows 0:300,000) was the only event-time bucket that
mechanically cleared all three of the horizon-sweep's "genuine structure"
criteria: MI clears the shuffled-label null, MI beats the trend-invariant
`volatility_regime_transition` reference by a wide margin (which itself does
not clear its own null in this bucket), and raw-return autocorrelation is
not near zero (lag-1 ≈ -0.079, ~4.4 SE from zero at n=3,180). Test 1 flagged
three unresolved alternative explanations and explicitly did NOT accept the
result as genuine: (a) it might be driven only by the noisy approximated
NFP/CPI calendar rather than the exact FOMC dates, (b) it might be a
bid/ask spread-widening microstructure artifact, never checked against the
CSV's `spread` column, and (c) it might be thin-sample noise.

This follow-up runs three narrow, pre-specified control analyses to test
those three explanations, using the same MI/null/autocorrelation
methodology, unchanged, on sub-strata of B_pre_event.

## DATA USED

`data/gold_seed_merged_full6yr.csv`, rows **0:300,000 only**, identical to
Test 1. The reserved OOS holdout was never read (verified: no reference to
the reserved-holdout row-count literal appears in the new script).

## PRE-ANALYSIS OBSERVATION: `spread` HAS ZERO VARIANCE IN THIS WINDOW

Before running any MI code, the `spread` column was inspected directly.
Within rows 0:300,000, **`spread` is exactly 20.0 for every single row** —
no exceptions. (Over the *entire* 6-year file, spread==20.0 only 98.9% of
the time and takes on values up to 29+ in the remainder; that variation
exists almost entirely later in the file, outside this training window.)

This has one direct consequence for the planned spread-control test: the
"spread > 20 (widened)" stratum of B_pre_event is **empty** (n=0) in this
window. The spread-based falsification test specified in the task cannot
be run in the form intended, because there is no widened-spread comparison
group to test against inside rows 0:300,000. This is reported honestly as
a limitation, not silently worked around. It also means the spread-matched
comparison sample from A_ordinary is trivially spread-matched (every row —
target and candidate pool alike — has spread==20.0).

One thing this observation does *rule out*: "logged bid/ask spread
widening, as this specific broker feed's `spread` field records it, during
this window" cannot be the mechanism, because the field never moves in this
window. It does **not** rule out unlogged/real market-microstructure
widening that this feed's spread column simply doesn't capture in this
period (this feed may have used a fixed nominal spread for retail/demo
pricing purposes during this stretch — a data-provenance question outside
this test's scope).

## METHOD (unchanged from Test 1)

`binned_mutual_information` / `mi_with_shuffle_control` (10-bin quantile MI,
20-permutation shuffled-label null) and `bucket_autocorrelation`, all
imported unchanged from `research/phase3a_representation_experiments.py`
and `research/genesis_event_time_test.py`. Representations: `momentum_scalar`
and `volatility_regime_transition` (trend-invariant reference), both
unchanged. Target: `forward_return` for horizon ∈ {5, 15} bars, unchanged.

## ANALYSIS 1 — EXACT FOMC vs APPROXIMATED NFP/CPI SPLIT

Within B_pre_event (n=3,180), each row's "next event" (the event driving
its B_pre_event label) was classified as exact-FOMC or approximated
NFP/CPI.

| Sub-bucket | n |
|---|---|
| exact_FOMC | 780 |
| approximated_NFP_CPI | 2,400 |
| **sum** | **3,180** (matches B_pre_event exactly) |

**Horizon = 5 bars**

| Cell | n | real MI (momentum) | null mean | null std | SE above null | vol_regime_transition clears null? | autocorr lag1 | near-zero? |
|---|---|---|---|---|---|---|---|---|
| B_pre_event (full) | 3,180 | 0.058740 | 0.012482 | 0.002036 | 22.7 | no | -0.0794 | no |
| exact_FOMC | 780 | 0.141097 | 0.054660 | 0.007366 | 11.7 | no | -0.0841 | no |
| approximated_NFP_CPI | 2,400 | 0.051928 | 0.017325 | 0.002492 | 13.9 | no | -0.0743 | no |

**Horizon = 15 bars**

| Cell | n | real MI (momentum) | null mean | null std | SE above null | vol_regime_transition clears null? | autocorr lag1 | near-zero? |
|---|---|---|---|---|---|---|---|---|
| B_pre_event (full) | 3,180 | 0.062164 | 0.012281 | 0.001820 | 27.4 | no | -0.0794 | no |
| exact_FOMC | 780 | 0.114317 | 0.054410 | 0.008042 | 7.4 | no | -0.0841 | no |
| approximated_NFP_CPI | 2,400 | 0.061795 | 0.016341 | 0.002455 | 18.5 | no | -0.0743 | no |

**Finding:** the effect does NOT disappear when isolated to the exact-FOMC
subset. If anything, the exact-FOMC subset (n=780) shows a *larger* raw
MI-over-null margin at horizon 5 and comparable or larger raw lag-1
autocorrelation magnitude than the full bucket. The approximated NFP/CPI
subset (n=2,400) independently reproduces essentially the same
autocorrelation signature (lag-1 ≈ -0.074) and also clears its null by a
wide margin. Both sub-buckets individually satisfy the same three-criteria
pattern as the full B_pre_event bucket (MI clears null, regime-transition
reference does not clear its own null, autocorrelation not near zero).
This is evidence AGAINST the "it's purely a CPI/NFP calendar-noise
artifact" explanation — the exact-dated FOMC subset shows the same pattern,
if not a stronger one, in its own right (n=780 is admittedly a further
~4x-smaller sample than the full bucket, so its own sampling variance is
correspondingly larger — see Caveats).

## ANALYSIS 2 — SPREAD CONTROL

| Cell | n |
|---|---|
| spread == 20 | 3,180 (100% of B_pre_event) |
| spread > 20 | 0 |

As documented above, this stratification is degenerate in this window: all
3,180 B_pre_event rows have spread==20.0, so "spread==20" numerically
reproduces the full-bucket result exactly (identical MI/null/autocorrelation
numbers to the "B_pre_event (full)" row above), and "spread>20" has zero
rows and cannot be evaluated (skipped: n<50).

**Finding:** the intended spread-widening falsification test could not be
run as specified in this training window, because the window contains no
observed spread variation at all. This piece of the investigation is
**inconclusive by data limitation**, not resolved in either direction — it
neither confirms nor rules out spread-widening as an explanation via this
column directly.

## ANALYSIS 3 — MATCHED-CONTROL COMPARISON (A_ordinary, spread + local-vol matched)

A comparison sample of n=3,180 rows was drawn from A_ordinary (n=290,367),
matched exactly on `spread` value (trivial here, since spread==20.0
everywhere) and matched by decile-binned trailing 30-bar realized
volatility (rolling std of raw 1-bar returns) to reproduce B_pre_event's
local-volatility distribution, using simple quantile-bin proportional
sampling (no nearest-neighbor algorithm, per task scope).

**Horizon = 5 bars**

| Cell | n | real MI (momentum) | null mean | null std | vol_regime_transition clears null? | autocorr lag1 | near-zero? |
|---|---|---|---|---|---|---|---|
| B_pre_event (full) | 3,180 | 0.058740 | 0.012482 | 0.002036 | no | -0.0794 | **no** |
| A_ordinary matched-control | 3,180 | 0.057184 | 0.012487 | 0.002190 | no | +0.0086 | **yes** |

**Horizon = 15 bars**

| Cell | n | real MI (momentum) | null mean | null std | vol_regime_transition clears null? | autocorr lag1 | near-zero? |
|---|---|---|---|---|---|---|---|
| B_pre_event (full) | 3,180 | 0.062164 | 0.012281 | 0.001820 | no | -0.0794 | **no** |
| A_ordinary matched-control | 3,180 | 0.059603 | 0.012401 | 0.001932 | no | +0.0086 | **yes** |

**Finding:** this is the most informative result in this follow-up. The
matched-control sample from A_ordinary — same spread, same local
realized-volatility distribution as B_pre_event, same sample size — shows
essentially the same momentum-scalar MI-vs-null clearance (both cells
clear the null by a similar order of magnitude, and `volatility_regime_transition`
fails to clear its own null in both — so criteria 1 and 2 alone do NOT
distinguish B_pre_event from a volatility-matched ordinary sample). **But
the autocorrelation signature does not appear in the matched control**
(lag-1 ≈ +0.009, near-zero, vs. -0.079 in B_pre_event, not near-zero). This
means the negative lag-1 autocorrelation is not merely an artifact of local
volatility level (which was explicitly matched away here) — something
about actual temporal proximity to a scheduled event, beyond what
matching on spread and realized volatility captures, is associated with the
autocorrelation anomaly.

## SUMMARY TABLE (lag-1 autocorrelation, the discriminating statistic)

| Cell | n | lag-1 autocorr | near-zero (\|r\|<0.05)? |
|---|---|---|---|
| B_pre_event (full, from Test 1) | 3,180 | -0.0794 | no |
| B_pre_event × exact FOMC | 780 | -0.0841 | no |
| B_pre_event × approximated NFP/CPI | 2,400 | -0.0743 | no |
| B_pre_event × spread==20 | 3,180 | -0.0794 | no (identical to full bucket — degenerate stratification) |
| B_pre_event × spread>20 | 0 | n/a | untestable, no data |
| A_ordinary, matched on spread + local vol | 3,180 | +0.0086 | **yes** |

(Lag-1 autocorrelation is identical across the horizon-5 and horizon-15
rows within a cell because it is computed on raw 1-bar returns,
independent of the forward-return horizon — same convention as Test 1.)

## DECISION RULE APPLIED

Per the pre-specified rule:
- **A — FALSIFIED** would require the effect to disappear after exact-FOMC
  isolation and/or spread control. It does not disappear after exact-FOMC
  isolation (if anything it is comparably or more pronounced in the n=780
  FOMC-only subset), and the spread control could not be run at all in this
  window (no widened-spread rows exist), so it cannot be said to have
  "disappeared" there either — there was simply no test to fail.
- **B — SURVIVES** requires the effect to remain materially above null and
  the trend-invariant reference after these controls, in a subset with
  adequate sample size. Both exact-FOMC (n=780) and approximated-NFP/CPI
  (n=2,400) subsets independently reproduce the full bucket's
  three-criteria pattern (MI clears null, regime-transition reference does
  not, autocorrelation not near zero), and the spread/local-vol-matched
  ordinary control shows the MI clearance alone is NOT sufficient to
  reproduce the anomaly — only cells at actual event-time proximity show
  the non-zero autocorrelation.
- **C — INCONCLUSIVE** applies squarely to the spread-widening question in
  isolation: this window's `spread` column has no variance, so that
  specific alternative explanation is neither confirmed nor ruled out by
  this data.

**Classification: B — ANOMALY SURVIVES**, with one explicit carve-out.

The effect (elevated momentum-scalar MI beyond what the trend-invariant
reference explains, combined with non-trivial negative lag-1 return
autocorrelation) is present in both the exact-FOMC and the
approximated-event sub-populations of B_pre_event at adequate sample sizes
(n=780 and n=2,400 respectively — both comfortably above the ~50-row floor
used throughout this line of work, though both smaller than the full
3,180-row bucket and therefore carrying more sampling variance than the
pooled numbers), and is absent from a spread- and local-volatility-matched
sample of otherwise-ordinary rows of the same size. This directly narrows
two of the three explanations Test 1 left open: it is not purely a CPI/NFP
calendar-noise artifact (the exact-FOMC-only subset shows it too, and more
strongly by MI margin), and it is not purely a local-realized-volatility
level artifact (the matched control removes exactly this and the
autocorrelation vanishes). The third explanation — logged spread widening —
could not be tested at all in this specific 300,000-row window because the
window contains no spread variation; this is preserved as an open,
unresolved caveat rather than treated as ruled out or ruled in.

## CAVEATS (read before treating "B" as strong)

1. **Sample sizes, while adequate to compute the statistics, are still
   small in absolute terms.** n=780 (exact-FOMC) and n=2,400
   (approximated NFP/CPI) are 4x and 1.3x smaller than the already-small
   original B_pre_event (n=3,180, itself ~1% of the 300,000-row training
   window). A single-lag autocorrelation test at these sample sizes, with
   multiple lags and multiple sub-buckets examined informally across this
   and the prior report, carries real multiple-comparisons exposure that
   was not formally corrected for.
2. **The spread-widening explanation is not resolved, only untestable here.**
   The complete absence of spread variation in this window is itself an
   interesting data-provenance fact (possibly reflecting a fixed nominal
   spread configuration for this particular broker/feed/period) that this
   follow-up did not investigate further, since doing so would go beyond
   the strict scope given.
3. **This remains a marginal-MI / autocorrelation information-content
   check only.** No OOS predictive test was run, no trading rule was
   built or implied, and no model of any kind was fit. "Survives" here
   means "survives these two specific falsification attempts," not
   "validated as tradeable structure."
4. **The matched-control method is deliberately simple** (exact-value
   spread filter plus decile-bin proportional sampling on a single
   rolling-volatility feature), not a rigorous causal-matching procedure.
   It is adequate to show the autocorrelation anomaly is not explained by
   this one volatility proxy, but does not rule out other unmeasured
   confounds correlated with event-time proximity.
5. Calendar-approximation caveats from Test 1 (CPI dates ±1-3 days, NFP
   dates occasional multi-day slippage) still apply to the
   approximated_NFP_CPI subset's internal composition.

## RECOMMENDATION ON PROCEEDING TO TEST 2

Per the classification rule given for this follow-up: **because this
result classifies as B (anomaly survives these controls), this report does
NOT recommend proceeding to Test 2 (the bounded tick pilot) on its own
authority.** B_pre_event, and specifically its exact-FOMC and
approximated-event sub-populations, should be flagged as a **preserved
candidate signal** for the project record — narrowed but not fully
resolved by this follow-up, with the spread-widening alternative explicitly
left open due to a data limitation rather than genuinely tested. **The user
should be consulted before any further investigation, tooling, or testing
is built around this candidate.** No such further work has been started
here.

## FILES TOUCHED (scope discipline)

- New: `research/genesis_event_time_test1b_falsification.py`
- New: `tests/research/test_genesis_event_time_test1b.py`
- New: `docs/superpowers/reports/2026-08-28-goldex-event-time-test1b-falsification.md`
- `research/genesis_event_time_test.py` was NOT modified (only imported from).
- Rows 300,000:400,000 of the CSV were never read; no reference to the
  reserved-holdout row-count literal appears in the new script (verified by
  `tests/research/test_genesis_event_time_test1b.py::test_script_never_references_the_reserved_holdout_row_count`).
