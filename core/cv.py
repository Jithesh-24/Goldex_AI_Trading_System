"""
Purged & embargoed cross-validation (de Prado, ch. 7).

Why this exists: triple-barrier labels have variable, overlapping outcome
windows [t0, t1] — label at bar t depends on price up to t+holding, and
neighboring events' windows overlap almost entirely. Plain chronological
walk-forward is NECESSARY (never train on the future) but NOT SUFFICIENT:
if a training event's window overlaps the test window at all, its label
was influenced by price action inside (or adjacent to) the test period,
which leaks information across the boundary and inflates backtested edge
versus what's achievable live. Two fixes, both implemented here:
  - PURGE: drop training events whose [t0, t1] interval overlaps the test
    interval.
  - EMBARGO: additionally drop training events starting within `embargo_bars`
    bars AFTER the test interval ends, to kill residual serial-correlation
    leakage that purging alone doesn't catch (a training row just after the
    test window can still be autocorrelated with it even with no label
    overlap).

Everything here operates on integer bar-positions (t0, t1 arrays), not
calendar dates, so it works identically on M1 or M5 data.
"""
import numpy as np


def purge_and_embargo_mask(t0: np.ndarray, t1: np.ndarray,
                            test_start: int, test_end: int,
                            embargo_bars: int) -> np.ndarray:
    """Boolean mask over all events (aligned to t0/t1 arrays), True = safe to
    keep in the TRAINING set for a test fold spanning bar positions
    [test_start, test_end] (inclusive, in underlying bar-index units, not
    event-index units)."""
    t0 = np.asarray(t0, dtype=np.int64)
    t1 = np.asarray(t1, dtype=np.int64)
    overlaps = ~((t1 < test_start) | (t0 > test_end))
    embargo_zone = (t0 > test_end) & (t0 <= test_end + embargo_bars)
    return ~(overlaps | embargo_zone)


class PurgedWalkForwardCV:
    """Chronological expanding-window walk-forward with purge + embargo at
    every fold boundary. This is the validation scheme to use for realistic
    OOS performance estimates (what the earlier system called "walk-forward"
    but without the purge/embargo step — this is that, fixed).

    Parameters
    ----------
    n_splits : number of chronological test folds.
    embargo_bars : bars to exclude from training immediately after each test
        fold ends. Pass ~= your max label holding period (de Prado's
        rule of thumb: embargo covers the longest label horizon in the set).
    min_train_bars : refuse to yield a fold if training set would be smaller
        than this (avoids degenerate tiny-train folds at the start).
    """

    def __init__(self, n_splits: int, embargo_bars: int, min_train_bars: int = 10_000):
        self.n_splits = n_splits
        self.embargo_bars = embargo_bars
        self.min_train_bars = min_train_bars

    def split(self, t0: np.ndarray, t1: np.ndarray):
        t0 = np.asarray(t0, dtype=np.int64)
        t1 = np.asarray(t1, dtype=np.int64)
        n = len(t0)
        order = np.argsort(t0, kind="mergesort")
        t0s, t1s = t0[order], t1[order]

        bar_lo, bar_hi = int(t0s[0]), int(t1s.max())
        fold_edges = np.linspace(bar_lo, bar_hi, self.n_splits + 1).astype(np.int64)

        for k in range(self.n_splits):
            test_start, test_end = int(fold_edges[k]), int(fold_edges[k + 1])
            test_mask = (t0s >= test_start) & (t0s <= test_end)
            train_eligible = t0s < test_start  # walk-forward: only the past
            if train_eligible.sum() < self.min_train_bars or test_mask.sum() == 0:
                continue
            keep = purge_and_embargo_mask(t0s, t1s, test_start, test_end, self.embargo_bars)
            train_mask = train_eligible & keep
            train_pos = order[train_mask]
            test_pos = order[test_mask]
            yield train_pos, test_pos


class PurgedKFold:
    """Classic de Prado purged K-fold: k contiguous chronological test blocks
    covering the whole timeline (not just walk-forward-forward), train =
    everything else after purge + embargo. Uses more of the data per fold
    than walk-forward (useful for hyperparameter search where you want every
    event to serve as a test point at least once) but individual folds can
    have training data from AFTER the test block — do not use this for the
    final reported OOS performance number, use PurgedWalkForwardCV for that.
    """

    def __init__(self, n_splits: int, embargo_bars: int):
        self.n_splits = n_splits
        self.embargo_bars = embargo_bars

    def split(self, t0: np.ndarray, t1: np.ndarray):
        t0 = np.asarray(t0, dtype=np.int64)
        t1 = np.asarray(t1, dtype=np.int64)
        order = np.argsort(t0, kind="mergesort")
        t0s, t1s = t0[order], t1[order]
        n = len(t0s)
        fold_bounds = np.array_split(np.arange(n), self.n_splits)

        for fold_positions in fold_bounds:
            if len(fold_positions) == 0:
                continue
            test_start = int(t0s[fold_positions[0]])
            test_end = int(t1s[fold_positions].max())
            keep = purge_and_embargo_mask(t0s, t1s, test_start, test_end, self.embargo_bars)
            keep[fold_positions] = False
            train_pos = order[keep]
            test_pos = order[fold_positions]
            yield train_pos, test_pos
