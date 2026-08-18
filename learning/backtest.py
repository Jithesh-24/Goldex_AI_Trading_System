"""
Sequential-only backtest: enforces "no new signal while a trade is open",
matching the user's real manual workflow (one signal at a time, wait for
close, then look again). core/evaluate.py's threshold sweep is UNCONDITIONAL
-- every event clearing the threshold counts as a trade even if it overlaps
in time with another, which inflates apparent frequency to ~50 signals/day.
This reuses the same honest OOF meta predictions and just adds a greedy
non-overlap filter: sort candidates by entry (t0), walk chronologically,
accept a candidate only if its t0 >= the previous accepted trade's t1.

Run: python3 -m core.backtest [--rows N]
"""
import argparse
import time

import numpy as np
import pandas as pd

from learning.train import assemble_dataset, label_events, train_primary_oof, build_meta_labels, TB_CFG_TRADE
from learning.evaluate import oof_meta_predictions, THRESHOLDS


def greedy_sequential(t0: np.ndarray, t1: np.ndarray, order: np.ndarray) -> np.ndarray:
    """order: indices into t0/t1, already sorted by t0 ascending. Returns
    the subset of `order` accepted under a one-trade-at-a-time constraint."""
    accepted = []
    last_t1 = -1
    for i in order:
        if t0[i] >= last_t1:
            accepted.append(i)
            last_t1 = t1[i]
    return np.array(accepted, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=None)
    args = ap.parse_args()

    t_start = time.time()
    feat, close, high, low, vol, t0_idx, feature_cols = assemble_dataset(rows=args.rows)
    X, y_bin, t0, t1, t0_nz = label_events(close, high, low, vol, t0_idx, feature_cols, feat)

    print("\n== primary OOF ==")
    oof_pred, oof_proba_primary, has_oof, fold_metrics = train_primary_oof(X, y_bin, t0, t1)

    print("\n== meta labels from primary OOF ==")
    has_oof, side, meta_labels = build_meta_labels(close, high, low, vol, t0_nz, oof_pred, has_oof)

    X_meta = X.loc[has_oof].reset_index(drop=True)
    X_meta["assumed_side"] = side
    y_meta = pd.Series(meta_labels["label"].to_numpy())
    t0_meta = pd.Series(meta_labels.index.to_numpy())
    t1_meta = pd.Series(meta_labels["t1"].to_numpy())

    print("\n== meta OOF (second purged walk-forward pass) ==")
    oof_meta_proba = oof_meta_predictions(X_meta, y_meta, t0_meta, t1_meta)
    valid_meta = np.isfinite(oof_meta_proba)

    spread_pts = feat["spread"].to_numpy()[t0_nz][has_oof]
    spread_price = np.where(np.isfinite(spread_pts), spread_pts, np.nanmedian(spread_pts)) * 0.01
    close_at_meta = close[t0_nz][has_oof]
    sl_dist_price = TB_CFG_TRADE.sl_mult * (vol[t0_nz][has_oof]) * close_at_meta

    t0_np = t0_meta.to_numpy()
    t1_np = t1_meta.to_numpy()
    times = pd.to_datetime(feat["time"].to_numpy())

    print(f"\n{'threshold':>9} {'n_trades':>9} {'trades/day':>11} {'win_rate':>9} "
          f"{'raw_R':>8} {'spread_R':>9} {'net_R':>8}")
    for thr in THRESHOLDS:
        m = valid_meta & (oof_meta_proba >= thr)
        cand_idx = np.where(m)[0]
        if len(cand_idx) < 30:
            print(f"{thr:>9.2f} {len(cand_idx):>9} (too few OOF candidates)")
            continue
        order = cand_idx[np.argsort(t0_np[cand_idx])]
        accepted = greedy_sequential(t0_np, t1_np, order)
        if len(accepted) < 30:
            print(f"{thr:>9.2f} {len(accepted):>9} (too few sequential trades)")
            continue

        win_rate = y_meta.to_numpy()[accepted].mean()
        raw_R = win_rate * TB_CFG_TRADE.pt_mult - (1 - win_rate) * TB_CFG_TRADE.sl_mult
        spread_R = float(np.mean(spread_price[accepted] / np.clip(sl_dist_price[accepted], 1e-9, None)))
        net_R = raw_R - spread_R

        first_t, last_t = times[t0_np[accepted[0]]], times[t0_np[accepted[-1]]]
        span_days = max((last_t - first_t).total_seconds() / 86400.0, 1.0)
        trades_per_day = len(accepted) / span_days

        print(f"{thr:>9.2f} {len(accepted):>9,} {trades_per_day:>11.2f} {win_rate:>9.4f} "
              f"{raw_R:>8.4f} {spread_R:>9.4f} {net_R:>8.4f}")

    print(f"\nDONE in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
