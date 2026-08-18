"""
Honest out-of-sample evaluation of the full two-stage pipeline: primary OOF
(reused from train.py's purged walk-forward) + a SECOND purged walk-forward
pass to get OOF meta predictions (the saved meta.cbm was fit in-sample on
100% of the meta training set -- fine for the deployed model, but useless
for judging whether the precision filter actually works OOS).

For each meta probability threshold, reports: coverage (trade frequency),
win rate, raw expectancy in R (win_rate*pt_mult - (1-win_rate)*sl_mult), and
expectancy net of the actual observed spread at each event (the honest
"does this really make money" number, not just a directional accuracy stat).

Run: python3 -m core.evaluate [--rows N]
"""
import argparse
import time

import numpy as np
import pandas as pd

from learning.train import (assemble_dataset, label_events, train_primary_oof,
                             build_meta_labels, TB_CFG_TRADE, EMBARGO_BARS, N_SPLITS,
                             _fit_with_early_stopping)
from learning.cv import PurgedWalkForwardCV

THRESHOLDS = [0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60, 0.65]


def oof_meta_predictions(X_meta: pd.DataFrame, y_meta: pd.Series, t0: pd.Series, t1: pd.Series):
    cv = PurgedWalkForwardCV(n_splits=N_SPLITS, embargo_bars=EMBARGO_BARS)
    oof_proba = np.full(len(X_meta), np.nan, dtype=np.float64)
    for fold, (train_pos, test_pos) in enumerate(cv.split(t0.to_numpy(), t1.to_numpy())):
        model = _fit_with_early_stopping(X_meta, y_meta, train_pos)
        oof_proba[test_pos] = model.predict_proba(X_meta.iloc[test_pos])[:, 1]
        print(f"  meta fold {fold}: train={len(train_pos):,} test={len(test_pos):,} "
              f"best_iter={model.get_best_iteration()}")
    return oof_proba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=None)
    args = ap.parse_args()

    t_start = time.time()
    feat, close, high, low, vol, t0_idx, feature_cols = assemble_dataset(rows=args.rows)
    X, y_bin, t0, t1, t0_nz = label_events(close, high, low, vol, t0_idx, feature_cols, feat)

    print("\n== primary OOF (reused methodology from train.py) ==")
    oof_pred, oof_proba_primary, has_oof, fold_metrics = train_primary_oof(X, y_bin, t0, t1)
    print(f"mean primary OOF accuracy: {np.mean([f['acc'] for f in fold_metrics]):.4f}")

    print("\n== building meta labels from primary OOF ==")
    has_oof, side, meta_labels = build_meta_labels(close, high, low, vol, t0_nz, oof_pred, has_oof)

    X_meta = X.loc[has_oof].reset_index(drop=True)
    X_meta["assumed_side"] = side
    y_meta = pd.Series(meta_labels["label"].to_numpy())
    t0_meta = pd.Series(meta_labels.index.to_numpy())
    t1_meta = pd.Series(meta_labels["t1"].to_numpy())

    print("\n== meta OOF (second purged walk-forward pass, honest precision test) ==")
    oof_meta_proba = oof_meta_predictions(X_meta, y_meta, t0_meta, t1_meta)
    valid_meta = np.isfinite(oof_meta_proba)
    print(f"meta OOF coverage: {valid_meta.sum():,}/{len(oof_meta_proba):,} "
          f"(first fold's worth has no OOF meta score, as expected)")

    # observed spread at each meta event, in price units (spread col is
    # broker points; XAUUSD here quotes 2 decimals -> 1 point = 0.01 price)
    spread_pts = feat["spread"].to_numpy()[t0_nz][has_oof]
    spread_price = np.where(np.isfinite(spread_pts), spread_pts, np.nanmedian(spread_pts)) * 0.01
    close_at_meta = close[t0_nz][has_oof]
    sl_dist_price = TB_CFG_TRADE.sl_mult * (vol[t0_nz][has_oof]) * close_at_meta

    print(f"\n{'threshold':>9} {'coverage':>9} {'n_trades':>9} {'win_rate':>9} "
          f"{'raw_R':>8} {'spread_R':>9} {'net_R':>8}")
    for thr in THRESHOLDS:
        m = valid_meta & (oof_meta_proba >= thr)
        if m.sum() < 30:
            print(f"{thr:>9.2f} {'--':>9} {m.sum():>9} (too few OOF trades to report)")
            continue
        win_rate = y_meta.to_numpy()[m].mean()
        raw_R = win_rate * TB_CFG_TRADE.pt_mult - (1 - win_rate) * TB_CFG_TRADE.sl_mult
        # round-trip spread cost expressed in R (fraction of the SL distance,
        # the natural per-trade risk unit)
        spread_R = float(np.mean(spread_price[m] / np.clip(sl_dist_price[m], 1e-9, None)))
        net_R = raw_R - spread_R
        coverage = m.sum() / valid_meta.sum()
        print(f"{thr:>9.2f} {coverage:>9.3f} {m.sum():>9,} {win_rate:>9.4f} "
              f"{raw_R:>8.4f} {spread_R:>9.4f} {net_R:>8.4f}")

    print(f"\ndone in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
