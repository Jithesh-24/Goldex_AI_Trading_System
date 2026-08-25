"""
PHASE 1A -- Empirical edge audit. Research-only: does NOT touch production
models, does NOT retrain deployed models, does NOT change core/train.py's
defaults. Reuses the exact same feature set (28 cols), CUSUM event gate,
triple-barrier config, and purged/embargoed walk-forward CV the deployed
system uses -- this script only adds extra instrumentation (fold-id
tracking, importances, ablations, regime slicing, MAE/MFE, bootstrap CIs)
around that same methodology to answer: does the current architecture have
a robust, calibrated, cost-resilient OOS edge.

Run (background, this takes a while -- ~70 CatBoost OOF fits total):
  /home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.audit_edge [--rows N] [--quick]
Output persisted to research/output/*.json, *.csv, and a text summary.
"""
import argparse
import json
import os
import time

import numba
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from learning.data import load_raw_m1
from features.features import build_features
from features.labeling import TripleBarrierConfig, cusum_filter, triple_barrier_labels
from learning.cv import PurgedWalkForwardCV
from learning.train import (TB_CFG_DIR, TB_CFG_TRADE, HORIZON_VOL_SCALE, CUSUM_K,
                             N_SPLITS, EMBARGO_BARS, CATBOOST_KW, VAL_FRACTION,
                             META_PROB_THRESHOLD)
from learning.backtest import greedy_sequential

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "research", "output")
os.makedirs(OUT, exist_ok=True)

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

FAMILIES = {
    "base_return": ["ret_1", "sign_ret_1", "ret_5", "sign_ret_5", "ret_15",
                     "sign_ret_15", "ret_60", "sign_ret_60"],
    "volatility": ["ewma_vol", "gk_vol_20", "rs_vol_20", "yz_vol_20",
                    "gk_vol_60", "rs_vol_60", "yz_vol_60",
                    "gk_vol_240", "rs_vol_240", "yz_vol_240"],
    "jump": ["bipower_var_60", "jump_component_60"],
    "kalman": ["kalman_level_dist", "kalman_velocity", "kalman_residual_z"],
    "hurst_fracdiff": ["hurst_120", "hurst_480", "fracdiff_log_price"],
    "spread": ["spread"],
    "tick_volume": ["tick_volume"],
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _fit(X, y, train_pos):
    cut = int(len(train_pos) * (1 - VAL_FRACTION))
    tr, va = train_pos[:cut], train_pos[cut:]
    model = CatBoostClassifier(**CATBOOST_KW)
    model.fit(X.iloc[tr], y.iloc[tr], eval_set=(X.iloc[va], y.iloc[va]))
    return model


def manual_log_loss(y_true, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    phat = k / n
    denom = 1 + z ** 2 / n
    center = phat + z ** 2 / (2 * n)
    half = z * np.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2))
    return ((center - half) / denom, (center + half) / denom)


def block_bootstrap(values, block_size=20, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n < block_size * 2:
        return (np.nan, np.nan, np.nan)
    n_blocks = int(np.ceil(n / block_size))
    starts_pool = np.arange(0, n - block_size + 1)
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.choice(starts_pool, size=n_blocks, replace=True)
        sample = np.concatenate([values[s:s + block_size] for s in starts])[:n]
        means[b] = sample.mean()
    return (float(np.percentile(means, 2.5)), float(np.mean(means)), float(np.percentile(means, 97.5)))


def oof_run(X, y_bin, t0, t1, tag, want_importance=True):
    """Purged walk-forward OOF with fold_id tracking + per-fold importances."""
    cv = PurgedWalkForwardCV(n_splits=N_SPLITS, embargo_bars=EMBARGO_BARS)
    n = len(X)
    oof_pred = np.full(n, -1, dtype=np.int64)
    oof_proba = np.full(n, np.nan, dtype=np.float64)
    fold_id = np.full(n, -1, dtype=np.int64)
    fold_metrics = []
    importances = []
    for fold, (train_pos, test_pos) in enumerate(cv.split(t0.to_numpy(), t1.to_numpy())):
        model = _fit(X, y_bin, train_pos)
        proba = model.predict_proba(X.iloc[test_pos])[:, 1]
        pred = (proba >= 0.5).astype(np.int64)
        oof_pred[test_pos] = pred
        oof_proba[test_pos] = proba
        fold_id[test_pos] = fold
        y_true = y_bin.iloc[test_pos].to_numpy()
        acc = float((pred == y_true).mean())
        rec_pos = float((pred[y_true == 1] == 1).mean()) if (y_true == 1).any() else float("nan")
        rec_neg = float((pred[y_true == 0] == 0).mean()) if (y_true == 0).any() else float("nan")
        bal_acc = float(np.nanmean([rec_pos, rec_neg]))
        ll = manual_log_loss(y_true, proba)
        fold_metrics.append({"fold": fold, "n_train": len(train_pos), "n_test": len(test_pos),
                              "acc": acc, "bal_acc": bal_acc, "logloss": ll,
                              "best_iter": model.get_best_iteration()})
        if want_importance:
            importances.append(dict(zip(X.columns, [float(v) for v in model.get_feature_importance()])))
        log(f"  [{tag}] fold {fold}: train={len(train_pos):,} test={len(test_pos):,} "
            f"acc={acc:.4f} bal_acc={bal_acc:.4f} logloss={ll:.4f} best_iter={model.get_best_iteration()}")
    has_oof = oof_pred >= 0
    return {"oof_pred": oof_pred, "oof_proba": oof_proba, "fold_id": fold_id,
            "has_oof": has_oof, "fold_metrics": fold_metrics, "importances": importances}


def build_meta(close, high, low, vol, t0_nz, side, has_oof):
    """Builds the meta-labeling target for a caller-supplied side.
    `side` must already be a signed +-1.0 array aligned to t0_nz's full
    index space (e.g. research.direction_side.compute_direction_oof's
    `side` output) -- this function does NOT derive a side from a raw
    classifier prediction anymore (Phase 5A: every downstream specialist
    conditions on Direction's side, never its own)."""
    side_at_oof = side[has_oof]
    assert np.all(np.isin(side_at_oof, (-1.0, 1.0))), (
        "build_meta's `side` argument must already be a signed +-1.0 array "
        "(e.g. research.direction_side.compute_direction_oof's `side` output, "
        "or np.where(oof_pred == 1, 1.0, -1.0)) -- NOT a raw 0/1 classifier "
        "prediction. Passing raw oof_pred silently mislabels every short event."
    )
    side_sub = side_at_oof
    t0_sub = t0_nz[has_oof]
    meta_labels = triple_barrier_labels(close, high, low, t0_sub, vol, TB_CFG_TRADE, side=side_sub)
    return side_sub, meta_labels


@numba.njit(cache=True)
def _mae_mfe_core(close, high, low, t0_idx, t1_idx, side, vol_at_t0):
    n = len(t0_idx)
    mae = np.empty(n, dtype=np.float64)
    mfe = np.empty(n, dtype=np.float64)
    for e in range(n):
        t0 = t0_idx[e]
        t1 = t1_idx[e]
        s = side[e]
        p0 = close[t0]
        worst = 0.0
        best = 0.0
        for j in range(t0 + 1, t1 + 1):
            if s >= 0:
                fav = (high[j] - p0) / p0
                adv = (low[j] - p0) / p0
            else:
                fav = (p0 - low[j]) / p0
                adv = (p0 - high[j]) / p0
            if fav > best:
                best = fav
            if adv < worst:
                worst = adv
        v = vol_at_t0[e] if vol_at_t0[e] > 1e-9 else 1e-9
        mae[e] = -worst / v
        mfe[e] = best / v
    return mae, mfe


def trade_stats(y_meta, side, ret, vol_at_t0, mask, label="trades"):
    """R-multiple based expectancy stats for a boolean mask into the meta arrays."""
    n = int(mask.sum())
    if n == 0:
        return {"n": 0}
    lab = y_meta[mask]
    win_rate = float(lab.mean())
    signed_pnl = side[mask] * ret[mask]
    R = signed_pnl / np.clip(vol_at_t0[mask] * TB_CFG_TRADE.sl_mult, 1e-9, None)
    avg_win_R = float(R[lab == 1].mean()) if (lab == 1).any() else float("nan")
    avg_loss_R = float(R[lab == 0].mean()) if (lab == 0).any() else float("nan")
    nominal_raw_R = win_rate * TB_CFG_TRADE.pt_mult - (1 - win_rate) * TB_CFG_TRADE.sl_mult
    realized_mean_R = float(R.mean())
    lo, mid, hi = block_bootstrap(R, block_size=20, n_boot=1000)
    wlo, whi = wilson_ci(int(lab.sum()), n)
    # max losing streak (chronological order assumed already in mask application)
    streak = 0
    max_streak = 0
    for v in lab:
        if v == 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return {"n": n, "win_rate": win_rate, "win_rate_wilson_95": [wlo, whi],
            "avg_win_R": avg_win_R, "avg_loss_R": avg_loss_R,
            "nominal_raw_R": nominal_raw_R, "realized_mean_R": realized_mean_R,
            "realized_mean_R_bootstrap_95": [lo, hi],
            "max_losing_streak": int(max_streak)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=None)
    ap.add_argument("--quick", action="store_true", help="skip family ablations + bootstrap for a fast smoke test")
    args = ap.parse_args()
    t_start = time.time()

    log("loading raw M1 + building 28-feature matrix (deployed feature set, unchanged)")
    df = load_raw_m1()
    if args.rows:
        df = df.tail(args.rows).reset_index(drop=True)
    feat = build_features(df)
    close = df["close"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    times = pd.to_datetime(feat["time"].to_numpy())

    vol = feat["ewma_vol"].to_numpy(dtype=np.float64)
    vol_filled = np.where(np.isfinite(vol) & (vol > 0), vol, np.nanmedian(vol[np.isfinite(vol)]))
    cusum_threshold = np.clip(CUSUM_K * vol_filled * close, 1e-6, None)
    event_mask = cusum_filter(close, cusum_threshold)
    vol_tb = vol_filled * np.sqrt(TB_CFG_DIR.max_holding) * HORIZON_VOL_SCALE

    feature_cols = [c for c in feat.columns if c != "time"]
    warmup_ok = feat[feature_cols].notna().all(axis=1).to_numpy()
    horizon_ok = np.arange(len(df)) < (len(df) - TB_CFG_DIR.max_holding - 1)
    valid = event_mask & warmup_ok & horizon_ok
    t0_idx = np.where(valid)[0]
    eligible = warmup_ok & horizon_ok

    log(f"bars={len(df):,} cusum_events={len(t0_idx):,} ({len(t0_idx)/len(df)*100:.2f}% of bars) "
        f"features={len(feature_cols)}")

    # ---- item 13: CUSUM event stats ----
    ev_pos = np.where(event_mask & eligible)[0]
    gaps = np.diff(ev_pos) if len(ev_pos) > 1 else np.array([])
    cusum_stats = {
        "total_bars": int(len(df)), "cusum_events": int(len(t0_idx)),
        "event_frequency_pct": float(len(t0_idx) / len(df) * 100),
        "gap_bars_mean": float(gaps.mean()) if len(gaps) else None,
        "gap_bars_median": float(np.median(gaps)) if len(gaps) else None,
        "gap_bars_p10": float(np.percentile(gaps, 10)) if len(gaps) else None,
        "gap_bars_p90": float(np.percentile(gaps, 90)) if len(gaps) else None,
    }

    # ---- direction labels (symmetric TB_CFG_DIR) ----
    labels = triple_barrier_labels(close, high, low, t0_idx, vol_tb, TB_CFG_DIR, side=None)
    y_raw = labels["label"].to_numpy()
    t1_raw = labels["t1"].to_numpy()
    nz = y_raw != 0
    t0_nz = t0_idx[nz]
    y_bin = pd.Series((y_raw[nz] == 1).astype(np.int64)).reset_index(drop=True)
    t0 = pd.Series(t0_nz).reset_index(drop=True)
    t1 = pd.Series(t1_raw[nz]).reset_index(drop=True)
    X_full = feat.loc[t0_nz, feature_cols].reset_index(drop=True)
    log(f"directional events={len(t0_nz):,} ({(~nz).mean()*100:.1f}% dropped as vertical timeout)")

    results = {"meta": {
        "n_bars": int(len(df)), "n_cusum_events": int(len(t0_idx)),
        "n_directional_events": int(len(t0_nz)),
        "date_range": [str(times[0]), str(times[-1])],
        "feature_cols": feature_cols, "n_features": len(feature_cols),
        "tb_cfg_dir": TB_CFG_DIR.__dict__, "tb_cfg_trade": TB_CFG_TRADE.__dict__,
        "horizon_vol_scale": HORIZON_VOL_SCALE, "cusum_k": CUSUM_K,
        "n_splits": N_SPLITS, "embargo_bars": EMBARGO_BARS,
        "catboost_kw": CATBOOST_KW, "val_fraction": VAL_FRACTION,
        "deployed_meta_prob_threshold": META_PROB_THRESHOLD,
    }, "cusum": cusum_stats}

    # ================= BASELINE FULL RUN =================
    log("== baseline primary OOF (28 features) ==")
    prim = oof_run(X_full, y_bin, t0, t1, tag="baseline-primary")
    results["primary_baseline"] = {"fold_metrics": prim["fold_metrics"]}

    side_in = np.where(prim["oof_pred"] == 1, 1.0, -1.0)
    side, meta_labels = build_meta(close, high, low, vol_tb, t0_nz, side_in, prim["has_oof"])
    has_oof = prim["has_oof"]
    X_meta = X_full.loc[has_oof].reset_index(drop=True)
    X_meta["assumed_side"] = side
    y_meta = pd.Series(meta_labels["label"].to_numpy())
    t0_meta = pd.Series(meta_labels.index.to_numpy())
    t1_meta = pd.Series(meta_labels["t1"].to_numpy())
    ret_meta = meta_labels["ret"].to_numpy()
    touch_meta = meta_labels["touch"].to_numpy()
    holding_meta = meta_labels["holding_bars"].to_numpy()
    vol_at_meta = vol_tb[t0_nz][has_oof]

    log("== baseline meta OOF ==")
    meta = oof_run(X_meta, y_meta, t0_meta, t1_meta, tag="baseline-meta")
    results["meta_baseline"] = {"fold_metrics": meta["fold_metrics"]}

    oof_meta_proba = meta["oof_proba"]
    valid_meta = meta["has_oof"]
    meta_fold_id = meta["fold_id"]
    y_meta_np = y_meta.to_numpy()
    side_np = side
    t0_meta_np = t0_meta.to_numpy()
    t1_meta_np = t1_meta.to_numpy()
    meta_times = times.to_numpy()[t0_meta_np]
    meta_years = pd.to_datetime(meta_times).year

    spread_pts = feat["spread"].to_numpy()[t0_nz][has_oof]
    spread_price = np.where(np.isfinite(spread_pts), spread_pts, np.nanmedian(spread_pts)) * 0.01
    close_at_meta = close[t0_nz][has_oof]
    sl_dist_price = TB_CFG_TRADE.sl_mult * vol_at_meta * close_at_meta
    spread_R = spread_price / np.clip(sl_dist_price, 1e-9, None)

    # ================= 2. CALIBRATION =================
    log("== calibration ==")
    calib_bins = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.01]
    calib_rows = []
    for i in range(len(calib_bins) - 1):
        lo, hi = calib_bins[i], calib_bins[i + 1]
        m = valid_meta & (oof_meta_proba >= lo) & (oof_meta_proba < hi)
        n = int(m.sum())
        if n == 0:
            calib_rows.append({"bin": f"[{lo:.2f},{hi:.2f})", "n": 0})
            continue
        pred_mean = float(oof_meta_proba[m].mean())
        obs_rate = float(y_meta_np[m].mean())
        calib_rows.append({"bin": f"[{lo:.2f},{hi:.2f})", "n": n,
                            "mean_predicted_p": pred_mean, "observed_win_rate": obs_rate,
                            "gap": obs_rate - pred_mean})
    brier = float(np.mean((oof_meta_proba[valid_meta] - y_meta_np[valid_meta]) ** 2))
    ll_meta = manual_log_loss(y_meta_np[valid_meta], oof_meta_proba[valid_meta])
    # calibration slope/intercept via simple logistic regression of y on logit(p) (Newton's method, no sklearn dep)
    p = np.clip(oof_meta_proba[valid_meta], 1e-6, 1 - 1e-6)
    logit_p = np.log(p / (1 - p))
    y_c = y_meta_np[valid_meta].astype(np.float64)
    a, b = 0.0, 1.0  # intercept, slope
    for _ in range(50):
        z = a + b * logit_p
        pr = 1 / (1 + np.exp(-z))
        w = pr * (1 - pr)
        w = np.clip(w, 1e-6, None)
        grad_a = np.sum(y_c - pr)
        grad_b = np.sum((y_c - pr) * logit_p)
        h_aa = -np.sum(w)
        h_bb = -np.sum(w * logit_p ** 2)
        h_ab = -np.sum(w * logit_p)
        det = h_aa * h_bb - h_ab ** 2
        if abs(det) < 1e-12:
            break
        da = (grad_a * h_bb - grad_b * h_ab) / det
        db = (grad_b * h_aa - grad_a * h_ab) / det
        a -= da
        b -= db
    results["calibration"] = {"bins": calib_rows, "brier_score": brier, "logloss": ll_meta,
                               "calibration_intercept": float(a), "calibration_slope": float(b),
                               "note": "slope=1,intercept=0 is perfect calibration; slope<1 means "
                                       "probabilities are overconfident (too spread out), slope>1 underconfident."}

    # ================= 3+4. THRESHOLD SWEEP (unconditional + sequential) =================
    log("== threshold sweep (deployed rule + sensitivity) ==")
    threshold_rows = []
    for thr in THRESHOLDS:
        m = valid_meta & (oof_meta_proba >= thr)
        cand_idx = np.where(m)[0]
        row = {"threshold": thr, "unconditional_n": int(len(cand_idx))}
        if len(cand_idx) >= 30:
            uw = float(y_meta_np[cand_idx].mean())
            row["unconditional_win_rate"] = uw
            row["unconditional_raw_R"] = uw * TB_CFG_TRADE.pt_mult - (1 - uw) * TB_CFG_TRADE.sl_mult
            row["unconditional_spread_R"] = float(spread_R[cand_idx].mean())
            row["unconditional_net_R"] = row["unconditional_raw_R"] - row["unconditional_spread_R"]
        order = cand_idx[np.argsort(t0_meta_np[cand_idx])]
        accepted = greedy_sequential(t0_meta_np, t1_meta_np, order)
        row["sequential_n"] = int(len(accepted))
        if len(accepted) >= 30:
            stats = trade_stats(y_meta_np, side_np, ret_meta, vol_at_meta, np.isin(np.arange(len(y_meta_np)), accepted))
            first_t, last_t = pd.Timestamp(meta_times[accepted[0]]), pd.Timestamp(meta_times[accepted[-1]])
            span_days = max((last_t - first_t).total_seconds() / 86400.0, 1.0)
            stats["trades_per_day"] = len(accepted) / span_days
            stats["spread_R_mean"] = float(spread_R[accepted].mean())
            stats["net_R_after_spread"] = stats["realized_mean_R"] - stats["spread_R_mean"]
            # per-fold breakdown
            fold_rows = []
            for f in sorted(set(meta_fold_id[accepted].tolist())):
                fm = accepted[meta_fold_id[accepted] == f]
                if len(fm) >= 10:
                    fs = trade_stats(y_meta_np, side_np, ret_meta, vol_at_meta,
                                      np.isin(np.arange(len(y_meta_np)), fm))
                    fold_rows.append({"fold": int(f), **fs})
            # per-year breakdown
            year_rows = []
            for yr in sorted(set(meta_years[accepted].tolist())):
                ym = accepted[meta_years[accepted] == yr]
                if len(ym) >= 5:
                    ys = trade_stats(y_meta_np, side_np, ret_meta, vol_at_meta,
                                      np.isin(np.arange(len(y_meta_np)), ym))
                    year_rows.append({"year": int(yr), **ys})
            row["sequential"] = stats
            row["sequential_by_fold"] = fold_rows
            row["sequential_by_year"] = year_rows
        threshold_rows.append(row)
    results["threshold_sweep"] = threshold_rows

    # deployed-rule detail already inside threshold_rows where threshold==0.60
    results["deployed_rule_0.60"] = next(r for r in threshold_rows if abs(r["threshold"] - 0.60) < 1e-9)

    # ================= 9. VOLATILITY REGIME =================
    log("== volatility regime (causal trailing-252-day terciles) ==")
    vol_series = feat["ewma_vol"].to_numpy()
    daily = pd.Series(vol_series, index=times).resample("1D").mean().dropna()
    q33 = daily.rolling(252, min_periods=20).quantile(0.3333).shift(1)
    q66 = daily.rolling(252, min_periods=20).quantile(0.6667).shift(1)
    cutoffs = pd.DataFrame({"q33": q33, "q66": q66}).dropna()
    ev_dates = pd.DatetimeIndex(meta_times).normalize()
    cutoffs_at_event = cutoffs.reindex(ev_dates, method="ffill")
    ev_vol = vol_series[t0_meta_np]
    regime = np.full(len(t0_meta_np), "unknown", dtype=object)
    have_cut = cutoffs_at_event["q33"].notna().to_numpy()
    lo_c = cutoffs_at_event["q33"].to_numpy()
    hi_c = cutoffs_at_event["q66"].to_numpy()
    regime[have_cut & (ev_vol <= lo_c)] = "low"
    regime[have_cut & (ev_vol > lo_c) & (ev_vol <= hi_c)] = "medium"
    regime[have_cut & (ev_vol > hi_c)] = "high"

    thr060_mask = valid_meta & (oof_meta_proba >= 0.60)
    regime_rows = []
    for rg in ["low", "medium", "high"]:
        rmask_all = (regime == rg) & thr060_mask
        cand_idx = np.where(rmask_all)[0]
        order = cand_idx[np.argsort(t0_meta_np[cand_idx])] if len(cand_idx) else cand_idx
        accepted = greedy_sequential(t0_meta_np, t1_meta_np, order) if len(order) else np.array([], dtype=np.int64)
        row = {"regime": rg, "cusum_events_in_regime": int((regime == rg).sum()),
               "accepted_trades": int(len(accepted))}
        if len(accepted) >= 10:
            row.update(trade_stats(y_meta_np, side_np, ret_meta, vol_at_meta,
                                    np.isin(np.arange(len(y_meta_np)), accepted)))
        regime_rows.append(row)
    results["volatility_regime"] = regime_rows

    # ================= 10. YEAR-BY-YEAR (already embedded in deployed rule above, restate top-level) =================
    results["year_by_year_at_0.60"] = results["deployed_rule_0.60"].get("sequential_by_year", [])

    # ================= 11. HUMAN EXECUTION DELAY =================
    log("== execution delay sensitivity (whole-bar shifts, M1 resolution -- NOT sub-minute) ==")
    accepted060 = greedy_sequential(
        t0_meta_np, t1_meta_np,
        np.where(thr060_mask)[0][np.argsort(t0_meta_np[np.where(thr060_mask)[0]])])
    delay_rows = []
    for shift in [0, 1, 3, 5, 10, 15, 30]:
        t0_shift = t0_meta_np[accepted060] + shift
        ok = t0_shift < (len(close) - TB_CFG_TRADE.max_holding - 1)
        t0_shift = t0_shift[ok]
        side_shift = side_np[accepted060][ok]
        vol_shift = vol_tb[t0_shift]
        shifted = triple_barrier_labels(close, high, low, t0_shift, vol_tb, TB_CFG_TRADE, side=side_shift)
        wr = float(shifted["label"].mean())
        signed = side_shift * shifted["ret"].to_numpy()
        R = signed / np.clip(vol_shift * TB_CFG_TRADE.sl_mult, 1e-9, None)
        delay_rows.append({"shift_bars": shift, "approx_seconds": shift * 60, "n": int(ok.sum()),
                            "win_rate": wr, "mean_R": float(R.mean())})
    results["execution_delay"] = {
        "note": "M1 bar data only -- true sub-minute (1s/3s/5s...) fill simulation is NOT supported by "
                "this dataset and is not attempted. shift_bars=1 is the finest honest resolution "
                "available (~60s proxy). Same accepted-trade set (threshold 0.60, sequential) re-entered "
                "at t0+shift bars, same assumed side, re-scaled vol at the new entry point.",
        "rows": delay_rows}

    # ================= 12. SL/TP GEOMETRY (MAE/MFE + touch distribution) =================
    log("== SL/TP geometry: MAE/MFE + touch distribution on deployed accepted trades ==")
    touch_acc = touch_meta[accepted060]
    holding_acc = holding_meta[accepted060]
    n_acc = len(accepted060)
    tp_first = int((touch_acc == np.where(side_np[accepted060] >= 0, 1, -1)).sum())
    sl_first = int((touch_acc == np.where(side_np[accepted060] >= 0, -1, 1)).sum())
    timeout = int((touch_acc == 0).sum())
    mae, mfe = _mae_mfe_core(close, high, low, t0_meta_np[accepted060], t1_meta_np[accepted060],
                              side_np[accepted060], vol_at_meta[accepted060])
    results["sl_tp_geometry"] = {
        "n_trades": n_acc, "tp_first_frac": tp_first / n_acc, "sl_first_frac": sl_first / n_acc,
        "timeout_frac": timeout / n_acc,
        "holding_bars_mean": float(holding_acc.mean()), "holding_bars_median": float(np.median(holding_acc)),
        "time_to_tp_mean_bars": float(holding_acc[touch_acc == np.where(side_np[accepted060] >= 0, 1, -1)].mean()) if tp_first else None,
        "time_to_sl_mean_bars": float(holding_acc[touch_acc == np.where(side_np[accepted060] >= 0, -1, 1)].mean()) if sl_first else None,
        "mae_mean_R": float(mae.mean()), "mae_median_R": float(np.median(mae)), "mae_p90_R": float(np.percentile(mae, 90)),
        "mfe_mean_R": float(mfe.mean()), "mfe_median_R": float(np.median(mfe)), "mfe_p90_R": float(np.percentile(mfe, 90)),
        "frac_mae_exceeds_sl_1.0R": float((mae >= 1.0).mean()),
        "frac_mfe_exceeds_tp_1.5R": float((mfe >= 1.5).mean()),
    }

    # ================= 14. STATISTICAL ROBUSTNESS =================
    log("== bootstrap / robustness on deployed rule ==")
    R_acc = (side_np[accepted060] * ret_meta[accepted060]) / np.clip(vol_at_meta[accepted060] * TB_CFG_TRADE.sl_mult, 1e-9, None)
    lo, mid, hi = block_bootstrap(R_acc, block_size=20, n_boot=3000)
    lab_acc = y_meta_np[accepted060]
    fold_winrates = [f["win_rate"] for f in results["deployed_rule_0.60"].get("sequential_by_fold", []) if "win_rate" in f]
    year_winrates = [f["win_rate"] for f in results["deployed_rule_0.60"].get("sequential_by_year", []) if "win_rate" in f]
    results["robustness"] = {
        "n_sequential_trades": int(n_acc),
        "win_rate_wilson_95": list(wilson_ci(int(lab_acc.sum()), n_acc)),
        "mean_R_block_bootstrap_95": [lo, hi], "mean_R_block_bootstrap_center": mid,
        "fold_win_rate_std": float(np.std(fold_winrates)) if len(fold_winrates) > 1 else None,
        "fold_win_rate_values": fold_winrates,
        "year_win_rate_std": float(np.std(year_winrates)) if len(year_winrates) > 1 else None,
        "year_win_rate_values": year_winrates,
        "caveat": "Trades are non-overlapping in time by construction (sequential-only) but returns can "
                  "still be serially correlated (shared volatility regime, autocorrelated CUSUM triggers) "
                  "-- treat the Wilson CI (assumes IID Bernoulli) as optimistic; the block bootstrap "
                  "(resamples 20-trade chunks) is the more honest interval.",
    }

    # ================= 6. TICK_VOLUME ABLATION (full primary+meta) =================
    log("== ablation: full pipeline WITHOUT tick_volume ==")
    cols_no_tv = [c for c in feature_cols if c != "tick_volume"]
    X_no_tv = feat.loc[t0_nz, cols_no_tv].reset_index(drop=True)
    prim_ntv = oof_run(X_no_tv, y_bin, t0, t1, tag="no-tickvol-primary", want_importance=False)
    side_in_ntv = np.where(prim_ntv["oof_pred"] == 1, 1.0, -1.0)
    side_ntv, meta_labels_ntv = build_meta(close, high, low, vol_tb, t0_nz, side_in_ntv, prim_ntv["has_oof"])
    has_oof_ntv = prim_ntv["has_oof"]
    X_meta_ntv = X_no_tv.loc[has_oof_ntv].reset_index(drop=True)
    X_meta_ntv["assumed_side"] = side_ntv
    y_meta_ntv = pd.Series(meta_labels_ntv["label"].to_numpy())
    t0_meta_ntv = pd.Series(meta_labels_ntv.index.to_numpy())
    t1_meta_ntv = pd.Series(meta_labels_ntv["t1"].to_numpy())
    ret_meta_ntv = meta_labels_ntv["ret"].to_numpy()
    vol_at_meta_ntv = vol_tb[t0_nz][has_oof_ntv]
    meta_ntv = oof_run(X_meta_ntv, y_meta_ntv, t0_meta_ntv, t1_meta_ntv, tag="no-tickvol-meta", want_importance=False)
    valid_meta_ntv = meta_ntv["has_oof"]
    m060_ntv = valid_meta_ntv & (meta_ntv["oof_proba"] >= 0.60)
    cand_ntv = np.where(m060_ntv)[0]
    order_ntv = cand_ntv[np.argsort(t0_meta_ntv.to_numpy()[cand_ntv])]
    accepted_ntv = greedy_sequential(t0_meta_ntv.to_numpy(), t1_meta_ntv.to_numpy(), order_ntv)
    stats_ntv = trade_stats(y_meta_ntv.to_numpy(), side_ntv, ret_meta_ntv, vol_at_meta_ntv,
                             np.isin(np.arange(len(y_meta_ntv)), accepted_ntv)) if len(accepted_ntv) >= 10 else {"n": len(accepted_ntv)}
    results["ablation_no_tick_volume"] = {
        "primary_fold_metrics": prim_ntv["fold_metrics"], "meta_fold_metrics": meta_ntv["fold_metrics"],
        "sequential_trades_at_0.60": stats_ntv,
        "primary_mean_acc_delta_vs_baseline": float(np.mean([f["acc"] for f in prim_ntv["fold_metrics"]]) -
                                                      np.mean([f["acc"] for f in prim["fold_metrics"]])),
    }

    # ================= 5. SPREAD ABLATION (full primary+meta, for scenario B) =================
    log("== ablation: full pipeline WITHOUT spread feature ==")
    cols_no_sp = [c for c in feature_cols if c != "spread"]
    X_no_sp = feat.loc[t0_nz, cols_no_sp].reset_index(drop=True)
    prim_nsp = oof_run(X_no_sp, y_bin, t0, t1, tag="no-spread-primary", want_importance=False)
    side_in_nsp = np.where(prim_nsp["oof_pred"] == 1, 1.0, -1.0)
    side_nsp, meta_labels_nsp = build_meta(close, high, low, vol_tb, t0_nz, side_in_nsp, prim_nsp["has_oof"])
    has_oof_nsp = prim_nsp["has_oof"]
    X_meta_nsp = X_no_sp.loc[has_oof_nsp].reset_index(drop=True)
    X_meta_nsp["assumed_side"] = side_nsp
    y_meta_nsp = pd.Series(meta_labels_nsp["label"].to_numpy())
    t0_meta_nsp = pd.Series(meta_labels_nsp.index.to_numpy())
    t1_meta_nsp = pd.Series(meta_labels_nsp["t1"].to_numpy())
    ret_meta_nsp = meta_labels_nsp["ret"].to_numpy()
    vol_at_meta_nsp = vol_tb[t0_nz][has_oof_nsp]
    meta_nsp = oof_run(X_meta_nsp, y_meta_nsp, t0_meta_nsp, t1_meta_nsp, tag="no-spread-meta", want_importance=False)
    valid_meta_nsp = meta_nsp["has_oof"]
    m060_nsp = valid_meta_nsp & (meta_nsp["oof_proba"] >= 0.60)
    cand_nsp = np.where(m060_nsp)[0]
    order_nsp = cand_nsp[np.argsort(t0_meta_nsp.to_numpy()[cand_nsp])]
    accepted_nsp = greedy_sequential(t0_meta_nsp.to_numpy(), t1_meta_nsp.to_numpy(), order_nsp)
    stats_nsp = trade_stats(y_meta_nsp.to_numpy(), side_nsp, ret_meta_nsp, vol_at_meta_nsp,
                             np.isin(np.arange(len(y_meta_nsp)), accepted_nsp)) if len(accepted_nsp) >= 10 else {"n": len(accepted_nsp)}
    results["ablation_no_spread"] = {
        "primary_fold_metrics": prim_nsp["fold_metrics"], "meta_fold_metrics": meta_nsp["fold_metrics"],
        "sequential_trades_at_0.60": stats_nsp,
    }

    # ================= 5. SPREAD SCENARIOS =================
    log("== spread cost scenarios ==")
    seed_path = os.path.join(BASE, "data", "gold_seed.csv")
    live_real = pd.read_csv(seed_path, usecols=["time", "spread", "src"], parse_dates=["time"])
    live_real = live_real[live_real["src"] == "xmlive"]
    empirical = {
        "n_live_minutes_observed": int(len(live_real)),
        "date_range": [str(live_real["time"].min()), str(live_real["time"].max())] if len(live_real) else None,
        "mean_pts": float(live_real["spread"].mean()) if len(live_real) else None,
        "median_pts": float(live_real["spread"].median()) if len(live_real) else None,
        "p10_pts": float(live_real["spread"].quantile(0.10)) if len(live_real) else None,
        "p90_pts": float(live_real["spread"].quantile(0.90)) if len(live_real) else None,
        "max_pts": float(live_real["spread"].max()) if len(live_real) else None,
    }
    baseline_seq = results["deployed_rule_0.60"]["sequential"]
    win_rate_060 = baseline_seq["win_rate"]
    conservative_rows = []
    for cost_pts in [15, 20, 25, 30, 40, 50]:
        cost_price = cost_pts * 0.01
        cost_R = float(np.mean(cost_price / np.clip(sl_dist_price[accepted060], 1e-9, None)))
        conservative_rows.append({"assumed_cost_points": cost_pts,
                                   "net_R": baseline_seq["realized_mean_R"] - cost_R})
    results["spread_analysis"] = {
        "scenario_A_current_synthetic_assumption": {
            "description": "98.9% of historical spread column is a fake constant (20.0 pts); this is what "
                            "the deployed backtest's cost model has been implicitly using.",
            "net_R_after_spread": baseline_seq.get("net_R_after_spread"),
        },
        "scenario_B_spread_excluded_from_model_features": {
            "description": "Model retrained without spread as a feature (still costed at inference); "
                            "see ablation_no_spread above for the full OOF comparison.",
            "primary_acc_delta_vs_baseline": float(np.mean([f["acc"] for f in prim_nsp["fold_metrics"]]) -
                                                     np.mean([f["acc"] for f in prim["fold_metrics"]])),
        },
        "scenario_C_conservative_sensitivity": {
            "description": "OBSERVED/ASSUMED: fixed flat cost assumptions in broker points applied to the "
                            "deployed rule's actual trades, independent of the (mostly fake) historical spread column.",
            "rows": conservative_rows,
        },
        "scenario_D_empirical_xm_spread": {
            "description": "OBSERVED DATA (not assumed): real spread from xm_ticker.py's live capture only "
                            "(src=='xmlive' rows in gold_seed.csv). Small, recent sample -- do not extrapolate "
                            "to the full 6.7yr history.",
            **empirical,
        },
    }

    # ================= 7+8. FEATURE IMPORTANCE + ABLATION =================
    if not args.quick:
        log("== feature importance (catboost + permutation, fold-level stability) ==")
        imp_df = pd.DataFrame(prim["importances"])
        imp_mean = imp_df.mean().sort_values(ascending=False)
        imp_std = imp_df.std()
        cv_ratio = (imp_std / imp_mean.replace(0, np.nan)).abs()

        # permutation importance on the LAST fold's test set using that fold's last-trained model
        log("  refitting last fold for permutation importance...")
        cv2 = PurgedWalkForwardCV(n_splits=N_SPLITS, embargo_bars=EMBARGO_BARS)
        splits = list(cv2.split(t0.to_numpy(), t1.to_numpy()))
        last_train, last_test = splits[-1]
        last_model = _fit(X_full, y_bin, last_train)
        base_pred = last_model.predict_proba(X_full.iloc[last_test])[:, 1]
        base_acc = float(((base_pred >= 0.5).astype(int) == y_bin.iloc[last_test].to_numpy()).mean())
        perm_importance = {}
        rng = np.random.default_rng(7)
        for col in feature_cols:
            Xp = X_full.iloc[last_test].copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())
            pp = last_model.predict_proba(Xp)[:, 1]
            acc = float(((pp >= 0.5).astype(int) == y_bin.iloc[last_test].to_numpy()).mean())
            perm_importance[col] = base_acc - acc

        feature_rows = []
        for c in feature_cols:
            cb_imp = float(imp_mean[c])
            cb_cv = float(cv_ratio[c]) if np.isfinite(cv_ratio[c]) else None
            perm = perm_importance[c]
            if cb_imp > imp_mean.median() * 1.5 and perm > 0.002 and (cb_cv is None or cb_cv < 0.5):
                cls = "CORE"
            elif cb_imp > imp_mean.median() and (perm > 0 or cb_imp > 0):
                cls = "SUPPORTING"
            elif cb_cv is not None and cb_cv >= 0.75:
                cls = "UNSTABLE"
            elif perm <= 0 and cb_imp <= imp_mean.median() * 0.5:
                cls = "WEAK"
            else:
                cls = "UNKNOWN"
            feature_rows.append({"feature": c, "catboost_importance_mean": cb_imp,
                                  "catboost_importance_cv": cb_cv, "permutation_delta_acc": perm,
                                  "classification": cls})
        feature_rows.sort(key=lambda r: -r["catboost_importance_mean"])
        results["feature_importance"] = feature_rows

        log("== family ablation (primary-OOF only, 5 remaining families) ==")
        family_rows = []
        baseline_acc = float(np.mean([f["acc"] for f in prim["fold_metrics"]]))
        baseline_bal = float(np.mean([f["bal_acc"] for f in prim["fold_metrics"]]))
        for fam, cols in FAMILIES.items():
            if fam in ("tick_volume", "spread"):
                continue  # covered by the full-pipeline ablations above
            remaining = [c for c in feature_cols if c not in cols]
            Xf = feat.loc[t0_nz, remaining].reset_index(drop=True)
            r = oof_run(Xf, y_bin, t0, t1, tag=f"ablate-{fam}", want_importance=False)
            acc = float(np.mean([f["acc"] for f in r["fold_metrics"]]))
            bal = float(np.mean([f["bal_acc"] for f in r["fold_metrics"]]))
            family_rows.append({"family_removed": fam, "n_features_removed": len(cols),
                                 "acc_with_family": baseline_acc, "acc_without_family": acc,
                                 "acc_delta": acc - baseline_acc,
                                 "bal_acc_with_family": baseline_bal, "bal_acc_without_family": bal,
                                 "bal_acc_delta": bal - baseline_bal})
        family_rows.append({"family_removed": "tick_volume", "n_features_removed": 1,
                             "acc_with_family": baseline_acc,
                             "acc_without_family": float(np.mean([f["acc"] for f in prim_ntv["fold_metrics"]])),
                             "acc_delta": float(np.mean([f["acc"] for f in prim_ntv["fold_metrics"]])) - baseline_acc})
        family_rows.append({"family_removed": "spread", "n_features_removed": 1,
                             "acc_with_family": baseline_acc,
                             "acc_without_family": float(np.mean([f["acc"] for f in prim_nsp["fold_metrics"]])),
                             "acc_delta": float(np.mean([f["acc"] for f in prim_nsp["fold_metrics"]])) - baseline_acc})
        results["family_ablation"] = family_rows

        # ================= 13b. CUSUM vs random-bar baseline =================
        log("== CUSUM vs random-bar baseline ==")
        rng2 = np.random.default_rng(11)
        non_event_pool = np.where(eligible & ~event_mask)[0]
        rand_t0 = rng2.choice(non_event_pool, size=min(len(t0_idx), len(non_event_pool)), replace=False)
        rand_t0.sort()
        rand_labels = triple_barrier_labels(close, high, low, rand_t0, vol_tb, TB_CFG_DIR, side=None)
        ry = rand_labels["label"].to_numpy()
        rt1 = rand_labels["t1"].to_numpy()
        rnz = ry != 0
        rt0_nz = rand_t0[rnz]
        ry_bin = pd.Series((ry[rnz] == 1).astype(np.int64)).reset_index(drop=True)
        rt0 = pd.Series(rt0_nz).reset_index(drop=True)
        rt1_nz = pd.Series(rt1[rnz]).reset_index(drop=True)
        rX = feat.loc[rt0_nz, feature_cols].reset_index(drop=True)
        rrun = oof_run(rX, ry_bin, rt0, rt1_nz, tag="random-bar-primary", want_importance=False)
        results["cusum_vs_random"] = {
            "cusum_n_directional": int(len(t0_nz)),
            "cusum_primary_acc": baseline_acc, "cusum_primary_bal_acc": baseline_bal,
            "random_n_directional": int(len(rt0_nz)),
            "random_primary_acc": float(np.mean([f["acc"] for f in rrun["fold_metrics"]])),
            "random_primary_bal_acc": float(np.mean([f["bal_acc"] for f in rrun["fold_metrics"]])),
        }
    else:
        results["feature_importance"] = "skipped (--quick)"
        results["family_ablation"] = "skipped (--quick)"
        results["cusum_vs_random"] = "skipped (--quick)"

    results["elapsed_sec"] = time.time() - t_start

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_json = os.path.join(OUT, f"audit_{stamp}.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(os.path.join(OUT, "latest.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    # CSV exports for the sweep + importance tables
    pd.DataFrame(threshold_rows).to_csv(os.path.join(OUT, f"threshold_sweep_{stamp}.csv"), index=False)
    if isinstance(results.get("feature_importance"), list):
        pd.DataFrame(results["feature_importance"]).to_csv(os.path.join(OUT, f"feature_importance_{stamp}.csv"), index=False)
    if isinstance(results.get("family_ablation"), list):
        pd.DataFrame(results["family_ablation"]).to_csv(os.path.join(OUT, f"family_ablation_{stamp}.csv"), index=False)

    log(f"DONE in {results['elapsed_sec']:.1f}s -> {out_json}")


if __name__ == "__main__":
    main()
