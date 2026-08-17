"""FINAL EDITION (v7.3f 2026-08-03) — Backtest using GENUINE walk-forward OOF.

v7.3f FIX: train_ai.py trains its FINAL deployed models on 100% of
gold_features.csv (no holdout — by design, so the live engine runs the
freshest possible model). The PRIOR version of this script loaded those
same final models and re-scored them on that same matrix: every WR / PF /
net-$ number it printed was the model grading its own homework, not a
generalization estimate.

Fix: use the walk-forward OUT-OF-FOLD probabilities train_ai.py /
train_direction.py already compute and cache (models/oof_probs.npy,
models/dir_oof_probs.npy) — real predictions from models that never saw
that time period during training. The earliest ~1/3 of history has no OOF
coverage (walk-forward needs a training burn-in before the first test
window) and is EXCLUDED rather than silently filled with in-sample scores.

Candidate (direction, sl_mult, tp_ratio) identity is recovered from each
row's OWN geometry columns, not from row position — build_full_matrix.py's
external sort only guarantees TIME order, not a fixed within-bar layout.

Everything else mirrors the live engine's decision math exactly: same
EV/expectancy formula, same calibration curves (already OOF-fit), same
first-touch SL/TP resolution. No gates (v7.1 — harness, not harden).
"""
import numpy as np
import pandas as pd
import json, sys, os, time, collections

BASE = "/home/jith/.hermes/profiles/trading/scripts"
sys.path.insert(0, BASE)
FEAT_CSV = f"{BASE}/gold_features_m5.csv"
MODEL_DIR = f"{BASE}/models"


def nearest_idx(v, grid):
    g = np.asarray(grid, dtype=np.float64)
    d = np.abs(g[None, :] - v[:, None])
    return d.argmin(axis=1)


def main():
    t0 = time.time()
    from features import SL_MULTS, TP_RATIOS
    from calibrate import load_calibration, apply_calibration
    from train_ai import walk_forward_splits, ROWS_PER_BAR

    cal = load_calibration()
    cal_by_rr = None
    _crr = f"{MODEL_DIR}/calibration_by_drr.json"
    if os.path.exists(_crr):
        with open(_crr) as f:
            cal_by_rr = json.load(f)
        print(f"Per-dir×RR calibration loaded: {list(cal_by_rr.keys())[:4]}...")
    print("Backtest v7.3f: genuine walk-forward OOF (no in-sample leakage), no gates")

    print("Loading feature matrix...")
    df = pd.read_csv(FEAT_CSV, dtype={c: "float32" for c in pd.read_csv(FEAT_CSV, nrows=0).columns if c != "time"})
    df["time"] = pd.to_datetime(df["time"])
    n = len(df)
    assert n % ROWS_PER_BAR == 0, f"{n} rows not a multiple of ROWS_PER_BAR={ROWS_PER_BAR} — matrix/code mismatch"

    oof_probs = np.load(f"{MODEL_DIR}/oof_probs.npy")
    assert len(oof_probs) == n, (f"oof_probs.npy has {len(oof_probs):,} rows but gold_features.csv has "
                                  f"{n:,} — stale cache, rerun train_ai.py before backtesting")
    splits = walk_forward_splits(n)
    test_mask = np.zeros(n, dtype=bool)
    for _, te_idx in splits:
        test_mask[te_idx] = True
    df["_oof_p"] = oof_probs
    df["_cov"] = test_mask

    # ── recover (direction, sl_mult, tp_ratio) identity from the row's OWN
    # geometry — NOT from position (external sort guarantees time order only)
    spread_d = df["spread"].values.astype(np.float64) / 100.0
    sl_dist_col = df["sl_dist_buy"].values.astype(np.float64)
    tp_dist_col = df["tp_dist_buy"].values.astype(np.float64)
    tp_ratio_recovered = tp_dist_col / (sl_dist_col + spread_d + 1e-9)
    sl_i = nearest_idx(df["sl_atr_buy"].values.astype(np.float64), SL_MULTS)
    tp_i = nearest_idx(tp_ratio_recovered, TP_RATIOS)
    err = np.abs(tp_ratio_recovered - np.asarray(TP_RATIOS)[tp_i])
    assert err.max() < 0.05, f"TP ratio recovery mismatch (max err {err.max():.4f}) — geometry formula drifted"
    dir_bit = (df["direction"].values > 0.5).astype(np.int64)
    n_tp, n_sl = len(TP_RATIOS), len(SL_MULTS)
    df["_cand"] = dir_bit * (n_sl * n_tp) + sl_i * n_tp + tp_i
    n_cand = 2 * n_sl * n_tp
    assert n_cand == ROWS_PER_BAR, f"grid size {n_cand} != ROWS_PER_BAR {ROWS_PER_BAR}"

    print("Sorting into canonical (time, candidate) layout...")
    df = df.sort_values(["time", "_cand"], kind="mergesort")  # stable
    first_block = np.sort(df["_cand"].values[:n_cand])
    assert (first_block == np.arange(n_cand)).all(), "candidate grid incomplete for first bar — build mismatch"

    n_bars = n // n_cand
    oof_grid = df["_oof_p"].values.reshape(n_bars, n_cand)
    cov_grid = df["_cov"].values.reshape(n_bars, n_cand)
    bar_covered = cov_grid.all(axis=1)
    assert (cov_grid.any(axis=1) == bar_covered).all(), "partial OOF coverage within one bar — split isn't bar-aligned"

    bars = df.iloc[::n_cand].reset_index(drop=True)  # one row/bar (market feats identical within block)
    assert len(bars) == n_bars

    first_cov = int(np.argmax(bar_covered)) if bar_covered.any() else n_bars
    print(f"Bars: {n_bars:,} total | {bar_covered.sum():,} with genuine OOF coverage "
          f"({bars['time'].iloc[first_cov] if bar_covered.any() else 'n/a'} -> {bars['time'].iloc[-1]})")
    print("(pre-coverage bars are walk-forward training burn-in — excluded, not scored)")

    # ── direction model OOF, aligned to `bars` by TIME (independent dataset —
    # built from gold_seed_multi.csv directly, not sliced from gold_features.csv)
    dp, dtp, dmp = (f"{MODEL_DIR}/dir_oof_probs.npy", f"{MODEL_DIR}/dir_oof_times.npy",
                    f"{MODEL_DIR}/dir_oof_mask.npy")
    if os.path.exists(dp) and os.path.exists(dtp):
        dir_probs = np.load(dp)
        dir_times = pd.to_datetime(np.load(dtp), unit="s")
        dir_mask = np.load(dmp) if os.path.exists(dmp) else np.ones(len(dir_probs), dtype=bool)
        dir_df = pd.DataFrame({"time": dir_times, "p_up": dir_probs, "_dcov": dir_mask})
        bars = bars.merge(dir_df, on="time", how="left")
        bars["p_up"] = bars["p_up"].fillna(0.5)
        bars["_dcov"] = bars["_dcov"].fillna(False)
        bar_covered = bar_covered & bars["_dcov"].values
        print(f"Direction OOF aligned by time: {int(bars['_dcov'].sum()):,}/{len(bars):,} bars matched")
    else:
        print("WARN: no dir_oof_times.npy cache (rerun train_direction.py to get it) — "
              "using neutral 0.5 direction prior for this backtest")
        bars["p_up"] = 0.5

    N = len(bars)
    hi = bars["high"].values.astype(float)
    lo = bars["low"].values.astype(float)
    close_v = bars["close"].values.astype(float)
    times = bars["time"].values
    spr = (bars["spread"].astype(float) / 100.0).values if "spread" in bars.columns else np.full(N, 0.2)
    p_ups = np.clip(bars["p_up"].values.astype(float), 0.05, 0.95)

    trs = np.maximum(hi[1:] - lo[1:], np.maximum(abs(hi[1:] - close_v[:-1]), abs(lo[1:] - close_v[:-1])))
    atrs = np.full(N, np.nan)
    atrs[1:] = pd.Series(trs).ewm(alpha=1 / 14, adjust=False).mean().values
    atrs[:15] = float(np.nanmean(trs[:14]))
    atrs = np.nan_to_num(atrs, nan=float(np.nanmean(trs)))

    def regime_of(row):
        te = row.get("trend_ema", 0.0); bb = row.get("bb_pctile", 0.5)
        if abs(te) > 1.2:
            return "TREND-UP" if te > 0 else "TREND-DOWN"
        if bb < 0.3:
            return "RANGE"
        return "MIXED"

    def score_direction(bar_i, direction, p_prior):
        """Max-EV sweep over the 24 (sl_mult, tp_ratio) candidates for one
        direction, using genuine OOF probabilities (oof_grid) — same
        selection rule as the live engine's best_placement()."""
        base = (n_sl * n_tp) if direction == "BUY" else 0
        atr = atrs[bar_i]; spread = max(spr[bar_i], 0.2)
        best = None
        for si, m in enumerate(SL_MULTS):
            sl_dist = max(atr * m, 0.30)
            true_sl = sl_dist + spread
            for ti, r in enumerate(TP_RATIOS):
                tp_dist = (sl_dist + spread) * r
                p_raw = oof_grid[bar_i, base + si * n_tp + ti]
                if cal_by_rr is not None:
                    knots = cal_by_rr.get(f"{direction}_{r}")
                    p = apply_calibration(p_raw, knots) if knots else p_raw
                else:
                    p = apply_calibration(p_raw, cal) if cal else p_raw
                rr = tp_dist / (true_sl + 1e-9)
                exp = p * rr - (1 - p)
                if best is None or exp > best[0]:
                    best = (exp, sl_dist, tp_dist, p)
        exp, sl_dist, tp_dist, p = best
        return exp * p_prior, (sl_dist, tp_dist, p, exp)

    trades = []
    open_t = None  # (dir, entry, sl, tp, entry_idx)
    start_i = first_cov if bar_covered.any() else N
    print("signal loop (OOF-covered bars only)...", flush=True)
    for i in range(start_i, N):
        if not bar_covered[i]:
            continue
        if open_t is not None:
            d, entry, sl, tp, ei = open_t
            hi_i, lo_i = hi[i], lo[i]
            if d == "BUY":
                if lo_i <= sl and hi_i < tp:
                    pnl, kind = sl - entry, "SL"
                elif hi_i >= tp:
                    pnl, kind = tp - entry, "TP"
                else:
                    continue
            else:
                if hi_i >= sl and lo_i > tp:
                    pnl, kind = entry - sl, "SL"
                elif lo_i <= tp:
                    pnl, kind = entry - tp, "TP"
                else:
                    continue
            trades.append({"t0": times[ei], "t1": times[i], "dir": d, "entry": entry,
                           "sl": sl, "tp": tp, "pnl": pnl, "kind": kind,
                           "reg": regime_of(bars.iloc[ei])})
            open_t = None
            continue

        p_up = p_ups[i]
        b_exp, b_tup = score_direction(i, "BUY", p_up)
        s_exp, s_tup = score_direction(i, "SELL", 1 - p_up)
        if b_exp <= 0 and s_exp <= 0:
            continue
        if b_exp >= s_exp:
            sl_dist, tp_dist, conf, exp = b_tup; direction = "BUY"
        else:
            sl_dist, tp_dist, conf, exp = s_tup; direction = "SELL"
        if exp <= 0:
            continue
        spread = max(spr[i], 0.2)
        if direction == "BUY":
            entry = close_v[i] + spread
            sl = entry - sl_dist - spread
            tp = entry + tp_dist
        else:
            entry = close_v[i]
            sl = entry + sl_dist + spread
            tp = entry - tp_dist
        open_t = (direction, entry, sl, tp, i)

    # ── report ──
    reg_map = collections.defaultdict(list)
    for tr in trades:
        reg_map[tr["reg"]].append(tr)
    net = sum(t["pnl"] for t in trades)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    kinds = collections.Counter(t["kind"] for t in trades)
    gw = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gl = -sum(t["pnl"] for t in trades if t["pnl"] < 0)
    print(f"diagnostic: SL hits={kinds.get('SL', 0)} TP hits={kinds.get('TP', 0)} | "
          f"gross_w=${gw:.2f} gross_l=${gl:.2f} implied_PF={gw/max(gl,1e-9):.3f}")
    with open(f"{BASE}/models/backtest_trades.json", "w") as f:
        json.dump([{k: (v if k not in ("t0", "t1") else str(v)) for k, v in t.items()}
                   for t in trades], f)
    print(f"trades saved: {BASE}/models/backtest_trades.json ({len(trades)})")
    print(f"\n=== V7.3f OOF BACKTEST: {len(trades)} trades | net ${net:+.2f} | WR {wins/max(len(trades),1):.1%}")
    print(f"BUY {sum(1 for t in trades if t['dir']=='BUY')} | SELL {sum(1 for t in trades if t['dir']=='SELL')}")
    rrs = [abs(t["tp"] - t["entry"]) / max(abs(t["entry"] - t["sl"]), 0.01) for t in trades]
    if rrs:
        print(f"RR: min {min(rrs):.2f} | mean {np.mean(rrs):.2f} | max {max(rrs):.2f} | all ≥1: {min(rrs) >= 1.0}")
    for reg, ts in sorted(reg_map.items()):
        if len(ts) < 5:
            continue
        rnet = sum(t["pnl"] for t in ts)
        rw = sum(1 for t in ts if t["pnl"] > 0)
        gross_w = sum(t["pnl"] for t in ts if t["pnl"] > 0)
        gross_l = -sum(t["pnl"] for t in ts if t["pnl"] < 0)
        pf = gross_w / max(gross_l, 1e-9)
        rvals = [t["pnl"] / max(abs(t["entry"] - t["sl"]), 1e-9) for t in ts]
        ev_r = float(np.mean(rvals)) if rvals else 0.0
        print(f"  {reg:12s}: {len(ts):5d} trades | WR {rw/len(ts):.1%} | PF {pf:.2f} | net ${rnet:+.2f} | EV {ev_r:+.3f}R")
    print(f"\n⏱ {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
