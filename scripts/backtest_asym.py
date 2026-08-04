"""PHASE 5b — v5 backtest: replicate the LIVE learned-placement sweep.
The live engine evaluates all 20 (direction × SL_mult × TP_ratio) candidates
with the model's CALIBRATED P(win|market+placement+direction), fires max-EV > 0.
This backtest does the SAME with walk-forward OOF probabilities (honest OOS)
and applies the SAME learned calibration curve the engine will use live.

v5 additions:
  - calibration applied to OOF probs before EV (raw LightGBM P is overconfident)
  - per-regime breakdown (TREND-UP / TREND-DOWN / RANGE / NEWS / QUIET) so we
    can see WHERE the edge lives and whether the model is profiting in every
    market condition (user demand: "profit in any market condition").
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import json, os, sys

BASE = "/home/jith/.hermes/profiles/trading/scripts"
FEAT_CSV = f"{BASE}/gold_features.csv"
FEATURE_EXCLUDE = {"time", "target", "fwd_return", "open", "high", "low", "close",
                   "tick_volume", "spread", "real_volume"}

# ── live engine's placement search space (mirrors features.py — v6 grid) ──
from features import SL_MULTS, TP_RATIOS


def load_calibration():
    try:
        with open(f"{BASE}/models/calibration.json") as f:
            return json.load(f)
    except Exception:
        return None


def reg_of(fx_row):
    """Regime bucket from features (mirrors engine's regime_label buckets)."""
    te = fx_row.get("trend_ema", 0.0); ts = fx_row.get("trend_slope", 0.0)
    bb = fx_row.get("bb_pctile", 0.5); vs = fx_row.get("vol_spike", 0.0)
    nc = fx_row.get("news_candle", 0.0)
    if nc > 0.4 or vs > 2.0:
        return "NEWS"
    if abs(te) > 1.2 and ts * te > 0:
        return "TREND-UP" if te > 0 else "TREND-DOWN"
    if bb < 0.3:
        return "RANGE"
    return "MIXED"


def simulate_v4(df, oof, spread=0.20, max_bars=60, select_p=0.0, select_p_buy=None, select_p_sell=None):
    """Walk bar-by-bar; for each bar evaluate its 20 placement rows' OOF probs,
    fire the max-EV candidate (EV>0), then first-touch simulate."""
    trades = []
    open_trade = None
    times = df["time"].values
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    sl_d_b = df["sl_dist_buy"].values
    tp_d_b = df["tp_dist_buy"].values
    sl_d_s = df["sl_dist_sell"].values
    tp_d_s = df["tp_dist_sell"].values
    direction = df["direction"].values
    n = len(df)
    from itertools import groupby
    bar_groups = groupby(range(n), key=lambda i: times[i])

    for i0, idxs in bar_groups:
        idxs = list(idxs)
        if open_trade is not None:
            d, entry, sl, tp, ei, et = open_trade
            for i in idxs:
                hi, lo = highs[i], lows[i]
                if d == "BUY":
                    if lo <= sl:
                        trades.append((et, times[i], d, entry, sl, tp, sl - entry, "SL")); open_trade = None; break
                    elif hi >= tp:
                        trades.append((et, times[i], d, entry, sl, tp, tp - entry, "TP")); open_trade = None; break
                else:
                    if hi >= sl:
                        trades.append((et, times[i], d, entry, sl, tp, entry - sl, "SL")); open_trade = None; break
                    elif lo <= tp:
                        trades.append((et, times[i], d, entry, sl, tp, entry - tp, "TP")); open_trade = None; break
            if open_trade is not None and (len(times) - 1 - ei) >= max_bars:
                d2, entry2, sl2, tp2, ei2, et2 = open_trade
                px = closes[idxs[-1]]
                pnl = (px - entry2) if d2 == "BUY" else (entry2 - px)
                trades.append((et2, times[idxs[-1]], d2, entry2, sl2, tp2, pnl, "TIME")); open_trade = None

        if open_trade is not None:
            continue

        # ── decide: max-EXPECTANCY over this bar's 20 placement rows ──
        # (v5.1: dollar EV grows with stop width and always picks the widest
        #  stop → 99.7% of trades expired TIME. Exp = P×RR − (1−P) is
        #  scale-free and purely P-driven — the honest, pro-standard metric.)
        # v5.3 LEARNED SELECTIVITY: skip candidates below the model's own
        # direction-specific OOF top-decile floor — same rule as live engine.
        best = None
        for i in idxs:
            p = float(oof[i])
            d = direction[i]
            floor = select_p_buy if d > 0.5 else select_p_sell
            if floor is None:
                floor = select_p
            if p < floor:
                continue
            if d > 0.5:
                sl_dist, tp_dist = sl_d_b[i], tp_d_b[i]
            else:
                sl_dist, tp_dist = sl_d_s[i], tp_d_s[i]
            true_sl = sl_dist + spread
            rr = tp_dist / (true_sl + 1e-9)
            exp = p * rr - (1 - p)
            if best is None or exp > best[3]:
                best = (sl_dist, tp_dist, p, exp, d, i)
        if best is None:
            continue
        sl_dist, tp_dist, p, exp, d, i_best = best
        if exp <= 0:
            continue
        entry = closes[idxs[0]]
        if d > 0.5:
            sl = entry - sl_dist - spread
            tp = entry + tp_dist
        else:
            sl = entry + sl_dist + spread
            tp = entry - tp_dist
        open_trade = ("BUY" if d > 0.5 else "SELL", entry, sl, tp, idxs[0], times[idxs[0]])
    return trades


def report(trades, label):
    if not trades:
        print(f"{label}: NO TRADES"); return
    df = pd.DataFrame(trades, columns=["entry_t","exit_t","dir","entry","sl","tp","pnl","result"])
    wins = df[df.pnl > 0]; losses = df[df.pnl < 0]
    wr = len(wins) / len(df)
    gross_win = wins.pnl.sum(); gross_loss = abs(losses.pnl.sum())
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    cum = df.pnl.cumsum(); dd = (cum - cum.cummax()).min()
    print(f"{label}: Trades={len(df)} WR={wr:.1%} PF={pf:.2f} Exp=${df.pnl.mean():.2f} Net=${df.pnl.sum():.2f} DD=${dd:.2f}")
    return df


def main():
    df = pd.read_csv(FEAT_CSV)
    df["time"] = pd.to_datetime(df["time"])
    feats = [c for c in df.columns if c not in FEATURE_EXCLUDE]
    X = df[feats].values.astype(np.float32)
    y = df["target"].values.astype(int)

    from train_ai import walk_forward_splits, LGB_PARAMS
    OOF_PROBS = f"{BASE}/models/oof_probs.npy"
    OOF_TARGETS = f"{BASE}/models/oof_targets.npy"
    if os.path.exists(OOF_PROBS) and os.path.exists(OOF_TARGETS) and \
       os.path.getmtime(OOF_PROBS) > os.path.getmtime(FEAT_CSV):
        # Reuse the exact walk-forward OOF from the last train_ai run
        # (same splits/params → identical predictions). Saves ~40 min.
        oof = np.load(OOF_PROBS)
        print(f"Loaded cached OOF ({len(oof)} rows) — skipping walk-forward re-train")
    else:
        splits = walk_forward_splits(len(df))
        oof = np.full(len(df), 0.5)
        for tr_idx, te_idx in splits:
            m = lgb.train(LGB_PARAMS, lgb.Dataset(X[tr_idx], label=y[tr_idx]), num_boost_round=600)
            oof[te_idx] = m.predict(X[te_idx])
            m.free_dataset()
        np.save(OOF_PROBS, oof.astype(np.float32))
        np.save(OOF_TARGETS, y.astype(np.int8))
        print("Walk-forward OOF computed + cached (reuse on next run)")
    df["prob"] = oof

    # v5: apply learned calibration to OOF probs (same curve the engine uses)
    cal = load_calibration()
    select_p = float(cal.get("select_p", 0.0)) if cal else 0.0
    select_p_buy = float(cal.get("select_p_buy", 0.0)) if cal else 0.0
    select_p_sell = float(cal.get("select_p_sell", 0.0)) if cal else 0.0
    if cal:
        from calibrate import apply_calibration
        df["prob"] = apply_calibration(df["prob"].values, cal)
        print(f"Calibration applied ({len(cal['knots_p'])} knots) — raw P → truthful P")
        print(f"LEARNED SELECTIVITY floors — BUY {select_p_buy:.4f} | SELL {select_p_sell:.4f} "
              f"(per-direction top-decile of own OOF)\n")

    print("=" * 70)
    print("v5.3 LEARNED-PLACEMENT BACKTEST (calibrated Exp-max + per-direction selectivity)")
    print("=" * 70)
    for spread in [0.20, 0.30, 0.41]:
        trades = simulate_v4(df, df["prob"].values, spread=spread, select_p=select_p,
                             select_p_buy=select_p_buy, select_p_sell=select_p_sell)
        report(trades, f"Exp-max sweep @ spread ${spread:.2f}")

    print("\n" + "=" * 70)
    print("PER-DIRECTION breakdown (which side carries the edge?)")
    print("=" * 70)
    trades = simulate_v4(df, df["prob"].values, select_p=select_p,
                         select_p_buy=select_p_buy, select_p_sell=select_p_sell)
    if trades:
        tdf = pd.DataFrame(trades, columns=["entry_t","exit_t","dir","entry","sl","tp","pnl","result"])
        for side in ["BUY", "SELL"]:
            sub = tdf[tdf.dir == side]
            report([tuple(r) for r in sub.values], f"{side} side")
        tdf.to_csv(f"{BASE}/backtest_v4.csv", index=False)
        print(f"\nSaved: {BASE}/backtest_v4.csv")

        # ── per-regime breakdown (user demand: profit in ANY market) ──
        print("\n" + "=" * 70)
        print("PER-REGIME breakdown (where does the edge live?)")
        print("=" * 70)
        # map each entry to its regime via the entry bar's features
        entry_t = pd.to_datetime(tdf["entry_t"])
        feats_df = df[["time"] + [c for c in ("trend_ema","trend_slope","bb_pctile",
                                              "vol_spike","news_candle") if c in df.columns]]
        feats_df = feats_df.drop_duplicates(subset="time").set_index("time")
        regs = []
        for et in entry_t:
            try:
                row = feats_df.loc[et]
                regs.append(reg_of(row.to_dict()))
            except Exception:
                regs.append("MIXED")
        tdf["regime"] = regs
        for reg in sorted(tdf["regime"].unique()):
            sub = tdf[tdf.regime == reg]
            report([tuple(r) for r in sub.drop(columns=["regime"]).values], f"{reg} regime")


if __name__ == "__main__":
    main()
