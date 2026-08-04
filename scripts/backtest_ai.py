"""PHASE 3 — Backtest: simulate REAL trades with the AI model.
Walk-forward OOF probabilities → entries when P ≥ threshold → ATR-based
adaptive SL/TP (TP always ≥ 1.3× SL) → 1-min bar-by-bar resolution.
Reports honest numbers: WR, expectancy, profit factor, drawdown, trade count.

NO lookahead: every decision uses only bars up to that point.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import json, sys

BASE = "/home/jith/.hermes/profiles/trading/scripts"
FEAT_CSV = f"{BASE}/gold_features.csv"

FEATURE_EXCLUDE = {"time", "target", "fwd_return", "open", "high", "low", "close",
                   "tick_volume", "spread", "real_volume"}
# Note: raw OHLC are kept in df for trade simulation but NOT used as features
# (features come from transformed columns). We exclude raw price cols here.

# ═══ CALIBRATION: what does P=0.65 actually mean? ═══
def calibration_report(y, probs, name):
    print(f"\n=== {name} — probability calibration ===")
    probs = np.array(probs); y = np.array(y)
    rows = []
    for th in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        for direction in ["up", "down"]:
            if direction == "up":
                mask = probs >= th
                acc = y[mask].mean() if mask.any() else float("nan")
                kind = "BUY"
            else:
                mask = probs <= (1 - th)
                acc = (1 - y[mask]).mean() if mask.any() else float("nan")
                kind = "SELL"
            n = mask.sum()
            if n >= 20:  # only report statistically meaningful buckets
                rows.append((th, kind, acc, n))
    for th, kind, acc, n in rows:
        print(f"  P≥{th:.2f} {kind}: correct={acc:.1%}  n={n}")
    return rows

# ═══ TRADE SIMULATION ═══
def simulate_trades(df, probs, conf_th, sl_atr=1.6, tp_ratio=1.3, spread=0.2):
    """Walk through bars; fire when prob crosses threshold; resolve with SL/TP.
    SL = atr_14 * sl_atr (+ spread buffer). TP = SL * tp_ratio (guaranteed > SL)."""
    trades = []
    open_trade = None  # (dir, entry, sl, tp, entry_idx, entry_time)
    p = probs
    atr = df["atr_14"].values
    times = df["time"].values
    closes = df["close"].values

    for i in range(60, len(df)):
        if open_trade is None:
            # Look for entry
            if p[i] >= conf_th:  # BUY
                entry = closes[i]
                sl = entry - max(atr[i] * sl_atr, 4.0) - spread  # spread buffer on SL
                tp = entry + max((entry - sl) * tp_ratio, atr[i] * 2.0)
                open_trade = ("BUY", entry, sl, tp, i, times[i])
            elif p[i] <= (1 - conf_th):  # SELL
                entry = closes[i]
                sl = entry + max(atr[i] * sl_atr, 4.0) + spread
                tp = entry - max((sl - entry) * tp_ratio, atr[i] * 2.0)
                open_trade = ("SELL", entry, sl, tp, i, times[i])
        else:
            d, entry, sl, tp, ei, et = open_trade
            hi, lo = df["high"].values[i], df["low"].values[i]
            if d == "BUY":
                if lo <= sl:  # SL first
                    pnl = (sl - entry)  # includes spread buffer
                    trades.append((et, times[i], d, entry, sl, tp, pnl, "SL"))
                    open_trade = None
                elif hi >= tp:
                    pnl = (tp - entry)
                    trades.append((et, times[i], d, entry, sl, tp, pnl, "TP"))
                    open_trade = None
            else:  # SELL
                if hi >= sl:
                    pnl = (entry - sl)
                    trades.append((et, times[i], d, entry, sl, tp, pnl, "SL"))
                    open_trade = None
                elif lo <= tp:
                    pnl = (entry - tp)
                    trades.append((et, times[i], d, entry, sl, tp, pnl, "TP"))
                    open_trade = None
            # Time stop: max 60 bars (1 hour) — force close at close price
            if open_trade is not None and (i - ei) >= 60:
                d2, entry2, sl2, tp2, ei2, et2 = open_trade
                px = closes[i]
                pnl = (px - entry2) if d2 == "BUY" else (entry2 - px)
                trades.append((et2, times[i], d2, entry2, sl2, tp2, pnl, "TIME"))
                open_trade = None

    return trades

def report(trades, label):
    if not trades:
        print(f"{label}: NO TRADES"); return
    df = pd.DataFrame(trades, columns=["entry_t","exit_t","dir","entry","sl","tp","pnl","result"])
    wins = df[df.pnl > 0]
    losses = df[df.pnl < 0]
    wr = len(wins) / len(df)
    gross_win = wins.pnl.sum()
    gross_loss = abs(losses.pnl.sum())
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    exp = df.pnl.mean()
    # Drawdown on cumulative pnl
    cum = df.pnl.cumsum()
    dd = (cum - cum.cummax()).min()
    bu = df[df.dir == "BUY"]; sd = df[df.dir == "SELL"]
    bu_wr = (bu.pnl > 0).mean() if len(bu) else float("nan")
    sd_wr = (sd.pnl > 0).mean() if len(sd) else float("nan")
    print(f"\n=== {label} ===")
    print(f"Trades: {len(df)} | WR: {wr:.1%} | Win: {len(wins)} Loss: {len(losses)}")
    print(f"Profit factor: {pf:.2f} | Expectancy: ${exp:.2f}/trade | Net: ${df.pnl.sum():.2f}")
    print(f"Max drawdown: ${dd:.2f} | BUY WR: {bu_wr:.1%} | SELL WR: {sd_wr:.1%}")
    print(f"Avg win: ${gross_win/max(len(wins),1):.2f} | Avg loss: ${gross_loss/max(len(losses),1):.2f}")
    print(f"Result mix: {df['result'].value_counts().to_dict()}")
    return df

def main():
    # Load features + model, recompute OOF probabilities via walk-forward (same split as train)
    df = pd.read_csv(FEAT_CSV)
    df["time"] = pd.to_datetime(df["time"])
    feats = [c for c in df.columns if c not in FEATURE_EXCLUDE]
    print(f"Data: {len(df)} rows | features: {len(feats)}")

    X = df[feats].values.astype(np.float32)
    y = df["target"].values.astype(int)

    # Rebuild OOF probabilities — identical walk-forward splits to training
    from train_ai import walk_forward_splits, LGB_PARAMS
    import lightgbm as lgb
    splits = walk_forward_splits(len(df))
    oof = np.full(len(df), 0.5)
    for tr_idx, te_idx in splits:
        m = lgb.train(LGB_PARAMS, lgb.Dataset(X[tr_idx], label=y[tr_idx]), num_boost_round=600)
        oof[te_idx] = m.predict(X[te_idx])
        m.free_dataset()

    df["prob"] = oof

    # Calibration
    calibration_report(y, oof, "Walk-forward OOF")

    # Trade simulation at multiple confidence thresholds
    print("\n" + "="*60)
    print("TRADE SIMULATION — real SL/TP, spread 0.2, ATR-based adaptive levels")
    print("="*60)
    for th in [0.55, 0.60, 0.65, 0.70]:
        trades = simulate_trades(df, oof, conf_th=th)
        report(trades, f"Confidence ≥ {th:.2f}")

    # Save trades for the final chosen threshold
    best_trades = simulate_trades(df, oof, conf_th=0.65)
    if best_trades:
        pd.DataFrame(best_trades, columns=["entry_t","exit_t","dir","entry","sl","tp","pnl","result"])\
          .to_csv(f"{BASE}/backtest_trades.csv", index=False)
        print(f"\nSaved: {BASE}/backtest_trades.csv")

if __name__ == "__main__":
    main()
