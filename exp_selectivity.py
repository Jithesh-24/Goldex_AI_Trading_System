#!/usr/bin/env python3
"""Quick v5.2 selectivity experiment: fire only when the best candidate's
calibrated P is in the top-q of the model's OWN OOF distribution (learned
threshold, not hardcoded). Uses cached OOF — runs in seconds."""
import numpy as np
import pandas as pd
import json, sys

BASE = "/home/jith/.hermes/profiles/trading/scripts"
FEAT_CSV = f"{BASE}/gold_features_m5.csv"
sys.path.insert(0, BASE)
from calibrate import apply_calibration

cal = json.load(open(f"{BASE}/models/calibration.json"))
df = pd.read_csv(FEAT_CSV)
df["time"] = pd.to_datetime(df["time"])
oof = np.load(f"{BASE}/models/oof_probs.npy")
pcal = apply_calibration(oof, cal)
df["pcal"] = pcal

sl_d_b = df["sl_dist_buy"].values; tp_d_b = df["tp_dist_buy"].values
sl_d_s = df["sl_dist_sell"].values; tp_d_s = df["tp_dist_sell"].values
direction = df["direction"].values
times = df["time"].values; closes = df["close"].values
highs = df["high"].values; lows = df["low"].values
n = len(df)

def run(q, spread=0.20, max_bars=60):
    th = np.percentile(pcal[pcal != 0], q) if q < 100 else 0.0
    trades = []
    open_trade = None
    fired = 0
    from itertools import groupby
    bar_groups = groupby(range(n), key=lambda i: times[i])  # fresh iterator per run
    for i0, idxs in bar_groups:
        idxs = list(idxs)
        if open_trade is not None:
            d, entry, sl, tp, ei, et = open_trade
            for i in idxs:
                hi, lo = highs[i], lows[i]
                if d == "BUY":
                    if lo <= sl: trades.append((et, times[i], d, entry, sl, tp, sl-entry, "SL")); open_trade=None; break
                    elif hi >= tp: trades.append((et, times[i], d, entry, sl, tp, tp-entry, "TP")); open_trade=None; break
                else:
                    if hi >= sl: trades.append((et, times[i], d, entry, sl, tp, entry-sl, "SL")); open_trade=None; break
                    elif lo <= tp: trades.append((et, times[i], d, entry, sl, tp, entry-tp, "TP")); open_trade=None; break
            if open_trade is not None and (len(times)-1-ei) >= max_bars:
                d2, e2, s2, t2, ei2, et2 = open_trade
                px = closes[idxs[-1]]
                pnl = (px-e2) if d2=="BUY" else (e2-px)
                trades.append((et2, times[idxs[-1]], d2, e2, s2, t2, pnl, "TIME")); open_trade=None
        if open_trade is not None:
            continue
        best = None
        for i in idxs:
            p = float(pcal[i]); d = direction[i]
            sl_dist, tp_dist = (sl_d_b[i], tp_d_b[i]) if d>0.5 else (sl_d_s[i], tp_d_s[i])
            true_sl = sl_dist + spread
            rr = tp_dist/(true_sl+1e-9)
            exp = p*rr - (1-p)
            if best is None or exp > best[3]:
                best = (sl_dist, tp_dist, p, exp, d, i)
        if best is None: continue
        sl_dist, tp_dist, p, exp, d, i_best = best
        if p < th: continue   # LEARNED selectivity: top-q of own OOF distribution
        fired += 1
        entry = closes[idxs[0]]
        if d>0.5: sl=entry-sl_dist-spread; tp=entry+tp_dist
        else: sl=entry+sl_dist+spread; tp=entry-tp_dist
        open_trade = ("BUY" if d>0.5 else "SELL", entry, sl, tp, idxs[0], times[idxs[0]])
    return trades, fired

def report(trades, label, fired=None):
    if not trades:
        print(f"{label}: NO TRADES"); return
    tdf = pd.DataFrame(trades, columns=["et","xt","dir","entry","sl","tp","pnl","result"])
    wins = tdf[tdf.pnl>0]; losses = tdf[tdf.pnl<0]
    wr = len(wins)/len(tdf)
    gw, gl = wins.pnl.sum(), abs(losses.pnl.sum())
    pf = gw/gl if gl>0 else float("inf")
    cum = tdf.pnl.cumsum(); dd = (cum-cum.cummax()).min()
    resolved = tdf[tdf.result!="TIME"]
    rwr = (resolved.result=="TP").mean() if len(resolved) else float("nan")
    extra = f" | fired={fired}" if fired is not None else ""
    print(f"{label}: Trades={len(tdf)} WR={wr:.1%} PF={pf:.2f} Net=${tdf.pnl.sum():.2f} DD=${dd:.2f} resolved={len(resolved)} (WR {rwr:.1%}){extra}")
    # TIME vs resolved breakdown
    t = tdf[tdf.result=="TIME"]; s = tdf[tdf.result=="SL"]; tp = tdf[tdf.result=="TP"]
    print(f"    TIME: n={len(t)} Net=${t.pnl.sum():.2f} | SL: n={len(s)} Net=${s.pnl.sum():.2f} | TP: n={len(tp)} Net=${tp.pnl.sum():.2f}")
    return tdf

print("v5.2 selectivity sweep (learned P threshold from own OOF):\n")
for q in [100, 90, 95, 99]:
    trades, fired = run(q)
    report(trades, f"P >= OOF top {100-q}% (q={q})" if q<100 else "no selectivity (q=100)", fired)
    print()
