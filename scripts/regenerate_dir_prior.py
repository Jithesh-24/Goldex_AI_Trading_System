"""Regenerate regime_dir_prior.json with the CURRENT trend-first regime_bin
at the TRADE horizon (60 bars = 180 min), matching how live trades resolve.

2026-08-06 FIX: the deployed prior was measured 2026-08-05 with the OLD
vol-first regime classifier and a broken per-row method. It claimed gold
mean-reverts (STRONG_UP -> P(up)=0.476), so the engine shorted strength and
got run over (5 SL / 1 TP in 38 min while gold climbed 4234->4244). A fresh
stream of the same matrix with the current classifier shows P(up) ~0.76-0.77
at the 180-min trade horizon in EVERY regime (secular bull). This file makes
the direction tilt match reality. Also emits the 15-min column for reference.
"""
import pandas as pd, numpy as np, sys, json
sys.path.insert(0, '.')
import features as F

HORIZONS = [5, 20, 60]   # 15min / 60min / 180min
CSV = 'gold_features.csv'
OUT = "/home/jith/.hermes/profiles/trading/scripts/models/regime_dir_prior.json"

regime_rows = {n: {'t': [], 'c': []} for n in F.REGIME_NAMES}

chunk = pd.read_csv(CSV, usecols=['time', 'close', 'trend_ema',
                                  'trend_slope', 'bb_pctile', 'atr_pctile',
                                  'vol_spike', 'news_candle', 'rsi_14',
                                  'm1_d1_vol_ratio'],
                    chunksize=300000)
for c in chunk:
    if len(c) == 0:
        continue
    f = c[['trend_ema', 'trend_slope', 'bb_pctile', 'atr_pctile', 'vol_spike',
           'news_candle', 'rsi_14', 'm1_d1_vol_ratio']].fillna(0.0)
    r = np.array([F.regime_bin(f.iloc[i]) for i in range(len(c))])
    for n in F.REGIME_NAMES:
        m = r == n
        if m.sum() == 0:
            continue
        times = pd.to_datetime(c['time'].values[m]).astype('int64') // 10**9
        closes = c['close'].values[m]
        _, idx = np.unique(times, return_index=True)
        regime_rows[n]['t'].append(times[idx])
        regime_rows[n]['c'].append(closes[idx])

out = {}
for n in F.REGIME_NAMES:
    if not regime_rows[n]['t']:
        continue
    t = np.concatenate(regime_rows[n]['t'])
    c = np.concatenate(regime_rows[n]['c'])
    o = np.argsort(t)
    t, c = t[o], c[o]
    out[n] = {}
    for h in HORIZONS:
        tgt = t + h * 180
        j = np.searchsorted(t, tgt, side='left')
        valid = j < len(t)
        i = np.arange(len(t))[valid]
        jj = j[valid]
        ups = int((c[jj] > c[i]).sum())
        tot = len(i)
        out[n][f'h{h}'] = round(ups / max(tot, 1), 4)
        out[n][f'n{h}'] = int(tot)
        print(f'  {n:15s} h={h:>3} ({h*3:>4}min): P(up)={out[n][f"h{h}"]:.4f}  n={tot:,}',
              flush=True)

deploy = {
    "_note": ("Regenerated 2026-08-06 with CURRENT trend-first regime_bin at the "
              "TRADE horizon. Old file (Aug 5) used the vol-first classifier and a "
              "broken method, claiming mean reversion (STRONG_UP P(up)=0.476) -> "
              "engine shorted strength, 5 SL / 1 TP in 38 min while gold climbed "
              "4234->4244. Fresh stream of the same 6.4yr matrix: P(up)=0.76-0.77 "
              "at 180 min in every regime (secular gold bull). Engine uses h60 as "
              "the direction tilt; h5 kept for reference."),
    "horizon_bars": 60,
    "bar_seconds": 180,
    "source": "gold_features.csv stream, trend-first regime_bin, one row/timestamp",
    "measured": "2026-08-06",
    "P_up_by_regime": {n: out[n]['h60'] for n in out},
    "P_up_15min": {n: out[n]['h5'] for n in out},
    "n": {n: out[n]['n60'] for n in out},
}
tmp = OUT + ".tmp"
with open(tmp, 'w') as f:
    json.dump(deploy, f, indent=2)
os_replace = __import__('os').replace
os_replace(tmp, OUT)
print(f"✅ wrote {OUT} — engine hot-reloads per signal (line 414)")
