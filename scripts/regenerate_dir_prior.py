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
import pandas as pd, numpy as np, sys, json, time
import os
sys.path.insert(0, '.')
import features as F

HORIZONS = [5, 20, 60]   # 15min / 60min / 180min (M1 default)
RECENT_WINDOW_DAYS = 90  # v8.3: direction prior adapts to the trailing 90d market
# v8 M5: matrix bars are 300s. The trade horizon (180 min) = 36 M5 bars.
# Env override keeps one script for both timeframes:
#   PRIOR_BAR_SECS=300 PRIOR_HORIZONS="3,12,36" → M5 (15/60/180 min)
BAR_SECS = int(os.environ.get("PRIOR_BAR_SECS", "180"))
if os.environ.get("PRIOR_HORIZONS"):
    HORIZONS = [int(x) for x in os.environ["PRIOR_HORIZONS"].split(",")]
CSV = os.environ.get('FEAT_CSV', 'gold_features.csv')
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
        times = pd.to_datetime(c['time'].values[m], utc=True).tz_localize(None).astype('datetime64[s]').astype('int64')
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
        def _pup(tt, cc):
            tgt = tt + h * BAR_SECS
            j = np.searchsorted(tt, tgt, side='left')
            valid = j < len(tt)
            i = np.arange(len(tt))[valid]
            jj = j[valid]
            ups = int((cc[jj] > cc[i]).sum())
            return ups / max(len(i), 1), len(i)
        # FULL-HISTORY prior (6yr secular average) — reference only
        p_full, n_full = _pup(t, c)
        # v8.3 (2026-08-07): RECENT-WINDOW prior — the deployed direction tilt
        # must adapt to the CURRENT market, not a static 6yr secular-bull
        # average. A 6yr P(up)=0.85 average kept buying dips through any
        # real turn (the engine's #1 structural loss factor). Measure P(up)
        # over the trailing window (default 90 days); shrink toward the
        # full-history value when the recent sample is thin (empirical-Bayes,
        # data-driven — not a gate).
        t_max = float(t[-1])
        recent = t >= (t_max - RECENT_WINDOW_DAYS * 86400.0)
        p_rec, n_rec = _pup(t[recent], c[recent]) if recent.sum() > 50 else (p_full, 0)
        w = min(n_rec / 500.0, 1.0)          # full weight at >=500 recent rows
        p_up = w * p_rec + (1 - w) * p_full  # shrink thin samples to history
        out[n][f'h{h}'] = round(p_up, 4)
        out[n][f'n{h}'] = int(n_full)
        out[n][f'n_rec{h}'] = int(n_rec)
        out[n][f'p_full_h{h}'] = round(p_full, 4)
        print(f'  {n:15s} h={h:>3} ({h*BAR_SECS//60:>4}min): P(up)={p_up:.4f} '
              f'(recent n={n_rec:,}, full={p_full:.4f}/{n_full:,})', flush=True)

# trade-horizon key: h60 (M1, 180min) or h36 (M5, 180min)
_trade_key = "h60" if BAR_SECS == 180 else f"h{HORIZONS[-1]}"
deploy = {
    "_note": (f"Regenerated {time.strftime('%Y-%m-%d')} with CURRENT trend-first "
              "regime_bin at the TRADE horizon. Old file (Aug 5) used the "
              "vol-first classifier and a broken method, claiming mean reversion "
              "(STRONG_UP P(up)=0.476) -> engine shorted strength, 5 SL / 1 TP "
              "in 38 min while gold climbed 4234->4244. Fresh stream of the same "
              "6.4yr matrix: P(up)=0.76-0.77 at 180 min in every regime (secular "
              f"gold bull). Engine uses {_trade_key} as the direction tilt; h5 kept "
              "for reference. v8: bars are M5 (300s) when PRIOR_BAR_SECS=300."),
    "horizon_bars": HORIZONS[-1],
    "bar_seconds": BAR_SECS,
    "source": f"{CSV} stream, trend-first regime_bin, one row/timestamp",
    "measured": time.strftime("%Y-%m-%d"),
    "P_up_by_regime": {n: out[n][_trade_key] for n in out},
    "P_up_15min": {n: out[n][f"h{HORIZONS[0]}"] for n in out},
    "n": {n: out[n][f'n{HORIZONS[-1]}'] for n in out},
}
tmp = OUT + ".tmp"
with open(tmp, 'w') as f:
    json.dump(deploy, f, indent=2)
os_replace = __import__('os').replace
os_replace(tmp, OUT)
print(f"✅ wrote {OUT} — engine hot-reloads per signal (line 414)")
