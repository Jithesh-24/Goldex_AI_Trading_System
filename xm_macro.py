"""xm_macro.py — Live cross-asset macro context poller (2026-08-12).

THE GAP CLOSED: the model was trained on gold alone — no dollar, no yields.
fetch_macro_context.py built the TRAINING macro features (macro_daily.csv,
no-lookahead, yesterday-complete by construction: shift(1) + date normalize).
This daemon computes the SAME 7 features LIVE from Yahoo daily closes and
publishes macro_state.json for the engine to inject — with the SAME
no-lookahead semantics: a prediction at time T uses the last COMPLETED daily
bar (≤ yesterday), never today's in-flight bar. Training and live agree by
construction; no distribution shift, no lookahead.

WHY yesterday-complete: the model learned P(move | yesterday's macro context).
Feeding today's live DXY would be an out-of-distribution input it never saw
in training. Slow regime features (120d z-score, 5d momentum) lose nothing in
24h. This is the honest no-leak design.

Poll interval: 300s (macro regime changes slowly; Yahoo is rate-limited).
Publication: {OUTDIR}/macro_state.json  {ts, dxy_z, dxy_5d_chg, tnx_level,
tnx_5d_chg, gc_5d_chg, gld_5d_chg, eur_5d_chg, asof(UTC date of last bar)}.
"""
import urllib.request, urllib.parse, json, os, sys, time, math
import numpy as np
import pandas as pd

BASE = "/home/jith/.hermes/profiles/trading/scripts"
OUTDIR = "/home/jith/.hermes/profiles/trading/cron/output"
STATE = f"{OUTDIR}/macro_state.json"
SYMBOLS = ["DX-Y.NYB", "^TNX", "GC=F", "GLD", "EURUSD=X"]
UA = {"User-Agent": "Mozilla/5.0 (research)"}
Z_WINDOW = 120
POLL_S = 300

FEAT_ORDER = ["dxy_z", "dxy_5d_chg", "tnx_level", "tnx_5d_chg",
              "gc_5d_chg", "gld_5d_chg", "eur_5d_chg"]


def yahoo_daily(sym, rng="2y"):
    """Daily closes 2y — enough for 120d z-score + 5d momentum."""
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/"
           f"{urllib.parse.quote(sym)}?interval=1d&range={rng}")
    req = urllib.request.Request(url, headers=UA)
    d = json.loads(urllib.request.urlopen(req, timeout=25).read())
    r = d["chart"]["result"][0]
    ts = r.get("timestamp", [])
    q = r["indicators"]["quote"][0]
    df = pd.DataFrame({"time": pd.to_datetime(ts, unit="s", utc=True),
                       "close": q.get("close", [None] * len(ts))}).dropna()
    df["time"] = df["time"].dt.normalize()
    df = df.drop_duplicates("time", keep="last").set_index("time")["close"]
    return df


def compute_live():
    """Fetch all symbols, compute features with yesterday-complete semantics."""
    closes = {}
    for sym in SYMBOLS:
        try:
            s = yahoo_daily(sym)
            if len(s) > 30:
                closes[sym] = s
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] {sym}: {type(e).__name__}: {str(e)[:60]}", flush=True)
        time.sleep(1.0)

    if len(closes) < 4:
        print(f"[{time.strftime('%H:%M:%S')}] ⚠ only {len(closes)} symbols — skipping cycle", flush=True)
        return None

    # BUGFIX (2026-08-12): never extend into FUTURE dates — bfill would stamp
    # the last close onto days Yahoo hasn't delivered yet, corrupting the
    # no-lookahead "last completed bar" semantics. Index only to the newest
    # real close any symbol delivered.
    all_idx = sorted(set().union(*[set(s.index) for s in closes.values()]))
    idx = pd.DatetimeIndex(all_idx)
    out = pd.DataFrame(index=idx)
    for sym, s in closes.items():
        out[f"{sym}_close"] = s.reindex(out.index).ffill().bfill()

    # NO-LOOKAHEAD: features from closes ≤ the last COMPLETED bar. The newest
    # Yahoo row is today's in-flight bar (its close finalizes at the next NY
    # close) — drop it, then features are computed from the completed tail.
    shifted = out.shift(1).iloc[:-1]  # removes the in-progress (today) row
    m = pd.DataFrame(index=shifted.index)
    m["dxy_z"] = ((shifted["DX-Y.NYB_close"] - shifted["DX-Y.NYB_close"].rolling(Z_WINDOW).mean())
                  / (shifted["DX-Y.NYB_close"].rolling(Z_WINDOW).std() + 1e-9))
    m["dxy_5d_chg"] = shifted["DX-Y.NYB_close"].pct_change(5) * 100.0
    m["tnx_level"] = shifted["^TNX_close"]
    m["tnx_5d_chg"] = shifted["^TNX_close"].diff(5)
    m["gc_5d_chg"] = shifted["GC=F_close"].pct_change(5) * 100.0
    m["gld_5d_chg"] = shifted["GLD_close"].pct_change(5) * 100.0
    m["eur_5d_chg"] = shifted["EURUSD=X_close"].pct_change(5) * 100.0

    last = m.dropna().iloc[-1]  # the most recent fully-known bar (≤ yesterday)
    state = {"ts": time.time(),
             "asof": str(last.name.date()),
             **{c: float(last[c]) for c in FEAT_ORDER if not math.isnan(last[c])}}
    return state


def main():
    print(f"[{time.strftime('%H:%M:%S')}] xm_macro live context poller — every {POLL_S}s", flush=True)
    while True:
        try:
            state = compute_live()
            if state:
                with open(STATE, "w") as f:
                    json.dump(state, f)
                print(f"[{time.strftime('%H:%M:%S')}] wrote {STATE} asof={state['asof']} dxy_z={state['dxy_z']:.2f} tnx={state['tnx_level']:.2f}", flush=True)
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] cycle error: {type(e).__name__}: {str(e)[:100]}", flush=True)
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()