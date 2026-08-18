"""fetch_macro_context.py — Cross-asset context for the gold AI (2026-08-12).

THE GAP: the model trades XAUUSD with ZERO knowledge of the dollar, yields, or
flows — gold's #1 macro driver is the USD/real-yield complex. Institutions
always trade gold against DXY + yields + futures premium.

WHAT: download 5y of daily closes for the cross-asset complex from Yahoo
(unofficial, free, no key) and emit a CSV aligned for matrix merging:
  DX-Y.NYB  US Dollar Index (DXY)
  ^TNX      10Y Treasury yield
  GC=F      Gold futures (COMEX) — futures premium vs XAUUSD spot
  GLD       SPDR Gold ETF — flow/positioning proxy
  EURUSD=X  EUR/USD — the most traded USD leg

NO LOOKAHEAD: every feature uses data available at the START of the M5 bar's
day. We shift the daily series +1 day before computing stats, so a bar on
2024-03-05 sees only closes ≤ 2024-03-04. Live engine applies the same rule
(yesterday-complete only) — training and live agree by construction.

FEATURES (per M5 bar, constant within its day):
  dxy_z         DXY close z-score vs trailing 120d (macro regime: strong/weak $)
  dxy_5d_chg    DXY 5-day change % (dollar momentum)
  tnx_level     10Y yield level (real-rate proxy)
  tnx_5d_chg    10Y yield 5-day change (rate shock)
  gc_5d_chg     Gold futures 5-day change % (futures momentum — leads spot)
  gld_5d_chg    GLD 5-day change % (ETF flow proxy)
  eur_5d_chg    EURUSD 5-day change % (USD leg momentum)
"""
import urllib.request, json, urllib.parse, os, sys, time
import numpy as np
import pandas as pd

BASE = "/home/jith/.hermes/profiles/trading/scripts"
OUT = f"{BASE}/macro_daily.csv"
SYMBOLS = ["DX-Y.NYB", "^TNX", "GC=F", "GLD", "EURUSD=X"]
UA = {"User-Agent": "Mozilla/5.0 (research)"}
Z_WINDOW = 120


def yahoo_chart(sym, interval="1d", rng="10y"):
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/"
           f"{urllib.parse.quote(sym)}?interval={interval}&range={rng}")
    req = urllib.request.Request(url, headers=UA)
    d = json.loads(urllib.request.urlopen(req, timeout=25).read())
    r = d["chart"]["result"][0]
    ts = r.get("timestamp", [])
    q = r["indicators"]["quote"][0]
    return pd.DataFrame({
        "time": pd.to_datetime(ts, unit="s", utc=True),
        "close": q.get("close", [None] * len(ts)),
    }).dropna().sort_values("time").reset_index(drop=True)


def main():
    t0 = time.time()
    frames = {}
    for sym in SYMBOLS:
        try:
            df = yahoo_chart(sym)
            frames[sym] = df.set_index("time")["close"]
            print(f"  {sym}: {len(df):,} daily closes ({df['time'].iloc[0].date()} → {df['time'].iloc[-1].date()})", flush=True)
        except Exception as e:
            print(f"  {sym}: FAIL {type(e).__name__}: {str(e)[:80]}", flush=True)
        time.sleep(1.2)  # polite to Yahoo

    if len(frames) < 4:
        print("❌ too many failures — aborting")
        sys.exit(1)

    out = pd.DataFrame(index=pd.date_range("2019-11-01", "2026-08-13", freq="D", tz="UTC"))
    for sym, s in frames.items():
        # BUGFIX (2026-08-12): Yahoo daily bars are stamped at 04:00 UTC (NYSE
        # close) — reindexing against midnight-UTC would match NOTHING. Normalize
        # to date-only so the merge is by calendar day.
        s = s.copy()
        s.index = s.index.normalize()
        s = s[~s.index.duplicated(keep="last")]
        out[f"{sym}_close"] = s.reindex(out.index).ffill().bfill()

    # NO-LOOKAHEAD: shift +1 day — a bar on day D sees closes ≤ D-1
    shifted = out.shift(1)

    m = pd.DataFrame(index=out.index)
    m["dxy_z"] = ((shifted["DX-Y.NYB_close"] - shifted["DX-Y.NYB_close"].rolling(Z_WINDOW).mean())
                  / (shifted["DX-Y.NYB_close"].rolling(Z_WINDOW).std() + 1e-9))
    m["dxy_5d_chg"] = shifted["DX-Y.NYB_close"].pct_change(5) * 100.0
    m["tnx_level"] = shifted["^TNX_close"]
    m["tnx_5d_chg"] = shifted["^TNX_close"].diff(5)
    m["gc_5d_chg"] = shifted["GC=F_close"].pct_change(5) * 100.0
    m["gld_5d_chg"] = shifted["GLD_close"].pct_change(5) * 100.0
    m["eur_5d_chg"] = shifted["EURUSD=X_close"].pct_change(5) * 100.0

    m = m.reset_index().rename(columns={"index": "date"})
    m.to_csv(OUT, index=False, float_format="%.6f")
    print(f"✅ saved {OUT}: {len(m):,} days × {len(m.columns)-1} features in {time.time()-t0:.0f}s", flush=True)
    print("cols:", list(m.columns))
    print(m.dropna().head(2).to_string(), flush=True)
    print(m.dropna().tail(2).to_string(), flush=True)


if __name__ == "__main__":
    main()