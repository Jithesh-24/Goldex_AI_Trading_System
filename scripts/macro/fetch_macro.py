"""Fetch macro market data from internet (yfinance) for training + live cache.
Saves: macro_history.csv (daily 2019->now, all symbols) + macro_live.json (latest, engine cache)
Symbols: DXY (DX-Y.NYB), US10Y (^TNX), VIX (^VIX), SP500 (^GSPC), Silver (SI=F), Oil (CL=F)
Gold is the dependent variable; these are its known drivers.
"""
import yfinance as yf
import pandas as pd, json, time, os

SYMS = {"dxy": "DX-Y.NYB", "us10y": "^TNX", "vix": "^VIX",
        "spx": "^GSPC", "silver": "SI=F", "oil": "CL=F"}
OUT = os.path.dirname(os.path.abspath(__file__))

def fetch_one(name, sym):
    try:
        df = yf.download(sym, start="2019-01-01", interval="1d", progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        s = df["Close"].dropna()
        s.name = name
        return s
    except Exception as e:
        print(f"  {name} FAIL: {e}")
        return None

def main():
    t0=time.time()
    print("fetching 6 macro series from internet...", flush=True)
    series=[]
    for name, sym in SYMS.items():
        s = fetch_one(name, sym)
        if s is not None:
            series.append(s)
            print(f"  {name}: {len(s)} daily closes | {s.index.min().date()} -> {s.index.max().date()}", flush=True)
        else:
            print(f"  {name}: NO DATA", flush=True)
    if not series:
        print("ALL MACRO FETCHES FAILED — no internet or symbols wrong"); return 1
    m = pd.concat(series, axis=1)
    m.index.name = "date"
    m.to_csv(f"{OUT}/macro_history.csv")
    print(f"macro_history.csv: {m.shape} | {time.time()-t0:.0f}s", flush=True)
    # live cache (latest row) — engine reads this
    latest = m.dropna().iloc[-1].to_dict()
    json.dump({"ts": time.time(), **latest}, open(f"{OUT}/macro_live.json","w"))
    print("macro_live.json written:", {k: round(v,3) for k,v in list(latest.items())[:6]}, flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())