"""download_2026_ab.py — fetch 2026-01..02 (missed by resume) and merge into gap_m1_2026.csv"""
import sys, time
sys.path.insert(0, "/home/jith/.hermes/profiles/trading/scripts")
from download_missing_gaps import fetch_day
from datetime import date, timedelta, datetime, timezone

BASE = "/home/jith/.hermes/profiles/trading/scripts"
t0 = time.time()
rows = []
d = date(2026, 1, 1)
while d <= date(2026, 2, 28):
    if d.weekday() < 5:
        b = fetch_day(d)
        if b:
            rows.extend(b)
    d += timedelta(days=1)
    if (d - date(2026, 1, 1)).days % 15 == 0:
        print(f"  {(d-date(2026,1,1)).days} days, {len(rows)} bars", flush=True)

seen = set(); uniq = []
for r in sorted(rows, key=lambda x: x[0]):
    if r[0] not in seen:
        seen.add(r[0]); uniq.append(r)

with open(f"{BASE}/gap_m1_2026ab.csv", "w") as f:
    f.write("time,open,high,low,close,tick_volume,spread,real_volume\n")
    for sec, o, h, l, c, v in uniq:
        ts = datetime.fromtimestamp(sec, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{ts},{o:.3f},{h:.3f},{l:.3f},{c:.3f},{int(v)},20.0,{int(v)}\n")
print(f"2026 Jan-Feb: {len(uniq)} bars -> gap_m1_2026ab.csv ({time.time()-t0:.0f}s)", flush=True)

# merge into the main 2026 gap file (it has Mar-May), keep sorted, dedup
import pandas as pd
a = pd.read_csv(f"{BASE}/gap_m1_2026ab.csv", parse_dates=["time"])
b = pd.read_csv(f"{BASE}/gap_m1_2026.csv", parse_dates=["time"])
m = pd.concat([a, b]).drop_duplicates(subset="time", keep="last").sort_values("time")
m.to_csv(f"{BASE}/gap_m1_2026.csv", index=False)
print(f"merged gap_m1_2026.csv: {len(m):,} bars ({m['time'].iloc[0]} -> {m['time'].iloc[-1]})", flush=True)
import os
os.remove(f"{BASE}/gap_m1_2026ab.csv")
print("✅ 2026 complete", flush=True)
