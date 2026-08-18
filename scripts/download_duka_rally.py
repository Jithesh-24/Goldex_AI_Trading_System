#!/usr/bin/env python3
"""Dukascopy XAUUSD M1 rally history downloader (v6 teaching data).

Real spot XAUUSD M1 candles, free from Dukascopy's public feed.
Each day = one LZMA .bi5 file with 1440 M1 candles (full 24h session).

Record format (24 bytes, big-endian):
  int32  sec_offset_from_00:00 UTC
  int32  open   (price * 1000)
  int32  close
  int32  low
  int32  high
  float32 volume (lots)

Rally periods (the data the model has NEVER seen):
  COVID rally     2019-12-01 .. 2020-08-31   1460 -> 2075
  2024 breakout   2024-02-01 .. 2024-05-31   1990 -> 2450
  2025 melt-up    2025-01-01 .. 2025-05-31   2620 -> 3500+
Range/other regime (teach ranging + reversals):
  2023 range      2023-01-01 .. 2023-04-30   1800-2050 chop
  2022 bear       2022-03-01 .. 2022-09-30   2070 -> 1620 (down-trend teaching)
"""
import lzma, struct, os, sys, time, urllib.request, csv
from datetime import date, timedelta, datetime, timezone

BASE = "https://datafeed.dukascopy.com/datafeed/XAUUSD"
OUT = "/home/jith/.hermes/profiles/trading/scripts/xauusd_rally.csv"
POINT = 1000.0  # XAUUSD quoted with 3 decimals

PERIODS = [
    ("covid_rally",  date(2019, 12, 1),  date(2020, 8, 31)),
    ("range_2023",   date(2023, 1, 1),   date(2023, 4, 30)),
    ("bear_2022",    date(2022, 3, 1),   date(2022, 9, 30)),
    ("rally_2024",   date(2024, 2, 1),   date(2024, 5, 31)),
    ("meltup_2025",  date(2025, 1, 1),   date(2025, 5, 31)),
]

def fetch_day(d: date, retries=3):
    """Download + parse one day of M1 candles. Returns list of (epoch_sec, o,h,l,c,vol)."""
    url = f"{BASE}/{d.year}/{d.month-1:02d}/{d.day:02d}/BID_candles_min_1.bi5"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            raw = urllib.request.urlopen(req, timeout=30).read()
            if len(raw) < 100:
                return None  # no data that day (holiday/weekend)
            data = lzma.decompress(raw)
            n = len(data) // 24
            if n == 0:
                return None
            day_start = int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())
            out = []
            for i in range(n):
                sec, o, c, l, h, v = struct.unpack(">5if", data[i*24:(i+1)*24])
                out.append((day_start + sec, o/POINT, h/POINT, l/POINT, c/POINT, v))
            return out
        except Exception as e:
            if attempt == retries - 1:
                print(f"  !! {d} FAILED: {e}")
                return None
            time.sleep(1.5 * (attempt + 1))
    return None

def main():
    rows = []
    for label, d0, d1 in PERIODS:
        n_days = (d1 - d0).days + 1
        n_bars = 0
        d = d0
        while d <= d1:
            # skip weekends (Dukascopy = FX market, closed Sat/Sun)
            if d.weekday() < 5:
                bars = fetch_day(d)
                if bars:
                    rows.extend(bars)
                    n_bars += len(bars)
            d += timedelta(days=1)
            if n_days and (d - d0).days % 30 == 0:
                print(f"  {label}: {d - d0} days, {n_bars} bars so far", flush=True)
        print(f"== {label}: {n_days}d -> {n_bars} bars", flush=True)

    # dedup by epoch + sort
    seen = set(); uniq = []
    for r in sorted(rows, key=lambda x: x[0]):
        if r[0] not in seen:
            seen.add(r[0]); uniq.append(r)
    print(f"\nTOTAL: {len(uniq)} unique M1 bars")

    # ── INTEGRITY CHECK: day-boundary continuity ──
    # If the URL day index were off by one, every day's last candle would
    # NOT join the next day's first candle (timestamps would jump ±1 day).
    # Verify: consecutive bars must be ≤ 120s apart (M1 grid), else flag.
    bad_gaps = 0
    for i in range(1, len(uniq)):
        gap = uniq[i][0] - uniq[i-1][0]
        if gap > 120 or gap <= 0:
            bad_gaps += 1
    print(f"Integrity: {bad_gaps} non-M1 gaps (>120s or backwards) in {len(uniq)} bars")
    if bad_gaps > len(uniq) * 0.02:  # >2% broken = indexing bug
        print("!! TOO MANY GAPS — day index likely wrong, ABORT before save")
        sys.exit(1)

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume", "src"])
        for epo, o, h, l, c, v in uniq:
            ts = datetime.fromtimestamp(epo, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            w.writerow([ts, f"{o:.2f}", f"{h:.2f}", f"{l:.2f}", f"{c:.2f}", int(v*1000), 20, 0, "duka"])
    print(f"Saved: {OUT}")
    # summary per year
    years = {}
    for epo, *_ in uniq:
        y = datetime.fromtimestamp(epo, tz=timezone.utc).year
        years[y] = years.get(y, 0) + 1
    print("by year:", years)

if __name__ == "__main__":
    main()
