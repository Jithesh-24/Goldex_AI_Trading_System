#!/usr/bin/env python3
"""Download FULL 2021 XAUUSD M1 from Dukascopy — fills the missing year.

Why: every source on disk (gold_features_rally.csv, xauusd_rally.csv,
gold_seed_full6yr.csv) skips 2021 — the downloader's PERIODS list never
included it. The model has therefore never seen a full post-COVID
consolidation year. This script fetches every trading day of 2021
(1 Jan → 31 Dec) with retry/backoff, validates M1 integrity, dedups, and
appends to the rally cache so the matrix rebuild includes 2021.

Record format (24 bytes, big-endian):
  int32  sec_offset_from_00:00 UTC
  int32  open   (price * 1000)
  int32  close
  int32  low
  int32  high
  float32 volume (lots)
"""
import lzma, struct, os, sys, time, urllib.request, csv
from datetime import date, timedelta, datetime, timezone

BASE = "https://datafeed.dukascopy.com/datafeed/XAUUSD"
OUT = "/home/jith/.hermes/profiles/trading/scripts/gold_m1_2021.csv"
POINT = 1000.0


def fetch_day(d: date, retries=6):
    """Download + parse one day of M1 candles. Returns list of (epoch_sec, o,h,l,c,vol)."""
    url = f"{BASE}/{d.year}/{d.month-1:02d}/{d.day:02d}/BID_candles_min_1.bi5"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Accept": "*/*"})
            raw = urllib.request.urlopen(req, timeout=45).read()
            if len(raw) < 100:
                return None  # holiday/weekend
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
                print(f"  !! {d} FAILED: {e}", flush=True)
                return None
            time.sleep(5 * (attempt + 1) + 2)
    return None


def main():
    rows = []
    n_bars = 0
    d = date(2021, 1, 1)
    d1 = date(2021, 12, 31)
    while d <= d1:
        if d.weekday() < 5:  # FX closed Sat/Sun
            bars = fetch_day(d)
            if bars:
                rows.extend(bars)
                n_bars += len(bars)
            if (d - date(2021, 1, 1)).days % 25 == 0:
                print(f"  {d}: {n_bars:,} bars so far ({100*(d-date(2021,1,1)).days/364:.0f}% of year)", flush=True)
        d += timedelta(days=1)
    print(f"== 2021 total: {n_bars:,} raw bars", flush=True)

    # dedup by epoch + sort
    seen = set(); uniq = []
    for r in sorted(rows, key=lambda x: x[0]):
        if r[0] not in seen:
            seen.add(r[0]); uniq.append(r)
    print(f"TOTAL: {len(uniq)} unique M1 bars")

    # integrity: consecutive bars ≤ 120s apart
    bad_gaps = sum(1 for i in range(1, len(uniq)) if uniq[i][0] - uniq[i-1][0] > 120 or uniq[i][0] - uniq[i-1][0] <= 0)
    print(f"Integrity: {bad_gaps} non-M1 gaps in {len(uniq)} bars")
    if bad_gaps > len(uniq) * 0.02:
        print("!! TOO MANY GAPS — ABORT before save")
        sys.exit(1)

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume", "src"])
        for epo, o, h, l, c, v in uniq:
            ts = datetime.fromtimestamp(epo, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            w.writerow([ts, f"{o:.2f}", f"{h:.2f}", f"{l:.2f}", f"{c:.2f}", int(v*1000), 20, 0, "duka"])
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
