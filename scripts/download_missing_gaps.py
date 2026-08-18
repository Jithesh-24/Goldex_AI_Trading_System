#!/usr/bin/env python3
"""download_missing_gaps.py — v8.4 (2026-08-08)

Download the STILL-MISSING XAUUSD spot M1 months from Dukascopy's free feed
and save them as seed-format CSVs (time,open,high,low,close,tick_volume,
spread,real_volume) for append_missing_layers.py to ingest.

Missing months (verified against the augmented matrix):
  2020-09..12  post-COVID correction 2075 -> 1775
  2022-01,02   pre-bear highs + Russia spike to 2070
  2022-10..12  bear bottom 1620 -> recovery 1825
  2023-05..12  range chop 1800-2050
  2026-01..05  recent consolidation before Jun-Aug 2026 rally

bi5 record format (Dukascopy BID_candles_min_1, LZMA-compressed):
  int32 sec_offset_from_00:00 UTC | int32 O | int32 C | int32 L | int32 H
  float32 vol — all prices * 1000, timestamps in TRUE UTC.

NOTE: Dukascopy now requires a Referer header (plain UA gets 503).
"""
import lzma, struct, os, sys, time, urllib.request
from datetime import date, timedelta, datetime, timezone

BASE = "https://datafeed.dukascopy.com/datafeed/XAUUSD"
OUT_DIR = "/home/jith/.hermes/profiles/trading/scripts"
POINT = 1000.0
SPREAD_PTS = 20.0  # XM-like 0.20$ spread, matches seed convention

GAP_PERIODS = [
    ("2020", date(2020, 9, 1), date(2020, 12, 31)),
    ("2022a", date(2022, 1, 1), date(2022, 2, 28)),
    ("2022b", date(2022, 10, 1), date(2022, 12, 31)),
    ("2023", date(2023, 5, 1), date(2023, 12, 31)),
    ("2026", date(2026, 1, 1), date(2026, 5, 31)),
]

def fetch_day(d: date, retries=4):
    url = f"{BASE}/{d.year}/{d.month-1:02d}/{d.day:02d}/BID_candles_min_1.bi5"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
                "Accept": "*/*",
                "Referer": "https://www.dukascopy.com/",
            })
            raw = urllib.request.urlopen(req, timeout=30).read()
            if len(raw) < 100:
                return None
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
            time.sleep(2.0 * (attempt + 1))
    return None

def main():
    t0 = time.time()
    # RESUME_FROM=YYYY-MM-DD skips earlier days (used after an interrupted run
    # so we don't re-download weeks of bars we already have)
    resume_from = None
    if os.environ.get("RESUME_FROM"):
        try:
            resume_from = datetime.strptime(os.environ["RESUME_FROM"], "%Y-%m-%d").date()
            print(f"resuming from {resume_from} (days before are skipped)", flush=True)
        except ValueError:
            print(f"ignoring bad RESUME_FROM={os.environ['RESUME_FROM']}", flush=True)
    for label, d0, d1 in GAP_PERIODS:
        out_path = f"{OUT_DIR}/gap_m1_{label}.csv"
        if os.path.exists(out_path):
            print(f"{label}: {out_path} already exists — skip", flush=True)
            continue
        rows = []
        n_days = (d1 - d0).days + 1
        n_bars = 0
        d = d0
        if resume_from and d < resume_from:
            print(f"{label}: {d} -> {resume_from} skipped (resume)", flush=True)
            d = resume_from
        while d <= d1:
            if d.weekday() < 5:
                bars = fetch_day(d)
                if bars:
                    rows.extend(bars)
                    n_bars += len(bars)
            d += timedelta(days=1)
            if (d - d0).days % 30 == 0:
                print(f"  {label}: {(d-d0).days}/{n_days} days, {n_bars} bars", flush=True)
        # dedup + sort
        seen = set(); uniq = []
        for r in sorted(rows, key=lambda x: x[0]):
            if r[0] not in seen:
                seen.add(r[0]); uniq.append(r)
        if not uniq:
            print(f"  {label}: NO data downloaded — skip", flush=True)
            continue
        with open(out_path, "w") as f:
            f.write("time,open,high,low,close,tick_volume,spread,real_volume\n")
            for sec, o, h, l, c, v in uniq:
                ts = datetime.fromtimestamp(sec, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{ts},{o:.3f},{h:.3f},{l:.3f},{c:.3f},{int(v):d},{SPREAD_PTS:.1f},{int(v):d}\n")
        print(f"== {label}: {n_days}d -> {len(uniq)} bars -> {out_path} ({time.time()-t0:.0f}s)", flush=True)
    print(f"✅ gaps downloaded in {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
