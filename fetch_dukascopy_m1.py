"""fetch_dukascopy_m1.py — 6-year XAUUSD M1 microstructure backfill.

v3 (2026-08-12): Uses curl instead of Python urllib (better SSL handling).
  - checkpoint/resume
  - per-day progress
  - error logging
  - curl with 30s timeout + retries

Usage:  python3 fetch_dukascopy_m1.py [--start 2019-12-01] [--end 2026-08-12]
        [--out dukascopy_m1_features.csv] [--symbol XAUUSD]
"""
import argparse, lzma, struct, subprocess, os, sys, time, json
import numpy as np
import pandas as pd

BASE = "https://datafeed.dukascopy.com/datafeed"
T0 = time.time()
CHECKPOINT_FILE = None
ERROR_LOG = None

def log(msg):
    print(f"[{time.time()-T0:.0f}s] {msg}", flush=True)

def log_error(msg):
    with open(ERROR_LOG, "a") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

def load_checkpoint():
    if CHECKPOINT_FILE and os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE) as f:
                return json.load(f)
        except:
            pass
    return {"completed_days": [], "last_date": None}

def save_checkpoint(ckpt):
    if CHECKPOINT_FILE:
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(ckpt, f)

def day_file(symbol, year, month01, day01, kind="BID_candles_min_1"):
    """Download one day's M1 data using curl."""
    url = f"{BASE}/{symbol}/{year:04d}/{month01:02d}/{day01:02d}/{kind}.bi5"
    tmp = f"/tmp/duka_{year}{month01:02d}{day01:02d}.bi5"

    for attempt in range(3):
        try:
            result = subprocess.run(
                ["curl", "-s", "-o", tmp,
                 "--connect-timeout", "30", "--max-time", "60",
                 "-H", "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                 url],
                capture_output=True, text=True, timeout=90
            )
            time.sleep(10.0)  # polite delay — Dukascopy rate-limits aggressively (30s ban after burst)

            if result.returncode != 0:
                log_error(f"curl failed ({result.returncode}) for {url}")
                time.sleep(10 * (attempt + 1))
                continue

            if not os.path.exists(tmp) or os.path.getsize(tmp) < 100:
                log_error(f"empty/tiny file for {url}")
                return None, 0

            # Check if HTML error page (503 etc)
            with open(tmp, "rb") as f:
                data = f.read()
            if data[:5] == b"<html" or len(data) < 200:
                log_error(f"HTML/error response ({len(data)} bytes) for {url}")
                os.remove(tmp)
                time.sleep(15 * (attempt + 1))
                continue

            with open(tmp, "rb") as f:
                data = f.read()
            os.remove(tmp)

            raw = lzma.decompress(data)
            return raw, len(raw) // 24

        except subprocess.TimeoutExpired:
            log_error(f"timeout for {url}")
            time.sleep(15 * (attempt + 1))
        except Exception as e:
            log_error(f"error for {url}: {e}")
            time.sleep(10 * (attempt + 1))

    log_error(f"FAILED after 3 attempts: {url}")
    return None, 0


def parse_m1(raw, day_start_utc):
    """Parse bi5 M1 bars. 24B/bar = uint32 ts + 4x uint32 price*1000 + float32 vol."""
    n = len(raw) // 24
    dt = np.dtype([("ts", ">u4"), ("o", ">u4"), ("c", ">u4"),
                   ("l", ">u4"), ("h", ">u4"), ("v", ">f4")])
    arr = np.frombuffer(raw, dtype=dt, count=n)
    times = day_start_utc + arr["ts"].astype(np.int64)
    opens = arr["o"].astype(np.float64) / 1000.0
    closes = arr["c"].astype(np.float64) / 1000.0
    vols = arr["v"].astype(np.float64)
    return times, opens, closes, vols


def main():
    global CHECKPOINT_FILE, ERROR_LOG

    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-12-01")
    ap.add_argument("--end", default="2026-08-12")
    ap.add_argument("--out", default="/home/jith/.hermes/profiles/trading/scripts/dukascopy_m1_features.csv")
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--fresh", action="store_true", default=False,
                    help="ignore checkpoint, start fresh")
    args = ap.parse_args()

    out_dir = os.path.dirname(args.out) or "."
    CHECKPOINT_FILE = os.path.join(out_dir, ".dukascopy_checkpoint.json")
    ERROR_LOG = os.path.join(out_dir, "dukascopy_fetch_errors.log")

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")

    # Resume from checkpoint
    ckpt = load_checkpoint()
    completed_set = set(ckpt.get("completed_days", []))
    if completed_set:
        log(f"resuming: {len(completed_set)} days already completed")

    all_t, all_o, all_c, all_v = [], [], [], []
    day = start.normalize()
    missing = 0
    days = 0
    skipped = 0
    total_days = (end - start).days + 1

    while day <= end:
        day_str = day.strftime("%Y-%m-%d")

        if day_str in completed_set:
            skipped += 1
            day += pd.Timedelta(days=1)
            continue

        y, m01, d01 = day.year, day.month - 1, day.day

        # DST offset for Dukascopy NY timestamps
        if y < 2007:
            utc_off = 5
        else:
            def nth_sunday(yy, mm, ww):
                d0 = pd.Timestamp(yy, mm, 1, tz="UTC")
                dow = d0.dayofweek
                first_sun = d0 + pd.Timedelta(days=(6 - dow) % 7)
                return first_sun + pd.Timedelta(weeks=ww - 1)
            dst_start = nth_sunday(y, 3, 2)
            dst_end = nth_sunday(y, 11, 1)
            utc_off = 4 if dst_start <= day < dst_end else 5

        raw, n = day_file(args.symbol, y, m01, d01)
        if raw is None or n == 0:
            missing += 1
        else:
            day_start_ny = day.value // 10**9
            times, opens, closes, vols = parse_m1(raw, day_start_ny + utc_off * 3600)
            all_t.append(times); all_o.append(opens); all_c.append(closes); all_v.append(vols)
            days += 1

        completed_set.add(day_str)
        ckpt["completed_days"] = list(completed_set)
        ckpt["last_date"] = day_str

        done_count = days + missing
        if done_count % 10 == 0 or done_count == total_days:
            el = time.time() - T0
            rate = el / max(done_count, 1)
            eta = rate * (total_days - done_count)
            log(f"progress: {done_count}/{total_days} ({days} ok, {missing} miss, {skipped} skip) "
                f"rate={rate:.1f}s/day ETA={eta/60:.0f}min")

        if done_count % 50 == 0:
            save_checkpoint(ckpt)

        day += pd.Timedelta(days=1)

    save_checkpoint(ckpt)

    if not all_t:
        log("NO DATA — check network/URLs. Aborting.")
        sys.exit(1)

    t = np.concatenate(all_t); o = np.concatenate(all_o)
    c = np.concatenate(all_c); v = np.concatenate(all_v)
    log(f"downloaded {len(t):,} M1 bars over {days} days (missing={missing}, resumed={skipped})")

    df = pd.DataFrame({"time": t, "open": o, "close": c, "vol": v})
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.sort_values("time").drop_duplicates("time").set_index("time")

    # Aggregate to M5
    df["up"] = (df["close"] > df["open"]).astype(float)
    df["up_vol"] = df["vol"] * df["up"]
    df["dn_vol"] = df["vol"] * (1 - df["up"])
    g = df.resample("5min").agg(
        ticks=("vol", "count"),
        up_vol=("up_vol", "sum"),
        dn_vol=("dn_vol", "sum"),
        tot_vol=("vol", "sum"),
    )
    g["dk_delta"] = (g["up_vol"] - g["dn_vol"]) / (g["up_vol"] + g["dn_vol"] + 1e-9)
    g["dk_cvd"] = g["dk_delta"].cumsum()
    g["dk_cvd"] = g["dk_cvd"] * 0.999 ** np.arange(len(g))
    g["dk_vol_rel"] = g["tot_vol"] / (g["tot_vol"].rolling(288, min_periods=1).mean() + 1e-9)
    g = g.reset_index().rename(columns={"index": "time"})

    g.to_csv(args.out, index=False)
    log(f"saved {len(g):,} M5 rows -> {args.out}")
    log(f"cols: {list(g.columns)}")
    log(f"done in {time.time()-T0:.0f}s")

    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        log("checkpoint cleaned up (success)")


if __name__ == "__main__":
    main()
