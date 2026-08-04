"""Atomic v6/v7.10 matrix builder — cgroup-safe, resumable from permanent sources.

v7.10 INCREMENTAL (2026-08-04): if a built gold_features.csv exists and its
schema still matches the current feature engine, the default fast path computes
features ONLY for seed rows NEWER than the matrix's last timestamp and appends
them. This avoids the ~58 CPU-min full 6.39M-row rebuild (and its 3600s timeouts)
just to add the ~1,440 rows each EOD contributes.

Full 6.39M-row build is performed when: the matrix is missing, --full is forced,
or the schema fingerprint has drifted (sidecar .matrix_schema.json mismatch).

Sources (never deleted):
  gold_features_rally.csv  (4.71M rows, static cache)
  gold_seed.csv            (61k XM bars)

Full-build pipeline (all on-disk, peak RSS < 400MB):
  1. rally subsample (every 3rd bar) streamed to .full_cat.csv
  2. XM features streamed (per-period) APPENDED to .full_cat.csv
  3. GNU external sort by time col (8 cores)
  4. chunked float32 conversion -> gold_features.csv
  5. cleanup temps (only at the very end)

Incremental pipeline:
  1. read last timestamp already in gold_features.csv
  2. compute features on a window of the seed that includes the trailing context
     + the new rows
  3. append ONLY rows strictly newer than the matrix's last timestamp
  4. write matrix.schema.json sidecar with the feature-engine fingerprint
"""
import os, sys, time, json, hashlib, shutil
import numpy as np
import pandas as pd

BASE = "/home/jith/.hermes/profiles/trading/scripts"
sys.path.insert(0, BASE)
import features as F

RALLY_FEAT = f"{BASE}/gold_features_rally.csv"
XM_SEED = f"{BASE}/gold_seed.csv"
CAT = f"{BASE}/.full_cat.csv"
SORTED = f"{BASE}/.full_sorted.csv"
OUT = f"{BASE}/gold_features.csv"
SCHEMA = f"{BASE}/.matrix_schema.json"
TIME_COL = None  # resolved dynamically from the header at build time


def schema_fingerprint():
    """Hash of the current feature engine's public surface. If this differs from
    the sidecar written at last build, the matrix is stale (schema drift) and
    must be fully rebuilt rather than incrementally appended."""
    names = sorted(n for n in dir(F) if not n.startswith("_"))
    h = hashlib.sha256("\n".join(names).encode()).hexdigest()[:16]
    return h


def _count_rows(path):
    with open(path, "rb") as f:
        return sum(1 for _ in f) - 1  # minus header


def _last_matrix_time(OUT):
    """Return the newest 'time' already in the built matrix (header-aware:
    'time' is NOT column 0 in this matrix — it lives at the header position)."""
    # find the time column index from the header
    ti = None
    with open(OUT, "rb") as f:
        hdr = f.readline().decode("utf-8", "replace").strip()
    for i, c in enumerate(hdr.split(",")):
        if c.strip() == "time":
            ti = i
            break
    if ti is None:
        return None
    with open(OUT, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        if size > 8192:
            f.seek(size - 8192)
        tail = f.read().decode("utf-8", "replace").splitlines()
    for ln in reversed(tail):
        s = ln.strip()
        if s and not s.lower().startswith("time,"):
            parts = s.split(",")
            if len(parts) > ti:
                return pd.to_datetime(parts[ti])
    return None


def incremental_append(seed_csv, OUT, t0):
    """Append only seed rows newer than the matrix's last timestamp.
    Returns (n_appended, total_rows)."""
    last_dt = _last_matrix_time(OUT)
    if last_dt is None:
        print("  incremental: could not read matrix last-ts — falling to full build")
        return None

    df = pd.read_csv(seed_csv)
    df["time"] = pd.to_datetime(df["time"])
    new = df[df["time"] > last_dt].copy()
    if len(new) == 0:
        print(f"  incremental: no new bars after {last_dt} — matrix current")
        return 0, _count_rows(OUT)

    # context window needed for rolling features (<300 bars => block dropped)
    ctx = df[df["time"] <= last_dt].tail(1500)
    work = pd.concat([ctx, new], ignore_index=True).sort_values("time").reset_index(drop=True)

    t = work["time"].values.astype("datetime64[s]").astype(np.int64)
    gaps = np.where(np.diff(t) > 6 * 3600)[0]
    bounds = [0] + [int(g) + 1 for g in gaps] + [len(work)]
    periods = [work.iloc[bounds[i]:bounds[i+1]] for i in range(len(bounds) - 1)
               if len(work.iloc[bounds[i]:bounds[i+1]]) >= 300]

    header = list(pd.read_csv(OUT, nrows=0).columns)
    non_feat = set(["time", "target", "fwd_return"]) | set(F.RAW_PRICE_COLS)
    geometry_cols = ["sl_dist_buy", "tp_dist_buy", "sl_dist_sell", "tp_dist_sell",
                     "sl_atr_buy", "sl_atr_sell", "rr_buy", "rr_sell"]

    OUT_tmp = OUT + ".inc_tmp"
    if os.path.exists(OUT_tmp):
        os.remove(OUT_tmp)
    appended = 0
    for p in periods:
        if p["time"].max() <= last_dt:
            continue
        fdf = F._feature_block(p).dropna().reset_index(drop=True)
        if len(fdf) < 100:
            continue
        # market_cols = matrix-header columns that the feature block produces
        # (direction/target/geometry/time are added explicitly below)
        market_cols = [c for c in header if c in fdf.columns]
        atr = fdf["atr_14"].values
        spr = (fdf["spread"].astype(float) / 100.0).values if "spread" in fdf.columns \
            else np.full(len(fdf), F.SPREAD)
        for direction in ("BUY", "SELL"):
            for m in F.SL_MULTS:
                for r in F.TP_RATIOS:
                    sl_dist = np.maximum(atr * m, F.MIN_SL_FLOOR)
                    tp_dist = (sl_dist + spr) * r
                    tdf = F.add_trade_target(fdf, max_bars=F.MAX_TARGET_BARS,
                                             sl_dist=sl_dist, tp_dist=tp_dist, direction=direction)
                    gdf = F.add_geometry_awareness(
                        fdf, sl_dist_buy=sl_dist, tp_dist_buy=tp_dist,
                        sl_dist_sell=sl_dist, tp_dist_sell=tp_dist)
                    out = fdf[market_cols].copy()
                    for c in ("open", "high", "low", "close", "spread"):
                        out[c] = fdf[c].values
                    out["time"] = fdf["time"].values
                    for c in geometry_cols:
                        out[c] = gdf[c].values
                    out["direction"] = 1.0 if direction == "BUY" else 0.0
                    out["target"] = tdf["target"].values
                    # BULLETPROOF: force EXACT matrix column order by construction
                    # (immune to any feature-block ordering drift). Extras -> drop.
                    out = out.reindex(columns=header).dropna().reset_index(drop=True)
                    # keep ONLY genuinely new rows (drop trailing context predating last_dt)
                    out = out[out["time"] > last_dt].reset_index(drop=True)
                    if len(out):
                        out.to_csv(OUT_tmp, mode="a", header=not os.path.exists(OUT_tmp), index=False)
                        appended += len(out)
                    del out, tdf, gdf
        del fdf
        print(f"  incremental: running appended {appended:,} rows", flush=True)

    if appended == 0 or not os.path.exists(OUT_tmp):
        if os.path.exists(OUT_tmp):
            os.remove(OUT_tmp)
        print("  incremental: 0 new feature rows (timestamps did not extend) — matrix unchanged")
        return 0, _count_rows(OUT)

    # float32-normalize to match live matrix dtype, then append atomically.
    # SORT BY TIME: the full build's GNU sort guarantees time-monotonic rows
    # (walk-forward OOF splits depend on it). The incremental append emits rows
    # in geometry-block order, so re-sort by time before appending — keeps the
    # whole matrix time-ordered, exactly like a full rebuild.
    inc = pd.read_csv(OUT_tmp)
    inc["time"] = pd.to_datetime(inc["time"])
    inc = inc.sort_values("time").reset_index(drop=True)
    inc["time"] = inc["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    for c in inc.columns:
        if c != "time":
            inc[c] = inc[c].astype(np.float32)
    # crash-safe: write to temp, fsync-rename semantics -> backup then append
    BACKUP = OUT + ".bak"
    shutil.copyfile(OUT, BACKUP)
    inc.to_csv(OUT, mode="a", header=False, index=False)
    os.remove(OUT_tmp)
    if os.path.exists(BACKUP):
        os.remove(BACKUP)
    total = _count_rows(OUT)
    print(f"  incremental: appended {appended:,} new rows after {last_dt} | matrix ~{total:,} rows ({time.time()-t0:.0f}s)", flush=True)
    return appended, total


def stream_rally_subsample(cat, every=3):
    """Pass 1: collect bar times. Pass 2: write kept rows to cat (header+body)."""
    keep = set()
    for chunk in pd.read_csv(RALLY_FEAT, usecols=["time"], chunksize=1_000_000):
        keep.update(pd.Series(chunk["time"].unique()).sort_values().values[::every])
    first = True
    for chunk in pd.read_csv(RALLY_FEAT, chunksize=500_000):
        m = chunk["time"].isin(keep)
        if m.any():
            chunk[m].to_csv(cat, mode="a", header=first, index=False)
            first = False
        del chunk
    del keep
    return first


def stream_xm_append(seed_csv, cat):
    """Build XM features per-period, APPEND to cat."""
    df = pd.read_csv(seed_csv)
    df["time"] = pd.to_datetime(df["time"])
    t = df["time"].values.astype("datetime64[s]").astype(np.int64)
    gaps = np.where(np.diff(t) > 6 * 3600)[0]
    bounds = [0] + [int(g) + 1 for g in gaps] + [len(df)]
    periods = [p for p in (df.iloc[bounds[i]:bounds[i+1]] for i in range(len(bounds) - 1)) if len(p) >= 300]
    print(f"XM: {len(df)} bars -> {len(periods)} periods", flush=True)
    total = 0
    first = not os.path.exists(cat) or os.path.getsize(cat) == 0
    for p in periods:
        fdf = F._feature_block(p).dropna().reset_index(drop=True)
        if len(fdf) < 100:
            continue
        atr = fdf["atr_14"].values
        spr = (fdf["spread"].astype(float) / 100.0).values if "spread" in fdf.columns else np.full(len(fdf), F.SPREAD)
        market_cols = [c for c in fdf.columns
                       if c not in ("time", "target", "fwd_return") and c not in F.RAW_PRICE_COLS]
        for direction in ("BUY", "SELL"):
            for m in F.SL_MULTS:
                for r in F.TP_RATIOS:
                    sl_dist = np.maximum(atr * m, F.MIN_SL_FLOOR)
                    tp_dist = (sl_dist + spr) * r
                    tdf = F.add_trade_target(fdf, max_bars=F.MAX_TARGET_BARS,
                                             sl_dist=sl_dist, tp_dist=tp_dist, direction=direction)
                    gdf = F.add_geometry_awareness(
                        fdf, sl_dist_buy=sl_dist, tp_dist_buy=tp_dist,
                        sl_dist_sell=sl_dist, tp_dist_sell=tp_dist)
                    out = fdf[market_cols].copy()
                    for c in ("open", "high", "low", "close", "spread"):
                        out[c] = fdf[c].values
                    out["time"] = fdf["time"].values
                    for c in ("sl_dist_buy", "tp_dist_buy", "sl_dist_sell", "tp_dist_sell",
                              "sl_atr_buy", "sl_atr_sell", "rr_buy", "rr_sell"):
                        out[c] = gdf[c].values
                    out["direction"] = 1.0 if direction == "BUY" else 0.0
                    out["target"] = tdf["target"].values
                    out = out.dropna().reset_index(drop=True)
                    out.to_csv(cat, mode="a", header=first, index=False)
                    first = False
                    total += len(out)
                    del out, tdf, gdf
        del fdf
    return total


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--incremental", action="store_true",
                    help="append only NEW bars to existing matrix (v7.10 default when matrix+schema ok)")
    ap.add_argument("--full", action="store_true",
                    help="force full 6.39M-row rebuild (first build, forced schema change)")
    args = ap.parse_args()
    t0 = time.time()

    # ── v7.10 INCREMENTAL FAST-PATH ──
    if (args.incremental or not args.full) and os.path.exists(OUT):
        try:
            with open(SCHEMA) as f:
                fp_ok = (json.load(f).get("fp") == schema_fingerprint())
        except Exception:
            fp_ok = False
        if fp_ok:
            print("═══ ATOMIC MATRIX BUILDER — INCREMENTAL ═══", flush=True)
            res = incremental_append(XM_SEED, OUT, t0)
            if res is not None and not args.full:
                n, total = res
                print(f"\n✅ INCREMENTAL: +{n:,} rows | total ~{total:,} | {time.time()-t0:.0f}s", flush=True)
                return
        else:
            print("  schema drift or no fingerprint — running FULL rebuild", flush=True)

    # ── FULL BUILD ──
    print("═══ ATOMIC FULL MATRIX BUILDER ═══", flush=True)
    for tmp in (CAT, SORTED):
        if os.path.exists(tmp):
            os.remove(tmp)

    nothing = True
    if os.path.exists(RALLY_FEAT):
        nothing = stream_rally_subsample(CAT)
        n_rally = sum(1 for _ in open(CAT)) - 1
        print(f"rally sub: {n_rally:,} rows ({time.time()-t0:.0f}s)", flush=True)
        if nothing:
            raise RuntimeError("rally subsample produced zero rows")

    n_xm = stream_xm_append(XM_SEED, CAT)
    print(f"xm: {n_xm:,} rows ({time.time()-t0:.0f}s)", flush=True)

    header_line = open(CAT).readline().rstrip("\n")
    time_col = header_line.split(",").index("time") + 1
    SORT_TMP = "/home/jith/.hermes/profiles/trading/tmp"
    os.makedirs(SORT_TMP, exist_ok=True)
    subprocess.run(["bash", "-c",
        f"(echo '{header_line}' && tail -n +2 {CAT} | LC_ALL=C sort -t, -k{time_col},{time_col} --parallel=8 --buffer-size=1G --temporary-directory={SORT_TMP}) > {SORTED}"],
        check=True)
    print(f"sorted: {os.path.getsize(SORTED)/1e9:.1f} GB ({time.time()-t0:.0f}s)", flush=True)

    if os.path.exists(OUT):
        os.remove(OUT)
    total = 0
    tb = {0.0: 0, 1.0: 0}
    first = True
    for chunk in pd.read_csv(SORTED, chunksize=400_000):
        for c in chunk.columns:
            if c != "time":
                chunk[c] = chunk[c].astype(np.float32)
        chunk.to_csv(OUT, mode="a", header=first, index=False)
        first = False
        total += len(chunk)
        tb[0.0] += int((chunk["target"] == 0).sum())
        tb[1.0] += int((chunk["target"] == 1).sum())
        del chunk
    os.remove(SORTED); os.remove(CAT)

    # write the schema sidecar so future runs can take the incremental path
    with open(SCHEMA, "w") as f:
        json.dump({"fp": schema_fingerprint(), "built_at": time.time(),
                   "rows": total}, f)
    print(f"\n✅ FINAL: {total:,} rows | balance {tb} | {time.time()-t0:.0f}s", flush=True)
    print(f"   saved: {OUT} | schema fingerprint: {schema_fingerprint()}", flush=True)


if __name__ == "__main__":
    main()