"""v8 M5 MATRIX BUILDER — 6yr raw M1 → M5 base timeframe + MFE/MFA excursion.

WHY M5 (user mandate 2026-08-07):
  M1 carries noise; M5 gives the same market structure with 5× less noise.
  M15 context layer (features.py add_htf_context) keeps the 15-min structure
  the user wants caught. Trade horizon stays the same real-time span as v7
  (MAX_TARGET_BARS = 36 M5 bars = 180 min, matching the proven direction
  prior horizon).

WHAT'S NEW vs v7 (build_full_matrix.py):
  1. RAW 6yr M1 (gold_seed_full6yr.csv + gold_seed.csv) resampled to M5 OHLCV
     (open=first, high=max, low=min, close=last, vol=sum, spread=mean).
  2. Every geometry row carries mfe_atr / mfa_atr — the max favorable and
     max adverse excursion (in ATR units) over the path UP TO first-touch
     resolution. These are the institutional placement-learning signals:
     "in this regime winners run X ATR favorable, losers dip Y ATR adverse".
  3. Output: gold_features_m5.csv (same 48-row geometry block per timestamp:
     2 dir × 6 SL × 4 TP) + .matrix_schema_m5.json fingerprint.

MEMORY: streamed per contiguous period (gap > 6h), same as v7. The M5 matrix
is ~9.3M rows × 110 cols → streamed chunks, peak ~600MB. Sorted by time via
GNU sort (time-monotonic for walk-forward splits).

Usage: python build_m5_matrix.py [--incremental] [--full]
"""
import os, sys, time, json, shutil, subprocess
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F

BASE = "/home/jith/.hermes/profiles/trading/scripts"
RAW_6YR = f"{BASE}/gold_seed_merged_full6yr.csv"  # FULL merged: 2.46M M1 bars, 81/81 months 2019-12 → 2026-08
RAW_CUR = f"{BASE}/gold_seed.csv"            # current XM seed (has newest bars)
OUT = f"{BASE}/gold_features_m5.csv"
SCHEMA = f"{BASE}/.matrix_schema_m5.json"
CAT = f"{BASE}/.full_cat_m5.csv"
SORTED = f"{BASE}/.full_sorted_m5.csv"
SORT_TMP = "/home/jith/.hermes/profiles/trading/tmp"
MAX_TARGET_BARS = 36          # 36 M5 bars = 180 min (matches direction prior)
MIN_SL_FLOOR = 0.30

def load_raw():
    """Concatenate raw M1 seeds, dedupe by time (newest source wins), sort."""
    a = pd.read_csv(RAW_6YR)
    a["time"] = pd.to_datetime(a["time"])
    # 6yr seed has no spread/real_volume — synthesize spread in POINTS
    # (engine/XM convention: /100 → $). F.SPREAD is $0.20 → 20 points.
    if "spread" not in a.columns:
        a["spread"] = F.SPREAD * 100.0
    if "real_volume" not in a.columns:
        a["real_volume"] = a["tick_volume"].fillna(0.0)
    last6 = a["time"].iloc[-1]
    b = pd.read_csv(RAW_CUR)
    b["time"] = pd.to_datetime(b["time"])
    b = b[b["time"] > last6]
    keep = ["time","open","high","low","close","tick_volume","spread","real_volume"]
    a = a[keep]; b = b[[c for c in keep if c in b.columns]]
    df = pd.concat([a, b], ignore_index=True).drop_duplicates(subset="time", keep="last")
    df = df.sort_values("time").reset_index(drop=True)
    return df

def to_m5(df):
    """Resample M1 → M5 OHLCV."""
    s = df.set_index("time")
    r = s.resample("5min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
        "tick_volume": "sum", "spread": "mean", "real_volume": "sum",
    }).dropna(subset=["open", "close"])
    r = r.reset_index()
    r["spread"] = r["spread"].fillna(F.SPREAD)
    return r

def _feature_block_m5(df_raw):
    """Feature-engineer one contiguous M5 period (mirrors F._feature_block)."""
    df = df_raw.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    keep = [c for c in ["time","open","high","low","close","tick_volume","spread","real_volume"] if c in df.columns]
    df = df[keep]
    df = F.add_returns(df); df = F.add_atr(df); df = F.add_bb_width(df)
    df = F.add_ewma_vol(df); df = F.add_rsi(df); df = F.add_macd(df)
    df = F.add_stoch(df); df = F.add_price_shape(df); df = F.add_structure(df)
    df = F.add_volume(df); df = F.add_regime(df)
    df = F.add_institutional_levels(df)
    df = F.add_scale_free(df)
    df = F.add_htf_context(df)     # v8: now includes M15 layer
    df = F.add_session_clock(df)
    df = F.add_event_proximity(df)
    df = F.add_order_flow(df)
    df = F.add_strategy_playbook(df)
    # v8.8: position-state columns must exist in matrix rows (0.0 baseline
    # for historical bars — live engine injects real values at prediction,
    # live outcome rows carry real values via merge_live_outcomes_appended).
    df = F.add_position_state(df)
    return df

# v8.9 TICK CLOSED-LOOP (2026-08-12): attach the Dukascopy tick block to a
# feature dataframe. BOTH the full build and incremental append must emit the
# SAME tick cols (imb_300s/vol_rel/cvd) or the matrix ↔ model feature set
# silently diverges. vol_rel from features.py (tick_volume ratio) is
# REPLACED by the Dukascopy activity-burst definition — training and live
# engine agree on the same formula.
TICK_COLS = ["imb_300s", "vol_rel", "cvd"]
_DK_TICK = None  # lazy-loaded tick block (indexed by UTC time)

def _attach_tick_block(df):
    """Merge dk tick cols by time (UTC). No-op if the dk file is missing —
    full-history rebuilds keep whatever vol_rel features.py already made."""
    global _DK_TICK
    if _DK_TICK is None:
        _p = os.path.join(os.path.dirname(OUT), "dukascopy_m1_features.csv")
        if os.path.exists(_p):
            try:
                _dk = pd.read_csv(_p, parse_dates=["time"])
                _dk["time"] = pd.to_datetime(_dk["time"], utc=True)
                _DK_TICK = _dk.set_index("time").sort_index()[TICK_COLS]
            except Exception as _e:
                print(f"tick-block load warn: {_e}")
                _DK_TICK = False
        else:
            _DK_TICK = False
    if _DK_TICK is False:
        return df
    df["time"] = pd.to_datetime(df["time"], utc=True)
    for c in TICK_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])
    df = df.merge(_DK_TICK, left_on="time", right_index=True, how="left")
    for c in TICK_COLS:
        df[c] = df[c].ffill(limit=3).fillna(0.0)
    return df

def build(out=OUT, incremental=True):
    t0 = time.time()
    # ── incremental fast path: only if schema fingerprint unchanged ──
    if incremental and os.path.exists(out):
        try:
            with open(SCHEMA) as f:
                fp_ok = json.load(f).get("fp") == schema_fingerprint()
        except Exception:
            fp_ok = False
        if fp_ok:
            print("═══ M5 BUILDER — INCREMENTAL ═══", flush=True)
            n = incremental_append(out, t0)
            print(f"✅ incremental: +{n:,} | {time.time()-t0:.0f}s", flush=True)
            return

    print("═══ M5 FULL BUILD ═══", flush=True)
    raw = load_raw()
    print(f"raw M1: {len(raw):,} bars ({raw['time'].iloc[0]} → {raw['time'].iloc[-1]})", flush=True)
    m5 = to_m5(raw)
    print(f"M5: {len(m5):,} bars", flush=True)
    del raw

    # split into contiguous periods (gap > 6h)
    t = m5["time"].values.astype("datetime64[s]").astype(np.int64)
    gaps = np.where(np.diff(t) > 6 * 3600)[0]
    bounds = [0] + [int(g) + 1 for g in gaps] + [len(m5)]
    periods = [m5.iloc[bounds[i]:bounds[i+1]] for i in range(len(bounds)-1)
               if len(m5.iloc[bounds[i]:bounds[i+1]]) >= 300]
    print(f"periods: {len(periods)}", flush=True)

    first = True
    for pi, p in enumerate(periods):
        fdf = _feature_block_m5(p).dropna().reset_index(drop=True)
        # v8.9 TICK CLOSED-LOOP: full build emits the SAME tick cols as
        # incremental + live engine (imb_300s/vol_rel/cvd).
        fdf = _attach_tick_block(fdf)
        if len(fdf) < 100:
            continue
        atr = fdf["atr_14"].values
        spr = (fdf["spread"].astype(float) / 100.0).values
        market_cols = [c for c in fdf.columns
                       if c not in ("time","target","fwd_return","mfe_atr","mfa_atr")
                       and c not in F.RAW_PRICE_COLS]
        for direction in ("BUY", "SELL"):
            for m in F.SL_MULTS:
                for r in F.TP_RATIOS:
                    sl_dist = np.maximum(atr * m, MIN_SL_FLOOR)
                    tp_dist = (sl_dist + spr) * r
                    tdf = F.add_trade_target(fdf, max_bars=MAX_TARGET_BARS,
                                             sl_dist=sl_dist, tp_dist=tp_dist,
                                             direction=direction)
                    gdf = F.add_geometry_awareness(
                        fdf, sl_dist_buy=sl_dist, tp_dist_buy=tp_dist,
                        sl_dist_sell=sl_dist, tp_dist_sell=tp_dist)
                    out_f = fdf[market_cols].copy()
                    for c in ("open","high","low","close","spread"):
                        out_f[c] = fdf[c].values
                    out_f["time"] = fdf["time"].values
                    for c in ("sl_dist_buy","tp_dist_buy","sl_dist_sell","tp_dist_sell",
                              "sl_atr_buy","sl_atr_sell","rr_buy","rr_sell"):
                        out_f[c] = gdf[c].values
                    out_f["direction"] = 1.0 if direction == "BUY" else 0.0
                    out_f["target"] = tdf["target"].values
                    out_f["mfe_atr"] = tdf["mfe_atr"].values
                    out_f["mfa_atr"] = tdf["mfa_atr"].values
                    out_f.to_csv(CAT, mode="a", header=first, index=False)
                    first = False
                    del out_f, tdf, gdf
        del fdf
        print(f"  period {pi+1}/{len(periods)} done ({time.time()-t0:.0f}s)", flush=True)

    # sort by time — CAT is ALREADY time-ordered (periods processed oldest→newest,
    # M5 resample sorted). External sort is a redundant safety; skip when
    # MATRIX_SKIP_SORT=1 to cut disk peak (2 copies vs 3).
    if os.environ.get("MATRIX_SKIP_SORT") == "1":
        print("skip-sort: CAT is time-ordered; streaming directly", flush=True)
        header_line = open(CAT).readline().rstrip("\n")
        SORTED = CAT
    else:
        header_line = open(CAT).readline().rstrip("\n")
        time_col = header_line.split(",").index("time") + 1
        os.makedirs(SORT_TMP, exist_ok=True)
        if os.path.exists(SORTED):
            os.remove(SORTED)
        subprocess.run(["bash", "-c",
            f"(echo '{header_line}' && tail -n +2 {CAT} | LC_ALL=C sort -t, -k{time_col},{time_col} --parallel=8 --buffer-size=1G --temporary-directory={SORT_TMP}) > {SORTED}"],
            check=True)
    print(f"sorted: {os.path.getsize(SORTED)/1e9:.1f} GB ({time.time()-t0:.0f}s)", flush=True)

    if os.path.exists(out):
        os.remove(out)
    total = 0
    tb = {0.0: 0, 1.0: 0}
    first = True
    for chunk in pd.read_csv(SORTED, chunksize=400_000):
        for c in chunk.columns:
            if c != "time":
                chunk[c] = chunk[c].astype(np.float32)
        chunk.to_csv(out, mode="a", header=first, index=False)
        first = False
        total += len(chunk)
        tb[0.0] += int((chunk["target"] == 0).sum())
        tb[1.0] += int((chunk["target"] == 1).sum())
        del chunk
    for f in {SORTED, CAT}:
        if os.path.exists(f):
            os.remove(f)
    with open(SCHEMA, "w") as f:
        json.dump({"fp": schema_fingerprint(), "built_at": time.time(), "rows": total}, f)
    print(f"✅ FINAL: {total:,} rows | balance {tb} | {time.time()-t0:.0f}s", flush=True)

def incremental_append(out, t0):
    """Append M5 rows newer than matrix's last ts (mirrors v7 flow)."""
    from build_full_matrix import _last_matrix_time
    last_dt = _last_matrix_time(out)
    if last_dt is None:
        return 0
    raw = load_raw()
    new = raw[raw["time"] > last_dt]
    if len(new) == 0:
        return 0
    ctx = raw[raw["time"] <= last_dt].tail(1500)
    work = pd.concat([ctx, new], ignore_index=True).sort_values("time").reset_index(drop=True)
    m5 = to_m5(work)
    t = m5["time"].values.astype("datetime64[s]").astype(np.int64)
    gaps = np.where(np.diff(t) > 6 * 3600)[0]
    bounds = [0] + [int(g) + 1 for g in gaps] + [len(m5)]
    periods = [m5.iloc[bounds[i]:bounds[i+1]] for i in range(len(bounds)-1)
               if len(m5.iloc[bounds[i]:bounds[i+1]]) >= 300]
    header = list(pd.read_csv(out, nrows=0).columns)
    non_feat = set(["time","target","fwd_return","mfe_atr","mfa_atr"]) | set(F.RAW_PRICE_COLS)
    geometry_cols = ["sl_dist_buy","tp_dist_buy","sl_dist_sell","tp_dist_sell",
                     "sl_atr_buy","sl_atr_sell","rr_buy","rr_sell"]
    # v8.9 TICK CLOSED-LOOP (2026-08-12): new rows must carry the tick block
    # (imb_300s/vol_rel/cvd) or reindex->dropna silently DROPS them and the
    # matrix stops growing. _attach_tick_block does the merge (no-op if dk
    # file missing). The EOD loop refreshes the dk window (--recent-days 10)
    # BEFORE this incremental build so today's bars have tick features.
    OUT_tmp = out + ".inc_tmp"
    if os.path.exists(OUT_tmp):
        os.remove(OUT_tmp)
    appended = 0
    for p in periods:
        if p["time"].max() <= last_dt:
            continue
        fdf = _feature_block_m5(p).dropna().reset_index(drop=True)
        if len(fdf) < 100:
            continue
        market_cols = [c for c in header if c in fdf.columns]
        atr = fdf["atr_14"].values
        spr = (fdf["spread"].astype(float) / 100.0).values
        for direction in ("BUY","SELL"):
            for m in F.SL_MULTS:
                for r in F.TP_RATIOS:
                    sl_dist = np.maximum(atr * m, MIN_SL_FLOOR)
                    tp_dist = (sl_dist + spr) * r
                    tdf = F.add_trade_target(fdf, max_bars=MAX_TARGET_BARS,
                                             sl_dist=sl_dist, tp_dist=tp_dist,
                                             direction=direction)
                    gdf = F.add_geometry_awareness(
                        fdf, sl_dist_buy=sl_dist, tp_dist_buy=tp_dist,
                        sl_dist_sell=sl_dist, tp_dist_sell=tp_dist)
                    o = fdf[market_cols].copy()
                    for c in ("open","high","low","close","spread"):
                        o[c] = fdf[c].values
                    o["time"] = fdf["time"].values
                    for c in geometry_cols:
                        o[c] = gdf[c].values
                    o["direction"] = 1.0 if direction == "BUY" else 0.0
                    o["target"] = tdf["target"].values
                    o["mfe_atr"] = tdf["mfe_atr"].values
                    o["mfa_atr"] = tdf["mfa_atr"].values
                    # v8.9 TICK CLOSED-LOOP: attach tick block to the new rows
                    o = _attach_tick_block(o)
                    o = o.reindex(columns=header).dropna().reset_index(drop=True)
                    o = o[o["time"] > last_dt].reset_index(drop=True)
                    if len(o):
                        o.to_csv(OUT_tmp, mode="a", header=not os.path.exists(OUT_tmp), index=False)
                        appended += len(o)
                    del o, tdf, gdf
        del fdf
    if appended == 0 or not os.path.exists(OUT_tmp):
        if os.path.exists(OUT_tmp):
            os.remove(OUT_tmp)
        return 0
    inc = pd.read_csv(OUT_tmp)
    inc["time"] = pd.to_datetime(inc["time"])
    inc = inc.sort_values("time").reset_index(drop=True)
    inc["time"] = inc["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    for c in inc.columns:
        if c != "time":
            inc[c] = inc[c].astype(np.float32)
    BACKUP = out + ".bak"
    shutil.copyfile(out, BACKUP)
    inc.to_csv(out, mode="a", header=False, index=False)
    os.remove(OUT_tmp)
    if os.path.exists(BACKUP):
        os.remove(BACKUP)
    return appended

def schema_fingerprint():
    """Fingerprint the feature schema so incremental builds detect drift.
    v8.4c: include the geometry grid — changing SL_MULTS/TP_RATIOS changes
    rows/bar (48→84); incremental append must NOT mix row counts."""
    try:
        hdr = list(pd.read_csv(CAT if os.path.exists(CAT) else OUT, nrows=0).columns)
    except Exception:
        hdr = []
    grid = f"{len(F.SL_MULTS)}x{len(F.TP_RATIOS)}"
    return f"{len(hdr)}|{hdr[:5]}|mfe_mfa_m5|grid{grid}"

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--incremental", action="store_true")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    build(OUT, incremental=(args.incremental or not args.full))
