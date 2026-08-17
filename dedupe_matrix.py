#!/usr/bin/env python3
"""Dedupe gold_features.csv after the 2026-08-05 double-append bug.

The old _last_matrix_time read the last LINE (a live-outcome row with an OLD
timestamp), so the second incremental run re-appended every seed bar after
that stale cut → ~16k duplicated seed rows + duplicated live outcome rows.

Dedupe key: (time, direction, sl_dist_buy, tp_dist_buy) — uniquely identifies
one geometry variant; 48 per bar. Streams with a seen-key set (6.4M keys ~
400MB) so the 8.3M×108 matrix never loads fully. Writes to .deduped then
atomically renames over the live matrix.
"""
import os, sys, time
import pandas as pd

OUT = "/home/jith/.hermes/profiles/trading/scripts/gold_features.csv"
TMP = OUT + ".deduped"
t0 = time.time()

header = list(pd.read_csv(OUT, nrows=0).columns)
ti = header.index("time")
di = header.index("direction")
si = header.index("sl_dist_buy")
pi = header.index("tp_dist_buy")

seen = set()
total = 0
dups = 0
first = True
for chunk in pd.read_csv(OUT, chunksize=400_000):
    t = chunk["time"].astype(str)
    keys = (t + "|" + chunk["direction"].astype(str) + "|"
            + chunk["sl_dist_buy"].astype(str) + "|" + chunk["tp_dist_buy"].astype(str))
    mask = ~keys.isin(seen)
    dup_count = int((~mask).sum())
    dups += dup_count
    keep = chunk[mask]
    seen.update(keys[mask].tolist())
    keep.to_csv(TMP, mode="a", header=first, index=False)
    first = False
    total += len(keep)
    if total % 1_000_000 < 400_000:
        print(f"  {total:,} kept | {dups:,} dups | {time.time()-t0:.0f}s", flush=True)

os.replace(TMP, OUT)
print(f"\n✅ DEDUPE: kept {total:,} | removed {dups:,} dup rows | {time.time()-t0:.0f}s", flush=True)
