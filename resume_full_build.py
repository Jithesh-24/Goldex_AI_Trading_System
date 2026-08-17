#!/usr/bin/env python3
"""Resume the full matrix build from the sort step.

The --full build crashed at line 306 with NameError: subprocess (import was
missing — now fixed in build_full_matrix.py). The expensive rally subsample
and XM append are already done in .full_cat.csv (12.5GB, unsorted). This
script resumes: GNU sort → float32 convert → schema sidecar → cleanup.
Identical to build_full_matrix.py lines 302-333.
"""
import os, sys, time, json, hashlib, subprocess
import numpy as np
import pandas as pd

BASE = "/home/jith/.hermes/profiles/trading/scripts"
sys.path.insert(0, BASE)
import features as F

def schema_fingerprint():
    names = sorted(n for n in dir(F) if not n.startswith("_"))
    return hashlib.sha256("\n".join(names).encode()).hexdigest()[:16]

CAT = f"{BASE}/.full_cat.csv"
SORTED = f"{BASE}/.full_sorted.csv"
OUT = f"{BASE}/gold_features.csv"
SCHEMA = f"{BASE}/.matrix_schema.json"
SORT_TMP = "/home/jith/.hermes/profiles/trading/tmp"
t0 = time.time()

assert os.path.exists(CAT), f"{CAT} missing — cannot resume"

header_line = open(CAT).readline().rstrip("\n")
time_col = header_line.split(",").index("time") + 1
os.makedirs(SORT_TMP, exist_ok=True)
print("sorting...", flush=True)
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

with open(SCHEMA, "w") as f:
    json.dump({"fp": schema_fingerprint(),
               "built_at": time.time(), "rows": total}, f)
print(f"\n✅ RESUMED FINAL: {total:,} rows | balance {tb} | {time.time()-t0:.0f}s", flush=True)
print(f"   saved: {OUT}", flush=True)
