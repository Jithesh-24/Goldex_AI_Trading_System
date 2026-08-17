#!/usr/bin/env python3
"""v8.8 TRANSITION STEP 1 — streaming column-add of position-state features.

Adds day_pnl / streak / trades_today (0.0 baseline) to gold_features_m5.csv
in pure streaming fashion (NO recompute — 33.7GB pure I/O, ~15-30 min).
The columns are inserted right BEFORE "open" — exactly where
_feature_block_m5 emits them (appended after the last market feature, with
open/high/low/close/spread appended after market_cols by the builder).

After the add, updates .matrix_schema_m5.json fp to the new value so the
next EOD incremental build stays on the fast path (no full rebuild).

Engine safety: the live engine never reads the matrix CSV (it reads
models/*.json + tick state), so rewriting it does NOT disturb the engine.
"""
import os, sys, time, json, csv, shutil

BASE = "/home/jith/.hermes/profiles/trading/scripts"
OUT = f"{BASE}/gold_features_m5.csv"
TMP = f"{OUT}.v88new"
SCHEMA = f"{BASE}/.matrix_schema_m5.json"
NEW_COLS = ["day_pnl", "streak", "trades_today"]
INSERT_VAL = "0.0"  # baseline: fresh day, no trades yet

t0 = time.time()
n = 0

with open(OUT, "r", newline="") as f_in, open(TMP, "w", newline="") as f_out:
    reader = csv.reader(f_in)
    writer = csv.writer(f_out)
    header = next(reader)
    assert "open" in header, f"'open' not in header ({len(header)} cols)"
    pos = header.index("open")
    assert len(header) == 116, f"unexpected header len {len(header)} (expected 116)"
    new_header = header[:pos] + NEW_COLS + header[pos:]
    assert len(new_header) == 119
    assert new_header[pos:pos + 3] == NEW_COLS
    writer.writerow(new_header)
    for row in reader:
        if len(row) != len(header):
            print(f"⚠️ row {n}: {len(row)} fields != {len(header)} — aborting (no partial write)")
            sys.exit(1)
        writer.writerow(row[:pos] + [INSERT_VAL, INSERT_VAL, INSERT_VAL] + row[pos:])
        n += 1
        if n % 2_000_000 == 0:
            print(f"  {n:,} rows ({time.time()-t0:.0f}s)", flush=True)

os.replace(TMP, OUT)
print(f"✅ column-add done: {n:,} rows × {len(new_header)} cols in {time.time()-t0:.0f}s")

# verify header
import pandas as pd
hdr = list(pd.read_csv(OUT, nrows=0).columns)
assert len(hdr) == 119 and hdr[pos:pos + 3] == NEW_COLS, hdr[pos:pos + 6]
print(f"✅ header verified: ...{hdr[pos-2:pos+5]}")

# update schema sidecar fp (mirror of build_m5_matrix.schema_fingerprint)
grid = "6x7"  # len(F.SL_MULTS) x len(F.TP_RATIOS) — unchanged by this transition
fp = f"{len(hdr)}|{hdr[:5]}|mfe_mfa_m5|grid{grid}"
with open(SCHEMA, "w") as f:
    json.dump({"fp": fp, "built_at": time.time(), "rows": n}, f, indent=2)
print(f"✅ schema sidecar updated: fp={fp}")
