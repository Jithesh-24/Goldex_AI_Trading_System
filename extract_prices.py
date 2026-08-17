#!/usr/bin/env python3
"""
extract_prices.py — Extract open/high/low/close from last 10M rows of CSV.
Fast: reads only 4 columns via csv.reader (no pandas overhead).
"""
import csv
import numpy as np
import time, os, sys

BASE = os.path.dirname(os.path.abspath(__file__))
CSV = f"{BASE}/gold_features_m5_full.csv"
TAIL = 10_000_000
TARGET_COLS = ['open', 'high', 'low', 'close']

def main():
    t0 = time.time()
    
    # Get header and find column indices
    with open(CSV) as f:
        reader = csv.reader(f)
        header = next(reader)
    
    col_idx = [header.index(c) for c in TARGET_COLS]
    print(f"CSV columns: {TARGET_COLS} at indices {col_idx}", flush=True)
    
    # Use known row count (avoids re-reading 36GB CSV)
    import json as _json
    meta = _json.load(open(f"{BASE}/train_data_meta.json"))
    row_count = meta['n_rows']
    
    start = max(0, row_count - TAIL)
    n_tail = row_count - start
    print(f"Extracting rows {start:,}–{row_count:,} ({n_tail:,} rows)", flush=True)
    
    # Create numpy array
    prices = np.zeros((n_tail, 4), dtype=np.float64)
    
    # Stream CSV, skip to start, extract last 10M rows
    with open(CSV) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        
        for i, row in enumerate(reader):
            if i < start:
                continue
            idx = i - start
            for j, ci in enumerate(col_idx):
                try:
                    prices[idx, j] = float(row[ci])
                except (ValueError, IndexError):
                    prices[idx, j] = 0.0
            
            if idx % 1_000_000 == 0 and idx > 0:
                el = time.time() - t0
                rate = idx / max(el, 1)
                eta = (n_tail - idx) / max(rate, 1)
                print(f"  {idx:,}/{n_tail:,} ({el:.0f}s, ~{eta:.0f}s remaining)", flush=True)
    
    # Save
    out = f"{BASE}/prices_tail.npy"
    np.save(out, prices)
    
    elapsed = time.time() - t0
    print(f"\n═══ DONE ═══ ({elapsed:.0f}s = {elapsed/60:.1f}min)", flush=True)
    print(f"Saved: {out} ({os.path.getsize(out)/1024**3:.1f} GB)", flush=True)
    print(f"Shape: {prices.shape} ({TARGET_COLS})", flush=True)
    print(f"Sample: {prices[-1]}", flush=True)

if __name__ == '__main__':
    main()
