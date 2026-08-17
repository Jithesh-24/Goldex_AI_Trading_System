#!/usr/bin/env python3
"""
extract_prices_fast.py — Extract open/high/low/close from last 10M rows using mmap-like seek.
Reads CSV in chunks, extracting only 4 columns.
"""
import numpy as np
import time, os, sys, json

BASE = os.path.dirname(os.path.abspath(__file__))
CSV = f"{BASE}/gold_features_m5_full.csv"
TAIL = 10_000_000
COL_INDICES = [102, 103, 104, 105]  # open, high, low, close (0-indexed)

def main():
    t0 = time.time()
    
    meta = json.load(open(f"{BASE}/train_data_meta.json"))
    total = meta['n_rows']
    start = max(0, total - TAIL)
    n_tail = total - start
    
    print(f"Extracting OHLC from rows {start:,}–{total:,} ({n_tail:,} rows)", flush=True)
    print(f"CSV columns: open(102), high(103), low(104), close(105)", flush=True)
    
    prices = np.zeros((n_tail, 4), dtype=np.float64)
    
    with open(CSV, 'rb') as f:
        # Skip header
        f.readline()
        
        # Skip to start row
        skip_count = 0
        while skip_count < start:
            f.readline()
            skip_count += 1
            if skip_count % 5_000_000 == 0:
                print(f"  Skipping: {skip_count:,}/{start:,}", flush=True)
        
        print(f"  Skipped {skip_count:,} rows, extracting...", flush=True)
        
        # Extract
        valid = 0
        for i in range(n_tail):
            line = f.readline()
            if not line:
                break
            
            parts = line.split(b',')
            try:
                for j, ci in enumerate(COL_INDICES):
                    prices[valid, j] = float(parts[ci])
                valid += 1
            except (ValueError, IndexError):
                continue
            
            if valid % 1_000_000 == 0:
                el = time.time() - t0
                print(f"  {valid:,}/{n_tail:,} ({el:.0f}s)", flush=True)
    
    # Trim to valid rows
    prices = prices[:valid]
    
    out = f"{BASE}/prices_tail.npy"
    np.save(out, prices)
    
    elapsed = time.time() - t0
    print(f"\n═══ DONE ═══ ({elapsed:.0f}s = {elapsed/60:.1f}min)", flush=True)
    print(f"Saved: {out} ({os.path.getsize(out)/1024**3:.2f} GB)", flush=True)
    print(f"Shape: {prices.shape}", flush=True)
    print(f"Close range: {prices[:,3].min():.2f} – {prices[:,3].max():.2f}", flush=True)
    print(f"Close last: {prices[-1,3]:.2f}", flush=True)

if __name__ == '__main__':
    main()
