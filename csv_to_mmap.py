#!/usr/bin/env python3
"""
csv_to_mmap.py — Convert 36GB CSV to numpy mmap using pure Python csv reader.
Saves X (features), Y (target), and times for walk-forward splits.
"""
import numpy as np
import time, os, sys, csv, json

BASE = os.path.dirname(os.path.abspath(__file__))
CSV = f"{BASE}/gold_features_m5_full.csv"
MMAP_X = f"{BASE}/train_data_x.npy"
MMAP_Y = f"{BASE}/train_data_y.npy"
MMAP_T = f"{BASE}/train_data_t.npy"
META = f"{BASE}/train_data_meta.json"

sys.path.insert(0, BASE)
from train_ai import FEATURE_EXCLUDE
from features import RAW_PRICE_COLS

def main():
    t0 = time.time()
    
    # Step 1: Read header
    print("1. Reading header...", flush=True)
    with open(CSV, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
    
    exclude = FEATURE_EXCLUDE | RAW_PRICE_COLS
    skip_cols = exclude | {'target', 'direction', 'time'}
    
    feat_idx = [i for i, c in enumerate(header) if c not in skip_cols]
    target_idx = header.index('target')
    time_idx = header.index('time')
    n_features = len(feat_idx)
    print(f"   {len(header)} cols → {n_features} features", flush=True)
    
    # Step 2: Count rows (fast binary scan)
    print("2. Counting rows...", flush=True)
    t_c = time.time()
    n_rows = 0
    with open(CSV, 'rb') as f:
        while True:
            buf = f.read(64 * 1024 * 1024)
            if not buf: break
            n_rows += buf.count(b'\n')
    n_rows -= 1  # header
    print(f"   {n_rows:,} rows ({time.time()-t_c:.1f}s)", flush=True)
    
    # Step 3: Create mmap files
    print(f"3. Creating mmap: X({n_rows},{n_features}) float32 + Y({n_rows}) int8 + T({n_rows}) int64", flush=True)
    X = np.memmap(MMAP_X, dtype=np.float32, mode='w+', shape=(n_rows, n_features))
    Y = np.memmap(MMAP_Y, dtype=np.int8, mode='w+', shape=(n_rows,))
    T = np.memmap(MMAP_T, dtype='datetime64[s]', mode='w+', shape=(n_rows,))
    
    # Step 4: Stream CSV
    print("4. Streaming CSV...", flush=True)
    i = 0
    bx = np.empty((500000, n_features), dtype=np.float32)
    by = np.empty(500000, dtype=np.int8)
    bt = np.empty(500000, dtype='datetime64[s]')
    bi = 0
    
    with open(CSV, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            for j, idx in enumerate(feat_idx):
                try: bx[bi, j] = float(row[idx])
                except: bx[bi, j] = 0.0
            try: by[bi] = int(float(row[target_idx]))
            except: by[bi] = 0
            try: bt[bi] = np.datetime64(row[time_idx][:19], 's')
            except: bt[bi] = np.datetime64('2020-01-01', 's')
            bi += 1
            if bi >= 500000:
                X[i:i+bi] = bx[:bi]
                Y[i:i+bi] = by[:bi]
                T[i:i+bi] = bt[:bi]
                i += bi; bi = 0
                el = time.time() - t0
                eta = (n_rows - i) / (i / el) if i > 0 else 0
                print(f"   {i:,}/{n_rows:,} ({el:.0f}s, ETA {eta:.0f}s = {eta/60:.0f}min)", flush=True)
    if bi > 0:
        X[i:i+bi] = bx[:bi]
        Y[i:i+bi] = by[:bi]
        T[i:i+bi] = bt[:bi]
        i += bi
    
    X.flush(); Y.flush(); T.flush()
    elapsed = time.time() - t0
    print(f"\n   ✅ DONE: {i:,} rows in {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
    
    # Save metadata
    meta = {'n_rows': int(n_rows), 'n_features': int(n_features),
            'features': [header[idx] for idx in feat_idx]}
    with open(META, 'w') as f:
        json.dump(meta, f)
    
    print(f"\n═══ FILES ═══", flush=True)
    for p in [MMAP_X, MMAP_Y, MMAP_T]:
        print(f"   {os.path.basename(p)}: {os.path.getsize(p)/1024**3:.2f} GB", flush=True)
    
    # Speed test
    t1 = time.time()
    _ = np.memmap(MMAP_X, dtype=np.float32, mode='r', shape=(n_rows, n_features))[0:1000]
    print(f"\n   Load speed: {time.time()-t1:.3f}s (vs 32h CSV = {32*3600/(time.time()-t1):.0f}x faster)", flush=True)

if __name__ == '__main__':
    main()
