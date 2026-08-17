"""Diagnose: dump resets + 6h+ gaps with sizes for first N rows."""
import sys

IN = "/home/jith/.hermes/profiles/trading/scripts/gold_features_m5.csv"
NMAX = 120_000

with open(IN) as f:
    header = f.readline()
    hdr = header.rstrip("\n").split(",")
    tc = hdr.index("time")

prev_t = None
row = 0
block_len = 0
block_sizes = []
period_start_row = 0
n_blocks = 0
last_print = 0
with open(IN) as f:
    next(f)
    for line in f:
        if row >= NMAX:
            break
        t = line.split(",", tc + 1)[tc]
        if prev_t is not None:
            if t < prev_t:
                block_sizes.append(block_len)
                block_len = 0
                n_blocks += 1
                if n_blocks <= 6 or n_blocks % 20 == 0:
                    print(f"  RESET at row {row}: closed block #{n_blocks} size {block_sizes[-1]}", flush=True)
            else:
                import datetime as _dt
                gap = (_dt.datetime.strptime(t, "%Y-%m-%d %H:%M:%S") -
                       _dt.datetime.strptime(prev_t, "%Y-%m-%d %H:%M:%S")).total_seconds() / 3600
                if gap >= 6:
                    print(f"  GAP {gap:.1f}h at row {row} (after block #{n_blocks}, cur len {block_len}, total blocks {len(block_sizes)})", flush=True)
        block_len += 1
        prev_t = t
        row += 1
print(f"scanned {row} rows | blocks closed: {len(block_sizes)} | first 10 sizes: {block_sizes[:10]} | last 3: {block_sizes[-3:]}")
