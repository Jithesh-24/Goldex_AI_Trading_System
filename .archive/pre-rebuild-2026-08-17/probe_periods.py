"""FULL analysis: build period map (M5 bars/period) vs matrix block structure."""
import sys
sys.path.insert(0, "/home/jith/.hermes/profiles/trading/scripts")
import numpy as np
import build_m5_matrix as B

# 1) reproduce build's period map
raw = B.load_raw()
m5 = B.to_m5(raw)
t = m5["time"].values.astype("int64") // 10**9
gaps = np.where(np.diff(t) > 6 * 3600)[0]
bounds = [0] + [int(g) + 1 for g in gaps] + [len(m5)]
periods = [m5.iloc[bounds[i]:bounds[i+1]] for i in range(len(bounds)-1)
           if len(m5.iloc[bounds[i]:bounds[i+1]]) >= 300]
bars_pp = [len(p) for p in periods]
print(f"M5 bars {len(m5)} | periods {len(periods)} | bars total {sum(bars_pp)} "
      f"| min {min(bars_pp)} max {max(bars_pp)}", flush=True)
print(f"expected matrix rows: {sum(bars_pp)*84:,}", flush=True)

# cumulative row offsets of each period in the matrix (84 rows per bar)
cum = [0]
for b in bars_pp:
    cum.append(cum[-1] + b * 84)
print(f"period row offsets: {cum[:5]}...{cum[-3:]}", flush=True)
print(f"last offset == total rows: {cum[-1] == 32491284}", flush=True)

# 2) scan matrix for block boundaries (t < prev) and 6h+ forward gaps
_DAYS = [0,31,28,31,30,31,30,31,31,30,31,30,31]
_CUM = {}
def _cum_days(y):
    if y in _CUM: return _CUM[y]
    days = 0
    for yy in range(1970, y):
        days += 366 if (yy%4==0 and (yy%100!=0 or yy%400==0)) else 365
    _CUM[y] = days
    return days
def to_min(t):
    y=int(t[0:4]); mo=int(t[5:7]); d=int(t[8:10]); h=int(t[11:13]); mi=int(t[14:16])
    cum = _cum_days(y)
    leap = 1 if (y%4==0 and (y%100!=0 or y%400==0)) else 0
    for m in range(1, mo): cum += _DAYS[m] + (1 if m==2 and leap else 0)
    cum += d-1
    return cum*1440 + h*60 + mi

IN = "/home/jith/.hermes/profiles/trading/scripts/gold_features_m5.csv"
with open(IN) as f:
    hdr = f.readline().rstrip("\n").split(",")
    tc = hdr.index("time")

prev = None
row = 0
block_resets = []     # rows where t < prev
fwd_gaps = []         # (row, gap_h) where t > prev and gap >= 6h
with open(IN) as f:
    next(f)
    for line in f:
        t = line.split(",", tc+1)[tc]
        m = to_min(t)
        if prev is not None:
            if m < prev:
                block_resets.append(row)
            elif m - prev >= 6*60:
                fwd_gaps.append((row, (m-prev)/60))
        prev = m
        row += 1
        if row % 8_000_000 == 0:
            print(f"  scanned {row:,} | resets {len(block_resets)} | fwd_gaps {len(fwd_gaps)}", flush=True)

print(f"\nmatrix: resets {len(block_resets)} | fwd_gaps>=6h {len(fwd_gaps)}", flush=True)
# period boundary in matrix = fwd_gap that coincides with a build-period offset
# block count within each fwd_gap-segment should be 83
seg_blocks = []
prev_gap_row = -1
cnt = 0
for r in block_resets:
    cnt += 1
# simpler: count resets between consecutive fwd_gaps
segs = []
last = 0
for r, g in fwd_gaps:
    n = sum(1 for b in block_resets if last < b < r)
    segs.append((r, g, n))
    last = r
print("fwd_gap segments (row, gap_h, resets_before):")
for s in segs[:10]:
    print(f"  {s}")
print("  ...")
for s in segs[-5:]:
    print(f"  {s}")
