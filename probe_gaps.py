"""Fast trace: manual timestamp parsing, no strptime."""
IN = "/home/jith/.hermes/profiles/trading/scripts/gold_features_m5.csv"

_DAYS = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

_CUM = {}
def _cum_days(y):
    """Days from 1970-01-01 to Jan 1 of year y (cached)."""
    if y in _CUM:
        return _CUM[y]
    y0 = 1970
    days = 0
    for yy in range(y0, y):
        days += 366 if (yy % 4 == 0 and (yy % 100 != 0 or yy % 400 == 0)) else 365
    _CUM[y] = days
    return days

def to_min(t):
    """'YYYY-MM-DD HH:MM:SS' -> minutes since epoch (fast, no strptime)."""
    y = int(t[0:4]); mo = int(t[5:7]); d = int(t[8:10])
    h = int(t[11:13]); mi = int(t[14:16])
    cum = _cum_days(y)
    leap = 1 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 0
    for m in range(1, mo):
        cum += _DAYS[m] + (1 if m == 2 and leap else 0)
    cum += d - 1
    return (cum * 1440) + h * 60 + mi

with open(IN) as f:
    header = f.readline()
    hdr = header.rstrip("\n").split(",")
    tc = hdr.index("time")

prev_m = None
row = 0
block_len = 0
n_blocks = 0
period_gaps = []
with open(IN) as f:
    next(f)
    for line in f:
        t = line.split(",", tc + 1)[tc]
        m = to_min(t)
        if prev_m is not None:
            if m < prev_m:
                n_blocks += 1
                block_len = 0
            else:
                gap = (m - prev_m) / 60.0
                if gap >= 6:
                    period_gaps.append((row, gap, n_blocks + 1, block_len + 1))
                    n_blocks = 0
                    block_len = 0
        block_len += 1
        prev_m = m
        row += 1
        if row % 5_000_000 == 0:
            print(f"  scanned {row:,} | periods so far {len(period_gaps)}", flush=True)

print(f"total rows: {row:,} | periods: {len(period_gaps)}", flush=True)
for i, (r, g, bl, sz) in enumerate(period_gaps[:6]):
    print(f"  period {i}: end row {r:,} | gap {g:.1f}h | blocks {bl} | last size {sz}", flush=True)
print("  ...")
for i, (r, g, bl, sz) in enumerate(period_gaps[-4:], start=max(0, len(period_gaps)-4)):
    print(f"  period {i}: end row {r:,} | gap {g:.1f}h | blocks {bl} | last size {sz}", flush=True)
