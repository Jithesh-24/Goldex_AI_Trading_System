"""Fast probe: raw byte-scan of time column. Layout inference only."""
import sys

IN = "/home/jith/.hermes/profiles/trading/scripts/gold_features_m5.csv"

with open(IN, "rb") as f:
    header = f.readline().decode()
hdr = header.rstrip("\n").split(",")
time_col = hdr.index("time")

resets = []
first_ts = last_ts = None
prev_t = None
row_idx = 0
n = 0
asc = 0
BUF = 64 * 1024 * 1024

with open(IN, "rb") as f:
    f.readline()  # header
    leftover = b""
    while True:
        chunk = f.read(BUF)
        if not chunk:
            break
        data = leftover + chunk
        lines = data.split(b"\n")
        leftover = lines.pop()
        for line in lines:
            if not line:
                continue
            # split at time_col-1 commas: walk bytes counting commas
            fields = line.split(b",")
            t = fields[time_col].decode()
            n += 1
            if first_ts is None:
                first_ts = t
            last_ts = t
            if prev_t is not None:
                if t < prev_t:
                    resets.append(n)
                elif t > prev_t:
                    asc += 1
            prev_t = t

print(f"rows: {n} | first: {first_ts} | last: {last_ts}", flush=True)
print(f"strictly-ascending steps: {asc} ({asc/n*100:.1f}%)", flush=True)
print(f"resets (t<prev): {len(resets)}", flush=True)
if resets:
    print(f"first 12 reset rows: {resets[:12]}", flush=True)
    print(f"last 6 reset rows: {resets[-6:]}", flush=True)
    diffs = [b - a for a, b in zip(resets, resets[1:])]
    print(f"spacing: min {min(diffs)} max {max(diffs)} uniq {len(set(diffs))}", flush=True)
    # rows per period = spacing/84? check divisibility
    if len(diffs) > 2:
        d0 = diffs[0]
        print(f"first spacing {d0} /84 = {d0/84:.1f} | /48 = {d0/48:.1f}", flush=True)
