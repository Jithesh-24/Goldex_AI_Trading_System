#!/usr/bin/env python3
"""
merge_matrix_runs.py v3 — time-major sort via reset-count block interleave.

Matrix layout (written by build_m5_matrix.py): period-major.
    [P0:G0..G83][P1:G0..G83]...[Pk:G0..G83]
Each period has EXACTLY 84 blocks (direction×SL×TP = 2×6×7), every block in a
period has the SAME row count = len(fdf) after dropna.

Reliable structure facts (probed on the 33.7GB file):
  * Block boundaries = time RESETS (t < prev): exactly 83 per period, 29,216 total.
  * Period boundaries = forward gaps (NO reset) — e.g. weekend 68.2h jumps.
  * Internal forward gaps (Christmas 28.3h NaN-drop artifacts) occur INSIDE
    blocks — gap-size detection is UNABLE to find period boundaries.
  * Therefore: count blocks via resets; close block 84 (G83) by row count.

Algorithm:
    stream rows; each reset closes a block (verify uniform size); after 83
    closed blocks, the 84th block (G83) is closed when it reaches block_size
    rows; then flush the period as time-major: for row i, write blocks[0..83][i]
    consecutively (84 rows per timestamp, preserving the geometry grid order).

Output: gold_features_m5_time.csv — same header, same values, time-major order.
"""
import os
import sys
import time

BASE = "/home/jith/.hermes/profiles/trading/scripts"
IN = f"{BASE}/gold_features_m5.csv"
OUT = f"{BASE}/gold_features_m5_time.csv"
LOG = f"{BASE}/merge_time_v3.log"
TIME_COL = 103  # 0-based index of "time" in header (col 104, 1-based)

EXPECTED_BLOCKS = 84
EXPECTED_ROWS = 32_491_284
EXPECTED_PERIODS = 352

def days_from_civil(y, m, d):
    """Exact days since 1970-01-01 (Hinnant civil algorithm, C++ reference)."""
    y -= m <= 2
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    doy = (153 * (m + (-3 if m > 2 else 9)) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468

def parse_time(s):
    """'2019-12-02 18:10:00' -> int seconds (manual, no strptime)."""
    y = int(s[0:4]); mo = int(s[5:7]); d = int(s[8:10])
    h = int(s[11:13]); mi = int(s[14:16]); sec = int(s[17:19])
    return days_from_civil(y, mo, d) * 86400 + h * 3600 + mi * 60 + sec

def log(msg):
    with open(LOG, "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    print(msg, flush=True)

def main():
    t0 = time.time()
    if os.path.exists(LOG):
        os.remove(LOG)
    if os.path.exists(OUT):
        os.remove(OUT)

    with open(IN) as fin, open(OUT, "w") as fout:
        header = fin.readline()
        fout.write(header)
        ncols = len(header.split(","))
        assert ncols > TIME_COL, f"header too narrow: {ncols}"

        cur = []          # rows of current (open) block
        blocks = []       # completed blocks of current period (list of lists of lines)
        block_size = None # uniform block size for current period
        prev_t = None
        first = True
        rows_in = 0
        periods = 0
        resets = 0

        for line in fin:
            rows_in += 1
            parts = line.split(",")
            t = parse_time(parts[TIME_COL])

            # Period-complete check (G83 closed by row count)
            if (len(blocks) == EXPECTED_BLOCKS - 1 and block_size is not None
                    and len(cur) == block_size):
                blocks.append(cur)
                # flush period time-major: row i, then block 0..83
                for i in range(block_size):
                    for b in range(EXPECTED_BLOCKS):
                        fout.write(blocks[b][i])
                periods += 1
                blocks = []
                block_size = None
                cur = [line]
                prev_t = t
                if periods % 25 == 0:
                    log(f"period {periods}/{EXPECTED_PERIODS} rows={rows_in} "
                        f"({(time.time()-t0):.0f}s)")
                continue

            if first:
                cur.append(line); prev_t = t; first = False
                continue

            if t < prev_t:
                # reset closes current block
                if block_size is None:
                    block_size = len(cur)
                elif len(cur) != block_size:
                    log(f"FATAL block-size mismatch: got {len(cur)} want {block_size} "
                        f"at row {rows_in} (period {periods})")
                    sys.exit(1)
                blocks.append(cur)
                resets += 1
                if len(blocks) == EXPECTED_BLOCKS:
                    log(f"FATAL 84 blocks closed without row-count close at row {rows_in}")
                    sys.exit(1)
                cur = [line]
            else:
                cur.append(line)
            prev_t = t

        # EOF: close final block / period
        if cur:
            if block_size is None:
                block_size = len(cur)
            elif len(cur) != block_size:
                log(f"FATAL EOF block-size mismatch: got {len(cur)} want {block_size}")
                sys.exit(1)
            blocks.append(cur)
        if blocks:
            if len(blocks) != EXPECTED_BLOCKS:
                log(f"FATAL EOF: {len(blocks)} blocks in final period (want {EXPECTED_BLOCKS})")
                sys.exit(1)
            for i in range(block_size):
                for b in range(EXPECTED_BLOCKS):
                    fout.write(blocks[b][i])
            periods += 1

    elapsed = time.time() - t0
    out_rows = 0
    with open(OUT) as f:
        out_rows = sum(1 for _ in f) - 1  # minus header
    log(f"DONE periods={periods} resets={resets} in_rows={rows_in} "
        f"out_rows={out_rows} size={os.path.getsize(OUT)/1e9:.1f}GB "
        f"elapsed={elapsed:.0f}s")
    ok = (periods == EXPECTED_PERIODS and out_rows == EXPECTED_ROWS
          and resets == 83 * EXPECTED_PERIODS)
    log(f"VERIFY {'PASS' if ok else 'FAIL'} "
        f"(periods {periods}/{EXPECTED_PERIODS}, rows {out_rows}/{EXPECTED_ROWS}, "
        f"resets {resets}/{83*EXPECTED_PERIODS})")
    sys.exit(0 if ok else 2)

if __name__ == "__main__":
    main()
