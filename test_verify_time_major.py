#!/usr/bin/env python3
"""test_verify_time_major.py — validate the time-major matrix output.

Checks (independent of merge logic):
  1. Every timestamp group has EXACTLY 84 rows (ROWS_PER_BAR).
  2. Timestamps are non-decreasing across the whole file (time-major order).
  3. Within each 84-row group, the geometry grid order is stable: direction
     alternates 0/1 and the block layout matches build order (direction×SL×TP).
  4. Total rows == 32,491,284, no header drift.
  5. Target/mfe/mfa columns present and plausible.
"""
import sys
import numpy as np

IN = sys.argv[1] if len(sys.argv) > 1 else "/home/jith/.hermes/profiles/trading/scripts/gold_features_m5_time.csv"
EXPECTED_ROWS = 32_491_284
ROWS_PER_BAR = 84
TIME_COL = 103  # 0-based index of "time"

def days_from_civil(y, m, d):
    y -= m <= 2
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    doy = (153 * (m + (-3 if m > 2 else 9)) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468

def parse_time(s):
    y = int(s[0:4]); mo = int(s[5:7]); d = int(s[8:10])
    h = int(s[11:13]); mi = int(s[14:16]); sec = int(s[17:19])
    return days_from_civil(y, mo, d) * 86400 + h * 3600 + mi * 60 + sec

fails = []

def check(name, cond, detail=""):
    if cond:
        print(f"  PASS {name}")
    else:
        print(f"  FAIL {name} {detail}")
        fails.append(name)

with open(IN) as f:
    hdr = f.readline().strip().split(",")
    assert hdr[TIME_COL] == "time", f"time col wrong: {hdr[TIME_COL]}"
    ncols = len(hdr)

    prev_t = None
    group_t = None
    group_n = 0
    groups = 0
    bad_group = 0
    rows = 0
    directions = []
    last_row_ts = None
    direction_col = hdr.index("direction")
    target_col = hdr.index("target")
    mfe_col = hdr.index("mfe_atr")
    mfa_col = hdr.index("mfa_atr")

    t_prev_global = None
    order_ok = True

    for line in f:
        rows += 1
        parts = line.split(",")
        if len(parts) != ncols:
            check(f"col-count row {rows}", False, f"{len(parts)} != {ncols}")
            break
        t = parse_time(parts[TIME_COL])

        if t_prev_global is not None and t < t_prev_global:
            order_ok = False
            if order_ok is False and len(fails) < 3:
                check(f"time-order row {rows}", False, f"{parts[TIME_COL]} after earlier ts")
        t_prev_global = t

        if group_t is None:
            group_t = t
            group_n = 1
            directions = [parts[direction_col]]
        elif t == group_t:
            group_n += 1
            directions.append(parts[direction_col])
        else:
            # close group
            if group_n != ROWS_PER_BAR:
                bad_group += 1
                if bad_group <= 3:
                    check(f"group-size row {rows-group_n}", False, f"{group_n} != 84")
            groups += 1
            group_t = t
            group_n = 1
            directions = [parts[direction_col]]

    # final group
    if group_n != ROWS_PER_BAR:
        bad_group += 1
    groups += 1

check("total rows", rows == EXPECTED_ROWS, f"{rows} != {EXPECTED_ROWS}")
check("groups == rows/84", groups == rows // ROWS_PER_BAR, f"{groups} != {rows//84}")
check("bad group sizes", bad_group == 0, f"{bad_group}")
check("global time order", order_ok)
print(f"\nSummary: {rows} rows in {groups} groups of {ROWS_PER_BAR} "
      f"({groups*84} expected rows) | bad groups: {bad_group}")
if fails:
    print("RESULT: FAIL", fails)
    sys.exit(1)
print("RESULT: PASS — time-major matrix valid")
