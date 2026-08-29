"""
Phase 3B Part 7 -- dynamic SL/TP research. Resimulates the triple-barrier
labeling at a GRID of alternate (SL_mult, TP_mult) geometries against the
same 212,108 events (same t0, side, vol -- only the barrier widths change),
overall and by volatility state. Research only: does NOT pick a winner or
change production geometry (still 1.0R SL / 1.5R TP everywhere live).

STATE -> MAE/MFE distribution -> TP-before-SL probability -> EV, per Phase 2
Step 7's "do not over-engineer" spirit: this is grid resimulation with the
existing numba triple-barrier core, not a new model.

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.dynamic_sltp_research
"""
import json
import os
import time

import numpy as np
import pandas as pd

from learning.data import load_raw_m1
from learning.train import assemble_dataset, TB_CFG_DIR
from features.labeling import triple_barrier_labels, TripleBarrierConfig

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "research", "output")

SL_GRID = [0.75, 1.0, 1.25, 1.5]
TP_GRID = [1.0, 1.5, 2.0, 2.5, 3.0]
BASELINE = (1.0, 1.5)  # current production (sl_mult, pt_mult)


def run_geometry(close, high, low, t0_idx, vol_tb, side, sl_mult, pt_mult):
    cfg = TripleBarrierConfig(pt_mult=pt_mult, sl_mult=sl_mult, max_holding=TB_CFG_DIR.max_holding)
    labels = triple_barrier_labels(close, high, low, t0_idx, vol_tb, cfg, side=side)
    touch = labels["touch"].to_numpy()
    favorable = np.where(side >= 0, 1, -1)
    p_tp = float((touch == favorable).mean())
    p_sl = float((touch == -favorable).mean())
    p_timeout = float((touch == 0).mean())
    ev = p_tp * pt_mult - p_sl * sl_mult
    return {"sl_mult": sl_mult, "pt_mult": pt_mult, "p_tp": p_tp, "p_sl": p_sl,
            "p_timeout": p_timeout, "ev_R": ev, "n": int(len(t0_idx))}


def main():
    t_start = time.time()
    df = pd.read_csv(os.path.join(OUT, "mae_mfe_dataset.csv"), parse_dates=["event_time"])

    print("== rebuilding close/high/low/vol_tb + matching events ==")
    feat, close, high, low, vol_tb, _, _ = assemble_dataset()
    times = pd.to_datetime(feat["time"].to_numpy())
    idx_map = pd.Series(np.arange(len(times)), index=times)
    t0_idx_all = idx_map.reindex(df["event_time"]).to_numpy()
    ok = np.isfinite(t0_idx_all)
    df = df.loc[ok].reset_index(drop=True)
    t0_idx_all = t0_idx_all[ok].astype(np.int64)
    side_all = np.where(df["direction"].to_numpy() == "BUY", 1.0, -1.0)
    df["cal_decile"] = pd.qcut(df["calibrated_proba"], 10, labels=False, duplicates="drop")
    print(f"matched {len(df):,} events")

    result = {"generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "n_events": int(len(df)), "sl_grid": SL_GRID, "tp_grid": TP_GRID,
              "baseline_sl_pt": BASELINE}

    print("\n== overall grid ==")
    grid_overall = []
    for sl in SL_GRID:
        for pt in TP_GRID:
            r = run_geometry(close, high, low, t0_idx_all, vol_tb, side_all, sl, pt)
            grid_overall.append(r)
    result["grid_overall"] = grid_overall
    baseline_row = next(r for r in grid_overall if r["sl_mult"] == BASELINE[0] and r["pt_mult"] == BASELINE[1])
    print(f"baseline (sl={BASELINE[0]}, tp={BASELINE[1]}): p_tp={baseline_row['p_tp']:.3f} "
          f"p_sl={baseline_row['p_sl']:.3f} ev={baseline_row['ev_R']:+.4f}R")
    best = max(grid_overall, key=lambda r: r["ev_R"])
    print(f"best-EV overall: sl={best['sl_mult']} tp={best['pt_mult']} ev={best['ev_R']:+.4f}R "
          f"(p_tp={best['p_tp']:.3f})")
    print(f"{'sl':>5} {'tp':>5} {'p_tp':>7} {'p_sl':>7} {'p_to':>7} {'ev_R':>9}")
    for r in grid_overall:
        flag = " <- baseline" if (r["sl_mult"], r["pt_mult"]) == BASELINE else (" <- best" if r is best else "")
        print(f"{r['sl_mult']:>5} {r['pt_mult']:>5} {r['p_tp']:>7.3f} {r['p_sl']:>7.3f} "
              f"{r['p_timeout']:>7.3f} {r['ev_R']:>+9.4f}{flag}")

    print("\n== grid by vol_state ==")
    result["grid_by_vol_state"] = {}
    for vs, sub in df.groupby("vol_state"):
        idx = t0_idx_all[df["vol_state"].to_numpy() == vs]
        s = side_all[df["vol_state"].to_numpy() == vs]
        rows = [run_geometry(close, high, low, idx, vol_tb, s, sl, pt) for sl in SL_GRID for pt in TP_GRID]
        result["grid_by_vol_state"][vs] = rows
        best_vs = max(rows, key=lambda r: r["ev_R"])
        print(f"  {vs}: best sl={best_vs['sl_mult']} tp={best_vs['pt_mult']} ev={best_vs['ev_R']:+.4f}R "
              f"(n={best_vs['n']:,})")

    print("\n== grid by calibrated-probability decile (top/bottom only shown) ==")
    result["grid_by_decile"] = {}
    for dec in sorted(df["cal_decile"].dropna().unique()):
        mask = df["cal_decile"].to_numpy() == dec
        idx = t0_idx_all[mask]
        s = side_all[mask]
        rows = [run_geometry(close, high, low, idx, vol_tb, s, sl, pt) for sl in SL_GRID for pt in TP_GRID]
        result["grid_by_decile"][int(dec)] = rows
        best_d = max(rows, key=lambda r: r["ev_R"])
        if dec in (0, 5, 9):
            print(f"  decile={int(dec)}: best sl={best_d['sl_mult']} tp={best_d['pt_mult']} "
                  f"ev={best_d['ev_R']:+.4f}R (n={best_d['n']:,})")

    out_path = os.path.join(OUT, "dynamic_sltp_research.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nsaved -> {out_path} ({time.time()-t_start:.1f}s)")


if __name__ == "__main__":
    main()
