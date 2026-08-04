#!/usr/bin/env python3
"""v7.3e: PER-DIRECTION × PER-RR calibration — unlocks the structural BUY edge.

Why: gold's 2020-2026 uptrend means BUY trades win MORE at identical geometry
(BUY ~33.6-34.6% vs SELL ~26.5-28.7% at RR 2.5-3.0). Pooling BUY+SELL into one
calibration curve FLATTENS that signal: the model sees both as "~30%" and
cannot express "today BUY is better". Splitting per (direction, RR) gives each
its own honest P(win) → the EV sweep naturally prefers BUY at wide RR, which
is the structural edge (BUY@RR3.0 = +0.34R vs SELL@RR3.0 = +0.06R).

Uses existing walk-forward OOF — no model retrain. Output:
models/calibration_by_drr.json  (keys: "BUY_1.3", "SELL_3.0", ...)
"""
import json
import numpy as np
import pandas as pd
import os

BASE = "/home/jith/.hermes/profiles/trading/scripts"
MODEL_DIR = f"{BASE}/models"
MATRIX = f"{BASE}/gold_features.csv"

TP_RATIOS = [1.3, 1.8, 2.5, 3.0]
DIRS = ["BUY", "SELL"]


def pava_bins(p, y, n_bins=100):
    """Bin p, PAVA-pool violators, return monotone (ps, ys)."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    qs = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    bin_ids = np.clip(np.searchsorted(qs, p, side="right") - 1, 0, len(qs) - 2)
    agg = {}
    for b in np.unique(bin_ids):
        m = bin_ids == b
        agg[b] = (p[m].mean(), y[m].sum(), int(m.sum()))
    bs = sorted(agg.keys())
    ps = np.array([agg[b][0] for b in bs])
    ys = np.array([agg[b][1] for b in bs])
    ws = np.array([agg[b][2] for b in bs], dtype=np.float64)
    blocks = [[ps[i], ys[i], ws[i]] for i in range(len(ps))]
    i = 0
    while i < len(blocks) - 1:
        mean_i = blocks[i][1] / blocks[i][2]
        mean_j = blocks[i + 1][1] / blocks[i + 1][2]
        if mean_j < mean_i - 1e-12:
            tot_w = blocks[i][2] + blocks[i + 1][2]
            blocks[i][0] = (blocks[i][0] * blocks[i][2] + blocks[i + 1][0] * blocks[i + 1][2]) / tot_w
            blocks[i][1] = blocks[i][1] + blocks[i + 1][1]
            blocks[i][2] = blocks[i][2] + blocks[i + 1][2]
            blocks.pop(i + 1)
            i = max(0, i - 1)
        else:
            i += 1
    return np.array([b[0] for b in blocks]), np.array([b[1] / b[2] for b in blocks])


def rr_bucket(rr):
    """Map an effective RR to the nearest grid ratio."""
    return min(TP_RATIOS, key=lambda t: abs(rr - t))


def main():
    print("Loading OOF...")
    oof = np.load(f"{MODEL_DIR}/oof_probs.npy")
    oofy = np.load(f"{MODEL_DIR}/oof_targets.npy")
    n = len(oof)
    print(f"OOF: {n:,} rows | base WR {oofy.mean():.1%}")

    print("Streaming matrix for per-row (direction, RR)...")
    drr = np.zeros(n, dtype=np.float32)   # 0..7: idx = dir*4 + rr_idx
    seen = 0
    for ch in pd.read_csv(MATRIX, usecols=["direction", "rr_buy", "rr_sell"],
                          dtype={"direction": np.int8, "rr_buy": np.float32, "rr_sell": np.float32},
                          chunksize=1_000_000):
        d = ch["direction"].values
        rr = np.where(d == 1, ch["rr_buy"].values, ch["rr_sell"].values)
        bucket = np.array([rr_bucket(r) for r in rr])
        idx = np.where(d == 1, 4, 0) + np.array([TP_RATIOS.index(t) for t in bucket])
        drr[seen:seen + len(ch)] = idx
        seen += len(ch)
    assert seen == n, f"matrix rows {seen} != OOF rows {n}"

    print("Fitting per-direction × per-RR calibration...")
    out = {}
    for di, dname in enumerate(DIRS):
        for ri, t in enumerate(TP_RATIOS):
            key = f"{dname}_{t}"
            m = drr == (di * 4 + ri)
            if m.sum() < 5000:
                print(f"  {key}: only {m.sum():,} rows — SKIP")
                continue
            ps, ys = pava_bins(oof[m], oofy[m])
            out[key] = {"knots_p": ps.tolist(), "knots_y": ys.tolist(), "n": int(m.sum())}
            samples = list(zip([round(x, 2) for x in ps[::max(1, len(ps)//5)]],
                               [round(y, 3) for y in ys[::max(1, len(ys)//5)]]))
            print(f"  {key}: n={m.sum():,} | base WR {oofy[m].mean():.1%} | curve samples {samples[:3]}...")

    outpath = f"{MODEL_DIR}/calibration_by_drr.json"
    with open(outpath + ".tmp", "w") as f:
        json.dump(out, f, indent=2)
    os.replace(outpath + ".tmp", outpath)  # atomic — matches the model swap pattern
    print(f"\nSaved: {outpath}")
    print("Done.")


if __name__ == "__main__":
    main()
