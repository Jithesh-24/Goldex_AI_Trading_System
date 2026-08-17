"""Smoke-test the FIXED emit block: build a fake 3-bar period, run the exact
emit code path, and assert every column lands in the right place."""
import sys, os
import numpy as np
import pandas as pd

BASE = "/home/jith/.hermes/profiles/trading/scripts"
sys.path.insert(0, BASE)
import features as F

n = 3
rng = np.random.default_rng(42)
hdr_market = ["spread", "ret_1", "ret_2", "ret_3", "ret_5", "atr_pct"]
market_cols = hdr_market
geometry_cols = ["sl_dist_buy","tp_dist_buy","sl_dist_sell","tp_dist_sell",
                 "sl_atr_buy","sl_atr_sell","rr_buy","rr_sell"]
out_cols = market_cols + ["open","high","low","close","spread","time"] + \
           geometry_cols + ["direction","target","mfe_atr","mfa_atr"]

fdf = pd.DataFrame({
    "spread": [20.0]*n, "ret_1": [0.0]*n, "ret_2": [0.0]*n, "ret_3": [0.0]*n,
    "ret_5": [0.0]*n, "atr_pct": [0.1]*n,
    "open": [100.0]*n, "high": [101.0]*n, "low": [99.0]*n,
    "close": [100.0]*n, "time": ["2026-01-01 00:00:00", "2026-01-01 00:05:00",
                                 "2026-01-01 00:10:00"],
})
n_mc = len(market_cols)
geom0 = n_mc + 6
dir0 = geom0 + len(geometry_cols)
tgt0 = dir0 + 1
mfe0 = tgt0 + 1
mfa0 = mfe0 + 1
assert len(out_cols) == mfa0 + 1, f"len {len(out_cols)} vs {mfa0+1}"

GEOMS = [(d, m, r) for m in F.SL_MULTS for r in F.TP_RATIOS for d in ("SELL","BUY")]
n_g = len(GEOMS)
targets = np.zeros((n_g, n), dtype=np.float32); targets[:, 1] = 1.0
mfe = np.full((n_g, n), 2.5, dtype=np.float32); mfe[0, 0] = 0.5
mfa = np.full((n_g, n), 1.5, dtype=np.float32)
dir_col = np.zeros((n, n_g), dtype=np.float32)
gcols = np.zeros((n, n_g, 8), dtype=np.float32)
for gi, (d, m, r) in enumerate(GEOMS):
    dir_col[:, gi] = 1.0 if d == "BUY" else 0.0
    gcols[:, gi, 0] = 0.86; gcols[:, gi, 1] = 1.38; gcols[:, gi, 2] = 0.86
    gcols[:, gi, 3] = 1.38; gcols[:, gi, 4] = 0.8; gcols[:, gi, 5] = 0.8
    gcols[:, gi, 6] = 1.6; gcols[:, gi, 7] = 1.6

feats_np = fdf[market_cols].values.astype(np.float32)
raw_np = fdf[["open","high","low","close","spread"]].values.astype(np.float32)
times = fdf["time"].values
blk = 2
first = True
tmp = "/tmp/smoke_out.csv"
if os.path.exists(tmp): os.remove(tmp)
for b0 in range(0, n, blk):
    b1 = min(b0 + blk, n); nb = b1 - b0
    block = np.empty((nb * n_g, len(out_cols)), dtype=np.float32)
    for j, b in enumerate(range(b0, b1)):
        rows = slice(j * n_g, (j+1) * n_g)
        for ci, c in enumerate(market_cols):
            block[rows, ci] = feats_np[b, ci]
        for ci, c in enumerate(["open","high","low","close","spread"]):
            block[rows, n_mc + ci] = raw_np[b, ci]
        block[rows, geom0:geom0 + len(geometry_cols)] = gcols[b]
        block[rows, dir0] = dir_col[b]
        block[rows, tgt0] = targets[:, b]
        block[rows, mfe0] = mfe[:, b]
        block[rows, mfa0] = mfa[:, b]
    out_df = pd.DataFrame(block, columns=out_cols)
    out_df["time"] = np.repeat(times[b0:b1], n_g)
    out_df.to_csv(tmp, mode="a", header=first, index=False)
    first = False

chk = pd.read_csv(tmp, dtype={"time": str})
dir_expect = np.tile(np.array([0.0, 1.0], dtype=np.float32), n_g // 2)  # SELL,BUY per geom
assert (chk["direction"].values == np.tile(dir_expect, n)).all(), "direction wrong"
b1 = chk[chk["time"] == "2026-01-01 00:05:00"]
assert (b1["target"] == 1.0).all(), "bar1 target wrong"
b0r = chk[chk["time"] == "2026-01-01 00:00:00"]
assert b0r["target"].iloc[0] == 0.0 and b0r["mfe_atr"].iloc[0] == 0.5, "bar0 mfe wrong"
assert (b0r["mfa_atr"] == 1.5).all(), "mfa wrong"
assert (b0r["sl_dist_buy"] == 0.86).all(), "sl wrong"
assert (b0r["rr_buy"] == 1.6).all(), "rr wrong"
assert b0r["spread"].iloc[0] == 20.0, "spread overwritten!"
assert (b0r["open"] == 100.0).all(), "open wrong"
assert b0r["time"].iloc[0] == "2026-01-01 00:00:00", "time wrong"
print("✅ SMOKE TEST PASS — direction/target/mfe/mfa/geometry/raw/time all in correct columns")
