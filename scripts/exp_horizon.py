#!/usr/bin/env python3
"""Fast horizon sensitivity: does a longer trade horizon give the model more
discriminative power (AUC)? Subsampled seed + quick LightGBM. Runs in ~3 min."""
import numpy as np
import pandas as pd
import lightgbm as lgb
import time, sys
sys.path.insert(0, "/home/jith/.hermes/profiles/trading/scripts")

df = pd.read_csv("/home/jith/.hermes/profiles/trading/scripts/gold_seed.csv", parse_dates=["time"])
sub = df.iloc[::5].reset_index(drop=True)   # every 5th bar = ~12k bars
print(f"seed {len(df)} -> subsample {len(sub)}")

def quick_auc(y, p):
    from scipy.stats import rankdata
    rp = rankdata(p); n1 = (y == 1).sum(); n0 = len(y) - n1
    return (rp[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 * n0 else 0

import features as F
for max_bars in [60, 240]:
    t0 = time.time()
    mdf = F.build_placement_dataset(sub, max_bars=max_bars)
    feats = [c for c in mdf.columns if c not in ("time", "target", "fwd_return",
             "open", "high", "low", "close", "tick_volume", "spread", "real_volume")]
    X = mdf[feats].values.astype(np.float32); y = mdf["target"].values.astype(int)
    m = lgb.train({"objective": "binary", "learning_rate": 0.08, "num_leaves": 63,
                   "feature_fraction": 0.85, "bagging_fraction": 0.85, "bagging_freq": 1,
                   "verbosity": -1}, lgb.Dataset(X, label=y), num_boost_round=200)
    p = m.predict(X)
    print(f"max_bars={max_bars}: rows={len(mdf)} base={y.mean():.3f} AUC(in-sample)={quick_auc(y, p):.4f} ({time.time()-t0:.0f}s)")
