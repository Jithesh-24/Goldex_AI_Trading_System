#!/usr/bin/env python3
"""Smoke-test features.vector_regime_bin against features.regime_bin (scalar)."""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "/home/jith/.hermes/profiles/trading/scripts")
import features as F

rng = np.random.default_rng(42)
N = 300_000
df = pd.DataFrame({
    "trend_ema": rng.normal(0, 1.2, N),
    "trend_slope": rng.normal(0, 0.3, N),
    "bb_pctile": rng.uniform(0, 1, N),
    "atr_pctile": rng.uniform(0, 1, N),
    "vol_spike": rng.exponential(0.8, N),
    "news_candle": rng.binomial(1, 0.05, N).astype(float),
    "rsi_14": rng.normal(50, 15, N),
    "m1_d1_vol_ratio": rng.lognormal(0, 0.4, N),
})
vec = F.vector_regime_bin(df)
scalar = np.array([F.regime_bin(r) for r in df.to_dict("records")])
match = (vec == scalar).mean()
print(f"features.vector_regime_bin match: {match:.4f} ({int((vec==scalar).sum())}/{N})")
if match < 0.999:
    bad = np.where(vec != scalar)[0][:10]
    for i in bad:
        print("  MISMATCH row", i, "vec=", vec[i], "scalar=", scalar[i])
    sys.exit(1)
print("SMOKE PASS")
