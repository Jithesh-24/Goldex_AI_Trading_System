#!/usr/bin/env python3
"""Smoke-test fit_signal_rating's vectorized helpers against features.regime_bin."""
import os, sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, "/home/jith/.hermes/profiles/trading/scripts")
import features as F
sys.path.insert(0, "/home/jith/.hermes/profiles/trading/scripts")
import importlib.util
spec = importlib.util.spec_from_file_location("fsr", "/home/jith/.hermes/profiles/trading/scripts/fit_signal_rating.py")
# load without running main
src = open("/home/jith/.hermes/profiles/trading/scripts/fit_signal_rating.py").read()
src = src.replace('if __name__ == "__main__":\n    main()', 'pass')
mod = importlib.util.module_from_spec(spec)
exec(compile(src, "fsr", "exec"), mod.__dict__)

rng = np.random.default_rng(42)
N = 200_000
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
vec = mod.vector_regime_bin(df)
scalar = np.array([F.regime_bin(r) for r in df.to_dict("records")])
match = (vec == scalar).mean()
print(f"regime_bin match: {match:.4f} ({int((vec==scalar).sum())}/{N})")
if match < 0.999:
    bad = np.where(vec != scalar)[0][:10]
    for i in bad:
        print("  MISMATCH row", i, "vec=", vec[i], "scalar=", scalar[i],
              {k: round(float(df.iloc[i][k]), 3) for k in ["trend_ema","trend_slope","bb_pctile","atr_pctile","vol_spike","news_candle","rsi_14","m1_d1_vol_ratio"]})
    sys.exit(1)

# test _regime_conf vector vs scalar
conf_vec = mod._regime_conf(df["trend_ema"].values, df["atr_pctile"].values)
def scalar_conf(row):
    te = float(row["trend_ema"]); ap = float(row["atr_pctile"])
    dist = min(abs(abs(te) - 0.4), abs(abs(te) - 1.2), 1.0)
    vol_dist = abs(ap - 0.5) * 2.0
    return float(np.clip(0.5 * (1 - dist) + 0.25 * vol_dist, 0, 1))
conf_sc = np.array([scalar_conf(r) for r in df.to_dict("records")])
conf_match = np.allclose(conf_vec, conf_sc, atol=1e-12)
print(f"_regime_conf vector vs scalar: {conf_match}")
if not conf_match:
    sys.exit(1)
print("SMOKE PASS")
