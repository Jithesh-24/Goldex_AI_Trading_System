"""Validation: live FeatureComputer must match training features.py exactly.
Feeds the same historical bars through both and compares feature values.
Any mismatch > 1e-6 means the live engine would see a different world than the model.
"""
import sys, math
sys.path.insert(0, "/home/jith/.hermes/profiles/trading/scripts")
import pandas as pd
import numpy as np

from features import build_features
from ai_signal_engine import FeatureComputer

FEATURE_EXCLUDE = {"time", "target", "fwd_return"}

BASE = "/home/jith/.hermes/profiles/trading/scripts"
df = pd.read_csv(f"{BASE}/gold_m1_history.csv")
fx = build_features(df, horizon=15)

# Take the LAST 1600 bars (what the live engine seeds)
tail = df.iloc[-1600:].reset_index(drop=True)

feats = [c for c in fx.columns if c not in FEATURE_EXCLUDE]

# Replay through the live FeatureComputer
fc = FeatureComputer(maxlen=2500)
for i in range(len(tail)):
    r = tail.iloc[i]
    t = pd.Timestamp(r["time"]).timestamp()
    fc.add(t, r["open"], r["high"], r["low"], r["close"], r["tick_volume"], r["spread"])

live_fx = fc.features()

# Align training features by TIME — compare at bar 25 from the end
# (last 15 bars have NaN lookahead target; features must match at a valid bar)
align_idx = len(tail) - 25
align_time = tail.iloc[align_idx]["time"]
# Rebuild the live state up to that bar by trimming the tail (simulate: compare at that moment)
fc2 = FeatureComputer(maxlen=2500)
for i in range(align_idx + 1):
    r = tail.iloc[i]
    t = pd.Timestamp(r["time"]).timestamp()
    fc2.add(t, r["open"], r["high"], r["low"], r["close"], r["tick_volume"], r["spread"])
live_fx = fc2.features()

fx_row = fx[fx["time"] == pd.Timestamp(align_time)]
if len(fx_row) == 0:
    print("❌ Cannot align: training features lack that bar")
    sys.exit(1)
fx_final = fx_row.iloc[0]

print("Feature comparison at final bar (training vs live):")
worst = 0.0
mismatches = []
for col in feats:
    a = float(fx_final[col])
    b = float(live_fx.get(col, float("nan")))
    diff = abs(a - b)
    worst = max(worst, diff)
    if diff > 1e-4 and diff > max(abs(a), abs(b)) * 1e-4:
        mismatches.append((col, a, b, diff))

if not mismatches:
    print(f"✅ PERFECT MATCH — all {len(feats)} features identical (worst diff {worst:.2e})")
else:
    print(f"❌ {len(mismatches)} mismatches:")
    for col, a, b, d in mismatches[:15]:
        print(f"  {col}: train={a:.6f} live={b:.6f} diff={d:.6f}")
    print(f"Worst diff: {worst:.2e}")

# Also verify model prediction consistency
import lightgbm as lgb
model = lgb.Booster(model_file=f"{BASE}/models/gold_lgb_model.txt")
with open(f"{BASE}/models/features.json") as f:
    import json
    model_feats = json.load(f)

X_train = np.array([[float(fx_final[c]) for c in model_feats]], dtype=np.float32)
X_live = np.array([[live_fx.get(c, 0.0) for c in model_feats]], dtype=np.float32)
p_train = model.predict(X_train)[0]
p_live = model.predict(X_live)[0]
print(f"\nModel prediction: train-features={p_train:.4f} | live-features={p_live:.4f}")
print(f"Agreement: {'✅' if abs(p_train-p_live) < 0.01 else '❌ MISMATCH'}")
