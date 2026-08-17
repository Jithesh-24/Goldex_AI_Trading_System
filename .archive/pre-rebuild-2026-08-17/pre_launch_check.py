#!/usr/bin/env python3
"""
pre_launch_check.py — Verify all components are wired correctly
before launching the gold trading system.
"""
import os
import sys
import json
import time

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "models")

errors = []
warnings = []

print("═══ PRE-LAUNCH CHECK ═══\n")

# 1. Check matrix exists
print("1. MATRIX FILES")
full_csv = os.path.join(BASE, "gold_features_m5_full.csv")
tick_csv = os.path.join(BASE, "gold_features_m5_tick.csv")
if os.path.exists(full_csv):
    sz = os.path.getsize(full_csv) / 1024/1024/1024
    print(f"   ✅ gold_features_m5_full.csv ({sz:.1f} GB)")
else:
    errors.append("gold_features_m5_full.csv missing!")
    print("   ❌ gold_features_m5_full.csv MISSING")

if os.path.exists(tick_csv):
    sz = os.path.getsize(tick_csv) / 1024/1024/1024
    print(f"   ✅ gold_features_m5_tick.csv ({sz:.1f} GB)")
else:
    warnings.append("gold_features_m5_tick.csv missing")

# 2. Check model files
print("\n2. MODEL FILES")
required_models = [
    "gold_lgb_model_s42.txt",
    "gold_lgb_model_s7.txt", 
    "gold_lgb_model_s2026.txt",
    "ensemble.json",
    "features.json",
    "signal_rating.json",
    "placement_prior.json",
    "calibration.json",
]

for model in required_models:
    path = os.path.join(MODEL_DIR, model)
    if os.path.exists(path):
        age_hours = (time.time() - os.path.getmtime(path)) / 3600
        if age_hours > 48:
            warnings.append(f"{model} is {age_hours:.0f}h old")
            print(f"   ⚠️ {model} ({age_hours:.0f}h old)")
        else:
            print(f"   ✅ {model}")
    else:
        errors.append(f"{model} missing!")
        print(f"   ❌ {model} MISSING")

# 3. Check ensemble config
print("\n3. ENSEMBLE CONFIG")
try:
    with open(os.path.join(MODEL_DIR, "ensemble.json")) as f:
        ens = json.load(f)
    print(f"   ✅ Seeds: {ens.get('seeds', [])}")
    print(f"   ✅ Models: {len(ens.get('models', []))} files")
except Exception as e:
    errors.append(f"ensemble.json read error: {e}")
    print(f"   ❌ {e}")

# 4. Check features
print("\n4. FEATURE CONFIG")
try:
    with open(os.path.join(MODEL_DIR, "features.json")) as f:
        feats = json.load(f)
    print(f"   ✅ Feature count: {len(feats.get('features', []))}")
    print(f"   ✅ Target: {feats.get('target', 'unknown')}")
except Exception as e:
    errors.append(f"features.json read error: {e}")
    print(f"   ❌ {e}")

# 5. Check Renaissance modules
print("\n5. RENAISSANCE MODULES")
try:
    sys.path.insert(0, BASE)
    from renaissance_modules import (
        HMMRegimeDetector, KalmanFilter, 
        OrnsteinUhlenbeckDetector, half_kelly,
        compute_renaissance_features
    )
    print("   ✅ HMMRegimeDetector imported")
    print("   ✅ KalmanFilter imported")
    print("   ✅ OrnsteinUhlenbeckDetector imported")
    print("   ✅ half_kelly imported")
    print("   ✅ compute_renaissance_features imported")
except Exception as e:
    errors.append(f"Renaissance modules error: {e}")
    print(f"   ❌ {e}")

# 6. Test Renaissance modules
print("\n6. RENAISSANCE MODULE TEST")
import numpy as np
np.random.seed(42)
prices = 2000 + np.cumsum(np.random.randn(200) * 0.5)
result = compute_renaissance_features(prices, window=100)
print(f"   ✅ HMM regime: {result['hmm_regime']}")
print(f"   ✅ HMM probs: {result['hmm_probs']}")
print(f"   ✅ Kalman trend: {result['kalman_trend']:.2f}")
print(f"   ✅ OU theta: {result['ou_theta']:.4f}")
print(f"   ✅ OU is_mr: {result['ou_is_mr']}")
print(f"   ✅ Kelly fraction: {result['kelly_fraction']:.4f}")

# 7. Check engine
print("\n7. ENGINE")
engine_path = os.path.join(BASE, "ai_signal_engine.py")
if os.path.exists(engine_path):
    with open(engine_path) as f:
        content = f.read()
    
    # Check Renaissance integration
    if 'renaissance_modules' in content:
        print("   ✅ Renaissance modules imported in engine")
    else:
        errors.append("Renaissance modules NOT imported in engine!")
        print("   ❌ Renaissance modules NOT imported")
    
    # Check circuit breaker removed
    if 'circuit_breaker_check' in content and 'CIRCUIT BREAKER REMOVED' not in content:
        warnings.append("Circuit breaker may still be active")
        print("   ⚠️ Circuit breaker check found")
    else:
        print("   ✅ Circuit breaker removed")
    
    # Check macro injection removed
    if 'CROSS-ASSET MACRO' in content and 'COMMENTED OUT' in content:
        print("   ✅ Macro injection commented out")
    else:
        warnings.append("Macro injection status unclear")
        print("   ⚠️ Macro injection status unclear")
    
    # Check advanced features
    if 'advanced_features' in content:
        print("   ✅ Advanced features imported")
    else:
        warnings.append("Advanced features not imported")
        print("   ⚠️ Advanced features not imported")
    
    print(f"   ✅ Engine size: {len(content):,} chars, {content.count(chr(10))} lines")
else:
    errors.append("Engine file missing!")
    print("   ❌ Engine file MISSING")

# 8. Check ticker
print("\n8. TICKER")
ticker_path = os.path.join(BASE, "xm_ticker.py")
if os.path.exists(ticker_path):
    print(f"   ✅ xm_ticker.py exists ({os.path.getsize(ticker_path)/1024:.1f} KB)")
else:
    errors.append("xm_ticker.py missing!")
    print("   ❌ xm_ticker.py MISSING")

# 9. Check retrain script
print("\n9. RETRAIN SCRIPT")
retrain_path = os.path.join(BASE, "retrain_m5.py")
if os.path.exists(retrain_path):
    print(f"   ✅ retrain_m5.py exists")
else:
    warnings.append("retrain_m5.py missing")
    print("   ⚠️ retrain_m5.py missing")

# 10. Check Renaissance retrain script
print("\n10. RENAISSANCE RETRAIN")
ren_path = os.path.join(BASE, "retrain_with_renaissance.py")
if os.path.exists(ren_path):
    print(f"   ✅ retrain_with_renaissance.py exists")
else:
    warnings.append("retrain_with_renaissance.py missing")
    print("   ⚠️ retrain_with_renaissance.py missing")

# Summary
print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"Errors: {len(errors)}")
print(f"Warnings: {len(warnings)}")

if errors:
    print("\n❌ ERRORS (must fix):")
    for e in errors:
        print(f"   - {e}")

if warnings:
    print("\n⚠️ WARNINGS:")
    for w in warnings:
        print(f"   - {w}")

if not errors:
    print("\n✅ ALL CHECKS PASSED — READY TO LAUNCH")
else:
    print("\n❌ FIX ERRORS BEFORE LAUNCHING")
