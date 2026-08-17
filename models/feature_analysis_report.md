# Feature Importance & Model Quality Analysis
## Gold Trading Model v9.3 — 130 Features, 3-Seed Ensemble

---

## 1. EXECUTIVE SUMMARY

| Metric | Value | Assessment |
|--------|-------|------------|
| Total features | 130 | 39 are DEAD (zero importance) |
| Active features | 91 | Top 10 = 51.4% of total gain |
| Train accuracy | 67.2% | Sampled on 500K rows |
| OOF accuracy | 81.5% | WORSE than baseline (82.0%) |
| OOF uplift | **-0.52%** | ⚠️ Model adds NO value vs majority class |
| OOF UP recall | 7.1% | Predicts UP only 3.1% of time (actual: 18%) |
| OOF DOWN recall | 97.8% | Almost always predicts DOWN |
| Max tree depth | 22-26 | ⚠️ Deep overfitting (num_leaves=63, no max_depth) |
| Regularization | None | λ_l1=0, λ_l2=0 |

**Bottom line: The model is a majority-class classifier with no real signal. It predicts "DOWN" 96.9% of the time and achieves 81.5% accuracy simply because DOWN is 82% of the data. The -0.52% uplift means it's WORSE than just always predicting DOWN.**

---

## 2. COMPLETE FEATURE RANKING

### Top 20 Most Important (73% of total gain)

| Rank | Feature | Gain | % | Category |
|------|---------|------|---|----------|
| 1 | rr_buy | 3,298,379 | 15.62% | Placement |
| 2 | min_since_event | 981,950 | 4.65% | Time |
| 3 | min_to_event | 940,879 | 4.46% | Time |
| 4 | sl_atr_buy | 940,868 | 4.46% | Placement |
| 5 | daily_range_pct | 904,921 | 4.29% | Range |
| 6 | **garch_vol** | 904,915 | 4.29% | Renaissance-GARCH ✅ |
| 7 | rr_sell | 826,409 | 3.91% | Placement |
| 8 | dist_prev_high | 698,282 | 3.31% | Distance |
| 9 | dow | 682,238 | 3.23% | Time |
| 10 | **hurst** | 674,530 | 3.19% | Microstructure ✅ |
| 11 | **amihud** | 638,052 | 3.02% | Microstructure ✅ |
| 12 | hour_sin | 604,906 | 2.86% | Time |
| 13 | dist_prev_low | 478,644 | 2.27% | Distance |
| 14 | round50_dist | 432,027 | 2.05% | Distance |
| 15 | **garch_forecast** | 421,154 | 1.99% | Renaissance-GARCH ✅ |
| 16 | **vol_shock** | 419,516 | 1.99% | Renaissance-VolShock ✅ |
| 17 | vol_ewma_60 | 402,908 | 1.91% | Volatility |
| 18 | **entropy** | 391,087 | 1.85% | Microstructure ✅ |
| 19 | **variance_ratio** | 387,858 | 1.84% | Microstructure ✅ |
| 20 | dow_sin | 375,348 | 1.78% | Time |

### Bottom 20 (ALL ZERO — completely dead)

| Rank | Feature | Gain | Issue |
|------|---------|------|-------|
| 111-130 | All 20 Renaissance features | 0.00 | **DATA PIPELINE BUG** |
| 101-110 | engulf, doji, hammer, pin, donch_pos, donch_break, day_pnl, streak, trades_today, cvd | 0.00 | Constant/zero in data |

---

## 3. RENAISSANCE FEATURE ANALYSIS

### ✅ ALIVE & IMPORTANT (11 features, 18.9% of total gain)

| Feature | Gain % | Status |
|---------|--------|--------|
| garch_vol | 4.29% | **Rank #6** — Critical |
| garch_forecast | 1.99% | Rank #15 |
| vol_shock | 1.99% | Rank #16 |
| hurst | 3.19% | **Rank #10** — Critical |
| amihud | 3.02% | **Rank #11** — Critical |
| entropy | 1.85% | Rank #18 |
| variance_ratio | 1.84% | Rank #19 |
| momentum_half_life | 1.13% | Rank #25 |
| return_autocorr | 0.77% | Rank #35 |
| kyle_lambda | 0.08% | Rank #66 |
| vpin | 0.04% | Rank #72 |

### ❌ DEAD — Data Pipeline Bug (15 features, 0% gain)

| Feature Group | Features | Root Cause |
|---------------|----------|------------|
| **HMM** (5) | hmm_regime, hmm_prob_0-3 | ALL ZEROS in mmap |
| **Kalman** (3) | kalman_trend, kalman_velocity, kalman_innovation | ALL ZEROS in mmap |
| **OU** (6) | ou_theta, ou_mu, ou_half_life, ou_is_mr, ou_signal, ou_z_score | ALL ZEROS in mmap |
| **VolRegime** (3) | vol_regime, vol_persistence, vol_asymmetry | ALL ZEROS in mmap |
| **Correlation** (1) | corr_vix | ALL ZEROS in mmap |

### Why They're Dead

**Root cause: `add_renaissance_to_mmap.py` failed to populate the features.**

The script computed Renaissance features every 50th row, then forward-filled gaps. However, the forward-fill logic (`if col[i] != 0.0: last_val = col[i]`) cannot distinguish between "computed 0.0" and "not computed yet 0.0". If the computation threw errors (try/except blocks), the values stayed at 0.0 and the forward-fill propagated zeros forward.

**Proof:** In the last 500K rows, ALL these features are exactly 0.0 with std=0.0000. The first 500K rows show some variation for HMM probs (range 0-0.25) but still mostly zero.

### What Happened to `corr_dxy`?

`corr_dxy` has gain=8,531 (0.04%) — it's barely alive but functional. `corr_vix` is all zeros.

---

## 4. OVERFITTING ANALYSIS

### Train vs OOF Accuracy

| Metric | Train (sampled) | OOF | Gap |
|--------|-----------------|-----|-----|
| Accuracy | 67.2% | 81.5% | -14.3% |
| UP recall | 73.1% | 7.1% | **-66.0%** |
| DOWN recall | 65.0% | 97.8% | +32.8% |
| Pred UP% | 45.6% | 3.1% | -42.5% |

**Critical observation:** The train accuracy (67.2%) is MUCH lower than OOF (81.5%). This seems backwards but is explained by:
- Train data = last 10M rows (balanced era)
- OOF data = full 32.5M rows (dominated by DOWN class from earlier eras)
- The model predicts DOWN 96.9% on OOF, inflating accuracy

### Why Train Accuracy is Lower Than OOF

The OOF data spans 32.5M rows where DOWN=82% of samples. The model learns to predict DOWN overwhelmingly. On the last 10M rows (train), the class balance is 72.3% DOWN / 27.7% UP, so the model is forced to predict UP more often, reducing accuracy.

### Overfitting Evidence

| Indicator | Finding |
|-----------|---------|
| Tree depth | **22-26 levels** (way too deep for 63 leaves) |
| max_depth | Not set (unlimited) |
| min_child_samples | Default=20 (too low) |
| λ_l1, λ_l2 | Both 0 (no regularization) |
| Feature fraction | 0.7 (OK) |
| Bagging fraction | 0.7 (OK) |
| num_boost_round | 200 (moderate) |

**The model is severely overfitting.** Trees reaching depth 22-26 with 63 leaves and no regularization means it's memorizing noise in the training data.

### OOF Prediction Distribution

| Confidence | Count | Accuracy | Actual UP% |
|------------|-------|----------|------------|
| 0.0-0.3 | 23,299,223 | 89.5% | 10.5% |
| 0.3-0.4 | 5,523,450 | 65.4% | 34.6% |
| 0.4-0.5 | 2,680,279 | 60.4% | 39.7% |
| 0.5-0.6 | 633,044 | 39.5% | 39.5% |
| 0.6-0.7 | 273,956 | 43.0% | 43.0% |
| 0.7-1.0 | 99,204 | 50.5% | 50.5% |

**Key insight:** When the model is confident (>0.5), it's barely better than coin flip (39-50% accuracy). The high overall accuracy comes entirely from the low-confidence majority-class predictions.

---

## 5. SIGNAL QUALITY ANALYSIS

Despite the model's poor overall performance, the signal rating deciles show something interesting:

| Decile | Win Rate | Avg R:R | Expectancy | Verdict |
|--------|----------|---------|------------|---------|
| 0-4 (low signal) | 20-41% | 2.1-5.7 | +0.22 to +0.37 | Marginal |
| 5-6 (mid signal) | 40% | 3.7-4.0 | +0.85 to +0.97 | **Good** |
| 7-8 (high signal) | 44-48% | 4.2-4.4 | +1.27 to +1.55 | **Very good** |
| 9 (top signal) | 48% | 5.4 | **+2.07** | **Excellent** |

**The model DOES have some signal in the tails.** When it's very confident (top decile), win rate is 48% with 5.4 R:R = expectancy +2.07. But this represents only 3,279 out of 32.5M bars (0.01%).

---

## 6. ACTIONABLE RECOMMENDATIONS

### 🔴 CRITICAL: Fix Renaissance Feature Pipeline (Priority 1)

The 15 dead Renaissance features (HMM, Kalman, OU, vol_regime, corr_vix) contain NO data. They are all zeros. This is a data pipeline bug in `add_renaissance_to_mmap.py`.

**Fix:** Re-run `add_renaissance_to_mmap.py` with proper error logging and verification:
```python
# Add after forward-fill:
for col_offset in range(len(REN_FEATS)):
    col = X_new[:, ren_start + col_offset]
    pct_nonzero = (col != 0).mean()
    print(f"  {REN_FEATS[col_offset]}: {pct_nonzero*100:.1f}% non-zero")
    if pct_nonzero < 0.1:
        print(f"  ⚠️ WARNING: {REN_FEATS[col_offset]} is mostly zeros!")
```

### 🔴 CRITICAL: Retrain Without 39 Dead Features (Priority 2)

Remove all zero-importance features and retrain:
- **Remove 39 features** (all with gain=0)
- **Keep 91 active features** (all with gain>0)
- This will reduce model complexity and improve generalization

**Features to REMOVE:**
```
ret_5, body_frac, wick_ratio, body_atr, vol_spike_bin, close_loc, 
body_signed, up_wick_frac, dn_wick_frac, engulf, doji, hammer, pin,
donch_pos, donch_break, day_pnl, streak, trades_today, cvd, 
support_dist, resistance_dist, hmm_regime, hmm_prob_0-3, 
kalman_trend, kalman_velocity, kalman_innovation,
ou_theta, ou_mu, ou_half_life, ou_is_mr, ou_signal, ou_z_score,
vol_regime, vol_persistence, vol_asymmetry, corr_vix
```

### 🟡 HIGH: Regularize the Model (Priority 3)

Current params are overfitting badly. Apply these changes:
```python
params = {
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 31,          # DOWN from 63
    "max_depth": 6,            # ADD max_depth limit
    "min_child_samples": 200,  # UP from default 20
    "learning_rate": 0.03,     # DOWN from 0.05
    "num_boost_round": 500,    # UP from 200 (more trees, lower lr)
    "feature_fraction": 0.7,   # Keep
    "bagging_fraction": 0.7,   # Keep
    "lambda_l1": 0.1,          # ADD L1 regularization
    "lambda_l2": 1.0,          # ADD L2 regularization
    "min_gain_to_split": 0.01, # ADD minimum gain threshold
    "is_unbalance": True,
}
```

### 🟡 HIGH: Fix Target Leakage in Placement Features (Priority 4)

The top feature `rr_buy` (15.6% of gain) is a placement/risk-reward feature. These features are computed from future price levels (SL/TP distances) which may introduce look-ahead bias. Verify that:
- `rr_buy`, `rr_sell`, `sl_atr_buy`, `sl_atr_sell`, `tp_dist_buy`, `tp_dist_sell` are computed from CURRENT price, not future prices
- If they use future MFE/MFA data, they need to be removed or recomputed

### 🟢 MEDIUM: Add Missing Features (Priority 5)

Based on the analysis, these feature categories are underrepresented:
1. **Momentum quality** — Add RSI divergence, MACD crossover strength
2. **Volatility regime transitions** — Add volatility change rate (Δvol)
3. **Market microstructure** — Add bid-ask spread dynamics, order flow imbalance trends
4. **Cross-asset signals** — Fix corr_dxy, add DXY momentum, US10Y changes
5. **Temporal patterns** — Add session-relative indicators (London/NY overlap)

### 🟢 MEDIUM: Reduce Feature Count (Priority 6)

With 91 active features, there's redundancy. Consider:
- Remove features below 0.1% gain (13 features): `session, ret_1, ret_15, corr_dxy, cci_20, vpin, vol_z, close_loc_mom, patt_dir, stoch_d, range_pos_20, ret_10, flow_mom`
- Target 60-70 features for optimal bias-variance tradeoff

---

## 7. CATEGORY GAIN SUMMARY

| Category | Gain % | Active/Total | Assessment |
|----------|--------|--------------|------------|
| Placement | 27.49% | 8/8 | ⚠️ Check for leakage |
| Time | 22.84% | 14/14 | ✅ Strong signal |
| Microstructure | 12.40% | 13/14 | ✅ Good (hurst, amihud critical) |
| Distance | 10.66% | 5/5 | ✅ Strong |
| Renaissance-GARCH | 8.27% | 3/3 | ✅ Best Renaissance group |
| Range | 8.24% | 9/9 | ✅ Good |
| Volatility | 5.74% | 11/12 | ✅ Good |
| Technical | 2.91% | 14/14 | ⚠️ Low for 14 features |
| Returns | 0.96% | 9/10 | ⚠️ Weak |
| Candle | 0.06% | 2/13 | ❌ Mostly dead |
| Renaissance-HMM | 0.00% | 0/5 | ❌ All zeros |
| Renaissance-Kalman | 0.00% | 0/3 | ❌ All zeros |
| Renaissance-OU | 0.00% | 0/6 | ❌ All zeros |
| Renaissance-VolRegime | 0.00% | 0/3 | ❌ All zeros |
| State | 0.00% | 1/6 | ❌ Mostly dead |
| Support/Resistance | 0.00% | 0/2 | ❌ All zeros |

---

## 8. FEATURE DRIFT ALERT

Several top features show **massive distributional drift** between early and late data:

| Feature | Mean (first) | Mean (last) | Drift |
|---------|-------------|-------------|-------|
| min_since_event | -740,227 | +10,125 | **5028%** |
| min_to_event | +740,227 | +8,869 | **4900%** |
| garch_vol | 102,063 | 248,363 | **3934%** |
| garch_forecast | 102,063 | 248,363 | **3934%** |
| daily_range_pct | 1.01 | 1.55 | **83%** |
| vol_shock | 7.62 | 8.21 | **387%** |

**These features will degrade model performance over time.** Consider:
- Normalizing time features to [0,1] range
- Adding drift detection and automatic retraining triggers
- Using rolling normalization windows

---

## 9. SUMMARY OF ISSUES

| Issue | Severity | Impact | Fix |
|-------|----------|--------|-----|
| 39 features with zero data | 🔴 Critical | Wasted capacity, adds noise | Remove or fix pipeline |
| Renaissance features all zeros | 🔴 Critical | 15 features wasted | Fix add_renaissance_to_mmap.py |
| No regularization (λ=0) | 🔴 Critical | Severe overfitting | Add L1/L2 regularization |
| Unlimited tree depth | 🔴 Critical | Trees depth 22-26 | Set max_depth=6 |
| Placement features may leak | 🟡 High | Artificially inflated importance | Verify no look-ahead bias |
| Model predicts DOWN 97% | 🟡 High | No real signal extraction | Retrain with balanced approach |
| Feature drift in top features | 🟡 High | Degraded performance over time | Add normalization |
| 13 features below 0.1% gain | 🟢 Medium | Minor noise | Remove to simplify |
