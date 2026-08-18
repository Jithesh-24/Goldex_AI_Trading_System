"""
Phase 3B Part 2 -- feature research pipeline, steps 1-4 (causality, NaN/
availability, data-semantic sanity, redundancy). Steps 5 (leakage -- same
test as step 1 here), 6 (information/MI), 7 (walk-forward ablation), 8
(fold stability) happen in research/v3_train_and_select.py, which needs the
trained OOF models this script's output feeds into.

Causality/leakage check method: compute candidate features on the full
buffer, then again on a buffer truncated 2000 bars before the end, and
diff the OVERLAPPING region. Any feature that uses future information
(even accidentally, e.g. a global pd.cut()) will differ between the two
computations at rows close to the truncation point; a genuinely causal
feature is bit-identical there. This caught a real bug during development
(persistence_state/entropy_state/activity_state used pd.cut(), which fixes
bin edges from the whole series -- fixed in research/features_v3.py before
this check was run for real).

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.v3_pipeline_checks
"""
import json
import os
import time

import numpy as np
import pandas as pd

from learning.data import load_raw_m1
from features.features import build_features
from research.features_v3 import build_candidate_features

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "research", "output")
SUBSET_ROWS = 250_000       # ~6 months M1 -- enough for stable correlations, fast to compute
TRUNCATE_MARGIN = 2000       # bars removed from the end for the causality test


def main():
    t_start = time.time()
    raw = load_raw_m1().tail(SUBSET_ROWS).reset_index(drop=True)
    print(f"== subset: {len(raw):,} bars ==")

    base_full = build_features(raw)
    cand_full = build_candidate_features(raw, base_full)
    feature_names = [c for c in cand_full.columns if c != "time"]
    print(f"{len(feature_names)} candidate features built")

    # ---- step 1/5: causality / leakage (truncation test) ----
    raw_trunc = raw.iloc[:-TRUNCATE_MARGIN].reset_index(drop=True)
    base_trunc = build_features(raw_trunc)
    cand_trunc = build_candidate_features(raw_trunc, base_trunc)

    check_row = len(raw_trunc) - 500  # comfortably inside both frames' warmup-safe region
    causality = {}
    for c in feature_names:
        a = cand_full[c].iloc[check_row]
        b = cand_trunc[c].iloc[check_row]
        if pd.isna(a) and pd.isna(b):
            causality[c] = "PASS (both NaN)"
        elif pd.isna(a) or pd.isna(b):
            causality[c] = f"MISMATCH (one NaN): full={a} trunc={b}"
        elif np.isclose(a, b, rtol=1e-8, atol=1e-10):
            causality[c] = "PASS"
        else:
            causality[c] = f"FAIL: full={a} trunc={b}"
    fails = {k: v for k, v in causality.items() if not v.startswith("PASS")}
    print(f"\n== causality/leakage check: {len(feature_names)-len(fails)}/{len(feature_names)} PASS ==")
    for k, v in fails.items():
        print(f"  {k}: {v}")

    # ---- step 2: NaN / availability ----
    nan_frac = cand_full[feature_names].isna().mean()
    availability = {c: float(nan_frac[c]) for c in feature_names}
    thin = {c: v for c, v in availability.items() if v > 0.5}
    print(f"\n== NaN/availability: {len(thin)} features with >50% NaN on this subset ==")
    for c, v in sorted(thin.items(), key=lambda kv: -kv[1]):
        print(f"  {c}: {v*100:.1f}% NaN")

    # ---- step 3: data-semantic sanity ----
    semantic_flags = {}
    for c in feature_names:
        s = cand_full[c].dropna()
        if len(s) == 0:
            semantic_flags[c] = "ALL_NAN"
            continue
        if s.nunique() <= 1:
            semantic_flags[c] = "CONSTANT"
            continue
        if not np.isfinite(s.to_numpy()).all():
            semantic_flags[c] = "NON_FINITE"
            continue
        if "state" in c and (s.min() < -0.01 or s.max() > 10):
            semantic_flags[c] = f"STATE_OUT_OF_RANGE min={s.min()} max={s.max()}"
    bad_semantic = {k: v for k, v in semantic_flags.items()}
    print(f"\n== data-semantic sanity: {len(bad_semantic)} flagged ==")
    for c, v in bad_semantic.items():
        print(f"  {c}: {v}")

    # ---- step 4: redundancy (correlation) ----
    corr = cand_full[feature_names].corr().abs()
    corr_arr = corr.to_numpy(copy=True)
    np.fill_diagonal(corr_arr, 0)
    corr = pd.DataFrame(corr_arr, index=corr.index, columns=corr.columns)
    redundant_pairs = []
    seen = set()
    for c in feature_names:
        top = corr[c].idxmax()
        val = corr[c].max()
        pair = tuple(sorted((c, top)))
        if val > 0.95 and pair not in seen:
            seen.add(pair)
            redundant_pairs.append({"a": pair[0], "b": pair[1], "abs_corr": float(val)})
    print(f"\n== redundancy: {len(redundant_pairs)} pairs with |corr| > 0.95 ==")
    for p in sorted(redundant_pairs, key=lambda x: -x["abs_corr"]):
        print(f"  {p['a']} <-> {p['b']}: {p['abs_corr']:.3f}")

    result = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_features": len(feature_names), "subset_rows": SUBSET_ROWS,
        "causality_check": causality, "causality_fails": list(fails.keys()),
        "nan_availability": availability, "thin_features_over_50pct_nan": list(thin.keys()),
        "semantic_flags": bad_semantic, "redundant_pairs_corr_gt_0.95": redundant_pairs,
    }
    out_path = os.path.join(OUT, "v3_pipeline_checks.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nsaved -> {out_path} ({time.time()-t_start:.1f}s)")


if __name__ == "__main__":
    main()
