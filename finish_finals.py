"""finish_finals.py — v8.8 FAST-PATH RECOVERY (2026-08-12 10:10).

train_ai.py was SIGTERM'd by the resume script's `timeout 36000` at 09:44
mid-final-models-phase (10h cap sized for a wrong speed estimate — MY BUG).
What survived on disk:
  ✅ oof_probs.npy / oof_targets.npy  (the entire 8.5h walk-forward)
  ✅ gold_lgb_model_s42.txt           (final model #1, atomic-swapped 09:37)
  ❌ gold_lgb_model_s7/s2026.txt      (still Aug-10 STALE)
  ❌ features.json / ensemble.json / metrics.json  (written only AFTER finals)

This script: stream matrix → memmap (reuses train_ai.load_data), SKIP the
walk-forward entirely (OOF is already saved), train ONLY the missing final
models (s7, s2026 — s42 exists and is fresh), then write features.json,
ensemble.json, metrics.json exactly as train_ai.main() would.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import json, os, sys, time, gc

BASE = "/home/jith/.hermes/profiles/trading/scripts"
sys.path.insert(0, BASE)
from train_ai import load_data, lgb_params, recency_weights, SEEDS, MODEL_DIR  # noqa: E402

def main():
    t0 = time.time()
    # Re-stream the matrix into a fresh memmap (~15 min). The old memmaps
    # were consumed by the dead process; nothing survives but the OOF cache.
    X, y, times, feats, mm_x, mm_y = load_data()
    n = len(X)
    w = recency_weights(times)
    print(f"finish_finals: {n:,} rows ready in {time.time()-t0:.0f}s", flush=True)

    # Which final models are still stale? (fresh = modified today, > s42's swap)
    fresh = set()
    for s in SEEDS:
        p = f"{MODEL_DIR}/gold_lgb_model_s{s}.txt"
        if os.path.exists(p) and os.path.getmtime(p) > 1_724_000_000_000 / 1_000 - 1e9:
            fresh.add(s)  # heuristic: mtime within last ~2h
    # Explicit: s42 was swapped 09:37 today. Use file mtime vs now.
    now = time.time()
    missing = [s for s in SEEDS
               if not os.path.exists(f"{MODEL_DIR}/gold_lgb_model_s{s}.txt")
               or now - os.path.getmtime(f"{MODEL_DIR}/gold_lgb_model_s{s}.txt") > 6 * 3600]
    print(f"skip (fresh): {[s for s in SEEDS if s not in missing]}", flush=True)
    print(f"train now  : {missing}", flush=True)

    for s in missing:
        t1 = time.time()
        final = lgb.train(lgb_params(s),
                          lgb.Dataset(X, label=y, weight=w, free_raw_data=True,
                                      params={"max_bin": 63, "num_threads": 8}),
                          num_boost_round=600)
        name = f"{MODEL_DIR}/gold_lgb_model_s{s}.txt"
        tmp = name + ".tmp"
        final.save_model(tmp)
        os.replace(tmp, name)
        print(f"Model saved: {name} (atomic swap, {time.time()-t1:.0f}s)", flush=True)
        del final; gc.collect()

    # ── configs exactly as train_ai.main() would have written them ──
    with open(f"{MODEL_DIR}/features.json", "w") as f:
        json.dump(feats, f)

    # metrics.json from the SAVED OOF (train_ai's `res` was in the dead
    # process; the ground truth is on disk — recompute identically).
    oof_probs = np.load(f"{MODEL_DIR}/oof_probs.npy")
    oof_y = np.load(f"{MODEL_DIR}/oof_targets.npy")
    mask = oof_y != 0  # oof_y was only set on TEST rows (train rows = 0)
    oof_probs = oof_probs[mask]; oof_y = oof_y[mask]
    pred = (oof_probs >= 0.5).astype(int)
    acc = float((pred == oof_y).mean())
    up_mask = pred == 1
    dn_mask = pred == 0
    p_up = float(oof_y[up_mask].mean()) if up_mask.any() else float("nan")
    p_dn = float((1 - oof_y[dn_mask]).mean()) if dn_mask.any() else float("nan")
    res = {"acc": round(acc, 4), "p_up": round(p_up, 4), "p_dn": round(p_dn, 4),
           "n": int(len(oof_y))}
    print(f"OOF recomputed from saved cache: {res}", flush=True)
    with open(f"{MODEL_DIR}/metrics.json", "w") as f:
        json.dump(res, f, indent=2)

    with open(f"{MODEL_DIR}/ensemble.json", "w") as f:
        json.dump({"type": "placement", "seeds": SEEDS,
                   "models": [f"gold_lgb_model_s{s}.txt" for s in SEEDS],
                   "recency_tau_days": 120.0,
                   "base_tf": "m5"}, f, indent=2)
    print(f"✅ FINISH_FINALS done in {time.time()-t0:.0f}s — all 3 final models + configs", flush=True)

    # drop memmaps like train_ai does
    try:
        del X, y, w, times
        gc.collect()
        os.remove(mm_x); os.remove(mm_y)
        print("memmaps removed", flush=True)
    except Exception as e:
        print(f"memmap cleanup warn: {e}", flush=True)

if __name__ == "__main__":
    main()