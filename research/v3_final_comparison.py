"""
Phase 3B Part C -- current (v2, 26-feature) vs expanded (26 + surviving
candidates) model comparison, using the SAME walk-forward OOF methodology
and full production CATBOOST_KW settings (not the faster screening config
used for ablation).

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.v3_final_comparison
"""
import json
import os
import time

import numpy as np
import pandas as pd

from research.audit_edge import oof_run, build_meta

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "research", "output")
DS = os.path.join(OUT, "v3_dataset")


def main():
    t_start = time.time()
    with open(os.path.join(DS, "columns.json")) as f:
        cols = json.load(f)
    with open(os.path.join(OUT, "v3_feature_survivors.json")) as f:
        surv = json.load(f)
    survivors = surv["survivors"]
    base_cols = cols["base_cols"]
    expanded_cols = base_cols + survivors
    print(f"base (v2): {len(base_cols)} cols. expanded: {len(expanded_cols)} cols "
          f"({len(survivors)} surviving candidates: {survivors})")

    X = pd.DataFrame(np.load(os.path.join(DS, "X_v3.npy")), columns=cols["all_cols"])
    y_bin = pd.Series(np.load(os.path.join(DS, "y_bin.npy")))
    t0 = pd.Series(np.load(os.path.join(DS, "t0.npy")))
    t1 = pd.Series(np.load(os.path.join(DS, "t1.npy")))
    t0_nz = np.load(os.path.join(DS, "t0_nz.npy"))
    close = np.load(os.path.join(DS, "close.npy"))
    high = np.load(os.path.join(DS, "high.npy"))
    low = np.load(os.path.join(DS, "low.npy"))
    vol_tb = np.load(os.path.join(DS, "vol_tb.npy"))

    def run(cols_subset, tag):
        Xs = X[cols_subset]
        prim = oof_run(Xs, y_bin, t0, t1, tag=f"{tag}-primary", want_importance=False)
        side, meta_labels = build_meta(close, high, low, vol_tb, t0_nz, prim["oof_pred"], prim["has_oof"])
        has_oof = prim["has_oof"]
        X_meta = Xs.loc[has_oof].reset_index(drop=True)
        X_meta["assumed_side"] = side
        y_meta = pd.Series(meta_labels["label"].to_numpy())
        t0_meta = pd.Series(meta_labels.index.to_numpy())
        t1_meta = pd.Series(meta_labels["t1"].to_numpy())
        meta = oof_run(X_meta, y_meta, t0_meta, t1_meta, tag=f"{tag}-meta", want_importance=False)
        return prim, meta

    print("\n== base (v2, 26-feature) ==")
    prim_base, meta_base = run(base_cols, "final-base26")
    print("\n== expanded (26 + survivors) ==")
    prim_exp, meta_exp = run(expanded_cols, "final-expanded")

    def summarize(prim, meta):
        return {"primary_mean_acc": float(np.mean([f["acc"] for f in prim["fold_metrics"]])),
                "primary_fold_acc": [f["acc"] for f in prim["fold_metrics"]],
                "meta_mean_acc": float(np.mean([f["acc"] for f in meta["fold_metrics"]])),
                "meta_fold_acc": [f["acc"] for f in meta["fold_metrics"]],
                "meta_mean_logloss": float(np.mean([f["logloss"] for f in meta["fold_metrics"]]))}

    result = {"generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "n_survivors": len(survivors), "survivors": survivors,
              "base26": summarize(prim_base, meta_base), "expanded": summarize(prim_exp, meta_exp)}
    result["delta_primary_acc"] = result["expanded"]["primary_mean_acc"] - result["base26"]["primary_mean_acc"]
    result["delta_meta_acc"] = result["expanded"]["meta_mean_acc"] - result["base26"]["meta_mean_acc"]

    out_path = os.path.join(OUT, "v3_final_comparison.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\nbase26:    primary={result['base26']['primary_mean_acc']:.4f} meta={result['base26']['meta_mean_acc']:.4f}")
    print(f"expanded:  primary={result['expanded']['primary_mean_acc']:.4f} meta={result['expanded']['meta_mean_acc']:.4f}")
    print(f"delta:     primary={result['delta_primary_acc']:+.4f} meta={result['delta_meta_acc']:+.4f}")
    print(f"\nsaved -> {out_path} ({time.time()-t_start:.1f}s)")


if __name__ == "__main__":
    main()
