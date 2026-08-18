"""
Phase 3B Part 8 -- feature selection from OOS evidence (MI, CatBoost
importance, fold-to-fold stability, family ablation, redundancy). Combines
research/output/v3_importance_mi.json + v3_family_ablation.json + a fresh
redundancy pass computed directly on the 300,816-event candidate matrix
(more representative than the earlier 250k-bar smoke-test subset).

Rule-based, transparent selection (not a black-box "auto-select-best-N"):
  KEEP a candidate if its META-stage CatBoost importance is non-negligible
  (>=0.02) AND fold-to-fold CV of that importance is <=1.5 (not wildly
  unstable), OR it ranks in the top 25 by meta mutual information even if
  CatBoost underweights it (possible redundancy-driven underuse, not
  necessarily "no information").
  Within any redundant pair (|corr|>0.95), only the higher-ranked survives.
  Family-level ablation is reported alongside as corroborating (not sole)
  evidence -- a family showing ~0 delta with individually-promising member
  features is flagged for scrutiny rather than an automatic veto, since
  family-level ablation removes ALL of a family's columns at once and can
  mask one useful member among several inert ones.

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.v3_feature_selection
"""
import json
import os
import time

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "research", "output")
DS = os.path.join(OUT, "v3_dataset")

IMPORTANCE_THRESHOLD = 0.3   # top-quartile-ish of the observed 0.0-1.1 importance range --
# CatBoost distributes SOME nonzero importance to nearly every candidate it's given a
# chance to split on (normal GBDT behavior with 118 simultaneous features), so a low bar
# like 0.02 "selects" nearly everything and is not real evidence of usefulness. The
# system-level result (full-118 accuracy ~= base-26 accuracy, every family-ablation delta
# inside noise) is the real headline; this threshold is calibrated to only pass features
# with a genuinely above-typical importance, not merely nonzero importance.
CV_THRESHOLD = 0.6
MI_TOP_N = 15
REDUNDANCY_THRESHOLD = 0.95


def main():
    with open(os.path.join(OUT, "v3_importance_mi.json")) as f:
        imi = json.load(f)
    with open(os.path.join(OUT, "v3_family_ablation.json")) as f:
        ablation = json.load(f)
    with open(os.path.join(DS, "columns.json")) as f:
        cols = json.load(f)
    cand_cols = cols["cand_cols"]

    X = pd.DataFrame(np.load(os.path.join(DS, "X_v3.npy")), columns=cols["all_cols"])
    corr = X[cand_cols].corr().abs()
    corr_arr = corr.to_numpy(copy=True)
    np.fill_diagonal(corr_arr, 0)
    corr = pd.DataFrame(corr_arr, index=corr.index, columns=corr.columns)

    meta_imp = imi["meta_importance"]
    mi_meta = imi["mi_meta"]
    mi_rank = {k: r for r, k in enumerate(sorted(mi_meta, key=lambda k: -mi_meta[k]))}

    # column -> family, for the ablation cross-reference
    col_to_family = {}
    for fam, fam_cols in __import__("research.v3_family_ablation", fromlist=["FAMILIES"]).FAMILIES.items():
        for c in fam_cols:
            col_to_family[c] = fam

    decisions = {}
    for c in cand_cols:
        imp = meta_imp.get(c, {}).get("mean_importance", 0.0)
        cv = meta_imp.get(c, {}).get("cv_importance", float("nan"))
        mi = mi_meta.get(c, 0.0)
        rank = mi_rank.get(c, 999)
        fam = col_to_family.get(c, "?")
        fam_delta = ablation["families"].get(fam, {}).get("delta_vs_full118")

        reasons = []
        keep = False
        if imp >= IMPORTANCE_THRESHOLD and (np.isnan(cv) or cv <= CV_THRESHOLD):
            keep = True
            reasons.append(f"catboost_importance={imp:.3f} (>= {IMPORTANCE_THRESHOLD}), stable (cv={cv:.2f})")
        elif rank < MI_TOP_N:
            keep = True
            reasons.append(f"top-{MI_TOP_N} by meta MI (rank {rank}, MI={mi:.4f}) despite catboost_importance={imp:.3f}")
        else:
            reasons.append(f"catboost_importance={imp:.3f} (<{IMPORTANCE_THRESHOLD}), MI rank {rank} (outside top {MI_TOP_N})")

        decisions[c] = {"keep": keep, "catboost_importance": imp, "cv_importance": cv,
                         "mi_meta": mi, "mi_rank": rank, "family": fam, "family_ablation_delta": fam_delta,
                         "reasons": reasons}

    # redundancy pass: within each >0.95-correlated pair, drop the lower-importance one
    for c in cand_cols:
        if not decisions[c]["keep"]:
            continue
        partner = corr[c].idxmax()
        val = corr[c].max()
        if val > REDUNDANCY_THRESHOLD and decisions.get(partner, {}).get("keep"):
            if decisions[c]["catboost_importance"] <= decisions[partner]["catboost_importance"]:
                decisions[c]["keep"] = False
                decisions[c]["reasons"].append(f"redundant with {partner} (corr={val:.3f}), lower importance, dropped")

    survivors = [c for c in cand_cols if decisions[c]["keep"]]
    dropped = [c for c in cand_cols if not decisions[c]["keep"]]

    result = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "thresholds": {"importance": IMPORTANCE_THRESHOLD, "cv": CV_THRESHOLD,
                        "mi_top_n": MI_TOP_N, "redundancy": REDUNDANCY_THRESHOLD},
        "n_candidates": len(cand_cols), "n_survivors": len(survivors), "n_dropped": len(dropped),
        "survivors": survivors, "dropped": dropped, "decisions": decisions,
        "primary_acc_full118": float(np.mean([fm["acc"] for fm in imi["primary_fold_metrics"]])),
        "meta_acc_full118": float(np.mean([fm["acc"] for fm in imi["meta_fold_metrics"]])),
    }
    out_path = os.path.join(OUT, "v3_feature_survivors.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"survivors: {len(survivors)}/{len(cand_cols)}")
    for c in survivors:
        d = decisions[c]
        print(f"  KEEP {c}: imp={d['catboost_importance']:.3f} cv={d['cv_importance']:.2f} "
              f"mi_rank={d['mi_rank']} family={d['family']}")
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
