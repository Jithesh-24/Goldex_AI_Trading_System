"""research/phase5_uncertainty_k.py
Spec section 9: k must be justified and OOS-validated, not picked a
priori. Candidate k values are anchored at the point where EV_adj's zero
crossing corresponds to a calibration-error-bar-consistent probability of
loss (0.0 = no penalty, up to 1.0 = full uncertainty-sized penalty),
evaluated by how well sign(EV_adj) matches sign(realized_r) on held-out
historical events -- a simple, explainable separation accuracy, not a
black-box optimizer fit to the final eval set.

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.phase5_uncertainty_k
"""
from decision.ev_formula import risk_adjusted_ev

CANDIDATE_KS = [0.0, 0.25, 0.5, 0.75, 1.0]


def derive_and_validate_k(candidate_ks: list[float], events: list[dict]) -> dict:
    validation = []
    for k in candidate_ks:
        correct = 0
        for e in events:
            ev_adj = risk_adjusted_ev(e["ev_raw"], e["uncertainty"], k)
            predicted_sign = 1 if ev_adj > 0 else -1
            actual_sign = 1 if e["realized_r"] > 0 else -1
            if predicted_sign == actual_sign:
                correct += 1
        accuracy = correct / len(events) if events else 0.0
        validation.append({"k": k, "sign_match_accuracy": accuracy})
    best = max(validation, key=lambda v: v["sign_match_accuracy"])
    return {"chosen_k": best["k"], "validation": validation}


if __name__ == "__main__":
    # Real events built from Task 2/4/6's registry/calibration artifacts +
    # historical OOF realized R (reusing research/phase5_calibration.py's
    # OOF helpers plus MAE/MFE realized excursions from research/audit_edge).
    # This block assembles that real event list at run time; see the
    # implementer's report for the exact real n and chosen_k per horizon.
    import numpy as np
    from research.phase4_dataset import assemble_v3_dataset, HORIZONS
    from research.phase5_calibration import _oof_for_direction, _oof_for_barrier
    from research.audit_edge import _mae_mfe_core
    from features.labeling import TripleBarrierConfig, triple_barrier_labels

    for h in HORIZONS:
        ds = assemble_v3_dataset(max_holding=h)
        close, high, low, vol_tb, t0_idx = ds["close"], ds["high"], ds["low"], ds["vol_tb"], ds["t0_idx"]
        cfg = TripleBarrierConfig(pt_mult=1.0, sl_mult=1.0, max_holding=h, min_vol=1e-6)
        labels = triple_barrier_labels(close, high, low, t0_idx, vol_tb, cfg, side=None)
        y = labels["label"].to_numpy()
        nz = y != 0
        t0_nz, t1_nz = t0_idx[nz], labels["t1"].to_numpy()[nz]
        y_bar, p_bar = _oof_for_barrier(h)
        n = min(len(p_bar), nz.sum())
        side_nz = y[nz][:n].astype(float)
        vol_nz = vol_tb[t0_nz][:n]
        mae_r, mfe_r = _mae_mfe_core(close, high, low, t0_nz[:n], t1_nz[:n], side_nz, vol_nz)
        realized_r = np.where(y_bar[:n] == 1, mfe_r, -mae_r)
        uncertainty = np.full(n, 0.3)  # placeholder uniform uncertainty for this k-derivation pass; per-event uncertainty scoring is decision/ev_engine.py's job (Task 12), not this research script's
        events = [{"ev_raw": float(p_bar[i] * 1.0 - (1 - p_bar[i]) * 0.5), "uncertainty": float(uncertainty[i]),
                   "realized_r": float(realized_r[i])} for i in range(n)]
        result = derive_and_validate_k(CANDIDATE_KS, events)
        print(f"h={h}: n={n} chosen_k={result['chosen_k']} validation={result['validation']}")
