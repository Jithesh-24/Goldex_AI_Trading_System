"""
Phase 3B Part 2 step 7 -- walk-forward family ablation. For each of the 10
NEW candidate families (A-J), drop that family's columns and rerun primary
OOF; the delta vs the full-118 baseline (research/output/v3_importance_mi.json)
is the family's incremental contribution. Uses reduced CatBoost iterations
(faster screening pass -- final survivor-set validation in
v3_final_comparison.py uses full settings) since this is 10 extra full
walk-forward runs.

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.v3_family_ablation
"""
import json
import os
import time

import numpy as np
import pandas as pd

from learning.train import CATBOOST_KW
import research.audit_edge as audit_edge
from research.audit_edge import oof_run

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "research", "output")
DS = os.path.join(OUT, "v3_dataset")

# screening-speed CatBoost config -- fewer iterations/earlier stop than production CATBOOST_KW
SCREEN_KW = dict(CATBOOST_KW)
SCREEN_KW.update(iterations=600, early_stopping_rounds=40)

FAMILIES = {
    "A_return_dynamics": ["ret_240", "sign_ret_240", "ret_accel_5_15", "ret_decel_15_60",
                           "run_length_signed", "return_autocorr_20", "return_autocorr_60",
                           "return_pacf1_60", "sign_flip_rate_20", "rolling_mean_ret_20",
                           "rolling_median_ret_20", "return_dispersion_20",
                           "upside_downside_asymmetry_60", "return_skew_60", "return_kurt_60",
                           "return_skew_240", "return_percentile_rank_60",
                           "return_quantile_pos_240", "directional_entropy_60"],
    "B_volatility": ["realized_variance_20", "realized_semivar_upside_20",
                      "realized_semivar_downside_20", "parkinson_vol_60", "vol_acceleration_30",
                      "vol_of_vol_60", "vol_percentile_252", "vol_zscore_60", "vol_compression_ratio"],
    "C_jump_change": ["cusum_distance_to_threshold", "jump_intensity_60", "jump_magnitude_mean_60",
                       "jump_direction_bias_60", "bars_since_last_changepoint",
                       "changepoint_intensity_240", "vol_shock_zscore"],
    "D_distribution_info": ["tail_probability_60", "shannon_entropy_returns_60",
                             "permutation_entropy_60", "sample_entropy_20",
                             "return_concentration_60", "mi_proxy_sign_lag5_240"],
    "E_market_geometry": ["dist_from_high_20", "dist_from_low_20", "range_position_20",
                           "range_position_60", "range_width_20", "range_width_ratio_20_60",
                           "displacement_from_equilibrium_60", "breakout_magnitude_20",
                           "breakout_failure_magnitude_20", "reversal_frequency_60",
                           "avg_run_length_60", "excursion_from_recent_distribution_20",
                           "high_low_density_60"],
    "F_mean_reversion": ["hurst_240", "mean_reversion_speed_60", "half_life_60",
                          "autocorr_decay_rate_60", "persistence_score",
                          "residual_mean_reversion_60", "fracdiff_slope_60"],
    "G_time_session": ["hour_sin", "hour_cos", "minute_sin", "minute_cos", "dow_sin", "dow_cos",
                        "session_asian", "session_london", "session_ny",
                        "session_london_ny_overlap", "session_transition_flag",
                        "vol_conditional_on_session", "ret_conditional_on_session",
                        "activity_conditional_on_session"],
    "H_microstructure": ["tick_volume_zscore_60", "tick_volume_accel_20", "spread_change_1",
                          "spread_percentile_252", "spread_volatility_60", "tick_volume_spread_ratio"],
    "I_regime_state": ["vol_state_tercile", "jump_state", "persistence_state", "entropy_state",
                        "activity_state", "changepoint_state", "composite_state_id"],
    "J_first_passage": ["hist_p_reach_10bps_10b_60", "hist_time_to_10bps_60",
                         "hist_barrier_hit_freq_60", "hist_path_asymmetry_60"],
}


def _fit_screen(X, y, train_pos):
    from catboost import CatBoostClassifier
    cut = int(len(train_pos) * (1 - 0.15))
    tr, va = train_pos[:cut], train_pos[cut:]
    model = CatBoostClassifier(**SCREEN_KW)
    model.fit(X.iloc[tr], y.iloc[tr], eval_set=(X.iloc[va], y.iloc[va]))
    return model


def main():
    t_start = time.time()
    with open(os.path.join(DS, "columns.json")) as f:
        cols = json.load(f)
    X = pd.DataFrame(np.load(os.path.join(DS, "X_v3.npy")), columns=cols["all_cols"])
    y_bin = pd.Series(np.load(os.path.join(DS, "y_bin.npy")))
    t0 = pd.Series(np.load(os.path.join(DS, "t0.npy")))
    t1 = pd.Series(np.load(os.path.join(DS, "t1.npy")))

    all_family_cols = set()
    for fam_cols in FAMILIES.values():
        all_family_cols.update(fam_cols)
    missing = all_family_cols - set(X.columns)
    assert not missing, f"family list references missing columns: {missing}"

    # monkey-patch audit_edge's _fit to the faster screening config for this script's runs only
    # -- baseline is ALSO rerun under SCREEN_KW here (not reused from v3_importance_mi.json's
    # full-CATBOOST_KW run) so every accuracy in this comparison uses the identical training
    # budget; comparing a full-iterations baseline against reduced-iterations ablations would
    # confound "family removed" with "trained less", an apples-to-oranges bug caught before running.
    orig_fit = audit_edge._fit
    audit_edge._fit = _fit_screen

    results = {}
    try:
        print("== full-118 baseline, rerun under SCREEN_KW for a fair comparison ==")
        r0 = oof_run(X, y_bin, t0, t1, tag="ablate-baseline-full118", want_importance=False)
        baseline_acc = float(np.mean([fm["acc"] for fm in r0["fold_metrics"]]))
        print(f"full-118 baseline (SCREEN_KW) primary acc: {baseline_acc:.4f}")

        for fam, fam_cols in FAMILIES.items():
            reduced_cols = [c for c in X.columns if c not in fam_cols]
            Xr = X[reduced_cols]
            r = oof_run(Xr, y_bin, t0, t1, tag=f"ablate-{fam}", want_importance=False)
            acc = float(np.mean([fm["acc"] for fm in r["fold_metrics"]]))
            delta = acc - baseline_acc
            results[fam] = {"n_cols_dropped": len(fam_cols), "acc_without_family": acc,
                             "delta_vs_full118": delta}
            print(f"  {fam} (-{len(fam_cols)} cols): acc={acc:.4f} delta={delta:+.4f}pp_frac "
                  f"({'family HELPS' if delta < -0.0005 else ('family HURTS/noise' if delta > 0.0005 else 'negligible')})")
    finally:
        audit_edge._fit = orig_fit

    out_path = os.path.join(OUT, "v3_family_ablation.json")
    with open(out_path, "w") as f:
        json.dump({"baseline_acc_full118": baseline_acc, "families": results,
                    "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f, indent=2)
    print(f"\nsaved -> {out_path} ({time.time()-t_start:.1f}s)")


if __name__ == "__main__":
    main()
