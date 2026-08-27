"""research/genesis_horizon_sweep.py
Genesis reset, Section 30: the horizon sweep.

Every one of the 26 prior hypotheses (Phase 3, 3A, 4) used a fixed 5-bar
forward-return target. That variable itself was never tested. This script
sweeps the forward-return horizon across several values and re-runs the
exact same MI-vs-shuffled-null estimator already validated in Phase 3A
against the representations already computed/validated in Phase 3A and
Phase 4 -- reusing that code directly, not reimplementing it.

DATA USED: data/gold_seed_merged_full6yr.csv rows 0:300,000 (training
partition only -- same convention as Phase 3/3A/4; rows 300,000:400,000,
the real Phase 3 validation split, are never read here).

REPRESENTATION: seven representations already validated in Phase 3A/4:
  1. momentum scalar (Phase 3A, research.phase3a_representation_experiments)
  2. raw path window projection (Phase 3A)
  3. multi-scale volatility ratio (Phase 3A)
  4. volatility-regime transition (Phase 3A) -- also used here as the
     trend-invariant confound-diagnostic reference series
  5. GARCH(1,1) conditional variance (Phase 4, fit_garch11, reused unchanged)
  6. Kalman filtered velocity + innovation (Phase 4, kalman_level_trend_filter,
     reused unchanged)
  7. rolling skew/kurtosis (Phase 4, _rolling_moment, reused unchanged)

MODEL: none. This is a marginal-MI sweep exactly matching Phase 3A's
Section B methodology (binned_mutual_information + mi_with_shuffle_control,
imported unchanged from research/phase3a_representation_experiments.py). It
is intentionally NOT a full OOS predictive-modeling check (that pattern is
Phase 3A's Section D / Phase 4's OOS-check) -- the point of a cheap 7x6=42
cell sweep is to identify which cells are even worth a proper OOS check, not
to run 42 full OOS checks.

TARGET: forward_return(closes, horizon=H) for H in HORIZONS, identical
construction to Phase 3A/4, just varying H.

TRAIN PERIOD: rows 0:300,000 of the training partition only.

CONFOUND CHECK: Phase 3A/4 already identified a trend confound -- a long
secular price drift inflates MI for any trend-correlated feature/target pair
even with zero local predictability, and the trend-invariant
volatility-regime-transition representation showed ~2 orders of magnitude
less MI than the trend-sensitive ones as a result. For every cell where real
MI clears null_mi_mean + 3*null_mi_std, this script classifies it as
trend-confounded unless it beats the regime-transition MI at that same
horizon by a wide margin AND the raw return autocorrelation at that horizon
is not itself near zero (reusing analyze_return_autocorrelation from
research/phase3_representation_research.py). Every cleared cell is labeled:
  (a) null-consistent  -- doesn't clear null
  (b) large-MI-but-likely-trend-confounded
  (c) large-MI-and-not-explained-by-trend-confound -- candidate for a
      follow-up OOS check (NOT run here; out of scope for this script)

LIMITATION: marginal MI only, no OOS predictive check, single fixed-seed
20-permutation shuffle null per cell, single lookback/window choice per
representation (not itself swept), from-scratch GARCH/Kalman implementations
(same limitation already disclosed in Phase 4).

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 research/genesis_horizon_sweep.py
"""
import numpy as np
import pandas as pd

from research.phase3a_representation_experiments import (
    TRAINING_ROWS, DATA_PATH, N_BINS, RNG_SEED,
    MOMENTUM_LOOKBACK, PATH_WINDOW, VOL_WINDOWS,
    binned_mutual_information, mi_with_shuffle_control,
    momentum_scalar, path_pca_projection, multiscale_vol_summary,
    vol_regime_transition,
)
from research.phase4_garch_volatility_mechanism import fit_garch11
from research.phase4_kalman_trend_mechanism import kalman_level_trend_filter
from research.phase4_distributional_mechanism import _rolling_moment, WINDOW as DIST_WINDOW
from research.phase3_representation_research import analyze_return_autocorrelation

HORIZONS = (1, 5, 15, 30, 60, 120)
CONFOUND_CLEAR_SIGMAS = 3.0
CONFOUND_MARGIN_MULT = 5.0  # real MI must exceed regime-transition MI by this much to not be "likely confounded"


def load_training_closes():
    df = pd.read_csv(DATA_PATH, parse_dates=["time"])
    df_training = df.iloc[:TRAINING_ROWS].reset_index(drop=True)
    return df_training["close"].to_numpy(dtype=np.float64)


def forward_return(closes, horizon):
    fwd = np.full(len(closes), np.nan)
    fwd[:-horizon] = closes[horizon:] - closes[:-horizon]
    return fwd


def build_representations(closes, returns):
    """Build every representation once (representations do not depend on
    the horizon -- only the target does), reusing Phase 3A/4 code unchanged."""
    reps = {}
    reps["momentum_scalar"] = momentum_scalar(closes, MOMENTUM_LOOKBACK)
    reps["raw_path_window_projection"] = path_pca_projection(closes, PATH_WINDOW)
    vol_ratio, vols = multiscale_vol_summary(closes, VOL_WINDOWS)
    reps["multiscale_volatility_ratio"] = vol_ratio
    regime_transition = vol_regime_transition(vols[min(VOL_WINDOWS)])
    reps["volatility_regime_transition"] = regime_transition

    (omega, alpha, beta), sigma2 = fit_garch11(returns)
    reps["garch11_conditional_variance"] = sigma2

    typical_move_var = float(np.var(np.diff(closes)))
    _, velocities, innovations = kalman_level_trend_filter(
        closes, process_var_level=typical_move_var * 0.1,
        process_var_velocity=typical_move_var * 1e-4, obs_var=typical_move_var,
    )
    reps["kalman_filtered_velocity"] = velocities
    reps["kalman_innovation"] = innovations

    reps["rolling_skew"] = _rolling_moment(returns, DIST_WINDOW, order=3)
    reps["rolling_excess_kurtosis"] = _rolling_moment(returns, DIST_WINDOW, order=4)

    return reps, regime_transition, {"garch": (omega, alpha, beta)}


def classify_cell(real_mi, null_mean, null_std, regime_mi_at_horizon, autocorr_near_zero):
    clears_null = real_mi > (null_mean + CONFOUND_CLEAR_SIGMAS * null_std)
    if not clears_null:
        return "a_null_consistent"
    # regime-transition is the trend-invariant reference; if this cell's
    # real MI isn't well above it, or the target itself still shows
    # near-zero raw-return autocorrelation at this horizon (i.e. no real
    # local structure for the confound to be "explaining away" a genuine
    # signal), treat it as trend-confounded rather than genuine.
    beats_regime_by_margin = real_mi > (regime_mi_at_horizon * CONFOUND_MARGIN_MULT)
    if autocorr_near_zero or not beats_regime_by_margin:
        return "b_large_mi_likely_trend_confounded"
    return "c_large_mi_not_explained_by_trend_confound"


def main():
    closes = load_training_closes()
    returns = np.diff(closes, prepend=closes[0])
    print(f"Loaded {len(closes)} training closes from {DATA_PATH} (rows 0:{TRAINING_ROWS})")

    reps, regime_transition, fit_info = build_representations(closes, returns)
    print(f"Fitted GARCH(1,1): omega={fit_info['garch'][0]:.6g} alpha={fit_info['garch'][1]:.4f} "
          f"beta={fit_info['garch'][2]:.4f}")

    autocorr_by_horizon = {}
    for h in HORIZONS:
        ac = analyze_return_autocorrelation(closes, max_lag=min(h, 20))
        lag_key = f"lag_{min(h, 20)}"
        autocorr_by_horizon[h] = ac.get(lag_key, 0.0)

    rows = []
    for h in HORIZONS:
        fwd = forward_return(closes, h)
        regime_mi_h = mi_with_shuffle_control(regime_transition, fwd, n_bins=N_BINS, n_shuffles=20, seed=RNG_SEED)
        regime_real_mi = regime_mi_h["real_mi_nats"]
        autocorr_near_zero = abs(autocorr_by_horizon[h]) < 0.05
        for name, series in reps.items():
            stats = mi_with_shuffle_control(series, fwd, n_bins=N_BINS, n_shuffles=20, seed=RNG_SEED)
            classification = classify_cell(
                stats["real_mi_nats"], stats["null_mi_mean"], stats["null_mi_std"],
                regime_real_mi, autocorr_near_zero,
            )
            rows.append({
                "representation": name,
                "horizon": h,
                "real_mi_nats": stats["real_mi_nats"],
                "null_mi_mean": stats["null_mi_mean"],
                "null_mi_std": stats["null_mi_std"],
                "null_mi_max": stats["null_mi_max"],
                "regime_transition_mi_same_horizon": regime_real_mi,
                "raw_return_autocorr_at_horizon": autocorr_by_horizon[h],
                "classification": classification,
            })

    df_results = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    print("\n" + df_results.to_string(index=False))

    n_c = int((df_results["classification"] == "c_large_mi_not_explained_by_trend_confound").sum())
    n_b = int((df_results["classification"] == "b_large_mi_likely_trend_confounded").sum())
    n_a = int((df_results["classification"] == "a_null_consistent").sum())
    print(f"\nCategory counts: a_null_consistent={n_a} b_trend_confounded={n_b} c_candidate={n_c}")
    if n_c > 0:
        print("\nCategory (c) candidate cells (real signal, not explained by trend confound):")
        print(df_results[df_results["classification"] == "c_large_mi_not_explained_by_trend_confound"]
              [["representation", "horizon", "real_mi_nats", "null_mi_mean", "null_mi_std"]].to_string(index=False))
    else:
        print("\nNo category (c) cells found -- the sweep is uniformly null/confounded.")

    return df_results


if __name__ == "__main__":
    main()
