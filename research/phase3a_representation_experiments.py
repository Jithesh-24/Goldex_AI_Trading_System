"""research/phase3a_representation_experiments.py
Phase 3A Section B: quantifies whether richer representations carry more
information about forward returns than Phase 3's engineered scalars did.
Uses ONLY the real training partition (first 300k rows of
data/gold_seed_merged_full6yr.csv, matching research/phase3_real_run.py's
convention) -- the Phase 3 validation split is never touched here.

All statistics use binned mutual information (a simple nonlinear-association
proxy that plain correlation misses) between a short forward return horizon
and four representations:
  1. Phase-3-style single momentum scalar (close[t] - close[t-N])
  2. A short raw price-path window (last WINDOW normalized closes)
  3. Multi-scale volatility (several lookback windows)
  4. Volatility-regime transition (change in discretized vol bin)

Each real MI value is reported next to a shuffled-target null control
computed with the identical estimator, so a real value can be judged
against what "no information" looks like for this exact test rather than
eyeballed.

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 research/phase3a_representation_experiments.py
"""
import numpy as np
import pandas as pd

TRAINING_ROWS = 300_000
DATA_PATH = "data/gold_seed_merged_full6yr.csv"

FORWARD_HORIZON = 5       # bars ahead for the forward-return target
MOMENTUM_LOOKBACK = 10    # Phase-3-style single momentum scalar lookback
PATH_WINDOW = 15          # raw price-path window length
VOL_WINDOWS = (10, 30, 100)  # multi-scale volatility lookbacks
N_BINS = 10               # bins used for discretizing continuous variables
RNG_SEED = 42


def load_training_closes():
    df = pd.read_csv(DATA_PATH, parse_dates=["time"])
    df_training = df.iloc[:TRAINING_ROWS].reset_index(drop=True)
    return df_training["close"].to_numpy(dtype=np.float64)


def forward_return(closes, horizon=FORWARD_HORIZON):
    fwd = np.full(len(closes), np.nan)
    fwd[:-horizon] = closes[horizon:] - closes[:-horizon]
    return fwd


def binned_mutual_information(x, y, n_bins=N_BINS):
    """Simple binned (histogram) mutual information estimate in nats,
    using equal-frequency (quantile) bins on both variables so relationships
    of any shape (not just linear) are captured. NaNs are dropped pairwise."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < n_bins * 5:
        return 0.0
    x_edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    y_edges = np.unique(np.quantile(y, np.linspace(0, 1, n_bins + 1)))
    if len(x_edges) < 2 or len(y_edges) < 2:
        return 0.0
    x_bins = np.clip(np.digitize(x, x_edges[1:-1]), 0, len(x_edges) - 2)
    y_bins = np.clip(np.digitize(y, y_edges[1:-1]), 0, len(y_edges) - 2)
    joint = np.zeros((len(x_edges) - 1, len(y_edges) - 1))
    for xb, yb in zip(x_bins, y_bins):
        joint[xb, yb] += 1
    joint /= joint.sum()
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = joint / (px * py)
        terms = joint * np.log(ratio)
    terms[~np.isfinite(terms)] = 0.0
    return float(terms.sum())


def mi_with_shuffle_control(x, y, n_bins=N_BINS, n_shuffles=20, seed=RNG_SEED):
    real_mi = binned_mutual_information(x, y, n_bins)
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=np.float64)
    shuffle_mis = []
    for _ in range(n_shuffles):
        y_shuffled = rng.permutation(y)
        shuffle_mis.append(binned_mutual_information(x, y_shuffled, n_bins))
    return {
        "real_mi_nats": real_mi,
        "null_mi_mean": float(np.mean(shuffle_mis)),
        "null_mi_std": float(np.std(shuffle_mis)),
        "null_mi_max": float(np.max(shuffle_mis)),
        "n_shuffles": n_shuffles,
    }


def path_pca_projection(closes, window=PATH_WINDOW):
    """Collapse a raw normalized price-path window to a single scalar (first
    principal-component-like projection: mean-subtracted, unit-norm-weighted
    slope) so it can be compared with binned MI the same way as the other
    single-scalar representations. This deliberately loses information
    relative to using the full window in a real model (see Section D for
    that), but gives an apples-to-apples MI number here."""
    n = len(closes)
    proj = np.full(n, np.nan)
    for i in range(window, n):
        w = closes[i - window:i]
        w_norm = (w - w[0]) / w[0] if w[0] != 0 else w - w[0]
        # slope of a simple linear fit as the single summary scalar
        x_idx = np.arange(window)
        slope = np.polyfit(x_idx, w_norm, 1)[0]
        proj[i] = slope
    return proj


def multiscale_vol_summary(closes, windows=VOL_WINDOWS):
    """Multi-scale volatility collapsed to a single scalar: ratio of the
    shortest-window realized vol to the longest-window realized vol (a
    "vol-of-vol regime" summary), so this too can be compared via the same
    single-variable MI estimator."""
    n = len(closes)
    returns = np.diff(closes, prepend=closes[0])
    vols = {}
    for w in windows:
        vol = np.full(n, np.nan)
        for i in range(w, n):
            vol[i] = np.std(returns[i - w:i])
        vols[w] = vol
    short_w, long_w = min(windows), max(windows)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = vols[short_w] / vols[long_w]
    ratio[~np.isfinite(ratio)] = np.nan
    return ratio, vols


def vol_regime_transition(vol_short, n_bins=3):
    """Discretize short-window vol into n_bins regimes and compute the
    bin-to-bin transition (current regime minus previous regime) as a
    single scalar representing "volatility state transition"."""
    finite = vol_short[np.isfinite(vol_short)]
    if len(finite) < n_bins * 5:
        return np.full(len(vol_short), np.nan)
    edges = np.quantile(finite, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    regime = np.full(len(vol_short), np.nan)
    valid = np.isfinite(vol_short)
    regime[valid] = np.clip(np.digitize(vol_short[valid], edges[1:-1]), 0, len(edges) - 2)
    transition = np.full(len(vol_short), np.nan)
    transition[1:] = regime[1:] - regime[:-1]
    return transition


def momentum_scalar(closes, lookback=MOMENTUM_LOOKBACK):
    mom = np.full(len(closes), np.nan)
    mom[lookback:] = closes[lookback:] - closes[:-lookback]
    return mom


def main():
    closes = load_training_closes()
    print(f"Loaded {len(closes)} training closes from {DATA_PATH} (rows 0:{TRAINING_ROWS})")

    fwd = forward_return(closes, FORWARD_HORIZON)

    representations = {}
    representations["1_momentum_scalar_phase3_style"] = momentum_scalar(closes, MOMENTUM_LOOKBACK)
    representations["2_raw_path_window_projection"] = path_pca_projection(closes, PATH_WINDOW)
    vol_ratio, vols = multiscale_vol_summary(closes, VOL_WINDOWS)
    representations["3_multiscale_volatility_ratio"] = vol_ratio
    representations["4_volatility_regime_transition"] = vol_regime_transition(vols[min(VOL_WINDOWS)])

    print(f"\nForward horizon: {FORWARD_HORIZON} bars. N_BINS={N_BINS}. Shuffle control: 20 permutations.\n")
    print(f"{'representation':45s} {'real_MI(nats)':>14s} {'null_mean':>10s} {'null_std':>10s} {'null_max':>10s}")
    results = {}
    for name, repr_series in representations.items():
        stats = mi_with_shuffle_control(repr_series, fwd)
        results[name] = stats
        print(f"{name:45s} {stats['real_mi_nats']:14.6f} {stats['null_mi_mean']:10.6f} "
              f"{stats['null_mi_std']:10.6f} {stats['null_mi_max']:10.6f}")

    print("\nInterpretation: a representation's real_MI is only meaningfully above")
    print("noise if it clears null_mi_mean + a few null_mi_std, or exceeds null_mi_max.")
    return results


if __name__ == "__main__":
    main()
