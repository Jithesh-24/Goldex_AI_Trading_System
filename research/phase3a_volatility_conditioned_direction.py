"""research/phase3a_volatility_conditioned_direction.py
Phase 3A Section C: statistics probe (NOT a candidate). Computes
P(next-bar-direction) unconditionally vs. conditional on a volatility-regime
bin, and conditional on volatility-regime + recent path direction, using
real training-partition data only (first 300k rows, same convention as
research/phase3_real_run.py). Reports sample counts per bin -- small bins
are not evidence.

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 research/phase3a_volatility_conditioned_direction.py
"""
import numpy as np
import pandas as pd

TRAINING_ROWS = 300_000
DATA_PATH = "data/gold_seed_merged_full6yr.csv"
VOL_WINDOW = 30
N_VOL_BINS = 3
PATH_LOOKBACK = 5
MIN_BIN_COUNT = 500


def load_training_closes():
    df = pd.read_csv(DATA_PATH, parse_dates=["time"])
    return df.iloc[:TRAINING_ROWS]["close"].to_numpy(dtype=np.float64)


def compute_arrays(closes):
    returns = np.diff(closes, prepend=closes[0])
    n = len(closes)
    vol = np.full(n, np.nan)
    for i in range(VOL_WINDOW, n):
        vol[i] = np.std(returns[i - VOL_WINDOW:i])
    next_dir = np.full(n, np.nan)
    next_dir[:-1] = np.sign(returns[1:])
    recent_dir = np.full(n, np.nan)
    for i in range(PATH_LOOKBACK, n):
        recent_dir[i] = np.sign(closes[i] - closes[i - PATH_LOOKBACK])
    return vol, next_dir, recent_dir


def bin_series(x, n_bins):
    finite = x[np.isfinite(x)]
    edges = np.unique(np.quantile(finite, np.linspace(0, 1, n_bins + 1)))
    bins = np.full(len(x), np.nan)
    valid = np.isfinite(x)
    bins[valid] = np.clip(np.digitize(x[valid], edges[1:-1]), 0, len(edges) - 2)
    return bins


def p_up(direction_values):
    direction_values = direction_values[np.isfinite(direction_values)]
    n = len(direction_values)
    if n == 0:
        return None, 0
    return float(np.mean(direction_values > 0)), n


def main():
    closes = load_training_closes()
    vol, next_dir, recent_dir = compute_arrays(closes)
    vol_bin = bin_series(vol, N_VOL_BINS)

    print(f"N training bars: {len(closes)} (rows 0:{TRAINING_ROWS})")
    p, n = p_up(next_dir)
    print(f"\nUnconditional P(next bar up): {p:.4f}  (n={n})")

    print(f"\nP(next bar up | volatility regime bin), {N_VOL_BINS} quantile bins of {VOL_WINDOW}-bar realized vol:")
    for b in range(N_VOL_BINS):
        mask = vol_bin == b
        p, n = p_up(next_dir[mask])
        flag = "" if n >= MIN_BIN_COUNT else "  ** LOW SAMPLE COUNT, not evidence **"
        print(f"  vol_bin={b}: P(up)={p:.4f}  n={n}{flag}")

    print(f"\nP(next bar up | volatility regime bin + recent {PATH_LOOKBACK}-bar direction):")
    for b in range(N_VOL_BINS):
        for d, label in [(1.0, "up"), (-1.0, "down"), (0.0, "flat")]:
            mask = (vol_bin == b) & (recent_dir == d)
            p, n = p_up(next_dir[mask])
            if n == 0:
                continue
            flag = "" if n >= MIN_BIN_COUNT else "  ** LOW SAMPLE COUNT, not evidence **"
            print(f"  vol_bin={b}, recent_dir={label}: P(up)={p:.4f}  n={n}{flag}")


if __name__ == "__main__":
    main()
