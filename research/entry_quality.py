"""
Phase 3B Part 6 -- entry quality research. Uses the same 212,108 v2 events
(research/output/mae_mfe_dataset.csv) matched back to raw M1 bars (by
event_time -> bar index) to measure short-horizon path behaviour AFTER the
signal fires but BEFORE any entry decision: does immediate entry dominate
waiting, and how fast does the eventual move happen (vs fade)?

Quantities (causal, backward-looking from each horizon k, no change to live
entry logic):
  fav_k / adv_k       -- favorable/adverse excursion by bar k (truncated,
                          same R units as the full-horizon MAE/MFE columns)
  frac_mfe_captured_k -- fav_k / mfe_R (full-horizon) -- how much of the
                          eventual favorable move has already happened by k
  frac_mae_incurred_k -- adv_k / mae_R (full-horizon) -- how much of the
                          eventual adverse move has already happened by k

Research only -- does not change live entry timing.

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.entry_quality
"""
import json
import os
import time

import numba
import numpy as np
import pandas as pd

from learning.data import load_raw_m1

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "research", "output")
HORIZONS = [1, 3, 5, 10, 15, 20, 30]


@numba.njit(cache=True)
def _truncated_excursion(close, high, low, t0_idx, side, vol_at_t0, horizons, holding_bars):
    """fav/adv accumulate only up to each event's OWN barrier-resolution bar
    (holding_bars[e]) -- price action after the position would already have
    closed is not a real 'wait longer' option, so it must not leak into the
    excursion-by-k curve (a live position exits at the barrier, it doesn't
    keep riding the market). Horizons beyond holding_bars[e] just freeze at
    the value-at-resolution."""
    n = len(t0_idx)
    k = len(horizons)
    fav = np.zeros((n, k), dtype=np.float64)
    adv = np.zeros((n, k), dtype=np.float64)
    for e in range(n):
        t0 = t0_idx[e]
        s = side[e]
        p0 = close[t0]
        v = vol_at_t0[e] if vol_at_t0[e] > 1e-9 else 1e-9
        hold = holding_bars[e]
        best, worst = 0.0, 0.0
        hi = 0
        for j in range(t0 + 1, t0 + horizons[-1] + 1):
            if j >= len(close):
                break
            step = j - t0
            if step > hold:
                # frozen: fill all remaining horizon slots with the value as of resolution
                while hi < k:
                    fav[e, hi] = best / v
                    adv[e, hi] = -worst / v
                    hi += 1
                break
            if s >= 0:
                f = (high[j] - p0) / p0
                a = (low[j] - p0) / p0
            else:
                f = (p0 - low[j]) / p0
                a = (p0 - high[j]) / p0
            if f > best:
                best = f
            if a < worst:
                worst = a
            step = j - t0
            while hi < k and horizons[hi] == step:
                fav[e, hi] = best / v
                adv[e, hi] = -worst / v
                hi += 1
        while hi < k:
            fav[e, hi] = best / v
            adv[e, hi] = -worst / v
            hi += 1
    return fav, adv


def main():
    t_start = time.time()
    df = pd.read_csv(os.path.join(OUT, "mae_mfe_dataset.csv"), parse_dates=["event_time"])
    raw = load_raw_m1()
    close = raw["close"].to_numpy(dtype=np.float64)
    high = raw["high"].to_numpy(dtype=np.float64)
    low = raw["low"].to_numpy(dtype=np.float64)
    times = pd.to_datetime(raw["time"].to_numpy())
    idx_map = pd.Series(np.arange(len(times)), index=times)

    t0_idx = idx_map.reindex(df["event_time"]).to_numpy()
    ok = np.isfinite(t0_idx)
    print(f"matched {ok.sum():,}/{len(df):,} events back to raw bar index "
          f"({(~ok).sum():,} unmatched, dropped)")
    df = df.loc[ok].reset_index(drop=True)
    t0_idx = t0_idx[ok].astype(np.int64)
    side = np.where(df["direction"].to_numpy() == "BUY", 1.0, -1.0)
    vol_at_t0 = df["vol_at_event"].to_numpy(dtype=np.float64)
    holding_bars = df["holding_bars"].to_numpy(dtype=np.int64)
    horizons = np.array(HORIZONS, dtype=np.int64)

    fav, adv = _truncated_excursion(close, high, low, t0_idx, side, vol_at_t0, horizons, holding_bars)
    for i, k in enumerate(HORIZONS):
        df[f"fav_{k}b_R"] = fav[:, i]
        df[f"adv_{k}b_R"] = adv[:, i]
        df[f"frac_mfe_captured_{k}b"] = np.where(df["mfe_R"] > 1e-9, fav[:, i] / df["mfe_R"], np.nan)
        df[f"frac_mae_incurred_{k}b"] = np.where(df["mae_R"] > 1e-9, adv[:, i] / df["mae_R"], np.nan)

    result = {"n_events": int(len(df)), "horizons_bars": HORIZONS,
              "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    overall = {}
    for k in HORIZONS:
        overall[k] = {
            "mean_fav_R": float(df[f"fav_{k}b_R"].mean()), "mean_adv_R": float(df[f"adv_{k}b_R"].mean()),
            "frac_mfe_captured": float(df[f"frac_mfe_captured_{k}b"].mean()),
            "frac_mae_incurred": float(df[f"frac_mae_incurred_{k}b"].mean()),
            "fav_over_adv_ratio": float(df[f"fav_{k}b_R"].mean() / max(df[f"adv_{k}b_R"].mean(), 1e-9)),
        }
    result["overall_by_horizon"] = overall

    by_vs = {}
    for vs, sub in df.groupby("vol_state"):
        by_vs[vs] = {k: {"frac_mfe_captured": float(sub[f"frac_mfe_captured_{k}b"].mean()),
                          "frac_mae_incurred": float(sub[f"frac_mae_incurred_{k}b"].mean())}
                     for k in HORIZONS}
    result["by_vol_state"] = by_vs

    out_path = os.path.join(OUT, "entry_quality.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print("== immediate-entry vs waiting: fraction of eventual move already happened by bar k ==")
    print(f"{'k':>4} {'mean_fav_R':>11} {'mean_adv_R':>11} {'frac_MFE_capt':>14} {'frac_MAE_incur':>15} {'fav/adv':>8}")
    for k in HORIZONS:
        o = overall[k]
        print(f"{k:>4} {o['mean_fav_R']:>11.4f} {o['mean_adv_R']:>11.4f} "
              f"{o['frac_mfe_captured']:>14.3f} {o['frac_mae_incurred']:>15.3f} {o['fav_over_adv_ratio']:>8.3f}")
    print(f"\nsaved -> {out_path} ({time.time()-t_start:.1f}s)")


if __name__ == "__main__":
    main()
