"""
Phase 3B Parts 3/4/5 -- conditional MAE/MFE distributions, TP/SL/timeout
probability, and expected payoff/loss, conditioned on calibrated-probability
decile x direction x volatility state (and 2-way interactions where sample
size supports it). Uses the existing 212,108-event v2 dataset
(research/output/mae_mfe_dataset.csv) -- no new model, no new data pull.

Research/reporting only. Does not touch production SL/TP.

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.conditional_distributions
"""
import json
import os
import time

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "research", "output")
MIN_N = 300  # below this, a conditional cell is too thin to report a quantile


def quantiles(s, qs=(0.5, 0.75, 0.9, 0.95)):
    return {f"p{int(q*100)}": float(s.quantile(q)) for q in qs}


def cell_stats(sub):
    n = len(sub)
    if n < MIN_N:
        return {"n": n, "note": f"below MIN_N={MIN_N}, not reported"}
    wins = sub[sub.touch == "TP"]
    losses = sub[sub.touch == "SL"]
    return {
        "n": int(n),
        "mae_R": quantiles(sub.mae_R), "mfe_R": quantiles(sub.mfe_R),
        "p_tp": float((sub.touch == "TP").mean()),
        "p_sl": float((sub.touch == "SL").mean()),
        "p_timeout": float((sub.touch == "TIMEOUT").mean()),
        "expected_win_R_realized": float(wins.mfe_R.mean()) if len(wins) >= 10 else None,  # informational -- realized win is exactly tp_R by barrier construction
        "expected_loss_mae_R": float(losses.mae_R.mean()) if len(losses) >= 10 else None,
        "mean_holding_bars": float(sub.holding_bars.mean()),
        "naive_ev_R_at_1.5_1.0": float((sub.touch == "TP").mean() * 1.5 - (sub.touch == "SL").mean() * 1.0),
    }


def main():
    df = pd.read_csv(os.path.join(OUT, "mae_mfe_dataset.csv"))
    df["cal_decile"] = pd.qcut(df["calibrated_proba"], 10, labels=False, duplicates="drop")
    result = {"n_events": int(len(df)), "min_cell_n": MIN_N,
              "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    result["by_decile"] = {int(g): cell_stats(sub) for g, sub in df.groupby("cal_decile")}
    result["by_direction"] = {g: cell_stats(sub) for g, sub in df.groupby("direction")}
    result["by_vol_state"] = {g: cell_stats(sub) for g, sub in df.groupby("vol_state")}
    result["by_decile_x_vol_state"] = {
        f"{d}|{v}": cell_stats(sub) for (d, v), sub in df.groupby(["cal_decile", "vol_state"])
    }
    result["by_decile_x_direction"] = {
        f"{d}|{dir_}": cell_stats(sub) for (d, dir_), sub in df.groupby(["cal_decile", "direction"])
    }
    result["by_vol_state_x_direction"] = {
        f"{v}|{dir_}": cell_stats(sub) for (v, dir_), sub in df.groupby(["vol_state", "direction"])
    }

    out_path = os.path.join(OUT, "conditional_distributions.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print("== P(TP)/P(SL)/P(timeout) by vol_state ==")
    for k, v in result["by_vol_state"].items():
        if "note" in v:
            print(f"  {k}: {v['note']}")
            continue
        print(f"  {k}: n={v['n']:>7,} p_tp={v['p_tp']:.3f} p_sl={v['p_sl']:.3f} "
              f"p_timeout={v['p_timeout']:.3f} mae_p90={v['mae_R']['p90']:.3f} mfe_p90={v['mfe_R']['p90']:.3f} "
              f"ev={v['naive_ev_R_at_1.5_1.0']:+.4f}R")
    print("\n== decile x vol_state EV (thin cells noted) ==")
    for k, v in result["by_decile_x_vol_state"].items():
        if "note" in v:
            continue
        print(f"  {k}: n={v['n']:>6,} p_tp={v['p_tp']:.3f} ev={v['naive_ev_R_at_1.5_1.0']:+.4f}R "
              f"mae_p90={v['mae_R']['p90']:.3f} mfe_p90={v['mfe_R']['p90']:.3f}")
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
