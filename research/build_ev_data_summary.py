"""
Phase 3A step 8 -- EV research data prep. NOT an EV gate: descriptive
empirical conditional payoff/loss stats only, computed from
research/output/mae_mfe_dataset.csv, broken down by calibrated-probability
decile and volatility state. This is the input a future EV engine would
need (P(win), empirical E[win]/E[loss] instead of the assumed 1.5R/1.0R
constants) -- no decision threshold, no gate, production TP/SL unchanged.

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.build_ev_data_summary
"""
import json
import os
import time

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "research", "output")


def summarize(df, group_col):
    rows = []
    for g, sub in df.groupby(group_col):
        wins = sub[sub.touch == "TP"]
        losses = sub[sub.touch == "SL"]
        rows.append({
            group_col: g, "n": int(len(sub)),
            "p_tp": float((sub.touch == "TP").mean()),
            "p_sl": float((sub.touch == "SL").mean()),
            "p_timeout": float((sub.touch == "TIMEOUT").mean()),
            "expected_win_R": float(wins.mfe_R.mean()) if len(wins) else None,  # realized win = TP_R by construction; mfe_R shown for context
            "expected_win_R_assumed_TP": float(sub.tp_R.iloc[0]) if len(sub) else None,
            "expected_loss_R_assumed_SL": float(sub.sl_R.iloc[0]) if len(sub) else None,
            "realized_avg_mae_on_losses_R": float(losses.mae_R.mean()) if len(losses) else None,
            "realized_avg_mfe_on_wins_R": float(wins.mfe_R.mean()) if len(wins) else None,
            "naive_ev_R_at_assumed_1.5_1.0": float(
                (sub.touch == "TP").mean() * 1.5 - (sub.touch == "SL").mean() * 1.0
            ),
        })
    return rows


def main():
    df = pd.read_csv(os.path.join(OUT, "mae_mfe_dataset.csv"))
    df["cal_decile"] = pd.qcut(df["calibrated_proba"], 10, labels=False, duplicates="drop")

    by_decile = summarize(df, "cal_decile")
    by_vol_state = summarize(df, "vol_state")

    result = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "research/output/mae_mfe_dataset.csv",
        "n_events": int(len(df)),
        "note": "Descriptive only -- NOT an EV gate, NOT used to change production TP/SL "
                "(still the trained 1.5R/1.0R triple-barrier). naive_ev_R_at_assumed_1.5_1.0 "
                "uses the CURRENT production assumption for comparison; expected_win/loss_R "
                "columns are what a future empirical-payoff EV engine would use instead.",
        "by_calibrated_probability_decile": by_decile,
        "by_volatility_state": by_vol_state,
        "overall": summarize(df.assign(_all=1), "_all")[0],
    }
    out_path = os.path.join(OUT, "ev_data_summary.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"overall: n={result['overall']['n']:,} p_tp={result['overall']['p_tp']:.3f} "
          f"p_sl={result['overall']['p_sl']:.3f} naive_ev={result['overall']['naive_ev_R_at_assumed_1.5_1.0']:.4f}R")
    print("by calibrated-probability decile:")
    for r in by_decile:
        print(f"  decile={r['cal_decile']} n={r['n']:>6,} p_tp={r['p_tp']:.3f} "
              f"naive_ev={r['naive_ev_R_at_assumed_1.5_1.0']:+.4f}R")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
