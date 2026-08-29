"""
Phase 3B Part 11 -- EV surface (research only, no Telegram, no live threshold
change, no auto-trading). Combines:
  - conditional P(win)/P(loss) by calibrated-probability decile x vol_state
    (research/output/conditional_distributions.json)
  - candidate SL/TP geometries (research/output/dynamic_sltp_research.json)
  - a REAL execution-cost estimate, restricted to 2025-2026 events only --
    Phase 1A found spread is constant-filled/unreliable before ~2025 (only
    98.9% zero-filled historical spread), so cost-adjusted EV for older
    years would be fabricated precision. Nominal (no-cost) EV for the full
    2021-2026 history is reported separately, cost is layered on only where
    the underlying spread data is real.

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.ev_surface
"""
import json
import os
import time

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "research", "output")

REAL_SPREAD_POINTS = 25.0     # mean spread, xmlive-sourced rows (gold_seed.csv, src=="xmlive")
REAL_SPREAD_MEAN_PRICE = 4251.4  # mean close over those same rows
SPREAD_PRICE_DIST = REAL_SPREAD_POINTS * 0.01
COST_ERA_START_YEAR = 2025


def main():
    df = pd.read_csv(os.path.join(OUT, "mae_mfe_dataset.csv"), parse_dates=["event_time"])
    df["cal_decile"] = pd.qcut(df["calibrated_proba"], 10, labels=False, duplicates="drop")
    df["cost_R"] = (SPREAD_PRICE_DIST / REAL_SPREAD_MEAN_PRICE) / df["vol_at_event"].clip(lower=1e-9)
    recent = df[df["event_time"].dt.year >= COST_ERA_START_YEAR].copy()
    print(f"cost model: spread={REAL_SPREAD_POINTS}pts (~{SPREAD_PRICE_DIST:.2f} price) at mean price "
          f"{REAL_SPREAD_MEAN_PRICE:.1f} -> return-space cost {SPREAD_PRICE_DIST/REAL_SPREAD_MEAN_PRICE:.2e}, "
          f"applied only to {COST_ERA_START_YEAR}+ events (n={len(recent):,}/{len(df):,}) -- pre-{COST_ERA_START_YEAR} "
          f"spread is unreliable (Phase 1A finding), NOT cost-adjusted here.")

    def surface(sub, cost_col=None):
        rows = []
        for dec, g in sub.groupby("cal_decile"):
            wins = (g.touch == "TP").mean()
            losses = (g.touch == "SL").mean()
            nominal_ev = wins * 1.5 - losses * 1.0
            row = {"cal_decile": int(dec), "n": int(len(g)), "p_tp": float(wins), "p_sl": float(losses),
                   "nominal_ev_R": float(nominal_ev)}
            if cost_col:
                row["mean_cost_R"] = float(g[cost_col].mean())
                row["cost_adjusted_ev_R"] = float(nominal_ev - g[cost_col].mean())
            rows.append(row)
        return rows

    result = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cost_model": {"spread_points": REAL_SPREAD_POINTS, "mean_price": REAL_SPREAD_MEAN_PRICE,
                        "applied_from_year": COST_ERA_START_YEAR},
        "full_history_nominal_ev_by_decile": surface(df),
        "cost_adjusted_ev_by_decile_2025_2026_only": surface(recent, cost_col="cost_R"),
    }
    result["by_decile_x_vol_state_cost_adjusted_2025_2026"] = []
    for (dec, vs), g in recent.groupby(["cal_decile", "vol_state"]):
        if len(g) < 200:
            continue
        wins = (g.touch == "TP").mean()
        losses = (g.touch == "SL").mean()
        nominal = wins * 1.5 - losses * 1.0
        cost = g["cost_R"].mean()
        result["by_decile_x_vol_state_cost_adjusted_2025_2026"].append({
            "cal_decile": int(dec), "vol_state": vs, "n": int(len(g)), "p_tp": float(wins),
            "nominal_ev_R": float(nominal), "mean_cost_R": float(cost),
            "cost_adjusted_ev_R": float(nominal - cost),
        })

    out_path = os.path.join(OUT, "ev_surface.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print("\n== cost-adjusted EV by decile, 2025-2026 only (real spread era) ==")
    for r in result["cost_adjusted_ev_by_decile_2025_2026_only"]:
        print(f"  decile={r['cal_decile']} n={r['n']:>6,} p_tp={r['p_tp']:.3f} "
              f"nominal_ev={r['nominal_ev_R']:+.4f}R mean_cost={r['mean_cost_R']:.4f}R "
              f"cost_adj_ev={r['cost_adjusted_ev_R']:+.4f}R")
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
