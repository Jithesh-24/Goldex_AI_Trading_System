"""v8 PLACEMENT PRIOR FITTER — learn SL/TP placement from MFE/MFA excursions.

USER MANDATE (2026-08-07): "no hardcoded values… it should know where to keep
sl and tp not hardcoded it should learn everything from last 6 years of data."

WHAT THIS DOES:
  Streams the M5 matrix (gold_features_m5.csv), assigns regime_bin() per row,
  then for each regime × direction measures the MFE/MFA excursion
  distributions across ALL geometry blocks:
    - winners (target=1): how far price runs in our favor before TP
    - losers  (target=0): how far price dips against us before SL
  From those 6-year distributions it derives the LEARNED placement:
    sl_mult   = quantile of loser-MFA over the geometry grid where losers
                stop "dipping past SL on noise" (institutional: place SL
                beyond the typical adverse excursion of the REGIME)
    tp_ratio  = quantile of winner-MFE relative to that SL (capture the
                typical favorable run)
  Per regime, per direction → placement_prior.json. The engine reads this
  fresh per signal (like regime_dir_prior.json) instead of a hardcoded grid.

OUTPUT: models/placement_prior.json
  {
    "version": 8, "base_tf": "M5", "horizon_bars": 36,
    "regimes": {
      "STRONG_UP": {
        "BUY":  {"sl_atr": 2.1, "tp_ratio": 1.6, "mfe_p50": 3.2,
                 "mfa_p90": 2.6, "n_winners": 123456, "n_losers": 234567},
        "SELL": {...}
      }, ...
    },
    "global": {"sl_pct": 0.90, "tp_pct": 0.50, "note": "quantiles over winners/losers"}
  }

Usage: python fit_placement_prior.py
"""
import os, sys, json, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F

BASE = "/home/jith/.hermes/profiles/trading/scripts"
# v8.9c: PURE GOLD matrix — no macro features (1-day lag removed)
_pure = f"{BASE}/gold_features_m5_pure.csv"
_tick = f"{BASE}/gold_features_m5_tick.csv"
M5 = f"{BASE}/gold_features_m5.csv"
if os.path.exists(_pure):
    MATRIX = _pure
elif os.path.exists(_tick):
    MATRIX = _tick
else:
    MATRIX = M5
MODEL_DIR = f"{BASE}/models"
OUT = f"{MODEL_DIR}/placement_prior.json"
REGIMES = F.REGIME_NAMES  # MUST match features.py exactly ("HIGH_VOL", not "HIGH_VOL_SPIKE")
SL_PCT = 0.90    # SL beyond this share of WINNERS' adverse excursion (don't kill winners)
TP_PCT = 0.50    # TP at this share of WINNERS' favorable excursion (capture typical winner)
CHUNK = 500_000

def main():
    t0 = time.time()
    if not os.path.exists(MATRIX):
        print(f"❌ matrix not found: {MATRIX}")
        sys.exit(1)
    # per (regime, direction): for WINNERS collect favorable excursion (mfe)
    # and adverse dip survived (mfa); for LOSERS collect adverse run (mfa_l,
    # for the loss-cause band — how deep losers actually go before dying).
    acc = {r: {"BUY": {"mfe_w": [], "mfa_w": [], "mfa_l": []},
               "SELL": {"mfe_w": [], "mfa_w": [], "mfa_l": []}} for r in REGIMES}
    need = F.REGIME_KEYS + ["sl_atr_buy", "sl_atr_sell", "mfe_atr", "mfa_atr",
                            "target", "direction"]
    total = 0
    for chunk in pd.read_csv(MATRIX, usecols=need, chunksize=CHUNK):
        chunk = chunk.dropna(subset=["mfe_atr", "mfa_atr", "target", "direction"])
        if chunk.empty:
            continue
        r = chunk.to_dict("records")
        fx_rows = [{c: float(x[c]) for c in F.REGIME_KEYS} for x in r]
        bins = [F.regime_bin(fx) for fx in fx_rows]
        for row, b in zip(r, bins):
            if b not in acc:
                continue
            d = "BUY" if float(row["direction"]) == 1.0 else "SELL"
            tgt = float(row["target"])
            mfe = float(row["mfe_atr"]); mfa = float(row["mfa_atr"])
            if tgt == 1.0:
                acc[b][d]["mfe_w"].append(mfe)
                acc[b][d]["mfa_w"].append(mfa)
            else:
                acc[b][d]["mfa_l"].append(mfa)
        total += len(chunk)
        del chunk, r, fx_rows, bins
        print(f"  streamed {total:,} rows ({time.time()-t0:.0f}s)", flush=True)

    out = {"version": 8, "base_tf": "M5", "horizon_bars": 36,
           "sl_pct": SL_PCT, "tp_pct": TP_PCT,
           "note": "SL beyond p90 loser-MFA; TP at p50 winner-MFE (ATR units)",
           "built_at": time.time(), "regimes": {}}
    for rname in REGIMES:
        out["regimes"][rname] = {}
        for d in ("BUY", "SELL"):
            a = acc[rname][d]
            mfe_w = np.array(a["mfe_w"]); mfa_w = np.array(a["mfa_w"])
            mfa_l = np.array(a["mfa_l"])
            if len(mfe_w) < 200 or len(mfa_l) < 200:
                out["regimes"][rname][d] = {"n_winners": len(mfe_w),
                                            "n_losers": len(mfa_l),
                                            "sl_atr": None, "tp_ratio": None}
                continue
            # LEARNED SL (institutional): beyond the p90 of the ADVERSE dip
            # WINNERS survive. Loser-MFA is circular (losers die AT their SL),
            # winner-MFA is the true noise band — stops beyond it don't kill
            # trades that would have recovered to TP.
            sl_learned = float(np.quantile(mfa_w, SL_PCT))
            # LEARNED TP: capture the p50 favorable run of winners, expressed
            # as ratio of the learned SL (institutional RR from excursion data).
            mfe_p50 = float(np.quantile(mfe_w, TP_PCT))
            tp_ratio = max(mfe_p50 / sl_learned, 1.0) if sl_learned > 0 else 1.0
            out["regimes"][rname][d] = {
                "n_winners": len(mfe_w), "n_losers": len(mfa_l),
                "sl_atr": round(sl_learned, 3),
                "tp_ratio": round(tp_ratio, 3),
                "mfe_p50": round(mfe_p50, 3),
                "mfa_w_p90": round(sl_learned, 3),
                "mfa_l_p90": round(float(np.quantile(mfa_l, 0.90)), 3),
                "mfa_l_p50": round(float(np.quantile(mfa_l, 0.50)), 3),
            }
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"✅ placement_prior.json written ({time.time()-t0:.0f}s)")
    for rname in REGIMES:
        row = out["regimes"][rname]
        b = row.get("BUY", {}); s = row.get("SELL", {})
        print(f"  {rname:14s} BUY  sl={b.get('sl_atr')} tp={b.get('tp_ratio')} "
              f"(n_w={b.get('n_winners')}) | SELL sl={s.get('sl_atr')} tp={s.get('tp_ratio')} "
              f"(n_w={s.get('n_winners')})")

if __name__ == "__main__":
    main()
