"""
Phase 3A step 7 -- MAE/MFE research dataset. For every historical CUSUM
event resolved under the v2 (26-feature) meta-labeling pipeline, persist
the full excursion path summary (MAE, MFE, time-to-MAE, time-to-MFE,
time-to-TP, time-to-SL/timeout) alongside model probability, calibrated
probability, and market state -- the data layer a future empirical SL/TP
system needs for P(MAE<=x|state), P(MFE>=x|state), P(TP before SL|state).

Does NOT build the SL/TP optimizer or change live SL/TP -- data only.

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.build_mae_mfe_dataset
"""
import json
import os
import time

import numba
import numpy as np
import pandas as pd

from learning.data import load_raw_m1
from features.features import build_features
from features.labeling import cusum_filter, triple_barrier_labels
from learning.train import (TB_CFG_DIR, TB_CFG_TRADE, HORIZON_VOL_SCALE, CUSUM_K, assemble_dataset,
                             label_events)
from decision.calibration import PlattCalibrator
from research.audit_edge import oof_run, build_meta

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "research", "output")
EXCLUDE = frozenset({"spread", "tick_volume"})


@numba.njit(cache=True)
def _mae_mfe_timed_core(close, high, low, t0_idx, t1_idx, side, vol_at_t0):
    """Same excursion definition as audit_edge._mae_mfe_core, plus WHEN the
    running worst/best is reached (bars from t0) -- audit_edge only needed
    the magnitude, this dataset also needs timing for P(MAE<=x by time y)."""
    n = len(t0_idx)
    mae = np.empty(n, dtype=np.float64)
    mfe = np.empty(n, dtype=np.float64)
    t_mae = np.empty(n, dtype=np.int64)
    t_mfe = np.empty(n, dtype=np.int64)
    for e in range(n):
        t0 = t0_idx[e]
        t1 = t1_idx[e]
        s = side[e]
        p0 = close[t0]
        worst, best = 0.0, 0.0
        worst_t, best_t = 0, 0
        for j in range(t0 + 1, t1 + 1):
            if s >= 0:
                fav = (high[j] - p0) / p0
                adv = (low[j] - p0) / p0
            else:
                fav = (p0 - low[j]) / p0
                adv = (p0 - high[j]) / p0
            if fav > best:
                best = fav
                best_t = j - t0
            if adv < worst:
                worst = adv
                worst_t = j - t0
        v = vol_at_t0[e] if vol_at_t0[e] > 1e-9 else 1e-9
        mae[e] = -worst / v
        mfe[e] = best / v
        t_mae[e] = worst_t
        t_mfe[e] = best_t
    return mae, mfe, t_mae, t_mfe


def vol_terciles(feat, close):
    daily = pd.DataFrame({"close": close}, index=pd.to_datetime(feat["time"])).resample("1D").last()
    ret = np.log(daily["close"]).diff()
    ewma = ret.ewm(span=20).std()
    trailing = ewma.rolling(252, min_periods=60).median().shift(1)
    return trailing.reindex(pd.to_datetime(feat["time"]), method="ffill")


def main():
    t_start = time.time()
    print("== rebuilding v2 (26-feature) dataset for MAE/MFE research layer ==")
    feat, close, high, low, vol, t0_idx, feature_cols = assemble_dataset(exclude=EXCLUDE)
    times = pd.to_datetime(feat["time"].to_numpy())
    X, y_bin, t0, t1, t0_nz = label_events(close, high, low, vol, t0_idx, feature_cols, feat)

    print("== v2 primary + meta OOF (for model proba / calibrated proba columns) ==")
    prim = oof_run(X, y_bin, t0, t1, tag="mae-mfe-primary", want_importance=False)
    side_in = np.where(prim["oof_pred"] == 1, 1.0, -1.0)
    side, meta_labels = build_meta(close, high, low, vol, t0_nz, side_in, prim["has_oof"])
    has_oof = prim["has_oof"]
    X_meta = X.loc[has_oof].reset_index(drop=True)
    X_meta["assumed_side"] = side
    y_meta = pd.Series(meta_labels["label"].to_numpy())
    t0_meta = pd.Series(meta_labels.index.to_numpy())
    t1_meta = pd.Series(meta_labels["t1"].to_numpy())
    meta = oof_run(X_meta, y_meta, t0_meta, t1_meta, tag="mae-mfe-meta", want_importance=False)
    oof_proba = meta["oof_proba"]
    valid_meta = meta["has_oof"]

    t0_meta_np = t0_meta.to_numpy()[valid_meta]
    t1_meta_np = t1_meta.to_numpy()[valid_meta]
    side_np = side[valid_meta]
    y_np = y_meta.to_numpy()[valid_meta]
    raw_p = oof_proba[valid_meta]
    vol_at_meta = vol[t0_nz][has_oof][valid_meta]
    touch = meta_labels["touch"].to_numpy()[valid_meta]
    holding = meta_labels["holding_bars"].to_numpy()[valid_meta]

    calib_path = os.path.join(BASE, "models", "v2", "calibration_global_fallback.json")
    calibrator = PlattCalibrator.load(calib_path) if os.path.exists(calib_path) else PlattCalibrator.identity()
    cal_p = calibrator.apply(raw_p)

    vol_regime_series = vol_terciles(feat, close)
    lo_thr, hi_thr = np.nanpercentile(vol_regime_series.dropna(), [33.3, 66.7])
    vr_at_events = vol_regime_series.to_numpy()[t0_meta_np]
    vol_state = np.where(np.isnan(vr_at_events), "unknown",
                          np.where(vr_at_events <= lo_thr, "low",
                                   np.where(vr_at_events >= hi_thr, "high", "medium")))

    print(f"== MAE/MFE excursion scan ({len(t0_meta_np):,} resolved events) ==")
    mae, mfe, t_mae, t_mfe = _mae_mfe_timed_core(close, high, low, t0_meta_np, t1_meta_np,
                                                  side_np, vol_at_meta)

    tp_first = touch == np.where(side_np >= 0, 1, -1)
    sl_first = touch == np.where(side_np >= 0, -1, 1)
    is_timeout = ~tp_first & ~sl_first
    time_to_tp = np.where(tp_first, holding, -1)
    time_to_sl = np.where(sl_first, holding, -1)

    df = pd.DataFrame({
        "event_time": times[t0_meta_np], "resolution_time": times[t1_meta_np],
        "direction": np.where(side_np >= 0, "BUY", "SELL"),
        "raw_meta_proba": raw_p, "calibrated_proba": cal_p,
        "vol_state": vol_state, "vol_at_event": vol_at_meta,
        "mae_R": mae, "mfe_R": mfe, "time_to_mae_bars": t_mae, "time_to_mfe_bars": t_mfe,
        "touch": np.where(tp_first, "TP", np.where(sl_first, "SL", "TIMEOUT")),
        "time_to_tp_bars": time_to_tp, "time_to_sl_bars": time_to_sl,
        "holding_bars": holding, "label_win": y_np,
        "tp_R": TB_CFG_TRADE.pt_mult, "sl_R": TB_CFG_TRADE.sl_mult,
    })
    out_csv = os.path.join(OUT, "mae_mfe_dataset.csv")
    df.to_csv(out_csv, index=False)

    schema = {
        "n_events": int(len(df)), "columns": {c: str(df[c].dtype) for c in df.columns},
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_model": "v2 (26-feature, spread+tick_volume excluded)",
        "calibrator": os.path.basename(calib_path) if os.path.exists(calib_path) else "identity (no artifact found)",
        "tb_cfg_trade": {"pt_mult": TB_CFG_TRADE.pt_mult, "sl_mult": TB_CFG_TRADE.sl_mult,
                          "max_holding": TB_CFG_TRADE.max_holding},
        "notes": "M1 close-resolution only, no intrabar tick path -- same limitation as "
                 "Phase 1A's realized-vs-nominal R gap. mae_R/mfe_R are vol-normalized R units "
                 "(R = horizon-scaled vol at event time = the SL distance, sl_mult=1.0).",
    }
    with open(os.path.join(OUT, "mae_mfe_dataset_schema.json"), "w") as f:
        json.dump(schema, f, indent=2, default=str)

    print(f"n_events={len(df):,}")
    print(f"MAE: mean={df.mae_R.mean():.3f} p50={df.mae_R.median():.3f} p90={df.mae_R.quantile(0.9):.3f}")
    print(f"MFE: mean={df.mfe_R.mean():.3f} p50={df.mfe_R.median():.3f} p90={df.mfe_R.quantile(0.9):.3f}")
    print(f"touch: TP={float((df.touch=='TP').mean()):.3f} SL={float((df.touch=='SL').mean()):.3f} "
          f"TIMEOUT={float((df.touch=='TIMEOUT').mean()):.3f}")
    print(f"saved -> {out_csv} ({time.time()-t_start:.1f}s)")


if __name__ == "__main__":
    main()
