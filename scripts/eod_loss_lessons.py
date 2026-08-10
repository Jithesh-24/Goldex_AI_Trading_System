#!/usr/bin/env python3
"""EOD LOSS-LESSON REPLAY (v8.7, 2026-08-10) — adaptive loss learning.

User mandate: "the losses it took today should be learned like why it was a
loss and if the exact same setup comes in future what decision should i make
not completely avoid the trade… this should be a adaptive ai engine not
something hardcoded."

This runs at EOD AFTER the retrain chain (train_continue → direction →
calibration → regime specialists → OOF → rating → dir prior). For every
trade that closed today, it reconstructs the EXACT feature vector the engine
saw, re-runs the FULL live decision path with the FRESHLY retrained models,
and reports:

  OLD decision  (what the engine actually did — stored in live_outcomes)
  NEW decision  (what the retrained engine would do NOW for the same setup)

If the retrain learned the lesson, the same input produces a different
output (lower P / different side / held by the re-learned rating gate / or
still fires — which is also correct: the data said it was fine). Nothing
here is hardcoded: both decisions come from the SAME code path the live
engine uses (best_placement + direction_prior + rate_signal), just with the
new weights.

Output: cron/output/loss_lessons.jsonl (per-trade) + console summary.
"""
import json, os, sys, time
import numpy as np

BASE = "/home/jith/.hermes/profiles/trading/scripts"
OUTDIR = "/home/jith/.hermes/profiles/trading/cron/output"
MODEL = f"{BASE}/models"
sys.path.insert(0, BASE)

import lightgbm as lgb
import ai_signal_engine as E          # reuse the LIVE decision functions
from features import regime_bin
from calibrate import apply_calibration

OUTCOMES = f"{OUTDIR}/live_outcomes.jsonl"
LESSONS = f"{OUTDIR}/loss_lessons.jsonl"

# ── model loaders (mirror engine's nested load_ensemble) ──
def load_ensemble(cfg_path):
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
        ms = []
        for m in cfg["models"]:
            try:
                ms.append(lgb.Booster(model_file=f"{MODEL}/{m}"))
            except Exception:
                pass  # v8.7: skip missing model files (e.g. direction_s42 not
                # deployed) — a partial ensemble still beats a crash
        return ms
    except Exception:
        return []

def load_specialists():
    try:
        with open(f"{MODEL}/regime_specialists.json") as f:
            cfg = json.load(f)
        out, calib = {}, {}
        for regime, meta in (cfg.get("bins") or {}).items():
            try:
                out[regime] = [lgb.Booster(model_file=f"{MODEL}/{m}")
                               for m in meta["models"]]
            except Exception:
                out[regime] = None
            _cp = f"{MODEL}/calibration_by_drr_spec_{regime.lower()}.json"
            try:
                if os.path.exists(_cp):
                    with open(_cp) as f:
                        calib[regime] = json.load(f)
            except Exception:
                calib[regime] = None
        return out, calib
    except Exception:
        return {}, None

def main():
    # ── load FRESH (retrained) artifacts ──
    with open(f"{MODEL}/features.json") as f:
        feats = json.load(f)
    models = load_ensemble(f"{MODEL}/ensemble.json")
    dir_models = load_ensemble(f"{MODEL}/direction_ensemble.json")
    spec_models, spec_cal = load_specialists()
    try:
        with open(f"{MODEL}/calibration.json") as f:
            cal_knots = json.load(f)
    except Exception:
        cal_knots = None
    try:
        with open(f"{MODEL}/calibration_by_drr.json") as f:
            cal_by_rr = json.load(f)
    except Exception:
        cal_by_rr = None
    try:
        with open(f"{MODEL}/direction_features.json") as f:
            dir_feats = json.load(f)
    except Exception:
        dir_feats = []
    try:
        from signal_rating import rate_signal, rating_threshold
        rating_gate = float(rating_threshold())
    except Exception:
        rating_gate = 0.0

    print(f"═══ EOD LOSS-LESSON REPLAY ═══")
    print(f"feats: {len(feats)} (M5-only) | placement {len(models)} | dir {len(dir_models)} | specialists {len(spec_models)} | rating gate {rating_gate:.1f}")

    # ── today's closed trades ──
    rows = []
    with open(OUTCOMES) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("result") in ("TP", "SL"):
                rows.append(r)
    if not rows:
        print("no closed trades today — nothing to learn (clean slate)")
        return 0

    n_learned = 0
    # v8.7: replay each trade ONCE. EOD runs daily — without a dedupe the
    # audit trail re-appends every historical trade every night. Keyed by
    # (t, dir) — the same key merge_live_outcomes_appended uses.
    try:
        with open(LESSONS) as f:
            replayed = {(json.loads(l).get("t"), json.loads(l).get("old_dir"))
                        for l in f if l.strip()}
    except Exception:
        replayed = set()
    with open(LESSONS, "a") as out:
        for r in rows:
            if (r.get("t"), r.get("dir")) in replayed:
                continue  # already learned in a prior EOD run
            fx_raw = r.get("feats") or {}
            if not fx_raw:
                continue
            d_old = r.get("dir"); result = r.get("result")
            entry, sl, tp = r.get("entry"), r.get("sl"), r.get("tp")
            if not all(x is not None for x in (entry, sl, tp)):
                continue
            # ── reconstruct live inputs from the STORED snapshot ──
            # atr from stored geometry: sl_dist = |entry - sl| = atr * sl_mult;
            # sl_atr_buy = sl_dist / atr  →  atr = sl_dist / sl_atr
            sl_dist = abs(entry - sl)
            sl_atr = fx_raw.get("sl_atr_buy") or fx_raw.get("sl_atr_sell") or 1.0
            atr = sl_dist / max(float(sl_atr), 1e-9)
            spread = float(fx_raw.get("spread", 24.0)) / 100.0  # points → $
            # fx = the stored MARKET features (all of them — engine's X build
            # only reads feats[] entries, so HTF cols present-but-unlisted are
            # ignored exactly as in live).
            fx = {k: v for k, v in fx_raw.items() if k != "time"}

            # ── replay LIVE decision path (same functions the engine calls) ──
            regime = regime_bin(fx)
            route_models = (spec_models or {}).get(regime) or models
            route_cal = (spec_cal or {}).get(regime)
            _cal = route_cal if route_cal is not None else cal_by_rr
            buy = E.best_placement(route_models, feats, fx, atr, spread, "BUY",
                                   cal_knots, _cal, regime)
            sell = E.best_placement(route_models, feats, fx, atr, spread, "SELL",
                                    cal_knots, _cal, regime)
            fx["spread"] = round(spread * 100.0, 2)
            p_up = E.direction_prior(dir_models, dir_feats, fx)
            buy_w = buy[4] * p_up if buy else -1e9
            sell_w = sell[4] * (1 - p_up) if sell else -1e9
            if buy_w >= sell_w and buy:
                new_dir, new_conf, new_exp, new_sl, new_tp = "BUY", buy[2], buy[4], buy[0], buy[1]
            elif sell:
                new_dir, new_conf, new_exp, new_sl, new_tp = "SELL", sell[2], sell[4], sell[0], sell[1]
            else:
                new_dir = new_conf = new_exp = new_sl = new_tp = None
            # rating gate on the new candidate
            new_rating = None
            fires = False
            if new_dir is not None:
                try:
                    new_rating, _ = rate_signal(fx, new_dir, new_conf, new_exp,
                                                float(new_sl / max(atr, 1e-9)), regime)
                    fires = new_rating >= rating_gate
                except Exception:
                    fires = True
            # ── decision change? ──
            changed = (new_dir != d_old)
            verdict = "LEARNED-DIFFERENT" if changed else "SAME-DECISION"
            lesson = {
                "t": r.get("t"), "old_dir": d_old, "result": result,
                "pnl": round(r.get("pnl", 0.0), 2), "regime": regime,
                "old_conf": round(r.get("conf", 0.0), 3),
                "old_sl": sl, "old_tp": tp,
                "new_dir": new_dir,
                "new_conf": round(new_conf, 3) if new_conf is not None else None,
                "new_exp": round(new_exp, 4) if new_exp is not None else None,
                "new_sl": round(new_sl, 2) if new_sl is not None else None,
                "new_tp": round(new_tp, 2) if new_tp is not None else None,
                "new_rating": round(new_rating, 1) if new_rating is not None else None,
                "rating_gate": rating_gate,
                "fires_now": fires,
                "p_up": round(p_up, 3),
                "verdict": verdict,
            }
            out.write(json.dumps(lesson) + "\n")
            n_learned += 1
            replayed.add((r.get("t"), r.get("dir")))
            print(f"  {r.get('t')} {d_old} {result} ${r.get('pnl',0):+.2f} "
                  f"({regime}, conf {r.get('conf',0):.2f})")
            print(f"      → retrained: {new_dir} conf {lesson['new_conf']} exp {lesson['new_exp']} "
                  f"rating {lesson['new_rating']}/{rating_gate:.0f} fires={fires} :: {verdict}")

    print(f"═══ done: {n_learned} trades replayed → {LESSONS} ═══")
    return 0

if __name__ == "__main__":
    sys.exit(main())
