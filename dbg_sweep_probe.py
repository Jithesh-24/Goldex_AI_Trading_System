#!/usr/bin/env python3
"""EXACT replica of engine feature state + sweep. Seeds like main() does
(gold_seed.csv last 90000), appends ALL live bars from xm_live_bars.jsonl,
then the forming bar from tick state — then runs the same best_placement sweep.
Read-only. Mirrors engine lines 713-737, 942-956, 1160-1276."""
import sys, os, json, time
from datetime import datetime, timezone

sys.path.insert(0, "/home/jith/.hermes/profiles/trading/scripts")
os.chdir("/home/jith/.hermes/profiles/trading/scripts")

from ai_signal_engine import FeatureComputer, load_specialists, best_placement, direction_prior, read_tick_state

BASE = "/home/jith/.hermes/profiles/trading/scripts"
OUT = "/home/jith/.hermes/profiles/trading/cron/output"
MODEL = f"{BASE}/models"

# ── 1. replicate model loading (engine main() lines 619-629, 678-693) ──
import lightgbm as lgb
ENGINE_TF = "m5"
def _base_tf_ok(path):
    try:
        with open(path) as f:
            return json.load(f).get("base_tf", "m1") == ENGINE_TF
    except Exception:
        return False
def _load_ensemble(cfg_path):
    with open(cfg_path) as f:
        cfg = json.load(f)
    return [lgb.Booster(model_file=f"{MODEL}/{m}") for m in cfg["models"]], cfg.get("seeds")
models, ens_seeds = _load_ensemble(f"{MODEL}/ensemble.json")
dir_models, dir_seeds = [], None
if _base_tf_ok(f"{MODEL}/direction_ensemble.json"):
    try:
        dir_models, dir_seeds = _load_ensemble(f"{MODEL}/direction_ensemble.json")
    except Exception as e:
        print(f"direction load skipped: {e}")
else:
    print("direction ensemble: TF-guard refused (legacy/absent) → empirical regime prior")
with open(f"{MODEL}/features.json") as f:
    feats = json.load(f)
dir_feats = []
try:
    with open(f"{MODEL}/direction_features.json") as f:
        dir_feats = json.load(f)
except Exception:
    dir_feats = feats
spec_models, spec_cal = load_specialists(f"{MODEL}/regime_specialists.json", MODEL)
from calibrate import load_calibration
cal_knots = load_calibration()
print(f"placement: {len(models)} | direction: {len(dir_models)} | feats: {len(feats)}")

# ── 2. replicate feature seeding EXACTLY (engine lines 721-737) ──
fc = FeatureComputer(maxlen=90000)
hist = []
with open(f"{BASE}/gold_seed.csv") as f:
    next(f)
    for line in f:
        p = line.strip().split(",")
        try:
            t = datetime.strptime(p[0][:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        except Exception:
            continue
        hist.append((t.timestamp(), float(p[1]), float(p[2]), float(p[3]), float(p[4]), int(p[5]), int(p[6])))
for row in hist[-90000:]:
    fc.add(*row)
fc._xm_last_bar_ts = hist[-1][0]
print(f"seeded {len(hist[-90000:])} bars, last={datetime.utcfromtimestamp(hist[-1][0]):%Y-%m-%d %H:%M}")

# ── 3. append ALL live bars (engine lines 942-944) ──
n = 0
with open(f"{OUT}/xm_live_bars.jsonl") as f:
    for line in f:
        try:
            b = json.loads(line)
        except Exception:
            continue
        if b["t"] > fc._xm_last_bar_ts:
            fc.add(b["t"], b["o"], b["h"], b["l"], b["c"], b.get("v", 0), b.get("spread"))
            fc._xm_last_bar_ts = b["t"]
            n += 1
print(f"appended {n} live bars → total {len(fc.times)}")

# ── 4. forming bar from tick state (engine lines 946-956) ──
st = read_tick_state()
if st and st.get("cur_bar") and st["cur_bar"].get("t"):
    cb = st["cur_bar"]
    fc.add(float(cb["t"]), cb["o"], cb["h"], cb["l"], cb["c"], cb.get("v", 0), cb.get("spread", 25))

# ── 5. compute features + sweep (engine lines 1148-1276) ──
fx = fc.features()
if fx is None:
    print("❌ features() returned None"); sys.exit(1)
atr = fx.get("atr_14", 3.0)
xm_bid, xm_ask = st["bid"], st["ask"]
xm_spread = max(xm_ask - xm_bid, 0.05)
px = (xm_bid + xm_ask) / 2.0
print(f"fx: close={fx.get('close',0):.2f} atr={atr:.2f} regime={fx.get('regime')}")

from features import regime_bin
route_regime = regime_bin(fx)
route_models = spec_models.get(route_regime) if spec_models else None
route_cal = (spec_cal or {}).get(route_regime) if route_regime else None
if route_models:
    print(f"routing: SPECIALIST {route_regime}")
else:
    route_models = models
    print(f"routing: GLOBAL (no spec for {route_regime})")
_cal_used = route_cal if route_cal is not None else json.load(open(f"{MODEL}/calibration_by_drr.json"))

buy = best_placement(route_models, feats, fx, atr, xm_spread, "BUY", cal_knots, _cal_used, route_regime)
sell = best_placement(route_models, feats, fx, atr, xm_spread, "SELL", cal_knots, _cal_used, route_regime)

fx["spread"] = round(xm_spread * 100.0, 2)
p_up = direction_prior(dir_models, dir_feats, fx)
buy_w = buy[4] * p_up if buy else -1e9
sell_w = sell[4] * (1 - p_up) if sell else -1e9

print(f"\n=== LIVE SWEEP ({time.strftime('%H:%M:%S')} UTC) ===")
if buy: print(f"BUY:  P(win)={buy[2]:.3f} exp={buy[4]:+.3f} → ×P(up){p_up:.3f} = {buy_w:+.3f} (sl={buy[0]:.2f} tp={buy[1]:.2f} rr={buy[1]/max(buy[0],1e-9):.1f})")
if sell: print(f"SELL: P(win)={sell[2]:.3f} exp={sell[4]:+.3f} → ×(1-P){1-p_up:.3f} = {sell_w:+.3f} (sl={sell[0]:.2f} tp={sell[1]:.2f} rr={sell[1]/max(sell[0],1e-9):.1f})")
best = max(buy_w, sell_w)
print(f"best weighted exp = {best:+.3f} → {'🔥 FIRE' if best > 0 else '❄️ NO FIRE (conservative — no positive-EV setup)'}")
