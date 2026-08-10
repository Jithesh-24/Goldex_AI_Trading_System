"""PHASE 5 — AI Signal Engine (v4 2026-08-01: XM-native + learned placement).
TRUE AI: LightGBM model trained on real XAUUSD spot bars decides direction
AND SL/TP placement from its learned probability surface.

  - Live bars: REAL XM M1 bars (xm_ticker.py owns MT5, writes tick-built bars)
  - Model: LightGBM P(win | market + placement + direction) over grid sweep
  - SL/TP: LEARNED — evaluate every (SL×TP×dir) candidate, fire max-EV > 0
  - NO gates, NO hardcoded floors/multipliers (harness, not harden)
  - Journal: every outcome appended (retrain loop → closed loop)
"""
import json, os, sys, time, math, subprocess
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime, timezone, timedelta

BASE = "/home/jith/.hermes/profiles/trading/scripts"
OUTDIR = "/home/jith/.hermes/profiles/trading/cron/output"
MODEL = f"{BASE}/models/gold_lgb_model.txt"          # v6 fallback (compat)
ENSEMBLE_CFG = f"{BASE}/models/ensemble.json"        # v7 placement ensemble
SPEC_CFG = f"{BASE}/models/regime_specialists.json"   # v7.7 regime placement specialists
DIR_ENSEMBLE_CFG = f"{BASE}/models/direction_ensemble.json"  # v7 direction model
FEATURES = f"{BASE}/models/features.json"
DIR_FEATURES = f"{BASE}/models/direction_features.json"

# ── Telegram (same as before, correct token in signals .env) ──
def _tg_once(text):
    """Single Telegram send attempt via signals/.env token. Returns (ok, info)."""
    try:
        env = {}
        with open("/home/jith/.hermes/profiles/signals/.env") as f:
            for line in f:
                line = line.strip()
                if line and "=" in line:
                    k, v = line.split("=", 1)
                    env[k] = v
        token = env.get("TELEGRAM_BOT_TOKEN", "").strip()
        chat = (env.get("TELEGRAM_CHAT_ID") or "5376343193").strip()
        if not token:
            return False, "NO TOKEN"
        r = subprocess.run(
            ["curl", "-s", "-X", "POST",
             f"https://api.telegram.org/bot{token}/sendMessage",
             "--data-urlencode", f"chat_id={chat}",
             "--data-urlencode", f"text={text}",
             "-d", "parse_mode=HTML"],
            capture_output=True, timeout=15)
        out = r.stdout.decode("utf-8", "replace")
        if '"ok":true' in out:
            return True, "ok"
        desc = "?"
        try:
            desc = out.split('"description":"', 1)[1].split('"', 1)[0]
        except Exception:
            pass
        return False, desc
    except Exception as e:
        return False, str(e)

TG_FAIL_LOG = f"{OUTDIR}/.tg_delivery_failures.jsonl"

def tg(text):
    """Send a Telegram message. VERIFIES the API response — never prints
    success unless the message actually delivered (v7.3e 2026-08-03).
    v7.3f (2026-08-03): one retry on transient failure, then falls back to
    the platform-level `hermes send` path (independent token/config from
    signals/.env) so a signal is never silently dropped on a single bad
    path. Every failure is breadcrumbed to .tg_delivery_failures.jsonl.
    Returns True on confirmed delivery via EITHER path."""
    ok, info = _tg_once(text)
    if not ok:
        time.sleep(2)
        ok, info = _tg_once(text)  # one retry — covers transient blips
    if ok:
        print(f"[{ts()}] 📨 TG ok: {text[:100]}")
        return True
    print(f"[{ts()}] ❌ TG SEND FAILED (after retry): {info}")
    # v7.3f DELIVERY-DISCIPLINE (2026-08-03): previously this fell back to
    # `hermes send -t telegram`, which routes to the Hermes HOME CHANNEL — i.e.
    # THIS assistant chat. That's how signals were leaking 'here' when they
    # must go ONLY to @Goldrigging_bot. Signals now FAIL CLOSED: we never
    # redirect a signal into this chat. The message is breadcrumbed to
    # .tg_delivery_failures.jsonl (with full text) so nothing is silently lost
    # and it can be re-delivered manually if desired.
    fb_ok = False
    try:
        with open(TG_FAIL_LOG, "a") as f:
            f.write(json.dumps({"t": time.time(), "reason": info, "fallback_ok": False,
                                 "text": text}) + "\n")
    except Exception:
        pass
    return fb_ok

def ts():
    return datetime.now().strftime("%H:%M:%S")


def weekend_close_minutes():
    """Minutes until XM weekly close (Fri 23:55 server = Fri 20:55 UTC with
    the detected +3 summer offset), or 0 if already closed for the weekend.
    Server offset is read from the ticker's persisted measurement so DST is
    handled by what the broker actually reports. Returns None if offset
    unknown (treat as no weekend info)."""
    try:
        import json as _j
        off_p = os.path.join(BASE, "cron/output/xm_server_offset.json")
        off = 3.0
        try:
            with open(off_p) as _f:
                off = float(_j.load(_f).get("offset_h", 3.0))
        except Exception:
            pass
        now = datetime.now(timezone.utc)
        # weekly close: Friday 23:55 server time → UTC (server = UTC+off)
        fri_close_utc = datetime(now.year, now.month, now.day, 23, 55,
                                 tzinfo=timezone.utc) - timedelta(hours=off)
        days_ahead = (4 - now.weekday()) % 7   # 0 today..6 if weekend passed
        close_dt = fri_close_utc + timedelta(days=days_ahead)
        # if we've already passed this Friday's close and it's the weekend
        # (Sat/Sun), report 0 — market closed; Mon morning: next Friday
        if now > close_dt and now.weekday() in (5, 6):
            return 0
        if now > close_dt and now.weekday() == 4:
            return 0
        if now > close_dt:
            close_dt = close_dt + timedelta(days=7)
        return int((close_dt - now).total_seconds() // 60)
    except Exception:
        return None

# ── State (1 signal at a time) ──
ACTIVE = f"{OUTDIR}/.active_signal_ai.json"
def load_active():
    try:
        with open(ACTIVE) as f:
            return json.load(f)
    except Exception:
        return None

def save_active(a):
    with open(ACTIVE, "w") as f:
        json.dump(a, f)

JOURNAL = f"{OUTDIR}/trade_journal_ai.jsonl"
OUTCOMES = f"{OUTDIR}/live_outcomes.jsonl"   # CLOSED LOOP: feature vector + result per real trade
def append_journal(entry):
    with open(JOURNAL, "a") as f:
        f.write(json.dumps(entry) + "\n")

def append_outcome(entry):
    """Append a resolved trade's feature vector + outcome.
    retrain_loop merges these rows into the training matrix so the model
    learns from ACTUAL outcomes, not just simulated price paths."""
    try:
        with open(OUTCOMES, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"[{ts()}] outcome log warn: {e}")

# ── Feature computation (mirrors features.py, single-bar update) ──
class FeatureComputer:
    """Keeps rolling windows of OHLC data, computes the v7 features at the
    last bar by delegating to features.py — GUARANTEES the live engine sees
    the exact feature space the model was trained on (63→95 v7 features)."""
    def __init__(self, maxlen=32000):
        self.times = []
        self.opens, self.highs, self.lows, self.closes, self.vols = [], [], [], [], []
        self.spreads = []
        self.maxlen = maxlen

    def add(self, t, o, h, l, c, v, spread=None):
        if self.times and t <= self.times[-1]:
            # update last bar (replace) — keep max high / min low so range never shrinks
            self.times[-1] = t
            self.opens[-1] = o; self.closes[-1] = c; self.vols[-1] = v
            self.highs[-1] = max(self.highs[-1], h)
            self.lows[-1] = min(self.lows[-1], l)
            if spread is not None: self.spreads[-1] = spread
            return
        self.times.append(t); self.opens.append(o); self.highs.append(h)
        self.lows.append(l); self.closes.append(c); self.vols.append(v)
        self.spreads.append(spread if spread is not None else (self.spreads[-1] if self.spreads else 20))
        if len(self.times) > self.maxlen:
            self.times.pop(0); self.opens.pop(0); self.highs.pop(0)
            self.lows.pop(0); self.closes.pop(0); self.vols.pop(0)
            self.spreads.pop(0)

    def _a(self, arr):  # np array alias
        return np.array(arr, dtype=float)

    def features(self, geometry=None):
        """Compute MARKET features (v8: M5 base) by delegating to features.py —
        IDENTICAL pipeline to the M5 training matrix. The M1 bar buffer is
        aggregated into M5 OHLCV first (open=first, high=max, low=min,
        close=last, vol=sum) — same resample the matrix builder uses.

        CACHED by last M5-bar ts: HTF/session/event features only change when
        a bar COMPLETES (once/5min), not on every 3s poll."""
        if len(self.times) < 200:
            return None
        try:
            last_m5 = int(self.times[-1] // 300) * 300
        except Exception:
            last_m5 = None
        if getattr(self, "_fx_cache_ts", None) == last_m5 and getattr(self, "_fx_cache", None) is not None:
            return self._fx_cache
        import sys
        sys.path.insert(0, "/home/jith/.hermes/profiles/trading/scripts")
        from features import _feature_block
        # ── M1 → M5 aggregation (matches build_m5_matrix.to_m5 exactly) ──
        m5_times, m5_o, m5_h, m5_l, m5_c, m5_v = [], [], [], [], [], []
        cur = None
        for t, o, h, l, c, v in zip(self.times, self.opens, self.highs,
                                    self.lows, self.closes, self.vols):
            b = int(t // 300) * 300
            if cur is None or b != cur[0]:
                if cur is not None:
                    m5_times.append(cur[0]); m5_o.append(cur[1]); m5_h.append(cur[2])
                    m5_l.append(cur[3]); m5_c.append(cur[4]); m5_v.append(cur[5])
                cur = [b, o, h, l, c, v]
            else:
                cur[2] = max(cur[2], h); cur[3] = min(cur[3], l)
                cur[4] = c; cur[5] += v
        if cur is not None:
            m5_times.append(cur[0]); m5_o.append(cur[1]); m5_h.append(cur[2])
            m5_l.append(cur[3]); m5_c.append(cur[4]); m5_v.append(cur[5])
        if len(m5_times) < 200:
            return None
        df = pd.DataFrame({
            "time": [datetime.utcfromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S") for t in m5_times],
            "open": m5_o, "high": m5_h, "low": m5_l,
            "close": m5_c, "tick_volume": m5_v,
            "spread": [self.spreads[-1]] * len(m5_times) if self.spreads else [20] * len(m5_times),
            "real_volume": [0] * len(m5_times),
        })
        df["time"] = pd.to_datetime(df["time"])
        df = _feature_block(df)
        fx = {col: float(df.iloc[-1][col]) for col in df.columns if col not in ("time", "spread")}
        self._fx_cache = fx
        self._fx_cache_ts = last_m5
        return fx

    def placement_row(self, fx, atr, spread, direction, sl_mult, tp_ratio):
        """Build the FULL model row for one candidate placement: 42 market
        features + 8 geometry features (this placement) + direction flag.
        Mirrors build_placement_dataset() exactly."""
        sl_dist = max(atr * sl_mult, 0.30)
        tp_dist = (sl_dist + spread) * tp_ratio
        d = 1.0 if direction == "BUY" else 0.0
        row = dict(fx)
        # spread IS a model feature (training: points, e.g. 26 = $0.26).
        # features() excludes it — inject the REAL XM spread here (in points).
        row["spread"] = round(spread * 100.0, 2)
        row["sl_dist_buy"] = sl_dist
        row["tp_dist_buy"] = tp_dist
        row["sl_dist_sell"] = sl_dist
        row["tp_dist_sell"] = tp_dist
        row["sl_atr_buy"] = sl_dist / (atr + 1e-9)
        row["sl_atr_sell"] = sl_dist / (atr + 1e-9)
        row["rr_buy"] = tp_dist / (sl_dist + 1e-9)
        row["rr_sell"] = tp_dist / (sl_dist + 1e-9)
        row["direction"] = d
        return row, sl_dist, tp_dist

# ── XM-NATIVE BARS (v4 2026-08-01) ──
# TradingView is DELETED from the pipeline. xm_ticker.py owns the MT5
# connection (Wine allows one IPC client) and writes:
#   * xm_tick_state.json — bid/ask + cur_bar (forming M1 bar from REAL XM ticks)
#   * xm_live_bars.jsonl — completed M1 bars (real XM, true UTC)
# The engine reads ONLY these. No TV, no freeze, no divergence, exact spread.
BARS_LIVE = f"{OUTDIR}/xm_live_bars.jsonl"

def read_completed_bars(last_ts=0.0):
    """Return list of (t, o, h, l, c, v, spread) completed bars with t > last_ts."""
    bars = []
    try:
        with open(BARS_LIVE) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if float(d["t"]) > last_ts:
                        bars.append((float(d["t"]), float(d["o"]), float(d["h"]),
                                     float(d["l"]), float(d["c"]), int(d.get("v", 0)),
                                     int(d.get("spread", 20))))
                except Exception:
                    continue
    except Exception:
        pass
    return bars

# ── XM broker quote via MT5 (ground truth for SL/TP — the user's fills are on XM) ──
# FILL-SIDE semantics (08-04 FIX — TP sides were swapped before, causing false
# "TP HIT" acks one spread early on SELL and one spread late on BUY):
#   BUY  → close = SELL at BID: SL on bid<=sl, TP on bid>=tp
#   SELL → close = BUY  at ASK: SL on ask>=sl, TP on ask<=tp
# v2: xm_ticker.py daemon (25ms polling, first-touch path tracking) owns the MT5
# connection. The engine reads its state file — ZERO subprocess cost. The old
# spawn-based xm_tick survives only as a rare fallback if the ticker is stale.
_MT5_QUOTE = {"bid": None, "ask": None, "ts": 0}
TICK_STATE = f"{OUTDIR}/xm_tick_state.json"

def read_tick_state():
    """Read the ticker's 25ms state file. Returns dict or None (stale/missing)."""
    try:
        with open(TICK_STATE) as f:
            st = json.load(f)
        if st.get("bid") is None or time.time() - st.get("ts", 0) > 30:
            return None  # stale — ticker down or market closed
        return st
    except Exception:
        return None

def xm_tick(max_age=25):
    """FALLBACK ONLY: spawn Wine Python for a one-shot quote when ticker stale."""
    now = time.time()
    if _MT5_QUOTE["bid"] is not None and now - _MT5_QUOTE["ts"] < max_age:
        return _MT5_QUOTE
    try:
        r = subprocess.run(
            ["wine", "/home/jith/.wine/drive_c/users/jith/AppData/Local/Programs/Python/Python311/python.exe",
             "-c", "import MetaTrader5 as m;m.initialize();t=m.symbol_info_tick('GOLD.i#');"
                   "print(f'{t.bid:.3f},{t.ask:.3f}' if t else 'ERR');m.shutdown()"],
            capture_output=True, timeout=20, env={"WINEPREFIX": "/home/jith/.wine",
            "DISPLAY": ":99", "WINEDEBUG": "-all", "PATH": "/usr/bin:/bin"})
        out = (r.stdout or b"").decode().strip().splitlines()
        bid, ask = [float(x) for x in out[-1].split(",")]
        _MT5_QUOTE.update(bid=bid, ask=ask, ts=now)
        return _MT5_QUOTE
    except Exception:
        return None

def resolve_pnl(d, r_kind, active):
    """PnL for a resolved trade (first-touch semantics)."""
    if d == "BUY":
        return active["sl"] - active["entry"] if r_kind == "SL" else active["tp"] - active["entry"]
    return active["entry"] - active["sl"] if r_kind == "SL" else active["entry"] - active["tp"]

def load_specialists(cfg_path, fallback_single):
    """v7.7 REGIME ROUTER: load {regime: [3 Booster models]} from
    regime_specialists.json. Returns a dict regime->list-of-Boosters, or None
    if no specialist config exists (engine falls back to the global ensemble).
    Lightweight — one Booster per regime file, loaded once and hot-reloaded
    whenever the config mtime changes (mirrors load_ensemble).

    v7.7b (2026-08-06): also loads per-regime calibration curves
    (calibration_by_drr_spec_<regime>.json, emitted by train_regime_spec).
    Returns (models, calib): calib[regime] = {dir_RR: knots} so the router
    calibrates specialist probabilities with the specialist's OWN OOF curves
    instead of the base model's."""
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict) or "bins" not in cfg:
            return None, None
        out = {}
        calib = {}
        for regime, meta in (cfg.get("bins") or {}).items():
            try:
                out[regime] = [lgb.Booster(model_file=f"{BASE}/models/{m}")
                               for m in meta["models"]]
            except Exception:
                out[regime] = None
            # v7.7b: per-regime calibration (optional — fallback to base)
            _cp = f"{BASE}/models/calibration_by_drr_spec_{regime.lower()}.json"
            try:
                if os.path.exists(_cp):
                    with open(_cp) as f:
                        calib[regime] = json.load(f)
            except Exception:
                calib[regime] = None
        return (out if out else None), (calib if calib else None)
    except Exception:
        return None, None

# ── LEARNED SL/TP PLACEMENT (v4 2026-08-01 → v8.2 2026-08-07) ──
# NO hardcoded geometry. The model was trained over a grid of placements
# (SL_MULTS × TP_RATIOS × direction) and learned P(win | market + placement).
# v8.2 (2026-08-07): state-dependent placement — the engine ANALYSES the
# current market and picks the placement with the highest calibrated
# expectancy across the FULL candidate set. No fixed ratio: in range states
# the model prefers tight geometry (~1:1.3); in strong-trend states with room
# to run it can pick far geometry (up to the training grid's max ratio).
# The learned placement prior (placement_prior.json, fit from 6yr MFE/MFA
# excursions per regime×direction) is IN the candidate set as the
# institutional anchor and wins any expectancy tie — but it never constrains
# the sweep. The grid is the search space (data-sampled), not a hard rule.
def best_placement(models, feats, fx, atr, spread, direction, cal_knots=None, cal_by_rr=None, regime=None):
    """Evaluate ALL (sl_mult, tp_ratio) candidates for one direction using the
    ENSEMBLE's averaged P(win | market + placement), return the best
    (sl_dist, tp_dist, conf, ev, exp, sl_mult, tp_ratio) or None.

    Selection metric = EXPECTANCY per dollar risked (scale-free, pro standard):
        Exp = P × RR − (1−P)   where RR = TP / (SL+spread)
    conf = CALIBRATED ensemble P (raw model P is overconfident; v5 fixes it).
    v7.3c: PER-RR calibration — each TP ratio has its OWN honest P(win) curve.
    v8: the sweep is ANCHORED by the learned placement prior.
    v8.1: learned placement made authoritative (user mandate) — but that
    locked SL/TP to a fixed regime×direction pair (~1:1.3 always), which can
    never catch a 1:10 runner. User corrected (2026-08-07): "it should
    analyse and give the perfect placement… no constraints".
    v8.2: restored the FULL state-dependent sweep — the learned prior is the
    anchor (in the set, wins ties) but the calibrated expectancy surface
    decides. Geometry varies with the market, never constrained to a ratio.
    regime=None → grid-only (backward compat, never the default)."""
    from calibrate import apply_calibration
    from features import SL_MULTS, TP_RATIOS
    fc = FeatureComputer(maxlen=2)
    best = None
    candidates = [(m, r) for m in SL_MULTS for r in TP_RATIOS]
    learned_pair = None
    # v8: learned anchor — the regime's institutional SL/TP from 6yr MFE/MFA
    # excursion data. IN the set, wins expectancy ties, never constrains.
    try:
        if regime:
            with open(f"{BASE}/models/placement_prior.json") as f:
                pp = json.load(f)
            pd_ = pp.get("regimes", {}).get(regime, {}).get(direction, {})
            learned_sl = pd_.get("sl_atr")
            learned_tp = pd_.get("tp_ratio")
            mfe_p50 = pd_.get("mfe_p50")
            if learned_sl and learned_tp:
                learned_pair = (float(learned_sl), float(learned_tp))
                # v8.6 (2026-08-10): REACHABLE-BAND constraint — the sweep is
                # capped by what 6yr MFE/MFA data says is actually reachable:
                #   SL  ≤ learned p90 loser-adverse band (sl_atr)
                #   TP  ≤ learned median favorable excursion (mfe_p50)
                # A 7R TP at 20+ ATR is a multi-day runner — positive on a
                # time-blind exp surface but NOT the clean, human-speed trade
                # the user wants. The cap is LEARNED per regime×direction
                # (expands automatically when the regime has room to run) —
                # not a hardcoded ratio. The learned pair stays in the set.
                sl_cap = float(learned_sl)
                tp_cap_atr = float(mfe_p50) if mfe_p50 else float(learned_sl) * float(learned_tp)
    except Exception:
        pass
    # learned_pair (regime's institutional SL/TP from 6yr MFE/MFA data) is
    # ALWAYS evaluated — it is the anchor candidate. Evaluated alongside the
    # grid (never instead of it): the model's calibrated expectancy surface
    # decides the winner; the learned pair wins any near-tie.
    if learned_pair is not None:
        candidates.append(learned_pair)
    for m, r in candidates:
        row, sl_dist, tp_dist = fc.placement_row(fx, atr, spread, direction, m, r)
        # v8.6 reachable-band filter: skip geometry the data says is out of
        # reach (multi-day runners). True SL in ATR must be within the
        # learned adverse band; TP distance must be within the learned
        # median favorable excursion. The learned anchor pair DEFINES the
        # band — never filter it out (its TP ≈ mfe_p50 by construction).
        if learned_pair is not None and (m, r) != learned_pair:
            if sl_dist / max(atr, 1e-9) > sl_cap:
                continue
            if tp_dist / max(atr, 1e-9) > tp_cap_atr:
                continue
        X = np.array([[row.get(c, 0.0) for c in feats]], dtype=np.float32)
        # 3-seed ensemble: average raw probs, THEN calibrate once.
        p_raw = float(np.mean([mdl.predict(X)[0] for mdl in models]))
        if cal_by_rr is not None:
            # v7.3e per-dir×RR curve: honest P(win) for THIS geometry+direction.
            # v8.1: learned TP ratios (e.g. 1.29) rarely equal the grid ratios
            # the curves were fit on (1.3/1.8/2.5/3.0) — look up the NEAREST
            # available ratio so the learned geometry is still calibrated
            # (exact-key miss would silently fall back to RAW overconfident P).
            key = f"{direction}_{r}"
            if key not in cal_by_rr and r > 0:
                ratios = sorted(float(k.split('_')[1]) for k in cal_by_rr
                                if k.startswith(direction + "_"))
                if ratios:
                    rn = min(ratios, key=lambda x: abs(x - r))
                    key = f"{direction}_{rn}"
            knots = cal_by_rr.get(key)
            p = apply_calibration(p_raw, knots) if knots else p_raw
        else:
            p = apply_calibration(p_raw, cal_knots) if cal_knots else p_raw
        # Entry AND exit eat spread → true risk = SL + spread.
        true_sl = sl_dist + spread
        rr = tp_dist / (true_sl + 1e-9)
        exp = p * rr - (1 - p)          # expectancy per $ risked
        ev = p * tp_dist - (1 - p) * true_sl  # dollar EV (reporting only)
        if best is None or exp > best[4] + 1e-9:
            best = (sl_dist, tp_dist, p, ev, exp, m, r)
        elif learned_pair is not None and abs(exp - best[4]) <= 1e-9 and (m, r) == learned_pair:
            # v8.2: anchor wins ties — institutional geometry preferred when
            # the model sees no material difference between candidates.
            best = (sl_dist, tp_dist, p, ev, exp, m, r)
    return best


def direction_prior(dir_models, dir_feats, fx):
    """DIRECTION MODEL (v7, T2-1): P(up-move | market state) from the
    3-seed direction ensemble. Pure learned probability — the engine
    multiplies it into expectancy:
        final_exp(BUY)  = Exp(BUY)  × P(up)
        final_exp(SELL) = Exp(SELL) × (1 − P(up))
    In a downtrend P(up) is low → BUY expectancy gets crushed → the model
    itself refuses to catch falling knives. No gates, no thresholds.

    NEUTRALIZATION (v7.4, 2026-08-03): the direction model must DEMONSTRATE
    a real directional edge before it may tilt the placement decision. An
    essentially-random direction model (accuracy ≈ 0.50 within sampling
    noise) is pure noise — multiplying it in would crush one side for no
    real reason (observed: acc 0.4998 → 15 SELL / 0 BUY on a range day while
    the placement calibration actually favors BUY). So if the held-out
    metrics show no edge, we return a NEUTRAL prior (0.5) and let the
    placement model's own BUY/SELL expectancy decide. This is not a gate on
    entries — it only stops a coin-flip from dictating the side.
    """
    try:
        acc = None
        acc_price = None
        base_maj = None
        try:
            import json as _json
            with open(f"{BASE}/models/direction_metrics.json") as _f:
                _m = _json.load(_f)
                acc = _m.get("acc")
                acc_price = _m.get("acc_price")
                _bu = _m.get("base_up")
                base_maj = max(_bu, 1 - _bu) if _bu is not None else None
        except Exception:
            acc = None
        # v7.11 (2026-08-05): EMPIRICAL REGIME PRIOR. The flat 0.5 was itself
        # information-free — it let the placement model's geometry statistics
        # decide the side, and in a HIGH_VOL-mislabeled rally the fade
        # specialist fired SELL at P=92% into a +$47 trend (the 'trading
        # inversely' reported). Measured over the full 6.4yr matrix (133,136
        # 3-min bars), gold MEAN-REVERTS at this scale: P(up) after STRONG_UP
        # = 0.476, after STRONG_DOWN = 0.525, RANGE_TIGHT 0.516, HIGH_VOL
        # 0.504. So the regime-conditional empirical P(up) from the training
        # data becomes the prior when the ML direction model has no edge —
        # the market condition the model was trained on steers the side.
        # Data-driven (measured), not a gate, not hardcoded.
        def _empirical_prior():
            try:
                import json as _j
                with open(f"{BASE}/models/regime_dir_prior.json") as _f2:
                    _p = _j.load(_f2)
                reg = None
                try:
                    from features import regime_bin
                    reg = regime_bin(fx)
                except Exception:
                    reg = None
                if reg and reg in _p.get("P_up_by_regime", {}):
                    return float(_p["P_up_by_regime"][reg])
            except Exception:
                pass
            return 0.5
        # v7.4 gate: persistence acc <= 0.53 → coin flip → empirical prior.
        if acc is not None and acc <= 0.53:
            return _empirical_prior()
        # v7.6b gate (2026-08-04): the engine consumes p_up as P(PRICE up),
        # but the label measures H1-trend PERSISTENCE — trivially learnable
        # and saturating (p_up pinned ~0.00 in any falling H1 regime). A
        # persistence-parrot with acc 0.905 but acc_price 0.41 (vs 0.81
        # majority) forced SELLs into the 08-04 rally — the exact "selling
        # in a buy market" the user reported. If the model cannot beat
        # 'always bet the majority' on ACTUAL price direction, it has no
        # real directional edge → NEUTRAL prior; the placement model's own
        # learned BUY/SELL expectancy decides the side (pure teaching, not
        # a gate on entries — it only stops a saturating label from
        # overriding placement's real call).
        if acc_price is not None and base_maj is not None and acc_price <= base_maj + 0.02:
            return _empirical_prior()
        if acc_price is not None and acc_price <= 0.53:
            return _empirical_prior()
        # model has an edge → use its P(up), softly clipped so neither side mutes
        X = np.array([[fx.get(c, 0.0) for c in dir_feats]], dtype=np.float32)
        p_up = float(np.mean([mdl.predict(X)[0] for mdl in dir_models]))
        return max(min(p_up, 0.95), 0.05)
    except Exception:
        return 0.5  # no direction model → neutral prior

# ── Market regime label for signals (v5) ──
def regime_label(fx):
    """Human-readable market state from the regime features — pure description
    of what the model sees, NOT a gate (the model decides, this just labels)."""
    try:
        te = fx.get("trend_ema", 0.0); ts = fx.get("trend_slope", 0.0)
        bb = fx.get("bb_pctile", 0.5); ap = fx.get("atr_pctile", 0.5)
        vs = fx.get("vol_spike", 0.0); ns = fx.get("news_candle", 0.0)
        parts = []
        if abs(te) > 1.2 and ts * te > 0:
            parts.append("TRENDING " + ("UP 📈" if te > 0 else "DOWN 📉"))
        elif bb < 0.3:
            parts.append("RANGING ⟷")
        else:
            parts.append("MIXED ➰")
        if ns > 0.4 or vs > 2.0:
            parts.append("NEWS-SPIKE ⚡")
        elif ap < 0.3:
            parts.append("QUIET 😴")
        elif ap > 0.7:
            parts.append("VOLATILE 🌊")
        return " | ".join(parts) if parts else "UNKNOWN"
    except Exception:
        return "UNKNOWN"

# ── Main loop ──
def main():
    print(f"[{ts()}] ═══ XAUUSD AI — LightGBM Signal Engine v7 ═══")

    # ── ONE-INSTANCE PID LOCK (v7.3e 2026-08-03) ──
    # Two engines booting on the same active file caused duplicate signals
    # (08-03 dual SELL). Enforce a single live engine: if another instance is
    # already running, refuse to start. Prevents any startup race (manual
    # launch colliding with the watchdog) from ever firing a second signal.
    PID_LOCK = f"{OUTDIR}/.engine.pid"
    def _alive_engine_pid():
        try:
            with open(PID_LOCK) as f:
                pid = int(f.read().strip())
            os.kill(pid, 0)          # signal 0 = existence check
            # confirm it's actually us (not a reused pid)
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    return b"ai_signal_engine" in f.read()
            except Exception:
                return True
        except (ValueError, FileNotFoundError, ProcessLookupError, PermissionError):
            return False
    if _alive_engine_pid():
        print(f"[{ts()}] ⛔ Refusing to start: another engine instance is running (one-at-a-time guard).")
        sys.exit(0)
    try:
        with open(PID_LOCK, "w") as f:
            f.write(str(os.getpid()))
    except Exception:
        pass
    try:
        import atexit
        atexit.register(lambda: (os.path.exists(PID_LOCK) and os.remove(PID_LOCK)))
    except Exception:
        pass

    print(f"[{ts()}] ═══ XAUUSD AI — LightGBM Signal Engine v7 ═══")

    # ── v7 DUAL-MODEL ENSEMBLE ──
    # Placement: 3 seeds × P(win | market + placement + direction), averaged.
    # Direction: 3 seeds × P(up-move | market), a learned multiplier on
    # expectancy (teaches "don't catch falling knives" — no gates).
    def load_ensemble(cfg_path, fallback_single):
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
            ms = [lgb.Booster(model_file=f"{BASE}/models/{m}") for m in cfg["models"]]
            return ms, cfg["seeds"]
        except Exception:
            return [lgb.Booster(model_file=fallback_single)], None

    models, ens_seeds = load_ensemble(ENSEMBLE_CFG, MODEL)
    dir_models, dir_seeds = load_ensemble(DIR_ENSEMBLE_CFG, MODEL)
    # v8 M5 TF GUARD (boot): only load configs stamped base_tf=m5. Legacy M1
    # configs (no base_tf) load into an M5 feature space → silent mismatch →
    # garbage probabilities. Refuse and run model-less (no signals) until the
    # M5 retrain deploys. Same rule the hot-reload path enforces.
    ENGINE_TF = "m5"
    def _base_tf_ok(path):
        try:
            with open(path) as f:
                return json.load(f).get("base_tf", "m1") == ENGINE_TF
        except Exception:
            return False
    if not _base_tf_ok(ENSEMBLE_CFG):
        print(f"[{ts()}] 🛑 v8 TF GUARD: placement config not base_tf=m5 "
              f"(engine={ENGINE_TF}) — refusing M1 models. No signals until the "
              f"M5 retrain (retrain_m5.py) deploys M5-stamped configs.")
        models, dir_models = [], []
    elif not _base_tf_ok(DIR_ENSEMBLE_CFG):
        # v8 M5: the direction model is OPTIONAL. The M5 retrain only deploys
        # it if OOS acc > 0.53 (no-fake-edge gate); at the 180-min horizon the
        # honest result can be coin-flip → no M5 direction model exists yet.
        # The engine must NOT idle for that: direction_prior() falls back to
        # the EMPIRICAL REGIME PRIOR (measured P(up) per regime over 6yr) —
        # market-condition steering, not a gate. Keep the placement ensemble.
        print(f"[{ts()}] ℹ️ v8 TF GUARD: no base_tf=m5 direction model "
              f"(legacy/absent). Placement ensemble loaded; direction = "
              f"empirical regime prior (neutral-tilt, no fake edge).")
        dir_models = []
        # models from load_ensemble() above stay as-is (placement ensemble).
    spec_models, spec_cal = load_specialists(SPEC_CFG, MODEL)
    if not _base_tf_ok(SPEC_CFG) and spec_models:
        # v8 M5: specialists must be same TF lineage as the engine. Legacy
        # M1 specialists (no base_tf field) would route ticks to M1 models
        # computing M5 features — silent corruption. Drop → global ensemble.
        print(f"[{ts()}] 🛑 v8 TF GUARD: clearing {len(spec_models)} legacy specialists "
              f"(base_tf != {ENGINE_TF}) — using global ensemble until M5 specialists train.")
        spec_models, spec_cal = {}, None
    elif not models and spec_models:
        # v8 M5: the global ensemble was refused (TF mismatch) — specialists
        # are equally suspect (trained at the same TF as the ensemble). Drop
        # them so the engine idles cleanly until the M5 retrain deploys.
        print(f"[{ts()}] 🛑 v8 TF GUARD: clearing {len(spec_models)} specialists "
              f"(same TF lineage as refused ensemble) — idle until M5 retrain.")
        spec_models, spec_cal = {}, None
    if spec_models:
        print(f"[{ts()}] REGIME ROUTER: {len(spec_models)} specialists loaded "
              f"({', '.join(sorted(spec_models.keys()))})")
    else:
        print(f"[{ts()}] REGIME ROUTER: no regime_specialists.json yet — using global ensemble")
    with open(FEATURES) as f:
        feats = json.load(f)
    dir_feats = []
    try:
        with open(DIR_FEATURES) as f:
            dir_feats = json.load(f)
    except Exception:
        dir_feats = feats
    ens_tag = f"3-seed ensemble" if ens_seeds else "single (v6 fallback)"
    dir_tag = f"3-seed direction" if dir_seeds else "none (neutral prior)"
    print(f"[{ts()}] Placement: {len(models)} models ({ens_tag}) | {len(feats)} features")
    print(f"[{ts()}] Direction: {len(dir_models)} models ({dir_tag}) | {len(dir_feats)} features")
    print(f"[{ts()}] LEARNED PLACEMENT sweep × learned direction prior (no gates)")
    # v5: learned probability calibration (fits raw LightGBM P → truthful P)
    from calibrate import load_calibration
    cal_knots = load_calibration()
    if cal_knots:
        from calibrate import apply_calibration
        _chk = apply_calibration(np.array([0.5, 0.7]), cal_knots)
        print(f"[{ts()}] Calibration loaded ({len(cal_knots['knots_p'])} knots): raw 0.50→{_chk[0]:.2f} raw 0.70→{_chk[1]:.2f}")
    else:
        print(f"[{ts()}] WARN: no calibration.json — using RAW model probabilities")
    # v7.3c: PER-RR calibration (adaptive trade management). Each TP ratio has
    # its own honest P(win) curve, so the EV sweep can genuinely pick 1:2 vs
    # 1:8 per market state instead of always defaulting to RR 3.0.
    cal_by_rr = None
    _crr_path = f"{BASE}/models/calibration_by_drr.json"
    try:
        if os.path.exists(_crr_path):
            with open(_crr_path) as f:
                cal_by_rr = json.load(f)
            print(f"[{ts()}] Per-dir×RR calibration loaded: {len(cal_by_rr)} curves")
    except Exception as e:
        print(f"[{ts()}] WARN: per-dir×RR calibration load failed: {e}")

    fc = FeatureComputer(maxlen=90000)  # v7: hold ~60d of XM M1 for H1/D1 HTF context

    # Refresh seed: merge MT5 history + live bars so no gap at startup
    try:
        subprocess.run([sys.executable, f"{BASE}/merge_seed.py"], capture_output=True, timeout=120)
    except Exception as e:
        print(f"[{ts()}] merge warn: {e}")

    # Seed with recent history (continuous to last live bar)
    try:
        hist = []
        with open(f"{BASE}/gold_seed.csv") as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split(",")
                try:
                    # Seed times are naive UTC — parse AS UTC so epoch is true.
                    t = datetime.strptime(parts[0][:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                except Exception:
                    continue
                hist.append((t.timestamp(), float(parts[1]), float(parts[2]),
                             float(parts[3]), float(parts[4]), int(parts[5]), int(parts[6])))
        for row in hist[-90000:]:
            fc.add(*row)
        fc._xm_last_bar_ts = hist[-1][0]  # completed-bar watermark (XM-native)
        print(f"[{ts()}] Seeded with {len(hist[-90000:])} REAL XM M1 bars "
              f"(H1/D1 HTF context needs ~60d — v7)")
    except Exception as e:
        print(f"[{ts()}] Seed warn: {e}")

    active = load_active()

    # ── Startup reconciliation ──
    # v2: FIRST-TOUCH resolution via the ticker's 25ms path tracking. The ticker
    # records which level (SL/TP) was touched FIRST and when — so even if the
    # engine was down while BOTH were crossed, we know the true outcome.
    # If the ticker has partial coverage (launched mid-trade), we do NOT guess:
    # we ack honestly as UNVERIFIED and let the user confirm on their terminal.
    if active:
        st = read_tick_state()
        resolved = None
        if st:
            d0 = active["direction"]
            bid0, ask0 = st["bid"], st["ask"]
            verdict = st.get("verdict")
            partial = st.get("coverage_partial", False)
            if verdict in ("SL", "TP") and not partial:
                # FULL coverage — first-touch is definitive (no guessing).
                pnl = resolve_pnl(d0, verdict, active)
                emoji = "❌" if verdict == "SL" else "✅"
                sign = "" if verdict == "SL" else "+"
                tg(f"{emoji} <b>{verdict} HIT (while engine offline)</b> — {d0} @ ${active['entry']:.2f}\n"
                   f"{verdict}: ${active['sl' if verdict=='SL' else 'tp']:.2f} | PnL: {sign}${pnl:.2f} | Conf: {active['conf']:.0%}\n"
                   f"path: low ${st['min_bid']:.2f} / high ${st['max_ask']:.2f}")
                append_journal({"t": time.time(), "dir": d0, "entry": active["entry"],
                                "sl": active["sl"], "tp": active["tp"], "pnl": pnl,
                                "result": verdict, "conf": active["conf"], "reconciled": True,
                                "src": "ticker_path"})
                # CLOSED LOOP: persist feature vector + outcome so the retrain
                # learns from this REAL result too (reconciled wins/losses must
                # not escape the training loop — 07-31 TP would have been lost).
                try:
                    append_outcome({"t": time.time(), "dir": d0, "entry": active["entry"],
                                    "sl": active["sl"], "tp": active["tp"], "pnl": pnl,
                                    "result": verdict, "conf": active["conf"],
                                    "feats": active.get("feats", {})})
                except Exception:
                    pass
                print(f"[{ts()}] 🔔 Reconciled (ticker path): {verdict} HIT for {d0} @ {active['entry']:.2f} (PnL {pnl:+.2f})")
                active = None; save_active(None)
            else:
                # No verdict or partial coverage — check current price as a hint,
                # but ONLY ack if unambiguous (price sits beyond ONE level and
                # ticker has full coverage of the whole trade).
                if verdict is None and not partial:
                    if d0 == "BUY":
                        if bid0 <= active["sl"]:
                            resolved = ("SL", active["sl"] - active["entry"])
                        elif bid0 >= active["tp"]:
                            resolved = ("TP", active["tp"] - active["entry"])
                    else:
                        if ask0 >= active["sl"]:
                            resolved = ("SL", active["entry"] - active["sl"])
                        elif ask0 <= active["tp"]:
                            resolved = ("TP", active["entry"] - active["tp"])
                if resolved:
                    r_kind, pnl = resolved
                    emoji = "❌" if r_kind == "SL" else "✅"
                    sign = "" if r_kind == "SL" else "+"
                    tg(f"{emoji} <b>{r_kind} HIT (while engine offline)</b> — {d0} @ ${active['entry']:.2f}\n"
                       f"{r_kind}: ${active['sl' if r_kind=='SL' else 'tp']:.2f} | PnL: {sign}${pnl:.2f} | Conf: {active['conf']:.0%}")
                    append_journal({"t": time.time(), "dir": d0, "entry": active["entry"],
                                    "sl": active["sl"], "tp": active["tp"], "pnl": pnl,
                                    "result": r_kind, "conf": active["conf"], "reconciled": True})
                    print(f"[{ts()}] 🔔 Reconciled (current px): {r_kind} HIT for {d0} @ {active['entry']:.2f} (PnL {pnl:+.2f})")
                    active = None; save_active(None)
                elif partial:
                    # Ticker missed trade open — path gap. If the trade is still
                    # OPEN (price between levels, no verdict), restore it: the
                    # ticker now has full coverage going forward. Only if the
                    # trade LOOKS resolved do we ack honestly as UNVERIFIED.
                    beyond = False
                    if d0 == "BUY":
                        beyond = bid0 <= active["sl"] or bid0 >= active["tp"]
                    else:
                        beyond = ask0 >= active["sl"] or ask0 <= active["tp"]
                    if beyond:
                        tg(f"⚠️ <b>Trade outcome UNVERIFIED</b> — engine was offline during part of this trade.\n"
                           f"{d0} @ ${active['entry']:.2f} | SL ${active['sl']:.2f} | TP ${active['tp']:.2f} | Conf: {active['conf']:.0%}\n"
                           f"Current XM bid/ask: ${bid0:.2f}/${ask0:.2f}\n"
                           f"<i>Check your MT5 terminal for the actual fill — I will not guess.</i>")
                        print(f"[{ts()}] ⚠️ UNVERIFIED (partial coverage): {d0} @ {active['entry']:.2f} — awaiting user confirmation")
                        active = None; save_active(None)
                    else:
                        print(f"[{ts()}] 🔄 Restored open {d0} @ {active['entry']:.2f} (partial coverage — ticker now tracking full path) | bid/ask {bid0:.2f}/{ask0:.2f}")
                else:
                    print(f"[{ts()}] 🔄 Restored open {d0} @ {active['entry']:.2f} | SL {active['sl']} | TP {active['tp']} | XM bid/ask {bid0:.2f}/{ask0:.2f}")
        else:
            print(f"[{ts()}] ⚠️ ticker state unavailable at boot — waiting for ticker before restoring trade")

    last_bar_min = None
    poll = 3           # base poll interval (v7.5 2026-08-03: 10→3s — idle signal
                       # latency was ~13s worst case; measured signal compute is
                       # 2.6s/cycle, so 3s poll keeps CPU sane while signals fire
                       # ~6s after a bar close. Ack path is independent: 1s tight
                       # loop when a trade is open.)
    heartbeat = 0      # periodic status print
    loop_err_times = []    # v7.3f: sliding 5-min window of loop exceptions —
    last_degraded_alert = 0  # detects a HUNG-but-alive engine (watchdog only
                              # checks PID existence, not progress)
    # ── v7 POSITION-STATE (T3-1): the engine's own record, injected into
    # closed-loop outcome rows so the model learns from its own performance
    # (day P&L, win/loss streak, trades so far today). Learned, not gated.
    day_pnl_engine = 0.0
    streak_engine = 0
    trades_today_engine = 0
    _day_key = None
    model_mtime = None # hot-reload: pick up retrained model without restart
    spec_mtime = None  # v7.7 regime router: reload specialists on retrain
    cal_mtime = None    # v7.3f: calibration.json / calibration_by_drr.json were
                         # loaded ONCE at startup and never refreshed — a retrain
                         # could update calibration for days before the engine
                         # picked it up (only a restart did it). Hot-reload same
                         # as the model ensemble.
    CALIB_PATH = f"{BASE}/models/calibration.json"
    # (v8 TF guard: ENGINE_TF / _base_tf_ok defined at boot above — shared by
    # the hot-reload path below. A silent TF mismatch is catastrophic; the boot
    # guard refuses M1 configs and the reload guard does the same.)
    def maybe_reload_model():
        nonlocal models, dir_models, spec_models, spec_cal, model_mtime, spec_mtime, feats, dir_feats, cal_knots, cal_by_rr, cal_mtime
        try:
            m = os.path.getmtime(ENSEMBLE_CFG)
            dm = os.path.getmtime(DIR_ENSEMBLE_CFG)
            if model_mtime is not None and max(m, dm) > model_mtime:
                if not _base_tf_ok(ENSEMBLE_CFG):
                    print(f"[{ts()}] ⚠️ ENSEMBLE RELOAD REFUSED — placement not base_tf=m5 "
                          f"(engine={ENGINE_TF}, model files not M5). Keeping last valid models.")
                    model_mtime = max(m, dm)
                    return
                models, _ = load_ensemble(ENSEMBLE_CFG, MODEL)
                if _base_tf_ok(DIR_ENSEMBLE_CFG):
                    dir_models, _ = load_ensemble(DIR_ENSEMBLE_CFG, MODEL)
                else:
                    # direction model optional (no-fake-edge gate) — keep
                    # whatever direction models we have; if none, the
                    # empirical regime prior steers the side.
                    print(f"[{ts()}] ℹ️ reload: direction model absent/not-M5 — keeping prior direction state")
                with open(FEATURES) as f:
                    feats.clear(); feats.extend(json.load(f))
                try:
                    with open(DIR_FEATURES) as f:
                        dir_feats.clear(); dir_feats.extend(json.load(f))
                except Exception:
                    pass
                print(f"[{ts()}] 🔄 Ensemble hot-reloaded (retrain picked up) | {len(models)} placement + {len(dir_models)} direction | {len(feats)} feats")
            model_mtime = max(m, dm)
        except Exception as e:
            print(f"[{ts()}] reload warn: {e}")
        # v7.7 regime router: hot-reload specialists when a new set is trained
        try:
            sm = os.path.getmtime(SPEC_CFG)
            if spec_mtime is not None and sm > spec_mtime:
                if not _base_tf_ok(SPEC_CFG):
                    print(f"[{ts()}] ⚠️ SPECIALIST RELOAD REFUSED — base_tf mismatch (engine={ENGINE_TF}). Keeping last valid specialists.")
                else:
                    spec_models, spec_cal = load_specialists(SPEC_CFG, MODEL)
                    if spec_models:
                        print(f"[{ts()}] 🔄 REGIME ROUTER hot-reloaded: {len(spec_models)} specialists")
                    else:
                        print(f"[{ts()}] 🔄 REGIME ROUTER hot-reload: specialists removed, falling back to global ensemble")
            spec_mtime = sm
        except Exception as e:
            print(f"[{ts()}] specialist reload warn: {e}")
        try:
            cm = os.path.getmtime(CALIB_PATH) if os.path.exists(CALIB_PATH) else 0
            crm = os.path.getmtime(_crr_path) if os.path.exists(_crr_path) else 0
            latest_cal = max(cm, crm)
            if cal_mtime is not None and latest_cal > cal_mtime:
                cal_knots = load_calibration()
                if os.path.exists(_crr_path):
                    with open(_crr_path) as f:
                        cal_by_rr = json.load(f)
                else:
                    cal_by_rr = None
                print(f"[{ts()}] 🔄 Calibration hot-reloaded (base + per-dir×RR)")
            cal_mtime = latest_cal
        except Exception as e:
            print(f"[{ts()}] calib reload warn: {e}")

    while True:
        try:
            # ── XM-NATIVE BARS (v4): read ONLY the XM ticker's real data.
            # The ticker (25ms, owns MT5 connection) writes bid/ask + the
            # forming M1 bar (built from REAL XM ticks) to xm_tick_state.json,
            # and completed bars to xm_live_bars.jsonl. NO TradingView.
            st_now = read_tick_state()
            if st_now is None:
                # Market closed (Fri close → Mon open) or ticker down. Never
                # fire signals on stale data, never fabricate bars. The
                # watchdog restarts a dead ticker; Monday reopen resumes.
                time.sleep(poll); continue
            q = {"close": (st_now["bid"] + st_now["ask"]) / 2.0, "time": st_now["ts"]}

            # heartbeat every ~2 min so we know it's alive
            heartbeat += 1
            if heartbeat % 12 == 0:
                print(f"[{ts()}] ♥ polling OK | XM ${q['close']:.2f} | spread ${st_now['ask']-st_now['bid']:.2f} | bars {len(fc.times)}")

            # completed bars (real XM, from ticker's tick-built stream)
            for bar in read_completed_bars(fc._xm_last_bar_ts):
                fc.add(*bar)
                fc._xm_last_bar_ts = bar[0]

            # forming bar from ticker state (real XM OHLC, exact spread)
            cb = st_now.get("cur_bar")
            if cb and cb.get("t"):
                bar_ts = float(cb["t"])
                xm_spread_pts = int(cb.get("spread", 20))
                if bar_ts != last_bar_min:
                    fc.add(bar_ts, cb["o"], cb["h"], cb["l"], cb["c"], cb.get("v", 0), xm_spread_pts)
                    last_bar_min = datetime.utcfromtimestamp(bar_ts).strftime("%Y-%m-%d %H:%M")
                else:
                    # same-minute update — merge range (max high/min low)
                    fc.add(bar_ts, cb["o"], cb["h"], cb["l"], cb["c"], cb.get("v", 0), xm_spread_pts)

            # ── SL/TP check — v2: ticker first-touch (25ms), subprocess fallback ──
            if active:
                st = read_tick_state()
                qx = None
                if st:
                    bid, ask = st["bid"], st["ask"]
                    px_disp = q["close"] if q else bid
                else:
                    qx = xm_tick()  # rare fallback: ticker stale/down
                    if qx and qx.get("bid") is not None:
                        bid, ask = qx["bid"], qx["ask"]
                        px_disp = q["close"] if q else bid
                    else:
                        # NO live quote anywhere (ticker stale AND MT5 spawn
                        # failed) = market closed or both paths down. Never
                        # resolve an open trade on TV's stale close — keep it
                        # open until a REAL quote returns (Monday reopen).
                        bid = ask = None
                        px_disp = q["close"] if q else None
                d = active["direction"]
                hit = None  # ("SL"/"TP"/"TIME", pnl)
                # ── v7.5 TIME-HOLD (2026-08-03): NO artificial 60-bar close ──
                # USER MANDATE: "time-holding is unnecessary — after a few mins
                # past the old timeout the market moves to TP; I hold no matter
                # what." The engine now mirrors real manual trading: a virtual
                # position rides until a REAL SL/TP touch (ticker first-touch
                # verdict) or a real live quote crosses — it is NEVER force-closed
                # by a bar clock. This stops the "SL hit / TP fly" bleed the old
                # 60-bar TIME close caused (booking a small winner early while the
                # market ran on to TP). The user is kept AWARE of the hold, but
                # not told to close.
                max_bars = 60  # informational horizon only — no longer resolves
                bars_held = 0
                if active.get("entry_bar_ts"):
                    bars_held = int((time.time() - active["entry_bar_ts"]) // 60)
                elif active.get("time"):
                    bars_held = int((time.time() - active["time"]) // 60)
                # Awareness note at the old-timeout mark (once) — keeps the user
                # informed that the trade remains open and is being held to TP/SL,
                # WITHOUT issuing a forced-close command.
                if bars_held >= max_bars and not active.get("hold_note_sent"):
                    active["hold_note_sent"] = True
                    save_active(active)
                    tg(f"⏳ <b>HOLDING — {d} STILL OPEN past {max_bars} bars</b>\n"
                       f"━━━━━━━━━━━━━━━\n"
                       f"{d} @ ${active['entry']:.2f} | SL ${active['sl']:.2f} | TP ${active['tp']:.2f}\n"
                       f"<b>Neither SL nor TP hit yet ({bars_held} bars).</b>\n"
                       f"👉 <i>You are holding to TP/SL per your plan. The engine\n"
                       f"   stays tracking — no premature close.</i>\n"
                       f"━━━━━━━━━━━━━━━\n"
                       f"💡 This trade is NOT time-closed — it resolves only on a\n"
                       f"   real SL {active['sl']:.2f} / TP {active['tp']:.2f} touch.")
                    print(f"[{ts()}] ⏳ HOLD note sent (bar {bars_held}) — {d} riding to SL/TP")
                # ── WEEKEND GAP WARNING (v8.3): once per open trade, when <90
                # min remain before Friday's XM close and the trade is still
                # open. Informational only — mirrors the hold-note contract:
                # the engine keeps tracking, the USER decides whether to hold
                # the weekend gap. (SL/TP resolution continues through the
                # close; a gap through SL on Sunday reopen is real risk.)
                if not active.get("weekend_warn_sent"):
                    _wk = weekend_close_minutes()
                    if _wk is not None and 0 < _wk <= 90:
                        active["weekend_warn_sent"] = True
                        save_active(active)
                        tg(f"⚠️ <b>WEEKEND GAP RISK — {d} STILL OPEN</b>\n"
                           f"━━━━━━━━━━━━━━━\n"
                           f"{d} @ ${active['entry']:.2f} | SL ${active['sl']:.2f} | TP ${active['tp']:.2f}\n"
                           f"⏳ <b>XM closes in {_wk} min</b> (Friday 23:55 server).\n"
                           f"👉 <i>If not resolved by close, this position rides the\n"
                           f"   weekend and may GAP through SL on Sunday reopen.\n"
                           f"   Decide now: hold (your plan) or flatten before close.</i>\n"
                           f"━━━━━━━━━━━━━━━\n"
                           f"Engine keeps tracking — resolves on first real touch.")
                        print(f"[{ts()}] ⚠️ Weekend gap warning sent ({_wk} min to Friday close) — {d} open")
                # NO hit is forced here. Resolution only via real first-touch below.
                # 1) First-touch verdict from the ticker's 25ms path — definitive
                #    when coverage is full (ticker was running when trade opened).
                if st and st.get("verdict") in ("SL", "TP") and not st.get("coverage_partial"):
                    hit = (st["verdict"], resolve_pnl(d, st["verdict"], active))
                elif st and st.get("verdict") in ("SL", "TP") and st.get("coverage_partial"):
                    # Ticker saw a touch but missed the trade's open path — we
                    # can't be sure it was FIRST. Don't guess: report honestly.
                    tg(f"⚠️ <b>Level touched but UNVERIFIED order</b> — ticker had partial coverage.\n"
                       f"{d} @ ${active['entry']:.2f} | SL ${active['sl']:.2f} | TP ${active['tp']:.2f}\n"
                       f"ticker saw: {st['verdict']} | current bid/ask ${bid:.2f}/${ask:.2f}\n"
                       f"<i>Confirm on your MT5 terminal.</i>")
                    print(f"[{ts()}] ⚠️ partial-coverage touch {st['verdict']} — acked UNVERIFIED, user confirms")
                    append_journal({"t": time.time(), "dir": d, "entry": active["entry"],
                                    "sl": active["sl"], "tp": active["tp"], "pnl": 0.0,
                                    "result": "UNVERIFIED", "conf": active["conf"],
                                    "src": "ticker_partial"})
                    active = None; save_active(None); continue
                elif st is None:
                    # 2) Fallback: live bid/ask check (only when ticker unreachable)
                    if d == "BUY":
                        if bid is not None and bid <= active["sl"]:
                            hit = ("SL", active["sl"] - active["entry"])
                        elif bid is not None and bid >= active["tp"]:
                            hit = ("TP", active["tp"] - active["entry"])
                    else:
                        if ask is not None and ask >= active["sl"]:
                            hit = ("SL", active["entry"] - active["sl"])
                        elif ask is not None and ask <= active["tp"]:
                            hit = ("TP", active["entry"] - active["tp"])
                # (no direct price check when ticker alive & no verdict — ticker
                #  already knows the full path; a fresh cross will set verdict
                #  within 25ms, far faster than this loop)
                if hit:
                    r_kind, pnl = hit
                    # ── v7 POSITION-STATE update (T3-1): day P&L / streak /
                    # trades-today — the model's own record, learned over time
                    # via the closed loop (never a gate on entries).
                    try:
                        _dk = datetime.now(timezone.utc).date().isoformat()
                        if _day_key != _dk:
                            _day_key = _dk
                            day_pnl_engine = 0.0
                            trades_today_engine = 0
                        day_pnl_engine += pnl
                        trades_today_engine += 1
                        streak_engine = streak_engine + 1 if pnl > 0 else (-1 if pnl < 0 else streak_engine)
                    except Exception:
                        pass
                    if r_kind == "TIME":
                        emoji = "⏰"
                        head = "⚠️ ACTION NEEDED — TIME LIMIT REACHED"
                        sign = "+"
                        px_txt = (f"XM bid/ask: ${bid:.2f}/${ask:.2f}" if bid is not None else "no live quote")
                        lvl_txt = (f"<b>60 bars up — neither SL nor TP hit.</b>\n"
                                   f"👉 <b>CLOSE YOUR {d} TRADE NOW</b> (market price)\n{px_txt}\n"
                                   f"<i>Virtual trade closed at market for the learning loop; your real position is still open until YOU close it.</i>")
                    else:
                        emoji = "❌" if r_kind == "SL" else "✅"
                        head = f"TRADE CLOSED — {r_kind} HIT"
                        sign = "" if r_kind == "SL" else "+"
                        lvl_txt = f"{'🛑' if r_kind=='SL' else '✅'} {r_kind}: ${active['sl' if r_kind=='SL' else 'tp']:.2f}\n<i>This position is resolved — no manual action needed.</i>"
                    tg(f"{emoji} <b>{head}</b>\n"
                       f"━━━━━━━━━━━━━━━\n"
                       f"{d} @ ${active['entry']:.2f}\n"
                       f"{lvl_txt}\n"
                       f"💰 <b>PnL: {sign}${pnl:.2f}</b> | Conf: {active['conf']:.0%}\n"
                       f"━━━━━━━━━━━━━━━\n"
                       f"🧠 <i>Outcome recorded &amp; queued for the learning loop.</i>")
                    append_journal({"t": time.time(), "dir": d, "entry": active["entry"],
                                    "sl": active["sl"], "tp": active["tp"], "pnl": pnl,
                                    "result": r_kind, "conf": active["conf"], "src": "ticker_path" if st else "xm_fallback"})
                    # CLOSED LOOP: persist feature vector + outcome for retrain
                    try:
                        append_outcome({"t": time.time(), "dir": d, "entry": active["entry"],
                                        "sl": active["sl"], "tp": active["tp"], "pnl": pnl,
                                        "result": r_kind, "conf": active["conf"],
                                        "regime": active.get("regime", ""),
                                        "sl_atr": active.get("sl_atr", None),
                                        "feats": active.get("feats", {})})
                    except Exception:
                        pass
                    # SELF-ANALYSIS: on a loss (SL hit OR negative TIME drift),
                    # run the audit (root-cause + lesson). v8 passes the regime
                    # and SL-in-ATR so the classifier can compare against the
                    # learned MFE/MFA excursion band (EXCURSION_STOP diagnosis).
                    if pnl < 0 and active.get("feats"):
                        try:
                            from trade_audit import audit_trade
                            audit_trade({"dir": d, "entry": active["entry"], "sl": active["sl"],
                                         "tp": active["tp"], "pnl": pnl, "conf": active["conf"],
                                         "regime": active.get("regime", ""),
                                         "sl_atr": active.get("sl_atr", None),
                                         "feats": active.get("feats", {})})
                        except Exception as e:
                            print(f"[{ts()}] audit warn: {e}")
                    print(f"[{ts()}] 🔔 {r_kind} HIT — {d} @ {active['entry']:.2f} (PnL {pnl:+.2f}) bid/ask {bid:.2f}/{ask:.2f}")
                    active = None; save_active(None); continue
                time.sleep(1)  # tight SL/TP loop — ticker resolves wicks in 25ms
                continue  # trade open — no new signal

            # ── Signal generation ──
            # MARKET-CLOSED GUARD: never fire signals on stale data. The XM
            # ticker is the ground truth — if it's stale (30s), the market is
            # closed (Fri close → Mon open) or the daemon is down. TV keeps
            # serving Friday's last close all weekend, and tv_price() stamps
            # time=now, so WITHOUT this guard the engine would fire phantom
            # signals on dead bars and poison the training data. (Watchdog
            # restarts a dead ticker within 5m; no Wine spawn needed here.)
            # REAL-MARKET GUARD: also block when the TV feed itself is frozen
            # mid-session (rate-limited) while XM keeps moving — a signal on a
            # stale vector fires at a real XM price $5 away from the model's
            # view, corrupting the entry basis (07-31: two identical feats).
            if read_tick_state() is None:
                time.sleep(poll); continue
            maybe_reload_model()  # pick up retrained model if swapped
            fx = fc.features()
            if fx is None:
                time.sleep(poll); continue
            atr = fx.get("atr_14", 3.0)
            px = q["close"]

            # ── LEARNED PLACEMENT SWEEP (v4 2026-08-01) ──
            # NO gates, NO hardcoded geometry. The model learned
            # P(win | market + placement + direction) over the full grid.
            # Evaluate every candidate, pick the one with max expected value
            # across BOTH directions; fire only if EV > 0. The model decides
            # direction, SL width, AND TP distance from its learned surface.
            st = read_tick_state()
            if st:
                xm_bid, xm_ask = st["bid"], st["ask"]
                xm_spread = max(xm_ask - xm_bid, 0.05)  # REAL XM spread ($)
            else:
                xm_bid, xm_ask = px, px + 0.20
                xm_spread = 0.20
            # ── REGIME ROUTER (v7.7, 2026-08-04) ──
            # A range-day specialist sells overbought; a trend specialist buys
            # pullbacks. Route each tick to the specialist for the CURRENT
            # regime (computed from fx via the SAME pure function used to bin
            # training rows — train/live routing can never diverge). If the
            # current regime has no specialist (or specialists are absent),
            # fall back to the global ensemble. Safe: never crashes the sweep.
            route_models = models
            route_regime = None
            route_cal = None   # v7.7b: per-regime calibration (specialist OOF)
            # v8 M5: compute the regime UNCONDITIONALLY — the learned placement
            # prior (placement_prior.json) is keyed by regime×direction and must
            # anchor the sweep even when no specialist exists for this regime.
            # With spec_models empty, route_regime still feeds best_placement's
            # learned-placement injection (SL/TP from 6yr MFE/MFA excursions).
            try:
                from features import regime_bin
                route_regime = regime_bin(fx)
            except Exception:
                route_regime = None
            if spec_models:
                try:
                    rmod = spec_models.get(route_regime)
                    if rmod:
                        route_models = rmod
                        route_cal = (spec_cal or {}).get(route_regime)
                except Exception:
                    route_models = models
                if route_regime:
                    _rr = f" [{route_regime} specialist]"
                else:
                    _rr = " [global ensemble]"
            else:
                _rr = ""
            # v8 M5: TF-guard may have refused the old M1 models (engine boots
            # before the M5 retrain deploys). NO models → NO sweep → NO fire.
            # The engine idles safely, polls the ticker, and comes alive the
            # moment the retrain writes M5-stamped configs (hot-reload).
            if not route_models:
                if not getattr(globals(), "_no_models_notice", False):
                    print(f"[{ts()}] ⏳ no models loaded (M5 retrain pending) — idle, no signals")
                    globals()["_no_models_notice"] = True
                time.sleep(poll); continue
            _cal_used = route_cal if route_cal is not None else cal_by_rr
            buy = best_placement(route_models, feats, fx, atr, xm_spread, "BUY", cal_knots, _cal_used, route_regime)
            sell = best_placement(route_models, feats, fx, atr, xm_spread, "SELL", cal_knots, _cal_used, route_regime)

            # ── SAME-IDEA SUPPRESSION (2026-08-04, v7.9: MARKET-STATE based) ──
            # A resolved trade frees the engine to re-evaluate — and in a flat
            # market it re-fires the IDENTICAL idea (same direction, near-same
            # TP/SL) with no new information, because the feature bar only
            # recomputes on bar completion. That is the SAME bet, not a new
            # opportunity — re-taking it right after a loss compounds losers.
            # v7.9 HARNESS (not hardening): suppression is driven by MARKET
            # STATE, not a clock. A re-fire is only held back while the market
            # is genuinely unchanged: same regime + price within 0.5×ATR of the
            # last signal + same direction + same TP zone. The moment price
            # moves ≥0.5×ATR (or the regime flips), the market is in a NEW
            # state — the learned exp>0 IS the real-signal gate, so it fires
            # immediately. This is "wait until the model finds a REAL new
            # opportunity" taught by the market itself, never an arbitrary time
            # gate. No loss-side stacking: a same-state same-bet is held back.
            MIN_SIGNAL_GAP = 30        # safety floor: >30s between ANY two fires
                                       # (features only recompute once/min, so two
                                       # materially-different signals can't be <30s)
            SAME_IDEA_BAND = 1.00      # $ TP-distance proximity → "same idea"
            SAME_IDEA_MOVE = 0.50      # ATR-multiplier: price moved <0.5×ATR → unchanged state
            _last = globals().get("_last_signal")
            _sup = False
            if _last is not None:
                _tago = time.time() - _last["time"]
                _is_buy_now = (buy[4] >= sell[4]) if (buy and sell) else (buy is not None)
                _same_dir = (_is_buy_now == _last.get("dir_buy"))
                if _tago < MIN_SIGNAL_GAP:
                    _sup = True
                elif _same_dir and _last.get("regime") == route_regime:
                    _moved = abs(px - _last.get("px", px))
                    _best = buy if _is_buy_now else sell
                    if _moved < SAME_IDEA_MOVE * atr and _best is not None and abs(_best[1] - _last.get("tp", -1)) <= SAME_IDEA_BAND:
                        _sup = True
            if _sup:
                print(f"[{ts()}] ⏸ hold same-idea (state unchanged {time.time()-_last['time']:.0f}s, regime {route_regime}, Δ{abs(px-_last.get('px',px)):.2f}) — waiting for a REAL new move")
                time.sleep(poll); continue

            # ── v7 LEARNED DIRECTION PRIOR (T2-1) ──
            # P(up-move | market) from the direction ensemble multiplies each
            # side's expectancy: final_exp(BUY) = Exp × P(up), SELL × (1−P(up)).
            # Pure learned multiplier — in a downtrend P(up) is low and BUY
            # expectancy gets crushed → the model refuses falling knives.
            # NOTE: spread IS a direction-model feature (points) — inject the
            # real XM spread into fx before the prior, same as placement_row.
            fx["spread"] = round(xm_spread * 100.0, 2)
            p_up = direction_prior(dir_models, dir_feats, fx)
            buy_exp_w = buy[4] * p_up if buy else -1e9
            sell_exp_w = sell[4] * (1 - p_up) if sell else -1e9

            # Max-EXPECTANCY decision across directions — the model's call,
            # not mine. (Exp per $ risked is scale-free; dollar EV would
            # always pick the widest stop. v5.1 2026-08-01.)
            if buy and sell:
                if buy_exp_w >= sell_exp_w:
                    sl_dist, tp_dist, conf, ev, exp, sl_mult, tp_ratio = buy; direction = "BUY"
                else:
                    sl_dist, tp_dist, conf, ev, exp, sl_mult, tp_ratio = sell; direction = "SELL"
            elif buy:
                sl_dist, tp_dist, conf, ev, exp, sl_mult, tp_ratio = buy; direction = "BUY"
            elif sell:
                sl_dist, tp_dist, conf, ev, exp, sl_mult, tp_ratio = sell; direction = "SELL"
            else:
                sl_dist = tp_dist = conf = ev = exp = sl_mult = tp_ratio = None; direction = None

            if direction is not None and exp > 0:
                # ── v8 SIGNAL RATING GATE (2026-08-07) ──
                # "Not every trade it sees should be taken" — institutional
                # quality score. The rating combines calibrated P(win),
                # expectancy, regime confidence and MFE/MFA excursion headroom
                # with WEIGHTS LEARNED from 6yr data (signal_rating.json). Fire
                # only when rating >= learned threshold (the rating decile where
                # historical realized expectancy turns positive). Pure learning,
                # no hardcoded selectivity floor. If no rating config exists yet
                # (pre-training), the fallback never blocks (rating always
                # passes) so the engine behaves exactly like v7.
                try:
                    from signal_rating import rate_signal, rating_threshold
                    rating, rating_parts = rate_signal(
                        fx, direction, conf, exp,
                        float(sl_dist / max(atr, 1e-9)), route_regime)
                    rating_gate = float(rating_threshold())
                    if rating < rating_gate:
                        print(f"[{ts()}] ⭐ rating {rating:.1f} < threshold {rating_gate:.1f} "
                              f"— quality gate holds ({direction}, regime {route_regime})")
                        time.sleep(poll); continue
                except Exception as e:
                    rating, rating_parts, rating_gate = None, {}, 0.0
                    print(f"[{ts()}] rating warn: {e}")
                if direction == "BUY":
                    entry = xm_ask          # pay the spread to enter
                    sl = entry - sl_dist - xm_spread
                    tp = entry + tp_dist
                else:
                    entry = xm_bid          # sell at bid
                    sl = entry + sl_dist + xm_spread
                    tp = entry - tp_dist
                # FULL model row for the chosen placement (market + geometry + direction)
                _fc = FeatureComputer(maxlen=2)
                _row, _, _ = _fc.placement_row(fx, atr, xm_spread, direction, sl_mult, tp_ratio)
                # ── v7 POSITION-STATE + MICROSTRUCTURE (T3-1, T2-3) ──
                # Day P&L, streak, trades-today, tick imbalance/spread dynamics.
                # These are LIVE-ONLY signals the model hasn't seen in training
                # yet — they ride along in the closed-loop outcome row so the
                # NEXT retrain teaches them (pure learning, no gates).
                try:
                    _row["day_pnl"] = day_pnl_engine
                    _row["streak"] = streak_engine
                    _row["trades_today"] = trades_today_engine
                    if st and st.get("ms"):
                        _row["ticks_60s"] = st["ms"].get("ticks_60s", 0)
                        _row["imb_60s"] = st["ms"].get("imb_60s", 0)
                        _row["imb_300s"] = st["ms"].get("imb_300s", 0)
                        _row["spread_z60"] = (st["ms"].get("spread_now", 0.2) - st["ms"].get("spread_mean_60s", 0.2)) / max(st["ms"].get("spread_std_60s", 1e-6), 1e-6)
                except Exception:
                    pass
                active = {"direction": direction, "entry": entry, "sl": round(sl, 2),
                          "tp": round(tp, 2), "conf": conf, "time": time.time(),
                          "entry_bar_ts": fc.times[-1] if fc.times else time.time(),
                          "p_up": p_up,
                          "regime": route_regime,   # v8: regime for loss self-analysis
                          "sl_atr": float(sl_dist / max(atr, 1e-9)),  # v8: SL in ATR units
                          "feats": {k: _row.get(k, 0.0) for k in feats}}
                save_active(active)
                # record this fired signal for same-idea suppression (next fire)
                globals()["_last_signal"] = {
                    "time": time.time(),
                    "dir_buy": (direction == "BUY"),
                    "tp": abs(tp - entry),
                    "px": px,               # v7.9: market price at last signal
                    "regime": route_regime, # v7.9: regime at last signal
                }
                emoji = "🟢" if direction == "BUY" else "🔴"
                rr = abs(tp - entry) / max(abs(entry - sl), 0.01)
                reg = regime_label(fx)
                dir_pct = f"{p_up:.0%}" if direction == "BUY" else f"{1-p_up:.0%}"
                # ── NEWS PROXIMITY (v7.3): surface the next scheduled macro event.
                # Pure schedule info for the HUMAN — the model already saw
                # min_to_event as a feature; this line is for your execution
                # decision (skip/scale near NFP/FOMC/CPI). Never a gate.
                try:
                    from features import _events as _ev_factory
                    _now = time.time()
                    _evts = _ev_factory()
                    _nxt = next((e for e in _evts if e[0] >= _now - 60), None)
                    if _nxt:
                        _mins = (_nxt[0] - _now) / 60.0
                        if _mins <= 180:
                            news_line = f"⚠️ <b>News:</b> {_nxt[1].upper()} in {_mins:.0f} min\n"
                        else:
                            news_line = f"🗓 <b>Next event:</b> {_nxt[1].upper()} in {_mins/60:.1f}h\n"
                    else:
                        news_line = ""
                except Exception:
                    news_line = ""
                # ── WEEKEND GAP AWARENESS (v8.3): surface how much trading time
                # remains before the XM weekly close. If a 180-min trade can't
                # complete before Friday's close, the USER must decide whether
                # to hold the weekend gap (no gate — the model's learned
                # placement still fires; this is execution information, same
                # contract as the news line).
                wk_line = ""
                try:
                    _wk = weekend_close_minutes()
                    if _wk is not None:
                        if _wk <= 0:
                            wk_line = "⛔ <b>Market closed for the weekend</b> (XM) — no new entries until Sunday reopen\n"
                        elif _wk <= 240:
                            wk_line = f"⏳ <b>Friday close in {_wk} min</b> — 180-min trade may ride the weekend gap\n"
                        elif _wk <= 720:
                            wk_line = f"🗓 <b>Friday close in {_wk//60}h</b> — plan entry/exit accordingly\n"
                except Exception:
                    wk_line = ""
                tg(f"📡 <b>AI SIGNAL — {direction}</b> {emoji}\n"
                   f"━━━━━━━━━━━━━━━\n"
                   f"🎯 <b>Entry:</b> ${entry:.2f}\n"
                   f"🛑 <b>SL:</b> ${sl:.2f} (${sl_dist:.2f} away)\n"
                   f"✅ <b>TP:</b> ${tp:.2f} (${tp_dist:.2f} away) | R:R {rr:.2f}\n"
                   f"📊 <b>P(win):</b> {conf:.0%} | <b>P(dir):</b> {dir_pct} | <b>Exp:</b> {exp:+.2f}\n"
                   f"⭐ <b>Rating:</b> {rating if rating is not None else '—'}/100\n"
                   f"🌍 <b>Market:</b> {reg}\n"
                   f"{news_line}"
                   f"{wk_line}"
                   f"━━━━━━━━━━━━━━━\n"
                   f"ATR {atr:.2f} | XM bid/ask ${xm_bid:.2f}/${xm_ask:.2f} | spread ${xm_spread:.2f}\n"
                   f"<i>Closed-loop: outcome will be learned &amp; audited.</i>")
                print(f"[{ts()}] 🔥 AI {direction} @ {entry:.2f} | P={conf:.0%} | P(dir)={p_up:.2f} | Exp={exp:+.2f} | ⭐{rating if rating is not None else '—'} | SL {sl} TP {tp} | {reg}")
                time.sleep(1)

            time.sleep(poll)
        except Exception as e:
            now_e = time.time()
            loop_err_times.append(now_e)
            loop_err_times[:] = [t for t in loop_err_times if now_e - t < 300]
            print(f"[{ts()}] LOOP ERR ({len(loop_err_times)} in last 5m): {e}")
            # v7.3f: a repeatedly-erroring loop is "alive" to the watchdog's
            # PID check but produces zero signals — alert explicitly instead
            # of spinning silently forever. 15-min cooldown between alerts.
            if len(loop_err_times) >= 5 and now_e - last_degraded_alert > 900:
                last_degraded_alert = now_e
                tg(f"⚠️ <b>Engine loop degraded</b> — {len(loop_err_times)} errors in the last 5 min.\n"
                   f"Last: {e}\nSignals may be silently stalled. Check engine.log.")
            time.sleep(poll)

if __name__ == "__main__":
    main()
