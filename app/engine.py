#!/usr/bin/env python3
"""AI Signal Engine v8 (2026-08-17 rebuild, relocated into app/ for V3
Phase 1) -- pure-learned two-stage model (learning.train / decision.signal),
XM-native live bars (market/mt5_feed.py, Phase 2's rewrite of the old
xm_ticker.py), signal-only Telegram delivery. No hardcoded direction/entry
logic anywhere: primary+meta CatBoost models decide side and precision;
TP/SL are the same vol-scaled barrier widths the models were trained
against (decision/signal.py).

Contract with market/mt5_feed.py (unchanged from the old ticker -- Phase 2
kept every existing file output behavior-preserving, see market/README.md):
  ACTIVE (.active_signal_ai.json) : engine writes {time,direction,sl,tp} to
    open a trade, {} to clear it. Ticker tracks first-touch TP/SL against its
    own 25ms XM tick stream and reports the verdict in STATE.
  STATE (xm_tick_state.json)      : ticker's bid/ask/verdict/market_closed,
    read-only here.

One trade at a time (matches the user's manual "take signal, wait for
close, then next" workflow -- see learning/backtest.py's sequential-only OOF
eval, which is what calibrated the deployed meta_prob_threshold in
config/decision.yaml).

Run under systemd: `systemd-run --user --unit=ai-engine.service ...
python3 -u -m app.engine` (trading/watchdog.py already does this on restart).
"""
import json
import os
import subprocess
import time
from datetime import datetime

import numpy as np
import pandas as pd

from decision.signal import SignalEngine
from decision.router import ModelRouter
from features.features import build_features
from features.labeling import cusum_filter
from config.loader import load_config
from market.feed_listener import FeedListener

_cfg = load_config()
BASE = _cfg.runtime.base_dir
OUTDIR = _cfg.runtime.outdir
SEED = f"{BASE}/data/gold_seed.csv"
BARS_LIVE = f"{OUTDIR}/{_cfg.market.bars_file}"
ACTIVE = f"{OUTDIR}/{_cfg.market.active_signal_file}"
STATE = f"{OUTDIR}/{_cfg.market.tick_state_file}"
JOURNAL = f"{OUTDIR}/trade_journal_ai.jsonl"
OUTCOMES = f"{OUTDIR}/live_outcomes.jsonl"
TG_ENV = _cfg.telegram.env_path
TG_FAIL_LOG = f"{OUTDIR}/.tg_delivery_failures.jsonl"

SYMBOL = _cfg.market.symbol


def live_mae_mfe(buf, entry_time, side, entry_price, risk_unit):
    """Scan buffered M1 bars from entry_time (exclusive) to the newest
    buffered bar for running worst/best excursion, in R units (risk_unit =
    entry-to-SL price distance). Same fav/adv definition as
    research/audit_edge.py's _mae_mfe_core, just python (<=45 bars/trade,
    no numba needed) and bounded by whatever bars have arrived so far --
    this is a REAL M1-bar-derived measurement, not a placeholder, but it
    is still M1-close resolution (no intrabar tick path), same limitation
    as the historical dataset."""
    if risk_unit <= 1e-9:
        return None, None, 0
    window = buf[buf["time"] > entry_time]
    if len(window) == 0:
        return 0.0, 0.0, 0
    worst, best = 0.0, 0.0
    for h, l in zip(window["high"].to_numpy(), window["low"].to_numpy()):
        if side == 1:
            fav, adv = h - entry_price, l - entry_price
        else:
            fav, adv = entry_price - l, entry_price - h
        best = max(best, fav)
        worst = min(worst, adv)
    return -worst / risk_unit, best / risk_unit, len(window)


def vol_state_bucket(ewma_vol_series, current_vol):
    """Causal tercile bucket of current_vol against the trailing in-buffer
    distribution (buffer only ever holds past bars) -- approximate, matches
    Phase 1A's regime-breakdown methodology at reduced (in-buffer) sample
    size rather than a full trailing-252-day baseline."""
    v = ewma_vol_series[np.isfinite(ewma_vol_series) & (ewma_vol_series > 0)]
    if len(v) < 200 or not np.isfinite(current_vol):
        return "unknown"
    lo, hi = np.percentile(v, [33.3, 66.7])
    if current_vol <= lo:
        return "low"
    if current_vol >= hi:
        return "high"
    return "medium"

BUFFER_BARS = 8000        # ~5.5 days of M1 -- plenty for every feature's warmup
LOOP_SLEEP = 5.0          # verdict-poll cadence; ticker updates state every 0.25s
BAR_POLL_EVERY = 1        # check for a new completed bar every loop tick


def ts():
    return datetime.now().strftime("%H:%M:%S")


def log(msg):
    print(f"[{ts()}] {msg}", flush=True)


# ---- Telegram: fail-closed, signal-only chat, never falls back to this session ----
def _tg_once(text):
    try:
        env = {}
        with open(TG_ENV) as f:
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
            ["curl", "-s", "-X", "POST", f"https://api.telegram.org/bot{token}/sendMessage",
             "--data-urlencode", f"chat_id={chat}", "--data-urlencode", f"text={text}",
             "-d", "parse_mode=HTML"], capture_output=True, timeout=15)
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


def tg(text):
    ok, info = _tg_once(text)
    if not ok:
        time.sleep(2)
        ok, info = _tg_once(text)
    if ok:
        log(f"TG ok: {text[:80]}")
        return True
    log(f"TG SEND FAILED (after retry): {info}")
    try:
        with open(TG_FAIL_LOG, "a") as f:
            f.write(json.dumps({"t": time.time(), "reason": info, "text": text}) + "\n")
    except Exception:
        pass
    return False


def load_active():
    try:
        with open(ACTIVE) as f:
            d = json.load(f)
            return d if d else None
    except Exception:
        return None


def save_active(d):
    tmp = ACTIVE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f)
    os.replace(tmp, ACTIVE)


def append_jsonl(path, entry):
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def read_tick_state():
    try:
        with open(STATE) as f:
            return json.load(f)
    except Exception:
        return None


def load_buffer_tail(n=BUFFER_BARS):
    """Bootstrap the in-memory bar buffer from gold_seed.csv's tail."""
    df = pd.read_csv(SEED, usecols=["time", "open", "high", "low", "close",
                                     "tick_volume", "spread"],
                      parse_dates=["time"])
    df = df.sort_values("time").drop_duplicates(subset="time", keep="last").tail(n)
    return df.reset_index(drop=True)


class LiveEngine:
    def __init__(self):
        self.signal_engine = SignalEngine(
            router=ModelRouter(role_map=_cfg.models.model_dump()),
            meta_prob_threshold=_cfg.decision.meta_prob_threshold,
        )
        with open(os.path.join(BASE, "models", "active", "feature_cols.json")) as f:
            cfg = json.load(f)
        self.cusum_k = cfg["cusum_k"]
        # v1 (currently live) predates schema versioning -- fall back to a
        # named placeholder rather than crash/omit the field, so every
        # journal row is still self-describing (Phase 3A logging requirement).
        self.model_version = cfg.get("schema_version", "v1-legacy-unversioned")
        self.feature_schema_version = self.model_version
        primary_path = os.path.join(BASE, "models", "active", "primary.cbm")
        self.model_trained_at = cfg.get("trained_at_utc") or datetime.fromtimestamp(
            os.path.getmtime(primary_path)).isoformat()
        self.buf = load_buffer_tail()
        self.last_bar_t = int(self.buf["time"].iloc[-1].timestamp()) if len(self.buf) else 0
        self.active = load_active()
        log(f"engine start: buffer={len(self.buf):,} bars, last_bar={self.buf['time'].iloc[-1]}, "
            f"active_trade={'yes' if self.active else 'no'}")

        # Phase 2: managed market-state feed listener. Additive and inert --
        # proves MT5 -> market/mt5_feed.py -> this listener -> MarketState
        # reaches the V3 application boundary. Not wired into the signal
        # generation loop above (that stays on gold_seed.csv buffer +
        # build_features(), unchanged) -- get_market_state() is a standalone
        # accessor for later-phase use.
        self.feed_listener = FeedListener(
            symbol="GOLD.i#", host=_cfg.market.feed_host, port=_cfg.market.feed_port,
        )
        self.feed_listener.start()
        log(f"market feed listener started on {_cfg.market.feed_host}:{_cfg.market.feed_port} "
            f"(inert until market/mt5_feed.py connects)")

    def get_market_state(self):
        """Phase 2 accessor: proves MT5 -> managed feed -> MarketState -> V3
        application boundary. Not yet wired into the signal-generation
        loop (that's later-phase work) -- additive and inert."""
        t0 = time.time()
        state = self.feed_listener.get_latest_state()
        if state is not None:
            decision_ready_latency = time.time() - t0
            log(f"market_state accessed: seq={state.sequence} bid={state.bid} "
                f"ask={state.ask} feed_health={state.feed_health.value} "
                f"decision_ready_latency_sec={decision_ready_latency:.6f}")
        return state

    def poll_new_bars(self):
        """Tail xm_live_bars.jsonl for bars newer than self.last_bar_t."""
        new_rows = []
        try:
            with open(BARS_LIVE) as f:
                for line in f:
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    if d["t"] > self.last_bar_t:
                        new_rows.append(d)
        except Exception as e:
            log(f"bar tail read warn: {e}")
            return False
        if not new_rows:
            return False
        new_rows.sort(key=lambda d: d["t"])
        rows = [{"time": pd.Timestamp(d["t"], unit="s"), "open": d["o"], "high": d["h"],
                  "low": d["l"], "close": d["c"], "tick_volume": d["v"], "spread": d["spread"]}
                 for d in new_rows]
        self.buf = pd.concat([self.buf, pd.DataFrame(rows)], ignore_index=True)
        self.buf = self.buf.drop_duplicates(subset="time", keep="last").tail(BUFFER_BARS).reset_index(drop=True)
        self.last_bar_t = new_rows[-1]["t"]
        return True

    def check_for_signal(self):
        """No open trade: build features on the buffer, check the newest bar
        for a CUSUM event, score it. Fires a Telegram signal + opens ACTIVE
        on a qualifying call."""
        feat = build_features(self.buf)
        last = feat.iloc[-1]
        primary_cols = self.signal_engine.primary_cols
        if last[primary_cols].isna().any():
            log("warmup: feature NaNs on latest bar, skipping")
            return

        close = self.buf["close"].to_numpy(dtype=np.float64)
        ev = feat["ewma_vol"].to_numpy(dtype=np.float64)
        ev_filled = np.where(np.isfinite(ev) & (ev > 0), ev, np.nanmedian(ev[np.isfinite(ev)]))
        threshold = np.clip(self.cusum_k * ev_filled * close, 1e-6, None)
        event_mask = cusum_filter(close, threshold)
        if not event_mask[-1]:
            return

        raw_vol = float(last["ewma_vol"])
        sig = self.signal_engine.score(last, float(close[-1]), raw_vol, is_cusum_event=True)
        if sig is None:
            return

        trade_id = time.time()
        direction = "BUY" if sig.side == 1 else "SELL"
        entry_time = self.buf["time"].iloc[-1]
        vstate = vol_state_bucket(feat["ewma_vol"].to_numpy(dtype=np.float64), raw_vol)
        risk_unit = abs(sig.entry - sig.sl)
        save_active({"time": trade_id, "direction": direction, "sl": sig.sl, "tp": sig.tp})
        self.active = {"time": trade_id, "direction": direction, "sl": sig.sl, "tp": sig.tp,
                        "entry": sig.entry, "p_win": sig.p_win, "primary_proba": sig.primary_proba,
                        "opened_at": time.time(), "entry_time": entry_time, "risk_unit": risk_unit,
                        "feat_row": last.drop(labels=["time"]).to_dict()}
        msg = (f"<b>{direction} XAUUSD</b>\n"
               f"Entry: {sig.entry:.2f}\nTP: {sig.tp:.2f}\nSL: {sig.sl:.2f}\n"
               f"P(win): {sig.p_win:.1%} | primary conf: {sig.primary_proba:.1%}")
        tg(msg)
        # Additive logging (Phase 3A): every field below is new; nothing existing removed
        # or renamed, so any code reading the old fields keeps working unchanged.
        append_jsonl(JOURNAL, {
            "t": trade_id, "signal_id": trade_id, "symbol": SYMBOL, "direction": direction,
            "entry": sig.entry, "tp": sig.tp, "sl": sig.sl,
            "raw_primary_proba": sig.primary_proba, "raw_meta_proba": sig.p_win,
            "p_win": sig.p_win, "primary_proba": sig.primary_proba,
            "calibrated_proba": None,  # v1 live path has no calibration wired yet (Phase 2/3A)
            "model_version": self.model_version, "feature_schema_version": self.feature_schema_version,
            "model_trained_at": self.model_trained_at,
            "spread": float(last.get("spread", float("nan"))) if "spread" in last else None,
            "vol_state": vstate, "ewma_vol": raw_vol, "risk_unit_price": risk_unit,
            "entry_time": str(entry_time),
        })
        log(f"SIGNAL {direction} entry={sig.entry:.2f} tp={sig.tp:.2f} sl={sig.sl:.2f} "
            f"p_win={sig.p_win:.3f} vol_state={vstate}")

    def check_for_close(self):
        """Open trade: see if the ticker's first-touch tracker has a verdict
        for THIS trade (matched by trade_id, so a stale verdict from a
        previous trade is never misread as this one's outcome)."""
        st = read_tick_state()
        if st is None:
            return
        if st.get("trade_id") != self.active["time"]:
            return  # ticker hasn't picked up this trade yet
        verdict = st.get("verdict")
        if verdict is None:
            return

        exit_px = st.get("tp_first_px") if verdict == "TP" else st.get("sl_first_px")
        result = 1 if verdict == "TP" else 0
        direction = self.active.get("direction", "?")
        log(f"CLOSE {direction} verdict={verdict} exit={exit_px}")
        # feat_row absent -> this trade wasn't opened by this engine (e.g. a
        # stale entry left by a prior process); still close it out cleanly,
        # just without a training row.
        if "feat_row" in self.active:
            tg(f"<b>{verdict}</b> hit -- {direction} closed @ {exit_px}")

        if "feat_row" in self.active:
            mae_R = mfe_R = None
            entry_time = self.active.get("entry_time")
            risk_unit = self.active.get("risk_unit")
            side = 1 if direction == "BUY" else -1
            if entry_time is not None and risk_unit:
                mae_R, mfe_R, n_bars = live_mae_mfe(self.buf, entry_time, side,
                                                     self.active.get("entry"), risk_unit)
                log(f"  MAE={mae_R} MFE={mfe_R} (scanned {n_bars} bars)" if mae_R is not None else "")
            # Additive: existing 't/direction/entry/tp/sl/verdict/exit_px/label/feat_row'
            # fields unchanged, everything below is new.
            outcome = {"t": self.active.get("time"), "signal_id": self.active.get("time"),
                       "symbol": SYMBOL, "direction": direction,
                       "entry": self.active.get("entry"), "tp": self.active.get("tp"),
                       "sl": self.active.get("sl"), "verdict": verdict, "exit_px": exit_px,
                       "label": result, "outcome": verdict, "resolution_reason": verdict,
                       "exit_time": time.time(), "realized_outcome": result,
                       "mae_R": mae_R, "mfe_R": mfe_R,
                       "max_adverse_excursion_R": mae_R, "max_favorable_excursion_R": mfe_R,
                       "raw_primary_proba": self.active.get("primary_proba"),
                       "raw_meta_proba": self.active.get("p_win"),
                       "calibrated_proba": None,
                       "model_version": self.model_version,
                       "feature_schema_version": self.feature_schema_version,
                       "feat_row": self.active.get("feat_row")}
            append_jsonl(OUTCOMES, outcome)
        append_jsonl(JOURNAL, {"t": time.time(), "closed": self.active.get("time"), "verdict": verdict,
                                "resolution_reason": verdict, "exit_time": time.time(), "exit_px": exit_px})

        save_active({})
        self.active = None

    def run(self):
        log("live engine running")
        try:
            while True:
                try:
                    st = read_tick_state()
                    if st and st.get("market_closed"):
                        time.sleep(30)
                        continue

                    got_new_bar = self.poll_new_bars()

                    if self.active is None:
                        self.active = load_active()  # pick up externally-cleared/opened state on restart
                    if self.active is not None:
                        self.check_for_close()
                    elif got_new_bar and len(self.buf) >= 500:  # only recompute on a fresh bar
                        self.check_for_signal()
                except Exception as e:
                    log(f"loop error: {e}")
                time.sleep(LOOP_SLEEP)
        finally:
            self.feed_listener.stop()


if __name__ == "__main__":
    LiveEngine().run()
