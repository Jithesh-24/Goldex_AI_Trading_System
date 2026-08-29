#!/usr/bin/env python3
"""Phase 3A -- shadow paper-trading process for the v2 (26-feature) model +
rolling Platt calibration. Runs fully independently of app/engine.py:
own process, own buffer (read-only from the same market-data files), own
journal file, own hypothetical position state. Never touches ACTIVE,
Telegram, or the real journal/outcomes files -- a crash here cannot affect
the trader-facing engine, and stopping this process changes nothing about
production (systemctl --user stop gold-shadow.service).

What it does each cycle:
  1. poll the same bar sources app/engine.py reads (read-only)
  2. on a fresh bar with a CUSUM event, score it with the v2 model
     (models/candidates/v2/primary.cbm + meta.cbm via decision.signal.SignalEngine)
  3. apply rolling causal Platt calibration to the raw meta probability
  4. log EVERY such opportunity (not just qualifying ones) to
     shadow_journal.jsonl, with raw p, calibrated p, the real production
     engine's raw p if it fired on the same bar, and the hypothetical
     threshold decision
  5. if the hypothetical decision qualifies and no shadow position is
     already open, open ONE hypothetical position (v2's own vol-scaled
     TP/SL, same one-at-a-time discipline as the real engine) and track it
     bar-by-bar (M1 resolution, same limitation as the historical dataset)
     until TP/SL/timeout, then log the resolution with MAE/MFE
  6. periodically (once per calendar day, i.e. "nightly") refit the rolling
     Platt calibrator against a resolved-outcomes window seeded from the
     offline v2 OOF stream (research/output/v2_oof_outcomes.csv) plus every
     shadow-resolved trade accumulated since -- a live shadow process would
     otherwise take ~a year to reach RollingCalibrationConfig.min_samples
     on its own trades alone.

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 -u -m app.shadow
"""
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from decision.signal import SignalEngine
from decision.router import ModelRouter
from features.features import build_features
from features.labeling import cusum_filter
from decision.calibration import PlattCalibrator, RollingCalibrationConfig, fit_rolling
from config.loader import load_config
from app.engine import (BASE, SEED, BARS_LIVE, live_mae_mfe, vol_state_bucket, SYMBOL,
                         log as _base_log)

_cfg = load_config()
OUTDIR = _cfg.runtime.outdir
RESEARCH_OUT = os.path.join(BASE, "research", "output")
SHADOW_JOURNAL = f"{OUTDIR}/shadow_journal.jsonl"
SHADOW_RESOLVED = os.path.join(RESEARCH_OUT, "shadow_resolved_outcomes.csv")
JOURNAL_V1 = f"{OUTDIR}/trade_journal_ai.jsonl"  # read-only, for "current production p" cross-ref
MODEL_DIR_V2 = os.path.join(BASE, "models", "candidates", "v2")

BUFFER_BARS = 8000
LOOP_SLEEP = 5.0
CAL_CFG = RollingCalibrationConfig(window_days=180, min_samples=500)


def log(msg):
    _base_log(f"[SHADOW] {msg}")


def load_buffer_tail(n=BUFFER_BARS):
    df = pd.read_csv(SEED, usecols=["time", "open", "high", "low", "close",
                                     "tick_volume", "spread"], parse_dates=["time"])
    df = df.sort_values("time").drop_duplicates(subset="time", keep="last").tail(n)
    return df.reset_index(drop=True)


def append_jsonl(path, entry):
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def last_v1_signal_near(target_time, tolerance_sec=90):
    """Best-effort: did the real production engine ALSO fire within
    `tolerance_sec` of this bar? Read-only tail scan, never blocks/raises --
    shadow logging must survive the real journal being absent or mid-write."""
    try:
        with open(JOURNAL_V1) as f:
            lines = f.readlines()[-50:]
        for line in reversed(lines):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if "raw_meta_proba" not in d or "entry_time" not in d:
                continue
            et = pd.Timestamp(d["entry_time"])
            if abs((et - target_time).total_seconds()) <= tolerance_sec:
                return d.get("raw_meta_proba")
    except Exception:
        pass
    return None


class ShadowEngine:
    def __init__(self):
        v2_router = ModelRouter(role_map={
            "direction": "direction_catboost_v2_20260818",
            "opportunity_meta": "opportunity_meta_catboost_v2_20260818",
        })
        self.v2_engine = SignalEngine(router=v2_router, meta_prob_threshold=_cfg.decision.meta_prob_threshold)
        with open(os.path.join(MODEL_DIR_V2, "feature_cols.json")) as f:
            self.v2_cfg = json.load(f)
        self.cusum_k = self.v2_cfg["cusum_k"]
        # Single source of truth (config/decision.yaml), not a second copy
        # baked into the candidate's own feature_cols.json -- both happen
        # to be 0.6 today, this just removes the duplicate source.
        self.threshold = _cfg.decision.meta_prob_threshold
        self.model_version = self.v2_cfg["schema_version"]

        self.buf = load_buffer_tail()
        self.last_bar_t = int(self.buf["time"].iloc[-1].timestamp()) if len(self.buf) else 0
        self.position = None  # {side, entry, tp, sl, entry_time, risk_unit, raw_p, cal_p, signal_id}
        self.calibrator = self._load_bootstrap_calibrator()
        self.outcomes = self._load_outcomes_seed()
        self.last_refit_day = None
        log(f"start: v2 model_version={self.model_version} buffer={len(self.buf):,} bars "
            f"calibrator(a={self.calibrator.a:.3f}, b={self.calibrator.b:.3f}, n={self.calibrator.n_samples})")

    def _load_bootstrap_calibrator(self):
        boot = os.path.join(MODEL_DIR_V2, "calibration_bootstrap.json")
        glob = os.path.join(MODEL_DIR_V2, "calibration_global_fallback.json")
        if os.path.exists(boot):
            return PlattCalibrator.load(boot)
        if os.path.exists(glob):
            return PlattCalibrator.load(glob)
        log("WARN: no offline calibrator artifact found, starting from identity")
        return PlattCalibrator.identity()

    def _load_outcomes_seed(self):
        seed_path = os.path.join(RESEARCH_OUT, "v2_oof_outcomes.csv")
        cols = ["t0_time", "t1_time", "raw_proba", "label"]
        frames = []
        if os.path.exists(seed_path):
            frames.append(pd.read_csv(seed_path, parse_dates=["t0_time", "t1_time"])[cols])
        if os.path.exists(SHADOW_RESOLVED):
            frames.append(pd.read_csv(SHADOW_RESOLVED, parse_dates=["t0_time", "t1_time"])[cols])
        if not frames:
            log("WARN: no offline OOF outcomes seed found -- rolling calibration will use "
                "the bootstrap fit until enough shadow trades resolve on their own")
            return pd.DataFrame(columns=cols)
        return pd.concat(frames, ignore_index=True).sort_values("t0_time").reset_index(drop=True)

    def maybe_refit_calibrator(self):
        today = datetime.now(timezone.utc).date()
        if today == self.last_refit_day or len(self.outcomes) == 0:
            return
        asof = pd.Timestamp.now(tz="UTC").tz_localize(None)
        cal = fit_rolling(self.outcomes, asof=asof, cfg=CAL_CFG, global_fallback=self.calibrator)
        self.calibrator = cal
        self.last_refit_day = today
        log(f"nightly refit: a={cal.a:.4f} b={cal.b:.4f} n_samples={cal.n_samples}")

    def poll_new_bars(self):
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

    def check_opportunity(self):
        feat = build_features(self.buf)
        last = feat.iloc[-1]
        primary_cols = self.v2_engine.primary_cols
        if last[primary_cols].isna().any():
            return

        close = self.buf["close"].to_numpy(dtype=np.float64)
        ev = feat["ewma_vol"].to_numpy(dtype=np.float64)
        ev_filled = np.where(np.isfinite(ev) & (ev > 0), ev, np.nanmedian(ev[np.isfinite(ev)]))
        threshold = np.clip(self.cusum_k * ev_filled * close, 1e-6, None)
        event_mask = cusum_filter(close, threshold)
        if not event_mask[-1]:
            return

        raw_vol = float(last["ewma_vol"])
        # prob_threshold=0.0 forces score() to always return a Signal (it only ever
        # gates on p_win < threshold) -- shadow logs EVERY opportunity, qualifying or not.
        sig = self.v2_engine.score(last, float(close[-1]), raw_vol, is_cusum_event=True,
                                    prob_threshold=0.0)
        if sig is None:
            return

        cal_p = float(self.calibrator.apply(np.array([sig.p_win]))[0])
        qualifies = cal_p >= self.threshold
        entry_time = self.buf["time"].iloc[-1]
        vstate = vol_state_bucket(ev_filled, raw_vol)
        v1_p = last_v1_signal_near(entry_time)
        signal_id = time.time()

        append_jsonl(SHADOW_JOURNAL, {
            "event": "opportunity", "signal_id": signal_id, "symbol": SYMBOL,
            "timestamp": str(entry_time), "direction": "BUY" if sig.side == 1 else "SELL",
            "raw_proba": sig.p_win, "calibrated_proba": cal_p,
            "current_production_proba": v1_p, "v2_model_version": self.model_version,
            "hypothetical_threshold_decision": bool(qualifies),
            "entry_reference": sig.entry, "spread": float(last.get("spread", float("nan"))),
            "volatility_state": vstate, "feature_schema": self.model_version,
            "signal_validity": True, "position_already_open": self.position is not None,
        })

        if qualifies and self.position is None:
            self.position = {
                "signal_id": signal_id, "side": sig.side, "entry": sig.entry, "tp": sig.tp,
                "sl": sig.sl, "entry_time": entry_time, "risk_unit": abs(sig.entry - sig.sl),
                "raw_proba": sig.p_win, "calibrated_proba": cal_p, "opened_bars": 0,
            }
            log(f"opened hypothetical {'BUY' if sig.side == 1 else 'SELL'} entry={sig.entry:.2f} "
                f"raw_p={sig.p_win:.3f} cal_p={cal_p:.3f}")

    def check_position(self):
        pos = self.position
        window = self.buf[self.buf["time"] > pos["entry_time"]]
        if len(window) == 0:
            return
        pos["opened_bars"] = len(window)
        max_holding = self.v2_cfg["max_holding"]
        touch = None
        exit_px = None
        for _, row in window.iterrows():
            if pos["side"] == 1:
                if row["high"] >= pos["tp"]:
                    touch, exit_px = "TP", pos["tp"]
                    break
                if row["low"] <= pos["sl"]:
                    touch, exit_px = "SL", pos["sl"]
                    break
            else:
                if row["low"] <= pos["tp"]:
                    touch, exit_px = "TP", pos["tp"]
                    break
                if row["high"] >= pos["sl"]:
                    touch, exit_px = "SL", pos["sl"]
                    break
        if touch is None and pos["opened_bars"] < max_holding:
            return  # still open, wait for the next bar
        if touch is None:
            touch, exit_px = "TIMEOUT", float(window["close"].iloc[-1])
        label = 1 if touch == "TP" else 0

        mae_R, mfe_R, n_bars = live_mae_mfe(self.buf, pos["entry_time"], pos["side"],
                                             pos["entry"], pos["risk_unit"])
        exit_time = pos["entry_time"] + pd.Timedelta(minutes=pos["opened_bars"])
        append_jsonl(SHADOW_JOURNAL, {
            "event": "resolution", "signal_id": pos["signal_id"], "symbol": SYMBOL,
            "outcome": touch, "resolution_reason": touch, "label": label,
            "exit_timestamp": str(exit_time), "exit_price": exit_px,
            "realized_outcome": label, "mae_R": mae_R, "mfe_R": mfe_R,
            "max_adverse_excursion_R": mae_R, "max_favorable_excursion_R": mfe_R,
            "excursion_path_bars_scanned": n_bars, "time_to_resolution_bars": pos["opened_bars"],
            "raw_proba": pos["raw_proba"], "calibrated_proba": pos["calibrated_proba"],
        })
        log(f"resolved {touch} label={label} mae={mae_R} mfe={mfe_R} bars={pos['opened_bars']}")

        row = pd.DataFrame([{"t0_time": pos["entry_time"], "t1_time": exit_time,
                              "raw_proba": pos["raw_proba"], "label": label}])
        self.outcomes = pd.concat([self.outcomes, row], ignore_index=True)
        header = not os.path.exists(SHADOW_RESOLVED)
        row.to_csv(SHADOW_RESOLVED, mode="a", header=header, index=False)
        self.position = None

    def run(self):
        log("shadow engine running (paper-trading only, no telegram, no real trades)")
        while True:
            try:
                self.maybe_refit_calibrator()
                got_new_bar = self.poll_new_bars()
                if self.position is not None:
                    if got_new_bar:
                        self.check_position()
                elif got_new_bar and len(self.buf) >= 500:
                    self.check_opportunity()
            except Exception as e:
                log(f"loop error (isolated -- production engine unaffected): {e}")
            time.sleep(LOOP_SLEEP)


if __name__ == "__main__":
    ShadowEngine().run()
