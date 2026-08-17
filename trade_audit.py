#!/usr/bin/env python3
"""TRADE AUDIT — post-trade root-cause self-analysis (v5 2026-08-01).

The user's demand: "if anything goes wrong a self analysis should be done —
why did I lose this trade, find the root cause, learn so it never repeats."

On every SL hit, this module analyzes the trade's stored feature vector and
classifies WHY it lost, from evidence (not vibes):
  1. DIRECTION_WRONG   — the market moved the other way (model picked the
                         wrong side; trend features contradict the entry)
  2. STOP_TIGHT        — direction was right but price whipsawed through the
                         stop then continued toward TP (stop smaller than the
                         regime's normal noise)
  3. NEWS_SPIKE        — a volatility/volume/spread spike hit (NFP/FOMC-style
                         event) — news candles present at entry
  4. RANGE_ENTRY       — model fired a breakout in a ranging/compressed market
  5. TREND_FADE        — model bought/sold a mature trend that then reversed
  6. SPREAD_BLOWOUT    — spread widened at entry, stop effectively tighter

Each audit writes:
  - cron/output/trade_audits.jsonl  (every audited trade)
  - cron/output/lessons.md          (accumulating, deduplicated lessons —
    the "never repeats mistake" memory that retrain can read)

Root-cause logic is deterministic on the stored features (the EXACT vector the
model saw at signal time) + the outcome. It is NOT a hardcoded gate on future
signals — it only explains the past and feeds learning.

Usage:  from trade_audit import audit_trade; audit_trade(trade_dict)
"""
import json
import os
from datetime import datetime, timezone

BASE = "/home/jith/.hermes/profiles/trading/scripts"
OUTDIR = "/home/jith/.hermes/profiles/trading/cron/output"
AUDITS = f"{OUTDIR}/trade_audits.jsonl"
LESSONS = f"{OUTDIR}/lessons.md"

LESSON_LIBRARY = {  # reason key -> human lesson
    "DIRECTION_WRONG": "Model entered against the active trend. Verify trend_ema/trend_slope/above_ema50 alignment before firing — the model must learn to respect regime direction.",
    "STOP_TIGHT": "Stop was tighter than the regime's noise. SL distance should scale with atr_pctile — quiet-market stops get whipped by noise.",
    "NEWS_SPIKE": "Entered into a news/volatility spike. Big candles + volume bursts + spread widening at entry are event signatures — the model must avoid or widen for them.",
    "RANGE_ENTRY": "Breakout fired in a compressed/ranging market. bb_pctile low = range — fades beat breakouts there. Model must learn range vs trend regime.",
    "TREND_FADE": "Counter-trend entry in a mature trend. High atr_pctile + strong trend_ema at entry means trend continuation is likely — don't fade without confirmation.",
    "SPREAD_BLOWOUT": "Spread widened at entry, silently tightening the real stop. Spread is a cost the model must price in — avoid firing into spread spikes.",
    "EXCURSION_STOP": "v8: SL was INSIDE the regime's learned adverse-excursion band — 6yr MFE/MFA data says losers here routinely dip past this stop before resolving. Placement prior says SL must sit beyond the regime's p90 loser-MFA.",
    "UNKNOWN": "No clear single cause — review the feature vector manually.",
}

_PLACEMENT = None  # lazy-loaded placement_prior.json (learned excursion bands)


def _placement():
    global _PLACEMENT
    if _PLACEMENT is None:
        try:
            p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "placement_prior.json")
            with open(p) as f:
                _PLACEMENT = json.load(f)
        except Exception:
            _PLACEMENT = {}
    return _PLACEMENT


def _ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def classify(trade):
    """Deterministic root-cause classification from stored features."""
    f = trade.get("feats", {}) or {}
    if not f:
        return "UNKNOWN"
    te = float(f.get("trend_ema", 0.0))
    ts = float(f.get("trend_slope", 0.0))
    bb = float(f.get("bb_pctile", 0.5))
    ap = float(f.get("atr_pctile", 0.5))
    vs = float(f.get("vol_spike", 0.0))
    nc = float(f.get("news_candle", 0.0))
    sz = float(f.get("spread_z", 0.0))
    d = trade.get("dir", "BUY")
    entry = float(trade.get("entry", 0.0))
    sl = float(trade.get("sl", 0.0))
    pnl = float(trade.get("pnl", 0.0))

    # 1. News spike at entry (event signature)
    if nc > 0.4 or vs > 2.0:
        return "NEWS_SPIKE"
    # 2. Spread blowout at entry
    if sz > 2.0:
        return "SPREAD_BLOWOUT"
    # 3. v8 EXCURSION CHECK: SL inside the learned adverse band of this regime
    #    (MFE/MFA placement prior from 6yr data — the institutional diagnosis:
    #     did the stop sit where losers routinely dip before resolving?)
    try:
        p = _placement()
        regime = trade.get("regime", "")
        sl_atr = trade.get("sl_atr", None)
        if p and regime and sl_atr is not None and regime in p.get("regimes", {}):
            band = p["regimes"][regime].get(d, {}).get("mfa_p50")
            learned_sl = p["regimes"][regime].get(d, {}).get("sl_atr")
            if band is not None and learned_sl is not None and sl_atr < band * 1.1:
                return "EXCURSION_STOP"
    except Exception:
        pass
    # 4. Direction wrong vs the trend the model itself saw
    if d == "BUY" and te < -1.0 and ts < 0:
        return "DIRECTION_WRONG"
    if d == "SELL" and te > 1.0 and ts > 0:
        return "DIRECTION_WRONG"
    # 5. Mature trend fade (counter-trend in strong regime)
    if d == "SELL" and te > 1.5 and ap > 0.6:
        return "TREND_FADE"
    if d == "BUY" and te < -1.5 and ap > 0.6:
        return "TREND_FADE"
    # 6. Range entry (compressed market, breakout)
    if bb < 0.3:
        return "RANGE_ENTRY"
    # 7. Stop tight relative to regime
    if ap > 0.5:  # volatile regime, stop should be wider
        return "STOP_TIGHT"
    return "UNKNOWN"


def append_lesson(reason, trade):
    """Deduplicated, accumulating lesson memory (never repeats mistake)."""
    lines = []
    if os.path.exists(LESSONS):
        with open(LESSONS) as f:
            lines = [l for l in f.read().splitlines() if l and not l.startswith("#")]
    line = f"- [{datetime.now(timezone.utc).strftime('%Y-%m-%d')}] {reason}: {trade.get('dir')} @ ${trade.get('entry', 0):.2f} (PnL ${trade.get('pnl', 0):+.2f}) — {LESSON_LIBRARY[reason]}"
    if reason in "\n".join(lines):
        return  # lesson already known — don't duplicate
    lines.append(line)
    with open(LESSONS, "w") as f:
        f.write("# AI Trading Lessons — self-learned from losses\n\n")
        f.write("\n".join(lines) + "\n")


def audit_trade(trade):
    """Audit one closed losing trade: classify + persist + learn."""
    reason = classify(trade)
    rec = {
        "t": _ts(),
        "dir": trade.get("dir"),
        "entry": trade.get("entry"),
        "sl": trade.get("sl"),
        "tp": trade.get("tp"),
        "pnl": trade.get("pnl"),
        "conf": trade.get("conf"),
        "regime": trade.get("regime"),     # v8: regime at entry (routing)
        "sl_atr": trade.get("sl_atr"),     # v8: SL in ATR units (excursion check)
        "reason": reason,
        "feats": {k: v for k, v in (trade.get("feats") or {}).items() if k in
                  ("trend_ema", "trend_slope", "bb_pctile", "atr_pctile",
                   "vol_spike", "news_candle", "spread_z")},
    }
    with open(AUDITS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    append_lesson(reason, trade)
    print(f"[audit] {reason} — {trade.get('dir')} @ ${trade.get('entry', 0):.2f} (PnL ${trade.get('pnl', 0):+.2f})")
    return reason


def summarize(n=20):
    """Summarize recent audits (for reports)."""
    recs = []
    if os.path.exists(AUDITS):
        with open(AUDITS) as f:
            for line in f:
                try:
                    recs.append(json.loads(line))
                except Exception:
                    pass
    recent = recs[-n:]
    counts = {}
    for r in recent:
        counts[r["reason"]] = counts.get(r["reason"], 0) + 1
    return counts


if __name__ == "__main__":
    import sys
    print("Recent audit summary:", summarize())
    print(f"Lessons file: {LESSONS}")
