#!/usr/bin/env python3
"""Manual market close of the tracked position (user request 2026-08-10).
Records the outcome for the closed-loop learning pipeline, then clears the
active position so the engine goes flat and evaluates fresh on v8.6."""
import json, os, time, glob

OUTDIR = "/home/jith/.hermes/profiles/trading/cron/output"
ACTIVE = f"{OUTDIR}/.active_signal_ai.json"
JOURNAL = f"{OUTDIR}/trade_journal_ai.jsonl"
OUTCOMES = f"{OUTDIR}/live_outcomes.jsonl"

active = None
if os.path.exists(ACTIVE):
    with open(ACTIVE) as f:
        active = json.load(f)

if not active:
    print("NO ACTIVE POSITION — already flat.")
    raise SystemExit(0)

# Latest bid from the ticker state if available, else use journal tail fallback.
bid = None
for p in [f"{OUTDIR}/xm_tick_state.json", f"{OUTDIR}/.ticker_state.json"]:
    if os.path.exists(p):
        try:
            with open(p) as f:
                st = json.load(f)
            bid = st.get("bid")
            break
        except Exception:
            pass
if bid is None:
    bid = float(input()) if False else None

d = active["direction"]
entry = active["entry"]
if bid is None:
    print("COULD NOT READ TICKER STATE — aborting (no fabricated close price).")
    raise SystemExit(1)

pnl = round(bid - entry if d == "BUY" else entry - bid, 2)
rec = {"t": time.time(), "dir": d, "entry": entry, "sl": active["sl"], "tp": active["tp"],
       "pnl": pnl, "result": "MANUAL", "conf": active["conf"], "src": "user_market_close"}
with open(JOURNAL, "a") as f:
    f.write(json.dumps(rec) + "\n")
try:
    outcome = dict(rec)
    outcome["feats"] = active.get("feats", {})
    with open(OUTCOMES, "a") as f:
        f.write(json.dumps(outcome) + "\n")
except Exception as e:
    print(f"outcome append warn: {e}")

os.replace(ACTIVE, ACTIVE + ".closed")
print(f"CLOSED {d} @ {entry:.2f} at market {bid:.2f} → PnL {pnl:+.2f} (manual close, result=MANUAL)")
print(f"active state moved to {ACTIVE}.closed — engine is now FLAT")
