#!/usr/bin/env python3
"""RECOVER from the 08-04 false "TP HIT" ack (SELL @ 4063.98).

WHAT HAPPENED: xm_ticker.py triggered SELL TP on `bid <= tp` — WRONG side of
the spread. A SELL closes by BUYING at the ASK; the TP limit-buy fills only
when ask <= tp. At 09:01:32 the bid (4055.83) brushed TP (4055.84) while the
ask (4056.12) never got there → engine acked a TP that never filled, journaled
a fake +8.14 win, cleared the active signal, and threw the next signal.

THIS SCRIPT:
  1. Purges the FALSE outcome row (SELL @ 4063.98, result TP) from
     live_outcomes.jsonl — it must not poison tonight's learning loop.
  2. Restores .active_signal_ai.json to the TRUE open trade (SELL @ 4063.98,
     SL 4068.50, TP 4055.84, conf 0.4612) with the ORIGINAL features, so the
     ticker re-tracks it to its real outcome with the corrected fill-side logic.
  3. Sends a CORRECTION ack to @Goldrigging_bot: previous TP ack was wrong,
     position still open, tracking resumed.
"""
import json, os, time

OUT = "/home/jith/.hermes/profiles/trading/cron/output"
ACTIVE = os.path.join(OUT, ".active_signal_ai.json")
OUTCOMES = os.path.join(OUT, "live_outcomes.jsonl")
BASE = "/home/jith/.hermes/profiles/trading/scripts"
ENV_PATH = "/home/jith/.hermes/profiles/trading/.env"

def tg(text=""):
    """Same mechanism as the engine: curl --data-urlencode (never -d raw)."""
    env = {}
    with open(ENV_PATH) as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    token = env.get("TELEGRAM_BOT_TOKEN") or env.get("GOLDRIGGING_BOT_TOKEN") or env.get("BOT_TOKEN")
    if not token:
        print("⚠ no bot token in signals/.env — correction NOT sent")
        return False
    import subprocess
    r = subprocess.run(
        ["curl", "-s", "--data-urlencode", f"chat_id=5376343193",
         "--data-urlencode", "parse_mode=HTML",
         "--data-urlencode",
         "text=⚠️ <b>CORRECTION — previous TP ack was WRONG</b>\n"
         "━━━━━━━━━━━━━━━\n"
         "SELL @ $4063.98 | TP $4055.84\n"
         "❌ The earlier \"TP HIT\" was a bug: the bid touched TP but the ask "
         "never did — <b>the position did NOT fill</b>. It is STILL OPEN.\n"
         "🔧 Fill-side trigger bug fixed. The engine is now re-tracking this "
         "trade to its real SL/TP.\n"
         "⛔ The SELL @ $4055.96 signal sent right after was thrown in error — "
         "if you did NOT take it, ignore it.",
         f"https://api.telegram.org/bot{token}/sendMessage"],
        capture_output=True, timeout=20)
    ok = '"ok":true' in (r.stdout or b"").decode()
    print(f"correction ack sent: {ok}")
    return ok

def main():
    # ── 1) purge the false outcome row ──
    rows = []
    purged = 0
    with open(OUTCOMES) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("entry") == 4063.98 and r.get("dir") == "SELL" and r.get("result") == "TP":
                purged += 1
                false_row = r  # keep for feature reconstruction
                continue
            rows.append(r)
    print(f"purged {purged} false outcome row(s) from live_outcomes.jsonl")
    with open(OUTCOMES, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # ── 2) restore the TRUE active trade ──
    # Original signal from the journal: SELL @ 4063.98, SL 4068.500357287197,
    # TP 4055.8433568830455, conf 0.4611993010515468, P(dir)=0.05.
    # Original signal fired 08:09:17 IST on 08-04.
    signal_epoch = 1785811157.0  # 2026-08-04 08:09:17 IST (verified from journal)
    if "false_row" in locals() and false_row and false_row.get("feats"):
        feats = false_row.get("feats", {})
    elif os.path.exists(ACTIVE):
        # idempotent rerun: reuse the already-restored active file
        with open(ACTIVE) as f:
            feats = json.load(f).get("feats", {})
    else:
        feats = {}
    active = {
        "direction": "SELL",
        "entry": 4063.98,
        "sl": 4068.500357287197,
        "tp": 4055.8433568830455,
        "conf": 0.4611993010515468,
        "time": signal_epoch,
        "entry_bar_ts": float(int(signal_epoch // 60) * 60),
        "p_up": 0.05,
        "feats": feats,
        "hold_note_sent": False,
        "restored": True,  # marker: re-tracking after false ack (08-04)
    }
    with open(ACTIVE, "w") as f:
        json.dump(active, f, indent=2)
    print(f"restored active trade: SELL @ 4063.98 | SL 4068.50 | TP 4055.84 | "
          f"conf {active['conf']:.2f} | feats {len(feats)}")

    # ── 3) correction ack to the bot ──
    tg()

    # summary
    with open(OUTCOMES) as f:
        n = sum(1 for _ in f)
    print(f"live_outcomes.jsonl now: {n} rows (false row removed)")

if __name__ == "__main__":
    main()
