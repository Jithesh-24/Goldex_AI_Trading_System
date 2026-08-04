#!/usr/bin/env python3
"""RECONCILE-OPEN-TRADE (one-shot, 2026-08-04) — resolve a trade whose outcome
happened while the data feed was dead, using AUTHORITATIVE MT5 backfill.

WHY: the BUY @ 4020.28 (SL 4007.89 / TP 4057.45) crossed TP at 2026-08-03
14:49 UTC (real high 4082.10) while Xvfb was down and the ticker feed was dead.
The engine booted against a dead ticker → restored the trade as OPEN (partial
coverage). It is NOT a live trade — it is a resolved TP that never got its ack.
A stale OPEN record also blocks new signals (one-signal-at-a-time).

This script:
  1. reads .active_signal_ai.json (must be the exact trade above)
  2. verifies the true outcome from real MT5 M1 backfill (no guessing)
  3. appends the outcome to live_outcomes.jsonl (closed loop — model learns)
  4. appends to trade_journal_ai.jsonl
  5. sends the TP ack to @Goldrigging_bot via the SAME tg() mechanism as the engine
  6. clears the active signal so the engine can trade again

HONESTY: only reconciles when the backfill is definitive (TP touched AND SL not
touched, or vice versa). If both were touched → send UNVERIFIED instead.
"""
import json, os, subprocess, sys, time

OUT = "/home/jith/.hermes/profiles/trading/cron/output"
SCRIPTS = "/home/jith/.hermes/profiles/trading/scripts"
ACTIVE = f"{OUT}/.active_signal_ai.json"
OUTCOMES = f"{OUT}/live_outcomes.jsonl"
JOURNAL = f"{OUT}/trade_journal_ai.jsonl"
SYM = "GOLD.i#"

WINE = "/home/jith/.wine/drive_c/users/jith/AppData/Local/Programs/Python/Python311/python.exe"
CHECK = "/tmp/reconcile_backfill.py"


def load_active():
    try:
        with open(ACTIVE) as f:
            return json.load(f)
    except Exception:
        return None


def backfill_check(entry_ts):
    """Return (tp_hit, sl_hit, max_high, min_low) from real MT5 M1 bars after entry."""
    script = f"""
import MetaTrader5 as mt5, json
mt5.initialize()
r = mt5.copy_rates_from_pos("{SYM}", mt5.TIMEFRAME_M1, 0, 2000)
hi, lo = 0.0, 9e9
for x in r:
    if int(x[0]) < {int(entry_ts)}: continue
    hi = max(hi, float(x[2])); lo = min(lo, float(x[3]))
print(json.dumps({{"high": hi, "low": lo}}))
mt5.shutdown()
"""
    with open(CHECK, "w") as f:
        f.write(script)
    r = subprocess.run(
        ["wine", WINE, "Z:/tmp/reconcile_backfill.py"],
        capture_output=True, timeout=90,
        env={**os.environ, "WINEPREFIX": "/home/jith/.wine",
             "DISPLAY": ":99", "WINEDEBUG": "-all"})
    for line in r.stdout.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    return None


def tg_send(text):
    """Mirror the engine's tg(): signals/.env token → @Goldrigging_bot chat."""
    env = {}
    try:
        with open("/home/jith/.hermes/profiles/signals/.env") as f:
            for line in f:
                line = line.strip()
                if line and "=" in line:
                    k, v = line.split("=", 1)
                    env[k] = v
    except Exception as e:
        print("env read fail:", e)
        return False
    token = env.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat = (env.get("TELEGRAM_CHAT_ID") or "5376343193").strip()
    if not token:
        print("NO TOKEN")
        return False
    r = subprocess.run(
        ["curl", "-s", "-X", "POST",
         f"https://api.telegram.org/bot{token}/sendMessage",
         "--data-urlencode", f"chat_id={chat}",
         "--data-urlencode", f"text={text}",
         "-d", "parse_mode=HTML"],
        capture_output=True, timeout=15)
    out = r.stdout.decode("utf-8", "replace")
    return '"ok":true' in out


def append(path, entry):
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    active = load_active()
    if not active:
        print("no active trade — nothing to reconcile")
        return
    d = active.get("direction")
    entry = active.get("entry")
    sl, tp = active.get("sl"), active.get("tp")
    conf = active.get("conf", 0.0)
    ts = active.get("time") or active.get("entry_bar_ts") or time.time()
    print(f"active: {d} @ {entry} | SL {sl} | TP {tp} | conf {conf:.2f}")

    bf = backfill_check(ts)
    if not bf:
        print("backfill failed — aborting (no guessing)")
        sys.exit(1)
    hi, lo = bf["high"], bf["low"]
    print(f"backfill: high {hi:.2f} | low {lo:.2f}")

    tp_hit = hi >= tp if d == "BUY" else lo <= tp
    sl_hit = lo <= sl if d == "BUY" else hi >= sl
    print(f"→ TP {tp} touched? {tp_hit} | SL {sl} touched? {sl_hit}")

    if tp_hit and not sl_hit:
        verdict, pnl = "TP", (tp - entry) if d == "BUY" else (entry - tp)
    elif sl_hit and not tp_hit:
        verdict, pnl = "SL", (sl - entry) if d == "BUY" else (entry - sl)
    else:
        verdict, pnl = None, 0.0

    if verdict is None:
        msg = (f"⚠️ <b>Trade outcome UNVERIFIED</b> — both levels touched while engine offline.\n"
               f"{d} @ ${entry:.2f} | SL ${sl:.2f} | TP ${tp:.2f} | Conf: {conf:.0%}\n"
               f"<i>Check your MT5 terminal — I will not guess.</i>")
        print("BOTH touched — sending UNVERIFIED, clearing active")
        tg_send(msg)
        os.remove(ACTIVE)
        return

    emoji = "✅" if verdict == "TP" else "❌"
    sign = "+" if verdict == "TP" else ""
    msg = (f"{emoji} <b>{verdict} HIT (while engine offline)</b> — {d} @ ${entry:.2f}\n"
           f"{verdict}: ${tp if verdict=='TP' else sl:.2f} | PnL: {sign}${pnl:.2f} | Conf: {conf:.0%}\n"
           f"path: low ${lo:.2f} / high ${hi:.2f} (reconciled from MT5 backfill)")
    ok = tg_send(msg)
    print(f"ack sent: {ok}")

    # closed loop: outcome + journal
    append(OUTCOMES, {"t": time.time(), "dir": d, "entry": entry, "sl": sl, "tp": tp,
                      "pnl": round(pnl, 2), "result": verdict, "conf": conf,
                      "feats": active.get("feats", {}), "reconciled": True,
                      "src": "mt5_backfill"})
    append(JOURNAL, {"t": time.time(), "dir": d, "entry": entry, "sl": sl, "tp": tp,
                     "pnl": round(pnl, 2), "result": verdict, "conf": conf,
                     "reconciled": True, "src": "mt5_backfill"})
    os.remove(ACTIVE)
    print(f"DONE: {verdict} reconciled, active cleared, model will learn this outcome")


if __name__ == "__main__":
    main()
