#!/usr/bin/env python3
"""Watchdog for ai_signal_engine.py + xm_ticker.py + Xvfb (v2, 2026-08-04).

v1 bug (2026-08-03→04): only checked PID EXISTENCE. A frozen-but-alive engine
(dead data feed → stale-loop sleep, no journal output for 7h) passed every
check. Root cause was Xvfb dying → MT5 couldn't render → ticker got bid=None →
engine starved silently.

v2 fixes:
  - ENGINE freshness = journal output within the last 120s (systemd-run unit
    contract: `journalctl --user -u ai-engine.service`). Alive but silent → STALE
    → restart via systemd-run (preserves journal contract + PID lock).
  - TICKER freshness = xm_tick_state.json mtime < 60s AND bid is not None.
    Alive but stale/None → restart via Wine Popen.
  - XVFB :99 freshness = Xvfb process alive AND X99 socket exists. Dead → relaunch
    Xvfb + xfwm4 (the display MT5 renders into). Without this, MT5 boots into a
    dead display, never logs in, and the whole feed dies — the 2026-08-04 outage.
  - MT5 terminal freshness = terminal64.exe process present. Missing → relaunch.
  - SILENT when healthy (cron no_agent contract).

Engine restarts go through `systemd-run --user --unit=ai-engine.service`
(WorkingDirectory + DISPLAY set) so the engine runs under the user's systemd
manager (survives gateway restarts), logs to the journal, and its own PID lock
guarantees single-instance. Ticker restarts spawn Wine Python (WINEPREFIX +
DISPLAY=:99) detached — it holds the ONLY MT5 IPC connection.
"""
import os, subprocess, sys, time, json

BASE = "/home/jith/.hermes/profiles/trading/scripts"
OUTDIR = "/home/jith/.hermes/profiles/trading/cron/output"
ENGINE_UNIT = "ai-engine.service"
TICKER_STATE = f"{OUTDIR}/xm_tick_state.json"
X_SOCKET = "/tmp/.X11-unix/X99"
WINE_PY = "/home/jith/.wine/drive_c/users/jith/AppData/Local/Programs/Python/Python311/python.exe"
WINE_ENV = {"WINEPREFIX": "/home/jith/.wine", "DISPLAY": ":99", "WINEDEBUG": "-all",
            "PATH": "/usr/bin:/bin"}
MT5_EXE = "/home/jith/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe"

ENGINE_STALE_S = 120      # journal silence beyond this → engine is stuck
TICKER_STALE_S = 60       # state file older than this → ticker is dead/stale
XVFB_DISPLAY = ":99"


def pids_of(pat):
    out = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True).stdout.strip()
    return [int(p) for p in out.splitlines()] if out else []


def is_alive(pat):
    return bool(pids_of(pat))


def engine_fresh():
    """True if the engine unit logged in the last ENGINE_STALE_S seconds."""
    try:
        r = subprocess.run(
            ["journalctl", "--user", "-u", ENGINE_UNIT, "--no-pager",
             "--since", f"{ENGINE_STALE_S} seconds ago"],
            capture_output=True, text=True, timeout=20)
        return bool(r.stdout.strip())
    except Exception:
        return False  # can't verify → treat as stale, restart is safer


def ticker_fresh():
    """True if ticker state file is fresh AND has a real bid."""
    try:
        age = time.time() - os.path.getmtime(TICKER_STATE)
        if age > TICKER_STALE_S:
            return False
        with open(TICKER_STATE) as f:
            st = json.load(f)
        return st.get("bid") is not None
    except Exception:
        return False


def xvfb_fresh():
    if not is_alive("Xvfb :99"):
        return False
    return os.path.exists(X_SOCKET)


def restart_engine():
    try:
        subprocess.run(
            ["systemd-run", "--user", "--unit=ai-engine.service",
             "--property=WorkingDirectory=" + BASE,
             "--setenv=DISPLAY=:99", "--setenv=HOME=/home/jith",
             "--setenv=PYTHONUNBUFFERED=1",
             "/usr/bin/bash", "-c",
             f"cd {BASE} && exec /home/jith/.hermes/hermes-agent/venv/bin/python3 -u ai_signal_engine.py"],
            capture_output=True, timeout=30)
        return True
    except Exception as e:
        print(f"[watchdog] engine systemd-run failed: {e}")
        return False


def restart_ticker():
    try:
        with open(f"{BASE}/ticker.log", "a") as log:
            log.write(f"\n=== [{time.strftime('%Y-%m-%d %H:%M:%S')}] ticker down — restarting ===\n")
        # CRITICAL (2026-08-04): must launch via `wine` — exec'ing the Windows
        # .exe directly gives "Exec format error" (exit 126) and Popen returns
        # before the failure, so the watchdog would believe the restart worked
        # while the ticker is dead. The ticker holds the ONLY MT5 IPC connection.
        subprocess.Popen(
            ["wine", WINE_PY, f"{BASE}/xm_ticker.py"],
            env={**os.environ, **WINE_ENV},
            stdout=open(f"{BASE}/ticker.log", "a"),
            stderr=subprocess.STDOUT)
        return True
    except Exception as e:
        print(f"[watchdog] ticker restart failed: {e}")
        return False


def restart_xvfb():
    """Relaunch Xvfb :99 + xfwm4 — the display MT5 renders into."""
    try:
        for pid in pids_of("Xvfb :99"):
            os.kill(pid, 9)
        time.sleep(1)
        subprocess.Popen(["Xvfb", XVFB_DISPLAY, "-screen", "0", "1920x1080x24",
                          "-ac", "+extension", "XTEST"],
                         stdout=open(f"{BASE}/xvfb.log", "a"),
                         stderr=subprocess.STDOUT)
        time.sleep(3)
        subprocess.Popen(["xfwm4", "--compositor=off", "--vblank=off"],
                         env={**os.environ, "DISPLAY": XVFB_DISPLAY},
                         stdout=open(f"{BASE}/xfwm4.log", "a"),
                         stderr=subprocess.STDOUT)
        time.sleep(2)
        return is_alive("Xvfb :99") and os.path.exists(X_SOCKET)
    except Exception as e:
        print(f"[watchdog] xvfb restart failed: {e}")
        return False


def restart_mt5():
    try:
        subprocess.Popen(
            ["wine", MT5_EXE, "/portable"],
            env={**os.environ, "WINEPREFIX": "/home/jith/.wine",
                 "DISPLAY": ":99", "WINEDEBUG": "-all"},
            cwd=os.path.dirname(MT5_EXE),
            stdout=open(f"{BASE}/mt5.log", "a"),
            stderr=subprocess.STDOUT)
        return True
    except Exception as e:
        print(f"[watchdog] mt5 restart failed: {e}")
        return False


def main():
    msgs = []

    # 1. Xvfb — the display layer. If it's dead, fix FIRST (everything below
    #    depends on it) before touching engine/ticker.
    if not xvfb_fresh():
        if restart_xvfb():
            msgs.append("Xvfb :99 restarted (display was dead)")
        else:
            msgs.append("Xvfb :99 restart FAILED — display layer down")
        # MT5 needs the display to login; if terminal missing, relaunch it
        if not is_alive("terminal64.exe"):
            if restart_mt5():
                msgs.append("MT5 terminal relaunched (needs display to boot)")
            else:
                msgs.append("MT5 relaunch FAILED")

    # 2. Engine — FRESHNESS (journal) not just existence
    if not is_alive("ai_signal_engine.py"):
        if restart_engine():
            msgs.append(f"engine restarted (was dead | unit {ENGINE_UNIT})")
        else:
            msgs.append("engine restart FAILED, needs attention")
    elif not engine_fresh():
        if restart_engine():
            msgs.append(f"engine RESTARTED (was FROZEN — journal silent >{ENGINE_STALE_S}s)")
        else:
            msgs.append("engine stale-restart FAILED, needs attention")

    # 3. Ticker — FRESHNESS (state file age + real bid)
    if not is_alive("xm_ticker.py"):
        if restart_ticker():
            msgs.append("ticker restarted (was dead)")
        else:
            msgs.append("ticker restart FAILED, needs attention")
    elif not ticker_fresh():
        if restart_ticker():
            msgs.append("ticker RESTARTED (state stale / bid None)")
        else:
            msgs.append("ticker stale-restart FAILED, needs attention")

    if msgs:
        print(f"[watchdog {time.strftime('%H:%M:%S')}] ⚠️ " + " | ".join(msgs))
    # healthy → print nothing (cron no_agent stays silent)


if __name__ == "__main__":
    main()
