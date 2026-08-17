#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
xm_ticker.py — Persistent XM tick watcher + REAL XM M1 bar builder (Wine, 25ms).

WHY: The engine previously used TradingView's scanner for bar SHAPES. TV can
FREEZE (rate-limit) while XM keeps trading — on 07-31 it served a frozen
4048.21 for hours, the engine wrote the same bar into the seed every minute,
and the model computed features on fake duplicated bars. Fix: this daemon owns
the ONLY MT5 connection (Wine allows one IPC client) and produces the REAL XM
bar stream itself:

  1) At startup: backfill recent M1 bars via mt5.copy_rates_from_pos → xm_bars_backfill.csv
  2) Every 25ms tick: build the current M1 bar from REAL XM bid ticks (MT5 bar
     convention) + real spread; on minute rollover, append the completed bar to
     xm_live_bars.jsonl
  3) Track the full price path of the open trade (first-touch SL/TP, 25ms)
  4) Write xm_tick_state.json (plain file, throttled 0.25s) — engine reads this

The engine seeds from gold_seed.csv which merge_seed.py builds from
gold_m1_history.csv + xm_bars_backfill.csv + xm_live_bars.jsonl — ALL real XM
data. TradingView is not used anywhere in the signal path.

FIRST-TOUCH SEMANTICS: whoever touches first wins — on the FILL side of the
spread (a TP/SL closes the position, so it must be executable there):
  BUY  → close = SELL at BID: SL on bid<=sl, TP on bid>=tp
  SELL → close = BUY  at ASK: SL on ask>=sl, TP on ask<=tp
(08-04 FIX: TP sides were swapped before — BUY used ask, SELL used bid. That
made SELL TP fire when the BID brushed TP while the ASK never reached it —
false "TP HIT" acks exactly one spread early; BUY TP fired one spread late =
"not real time". Verdict records WHICH level was hit first and when — so even
if the engine is down for 10 minutes, reconciliation knows the true outcome.

Runs: wine python.exe xm_ticker.py   (guard with watchdog.py)
"""
import MetaTrader5 as mt5
import json, os, time, datetime

OUT = "/home/jith/.hermes/profiles/trading/cron/output"
ACTIVE = os.path.join(OUT, ".active_signal_ai.json")
STATE = os.path.join(OUT, "xm_tick_state.json")
BARS_LIVE = os.path.join(OUT, "xm_live_bars.jsonl")       # completed M1 bars (true UTC)
BARS_BACKFILL = "/home/jith/.hermes/profiles/trading/scripts/xm_bars_backfill.csv"
SYM = "GOLD.i#"
POLL = 0.025          # 25ms tick polling (measured: 0.09ms/call, 4k/sec possible)
WRITE_EVERY = 0.25    # state file write throttle
STALE_AFTER = 30.0    # engine treats state older than this as stale
BACKFILL_N = 2000     # M1 bars to dump at startup (real XM, from terminal cache)

def ts():
    return datetime.datetime.now().strftime("%H:%M:%S")

def log(msg):
    print(f"[{ts()}] {msg}", flush=True)

def connect():
    try:
        if not mt5.initialize():
            return False
        # 2026-08-10 FIX: explicit symbol subscription. The weekend zombie
        # churn left the Market Watch unsubscribed — symbol_info_tick() then
        # returns None on an OPEN market, so the ticker (and engine) believe
        # the market is closed. symbol_select(True) re-subscribes; without it
        # GOLD.i# (and XAUUSD before) never delivers a tick.
        try:
            if not mt5.symbol_select(SYM, True):
                log(f"WARN symbol_select {SYM} -> {mt5.last_error()}")
        except Exception as e:
            log(f"symbol_select err: {e}")
        info = mt5.symbol_info(SYM)
        if info is None:
            log(f"WARN symbol {SYM} not found")
        return True
    except Exception as e:
        log(f"connect err: {e}")
        return False

OFFSET_FILE = os.path.join(OUT, "xm_server_offset.json")

def read_persisted_offset():
    """Last measured server offset (persisted) — survives weekends/market close."""
    try:
        with open(OFFSET_FILE) as f:
            return float(json.load(f).get("offset_h", 3.0))
    except Exception:
        return 3.0

def utc_offset(prev=None):
    """MT5 server epoch - true epoch (hours). XM summer = +3. Detected live ONLY
    from a FRESH tick (market open). When the market is closed, symbol_info_tick
    returns a stale Friday tick whose time is hours old → live detection is
    garbage. Fall back to the persisted offset from the last live session.
    FIX(2026-08-10): (a) t.time is SERVER epoch = true UTC + offset_h. Comparing
    raw server epoch against time.time() fails on an OPEN market (always >600s
    apart) → offset never detected, stayed 0.0, every fresh tick rejected as
    "in the future". (b) tick-based detection is ALSO poisoned by a corrupt
    persisted base (0.0): true_tick_utc = t.time - 0 still = server time →
    check fails → returns corrupt base forever (chicken-and-egg). Robust fix:
    derive the offset from copy_rates M1 bar timestamps, which are live even
    when symbol_info_tick hiccups (backfill proves bars flow)."""
    base = prev if prev is not None else read_persisted_offset()
    try:
        t = mt5.symbol_info_tick(SYM)
        if t is not None:
            true_tick_utc = float(t.time) - base * 3600.0
            if abs(true_tick_utc - time.time()) < 600:
                off = (t.time - time.time()) / 3600.0
                off = round(off * 2) / 2
                return off
    except Exception:
        pass
    # Tick path failed (None or corrupt base). Fall back to the LIVE bar
    # stream: last M1 bar server time vs true UTC. copy_rates works in the
    # same conditions that make symbol_info_tick return None.
    try:
        rates = mt5.copy_rates_from_pos(SYM, mt5.TIMEFRAME_M1, 0, 3)
        if rates is not None and len(rates):
            off = (float(rates[-1]["time"]) - time.time()) / 3600.0
            off = round(off * 2) / 2
            # Plausibility guard on the MEASUREMENT, not on deviation from the
            # (possibly corrupt) base: broker server offsets are 0.5–6.5h
            # (XM +3 summer). A stale weekend bar gives a huge negative off →
            # rejected. A corrupt base 0.0 with real +3.0 must NOT block the
            # heal (old abs(off-base)<=2 guard did exactly that).
            if 0.5 <= abs(off) <= 6.5:
                return off
            return base
    except Exception:
        pass
    # stale or absent tick (market closed) → use last measured
    return base

def backfill(offset_h):
    """Dump recent REAL XM M1 bars to CSV (server time -> TRUE UTC).
    MUST render with timezone-aware UTC — datetime.fromtimestamp() without tz
    renders LOCAL (IST) time, shifting timestamps +5.5h and corrupting the seed
    (this once wrote 02:27 Aug 1 for the Fri 20:57 UTC bar)."""
    try:
        rates = mt5.copy_rates_from_pos(SYM, mt5.TIMEFRAME_M1, 0, BACKFILL_N)
        if rates is None or not len(rates):
            log(f"backfill: copy_rates returned None ({mt5.last_error()})")
            return
        off = read_persisted_offset()  # ← authoritative; ignore the passed-in value
        with open(BARS_BACKFILL, "w") as f:
            f.write("time,open,high,low,close,tick_volume,spread,real_volume,src\n")
            for r in rates:
                t_utc = int(r["time"]) - int(off * 3600)
                tstr = datetime.datetime.fromtimestamp(t_utc, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{tstr},{r['open']:.2f},{r['high']:.2f},{r['low']:.2f},"
                        f"{r['close']:.2f},{r['tick_volume']},{r['spread']},0,mt5bar\n")
        t_last = int(rates[-1]["time"]) - int(off * 3600)
        if t_last > time.time() + 300:
            log(f"backfill WARN: last bar {datetime.datetime.fromtimestamp(t_last, tz=datetime.timezone.utc)} is in the future (offset {off:+.1f}h?)")
        log(f"backfill: {len(rates)} real XM M1 bars -> {os.path.basename(BARS_BACKFILL)} (offset {off:+.1f}h, last {datetime.datetime.fromtimestamp(t_last, tz=datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC)")
    except Exception as e:
        log(f"backfill err: {e}")

def main():
    log(f"xm_ticker starting — {SYM} @ {int(POLL*1000)}ms polling (XM-native bars)")
    connected = connect()
    if not connected:
        log("initial init failed — retrying in 5s")
        time.sleep(5)
        connected = connect()

    offset_h = utc_offset(None) if connected else read_persisted_offset()
    log(f"MT5 server offset: {offset_h:+.1f}h")
    if connected:
        backfill(offset_h)
        try:
            with open(OFFSET_FILE + ".tmp", "w") as f:
                json.dump({"offset_h": offset_h, "ts": time.time()}, f)
            os.replace(OFFSET_FILE + ".tmp", OFFSET_FILE)
        except Exception:
            pass

    state = {
        "bid": None, "ask": None, "ts": 0.0,
        "trade_id": None,
        "min_bid": None, "max_ask": None,
        "min_bid_ts": 0.0, "max_ask_ts": 0.0,
        "sl_first_ts": None, "tp_first_ts": None,
        "sl_first_px": None, "tp_first_px": None,
        "verdict": None, "verdict_ts": None,
        "tracking_since": None,   # when ticker started tracking current trade
        "coverage_partial": False,  # True if ticker missed trade open (gap in path)
        "status": "starting",
        "cur_bar": None,          # forming M1 bar built from REAL XM ticks
        "last_bar_ts": 0.0,       # UTC minute-start of most recent completed bar
        "offset_h": offset_h,
    }
    last_write = 0.0
    last_active_mtime = 0.0
    trade = None
    last_none_log = 0.0
    last_recon = 0.0
    last_hb = 0.0
    cur = None  # [min_start_utc, open, high, low, close, spread_pts, vol]
    # v7 microstructure ring buffers (timestamped; 5-min windows)
    tbuf, sbuf, upbuf, dnbuf = [], [], [], []
    prev_bid = None

    while True:
        now = time.time()

        # 1) refresh open-trade spec from active signal file (mtime check)
        try:
            m = os.path.getmtime(ACTIVE)
            if m != last_active_mtime:
                last_active_mtime = m
                if os.path.exists(ACTIVE) and os.path.getsize(ACTIVE) > 2:
                    with open(ACTIVE) as f:
                        trade = json.load(f)
                else:
                    trade = None
                tid = (trade or {}).get("time") if trade else None
                if tid != state["trade_id"]:
                    state["trade_id"] = tid
                    state["min_bid"] = state["max_ask"] = None
                    state["min_bid_ts"] = state["max_ask_ts"] = 0.0
                    state["sl_first_ts"] = state["tp_first_ts"] = None
                    state["sl_first_px"] = state["tp_first_px"] = None
                    state["verdict"] = None; state["verdict_ts"] = None
                    state["tracking_since"] = now if trade else None
                    # Gap check: if we only started watching AFTER the trade opened,
                    # the path before tracking_since is unknown → partial coverage.
                    if trade:
                        state["coverage_partial"] = (state["tracking_since"] - float(trade.get("time", now)) > 5.0)
                    else:
                        state["coverage_partial"] = False
                    log(f"trade {'OPEN' if trade else 'NONE'} | {tid if trade else '-'} | "
                        f"{trade.get('direction') if trade else ''} SL {trade.get('sl') if trade else ''} "
                        f"TP {trade.get('tp') if trade else ''} | partial={state['coverage_partial']}")
        except Exception:
            pass

        # 2) 25ms tick poll + M1 bar build + path tracking + first-touch verdict
        if connected:
            t = mt5.symbol_info_tick(SYM)
            if t is not None:
                # FRESH-TICK GUARD (v7.2 2026-08-02, FIXED 08-03): MT5 returns
                # the LAST tick even when the market is CLOSED (Friday's price
                # all weekend). But t.time is SERVER time = true UTC + offset_h
                # (XM = +3.0h). Naively comparing t.time to time.time() (true
                # UTC) rejects EVERY real tick as stale → ticker never sees an
                # open market. Must convert server time → true UTC first.
                # If the true-UTC tick age is >60s, the market is closed —
                # write bid/ask=None so the engine's market-closed guard
                # (read_tick_state → None) blocks phantom signals on dead data.
                true_tick_utc = float(t.time) - offset_h * 3600.0
                if abs(true_tick_utc - time.time()) < 60:
                    state["bid"], state["ask"] = float(t.bid), float(t.ask)
                    state["ts"] = now
                    # FIX(2026-08-10): reset market_closed on a FRESH tick.
                    # The flag was only ever set True (stale tick / wall-clock
                    # guard) and never cleared — after the first weekend the
                    # engine would see market_closed=True forever and never
                    # trade again until a ticker restart. A fresh tick is
                    # definitive proof the market is OPEN.
                    state["market_closed"] = False
                else:
                    state["bid"] = state["ask"] = None
                    state["ts"] = now
                    state["market_closed"] = True
            # ---- MICROSTRUCTURE + bar build + path tracking only on FRESH ticks
            # (market open). When bid/ask is None (weekend/closed), skip all of
            # it — writing stale price math would poison the bar stream.
            # FIX(2026-08-04): MUST also require t is not None — `t` is re-assigned
            # each loop and goes None on a transient disconnect, while state["bid"]
            # can still hold a stale non-None from a prior tick. Without this gate
            # we enter the block and hit t.time → AttributeError kills the ticker.
            if t is not None and state.get("bid") is not None:
                # ---- MICROSTRUCTURE (v7): real order flow from XM ticks ----
                # Rolling ring buffers (timestamped) → imbalance, spread
                # dynamics, intensity. Engine reads a few floats per cycle;
                # the closed loop learns these from LIVE outcomes.
                tbuf.append(now)
                sbuf.append(state["ask"] - state["bid"])
                if prev_bid is not None:
                    if state["bid"] > prev_bid:
                        upbuf.append(now)
                    elif state["bid"] < prev_bid:
                        dnbuf.append(now)
                prev_bid = state["bid"]
                while tbuf and now - tbuf[0] > 300:   # 5-min window
                    tbuf.pop(0)
                while upbuf and now - upbuf[0] > 300:
                    upbuf.pop(0)
                while dnbuf and now - dnbuf[0] > 300:
                    dnbuf.pop(0)
                while sbuf and now - sbuf[0] > 300:
                    sbuf.pop(0)
                win60 = sum(1 for x in tbuf if now - x <= 60)
                win60_up = sum(1 for x in upbuf if now - x <= 60)
                win60_dn = sum(1 for x in dnbuf if now - x <= 60)
                spreads = [sbuf[i] for i in range(len(sbuf)) if now - tbuf[i] <= 60]
                _sm = sum(spreads) / len(spreads) if spreads else state["ask"] - state["bid"]
                _sstd = (sum((x - _sm) ** 2 for x in spreads) / max(len(spreads) - 1, 1)) ** 0.5 if len(spreads) > 1 else 0.0
                state["ms"] = {
                    "ticks_60s": win60,
                    "ticks_300s": len(tbuf),
                    "imb_60s": (win60_up - win60_dn) / (win60_up + win60_dn + 1e-9),
                    "imb_300s": (len(upbuf) - len(dnbuf)) / (len(upbuf) + len(dnbuf) + 1e-9),
                    "spread_now": state["ask"] - state["bid"],
                    "spread_mean_60s": _sm,
                    "spread_std_60s": _sstd,
                }
                # ---- REAL XM M1 BAR BUILD (MT5 convention: OHLC from bid) ----
                t_utc = int(t.time) - int(offset_h * 3600)
                bar_min = float(int(t_utc // 60) * 60)
                spread_pts = int(round((state["ask"] - state["bid"]) * 100))
                if cur is None or cur[0] != bar_min:
                    # rollover: persist the completed bar first
                    if cur is not None:
                        try:
                            with open(BARS_LIVE, "a") as f:
                                f.write(json.dumps({
                                    "t": int(cur[0]), "o": round(cur[1], 2),
                                    "h": round(cur[2], 2), "l": round(cur[3], 2),
                                    "c": round(cur[4], 2), "spread": cur[5],
                                    "v": cur[6]}) + "\n")
                            state["last_bar_ts"] = int(cur[0])
                        except Exception:
                            pass
                    cur = [bar_min, state["bid"], state["bid"], state["bid"], state["bid"], spread_pts, 0]
                else:
                    cur[2] = max(cur[2], state["bid"])
                    cur[3] = min(cur[3], state["bid"])
                    cur[4] = state["bid"]
                    cur[5] = spread_pts
                    cur[6] += 1
                state["cur_bar"] = {"t": int(cur[0]), "o": round(cur[1], 2),
                                    "h": round(cur[2], 2), "l": round(cur[3], 2),
                                    "c": round(cur[4], 2), "spread": cur[5], "v": cur[6]}
                # ---- path tracking ----
                if state["min_bid"] is None or state["bid"] < state["min_bid"]:
                    state["min_bid"], state["min_bid_ts"] = state["bid"], now
                if state["max_ask"] is None or state["ask"] > state["max_ask"]:
                    state["max_ask"], state["max_ask_ts"] = state["ask"], now
                if trade and state["verdict"] is None:
                    d = trade["direction"]; sl = float(trade["sl"]); tp = float(trade["tp"])
                    sl_t, tp_t = state["sl_first_ts"], state["tp_first_ts"]
                    if d == "BUY":
                        if sl_t is None and state["bid"] <= sl:
                            state["sl_first_ts"] = now; state["sl_first_px"] = state["bid"]
                        if tp_t is None and state["bid"] >= tp:
                            state["tp_first_ts"] = now; state["tp_first_px"] = state["bid"]
                    else:  # SELL
                        if sl_t is None and state["ask"] >= sl:
                            state["sl_first_ts"] = now; state["sl_first_px"] = state["ask"]
                        if tp_t is None and state["ask"] <= tp:
                            state["tp_first_ts"] = now; state["tp_first_px"] = state["ask"]
                    sl_t, tp_t = state["sl_first_ts"], state["tp_first_ts"]
                    if sl_t is not None or tp_t is not None:
                        if sl_t is not None and (tp_t is None or sl_t <= tp_t):
                            state["verdict"] = "SL"
                        else:
                            state["verdict"] = "TP"
                        state["verdict_ts"] = min([x for x in (sl_t, tp_t) if x])
                        log(f"VERDICT {state['verdict']} | SL@{state['sl_first_ts'] and round(state['sl_first_ts']-now+now,1)} | "
                            f"first SL ts {state['sl_first_ts']} | first TP ts {state['tp_first_ts']}")
            else:
                if now - last_none_log > 30:
                    log("symbol_info_tick None (market closed / disconnect?)")
                    last_none_log = now
                    # FIX(2026-08-10): do NOT mt5.shutdown() on a transient
                    # None. Repeated shutdown→initialize churn (every 30s on a
                    # flaky tick) wedges the module's IPC: fresh probe
                    # processes get ticks instantly while the churning ticker
                    # gets None forever (verified live). On an open market a
                    # None tick is transient — keep polling; the fresh-tick
                    # guard and the wall-clock guard below still write
                    # market_closed correctly. Real disconnects (terminal
                    # death) are handled by the watchdog's 5-min process
                    # restart, which gets a clean handle.
                    connected = False
        else:
            if now - last_recon > 5:
                last_recon = now
                connected = connect()
                if connected:
                    log("reconnected")
                    # 2026-08-10 FIX: symbol_select() subscribes the symbol,
                    # but the FIRST symbol_info_tick() can still return None
                    # before the terminal delivers a tick. Polling once and
                    # tearing down on None reconnects forever (the 08:10
                    # outage: ticks flowed on a fresh probe, ticker looped
                    # "None → reconnect"). Give the subscription time.
                    for _ in range(20):
                        t0 = mt5.symbol_info_tick(SYM)
                        if t0 is not None:
                            break
                        time.sleep(0.25)
                    offset_h = utc_offset(offset_h)
                    state["offset_h"] = offset_h
                    backfill(offset_h)
                    try:
                        with open(OFFSET_FILE + ".tmp", "w") as f:
                            json.dump({"offset_h": offset_h, "ts": time.time()}, f)
                        os.replace(OFFSET_FILE + ".tmp", OFFSET_FILE)
                    except Exception:
                        pass

        # 3) write state file (throttled; immediate on verdict)
        if now - last_write >= WRITE_EVERY or state["verdict"]:
            state["status"] = "ok" if connected else "disconnected"
            # Weekend/closed-window guard: mark market_closed whenever the
            # UTC wall clock says XM is shut (Fri >=21:00 UTC → Mon <21:00
            # UTC). MT5 can return t=None on a weekend disconnect, which the
            # stale-tick path above never sees — without this the watchdog's
            # market_closed() falls through to False and restart-churns the
            # ticker all weekend. The flag is also how the engine's
            # market-closed guard blocks phantom signals on dead data.
            try:
                g = time.gmtime()
                # XM GOLD.i# real hours (verified from live bars 2026-08-10):
                #   daily break 21:00-22:00 UTC; weekend Fri 21:00 -> Sun 22:00.
                # So closed = Fri>=21:00, Sat all day, Sun<22:00, daily 21-22h.
                if g.tm_wday == 4 and g.tm_hour >= 21:
                    state["market_closed"] = True
                elif g.tm_wday == 5:
                    state["market_closed"] = True
                elif g.tm_wday == 6 and g.tm_hour < 22:
                    state["market_closed"] = True
                elif 21 <= g.tm_hour < 22 and g.tm_wday < 4:
                    # daily break 21:00-22:00 UTC (Mon-Thu; Fri/Sat/Sun
                    # handled above — Fri closes permanently at 21:00)
                    state["market_closed"] = True
            except Exception:
                pass
            try:
                with open(STATE + ".tmp", "w") as f:
                    json.dump(state, f)
                os.replace(STATE + ".tmp", STATE)
                last_write = now
            except Exception:
                pass

        # heartbeat every 60s
        if now - last_hb > 60:
            last_hb = now
            log(f"hb bid {state['bid']} ask {state['ask']} | min {state['min_bid']} max {state['max_ask']} | verdict {state['verdict']} | bar {state['cur_bar'] and state['cur_bar']['t']}")

        time.sleep(POLL)

if __name__ == "__main__":
    main()
