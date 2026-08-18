#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mt5_feed.py — Managed MT5 feed process (Wine, 25ms), V3 Phase 2.

Rewritten from market/xm_ticker.py: this file is xm_ticker.py's exact,
unchanged behavior (same MT5 connect/offset/backfill/reconnect logic,
same STATE/BARS_LIVE/BARS_BACKFILL/OFFSET_FILE file outputs, same
verdict-tracking for the open trade) PLUS one additive capability: on
every fresh tick it also pushes a normalized tick_protocol frame over a
TCP socket to market/feed_listener.py (native side), and sends one
backfill frame right after each successful MT5 backfill call. Nothing
existing is removed -- app/engine.py's real signal-generation,
market-closed gating, and trade-verdict logic still read the same STATE
file this process writes, unchanged, exactly as before. The new socket
path is a parallel, additive channel that market/state_engine.py
consumes to build the canonical MarketState -- proving the new pipeline
end-to-end without touching any live trading behavior (Design spec
Sections 19/20/22-24: this is infrastructure, not a trading change).

============================================================
SCOPE NOTE (read before touching this file further): the verdict-
tracking block below (trade_id/min_bid/max_ask/sl_first_ts/tp_first_ts/
verdict, and the microstructure ring buffers + M1 bar build that feed
the STATE file) is copied verbatim from xm_ticker.py. It is NOT
migrated onto the new socket protocol and is explicitly out of Phase 2
scope (design spec Section 2's confirmed scope boundary) -- market/
state_engine.py independently reconstructs M1 bars and activity state
from the NEW tick stream; it does not read or depend on this block.
============================================================

Run: wine python.exe market/mt5_feed.py   (guard with trading/watchdog.py)
"""
import MetaTrader5 as mt5
import json, os, socket, sys, time, datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from market.tick_protocol import encode_tick_frame, encode_backfill_frame

OUT = "/home/jith/.hermes/profiles/trading/cron/output"
ACTIVE = os.path.join(OUT, ".active_signal_ai.json")
STATE = os.path.join(OUT, "xm_tick_state.json")
BARS_LIVE = os.path.join(OUT, "xm_live_bars.jsonl")       # completed M1 bars (true UTC)
BARS_BACKFILL = "/home/jith/.hermes/profiles/trading/scripts/data/xm_bars_backfill.csv"
SYM = "GOLD.i#"
POLL = 0.025          # 25ms tick polling (measured: 0.09ms/call, 4k/sec possible)
WRITE_EVERY = 0.25    # state file write throttle
STALE_AFTER = 30.0    # engine treats state older than this as stale
BACKFILL_N = 2000     # M1 bars to dump at startup (real XM, from terminal cache)

# ---- Phase 2 additive: managed feed socket target (matches config/market.yaml;
# kept as plain constants, not loaded from config/, since the Wine side has no
# pydantic installed -- noted as a real seam in the Phase 2 completion report,
# not solved here by adding a native-only dependency to the Wine process) ----
FEED_HOST = "127.0.0.1"
FEED_PORT = 47115
FEED_RECONNECT_BACKOFF = [1, 2, 4, 8]  # seconds, capped at 8s, bounded, never busy-loops


def ts():
    return datetime.datetime.now().strftime("%H:%M:%S")


def log(msg):
    print(f"[{ts()}] {msg}", flush=True)


# ============================================================
# PHASE 2 ADDITIVE: socket client to market/feed_listener.py.
# A failure here must never crash or pause the MT5 tick loop -- the
# verdict-tracking half and the MT5 connection itself are independent of
# whether this link happens to be up.
# ============================================================
class FeedSocketClient:
    def __init__(self, host, port):
        self.host, self.port = host, port
        self.sock = None
        self._backoff_idx = 0
        self._next_attempt = 0.0
        self._seq = 0

    def _connect(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2.0)
            s.connect((self.host, self.port))
            self.sock = s
            self._backoff_idx = 0
            log(f"feed socket connected to {self.host}:{self.port}")
            return True
        except OSError as e:
            self.sock = None
            delay = FEED_RECONNECT_BACKOFF[min(self._backoff_idx, len(FEED_RECONNECT_BACKOFF) - 1)]
            self._backoff_idx += 1
            self._next_attempt = time.time() + delay
            log(f"feed socket connect failed ({e}), retry in {delay}s")
            return False

    def ensure_connected(self):
        if self.sock is not None:
            return True
        if time.time() < self._next_attempt:
            return False
        return self._connect()

    def send_line(self, line):
        if self.sock is None:
            return False
        try:
            self.sock.sendall(line.encode("utf-8"))
            return True
        except OSError as e:
            log(f"feed socket send failed ({e}), will reconnect")
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None
            self._next_attempt = time.time() + FEED_RECONNECT_BACKOFF[0]
            self._backoff_idx = 1
            return False

    def send_tick(self, market_timestamp_iso, bid, ask, tick_volume):
        self._seq += 1
        line = encode_tick_frame(SYM, market_timestamp_iso, bid, ask, tick_volume, "mt5_live", self._seq)
        self.send_line(line)

    def send_backfill(self, bars):
        line = encode_backfill_frame(SYM, bars)
        self.send_line(line)


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

def backfill(offset_h, feed_client=None):
    """Dump recent REAL XM M1 bars to CSV (server time -> TRUE UTC).
    MUST render with timezone-aware UTC — datetime.fromtimestamp() without tz
    renders LOCAL (IST) time, shifting timestamps +5.5h and corrupting the seed
    (this once wrote 02:27 Aug 1 for the Fri 20:57 UTC bar).

    PHASE 2 ADDITIVE: also sends one backfill frame over feed_client (if given
    and connected) built from the same rows written to CSV -- additive, the
    CSV write is unchanged."""
    try:
        rates = mt5.copy_rates_from_pos(SYM, mt5.TIMEFRAME_M1, 0, BACKFILL_N)
        if rates is None or not len(rates):
            log(f"backfill: copy_rates returned None ({mt5.last_error()})")
            return
        off = read_persisted_offset()  # ← authoritative; ignore the passed-in value
        bars_for_socket = []
        with open(BARS_BACKFILL, "w") as f:
            f.write("time,open,high,low,close,tick_volume,spread,real_volume,src\n")
            for r in rates:
                t_utc = int(r["time"]) - int(off * 3600)
                tstr = datetime.datetime.fromtimestamp(t_utc, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{tstr},{r['open']:.2f},{r['high']:.2f},{r['low']:.2f},"
                        f"{r['close']:.2f},{r['tick_volume']},{r['spread']},0,mt5bar\n")
                bars_for_socket.append({
                    "time_iso": datetime.datetime.fromtimestamp(t_utc, tz=datetime.timezone.utc).isoformat(),
                    "open": float(r["open"]), "high": float(r["high"]), "low": float(r["low"]),
                    "close": float(r["close"]), "tick_volume": int(r["tick_volume"]), "spread": int(r["spread"]),
                })
        t_last = int(rates[-1]["time"]) - int(off * 3600)
        if t_last > time.time() + 300:
            log(f"backfill WARN: last bar {datetime.datetime.fromtimestamp(t_last, tz=datetime.timezone.utc)} is in the future (offset {off:+.1f}h?)")
        log(f"backfill: {len(rates)} real XM M1 bars -> {os.path.basename(BARS_BACKFILL)} (offset {off:+.1f}h, last {datetime.datetime.fromtimestamp(t_last, tz=datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC)")
        if feed_client is not None:
            feed_client.send_backfill(bars_for_socket)
    except Exception as e:
        log(f"backfill err: {e}")

def main():
    log(f"mt5_feed starting — {SYM} @ {int(POLL*1000)}ms polling (XM-native bars)")
    feed_client = FeedSocketClient(FEED_HOST, FEED_PORT)
    connected = connect()
    if not connected:
        log("initial init failed — retrying in 5s")
        time.sleep(5)
        connected = connect()

    offset_h = utc_offset(None) if connected else read_persisted_offset()
    log(f"MT5 server offset: {offset_h:+.1f}h")
    if connected:
        feed_client.ensure_connected()
        backfill(offset_h, feed_client)
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

        # PHASE 2 ADDITIVE: keep the feed socket link alive independently of
        # everything else below -- a failure here never blocks the MT5 loop.
        feed_client.ensure_connected()

        # 1) refresh open-trade spec from active signal file (mtime check)
        # ============================================================
        # VERDICT TRACKING (trade-management, NOT market-state) --
        # unchanged from market/xm_ticker.py, deliberately not migrated onto
        # the new socket protocol. Out of Phase 2 scope by explicit user
        # confirmation (see design spec Section 2). Still writes STATE
        # (xm_tick_state.json) for this purpose only.
        # ============================================================
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
                    # PHASE 2 ADDITIVE: push this fresh tick over the socket,
                    # additive, does not alter anything above.
                    market_ts_iso = datetime.datetime.fromtimestamp(
                        true_tick_utc, tz=datetime.timezone.utc).isoformat()
                    feed_client.send_tick(market_ts_iso, state["bid"], state["ask"], None)
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
                    backfill(offset_h, feed_client)
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
            log(f"hb bid {state['bid']} ask {state['ask']} | min {state['min_bid']} max {state['max_ask']} | verdict {state['verdict']} | bar {state['cur_bar'] and state['cur_bar']['t']} | feed_socket {'connected' if feed_client.sock else 'disconnected'}")

        time.sleep(POLL)

if __name__ == "__main__":
    main()
