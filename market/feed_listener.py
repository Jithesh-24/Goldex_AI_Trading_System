"""Native-side TCP server. mt5_feed.py (Wine) is the client and owns
reconnect; this process is the server so it can persist/restart
independently of the Wine side. Runs its accept-loop on a background
thread so app/engine.py's main loop is never blocked by socket I/O."""
import socket
import threading
import time
from datetime import datetime, timezone

from contracts.tick import Tick
from contracts.market_state import FeedHealthState
from market.state_engine import StateEngine
from market.tick_protocol import decode_frame, FRAME_TICK, FRAME_BACKFILL

STALE_AFTER_SEC = 5.0


class FeedListener:
    def __init__(self, symbol, host="127.0.0.1", port=47115):
        self.symbol = symbol
        self.host, self.port = host, port
        self.engine = StateEngine(symbol)
        self._lock = threading.Lock()
        self._latest_state = None
        self._health = FeedHealthState.UNKNOWN
        self._last_tick_wall = None
        self._server_sock = None
        self._thread = None
        self._stop_flag = threading.Event()

    def get_latest_state(self):
        with self._lock:
            return self._latest_state

    def get_health(self):
        with self._lock:
            if self._last_tick_wall is not None and time.time() - self._last_tick_wall > STALE_AFTER_SEC:
                return FeedHealthState.STALE
            return self._health

    def start(self):
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((self.host, self.port))
        self._server_sock.listen(1)
        self._server_sock.settimeout(1.0)
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_flag.set()
        if self._server_sock is not None:
            self._server_sock.close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _accept_loop(self):
        while not self._stop_flag.is_set():
            try:
                conn, _ = self._server_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with self._lock:
                self._health = FeedHealthState.CONNECTED
            self._serve_connection(conn)
            with self._lock:
                self._health = FeedHealthState.DISCONNECTED

    def _serve_connection(self, conn):
        conn.settimeout(1.0)
        buf = b""
        with conn:
            while not self._stop_flag.is_set():
                try:
                    chunk = conn.recv(4096)
                except socket.timeout:
                    continue
                except OSError:
                    return
                if not chunk:
                    return  # client closed
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    self._handle_line(line.decode("utf-8", "replace"))

    def _handle_line(self, line):
        try:
            frame = decode_frame(line)
        except ValueError:
            return  # malformed frame: rejected, never crashes
        if frame["type"] == FRAME_BACKFILL:
            self.engine.bootstrap(frame["bars"])
            return
        if frame["type"] == FRAME_TICK:
            # Stamped here, unconditionally -- this IS "when feed_listener.py
            # received it." Never taken from the wire (see tick_protocol.py's
            # encode_tick_frame docstring for why).
            ingestion_timestamp = datetime.now(timezone.utc)
            try:
                tick = Tick(
                    symbol=frame["symbol"],
                    market_timestamp=frame["market_timestamp"],
                    ingestion_timestamp=ingestion_timestamp,
                    bid=frame["bid"], ask=frame["ask"],
                    mid=(frame["bid"] + frame["ask"]) / 2,
                    spread=frame["ask"] - frame["bid"],
                    tick_volume=frame.get("tick_volume"),
                    source=frame["source"], internal_seq=frame["internal_seq"],
                )
            except Exception:
                return  # invalid tick payload: rejected, never crashes the listener
            state = self.engine.on_tick(tick)
            if state is None:
                return  # duplicate/out-of-order: engine already rejected it
            with self._lock:
                self._latest_state = state
                self._last_tick_wall = time.time()
