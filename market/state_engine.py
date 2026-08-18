"""Pure incremental MarketState builder -- no I/O, no MT5, no sockets.
on_tick() is O(1)/O(window size), never reloads history. bootstrap() is
the one explicit, startup-only exception that seeds from a bounded
recent backfill (Section 13 of the design spec)."""
from collections import deque
from datetime import datetime, timezone
from statistics import pstdev

from contracts.tick import Tick
from contracts.market_state import MarketState, M1BarState, FeedHealthState, DataQuality

TICK_WINDOW_SEC = 300      # ring buffer retention, matches xm_ticker.py's proven 5-min window
M1_BUFFER_BARS = 480       # ~8 hours of completed M1 bars retained
STALE_AFTER_SEC = 5.0


def is_market_closed(utc_dt):
    """Empirically-derived XM GOLD.i# session hours, ported unchanged from
    market/xm_ticker.py (comment there: "verified from live bars
    2026-08-10"). Not from any MT5 session API -- none is assumed to exist."""
    wd, hr = utc_dt.weekday(), utc_dt.hour  # Mon=0..Sun=6
    if wd == 4 and hr >= 21:
        return True
    if wd == 5:
        return True
    if wd == 6 and hr < 22:
        return True
    if 21 <= hr < 22 and wd < 4:
        return True
    return False


def _bar_start(dt):
    epoch = dt.timestamp()
    bucket = int(epoch // 60) * 60
    return datetime.fromtimestamp(bucket, tz=timezone.utc)


class StateEngine:
    def __init__(self, symbol):
        self.symbol = symbol
        self.completed_m1 = deque(maxlen=M1_BUFFER_BARS)
        self.current_m1 = None       # M1BarState, complete=False
        self._tick_times = deque()   # ring buffer of market_timestamp (epoch floats)
        self._spreads = deque()      # parallel ring buffer of spread samples
        self._last_market_ts = None
        self._last_bid = None
        self._last_ask = None
        self._sequence = 0

    def bootstrap(self, bars):
        """bars: list of dicts (time_iso/open/high/low/close/tick_volume/
        spread) from mt5_feed.py's backfill frame. Startup-only -- never
        called mid-stream. Seeds completed_m1 only; ring buffers start
        empty and fill from live ticks (they're short, 5-min windows,
        filling in naturally within minutes)."""
        for b in bars[-M1_BUFFER_BARS:]:
            start = datetime.fromisoformat(b["time_iso"])
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            self.completed_m1.append(M1BarState(
                open=b["open"], high=b["high"], low=b["low"], close=b["close"],
                tick_count=b.get("tick_volume") or 0, start_time=start,
                end_time=start, complete=True,
            ))

    def on_tick(self, tick: Tick):
        market_ts = tick.market_timestamp.timestamp()

        # out-of-order: reject
        if self._last_market_ts is not None and market_ts < self._last_market_ts:
            return None
        # duplicate: identical market_timestamp + bid + ask
        if (self._last_market_ts == market_ts and self._last_bid == tick.bid
                and self._last_ask == tick.ask):
            return None

        self._last_market_ts = market_ts
        self._last_bid, self._last_ask = tick.bid, tick.ask

        # ring buffers
        self._tick_times.append(market_ts)
        self._spreads.append(tick.spread)
        while self._tick_times and market_ts - self._tick_times[0] > TICK_WINDOW_SEC:
            self._tick_times.popleft()
            self._spreads.popleft()

        # M1 construction
        bar_start = _bar_start(tick.market_timestamp)
        if self.current_m1 is None or self.current_m1.start_time != bar_start:
            if self.current_m1 is not None:
                completed = self.current_m1.model_copy(update={"complete": True, "end_time": tick.market_timestamp})
                self.completed_m1.append(completed)
            self.current_m1 = M1BarState(
                open=tick.bid, high=tick.bid, low=tick.bid, close=tick.bid,
                tick_count=1, start_time=bar_start, complete=False,
            )
        else:
            self.current_m1 = self.current_m1.model_copy(update={
                "high": max(self.current_m1.high, tick.bid),
                "low": min(self.current_m1.low, tick.bid),
                "close": tick.bid,
                "tick_count": self.current_m1.tick_count + 1,
            })

        # activity / volatility state, window-bounded only
        now_s = market_ts
        tick_count_60s = sum(1 for t in self._tick_times if now_s - t <= 60)
        tick_count_300s = len(self._tick_times)
        span = (self._tick_times[-1] - self._tick_times[0]) if len(self._tick_times) > 1 else None
        tick_rate = (len(self._tick_times) / span) if span and span > 0 else 0.0
        spread_window = list(self._spreads)
        spread_mean = sum(spread_window) / len(spread_window) if spread_window else None
        spread_std = pstdev(spread_window) if len(spread_window) > 1 else (0.0 if spread_window else None)

        self._sequence += 1
        now = datetime.now(timezone.utc)
        processing_ts = now

        return MarketState(
            symbol=self.symbol, source=tick.source, sequence=self._sequence,
            market_timestamp=tick.market_timestamp, ingestion_timestamp=tick.ingestion_timestamp,
            processing_timestamp=processing_ts, bid=tick.bid, ask=tick.ask, mid=tick.mid,
            spread=tick.spread, last=tick.last,
            last_quality=DataQuality.UNAVAILABLE if tick.last is None else DataQuality.VALID,
            tick_count_60s=tick_count_60s, tick_count_300s=tick_count_300s,
            tick_rate_per_sec=tick_rate, current_m1=self.current_m1,
            completed_m1=self.completed_m1[-1] if self.completed_m1 else None,
            realized_vol_60s=None, spread_mean_60s=spread_mean, spread_std_60s=spread_std,
            feed_health=FeedHealthState.CONNECTED,
            last_tick_age_sec=(processing_ts - tick.ingestion_timestamp).total_seconds(),
            feed_latency_sec=(tick.ingestion_timestamp - tick.market_timestamp).total_seconds(),
            state_update_latency_sec=(processing_ts - tick.ingestion_timestamp).total_seconds(),
        )
