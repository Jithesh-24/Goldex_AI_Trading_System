"""Pure incremental MarketState builder -- no I/O, no MT5, no sockets.
on_tick() is O(1)/O(window size), never reloads history. bootstrap() is
the one explicit, startup-only exception that seeds from a bounded
recent backfill (Section 13 of the design spec)."""
from collections import deque
from datetime import datetime, timezone

from contracts.tick import Tick
from contracts.market_state import MarketState, M1BarState, FeedHealthState, DataQuality
from contracts.data_quality import is_invalid_price as _is_invalid_price
from contracts.data_quality import is_anomalous_spread as _is_anomalous_spread

TICK_WINDOW_SEC = 300      # ring buffer retention, matches xm_ticker.py's proven 5-min window
M1_BUFFER_BARS = 480       # ~8 hours of completed M1 bars retained
STALE_AFTER_SEC = 5.0
VOL_LOOKBACK_BARS = 60     # matches simulator/market_state_builder.py's realized_vol_60s window


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
        self._tick_times = deque()   # ring buffer of market_timestamp (epoch floats), 300s window
        self._spreads = deque()      # parallel ring buffer of spread samples
        self._tick_times_60s = deque()  # separate 60s-window ring buffer -- kept apart from
                                         # _tick_times so tick_count_60s stays O(1) amortized
                                         # (evict from the front) instead of an O(window) rescan
                                         # of the 300s buffer on every single tick.
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

    def completed_m1_window(self, n: int) -> list:
        """Last n completed M1 bars (fewer during warmup), oldest first.
        Read-only view into the same bounded ring buffer MarketState's
        completed_m1 field already draws its single latest entry from --
        no separate storage, no unbounded growth (Phase 3 spec section 6)."""
        if n <= 0:
            return []
        return list(self.completed_m1)[-n:]

    def on_tick(self, tick: Tick):
        """Three-way return contract:
        - Out-of-order or duplicate tick (identical market_timestamp+bid+ask
          as the last accepted tick): returns None. Nothing is built or
          surfaced -- these are treated as noise, not data.
        - Invalid-price tick (zero/negative/NaN/inf bid/ask/mid, or a
          crossed market where ask < bid) or one with an anomalous spread:
          returns a MarketState, NOT None -- with data_quality=INVALID and
          bid/ask carried forward from the last known-good tick, so the bad
          reading is visible to the consumer instead of silently dropped.
          Ring buffers/last-known state are left untouched by an
          invalid-price tick so it can't poison later windows.
        - Normal tick: returns a MarketState with data_quality=VALID (or
          INVALID if its spread is anomalous relative to recent history).
        """
        market_ts = tick.market_timestamp.timestamp()

        # out-of-order: reject
        if self._last_market_ts is not None and market_ts < self._last_market_ts:
            return None
        # duplicate: identical market_timestamp + bid + ask
        if (self._last_market_ts == market_ts and self._last_bid == tick.bid
                and self._last_ask == tick.ask):
            return None

        # Task 12 -- invalid price (zero/negative/NaN/inf bid/ask/mid, or a
        # crossed market where ask < bid). Tick's own pydantic gt=0 on
        # bid/ask already blocks most of this at construction time; this is
        # defense-in-depth for callers that bypass validation (or corrupt
        # mid, which is unconstrained) and, crucially, is what lets us
        # surface the problem on a returned MarketState instead of the tick
        # simply vanishing. Unlike out-of-order/duplicate above, we do NOT
        # return None here: silently dropping would hide the corruption
        # from the consumer entirely. Instead emit a flagged snapshot built
        # from the last known-good bid/ask, and skip updating ring
        # buffers/last-known state with the bad reading so it can't poison
        # later ticks' spread stats or window calculations.
        invalid_price = (
            _is_invalid_price(tick.bid) or _is_invalid_price(tick.ask)
            or _is_invalid_price(tick.mid) or tick.ask < tick.bid
        )
        if invalid_price:
            self._sequence += 1
            now = datetime.now(timezone.utc)
            fallback_bid = self._last_bid if self._last_bid and self._last_bid > 0 else 0.01
            fallback_ask = self._last_ask if self._last_ask and self._last_ask > 0 else 0.01
            return MarketState(
                symbol=self.symbol, source=tick.source, sequence=self._sequence,
                market_timestamp=tick.market_timestamp, ingestion_timestamp=tick.ingestion_timestamp,
                processing_timestamp=now, bid=fallback_bid, ask=fallback_ask,
                mid=(fallback_bid + fallback_ask) / 2.0, spread=fallback_ask - fallback_bid,
                last=tick.last, last_quality=DataQuality.UNAVAILABLE if tick.last is None else DataQuality.VALID,
                data_quality=DataQuality.INVALID,
                tick_count_60s=len(self._tick_times_60s), tick_count_300s=len(self._tick_times),
                tick_rate_per_sec=0.0, current_m1=self.current_m1,
                completed_m1=self.completed_m1[-1] if self.completed_m1 else None,
                realized_vol_60s=None, spread_mean_60s=None, spread_std_60s=None,
                market_closed=is_market_closed(tick.market_timestamp),
                feed_health=FeedHealthState.CONNECTED,
                last_tick_age_sec=(now - tick.ingestion_timestamp).total_seconds(),
                feed_latency_sec=(tick.ingestion_timestamp - tick.market_timestamp).total_seconds(),
                state_update_latency_sec=(now - tick.ingestion_timestamp).total_seconds(),
            )

        # Snapshot spread history BEFORE this tick's spread is appended
        # below, so the anomaly check judges the new tick against prior
        # ticks only -- mirrors the historical path's _trailing_bar_window,
        # which likewise excludes the row being evaluated.
        prior_spreads = list(self._spreads)

        self._last_market_ts = market_ts
        self._last_bid, self._last_ask = tick.bid, tick.ask

        # ring buffers -- both evicted from the front only, O(1) amortized,
        # never rescanned in full on a tick
        self._tick_times.append(market_ts)
        self._spreads.append(tick.spread)
        while self._tick_times and market_ts - self._tick_times[0] > TICK_WINDOW_SEC:
            self._tick_times.popleft()
            self._spreads.popleft()
        self._tick_times_60s.append(market_ts)
        while self._tick_times_60s and market_ts - self._tick_times_60s[0] > 60:
            self._tick_times_60s.popleft()

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

        # activity / volatility state, window-bounded only (O(window), never
        # O(history) -- spec Section 9 explicitly allows O(window size))
        tick_count_60s = len(self._tick_times_60s)
        tick_count_300s = len(self._tick_times)
        span = (self._tick_times[-1] - self._tick_times[0]) if len(self._tick_times) > 1 else None
        tick_rate = (len(self._tick_times) / span) if span and span > 0 else 0.0
        spread_window = list(self._spreads)
        spread_mean = sum(spread_window) / len(spread_window) if spread_window else None
        if len(spread_window) > 1:
            # Plain float population std -- statistics.pstdev uses exact
            # Fraction/rational arithmetic internally for precision, which
            # profiled at ~94% of on_tick's total runtime on a several-
            # thousand-tick synthetic run. Not needed here: these are
            # already-noisy market spread samples, not a context where
            # rational-exact precision matters, and O(window)-per-tick
            # (spec-sanctioned, Section 9) must still be genuinely fast.
            variance = sum((x - spread_mean) ** 2 for x in spread_window) / len(spread_window)
            spread_std = variance ** 0.5
        else:
            spread_std = 0.0 if spread_window else None

        # realized_vol_60s -- mirrors simulator/market_state_builder.py's
        # convention exactly: std of pct-change returns over a trailing
        # window of up-to-60 completed M1 bar closes (VOL_LOOKBACK_BARS on
        # the historical side). Adapted to the live path's own bar buffer
        # (self.completed_m1, already the source completed_m1 below draws
        # its single latest entry from) instead of a DataFrame window.
        closes = [b.close for b in list(self.completed_m1)[-VOL_LOOKBACK_BARS:]]
        if len(closes) >= 2:
            returns = [(closes[k] - closes[k - 1]) / closes[k - 1] for k in range(1, len(closes))
                       if closes[k - 1] != 0]
            if len(returns) > 1:
                mean = sum(returns) / len(returns)
                variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
                realized_vol_60s = variance ** 0.5
            else:
                realized_vol_60s = None
        else:
            realized_vol_60s = None

        prior_mean = sum(prior_spreads) / len(prior_spreads) if prior_spreads else None
        prior_std = None
        if len(prior_spreads) > 1:
            prior_variance = sum((x - prior_mean) ** 2 for x in prior_spreads) / len(prior_spreads)
            prior_std = prior_variance ** 0.5
        data_quality = (
            DataQuality.INVALID if _is_anomalous_spread(tick.spread, prior_mean, prior_std)
            else DataQuality.VALID
        )

        self._sequence += 1
        now = datetime.now(timezone.utc)
        processing_ts = now

        return MarketState(
            symbol=self.symbol, source=tick.source, sequence=self._sequence,
            market_timestamp=tick.market_timestamp, ingestion_timestamp=tick.ingestion_timestamp,
            processing_timestamp=processing_ts, bid=tick.bid, ask=tick.ask, mid=tick.mid,
            spread=tick.spread, last=tick.last,
            last_quality=DataQuality.UNAVAILABLE if tick.last is None else DataQuality.VALID,
            data_quality=data_quality,
            tick_count_60s=tick_count_60s, tick_count_300s=tick_count_300s,
            tick_rate_per_sec=tick_rate, current_m1=self.current_m1,
            completed_m1=self.completed_m1[-1] if self.completed_m1 else None,
            realized_vol_60s=realized_vol_60s, spread_mean_60s=spread_mean, spread_std_60s=spread_std,
            market_closed=is_market_closed(tick.market_timestamp),
            feed_health=FeedHealthState.CONNECTED,
            last_tick_age_sec=(processing_ts - tick.ingestion_timestamp).total_seconds(),
            feed_latency_sec=(tick.ingestion_timestamp - tick.market_timestamp).total_seconds(),
            state_update_latency_sec=(processing_ts - tick.ingestion_timestamp).total_seconds(),
        )
