# GOLDEX V4 Phase 1 — Chronological Market/Account Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a chronological, leak-proof, account-aware market simulator (`simulator/`) that replays `data/gold_seed_merged_full6yr.csv` bar-by-bar, exposes a `decide()`/`manage()` policy interface with no fixed horizon and no mandatory SL/TP, and records full per-bar trade experience — with zero decision policy, zero production changes, zero Phase 2 work.

**Architecture:** New module tree `simulator/` (contracts, closure detection, MarketState snapshot builder, execution/fill model, position lifecycle engine, experience recorder, replay orchestrator), reusing `contracts.market_state.MarketState`, `decision.ev_cost.round_trip_cost_r`, and `features.labeling`'s same-bar ambiguity convention as a low-level fill tie-break only. Never imported by any live/production path.

**Tech Stack:** Python, pandas/numpy, pytest, existing repo contracts (Pydantic-based `contracts.market_state`).

**Spec:** `docs/superpowers/specs/2026-08-26-goldex-v4-phase1-simulator-design.md` (Section 13 is the corrected/final architecture — Sections 0-12 are superseded context; executors must read Section 13 before starting).

## Global Constraints

- No decision policy, learning algorithm, reward-function choice, or Phase 2 research-loop code — anywhere in this plan. Any task that starts to design "what makes a good trading decision" is out of scope; write a trivial test-only stub policy where a policy callable is needed for integration testing (labeled explicitly as a test fixture, not a Phase 2 policy).
- No mandatory SL/TP on any position — both optional per Section 13.C.
- No fixed holding-period/horizon anywhere in engine logic.
- No inter-trade cooldown — a position may reopen the bar immediately after the previous one closes.
- Outcome enum is exactly: `POLICY_EXIT | SL_HIT | TP_HIT | LIQUIDATION | END_OF_REPLAY_FORCED_CLOSE`.
- Reuse `decision.ev_cost.round_trip_cost_r` unmodified; reuse the real historical `spread` column (points, `* 0.01` to price units, matching `learning/backtest.py:60-61`'s existing conversion) — never the `REPRESENTATIVE_SPREAD` placeholder from `research/phase5_ev_dataset.py`.
- `simulator/` is never imported by `app/`, `decision/` (production paths), or any live-trading code.
- No changes to `config/models.yaml`, `models/registry/`, or any file under `app/`.
- Do not touch or resume Phase 5 Batch 2 (`research/phase5b_diagnostics/`).

---

## File Structure

- `simulator/contracts.py` — `EnvironmentTag`, `PositionOutcome`, `Side` enums; `SimulatedExecutionConfig`, `AccountState`, `Position`, `PositionView` dataclasses.
- `simulator/closure.py` — weekend/closure gap classification, reusing `learning/data.py`'s existing heuristic.
- `simulator/market_state_builder.py` — builds a `contracts.market_state.MarketState` snapshot at row `i` using only rows `[0..i-1]` plus row `i`'s own open price.
- `simulator/execution.py` — entry/exit fill price (spread-crossing + slippage), cost via `round_trip_cost_r`, same-bar SL/TP ambiguity tie-break.
- `simulator/engine.py` — position open/close/liquidation-check logic and account-state updates.
- `simulator/experience.py` — per-bar experience record builder (dataclass + serialization), environment-tag stamping.
- `simulator/replay.py` — orchestrator: clock + snapshot builder + closure detector + engine + experience recorder, wired together over a `(policy_decide, policy_manage)` pair.
- `tests/simulator/test_contracts.py`, `test_closure.py`, `test_market_state_builder.py`, `test_execution.py`, `test_engine.py`, `test_experience.py`, `test_replay.py`, `test_no_leakage.py`.

---

### Task 1: Core contracts

**Files:**
- Create: `simulator/__init__.py` (empty)
- Create: `simulator/contracts.py`
- Test: `tests/simulator/__init__.py` (empty), `tests/simulator/test_contracts.py`

**Interfaces:**
- Produces: `EnvironmentTag` (str Enum: `SIMULATED_TRAINING, SIMULATED_VALIDATION, SIMULATED_OOS_TEST, LIVE_DEMO, LIVE_REAL`), `PositionOutcome` (str Enum: `POLICY_EXIT, SL_HIT, TP_HIT, LIQUIDATION, END_OF_REPLAY_FORCED_CLOSE`), `Side` (str Enum: `LONG, SHORT`), `SimulatedExecutionConfig` dataclass, `AccountState` dataclass with classmethod `AccountState.initial(config: SimulatedExecutionConfig, timestamp: datetime) -> AccountState`, `Position` dataclass with method `unrealized_pnl(current_mid: float) -> float`, `PositionView` dataclass.

- [ ] **Step 1: Write the failing test**

```python
"""tests/simulator/test_contracts.py"""
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simulator.contracts import (
    EnvironmentTag, PositionOutcome, Side, SimulatedExecutionConfig, AccountState, Position, PositionView,
)


def test_environment_tag_values():
    assert set(EnvironmentTag) == {
        EnvironmentTag.SIMULATED_TRAINING, EnvironmentTag.SIMULATED_VALIDATION,
        EnvironmentTag.SIMULATED_OOS_TEST, EnvironmentTag.LIVE_DEMO, EnvironmentTag.LIVE_REAL,
    }


def test_position_outcome_values():
    assert set(PositionOutcome) == {
        PositionOutcome.POLICY_EXIT, PositionOutcome.SL_HIT, PositionOutcome.TP_HIT,
        PositionOutcome.LIQUIDATION, PositionOutcome.END_OF_REPLAY_FORCED_CLOSE,
    }


def test_account_state_initial():
    config = SimulatedExecutionConfig(starting_balance=5000.0)
    ts = datetime(2020, 1, 1, tzinfo=timezone.utc)
    acct = AccountState.initial(config, ts)
    assert acct.balance == 5000.0
    assert acct.equity == 5000.0
    assert acct.margin_used == 0.0
    assert acct.margin_free == 5000.0
    assert acct.open_position_id is None
    assert acct.simulation_timestamp == ts


def test_position_unrealized_pnl_long_and_short():
    ts = datetime(2020, 1, 1, tzinfo=timezone.utc)
    long_pos = Position(position_id="p1", side=Side.LONG, entry_time=ts, entry_price=100.0,
                         size=2.0, sl_price=None, tp_price=None, margin_used=10.0)
    assert long_pos.unrealized_pnl(110.0) == 20.0
    short_pos = Position(position_id="p2", side=Side.SHORT, entry_time=ts, entry_price=100.0,
                          size=2.0, sl_price=None, tp_price=None, margin_used=10.0)
    assert short_pos.unrealized_pnl(90.0) == 20.0


def test_position_view_construction():
    ts = datetime(2020, 1, 1, tzinfo=timezone.utc)
    view = PositionView(position_id="p1", side=Side.LONG, entry_time=ts, entry_price=100.0,
                         size=1.0, sl_price=None, tp_price=None, unrealized_pnl=5.0, bars_held=3)
    assert view.bars_held == 3
    assert view.sl_price is None and view.tp_price is None  # neither is ever mandatory


if __name__ == "__main__":
    test_environment_tag_values()
    test_position_outcome_values()
    test_account_state_initial()
    test_position_unrealized_pnl_long_and_short()
    test_position_view_construction()
    print("tests/simulator/test_contracts.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/simulator/test_contracts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'simulator'`

- [ ] **Step 3: Write `simulator/contracts.py`**

```python
"""simulator/contracts.py
GOLDEX V4 Phase 1 core data contracts. Neither sl_price nor tp_price is ever
mandatory on a Position -- a policy may run with either, both, or neither
(see docs/superpowers/specs/2026-08-26-goldex-v4-phase1-simulator-design.md
Section 13). No fixed holding horizon appears anywhere in this module."""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class EnvironmentTag(str, Enum):
    SIMULATED_TRAINING = "SIMULATED_TRAINING"
    SIMULATED_VALIDATION = "SIMULATED_VALIDATION"
    SIMULATED_OOS_TEST = "SIMULATED_OOS_TEST"
    LIVE_DEMO = "LIVE_DEMO"
    LIVE_REAL = "LIVE_REAL"


class PositionOutcome(str, Enum):
    POLICY_EXIT = "POLICY_EXIT"
    SL_HIT = "SL_HIT"
    TP_HIT = "TP_HIT"
    LIQUIDATION = "LIQUIDATION"
    END_OF_REPLAY_FORCED_CLOSE = "END_OF_REPLAY_FORCED_CLOSE"


class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class SimulatedExecutionConfig:
    starting_balance: float = 10000.0
    leverage: float = 100.0
    slippage_fraction_of_spread: float = 0.5
    latency_ms: float = 0.0
    margin_call_threshold: float = 0.5
    liquidation_threshold: float = 0.2
    risk_fraction_of_equity: float = 0.01
    max_staleness_seconds: float = 5.0


@dataclass
class AccountState:
    balance: float
    equity: float
    margin_used: float
    margin_free: float
    exposure: float
    open_position_id: Optional[str]
    simulation_timestamp: datetime

    @classmethod
    def initial(cls, config: SimulatedExecutionConfig, timestamp: datetime) -> "AccountState":
        return cls(balance=config.starting_balance, equity=config.starting_balance,
                    margin_used=0.0, margin_free=config.starting_balance,
                    exposure=0.0, open_position_id=None, simulation_timestamp=timestamp)


@dataclass
class Position:
    position_id: str
    side: Side
    entry_time: datetime
    entry_price: float
    size: float
    sl_price: Optional[float]
    tp_price: Optional[float]
    margin_used: float

    def unrealized_pnl(self, current_mid: float) -> float:
        direction = 1.0 if self.side == Side.LONG else -1.0
        return direction * (current_mid - self.entry_price) * self.size


@dataclass
class PositionView:
    position_id: str
    side: Side
    entry_time: datetime
    entry_price: float
    size: float
    sl_price: Optional[float]
    tp_price: Optional[float]
    unrealized_pnl: float
    bars_held: int
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/simulator/test_contracts.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add simulator/__init__.py simulator/contracts.py tests/simulator/__init__.py tests/simulator/test_contracts.py
git commit -m "feat: add GOLDEX V4 Phase 1 simulator core contracts"
```

---

### Task 2: Closure detection

**Files:**
- Create: `simulator/closure.py`
- Test: `tests/simulator/test_closure.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `classify_gap(prev_timestamp: datetime, current_timestamp: datetime) -> str` returning one of `"NORMAL"`, `"WEEKEND_CLOSURE"`, `"DATA_GAP"`. `is_weekend_close_start(timestamp: datetime) -> bool`.

**Note for implementer:** reuse the exact weekend-detection convention already established in `learning/data.py` (`dow == 4 and hour >= 20` marks the start of the weekend close, i.e. Friday evening) — read that file's gap-detection section first so this matches the existing convention rather than inventing a new one.

- [ ] **Step 1: Write the failing test**

```python
"""tests/simulator/test_closure.py"""
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simulator.closure import classify_gap, is_weekend_close_start


def test_is_weekend_close_start_friday_evening():
    friday_2100 = datetime(2020, 1, 3, 21, 0, tzinfo=timezone.utc)  # a Friday
    assert is_weekend_close_start(friday_2100) is True


def test_is_weekend_close_start_friday_afternoon_not_close():
    friday_1400 = datetime(2020, 1, 3, 14, 0, tzinfo=timezone.utc)
    assert is_weekend_close_start(friday_1400) is False


def test_classify_gap_normal_one_minute():
    prev = datetime(2020, 1, 6, 10, 0, tzinfo=timezone.utc)
    curr = prev + timedelta(minutes=1)
    assert classify_gap(prev, curr) == "NORMAL"


def test_classify_gap_weekend_closure():
    prev = datetime(2020, 1, 3, 21, 0, tzinfo=timezone.utc)  # Friday 21:00
    curr = datetime(2020, 1, 5, 22, 0, tzinfo=timezone.utc)  # Sunday 22:00
    assert classify_gap(prev, curr) == "WEEKEND_CLOSURE"


def test_classify_gap_data_gap_midweek():
    prev = datetime(2020, 1, 7, 10, 0, tzinfo=timezone.utc)  # Tuesday
    curr = prev + timedelta(hours=3)
    assert classify_gap(prev, curr) == "DATA_GAP"


if __name__ == "__main__":
    test_is_weekend_close_start_friday_evening()
    test_is_weekend_close_start_friday_afternoon_not_close()
    test_classify_gap_normal_one_minute()
    test_classify_gap_weekend_closure()
    test_classify_gap_data_gap_midweek()
    print("tests/simulator/test_closure.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/simulator/test_closure.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'simulator.closure'`

- [ ] **Step 3: Write `simulator/closure.py`**

```python
"""simulator/closure.py
Reuses the weekend-gap convention already established in learning/data.py's
gap-detection code (Friday >=20:00 marks the start of the weekend close) so
the simulator's closure handling matches the rest of the codebase rather than
inventing a second convention."""
from datetime import datetime, timedelta

NORMAL_BAR_SECONDS = 60
DATA_GAP_TOLERANCE_SECONDS = 90


def is_weekend_close_start(timestamp: datetime) -> bool:
    return timestamp.weekday() == 4 and timestamp.hour >= 20


def classify_gap(prev_timestamp: datetime, current_timestamp: datetime) -> str:
    gap_seconds = (current_timestamp - prev_timestamp).total_seconds()
    if gap_seconds <= DATA_GAP_TOLERANCE_SECONDS:
        return "NORMAL"
    if is_weekend_close_start(prev_timestamp) or prev_timestamp.weekday() in (5, 6):
        return "WEEKEND_CLOSURE"
    return "DATA_GAP"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/simulator/test_closure.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add simulator/closure.py tests/simulator/test_closure.py
git commit -m "feat: add simulator market-closure gap classification"
```

---

### Task 3: MarketState snapshot builder (no-future-leakage core)

**Files:**
- Create: `simulator/market_state_builder.py`
- Test: `tests/simulator/test_market_state_builder.py`

**Interfaces:**
- Consumes: `contracts.market_state.MarketState`, `contracts.market_state.M1BarState`, `contracts.market_state.DataQuality`, `contracts.market_state.FeedHealthState` (read `contracts/market_state.py` first — if any field name/optionality below doesn't match the actual class, fix the construction call to match the real contract; the actual contract is authoritative, this plan's code is reference).
- Produces: `build_snapshot(df: pandas.DataFrame, i: int, symbol: str = "XAUUSD", sequence: int = 0) -> MarketState`. `df` must have columns `time` (as `datetime64`), `open`, `high`, `low`, `close`, `tick_volume`, `spread`. **Critical invariant**: the returned `MarketState` must never be a function of `df.iloc[i]["high"]`, `df.iloc[i]["low"]`, or `df.iloc[i]["close"]`, or of any row with index `> i` — only `df.iloc[i]["open"]`, `df.iloc[i]["time"]`, `df.iloc[i]["spread"]`, and rows `< i`.

- [ ] **Step 1: Write the failing test**

```python
"""tests/simulator/test_market_state_builder.py"""
import os
import sys
from datetime import timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

from simulator.market_state_builder import build_snapshot


def _make_df():
    times = pd.date_range("2020-01-06 10:00:00", periods=5, freq="1min")
    return pd.DataFrame({
        "time": times,
        "open": [1500.0, 1501.0, 1502.0, 1503.0, 1504.0],
        "high": [1500.5, 1501.5, 1502.5, 1503.5, 1504.5],
        "low": [1499.5, 1500.5, 1501.5, 1502.5, 1503.5],
        "close": [1501.0, 1502.0, 1503.0, 1504.0, 1505.0],
        "tick_volume": [10, 12, 11, 9, 13],
        "spread": [20.0, 20.0, 21.0, 19.0, 20.0],
    })


def test_snapshot_mid_uses_current_bar_open_only():
    df = _make_df()
    snap = build_snapshot(df, 2)
    assert snap.mid == 1502.0  # row 2's open, not high/low/close
    assert snap.spread == pytest_approx(21.0 * 0.01)


def pytest_approx(x, tol=1e-9):
    return x


def test_snapshot_never_reads_current_row_high_low_close():
    df = _make_df()
    poisoned = df.copy()
    poisoned.loc[2, ["high", "low", "close"]] = [999999.0, -999999.0, 999999.0]
    snap_clean = build_snapshot(df, 2)
    snap_poisoned = build_snapshot(poisoned, 2)
    assert snap_clean.mid == snap_poisoned.mid
    assert snap_clean.current_m1.high == snap_poisoned.current_m1.high
    assert snap_clean.current_m1.low == snap_poisoned.current_m1.low
    assert snap_clean.current_m1.close == snap_poisoned.current_m1.close


def test_snapshot_never_reads_future_rows():
    df = _make_df()
    poisoned = df.copy()
    poisoned.loc[3:, ["open", "high", "low", "close", "spread"]] = -1.0
    snap_clean = build_snapshot(df, 2)
    snap_poisoned = build_snapshot(poisoned, 2)
    assert snap_clean.mid == snap_poisoned.mid
    assert snap_clean.spread == snap_poisoned.spread
    assert snap_clean.completed_m1.close == snap_poisoned.completed_m1.close


def test_snapshot_completed_m1_uses_previous_row():
    df = _make_df()
    snap = build_snapshot(df, 2)
    assert snap.completed_m1.close == 1502.0  # row 1's close
    assert snap.completed_m1.complete is True
    assert snap.current_m1.complete is False


def test_snapshot_first_row_has_no_completed_bar():
    df = _make_df()
    snap = build_snapshot(df, 0)
    assert snap.completed_m1 is None


if __name__ == "__main__":
    test_snapshot_mid_uses_current_bar_open_only()
    test_snapshot_never_reads_current_row_high_low_close()
    test_snapshot_never_reads_future_rows()
    test_snapshot_completed_m1_uses_previous_row()
    test_snapshot_first_row_has_no_completed_bar()
    print("tests/simulator/test_market_state_builder.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/simulator/test_market_state_builder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'simulator.market_state_builder'`

- [ ] **Step 3: Write `simulator/market_state_builder.py`**

First read `contracts/market_state.py` in full to confirm exact field names/optionality/enum members for `MarketState`, `M1BarState`, `DataQuality`, `FeedHealthState`, then implement:

```python
"""simulator/market_state_builder.py
Builds a contracts.market_state.MarketState snapshot as-of row i's bar-OPEN
timestamp (see docs/superpowers/specs/2026-08-26-goldex-v4-phase1-simulator-design.md
Section 4 for the bar-open-vs-close decision). Only rows [0..i-1] are treated
as "completed" (their high/low/close are known); row i itself contributes
ONLY its open price and timestamp -- its high/low/close are not yet known at
decision time and must never be read here. This function is the load-bearing
piece the no-leakage test harness (Task 8) audits.

Source is always "synthetic_replay" -- this is what
contracts.market_state.MarketState's existing source Literal was designed to
distinguish from "mt5_live"."""
from datetime import timezone

import pandas as pd

from contracts.market_state import MarketState, M1BarState, DataQuality, FeedHealthState

SPREAD_POINTS_TO_PRICE = 0.01
VOL_LOOKBACK_BARS = 60


def build_snapshot(df: pd.DataFrame, i: int, symbol: str = "XAUUSD", sequence: int = 0) -> MarketState:
    row = df.iloc[i]
    ts = row["time"].to_pydatetime()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    spread_price = float(row["spread"]) * SPREAD_POINTS_TO_PRICE
    mid = float(row["open"])
    bid = mid - spread_price / 2.0
    ask = mid + spread_price / 2.0

    if i > 0:
        prev = df.iloc[i - 1]
        prev_ts = prev["time"].to_pydatetime()
        if prev_ts.tzinfo is None:
            prev_ts = prev_ts.replace(tzinfo=timezone.utc)
        completed_m1 = M1BarState(
            open=float(prev["open"]), high=float(prev["high"]),
            low=float(prev["low"]), close=float(prev["close"]),
            tick_count=int(prev["tick_volume"]), start_time=prev_ts, end_time=ts, complete=True,
        )
    else:
        completed_m1 = None

    current_m1 = M1BarState(
        open=mid, high=mid, low=mid, close=mid,
        tick_count=0, start_time=ts, end_time=ts, complete=False,
    )

    window_start = max(0, i - VOL_LOOKBACK_BARS)
    window = df.iloc[window_start:i]
    if len(window) >= 2:
        returns = window["close"].pct_change().dropna()
        realized_vol_60s = float(returns.std()) if len(returns) > 0 else None
    else:
        realized_vol_60s = None

    return MarketState(
        symbol=symbol, source="synthetic_replay", state_version="v1", sequence=sequence,
        market_timestamp=ts, ingestion_timestamp=ts, processing_timestamp=ts,
        bid=bid, ask=ask, mid=mid, spread=spread_price, last=mid,
        last_quality=DataQuality.VALID,
        tick_count_60s=0, tick_count_300s=0, tick_rate_per_sec=0.0,
        current_m1=current_m1, completed_m1=completed_m1,
        realized_vol_60s=realized_vol_60s, spread_mean_60s=None, spread_std_60s=None,
        feed_health=FeedHealthState.CONNECTED, last_tick_age_sec=0.0,
        feed_latency_sec=0.0, state_update_latency_sec=0.0,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/simulator/test_market_state_builder.py -v`
Expected: PASS (5 tests). If a `MarketState`/`M1BarState` construction call fails due to a field mismatch, fix the field names/types to match the real contract in `contracts/market_state.py` — do not change the test's leakage assertions to make a wrong implementation pass.

- [ ] **Step 5: Commit**

```bash
git add simulator/market_state_builder.py tests/simulator/test_market_state_builder.py
git commit -m "feat: add leak-safe MarketState snapshot builder for replay"
```

---

### Task 4: Execution/fill model

**Files:**
- Create: `simulator/execution.py`
- Test: `tests/simulator/test_execution.py`

**Interfaces:**
- Consumes: `simulator.contracts.Side`, `simulator.contracts.SimulatedExecutionConfig`, `decision.ev_cost.round_trip_cost_r`.
- Produces: `entry_fill_price(side: Side, mid: float, spread: float, config: SimulatedExecutionConfig) -> float`, `exit_fill_price(side: Side, mid: float, spread: float, config: SimulatedExecutionConfig) -> float`, `compute_cost_r(market_state, sl_distance_r: Optional[float], config: SimulatedExecutionConfig) -> Optional[float]`, `resolve_same_bar_ambiguity(side: Side, bar_high: float, bar_low: float, sl_price: Optional[float], tp_price: Optional[float]) -> Optional[str]` returning `"SL_HIT"`, `"TP_HIT"`, or `None`.

**Note for implementer:** `resolve_same_bar_ambiguity` must reuse the exact same-bar tie-break convention as `features/labeling.py`'s `_triple_barrier_core` (when both SL and TP are touched within one bar, the adverse side wins — charge `"SL_HIT"`) — read that function first (around line 87-92) to confirm the convention before implementing, since this is a deliberate reuse of existing, already-relied-upon logic, not a new invention.

- [ ] **Step 1: Write the failing test**

```python
"""tests/simulator/test_execution.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simulator.contracts import Side, SimulatedExecutionConfig
from simulator.execution import entry_fill_price, exit_fill_price, resolve_same_bar_ambiguity


def test_entry_fill_long_pays_ask_plus_slippage():
    config = SimulatedExecutionConfig(slippage_fraction_of_spread=0.5)
    price = entry_fill_price(Side.LONG, mid=100.0, spread=1.0, config=config)
    assert price == 100.0 + 0.5 + 0.5  # half-spread + slippage(0.5*spread)


def test_entry_fill_short_receives_bid_minus_slippage():
    config = SimulatedExecutionConfig(slippage_fraction_of_spread=0.5)
    price = entry_fill_price(Side.SHORT, mid=100.0, spread=1.0, config=config)
    assert price == 100.0 - 0.5 - 0.5


def test_exit_fill_long_receives_bid_minus_slippage():
    config = SimulatedExecutionConfig(slippage_fraction_of_spread=0.5)
    price = exit_fill_price(Side.LONG, mid=100.0, spread=1.0, config=config)
    assert price == 100.0 - 0.5 - 0.5


def test_same_bar_ambiguity_both_touched_charges_adverse_side():
    result = resolve_same_bar_ambiguity(Side.LONG, bar_high=110.0, bar_low=90.0, sl_price=95.0, tp_price=105.0)
    assert result == "SL_HIT"


def test_same_bar_ambiguity_only_tp_touched():
    result = resolve_same_bar_ambiguity(Side.LONG, bar_high=110.0, bar_low=99.0, sl_price=95.0, tp_price=105.0)
    assert result == "TP_HIT"


def test_same_bar_ambiguity_neither_touched():
    result = resolve_same_bar_ambiguity(Side.LONG, bar_high=101.0, bar_low=99.0, sl_price=95.0, tp_price=105.0)
    assert result is None


def test_same_bar_ambiguity_no_sl_or_tp_set():
    result = resolve_same_bar_ambiguity(Side.LONG, bar_high=110.0, bar_low=90.0, sl_price=None, tp_price=None)
    assert result is None


if __name__ == "__main__":
    test_entry_fill_long_pays_ask_plus_slippage()
    test_entry_fill_short_receives_bid_minus_slippage()
    test_exit_fill_long_receives_bid_minus_slippage()
    test_same_bar_ambiguity_both_touched_charges_adverse_side()
    test_same_bar_ambiguity_only_tp_touched()
    test_same_bar_ambiguity_neither_touched()
    test_same_bar_ambiguity_no_sl_or_tp_set()
    print("tests/simulator/test_execution.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/simulator/test_execution.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'simulator.execution'`

- [ ] **Step 3: Write `simulator/execution.py`**

```python
"""simulator/execution.py
Spread-crossing entry/exit fills plus a configurable slippage fraction of
spread (slippage is unmodeled anywhere else in this codebase -- see
decision/ev_cost.py, which handles spread cost only, not slippage). Cost
still goes through decision.ev_cost.round_trip_cost_r unmodified.
resolve_same_bar_ambiguity reuses features/labeling.py's existing
_triple_barrier_core convention: if both SL and TP are touched within one
bar, the adverse side wins."""
from typing import Optional

from decision.ev_cost import round_trip_cost_r
from simulator.contracts import Side, SimulatedExecutionConfig


def entry_fill_price(side: Side, mid: float, spread: float, config: SimulatedExecutionConfig) -> float:
    half_spread = spread / 2.0
    slippage = spread * config.slippage_fraction_of_spread
    if side == Side.LONG:
        return mid + half_spread + slippage
    return mid - half_spread - slippage


def exit_fill_price(side: Side, mid: float, spread: float, config: SimulatedExecutionConfig) -> float:
    half_spread = spread / 2.0
    slippage = spread * config.slippage_fraction_of_spread
    if side == Side.LONG:
        return mid - half_spread - slippage
    return mid + half_spread + slippage


def compute_cost_r(market_state, sl_distance_r: Optional[float], config: SimulatedExecutionConfig) -> Optional[float]:
    if sl_distance_r is None or sl_distance_r <= 0:
        return None
    return round_trip_cost_r(market_state, sl_distance_r, config.max_staleness_seconds)


def resolve_same_bar_ambiguity(side: Side, bar_high: float, bar_low: float,
                                sl_price: Optional[float], tp_price: Optional[float]) -> Optional[str]:
    sl_touched = sl_price is not None and (
        (side == Side.LONG and bar_low <= sl_price) or (side == Side.SHORT and bar_high >= sl_price)
    )
    tp_touched = tp_price is not None and (
        (side == Side.LONG and bar_high >= tp_price) or (side == Side.SHORT and bar_low <= tp_price)
    )
    if sl_touched:
        return "SL_HIT"
    if tp_touched:
        return "TP_HIT"
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/simulator/test_execution.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add simulator/execution.py tests/simulator/test_execution.py
git commit -m "feat: add simulator execution/fill model"
```

---

### Task 5: Position lifecycle engine (open/close/liquidation, account updates)

**Files:**
- Create: `simulator/engine.py`
- Test: `tests/simulator/test_engine.py`

**Interfaces:**
- Consumes: `simulator.contracts.{Side, Position, PositionView, AccountState, SimulatedExecutionConfig, PositionOutcome}`, `simulator.execution.{entry_fill_price, exit_fill_price, compute_cost_r}`.
- Produces: `to_position_view(position: Position, current_mid: float, bars_held: int) -> PositionView`; `open_position(market_state, account: AccountState, side: Side, sl_price: Optional[float], tp_price: Optional[float], config: SimulatedExecutionConfig) -> tuple[Position, AccountState]`; `close_position(market_state, account: AccountState, position: Position, exit_price: float, config: SimulatedExecutionConfig) -> tuple[float, float, AccountState]` (returns `(net_pnl, cost_amount, new_account)`); `check_liquidation(account: AccountState, config: SimulatedExecutionConfig) -> bool`.

- [ ] **Step 1: Write the failing test**

```python
"""tests/simulator/test_engine.py"""
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simulator.contracts import Side, AccountState, SimulatedExecutionConfig, Position
from simulator.engine import open_position, close_position, check_liquidation, to_position_view


class _FakeMarketState:
    def __init__(self, mid, spread, market_timestamp, realized_vol_60s=0.001):
        self.mid = mid
        self.spread = spread
        self.market_timestamp = market_timestamp
        self.realized_vol_60s = realized_vol_60s


def test_open_position_reduces_margin_free():
    config = SimulatedExecutionConfig(starting_balance=10000.0, leverage=100.0, risk_fraction_of_equity=0.01)
    ts = datetime(2020, 1, 1, tzinfo=timezone.utc)
    account = AccountState.initial(config, ts)
    ms = _FakeMarketState(mid=1500.0, spread=0.2, market_timestamp=ts)
    position, new_account = open_position(ms, account, Side.LONG, sl_price=1495.0, tp_price=None, config=config)
    assert position.side == Side.LONG
    assert position.entry_price > 1500.0  # crossed the spread
    assert new_account.margin_used > 0.0
    assert new_account.margin_free < account.margin_free
    assert new_account.open_position_id == position.position_id


def test_close_position_updates_balance_with_realized_pnl_minus_cost():
    config = SimulatedExecutionConfig(starting_balance=10000.0, leverage=100.0, risk_fraction_of_equity=0.01)
    ts0 = datetime(2020, 1, 1, tzinfo=timezone.utc)
    account = AccountState.initial(config, ts0)
    ms0 = _FakeMarketState(mid=1500.0, spread=0.2, market_timestamp=ts0)
    position, account = open_position(ms0, account, Side.LONG, sl_price=1495.0, tp_price=None, config=config)
    ts1 = datetime(2020, 1, 1, 0, 10, tzinfo=timezone.utc)
    ms1 = _FakeMarketState(mid=1510.0, spread=0.2, market_timestamp=ts1)
    exit_price = 1509.5
    net_pnl, cost_amount, new_account = close_position(ms1, account, position, exit_price, config)
    assert new_account.open_position_id is None
    assert new_account.margin_used == 0.0
    assert new_account.balance == account.balance + net_pnl
    assert cost_amount >= 0.0


def test_check_liquidation_true_when_equity_below_threshold_ratio():
    config = SimulatedExecutionConfig(liquidation_threshold=0.2)
    ts = datetime(2020, 1, 1, tzinfo=timezone.utc)
    account = AccountState(balance=100.0, equity=15.0, margin_used=100.0, margin_free=-85.0,
                            exposure=1000.0, open_position_id="p1", simulation_timestamp=ts)
    assert check_liquidation(account, config) is True


def test_check_liquidation_false_when_flat():
    config = SimulatedExecutionConfig(liquidation_threshold=0.2)
    ts = datetime(2020, 1, 1, tzinfo=timezone.utc)
    account = AccountState.initial(config, ts)
    assert check_liquidation(account, config) is False


def test_to_position_view_tracks_bars_held():
    ts = datetime(2020, 1, 1, tzinfo=timezone.utc)
    position = Position(position_id="p1", side=Side.LONG, entry_time=ts, entry_price=1500.0,
                         size=1.0, sl_price=1495.0, tp_price=None, margin_used=15.0)
    view = to_position_view(position, current_mid=1510.0, bars_held=7)
    assert view.bars_held == 7
    assert view.unrealized_pnl == 10.0


if __name__ == "__main__":
    test_open_position_reduces_margin_free()
    test_close_position_updates_balance_with_realized_pnl_minus_cost()
    test_check_liquidation_true_when_equity_below_threshold_ratio()
    test_check_liquidation_false_when_flat()
    test_to_position_view_tracks_bars_held()
    print("tests/simulator/test_engine.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/simulator/test_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'simulator.engine'`

- [ ] **Step 3: Write `simulator/engine.py`**

```python
"""simulator/engine.py
Position open/close mechanics and account-state bookkeeping. No decision
logic lives here -- side/sl_price/tp_price/exit timing are all supplied by
the caller (a policy in Phase 2+, or a test stub here in Phase 1)."""
import uuid
from typing import Optional

from simulator.contracts import AccountState, Position, PositionView, Side, SimulatedExecutionConfig
from simulator.execution import entry_fill_price, compute_cost_r


def to_position_view(position: Position, current_mid: float, bars_held: int) -> PositionView:
    return PositionView(
        position_id=position.position_id, side=position.side, entry_time=position.entry_time,
        entry_price=position.entry_price, size=position.size, sl_price=position.sl_price,
        tp_price=position.tp_price, unrealized_pnl=position.unrealized_pnl(current_mid), bars_held=bars_held,
    )


def open_position(market_state, account: AccountState, side: Side, sl_price: Optional[float],
                   tp_price: Optional[float], config: SimulatedExecutionConfig):
    entry_price = entry_fill_price(side, market_state.mid, market_state.spread, config)
    size = (account.equity * config.risk_fraction_of_equity)
    margin_used = (size * entry_price) / config.leverage
    position = Position(position_id=str(uuid.uuid4()), side=side, entry_time=market_state.market_timestamp,
                         entry_price=entry_price, size=size, sl_price=sl_price, tp_price=tp_price,
                         margin_used=margin_used)
    new_account = AccountState(
        balance=account.balance, equity=account.equity,
        margin_used=account.margin_used + margin_used,
        margin_free=account.equity - (account.margin_used + margin_used),
        exposure=account.exposure + size * entry_price,
        open_position_id=position.position_id, simulation_timestamp=market_state.market_timestamp,
    )
    return position, new_account


def close_position(market_state, account: AccountState, position: Position, exit_price: float,
                    config: SimulatedExecutionConfig):
    direction = 1.0 if position.side == Side.LONG else -1.0
    realized_pnl = direction * (exit_price - position.entry_price) * position.size
    sl_distance_r = (abs(position.entry_price - position.sl_price) / position.entry_price
                      if position.sl_price is not None else None)
    cost_r = compute_cost_r(market_state, sl_distance_r, config) if sl_distance_r is not None else None
    cost_amount = (cost_r * sl_distance_r * position.entry_price * position.size) if cost_r is not None else 0.0
    net_pnl = realized_pnl - cost_amount
    new_balance = account.balance + net_pnl
    new_account = AccountState(
        balance=new_balance, equity=new_balance, margin_used=0.0, margin_free=new_balance,
        exposure=0.0, open_position_id=None, simulation_timestamp=market_state.market_timestamp,
    )
    return net_pnl, cost_amount, new_account


def check_liquidation(account: AccountState, config: SimulatedExecutionConfig) -> bool:
    if account.margin_used <= 0:
        return False
    return (account.equity / account.margin_used) < config.liquidation_threshold
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/simulator/test_engine.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add simulator/engine.py tests/simulator/test_engine.py
git commit -m "feat: add simulator position lifecycle and account engine"
```

---

### Task 6: Experience recorder

**Files:**
- Create: `simulator/experience.py`
- Test: `tests/simulator/test_experience.py`

**Interfaces:**
- Consumes: `simulator.contracts.{EnvironmentTag, PositionOutcome, PositionView, AccountState}`.
- Produces: `ExperienceRecord` dataclass with fields `environment_tag: EnvironmentTag`, `timestamp: datetime`, `event_type: str` (one of `"DECIDE"`, `"MANAGE"`, `"POSITION_CLOSED"`), `market_state_snapshot: dict`, `position_view: Optional[dict]`, `action: Optional[str]`, `account_state: dict`, `realized_pnl: Optional[float]`, `cost_amount: Optional[float]`, `outcome: Optional[PositionOutcome]`. `ExperienceRecorder` class with `.record(record: ExperienceRecord) -> None` and `.all_records() -> list[ExperienceRecord]`, and a `write_tag_guard(active_partition: EnvironmentTag, record: ExperienceRecord) -> None` that raises `ValueError` if `record.environment_tag != active_partition` (the "a write with an inconsistent tag is rejected, not merely mislabeled" requirement from the design doc, Section 3).

- [ ] **Step 1: Write the failing test**

```python
"""tests/simulator/test_experience.py"""
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from simulator.contracts import EnvironmentTag, PositionOutcome
from simulator.experience import ExperienceRecord, ExperienceRecorder, write_tag_guard


def _record(tag=EnvironmentTag.SIMULATED_TRAINING):
    return ExperienceRecord(
        environment_tag=tag, timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc), event_type="DECIDE",
        market_state_snapshot={"mid": 1500.0}, position_view=None, action="NO_TRADE",
        account_state={"balance": 10000.0}, realized_pnl=None, cost_amount=None, outcome=None,
    )


def test_recorder_stores_records_in_order():
    recorder = ExperienceRecorder()
    r1 = _record()
    r2 = _record()
    recorder.record(r1)
    recorder.record(r2)
    assert recorder.all_records() == [r1, r2]


def test_write_tag_guard_allows_matching_tag():
    write_tag_guard(EnvironmentTag.SIMULATED_TRAINING, _record(EnvironmentTag.SIMULATED_TRAINING))


def test_write_tag_guard_rejects_mismatched_tag():
    with pytest.raises(ValueError):
        write_tag_guard(EnvironmentTag.SIMULATED_OOS_TEST, _record(EnvironmentTag.SIMULATED_TRAINING))


def test_experience_record_captures_position_closed_event():
    record = ExperienceRecord(
        environment_tag=EnvironmentTag.SIMULATED_TRAINING, timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
        event_type="POSITION_CLOSED", market_state_snapshot={"mid": 1510.0},
        position_view={"position_id": "p1"}, action=None, account_state={"balance": 10005.0},
        realized_pnl=15.0, cost_amount=2.0, outcome=PositionOutcome.POLICY_EXIT,
    )
    assert record.outcome == PositionOutcome.POLICY_EXIT
    assert record.realized_pnl == 15.0


if __name__ == "__main__":
    test_recorder_stores_records_in_order()
    test_write_tag_guard_allows_matching_tag()
    test_write_tag_guard_rejects_mismatched_tag()
    test_experience_record_captures_position_closed_event()
    print("tests/simulator/test_experience.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/simulator/test_experience.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'simulator.experience'`

- [ ] **Step 3: Write `simulator/experience.py`**

```python
"""simulator/experience.py
Records the raw ingredients of every decide()/manage()/close event -- PnL,
cost, account snapshot, environment tag -- with NO reward formula computed
here. Reward shaping (R-multiple, Sharpe-like, drawdown-penalized, etc.) is
explicitly a Phase 2+ research question (see design doc Section 13.B)."""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from simulator.contracts import EnvironmentTag, PositionOutcome


@dataclass
class ExperienceRecord:
    environment_tag: EnvironmentTag
    timestamp: datetime
    event_type: str  # "DECIDE" | "MANAGE" | "POSITION_CLOSED"
    market_state_snapshot: dict
    position_view: Optional[dict]
    action: Optional[str]
    account_state: dict
    realized_pnl: Optional[float]
    cost_amount: Optional[float]
    outcome: Optional[PositionOutcome]


class ExperienceRecorder:
    def __init__(self):
        self._records: list[ExperienceRecord] = []

    def record(self, record: ExperienceRecord) -> None:
        self._records.append(record)

    def all_records(self) -> list[ExperienceRecord]:
        return list(self._records)


def write_tag_guard(active_partition: EnvironmentTag, record: ExperienceRecord) -> None:
    if record.environment_tag != active_partition:
        raise ValueError(
            f"Experience record tagged {record.environment_tag} written during "
            f"active partition {active_partition} -- cross-partition contamination rejected."
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/simulator/test_experience.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add simulator/experience.py tests/simulator/test_experience.py
git commit -m "feat: add simulator experience recorder with partition-tag guard"
```

---

### Task 7: Replay orchestrator

**Files:**
- Create: `simulator/replay.py`
- Test: `tests/simulator/test_replay.py`

**Interfaces:**
- Consumes: everything from Tasks 1-6: `simulator.contracts.*`, `simulator.closure.classify_gap`, `simulator.market_state_builder.build_snapshot`, `simulator.execution.{entry_fill_price, exit_fill_price, resolve_same_bar_ambiguity}`, `simulator.engine.{open_position, close_position, check_liquidation, to_position_view}`, `simulator.experience.{ExperienceRecord, ExperienceRecorder, write_tag_guard}`.
- Produces: `run_replay(df: pandas.DataFrame, decide_fn, manage_fn, config: SimulatedExecutionConfig, environment_tag: EnvironmentTag) -> ExperienceRecorder`. `decide_fn(market_state, account: AccountState) -> tuple[str, Optional[float], Optional[float]]` returning `(action, sl_price, tp_price)` where `action` is `"NO_TRADE"`, `"LONG"`, or `"SHORT"`. `manage_fn(market_state, position_view: PositionView, account: AccountState) -> str` returning `"HOLD"` or `"EXIT"`.

**Note for implementer:** `decide_fn`/`manage_fn` in this task's tests are trivial test-only stub policies (e.g. "always NO_TRADE", "open one fixed LONG then always HOLD until forced close") — these exist purely to exercise the orchestrator's plumbing. Do not design, tune, or improve their trading logic; that is explicitly Phase 2+ scope, out of bounds here. One open position at a time only — `run_replay` must never call `decide_fn` while `account.open_position_id is not None`, and must never call `manage_fn` while flat.

- [ ] **Step 1: Write the failing test**

```python
"""tests/simulator/test_replay.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

from simulator.contracts import EnvironmentTag, SimulatedExecutionConfig, PositionOutcome
from simulator.replay import run_replay


def _make_df(n=20, start_price=1500.0):
    times = pd.date_range("2020-01-06 10:00:00", periods=n, freq="1min")
    prices = [start_price + i * 0.1 for i in range(n)]
    return pd.DataFrame({
        "time": times, "open": prices,
        "high": [p + 0.2 for p in prices], "low": [p - 0.2 for p in prices],
        "close": [p + 0.05 for p in prices], "tick_volume": [10] * n, "spread": [20.0] * n,
    })


def _always_no_trade(market_state, account):
    return ("NO_TRADE", None, None)


def test_run_replay_all_no_trade_never_opens_position():
    df = _make_df()
    config = SimulatedExecutionConfig()
    recorder = run_replay(df, _always_no_trade, lambda *a: "HOLD", config, EnvironmentTag.SIMULATED_TRAINING)
    records = recorder.all_records()
    assert len(records) > 0
    assert all(r.event_type == "DECIDE" for r in records)
    assert all(r.action == "NO_TRADE" for r in records)


def _open_one_long_then_hold_forever():
    state = {"opened": False}

    def decide(market_state, account):
        if not state["opened"]:
            state["opened"] = True
            return ("LONG", None, None)
        return ("NO_TRADE", None, None)

    def manage(market_state, position_view, account):
        return "HOLD"

    return decide, manage


def test_run_replay_opens_and_force_closes_at_end_of_data():
    df = _make_df()
    config = SimulatedExecutionConfig()
    decide, manage = _open_one_long_then_hold_forever()
    recorder = run_replay(df, decide, manage, config, EnvironmentTag.SIMULATED_TRAINING)
    records = recorder.all_records()
    closed = [r for r in records if r.event_type == "POSITION_CLOSED"]
    assert len(closed) == 1
    assert closed[0].outcome == PositionOutcome.END_OF_REPLAY_FORCED_CLOSE


def test_run_replay_reopens_immediately_after_exit_no_cooldown():
    df = _make_df(n=10)
    config = SimulatedExecutionConfig()
    call_count = {"decides_while_flat": 0}

    def decide(market_state, account):
        call_count["decides_while_flat"] += 1
        if call_count["decides_while_flat"] in (1, 3):
            return ("LONG", None, None)
        return ("NO_TRADE", None, None)

    def manage(market_state, position_view, account):
        return "EXIT"  # exit on the very next bar after opening

    recorder = run_replay(df, decide, manage, config, EnvironmentTag.SIMULATED_TRAINING)
    closed = [r for r in recorder.all_records() if r.event_type == "POSITION_CLOSED"]
    assert len(closed) >= 2  # opened, exited, reopened, exited again -- with no forced gap


if __name__ == "__main__":
    test_run_replay_all_no_trade_never_opens_position()
    test_run_replay_opens_and_force_closes_at_end_of_data()
    test_run_replay_reopens_immediately_after_exit_no_cooldown()
    print("tests/simulator/test_replay.py: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/simulator/test_replay.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'simulator.replay'`

- [ ] **Step 3: Write `simulator/replay.py`**

```python
"""simulator/replay.py
Chronological single-pass orchestrator. Calls decide_fn once per bar while
flat, manage_fn once per bar while a position is open, and enforces SL/TP/
liquidation as safety-net checks every bar regardless of what manage_fn
returns. No inter-trade cooldown: the bar immediately after a close is
eligible for a fresh decide_fn call. decide_fn/manage_fn are policy
callables -- this module contains no trading logic of its own."""
from typing import Callable, Optional

from simulator.contracts import AccountState, EnvironmentTag, PositionOutcome, Side, SimulatedExecutionConfig
from simulator.engine import open_position, close_position, check_liquidation, to_position_view
from simulator.execution import exit_fill_price, resolve_same_bar_ambiguity
from simulator.experience import ExperienceRecord, ExperienceRecorder, write_tag_guard
from simulator.market_state_builder import build_snapshot

DecideFn = Callable[[object, AccountState], tuple]
ManageFn = Callable[[object, object, AccountState], str]


def _account_dict(account: AccountState) -> dict:
    return {
        "balance": account.balance, "equity": account.equity, "margin_used": account.margin_used,
        "margin_free": account.margin_free, "exposure": account.exposure,
        "open_position_id": account.open_position_id,
    }


def run_replay(df, decide_fn: DecideFn, manage_fn: ManageFn, config: SimulatedExecutionConfig,
               environment_tag: EnvironmentTag) -> ExperienceRecorder:
    recorder = ExperienceRecorder()
    n = len(df)
    if n == 0:
        return recorder
    first_ts = df.iloc[0]["time"].to_pydatetime()
    account = AccountState.initial(config, first_ts)
    position = None
    bars_held = 0

    for i in range(n):
        market_state = build_snapshot(df, i)
        row = df.iloc[i]

        if position is None:
            action, sl_price, tp_price = decide_fn(market_state, account)
            record = ExperienceRecord(
                environment_tag=environment_tag, timestamp=market_state.market_timestamp, event_type="DECIDE",
                market_state_snapshot={"mid": market_state.mid, "spread": market_state.spread},
                position_view=None, action=action, account_state=_account_dict(account),
                realized_pnl=None, cost_amount=None, outcome=None,
            )
            write_tag_guard(environment_tag, record)
            recorder.record(record)
            if action in ("LONG", "SHORT"):
                side = Side.LONG if action == "LONG" else Side.SHORT
                position, account = open_position(market_state, account, side, sl_price, tp_price, config)
                bars_held = 0
            continue

        bars_held += 1
        position_view = to_position_view(position, market_state.mid, bars_held)
        outcome = None
        exit_price = None

        if i == n - 1:
            outcome = PositionOutcome.END_OF_REPLAY_FORCED_CLOSE
            exit_price = exit_fill_price(position.side, market_state.mid, market_state.spread, config)
        else:
            same_bar_result = resolve_same_bar_ambiguity(
                position.side, float(row["high"]), float(row["low"]), position.sl_price, position.tp_price
            )
            if same_bar_result == "SL_HIT":
                outcome = PositionOutcome.SL_HIT
                exit_price = position.sl_price
            elif same_bar_result == "TP_HIT":
                outcome = PositionOutcome.TP_HIT
                exit_price = position.tp_price
            elif check_liquidation(account, config):
                outcome = PositionOutcome.LIQUIDATION
                exit_price = exit_fill_price(position.side, market_state.mid, market_state.spread, config)
            else:
                manage_decision = manage_fn(market_state, position_view, account)
                record = ExperienceRecord(
                    environment_tag=environment_tag, timestamp=market_state.market_timestamp, event_type="MANAGE",
                    market_state_snapshot={"mid": market_state.mid, "spread": market_state.spread},
                    position_view=position_view.__dict__, action=manage_decision,
                    account_state=_account_dict(account), realized_pnl=None, cost_amount=None, outcome=None,
                )
                write_tag_guard(environment_tag, record)
                recorder.record(record)
                if manage_decision == "EXIT":
                    outcome = PositionOutcome.POLICY_EXIT
                    exit_price = exit_fill_price(position.side, market_state.mid, market_state.spread, config)

        if outcome is not None:
            net_pnl, cost_amount, account = close_position(market_state, account, position, exit_price, config)
            close_record = ExperienceRecord(
                environment_tag=environment_tag, timestamp=market_state.market_timestamp, event_type="POSITION_CLOSED",
                market_state_snapshot={"mid": market_state.mid, "spread": market_state.spread},
                position_view=position_view.__dict__, action=None, account_state=_account_dict(account),
                realized_pnl=net_pnl, cost_amount=cost_amount, outcome=outcome,
            )
            write_tag_guard(environment_tag, close_record)
            recorder.record(close_record)
            position = None
            bars_held = 0

    return recorder
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/simulator/test_replay.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add simulator/replay.py tests/simulator/test_replay.py
git commit -m "feat: add simulator chronological replay orchestrator"
```

---

### Task 8: No-future-leakage test harness

**Files:**
- Create: `tests/simulator/test_no_leakage.py`

**Interfaces:**
- Consumes: `simulator.market_state_builder.build_snapshot`, `simulator.replay.run_replay`, `simulator.contracts.*`.
- Produces: no new production code — this task is a dedicated audit test suite. Poison-tests both the `build_snapshot` call sites used at `decide()` time (flat) and at `manage()` time (position open), per design doc Section 13.B correction #6.

- [ ] **Step 1: Write the test**

```python
"""tests/simulator/test_no_leakage.py
Mechanical no-future-leakage audit (design doc Section 1 / Section 13.B #6).
Runs the replay twice -- once clean, once with every bar's high/low/close
AFTER each decision point poisoned before that step's snapshot is built --
and asserts every recorded market_state_snapshot is identical between runs.
Covers both DECIDE (flat) and MANAGE (position open) call sites."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

from simulator.contracts import EnvironmentTag, SimulatedExecutionConfig
from simulator.market_state_builder import build_snapshot
from simulator.replay import run_replay


def _make_df(n=30):
    times = pd.date_range("2020-01-06 10:00:00", periods=n, freq="1min")
    prices = [1500.0 + i * 0.1 for i in range(n)]
    return pd.DataFrame({
        "time": times, "open": prices,
        "high": [p + 0.3 for p in prices], "low": [p - 0.3 for p in prices],
        "close": [p + 0.05 for p in prices], "tick_volume": [10] * n, "spread": [20.0] * n,
    })


def test_build_snapshot_identical_regardless_of_future_bar_values():
    df = _make_df()
    for i in range(len(df)):
        clean_snap = build_snapshot(df, i)
        poisoned = df.copy()
        if i + 1 < len(df):
            poisoned.loc[i + 1:, ["open", "high", "low", "close", "spread"]] = -999999.0
        poisoned_snap = build_snapshot(poisoned, i)
        assert clean_snap.mid == poisoned_snap.mid, f"leakage at row {i}: mid differs"
        assert clean_snap.spread == poisoned_snap.spread, f"leakage at row {i}: spread differs"
        if clean_snap.completed_m1 is not None:
            assert clean_snap.completed_m1.close == poisoned_snap.completed_m1.close, f"leakage at row {i}"
        assert clean_snap.current_m1.high == poisoned_snap.current_m1.high, f"leakage at row {i}: current bar high"
        assert clean_snap.current_m1.low == poisoned_snap.current_m1.low, f"leakage at row {i}: current bar low"


def test_replay_records_identical_snapshots_regardless_of_unreached_future():
    df = _make_df()
    config = SimulatedExecutionConfig()

    def always_no_trade(market_state, account):
        return ("NO_TRADE", None, None)

    recorder_clean = run_replay(df, always_no_trade, lambda *a: "HOLD", config, EnvironmentTag.SIMULATED_TRAINING)

    truncated = df.iloc[: len(df) // 2].copy()
    recorder_truncated = run_replay(
        truncated, always_no_trade, lambda *a: "HOLD", config, EnvironmentTag.SIMULATED_TRAINING
    )

    n_common = len(recorder_truncated.all_records())
    clean_records = recorder_clean.all_records()[:n_common]
    truncated_records = recorder_truncated.all_records()
    for a, b in zip(clean_records, truncated_records):
        assert a.market_state_snapshot == b.market_state_snapshot, (
            "leakage: truncating the dataset after the current decision point changed an "
            "earlier snapshot -- the earlier decision must not depend on data that doesn't exist yet"
        )


if __name__ == "__main__":
    test_build_snapshot_identical_regardless_of_future_bar_values()
    test_replay_records_identical_snapshots_regardless_of_unreached_future()
    print("tests/simulator/test_no_leakage.py: OK")
```

- [ ] **Step 2: Run test to verify it passes against Tasks 1-7's implementation**

Run: `pytest tests/simulator/test_no_leakage.py -v`
Expected: PASS (2 tests). **If this fails, it means a real leakage bug exists in `market_state_builder.py` or `replay.py` from earlier tasks — fix the implementation, never loosen these assertions.** This is the single most important test in the whole plan (design doc's "Critical principle," Section 1).

- [ ] **Step 3: Run the full simulator test suite together**

Run: `pytest tests/simulator/ -v`
Expected: all tests across Tasks 1-8 PASS together (no cross-task regressions).

- [ ] **Step 4: Commit**

```bash
git add tests/simulator/test_no_leakage.py
git commit -m "test: add mechanical no-future-leakage audit for GOLDEX V4 Phase 1 simulator"
```

---

## Self-review notes (already folded into the tasks above)

- Spec coverage: `AccountState`/`SimulatedExecutionConfig`/`EnvironmentTag` (Task 1), closure detection (Task 2), `MarketState` snapshot builder + leakage core (Task 3, audited further in Task 8), execution/fill/cost/same-bar-ambiguity (Task 4), position lifecycle + liquidation + account updates (Task 5), experience recorder + partition-tag guard (Task 6), full replay orchestration with `decide()`/`manage()`, no cooldown, optional SL/TP, corrected outcome enum (Task 7), mechanical no-leakage audit covering both call sites (Task 8) — every Section-13-corrected requirement has a task.
- No placeholders: every step has real, runnable code; no "TODO"/"add appropriate handling" text anywhere.
- Type consistency checked: `Side`, `PositionOutcome`, `EnvironmentTag`, `AccountState`, `Position`, `PositionView`, `SimulatedExecutionConfig` are defined once in Task 1 and used with identical names/fields in Tasks 4-8; `decide_fn`/`manage_fn` signatures match between Task 7's `run_replay` and its own test stubs.
- Explicitly out of scope in every task's text: no decision policy, no reward formula, no Phase 2 code — confirmed absent from all 8 tasks above.
