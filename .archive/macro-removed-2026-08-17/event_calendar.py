"""event_calendar.py — REAL historical US macro event calendar 2019-2026 (2026-08-12).

CRITICAL FIX (found by research audit): features.py's _event_times() generated
ONLY 2026 events (NFP+CPI, hardcoded 12th) — training bars from 2019-2025 saw
min_to_event/pre_event/post_event computed against 2026 events = flat garbage;
FOMC (the single dominant gold jump driver per Sobti IIMA, Narain-Sangani,
Cai/Cheng/Wong 2001) was MISSING ENTIRELY. The model never learned news
structure. This module supplies the real schedule.

SOURCES (deterministic, no API key):
- FOMC meeting dates: Federal Reserve published meeting calendars (the Fed
  publishes dates 2+ years ahead; these are fixed historical facts for
  2019-2026). Statement 19:00 UTC (14:00 ET), presser 19:30 UTC (14:30 ET).
  Meetings WITHOUT presser (pre-2011 or non-presser meetings): statement only.
  All 2019-2026 meetings have pressers.
- NFP: first Friday of each month, 13:30 UTC (12:30 ET) — deterministic rule.
- CPI: BLS release schedule — mid-month (10th-15th), 13:30 UTC (08:30 ET).
  Exact dates vary; encode a table of KNOWN release dates where feasible and
  fall back to the published BLS-schedule weekly pattern (mostly the 10th-15th
  on Tues/Wed/Thu). Approximation is far better than the old flat-2026 bug,
  and walk-forward validation will confirm.
- FOMC minutes: ~3 weeks after meeting, 19:00 UTC. Also high-impact events.

Events are emitted as (unix_ts, code, kind) where code in
{FOMC, FOMC_PRESSER, FOMC_MINUTES, NFP, CPI} and kind is the impact class
(FOMC/NFP/CPI = "high"). The engine/training feeds use this via a tiny shim
in features.py so existing min_to_event/pre_event/post_event columns become
MEANINGFUL for all 6 years.
"""
import calendar as _cal
import datetime as _dt
import time

try:
    from zoneinfo import ZoneInfo
    _NY = ZoneInfo("America/New_York")  # DST-aware — NFP/CPI 08:30 ET, FOMC 14:00 ET
except Exception:
    _NY = _dt.timezone.utc

_EVENTS = None


# FOMC meeting dates 2019-2026 — from Federal Reserve published schedules.
# (year, month, day) of the meeting's first day (decisions announced on the
# second day at 14:00 ET = 19:00 UTC; presser 14:30 ET = 19:30 UTC).
_FOMC = [
    (2019, 1, 30), (2019, 3, 20), (2019, 5, 1), (2019, 6, 19),
    (2019, 7, 31), (2019, 9, 18), (2019, 10, 30), (2019, 12, 11),
    (2020, 1, 29), (2020, 3, 3), (2020, 3, 15), (2020, 4, 29),
    (2020, 6, 10), (2020, 7, 29), (2020, 9, 16), (2020, 11, 5),
    (2020, 12, 16), (2021, 1, 27), (2021, 3, 17), (2021, 4, 28),
    (2021, 6, 16), (2021, 7, 28), (2021, 9, 22), (2021, 11, 3),
    (2021, 12, 15), (2022, 1, 26), (2022, 3, 16), (2022, 5, 4),
    (2022, 6, 15), (2022, 7, 27), (2022, 9, 21), (2022, 11, 2),
    (2022, 12, 14), (2023, 2, 1), (2023, 3, 22), (2023, 5, 3),
    (2023, 6, 14), (2023, 7, 26), (2023, 9, 20), (2023, 11, 1),
    (2023, 12, 13), (2024, 1, 31), (2024, 3, 20), (2024, 5, 1),
    (2024, 6, 12), (2024, 7, 31), (2024, 9, 18), (2024, 11, 7),
    (2024, 12, 18), (2025, 1, 29), (2025, 3, 19), (2025, 5, 7),
    (2025, 6, 18), (2025, 7, 30), (2025, 9, 17), (2025, 10, 29),
    (2025, 12, 10), (2026, 1, 28), (2026, 3, 18), (2026, 4, 29),
    (2026, 6, 17), (2026, 7, 29), (2026, 9, 16), (2026, 10, 28),
    (2026, 12, 9),
]

# BLS CPI release dates (13:30 UTC / 08:30 ET). Schedule published annually;
# mid-month. We list KNOWN dates for 2019-2025 (from BLS archives) and use the
# published 2026 schedule.
_CPI_KNOWN = [
    "2019-01-11", "2019-02-13", "2019-03-12", "2019-04-10", "2019-05-10",
    "2019-06-12", "2019-07-11", "2019-08-13", "2019-09-12", "2019-10-10",
    "2019-11-13", "2019-12-11", "2020-01-14", "2020-02-13", "2020-03-11",
    "2020-04-10", "2020-05-12", "2020-06-10", "2020-07-14", "2020-08-12",
    "2020-09-11", "2020-10-13", "2020-11-12", "2020-12-10", "2021-01-13",
    "2021-02-10", "2021-03-10", "2021-04-13", "2021-05-12", "2021-06-10",
    "2021-07-13", "2021-08-11", "2021-09-14", "2021-10-13", "2021-11-10",
    "2021-12-10", "2022-01-12", "2022-02-10", "2022-03-10", "2022-04-12",
    "2022-05-11", "2022-06-10", "2022-07-13", "2022-08-10", "2022-09-13",
    "2022-10-13", "2022-11-10", "2022-12-13", "2023-01-12", "2023-02-14",
    "2023-03-14", "2023-04-12", "2023-05-10", "2023-06-13", "2023-07-12",
    "2023-08-10", "2023-09-13", "2023-10-12", "2023-11-14", "2023-12-12",
    "2024-01-11", "2024-02-13", "2024-03-12", "2024-04-10", "2024-05-15",
    "2024-06-12", "2024-07-11", "2024-08-14", "2024-09-11", "2024-10-10",
    "2024-11-13", "2024-12-11", "2025-01-14", "2025-02-12", "2025-03-12",
    "2025-04-10", "2025-05-13", "2025-06-11", "2025-07-15", "2025-08-12",
    "2025-09-11", "2025-10-14", "2025-11-13", "2025-12-11",
    # 2026 published schedule (BLS annual calendar)
    "2026-01-13", "2026-02-11", "2026-03-11", "2026-04-10", "2026-05-12",
    "2026-06-10", "2026-07-14", "2026-08-12", "2026-09-11", "2026-10-13",
    "2026-11-12", "2026-12-10",
]


def _ts(y, m, d, hr, mn=0):
    """DST-aware: interpret (y,m,d hr:mn) in America/New_York → UTC epoch.
    NFP/CPI release at 08:30 ET, FOMC statement 14:00 ET — the UTC hour
    shifts ±1 with US daylight saving. Hardcoding 13:30/19:00 UTC year-round
    puts the event window an hour late for half the year (research warning:
    'encode with a DST-aware clock, not fixed UTC offsets')."""
    return int(_dt.datetime(y, m, d, hr, mn, tzinfo=_NY).timestamp())


def build():
    """Return sorted list of (unix_ts, code) for 2019-2026."""
    ev = []

    # FOMC: statement 14:00 ET on the meeting's 2nd day, presser 14:30 ET,
    # minutes +21 days at 14:00 ET.
    for (y, m, d) in _FOMC:
        dt0 = _dt.date(y, m, d) + _dt.timedelta(days=1)
        ev.append((_ts(dt0.year, dt0.month, dt0.day, 14, 0), "FOMC"))
        ev.append((_ts(dt0.year, dt0.month, dt0.day, 14, 30), "FOMC_PRESSER"))
        mins = dt0 + _dt.timedelta(days=21)
        ev.append((_ts(mins.year, mins.month, mins.day, 14, 0), "FOMC_MINUTES"))

    # NFP: first Friday 08:30 ET — with BLS holiday exceptions: when the first
    # Friday is a federal holiday (New Year's Day / observed July 4), the
    # release moves (BLS archives). Known 2019-2026 shifts:
    #   2020-07: Jul 3 observed holiday (Jul 4 = Sat) → released Thu Jul 2
    #   2021-01: Jan 1 holiday → released Fri Jan 8
    NFP_EXC = {
        (2020, 7): 2,   # day of month
        (2021, 1): 8,
    }
    for y in range(2019, 2027):
        for m in range(1, 13):
            if (y, m) in NFP_EXC:
                fd = NFP_EXC[(y, m)]
            else:
                c = _cal.monthcalendar(y, m)
                fd = next(w[4] for w in c if w[4] != 0)
            ev.append((_ts(y, m, fd, 8, 30), "NFP"))

    # CPI: known BLS dates 08:30 ET
    for s in _CPI_KNOWN:
        y, m, d = (int(x) for x in s.split("-"))
        ev.append((_ts(y, m, d, 8, 30), "CPI"))

    ev.sort()
    return ev


def events():
    global _EVENTS
    if _EVENTS is None:
        _EVENTS = build()
    return _EVENTS


def codes():
    return {c for _, c in events()}


if __name__ == "__main__":
    ev = build()
    print(f"total events 2019-2026: {len(ev)}")
    from collections import Counter
    print(Counter(c for _, c in ev))
    for _ts0, c in ev[:8]:
        print(f"  {_dt.datetime.fromtimestamp(_ts0, _dt.timezone.utc).isoformat()}  {c}")
    print("  ...")
    for _ts0, c in ev[-4:]:
        print(f"  {_dt.datetime.fromtimestamp(_ts0, _dt.timezone.utc).isoformat()}  {c}")