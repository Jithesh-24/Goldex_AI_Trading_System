#!/usr/bin/env python3
"""SPACE GUARD (v7.6, 2026-08-03) — keep the 24/7 trading stack bounded.

The training matrix (gold_features.csv) is REBUILT nightly from gold_seed.csv —
bounded by construction (seed capped at ~60-day XM window). A few line-append
journals need a retention cap so nothing grows without limit. This guard runs
at EOD (before training), trims those, cleans stale temp scratch, reports each
file's footprint.

NEVER trims: model weights, gold_seed.csv (rebuilt nightly), gold_features.csv
(rebuilt nightly), or live_outcomes.jsonl — the permanent learned journal.
No agent, silent when healthy.
"""
import json, os, time

OUTDIR = "/home/jith/.hermes/profiles/trading/cron/output"
SCRIPTS = "/home/jith/.hermes/profiles/trading/scripts"
TMPDIR = "/home/jith/.hermes/profiles/trading/tmp"

KEEP_LIVE_BARS = 22000       # recent live M1 bars
KEEP_TRADE_JOURNAL = 10000   # engine trade journal
KEEP_REGIME_DAYS = 90.0      # market-behavior digests
STALE_TEMP_AGE = 3600
ALERT_TOTAL_GB = 30.0


def _trim_last_lines(path, keep):
    try:
        with open(path, "rb") as f:
            lines = f.readlines()
        if len(lines) <= keep:
            return 0
        tmp = path + ".rot"
        with open(tmp, "wb") as f:
            f.writelines(lines[-keep:])
        os.replace(tmp, path)
        return len(lines) - keep
    except Exception as e:
        print(f"  [trim failed] {path}: {e}")
        return 0


def _prune_age(path, days):
    now = time.time()
    try:
        with open(path) as f:
            rows = [l for l in f if l.strip()]
        keep = [l for l in rows if _recent(l, now, days)]
        if len(keep) < len(rows):
            tmp = path + ".rot"
            with open(tmp, "w") as f:
                f.writelines(keep)
            os.replace(tmp, path)
            return len(rows) - len(keep)
        return 0
    except Exception as e:
        print(f"  [prune failed] {path}: {e}")
        return 0


def _recent(line, now, days):
    try:
        r = json.loads(line)
        raw = r.get("t") or r.get("time") or r.get("ts")
        if isinstance(raw, (int, float)):
            return now - raw <= days * 86400
        return True
    except Exception:
        return True


def _rel(p):
    for base in (OUTDIR, SCRIPTS, TMPDIR):
        if p.startswith(base):
            return p[len(base) + 1:]
    return p


def _gb(path):
    try:
        return os.path.getsize(path) / 1e9
    except Exception:
        return 0.0


def _dir_gb(root):
    return sum(os.path.getsize(os.path.join(dp, fn))
               for dp, _, fns in os.walk(root)
               for fn in fns if os.path.isfile(os.path.join(dp, fn))) / 1e9


def main():
    msgs = []

    # 1. cap line-append journals
    for fname, keep in [("xm_live_bars.jsonl", KEEP_LIVE_BARS),
                        ("trade_journal_ai.jsonl", KEEP_TRADE_JOURNAL)]:
        p = f"{OUTDIR}/{fname}"
        if os.path.exists(p):
            n = _trim_last_lines(p, keep)
            if n:
                msgs.append(f"{fname}→pruned {n}")

    # 2. age-prune market-behavior digests
    for fname in ["market_regime_journal.jsonl"]:
        p = f"{OUTDIR}/{fname}"
        if os.path.exists(p):
            n = _prune_age(p, KEEP_REGIME_DAYS)
            if n:
                msgs.append(f"{fname}→pruned {n}")

    # 3. clean stale temp files (>1h old) in scripts + tmp
    now = time.time()
    for base in [SCRIPTS, TMPDIR]:
        for dp, _, fns in os.walk(base):
            for fn in fns:
                if not (fn.startswith(".full_") or fn.endswith((".tmp", ".rot"))):
                    continue
                pa = os.path.join(dp, fn)
                try:
                    if now - os.path.getmtime(pa) > STALE_TEMP_AGE:
                        os.remove(pa)
                        msgs.append(f"removed stale {_fs(pa)}")
                except Exception:
                    pass

    # 4. report footprint
    mtx = _gb(f"{SCRIPTS}/gold_features.csv")
    journals = sum(_gb(f"{OUTDIR}/{f}") for f in os.listdir(OUTDIR) if f.endswith(".jsonl"))
    total = _dir_gb(OUTDIR) + _dir_gb(SCRIPTS) + _dir_gb(TMPDIR)
    if total > ALERT_TOTAL_GB:
        msgs.append(f"⚠️ util {total:.1f}GB > {ALERT_TOTAL_GB:.0f}GB")

    status = " | ".join(msgs) if msgs else "healthy"
    print(f"[space-guard] {status} | matrix {mtx:.2f}GB(bounded) | journals {journals:.2f}GB | total≈{total:.1f}GB")


if __name__ == "__main__":
    main()