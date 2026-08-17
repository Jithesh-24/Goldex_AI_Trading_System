"""PHASE 5 — Learning loop: retrain the AI on fresh data + live outcomes.

Runs the full pipeline end-to-end so the model continuously improves:
  1. merge_seed.py   — pull latest MT5 history + live scanner bars into gold_seed.csv
  2. features        — rebuild feature matrix WITH trade-realistic target
  3. train_ai.py     — walk-forward validation + atomic model swap (hot-reloaded by engine)
  4. journal stats   — report live signal accuracy & PnL so the system sees its own outcomes

Safe to run while the engine is live: model swap is atomic, engine hot-reloads.
"""
import json, os, subprocess, sys, time
import pandas as pd

# RAW_PRICE_COLS is needed by merge_live_outcomes_appended (streaming append).
# Import safely at module scope so we don't depend on a mid-function import.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "."))
try:
    from features import RAW_PRICE_COLS
except Exception:
    try:
        from features import RAW_PRICES as RAW_PRICE_COLS
    except Exception:
        RAW_PRICE_COLS = ()
from datetime import datetime

BASE = "/home/jith/.hermes/profiles/trading/scripts"
OUTDIR = "/home/jith/.hermes/profiles/trading/cron/output"
PY = sys.executable
SEED = f"{BASE}/gold_seed.csv"
FEAT_CSV = f"{BASE}/gold_features.csv"
MODEL_DIR = f"{BASE}/models"
RETRAIN_LOG = f"{OUTDIR}/retrain_log.jsonl"

def run(cmd, timeout=600):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {r.stderr[-800:]}")
    return r.stdout

def journal_stats():
    """Live signal performance: every resolved trade the engine logged."""
    rows = []
    try:
        with open(f"{OUTDIR}/trade_journal_ai.jsonl") as f:
            for line in f:
                try: rows.append(json.loads(line))
                except Exception: pass
    except Exception:
        return None
    if not rows:
        return None
    # UNVERIFIED outcomes (partial-coverage ack) are NOT counted — they are
    # not real results, only "confirm on your terminal" notices.
    real = [r for r in rows if r.get("result") in ("TP", "SL")]
    wins = [r for r in real if r.get("result") == "TP"]
    losses = [r for r in real if r.get("result") == "SL"]
    net = sum(r.get("pnl", 0) for r in real)
    return {
        "n": len(real), "wins": len(wins), "losses": len(losses),
        "unverified": len(rows) - len(real),
        "net": net,
        "by_dir": {}
    }

def merge_live_outcomes_appended(feat_csv):
    """CLOSED LOOP (streaming, OOM-safe): append resolved live trades
    (actual outcomes, TP→1 / SL→0) DIRECTLY to the feature-matrix CSV on disk.
    Does NOT load the full 6.2M-row matrix into RAM — reads only the header,
    builds the tiny live-outcome frame, appends to the file.
    Returns (n_live_rows, total_rows). (2026-08-03, plan Phase 0.0 fix.)"""
    outcomes_path = f"{OUTDIR}/live_outcomes.jsonl"
    n_rows = _count_file_lines(feat_csv)
    if not os.path.exists(outcomes_path):
        return 0, n_rows
    header = list(pd.read_csv(feat_csv, nrows=0).columns)
    non_feat = set(["time", "target", "fwd_return"]) | set(RAW_PRICE_COLS)
    feats_cols = [c for c in header if c not in non_feat]
    rows = []
    merged_path = f"{OUTDIR}/live_outcomes_merged.json"
    merged = set()
    if os.path.exists(merged_path):
        try:
            with open(merged_path) as f:
                merged = set(json.load(f))
        except Exception:
            merged = set()
    with open(outcomes_path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("result") not in ("TP", "SL"):
                continue  # unverified / pending
            feats = r.get("feats") or {}
            if not feats:
                continue
            # v5: outcomes stored under an OLD 51-feat space (no regime group)
            # would 0.0-fill the 12 new regime features → biased rows. Skip.
            if "trend_ema" not in feats or "bb_pctile" not in feats:
                continue
            uid = f"{r.get('t')}|{r.get('dir')}"
            if uid in merged:
                continue  # idempotent: already merged in a prior EOD run
            row = {c: (feats.get(c) if c in feats_cols else None) for c in header}
            # v8.8 (2026-08-11): carry position-state into the training matrix so
            # the retrain learns streak/day-pnl effects (pure learning, no gates).
            if "day_pnl" in feats_cols:
                row["day_pnl"] = (r.get("state") or {}).get("day_pnl", 0.0)
            if "streak" in feats_cols:
                row["streak"] = (r.get("state") or {}).get("streak", 0)
            if "trades_today" in feats_cols:
                row["trades_today"] = (r.get("state") or {}).get("trades_today", 0)
            if "direction" not in feats and r.get("dir") in ("BUY", "SELL"):
                row["direction"] = 1.0 if r["dir"] == "BUY" else 0.0
            row["time"] = pd.Timestamp(int(r["t"]), unit="s").strftime("%Y-%m-%d %H:%M:%S")  # int-second, matches matrix datetime64[s] format (no fraction → mixed-col parse crash)
            row["target"] = 1.0 if r.get("result") == "TP" else 0.0
            row["fwd_return"] = 0.0
            rows.append(row)
            merged.add(uid)
    if not rows:
        return 0, n_rows
    extra = pd.DataFrame(rows, columns=header).fillna(0.0)
    extra.to_csv(feat_csv, mode="a", header=False, index=False)  # append
    # persist merged UIDs so a re-run never re-appends (closed-loop idempotency)
    with open(merged_path, "w") as f:
        json.dump(sorted(merged), f)
    return len(rows), n_rows + len(rows)

def _count_file_lines(path):
    try:
        with open(path, "rb") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0

def main():
    t0 = time.time()
    print(f"[{datetime.now():%H:%M:%S}] ═══ AI RETRAIN LOOP ═══")

    # 1. Fresh seed (MT5 history + live bars)
    print("→ merging seed (MT5 + live bars)...")
    out = run([PY, f"{BASE}/merge_seed.py"], timeout=300)
    print(out.strip())

    # 1b. LIVE-ONLY (v7.4, 2026-08-03): downloaded Dukascopy backtest/rally
    #     history is ARCHIVED (archives/backtest-data-2026-08-03/). The model
    #     now trains EXCLUSIVELY on real live XM data (gold_seed.csv) + real
    #     live outcomes. No more old-data training for the next 5 days.
    #     Weekends re-train on that week's accumulated live data (see EOD cron).
    RALLY_FEAT = f"{BASE}/gold_features_rally.csv"
    if os.path.exists(RALLY_FEAT):
        print(f"→ rally cache still present at {RALLY_FEAT} (live-only: skipped)")
    else:
        print("→ LIVE-ONLY training: downloaded backtest data archived, using real XM live data")

    # 2. Rebuild features — v7.10 INCREMENTAL (2026-08-04): if a feature matrix
    #    already exists with a matching schema fingerprint, build_full_matrix.py
    #    appends ONLY the ~1,440 new bars (seconds, not the 58-CPU-min full
    #    6.39M-row rebuild that kept timing out at 3600s). A full rebuild runs
    #    only on first build, schema drift, or --full.
    print("→ building/updating feature matrix (v7.10 incremental by default)...")
    out = run([PY, f"{BASE}/build_full_matrix.py", "--incremental"], timeout=900)
    print(out.strip()[-800:])

    # CLOSED LOOP: append actual live outcomes (direct label: TP=1/SL=0).
    # v7.3f OOM-FIX (2026-08-03, plan Phase 0.0): previously we did
    # pd.read_csv(FEAT_CSV) = load the ENTIRE 6.2M-row float32 matrix into one
    # DataFrame (~2.5GB) + concat copy (~5GB peak) + rewrite — the exact
    # pattern that OOM'd the v6 build twice and the likely silent-kill of
    # retrains at this step. Now stream: read only the header, append live rows
    # directly to the CSV on disk. Peak memory stays ~ the few live rows.
    import pandas as pd
    sys.path.insert(0, BASE)
    n_out, feat_rows = merge_live_outcomes_appended(FEAT_CSV)
    if n_out:
        print(f"→ merged {n_out} LIVE outcome rows into training matrix (closed loop) | matrix ~{feat_rows:,} rows")
    else:
        print(f"→ feature matrix: ~{feat_rows:,} rows (no new live outcomes)")

    # 3. Retrain (walk-forward + atomic swap) — v7 DUAL MODEL:
    # 3a. WARM-START CONTINUATION (v7.4, 2026-08-03): keep the models trained
    #     on ALL XAUUSD history and continue learning from live data.
    #     train_continue.py loads each deployed seed's weights (history base),
    #     then adapts them on gold_features.csv (live data + live outcomes).
    #     No cold rebuild of a fresh seed.
    print("→ warm-start continuation (preserve history + adapt live)...")
    out = run([PY, f"{BASE}/train_continue.py"], timeout=10800)
    print("\n".join(out.strip().splitlines()[-20:]))
    print("→ training direction model (v7.6 HTF-regime PERSISTENCE, walk-forward, 3 seeds)...")
    out = run([PY, f"{BASE}/train_direction_htf.py"], timeout=3600)
    print("\n".join(out.strip().splitlines()[-20:]))

    # 3b. v7.3f: per-direction × per-RR calibration, from the SAME OOF
    #     preds train_ai.py just saved (oof_probs.npy / oof_targets.npy) and
    #     the SAME gold_features.csv just rebuilt above — reuses artifacts
    #     already on disk, no extra rebuild. Previously this only ran from
    #     an orphaned manual pipeline, so it went stale relative to the
    #     model that's actually retrained daily (calibration_by_drr.json
    #     mtime frozen while gold_lgb_model_s*.txt kept moving). Wiring it
    #     here keeps it in lockstep with every retrain.
    print("→ fitting per-direction × per-RR calibration...")
    out = run([PY, f"{BASE}/fit_calibration_by_rr.py"], timeout=600)
    print("\n".join(out.strip().splitlines()[-15:]))

    # 3c. MARKET REGIME JOURNAL (2026-08-03) — learn BEYOND trade outcomes:
    #     daily digest of O/C/range/volatility/session stats + significant
    #     move events, appended to market_regime_journal.jsonl. Reads the
    #     fresh seed; cheap (<10s).
    print("→ journaling market regime (daily market diary)...")
    try:
        out = run([PY, f"{BASE}/journal_market_regime.py"], timeout=300)
        print(out.strip()[-300:])
    except Exception as e:
        print(f"⚠ regime journal skipped: {e}")

    # 3d. REGIME-TRANSITION TRAINER (2026-08-03→04): learns which market
    #     STATE precedes a significant move (15-bar horizon, 3×ATR threshold
    #     → ~30% base rate, genuinely informative). Trains on the full
    #     freshly-built matrix. Honest gate: OOS acc > 55%; a saturated or
    #     uninformative label aborts itself. Model → regime_transition.json,
    #     hot-reloadable later by the engine (not yet consumed — learning first).
    print("→ training regime-transition model (state → big-move precursors)...")
    try:
        out = run([PY, f"{BASE}/train_regime_transition.py"], timeout=3600)
        print("\n".join(out.strip().splitlines()[-15:]))
    except Exception as e:
        print(f"⚠ regime-transition training skipped: {e}")

    # 3e. REGIME SPECIALISTS (2026-08-04, v7.7): one placement ensemble per
    #     regime bin (8 bins) trained on the 6-year matrix rows that fell in
    #     that bin. The engine routes each live tick to the CURRENT regime's
    #     specialist via the shared regime_bin() rule — train/live routing
    #     can never diverge. Streaming/bucketed so peak RSS stays ~1-2GB.
    print("→ training regime specialists (8 bins × 3 seeds, 6yr matrix)...")
    try:
        out = run([PY, f"{BASE}/train_regime_spec.py"], timeout=10800)
        print("\n".join(out.strip().splitlines()[-20:]))
    except Exception as e:
        print(f"⚠ regime-specialist training skipped: {e}")

    # 3f. SPACE GUARD (2026-08-03) — data growth is bounded by design
    #     (matrix rebuilt nightly, seed capped at 60-day XM window) but the
    #     append-journals grow forever; guard audits disk, cleans temp, and
    #     rotates append-journals under caps. Runs LAST (cleans training temp).
    print("→ space guard (disk audit + journal rotation)...")
    try:
        out = run([PY, f"{BASE}/space_guard.py"], timeout=300)
        print(out.strip()[-300:])
    except Exception as e:
        print(f"⚠ space guard skipped: {e}")

    # 4. Report live performance
    js = journal_stats()
    if js:
        wr = js["wins"] / js["n"] * 100 if js["n"] else 0
        unv = f" | {js['unverified']} unverified (excluded)" if js.get("unverified") else ""
        print(f"\n→ LIVE SIGNAL PERFORMANCE: {js['n']} trades | {js['wins']}W/{js['losses']}L "
              f"({wr:.0f}% WR) | net {js['net']:+.2f}{unv}")

    # log retrain event
    try:
        with open(RETRAIN_LOG, "a") as f:
            f.write(json.dumps({"t": time.time(), "bars": feat_rows,
                                "last": str(feat_rows)}) + "\n")
    except Exception:
        pass
    print(f"[{datetime.now():%H:%M:%S}] ✅ retrain complete in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
