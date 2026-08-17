#!/bin/bash
# ─────────────────────────────────────────────────────────────
# v8.8 FEATURE-SPACE TRANSITION (2026-08-11) — position-state learning
# Adds day_pnl / streak / trades_today to the M5 matrix and runs the
# FULL 6yr cold retrain chain so the models learn P(win | state) from
# live outcome rows. Engine keeps running the WHOLE time — reload guards
# (boot + hot) refuse mismatched model/features pairs and keep the last
# valid set; the loop-level try/except catches anything else.
#
# Runs as systemd unit eod-retrain.service — the SAME unit the EOD cron
# launcher checks, so the 03:00 cron auto-skips while this is active.
# ─────────────────────────────────────────────────────────────
set -uo pipefail
BASE=/home/jith/.hermes/profiles/trading/scripts
PY=/home/jith/.hermes/hermes-agent/venv/bin/python3
TMP=/home/jith/.hermes/profiles/trading/tmp
LOG=${BASE}/transition_v88.log
mkdir -p "${TMP}"

run() {  # run <label> <timeout> <script...>
    local label="$1" timeout="$2"; shift 2
    echo "[$(date +%H:%M:%S)] ▶ ${label}: $*" | tee -a "${LOG}"
    "${PY}" -u "$@" >> "${LOG}" 2>&1
    local rc=$?
    echo "[$(date +%H:%M:%S)] ${label} rc=${rc}" | tee -a "${LOG}"
    return ${rc}
}

export FEAT_CSV="${BASE}/gold_features_m5.csv"
export PRIOR_BAR_SECS=300
export PRIOR_HORIZONS=3      # v8.8 FIX: trade key MUST be h3 (15-min) — h36 was the Bug-A
export DIR_HORIZON_BARS=36
export TMPDIR="${TMP}"

echo "[$(date +%H:%M:%S)] ═══ v8.8 FEATURE-SPACE TRANSITION START ═══" | tee -a "${LOG}"

# STEP 1 — streaming column-add (33.7GB I/O, ~15-30 min). Engine-safe: the
# engine never reads the matrix CSV. Atomic os.replace on success; abort on
# any row-width mismatch (no partial write).
run "column-add" 3600 "${BASE}/transition_add_state_cols.py" || { echo "❌ column-add failed"; exit 1; }

# STEP 2 — fresh seed + incremental matrix append (picks up bars since Aug 10;
# _feature_block_m5 now emits the 3 new cols → consistent 119-col rows)
run "merge_seed" 600 "${BASE}/merge_seed.py" || { echo "❌ merge_seed failed"; exit 1; }
run "build_m5_matrix --incremental" 1800 "${BASE}/build_m5_matrix.py" --incremental || { echo "❌ matrix build failed"; exit 1; }

# STEP 3 — closed-loop outcomes merge (carries REAL position-state from live
# trades into the new columns — the actual learning signal)
"${PY}" -c "
import sys; sys.path.insert(0, '${BASE}')
from retrain_loop import merge_live_outcomes_appended
n, tot = merge_live_outcomes_appended('${BASE}/gold_features_m5.csv')
print(f'merged {n} live outcome rows | matrix ~{tot:,} rows', flush=True)
" >> "${LOG}" 2>&1

# STEP 4 — FULL 6yr cold start (feature count 93→96: warm-start impossible).
# train_ai.py writes features.json BEFORE ensemble.json → when the engine's
# reload trigger (ensemble.json mtime) fires, the pair is already consistent.
run "fit_placement_prior" 1800 "${BASE}/fit_placement_prior.py" || { echo "❌ placement prior failed"; exit 1; }
run "train_ai (FULL 6yr cold)" 36000 "${BASE}/train_ai.py" || { echo "❌ train_ai failed"; exit 1; }

# STEP 5 — direction model, calibration, 8 regime specialists, OOF, rating
run "train_direction_htf" 3600 "${BASE}/train_direction_htf.py" || { echo "❌ direction failed"; exit 1; }
run "fit_calibration_by_rr" 1800 "${BASE}/fit_calibration_by_rr.py" || { echo "❌ calibration failed"; exit 1; }
run "train_regime_spec" 36000 "${BASE}/train_regime_spec.py" || { echo "❌ specialists failed"; exit 1; }
run "build_spec_oof_full" 7200 "${BASE}/build_spec_oof_full.py" || { echo "❌ spec OOF failed"; exit 1; }
run "fit_signal_rating" 3600 "${BASE}/fit_signal_rating.py" || { echo "❌ rating failed"; exit 1; }

# STEP 6 — direction prior at h3 (PRIOR_HORIZONS=3 → Bug-A fix preserved)
run "regenerate_dir_prior" 1800 "${BASE}/regenerate_dir_prior.py" || { echo "❌ prior failed"; exit 1; }

# STEP 7 — loss-lesson replay through the FRESH models (adaptive proof)
run "eod_loss_lessons" 1800 "${BASE}/eod_loss_lessons.py" || { echo "❌ loss lessons failed"; exit 1; }

echo "[$(date +%H:%M:%S)] 🎉 v8.8 TRANSITION COMPLETE — engine hot-reloaded new 96-feat models" | tee -a "${LOG}"
