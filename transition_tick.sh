#!/bin/bash
# ─────────────────────────────────────────────────────────────
# v8.9 TICK TRANSITION (2026-08-12 12:30) — the user directive chain.
# Teaches the beast TICK FLOW (Dukascopy M1 6yr backfill → imb_300s,
# vol_rel, cvd) in addition to the M5 structure. Engine stays OFF until
# this completes AND is verified (watchdog report gates the green light).
#
# Precursor (runs BEFORE this, not here):
#   fetch_dukascopy_m1.py   → dukascopy_m1_features.csv (1920/2200 days done)
#   build_tick_matrix.py    → gold_features_m5_tick.csv  (99 + 7 macro = 106)
# This script is the TRAIN half of the chain, on the tick matrix.
#
# Timeouts: generous (24h+ per step). NO repeat of the rc=124 mistake.
# Specialists run AFTER the TICK COMPLETE marker (engine live first with
# ensemble+direction+calib+rating+prior; specialists hot-reload).
# ─────────────────────────────────────────────────────────────
set -uo pipefail
BASE=/home/jith/.hermes/profiles/trading/scripts
PY=/home/jith/.hermes/hermes-agent/venv/bin/python3
TMP=/home/jith/.hermes/profiles/trading/tmp
LOG=${BASE}/transition_tick.log
mkdir -p "${TMP}"

run() {  # run <label> <timeout> <script...>
    local label="$1" timeout="$2"; shift 2
    echo "[$(date +%H:%M:%S)] ▶ ${label}: $*" | tee -a "${LOG}"
    timeout "${timeout}" "${PY}" -u "$@" >> "${LOG}" 2>&1
    local rc=$?
    echo "[$(date +%H:%M:%S)] ${label} rc=${rc}" | tee -a "${LOG}"
    return ${rc}
}

export FEAT_CSV="${BASE}/gold_features_m5_tick.csv"   # ← TICK MATRIX (99 feats)
export PRIOR_BAR_SECS=300
export PRIOR_HORIZONS=3      # Bug-A fix preserved (h3 trade key)
export DIR_HORIZON_BARS=36
export TMPDIR="${TMP}"

echo "[$(date +%H:%M:%S)] ═══ v8.9 TICK TRANSITION RESUME (99-feat matrix, 8t) ═══" | tee -a "${LOG}"

# STEP 1 — full cold retrain on the TICK matrix (99 features).
run "train_ai TICK (full cold)" 90000 "${BASE}/train_ai.py" || { echo "❌ tick train_ai failed"; exit 1; }

# STEP 2 — direction + calibration + rating on tick features (engine critical)
run "train_direction_htf" 7200 "${BASE}/train_direction_htf.py" || { echo "❌ tick direction failed"; exit 1; }
run "fit_calibration_by_rr" 3600 "${BASE}/fit_calibration_by_rr.py" || { echo "❌ tick calibration failed"; exit 1; }
run "fit_signal_rating" 3600 "${BASE}/fit_signal_rating.py" || { echo "❌ tick rating failed"; exit 1; }

# STEP 3 — direction prior at h3 (Bug-A fix preserved)
run "regenerate_dir_prior" 3600 "${BASE}/regenerate_dir_prior.py" || { echo "❌ tick prior failed"; exit 1; }

# STEP 4 — loss-lesson replay through the FRESH tick models
run "eod_loss_lessons" 3600 "${BASE}/eod_loss_lessons.py" || { echo "❌ tick loss lessons failed"; exit 1; }

echo "[$(date +%H:%M:%S)] 🏆🏆 TICK TRANSITION COMPLETE — engine hot-reloaded 99-feat tick models" | tee -a "${LOG}"

# ── CANONICAL MATRIX SWAP (2026-08-12): the tick matrix becomes the one true
# matrix so the daily EOD loop (train_continue, incremental appends, schema
# fingerprint) operates on the SAME 99-feature space the deployed models
# expect. Atomic rename; the old 96-feat matrix is kept as .m5backup.
if [ -s "${BASE}/gold_features_m5_tick.csv" ]; then
    cp "${BASE}/gold_features_m5.csv" "${BASE}/gold_features_m5.m5backup" 2>/dev/null
    mv "${BASE}/gold_features_m5_tick.csv" "${BASE}/gold_features_m5.csv"
    echo "[$(date +%H:%M:%S)] ✅ canonical matrix = gold_features_m5.csv (99-feat tick)" | tee -a "${LOG}"
fi

# ── POST-LIVE: specialists hot-reload (deliberate — engine live first) ──
run "train_regime_spec TICK" 90000 "${BASE}/train_regime_spec.py" || echo "⚠ specialists failed (engine on global ensemble)"
run "build_spec_oof_full TICK" 14400 "${BASE}/build_spec_oof_full.py" || echo "⚠ spec OOF failed"
run "fit_signal_rating (spec upgrade)" 3600 "${BASE}/fit_signal_rating.py" || echo "⚠ rating upgrade failed"

echo "[$(date +%H:%M:%S)] 🏆 v8.9 TICK SPECIALISTS COMPLETE — per-regime + spec-OOF hot-reloaded" | tee -a "${LOG}"