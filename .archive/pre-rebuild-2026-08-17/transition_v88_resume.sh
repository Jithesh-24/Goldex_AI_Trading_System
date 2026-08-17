#!/bin/bash
# ─────────────────────────────────────────────────────────────
# v8.8 RESUME FAST-PATH (2026-08-11 23:45) — OOM-recovery resume chain.
# The original chain completed steps 1-3 (column-add, merge_seed,
# build_m5_matrix, fit_placement_prior) then was OOM-KILLED during
# train_ai at 17:55 (5.6G+5.7G swap on a 7.5Gi box — one-shot
# pd.read_csv of the 33.7GB matrix = ~12.9GB float32).
# train_ai.py now streams into a DISK-BACKED float32 memmap
# (LightGBM consumes it zero-copy, verified). Resume = step 4+.
# Same log file → restart watcher still catches the completion.
# Same eod-retrain.service name → 03:00 EOD cron keeps auto-skipping.
#
# FAST-PATH (23:45): engine-critical steps FIRST, specialists AFTER
# the COMPLETE marker. Rating uses raw 3-seed OOF fallback initially;
# after specialists + spec-OOF finish, rating is REFIT on spec OOF and
# hot-reloads (engine reload guard picks it up — no restart needed).
# num_threads=8 (engine/ticker/MT5 stopped — full 8 logical CPUs).
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
    timeout "${timeout}" "${PY}" -u "$@" >> "${LOG}" 2>&1
    local rc=$?
    echo "[$(date +%H:%M:%S)] ${label} rc=${rc}" | tee -a "${LOG}"
    return ${rc}
}

export FEAT_CSV="${BASE}/gold_features_m5.csv"
export PRIOR_BAR_SECS=300
export PRIOR_HORIZONS=3      # Bug-A fix preserved (h3 trade key)
export DIR_HORIZON_BARS=36
export TMPDIR="${TMP}"

echo "[$(date +%H:%M:%S)] ═══ v8.8 TRANSITION RESUME FAST-PATH (8 threads) ═══" | tee -a "${LOG}"

# STEP 4 — FINISH (2026-08-12 recovery): train_ai made it through the
# walk-forward (OOF saved) + final s42 before the 10h timeout SIGTERM'd it
# (rc=124 at 09:44 — my wrong-speed-estimate bug). finish_finals.py reuses
# the saved OOF, re-streams the matrix, trains ONLY the 2 missing finals
# (s7, s2026), writes the configs. Timeout generous — 20h, no repeat.
run "finish_finals (recovery, 2 missing finals)" 72000 "${BASE}/finish_finals.py" || { echo "❌ finish_finals failed"; exit 1; }

# STEP 5a — direction model + calibration + rating (raw-OOF fallback) — ENGINE CRITICAL
run "train_direction_htf" 3600 "${BASE}/train_direction_htf.py" || { echo "❌ direction failed"; exit 1; }
run "fit_calibration_by_rr" 1800 "${BASE}/fit_calibration_by_rr.py" || { echo "❌ calibration failed"; exit 1; }
run "fit_signal_rating" 3600 "${BASE}/fit_signal_rating.py" || { echo "❌ rating failed"; exit 1; }

# STEP 6 — direction prior at h3 (PRIOR_HORIZONS=3 → Bug-A fix preserved)
run "regenerate_dir_prior" 1800 "${BASE}/regenerate_dir_prior.py" || { echo "❌ prior failed"; exit 1; }

# STEP 7 — loss-lesson replay through the FRESH models (adaptive proof)
run "eod_loss_lessons" 1800 "${BASE}/eod_loss_lessons.py" || { echo "❌ loss lessons failed"; exit 1; }

echo "[$(date +%H:%M:%S)] 🎉 v8.8 TRANSITION COMPLETE — M5 BASELINE READY (96-feat)" | tee -a "${LOG}"

# 2026-08-12: M5 specialists DELIBERATELY SKIPPED — the TICK retrain
# (transition_tick.sh, 99-feat) immediately replaces all M5 models. Training
# M5 specialists now would waste ~2h of CPU for models thrown away. The tick
# chain trains its own specialists after ITS COMPLETE marker.
echo "[$(date +%H:%M:%S)] ⏳ M5 chain ends here — tick retrain (transition_tick.sh) takes over. Engine stays OFF until tick verification." | tee -a "${LOG}"
