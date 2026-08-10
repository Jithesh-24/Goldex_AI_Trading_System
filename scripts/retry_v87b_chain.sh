#!/usr/bin/env bash
# v8.7 RETRY: failed steps of the chain (specialists + downstream).
# Fixes applied: module-level `import features as F` in train_regime_spec.py;
# resilient model loaders in eod_loss_lessons.py; direction_ensemble.json
# pruned to existing files.
set -u
BASE=/home/jith/.hermes/profiles/trading/scripts
PY=/home/jith/.hermes/hermes-agent/venv/bin/python3
export FEAT_CSV=$BASE/gold_features_m5.csv
export PRIOR_BAR_SECS=300 PRIOR_HORIZONS=3,12,36 DIR_HORIZON_BARS=36
LOG=$BASE/retrain_v87b.log

echo "=== v8.7b retry start $(date) ===" >> $LOG

run() {
  echo "--- $1 $(date) ---" >> $LOG
  $PY -u $BASE/$1 >> $LOG 2>&1
  rc=$?
  echo "exit=$rc ($1)" >> $LOG
  return $rc
}

run train_regime_spec.py || { echo "FATAL: specialists failed — aborting retry" >> $LOG; exit 1; }
run build_spec_oof_full.py || { echo "FATAL: spec OOF failed" >> $LOG; exit 1; }
run fit_signal_rating.py || { echo "FATAL: rating failed" >> $LOG; exit 1; }
run regenerate_dir_prior.py || true
run eod_loss_lessons.py || true
echo "=== v8.7b retry COMPLETE $(date) ===" >> $LOG
