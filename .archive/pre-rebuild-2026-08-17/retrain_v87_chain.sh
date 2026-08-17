#!/usr/bin/env bash
# v8.7 M5-only EOD retrain chain — waits for train_continue to finish,
# then runs the remaining steps. Mirrors eod_m5.py order exactly.
set -u
BASE=/home/jith/.hermes/profiles/trading/scripts
PY=/home/jith/.hermes/hermes-agent/venv/bin/python3
export FEAT_CSV=$BASE/gold_features_m5.csv
export PRIOR_BAR_SECS=300 PRIOR_HORIZONS=3,12,36 DIR_HORIZON_BARS=36
LOG=$BASE/retrain_v87.log

echo "=== v8.7 M5-only chain start $(date) ===" >> $LOG

# wait for the running train_continue (pid 2776158) to exit
while kill -0 2776158 2>/dev/null; do
  sleep 20
done
echo "=== train_continue done $(date) ===" >> $LOG

# ── CLOSED-LOOP MERGE: append today's resolved live trades (3x SL) to the
# training matrix BEFORE the decision components train, so the losses are
# LEARNED (target=0 rows, full recency weight), not just replayed. Mirrors
# eod_m5.py's first step.
echo "--- merge_live_outcomes ---" >> $LOG
$PY -c "
import sys; sys.path.insert(0, '$BASE')
from retrain_loop import merge_live_outcomes_appended
n, tot = merge_live_outcomes_appended('$FEAT_CSV')
print(f'merged {n} live outcome rows | matrix ~{tot:,} rows')
" >> $LOG 2>&1
echo "exit=$? (merge_live_outcomes)" >> $LOG

# ── RE-RUN train_continue AFTER the merge: the matrix grew, so the OOF
# (oof_probs.npy) saved by the first pass is index-misaligned with the
# matrix (fit_calibration_by_rr pairs OOF rows with direction/rr columns
# by row position). Warm-starting from the just-trained models is cheap and
# ALSO gives today's SL rows their first real training weight. This is the
# same warm-start the nightly EOD applies.
echo "--- train_continue (post-merge warm-start) ---" >> $LOG
$PY -u $BASE/train_continue.py >> $LOG 2>&1
echo "exit=$? (train_continue post-merge)" >> $LOG

run() {
  echo "--- $1 $(date) ---" >> $LOG
  $PY -u $BASE/$1 >> $LOG 2>&1
  echo "exit=$? ($1)" >> $LOG
}

run train_direction_htf.py
run fit_calibration_by_rr.py
run train_regime_spec.py
run build_spec_oof_full.py
run fit_signal_rating.py
run regenerate_dir_prior.py
run eod_loss_lessons.py
echo "=== v8.7 M5-only chain COMPLETE $(date) ===" >> $LOG
