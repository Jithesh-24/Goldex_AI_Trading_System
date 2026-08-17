#!/bin/bash
# ─────────────────────────────────────────────────────────────
# EOD LEARNING LOOP launcher — M5 PIPELINE (v8.4c) — 2026-08-08
# Purpose: run eod_m5.py OUTSIDE the gateway's 2GB cgroup (own systemd
#   unit) so training does NOT OOM-kill the gateway. Survives gateway
#   restarts. THE CRITICAL FIX: the OLD launcher ran retrain_loop.py
#   which trains on the M1 matrix (gold_features.csv, 102 feats) and
#   overwrote the deployed M5 global ensemble (108 feats, base_tf=m5)
#   with M1-lineage models every night. The engine's TF guard refused
#   the poisoned reload in-memory (journal: ENSEMBLE RELOAD REFUSED),
#   but the disk state was destroyed. eod_m5.py runs the ENTIRE loop
#   at M5 (gold_features_m5.csv, PRIOR_BAR_SECS=300) — warm-start
#   continuation preserves the 6yr base and adapts to the last 180 days.
#
# Used as the no_agent cron script for job b0c10005e3c0.
# Prints ONLY when launch fails; silent when healthy/launched.
# ─────────────────────────────────────────────────────────────
set -uo pipefail
BASE=/home/jith/.hermes/profiles/trading/scripts
PY=/home/jith/.hermes/hermes-agent/venv/bin/python3
TMP=/home/jith/.hermes/profiles/trading/tmp
LOG=${BASE}/eod_learning.log
STONE=/home/jith/.hermes/profiles/trading/cron/output/.eod_last_start

# If an EOD run is already active (unit exists & running), skip silently.
if systemctl --user is-active --quiet eod-retrain.service 2>/dev/null; then
    exit 0
fi

# Mark start, then launch via systemd-run (own cgroup, survives restart).
systemctl --user reset-failed eod-retrain.service 2>/dev/null || true
systemd-run --user --unit=eod-retrain \
    --working-directory="${BASE}" \
    --setenv="TMPDIR=${TMP}" \
    --setenv="FEAT_CSV=${BASE}/gold_features_m5.csv" \
    --setenv="PRIOR_BAR_SECS=300" \
    --setenv="PRIOR_HORIZONS=3" \
    --setenv="DIR_HORIZON_BARS=36" \
    --property=MemoryMax=7G \
    --property=CPUAccounting=yes \
    "${PY}" -u "${BASE}/eod_m5.py" >> "${LOG}" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    echo "[eod] systemd-run failed rc=$rc — M5 learning loop NOT launched"
    exit 1
fi
date +%s > "${STONE}"
exit 0
