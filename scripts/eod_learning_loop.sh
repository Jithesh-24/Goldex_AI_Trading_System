#!/bin/bash
# ─────────────────────────────────────────────────────────────
# EOD LEARNING LOOP launcher (v7.3e) — 2026-08-03
# Purpose: run retrain_loop.py OUTSIDE the gateway's 2GB cgroup so
#   train_ai (needs ~5GB via Dataset.subset) does NOT OOM-kill the
#   gateway. Also survives gateway restarts (own user unit).
# The engine hot-reloads the atomic-swapped models on completion,
# so the loop can run WHILE the bot is live.
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
    --property=MemoryMax=7G \
    --property=CPUAccounting=yes \
    "${PY}" -u "${BASE}/retrain_loop.py" >> "${LOG}" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    echo "[eod] systemd-run failed rc=$rc — learning loop NOT launched"
    exit 1
fi
date +%s > "${STONE}"
exit 0