#!/bin/bash
# tick_pipeline_orchestrator.sh — waits for prerequisites, then launches the
# tick retrain automatically. Runs as a periodic cron check (every 5 min).
# Prerequisites:
#   1. M5 chain finished (v8.8 TRANSITION COMPLETE in transition_v88.log)
#   2. Dukascopy fetch finished (dukascopy_m1_features.csv exists, has data)
# Then: build_tick_matrix.py → transition_tick.sh (eod-tick.service)
# Idempotent: does nothing once eod-tick.service exists/started.
set -uo pipefail
BASE=/home/jith/.hermes/profiles/trading/scripts
LOG88=${BASE}/transition_v88.log
DKCSV=${BASE}/dukascopy_m1_features.csv
MATRIX=${BASE}/gold_features_m5_tick.csv
TICKLOG=${BASE}/transition_tick.log

# Already launched?
systemctl --user is-active eod-tick.service >/dev/null 2>&1 && exit 0
[ -f "${TICKLOG}" ] && grep -q "TICK TRANSITION RESUME" "${TICKLOG}" && exit 0

# Prereq 1: M5 chain complete — finish_finals.py alone writes models but no
# magic completion marker. transition_v88.sh is the only thing that prints
# "v8.8 TRANSITION COMPLETE" (final step of the v8.8 tail). If the marker is
# missing, the M5 chain tail has not finished — do NOT proceed.
grep -q "v8.8 TRANSITION COMPLETE" "${LOG88}" 2>/dev/null || exit 0

# Prereq 2: Dukascopy CSV present and non-trivial
[ -s "${DKCSV}" ] || exit 0
SIZE=$(stat -c%s "${DKCSV}" 2>/dev/null || echo 0)
[ "${SIZE}" -gt 1000000 ] || exit 0

# Oops — matrix already built but chain not started? Weird state; relaunch.
if [ -s "${MATRIX}" ] && ! systemctl --user is-active eod-tick.service >/dev/null 2>&1; then
    :
fi

echo "[$(date +%H:%M:%S)] ✅ prerequisites met — launching TICK RETRAIN (106-feat: 99 + 7 macro)"
# Build the tick matrix if not yet built
if [ ! -s "${MATRIX}" ]; then
    /home/jith/.hermes/hermes-agent/venv/bin/python3 -u "${BASE}/build_tick_matrix.py" >> "${BASE}/build_tick_matrix.log" 2>&1 || {
        echo "❌ build_tick_matrix failed — see build_tick_matrix.log"; exit 1; }
fi
# Launch the tick chain as its own unit (never blocks the M5 unit name)
systemd-run --user --unit=eod-tick.service --collect \
    /bin/bash "${BASE}/transition_tick.sh" 2>&1 | head -1
echo "eod-tick.service launched: $(date +%H:%M:%S)"