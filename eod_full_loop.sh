#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# eod_full_loop.sh — COMPLETE daily learning loop
# 1. EOD loop (data refresh + 108-feature retrain + direction + calibration)
# 2. 130-feature retrain (main model with Renaissance features)
# ═══════════════════════════════════════════════════════════════════
set -uo pipefail
BASE=/home/jith/.hermes/profiles/trading/scripts
PY=/home/jith/.hermes/hermes-agent/venv/bin/python3
LOG=${BASE}/eod_full_loop.log

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG"; }

log "═══ FULL EOD LOOP START ═══"

# Step 1: EOD loop (data refresh + 108-feature retrain)
log "Step 1: EOD loop (data refresh + 108-feature retrain)..."
if bash ${BASE}/eod_learning_loop_m5.sh; then
    log "✅ EOD loop completed"
else
    log "❌ EOD loop failed — check eod_learning.log"
    exit 1
fi

# Step 2: 130-feature retrain (main model)
log "Step 2: 130-feature retrain..."
if ${PY} -u ${BASE}/retrain_130_daily.py >> "$LOG" 2>&1; then
    log "✅ 130-feature retrain completed"
else
    log "❌ 130-feature retrain failed"
    exit 1
fi

log "═══ FULL EOD LOOP COMPLETE ═══"
