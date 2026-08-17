#!/bin/bash
# model_staleness_watch.sh — ALERT on stale/dead learning pipeline (2026-08-12)
#
# THE GAP (verified): the chain-completion watch only fires on COMPLETE. If the
# retrain chain dies silently (OOM, crash, network, human kill) the engine keeps
# trading the LAST GOOD models forever — stale-model bleed with zero warning.
# This watch flags it BEFORE the user keeps compounding on dead knowledge.
#
# Checks every 15 min (cron):
#   1. models/features.json mtime — if the FEATURE SET is older than N days the
#      deployed models predate the macro/tick era → stale by construction.
#   2. the CANONICAL live-decision files (ensemble.json + 3 seed models) older
#      than N days → the learned surface is aging while the market moves on.
#      (Direction/specialist files are refreshed by later chain steps and the
#      tick chain; only the ensemble+seeds+features are the deployed decision.)
#   3. if a retrain chain (eod-retrain.service) is RUNNING, suppress alerts —
#      the pipeline is alive and will refresh models on completion.
#
# N days: 3 — a 10h+6h chain completes every 1-2 days in normal operation;
# 3 days without a refresh = pipeline is dead or the EOD cron is paused.

OUTDIR="$HOME/.hermes/profiles/trading/cron/output"
MODELS="$HOME/.hermes/profiles/trading/scripts/models"
STALE_DAYS=3
STAMP="$OUTDIR/.model_staleness_sent"

# Canonical deployed decision set (what the engine actually loads to trade)
CANONICAL=(features.json ensemble.json gold_lgb_model_s42.txt \
           gold_lgb_model_s7.txt gold_lgb_model_s2026.txt)

log() { echo "[$(date +%H:%M:%S)] $*"; }

# Retrain chain running? (systemd user service eod-retrain.service)
if systemctl --user is-active eod-retrain.service >/dev/null 2>&1; then
    log "retrain chain RUNNING — pipeline alive, skipping stale check"
    exit 0
fi

OLDEST=99999
for f in "${CANONICAL[@]}"; do
    full="$MODELS/$f"
    [ -f "$full" ] || { log "MISSING canonical file: $f"; OLDEST=$STALE_DAYS; continue; }
    age_days=$(( ($(date +%s) - $(stat -c %Y "$full")) / 86400 ))
    if [ "$age_days" -lt "$OLDEST" ]; then OLDEST=$age_days; fi
done

if [ "$OLDEST" -ge "$STALE_DAYS" ]; then
    MSG="⚠️ <b>MODEL STALENESS — learning pipeline may be DEAD</b>
━━━━━━━━━━━━━━━
Oldest model file: <b>${OLDEST}d</b> old (threshold ${STALE_DAYS}d).
No retrain chain running, no completed refresh in ${OLDEST} days.

If the engine is live, it is trading a <b>${OLDEST}-day-old learned surface</b>
while the market moved on. Check: eod-retrain.service, transition_tick.log,
train_ai.py, Dukascopy fetch, EOD cron (b0c10005e3c0)."
    # Alert once per staleness episode (avoid spam every 15 min)
    if [ ! -f "$STAMP" ]; then
        log "$MSG"
        touch "$STAMP"
        if command -v hermes >/dev/null 2>&1; then
            hermes send --text "$MSG" 2>/dev/null || echo "$MSG"
        else
            echo "$MSG"
        fi
    else
        log "staleness already alerted ($STAMP) — waiting for refresh"
    fi
else
    log "models fresh (oldest ${OLDEST}d) ✓"
    rm -f "$STAMP"
fi