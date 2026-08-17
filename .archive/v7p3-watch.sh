#!/bin/bash
# v7.3 pipeline watchdog — notifies when the pipeline finishes or dies.
LOG=/home/jith/.hermes/profiles/trading/scripts/v7p3_pipeline.log
if systemctl --user is-active v7p3 >/dev/null 2>&1; then
  exit 0  # still running — silent
fi
if grep -q "PIPELINE DONE" "$LOG" 2>/dev/null; then
  echo "✅ v7.3 pipeline COMPLETE — new strategy-playbook models + backtest ready."
  tail -30 "$LOG"
else
  echo "⚠️ v7.3 pipeline STOPPED unexpectedly — check $LOG"
  tail -30 "$LOG"
fi
