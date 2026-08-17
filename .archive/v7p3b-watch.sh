#!/bin/bash
# v7.3b pipeline watchdog — notifies when the resume pipeline finishes or dies.
LOG=/home/jith/.hermes/profiles/trading/scripts/v7p3b_pipeline.log
if systemctl --user is-active v7p3b >/dev/null 2>&1; then
  exit 0  # still running — silent
fi
if grep -q "RESUME PIPELINE DONE" "$LOG" 2>/dev/null; then
  echo "✅ v7.3 pipeline COMPLETE — new strategy-playbook models + backtest ready."
  tail -40 "$LOG"
else
  echo "⚠️ v7.3b pipeline STOPPED unexpectedly — check $LOG"
  tail -40 "$LOG"
fi
