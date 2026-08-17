#!/bin/bash
# v7.3d pipeline watchdog — notifies when the label-fix pipeline finishes or dies.
LOG=/home/jith/.hermes/profiles/trading/scripts/v7p3d_pipeline.log
if systemctl --user is-active v73d >/dev/null 2>&1; then
  exit 0  # still running — silent
fi
if grep -q "PIPELINE COMPLETE" "$LOG" 2>/dev/null; then
  echo "v7.3d pipeline COMPLETE (label fix) — new honest-label models + backtest ready."
  grep -E "V7 BACKTEST|WR |RR:|MIXED|RANGE|TREND|PF |EV " "$LOG" | tail -20
else
  echo "v7.3d pipeline STOPPED unexpectedly — check $LOG"
  tail -30 "$LOG"
fi
