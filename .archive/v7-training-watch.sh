#!/bin/bash
# v7-training-watch.sh — silent until training finishes, then prints the result.
# Used as no_agent cron: empty stdout = no message; non-empty = delivered.
LOG=/home/jith/.hermes/profiles/trading/scripts/v7_training.log
if [ -f "$LOG" ] && grep -q "v7 training finished" "$LOG"; then
  echo "✅ v7 TRAINING COMPLETE"
  grep -E "OOS|ACC|P\(up\)|PF|Backtest|TREND|RANGE|WIN|ROWS|===" "$LOG" | tail -40
fi
