#!/bin/bash
# v7 training runner — launched via systemd-run (survives gateway restarts).
# Owns its own cgroup via systemd --user, NOT the gateway's 2GB slice.
cd /home/jith/.hermes/profiles/trading/scripts
export TMPDIR=/home/jith/.hermes/profiles/trading/tmp
PY=/home/jith/.hermes/hermes-agent/venv/bin/python3
LOG=/home/jith/.hermes/profiles/trading/scripts/v7_training.log
{
  echo "=== v7 training started $(date) ==="
  echo "--- train_ai.py (placement 3-seed) ---"
  $PY -u train_ai.py 2>&1
  echo "PLACEMENT_EXIT=$?"
  echo "--- train_direction.py (direction 3-seed) ---"
  $PY -u train_direction.py 2>&1
  echo "DIRECTION_EXIT=$?"
  echo "--- backtest_v7.py ---"
  $PY -u backtest_v7.py 2>&1
  echo "BACKTEST_EXIT=$?"
  echo "=== v7 training finished $(date) ==="
} >> "$LOG" 2>&1
