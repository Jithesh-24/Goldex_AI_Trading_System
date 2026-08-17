#!/bin/bash
# spec_done_restart.sh — wait for M5 specialist training to finish, verify
# base_tf stamp, then restart the engine so the regime router loads them.
# Chained via systemd-run after spec-m5-train.service.
set -u
UNIT="spec-m5-train.service"
SPEC="/home/jith/.hermes/profiles/trading/scripts/models/regime_specialists.json"
TICKER_PID_FILE="/home/jith/.hermes/profiles/trading/cron/output/xm_ticker.pid"
LOG="/tmp/spec_restart.log"

echo "[$(date +%H:%M:%S)] waiting for $UNIT to finish..." >> "$LOG"

# Wait up to 4h for the training unit to exit (active -> inactive/failed)
for i in $(seq 1 480); do
  state=$(systemctl --user is-active "$UNIT" 2>/dev/null)
  if [ "$state" != "active" ]; then
    echo "[$(date +%H:%M:%S)] unit state=$state" >> "$LOG"
    break
  fi
  sleep 30
done

# Give the JSON a moment to flush after process exit
sleep 5

if [ ! -s "$SPEC" ]; then
  echo "[$(date +%H:%M:%S)] ERROR: $SPEC missing/empty — NOT restarting" >> "$LOG"
  exit 1
fi

# Verify the base_tf stamp — engine TF guard would refuse otherwise
TF=$(/home/jith/.hermes/hermes-agent/venv/bin/python -c \
  "import json; print(json.load(open('$SPEC')).get('base_tf','?'))" 2>/dev/null)
if [ "$TF" != "m5" ]; then
  echo "[$(date +%H:%M:%S)] ERROR: base_tf='$TF' != m5 — NOT restarting" >> "$LOG"
  exit 1
fi
echo "[$(date +%H:%M:%S)] specialists base_tf=$TF OK — restarting engine" >> "$LOG"

systemctl --user restart ai-engine.service >> "$LOG" 2>&1
sleep 8
systemctl --user is-active ai-engine.service >> "$LOG" 2>&1
echo "[$(date +%H:%M:%S)] engine restart issued, state: $(systemctl --user is-active ai-engine.service)" >> "$LOG"
echo "DONE" >> "$LOG"
