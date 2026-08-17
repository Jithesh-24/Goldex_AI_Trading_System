#!/bin/bash
# Hermes Scalper Watchdog — checks every 1min if scalper is alive
# If dead, restarts it

PROCESS_NAME="hermes_scalper.py"
PID_FILE="/tmp/hermes_scalper.pid"

# Check if running
if pgrep -f "$PROCESS_NAME" > /dev/null 2>&1; then
    # Running — save PID
    pgrep -f "$PROCESS_NAME" | head -1 > "$PID_FILE"
    exit 0
fi

# Not running — nothing to do, let the cron job restart it
echo "SCALPER_DOWN"