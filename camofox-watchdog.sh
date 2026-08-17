#!/bin/bash
# Camofox watchdog — silent when healthy, restarts when dead (Linux/systemd)

CAMOFOX_PORT=9377
SERVICE="camofox-browser.service"
LOG="$HOME/.hermes/logs/camofox-watchdog.log"

# Check if healthy
if curl -sf "http://localhost:$CAMOFOX_PORT/health" > /dev/null 2>&1; then
    exit 0
fi

echo "[$(date)] Camofox down on port $CAMOFOX_PORT. Restarting via systemd..." >> "$LOG"

# Kill any orphaned processes
pkill -9 -f "camofox-browser" 2>/dev/null
pkill -9 -f "camoufox" 2>/dev/null
sleep 2

# Restart via systemd
systemctl --user restart "$SERVICE" 2>/dev/null
sleep 8

if curl -sf "http://localhost:$CAMOFOX_PORT/health" > /dev/null 2>&1; then
    echo "[$(date)] ✓ Camofox restarted OK via systemd" >> "$LOG"
else
    echo "[$(date)] ✗ Camofox failed — attempting direct start" >> "$LOG"
    cd ~/.hermes/hermes-agent/node_modules/@askjo/camofox-browser && nohup npm start > /dev/null 2>&1 &
    sleep 8
    if curl -sf "http://localhost:$CAMOFOX_PORT/health" > /dev/null 2>&1; then
        echo "[$(date)] ✓ Camofox started directly" >> "$LOG"
    else
        echo "[$(date)] ✗ Camofox FAILED to start" >> "$LOG"
    fi
fi
