#!/bin/bash
# v8.8 watcher: waits for the transition chain to finish, then restarts the
# engine ONCE so the new source code (fx state-injection + hardened reload
# guard) is loaded into memory. Python does not hot-apply source edits —
# the engine process from 15:30:17 predates the 16:19-16:29 patches.
LOG=/home/jith/.hermes/profiles/trading/scripts/transition_v88.log
while ! grep -q "v8.8 TRANSITION COMPLETE" "${LOG}" 2>/dev/null; do
    sleep 60
done
echo "[$(date +%H:%M:%S)] chain complete — restarting engine to load v8.8 code"
systemctl --user restart ai-engine.service
sleep 10
systemctl --user is-active ai-engine.service
journalctl --user -u ai-engine.service --since "30 sec ago" --no-pager | tail -3
echo "WATCHER DONE $(date +%H:%M:%S)"
