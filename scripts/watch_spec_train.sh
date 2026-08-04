#!/bin/bash
# Wait for spec-train unit to finish, report final outcome.
for i in $(seq 1 240); do
  st=$(systemctl --user is-active spec-train.service 2>/dev/null)
  if [ "$st" != "active" ]; then break; fi
  sleep 30
done
echo "=== [$(date +%H:%M:%S)] spec-train ended (state=$st) ==="
echo "=== journal tail ==="
journalctl --user -u spec-train.service --no-pager -n 30 2>/dev/null | tail -30
echo "=== log tail ==="
tail -30 /home/jith/.hermes/profiles/trading/cron/output/spec_train.log 2>/dev/null
echo "=== regime_specialists.json ==="
cat /home/jith/.hermes/profiles/trading/scripts/models/regime_specialists.json 2>/dev/null
echo "=== spec model files ==="
ls -la /home/jith/.hermes/profiles/trading/scripts/models/spec_*.txt 2>/dev/null