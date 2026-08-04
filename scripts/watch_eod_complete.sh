#!/bin/bash
# Robust EOD completion watcher — polls the retrain PIDs (transient units
# make `systemctl wait` unreliable). Prints final outcome when both the
# retrain_loop and its children are gone.
set -uo pipefail
BASE=/home/jith/.hermes/profiles/trading/scripts
MODELS=${BASE}/models
marker() { date '+%H:%M:%S'; }

# Wait until no retrain_loop.py and no build_full_matrix.py remain.
for i in $(seq 1 360); do   # up to 6h
    alive=$(pgrep -f "retrain_loop.py|build_full_matrix.py" | grep -v $$ || true)
    if [ -z "$alive" ]; then
        break
    fi
    sleep 30
done

echo "=== [$(marker)] retrain pipeline ended ==="
echo "=== unit state ==="
systemctl --user is-active eod-retrain.service 2>&1

echo "=== journal tail (last 30) ==="
journalctl --user -u eod-retrain.service --no-pager -n 30 2>/dev/null | tail -30

echo "=== matrix ==="
stat -c '%y | %s bytes' ${BASE}/gold_features.csv 2>/dev/null
wc -l ${BASE}/gold_features.csv 2>/dev/null

echo "=== models (placement + direction + calibration) ==="
ls -la ${MODELS}/gold_lgb_model_s*.txt ${MODELS}/direction_s*.txt 2>/dev/null
ls -la ${MODELS}/calibration*.json ${MODELS}/direction_metrics.json 2>/dev/null

echo "=== retrain log tail ==="
tail -3 /home/jith/.hermes/profiles/trading/cron/output/retrain_log.jsonl 2>/dev/null

echo "=== live signal stats (if journaled) ==="
grep -E "LIVE SIGNAL|retrain complete" <(journalctl --user -u eod-retrain.service --no-pager -n 200 2>/dev/null) | tail -3
