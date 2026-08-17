#!/bin/bash
# auto_retrain.sh — Monitor matrix build, then auto-launch retrain
# Runs in background. Output goes to /tmp/auto_retrain.log
set -euo pipefail
LOG="/tmp/auto_retrain.log"
SCRIPTS="/home/jith/.hermes/profiles/trading/scripts"
PY="/home/jith/.hermes/hermes-agent/venv/bin/python3"

exec > >(tee -a "$LOG") 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] auto_retrain STARTED"

# Wait for matrix build to complete
TICK_CSV="$SCRIPTS/gold_features_m5_tick.csv"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for tick matrix build to finish..."
while true; do
    # Check if build_tick_matrix.py is still running
    if ! pgrep -f "build_tick_matrix.py" > /dev/null 2>&1; then
        if [ -f "$TICK_CSV" ]; then
            ROWS=$(wc -l < "$TICK_CSV")
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Matrix build COMPLETE: $ROWS rows"
            break
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Build stopped but no output file"
            exit 1
        fi
    fi
    sleep 30
done

# Verify matrix has expected columns
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Verifying matrix columns..."
$PY -c "
import pandas as pd
df = pd.read_csv('$TICK_CSV', nrows=0)
print(f'Columns: {len(df.columns)}')
print('Tick cols present:', all(c in df.columns for c in ['imb_300s','vol_rel','cvd']))
print('Macro cols present:', all(c in df.columns for c in ['dxy_z','tnx_level','gc_5d_chg']))
"

# Launch retrain
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching retrain pipeline..."
cd "$SCRIPTS"
$PY -u retrain_m5.py 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ═══ RETRAIN COMPLETE ═══"
