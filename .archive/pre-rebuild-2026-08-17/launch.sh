#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# LAUNCH GOLD TRADING SYSTEM
# Renaissance-style quantitative gold trading AI
# ═══════════════════════════════════════════════════════════════════
set -e
BASE="/home/jith/.hermes/profiles/trading/scripts"
VENV="/home/jith/.hermes/hermes-agent/venv/bin"
PYTHON="$VENV/python3"
ENGINE="$BASE/ai_signal_engine.py"
TICKER="$BASE/xm_ticker.py"
LOGDIR="$BASE/logs"

mkdir -p "$LOGDIR"

echo "═══ GOLD TRADING SYSTEM LAUNCH ═══"
echo "Time: $(date)"
echo ""

# 1. Kill any existing processes
echo "1. Stopping existing processes..."
pkill -f "ai_signal_engine.py" 2>/dev/null || true
pkill -f "xm_ticker.py" 2>/dev/null || true
sleep 2
echo "   ✅ Existing processes stopped"

# 2. Check Wine/MT5
echo ""
echo "2. Checking Wine/MT5..."
if pgrep -f "mt5" > /dev/null 2>&1 || pgrep -f "terminal64" > /dev/null 2>&1; then
    echo "   ✅ MT5 is running"
else
    echo "   ⚠️ MT5 not running — starting ticker (will attempt connection)"
fi

# 3. Check model files
echo ""
echo "3. Checking model files..."
if [ -f "$BASE/models/gold_lgb_model_s42.txt" ] && \
   [ -f "$BASE/models/gold_lgb_model_s7.txt" ] && \
   [ -f "$BASE/models/gold_lgb_model_s2026.txt" ]; then
    echo "   ✅ All 3 model seeds present"
else
    echo "   ❌ Model files missing! Run retrain first."
    exit 1
fi

# 4. Check matrix
echo ""
echo "4. Checking matrix..."
if [ -f "$BASE/gold_features_m5_full.csv" ]; then
    SIZE=$(du -h "$BASE/gold_features_m5_full.csv" | cut -f1)
    echo "   ✅ Matrix: $SIZE"
else
    echo "   ❌ Matrix missing! Run retrain first."
    exit 1
fi

# 5. Start ticker (background)
echo ""
echo "5. Starting XM ticker..."
nohup $PYTHON "$TICKER" >> "$LOGDIR/ticker.log" 2>&1 &
TICKER_PID=$!
echo "   ✅ Ticker started (PID: $TICKER_PID)"

# Wait for ticker to initialize
sleep 5

# 6. Start engine (background)
echo ""
echo "6. Starting signal engine..."
nohup $PYTHON -u "$ENGINE" >> "$LOGDIR/engine.log" 2>&1 &
ENGINE_PID=$!
echo "   ✅ Engine started (PID: $ENGINE_PID)"

# 7. Verify both are running
echo ""
echo "7. Verifying processes..."
sleep 3
if kill -0 $TICKER_PID 2>/dev/null; then
    echo "   ✅ Ticker running (PID: $TICKER_PID)"
else
    echo "   ❌ Ticker died! Check $LOGDIR/ticker.log"
fi

if kill -0 $ENGINE_PID 2>/dev/null; then
    echo "   ✅ Engine running (PID: $ENGINE_PID)"
else
    echo "   ❌ Engine died! Check $LOGDIR/engine.log"
fi

echo ""
echo "═══ SYSTEM LAUNCHED ═══"
echo "Ticker PID: $TICKER_PID"
echo "Engine PID: $ENGINE_PID"
echo ""
echo "Logs:"
echo "  Ticker: $LOGDIR/ticker.log"
echo "  Engine: $LOGDIR/engine.log"
echo ""
echo "Monitoring:"
echo "  tail -f $LOGDIR/engine.log"
echo "  tail -f $LOGDIR/ticker.log"
echo ""
echo "To stop:"
echo "  kill $TICKER_PID $ENGINE_PID"
