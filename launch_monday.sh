#!/bin/bash
# launch_monday.sh — Start ticker + engine for Monday trading
# Run this when gold market opens (Sun 5pm EST / Mon 3am IST)

echo "═══ LAUNCHING AI TRADING SYSTEM ═══"
echo ""

# Kill any existing processes
echo "Cleaning up old processes..."
pkill -f "xm_ticker" 2>/dev/null
pkill -f "ensemble_signal_engine" 2>/dev/null
sleep 2

# Check market hours
HOUR_UTC=$(TZ=UTC date +%H)
DOW=$(TZ=UTC date +%u)  # 1=Mon, 7=Sun
echo "UTC hour: $HOUR_UTC, Day: $DOW"

if [ "$DOW" -eq 6 ]; then
    echo "❌ Saturday — market closed. Exiting."
    exit 1
fi
if [ "$DOW" -eq 7 ] && [ "$HOUR_UTC" -lt 22 ]; then
    echo "❌ Sunday before 10pm UTC — market closed. Exiting."
    echo "   Market opens at 22:00 UTC (5pm EST / 3am IST)"
    exit 1
fi
if [ "$DOW" -eq 5 ] && [ "$HOUR_UTC" -ge 21 ]; then
    echo "❌ Friday after 9pm UTC — market closed. Exiting."
    exit 1
fi

echo "✅ Market is OPEN"
echo ""

# Start ticker
echo "Starting XM ticker (Wine/MT5)..."
cd /home/jith/.hermes/profiles/trading/scripts
wine python.exe xm_ticker.py &
TICKER_PID=$!
echo "  Ticker PID: $TICKER_PID"

# Wait for ticker to initialize
echo "  Waiting 20s for ticker to connect..."
sleep 20

# Check ticker is running
if kill -0 $TICKER_PID 2>/dev/null; then
    echo "  ✅ Ticker running"
else
    echo "  ❌ Ticker failed to start!"
    exit 1
fi

# Start engine
echo ""
echo "Starting AI signal engine..."
/home/jith/.hermes/hermes-agent/venv/bin/python3 -u ensemble_signal_engine_v2.py &
ENGINE_PID=$!
echo "  Engine PID: $ENGINE_PID"

# Wait and verify
sleep 5
if kill -0 $ENGINE_PID 2>/dev/null; then
    echo "  ✅ Engine running"
else
    echo "  ❌ Engine failed to start!"
    kill $TICKER_PID 2>/dev/null
    exit 1
fi

echo ""
echo "═══ SYSTEM LIVE ═══"
echo "  Ticker: PID $TICKER_PID"
echo "  Engine: PID $ENGINE_PID"
echo "  Signals: @goldrigging_bot"
echo "  Stop: kill $TICKER_PID $ENGINE_PID"
echo ""
echo "Monitor with:"
echo "  tail -f ensemble_trade_journal.jsonl"
echo "  ps aux | grep -E 'ticker|engine'"
