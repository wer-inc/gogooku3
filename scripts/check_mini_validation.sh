#!/bin/bash
# Check mini validation progress

echo "📊 Mini Validation Progress"
echo "============================"
echo ""

# Check if process is running
if [ -f _logs/mini_validation.pid ]; then
    PID=$(cat _logs/mini_validation.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ Process running (PID: $PID)"
        ps -p $PID -o pid,stat,etime,%cpu,%mem --no-headers | awk '{printf "   Status: %s | Runtime: %s | CPU: %s%% | Memory: %s%%\n", $2, $3, $4, $5}'
    else
        echo "❌ Process not running (completed or failed)"
        rm -f _logs/mini_validation.pid
    fi
else
    echo "❌ No PID file found"
fi

echo ""
echo "📄 Latest Log Output:"
echo "--------------------"
tail -20 _logs/mini_validation_*.log 2>/dev/null | tail -15

echo ""
echo "🔍 Key Metrics (if available):"
echo "------------------------------"
grep -E "Val RankIC|Val IC|Val Sharpe|Epoch [0-9]+/[0-9]+:" _logs/mini_validation_*.log 2>/dev/null | tail -10 || echo "   (No metrics yet)"

echo ""
