#!/bin/bash
# Monitor overfitting prevention validation test progress

LOG_FILE=$(ls -t _logs/overfitting_test_*.log 2>/dev/null | head -1)

if [ -z "$LOG_FILE" ]; then
    echo "❌ No overfitting test log found"
    exit 1
fi

echo "📊 Overfitting Prevention Validation Test Status"
echo "================================================"
echo "Log file: $LOG_FILE"
echo ""

# Check if process is running
if [ -f _logs/overfitting_test.pid ]; then
    PID=$(cat _logs/overfitting_test.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ Process running (PID: $PID)"
    else
        echo "⚠️  Process not found (may have completed)"
    fi
fi

echo ""
echo "📈 Latest Metrics:"
echo "=================="

# Extract latest epoch metrics
grep -E "Epoch [0-9]+/[0-9]+:" "$LOG_FILE" | tail -5

echo ""
echo "🎯 RankIC Progress:"
echo "==================="

# Extract RankIC values from validation
grep -E "Val.*RankIC:" "$LOG_FILE" | tail -10

echo ""
echo "📋 Recent Log (last 20 lines):"
echo "=============================="
tail -20 "$LOG_FILE"
