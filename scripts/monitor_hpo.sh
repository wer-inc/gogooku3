#!/bin/bash
# HPO Progress Monitoring Script

HPO_LOG="${1:-/tmp/hpo_production.log}"
OUTPUT_DIR="${2:-output/hpo_production}"

echo "==================================================="
echo "HPO Progress Monitor"
echo "==================================================="
echo "Log file: $HPO_LOG"
echo "Output dir: $OUTPUT_DIR"
echo ""

# Check if HPO is running
HPO_PID=$(ps aux | grep "run_optuna_atft.py" | grep -v grep | awk '{print $2}' | head -1)
if [ -z "$HPO_PID" ]; then
    echo "❌ No HPO process found"
else
    echo "✅ HPO running (PID: $HPO_PID)"
    ps -p $HPO_PID -o pid,%cpu,%mem,etime | tail -1
    echo ""
fi

# Show latest trial info
echo "---------------------------------------------------"
echo "Latest Trial Info:"
echo "---------------------------------------------------"
if [ -f "$HPO_LOG" ]; then
    tail -50 "$HPO_LOG" | grep -E "Trial [0-9]+:" | tail -5
else
    echo "Log file not found"
fi

echo ""
echo "---------------------------------------------------"
echo "Completed Trials:"
echo "---------------------------------------------------"
if [ -f "$OUTPUT_DIR/all_trials.json" ]; then
    python3 -c "
import json
with open('$OUTPUT_DIR/all_trials.json') as f:
    trials = json.load(f)
print(f'Total trials: {len(trials)}')
completed = [t for t in trials if t['value'] is not None]
print(f'Completed: {len(completed)}')
if completed:
    best = max(completed, key=lambda t: t['value'] if t['value'] else -999)
    print(f\"Best Sharpe: {best['value']:.4f} (Trial {best['number']})\")
    print(f\"  Params: lr={best['params']['lr']:.2e}, batch={best['params']['batch_size']}, hidden={best['params']['hidden_size']}\")
" 2>/dev/null || echo "No trials completed yet"
else
    echo "No results file yet"
fi

echo ""
echo "---------------------------------------------------"
echo "Recent Log Entries:"
echo "---------------------------------------------------"
if [ -f "$HPO_LOG" ]; then
    tail -15 "$HPO_LOG"
else
    echo "Log file not found"
fi

echo ""
echo "==================================================="
echo "Use: watch -n 30 $0 $HPO_LOG $OUTPUT_DIR"
echo "==================================================="
