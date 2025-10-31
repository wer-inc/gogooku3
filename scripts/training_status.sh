#!/bin/bash
# Quick training status snapshot
# Usage: ./scripts/training_status.sh

LOGFILE=$(ls -t _logs/training/prod_validation_50ep_*.log 2>/dev/null | head -1)
PID=$(cat _logs/training/prod_validation_50ep.pid 2>/dev/null || echo "")

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║          50-Epoch Training Status - Quick Snapshot           ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Process status
if [[ -n "$PID" ]] && ps -p "$PID" > /dev/null 2>&1; then
    echo "✅ Training ACTIVE (PID: $PID)"
    ps -p "$PID" -o pid,stat,%cpu,%mem,etime --no-headers | \
        awk '{printf "   CPU: %s%% | Memory: %s%% | Runtime: %s\n", $3, $4, $5}'
else
    echo "⏸️  Training STOPPED/COMPLETED"
fi

echo ""

# Latest epoch
LATEST_EPOCH=$(grep -E "Epoch [0-9]+/50" "$LOGFILE" 2>/dev/null | tail -1 | grep -oP "Epoch \K[0-9]+" || echo "0")
PROGRESS=$((LATEST_EPOCH * 100 / 50))
echo "📊 Progress: Epoch $LATEST_EPOCH/50 (${PROGRESS}%)"

# Degeneracy resets
RESET_COUNT=$(grep -c "DEGENERACY-GUARD.*reset applied" "$LOGFILE" 2>/dev/null || echo "0")
if [[ $RESET_COUNT -lt 5 ]]; then
    echo "✅ Degeneracy Resets: $RESET_COUNT (excellent)"
else
    echo "⚠️  Degeneracy Resets: $RESET_COUNT (monitor)"
fi

# Gradient warnings
GRAD_WARNINGS=$(grep -c "GRAD-MONITOR.*< 1e-07" "$LOGFILE" 2>/dev/null || echo "0")
if [[ $GRAD_WARNINGS -eq 0 ]]; then
    echo "✅ Gradient Health: No warnings"
else
    echo "⚠️  Gradient Warnings: $GRAD_WARNINGS"
fi

echo ""

# GPU status
echo "🖥️  GPU Status:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | \
    awk -F', ' '{printf "   Utilization: %s | Memory: %s / %s MiB\n", $1, $2, $3}'

echo ""

# Latest Sharpe (if available)
LATEST_SHARPE=$(grep "Achieved Sharpe Ratio:" "$LOGFILE" 2>/dev/null | tail -1 | grep -oP ": \K[0-9.]+" || echo "")
if [[ -n "$LATEST_SHARPE" ]]; then
    echo "📈 Latest Sharpe: $LATEST_SHARPE"
fi

echo ""
echo "Log: $LOGFILE"
echo ""
echo "Commands:"
echo "  Watch progress:  tail -f _logs/training/prod_validation_50ep_*.log | grep Epoch"
echo "  Check gradients: grep GRAD-MONITOR _logs/training/prod_validation_50ep_*.log | tail -10"
echo "  Monitor live:    ./scripts/monitor_training.sh"
