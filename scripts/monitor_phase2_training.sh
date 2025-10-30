#!/bin/bash
# Real-time Phase 2 Training Monitor

LOG_FILE="_logs/training/phase2_long_20251029_131446.log"

echo "=== Phase 2 Training Monitor ==="
echo "Log: $LOG_FILE"
echo

# Check if training is running
if [ -f "_logs/training/phase2_long.pid" ]; then
    PID=$(cat _logs/training/phase2_long.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ Training process running (PID: $PID)"
    else
        echo "❌ Training process not found"
    fi
else
    echo "⚠️  No PID file found"
fi

echo
echo "=== Latest Epoch Progress ==="
tail -100 "$LOG_FILE" | grep -E "Epoch [0-9]+/" | tail -5

echo
echo "=== GAT Gradient Health (last 10) ==="
grep "\[GAT-GRAD\] gradient norm:" "$LOG_FILE" | tail -10 | \
  awk '{print NR, $NF}' | column -t

echo
echo "=== GAT Gate Evolution ==="
grep "gate_value=" "$LOG_FILE" | grep -oE "gate_value=-?[0-9.e-]+" | tail -10

echo
echo "=== NaN Detection ==="
nan_count=$(grep "\[NaN-DEBUG\]" "$LOG_FILE" | wc -l)
if [ $nan_count -gt 0 ]; then
    echo "⚠️  $nan_count NaN events detected"
    grep "\[NaN-DEBUG\]" "$LOG_FILE" | tail -20
else
    echo "✅ No NaN events"
fi

echo
echo "=== Validation Metrics (last 5 epochs) ==="
grep -E "val.*rank_ic|val/loss" "$LOG_FILE" | tail -10

echo
echo "=== Training ETA ==="
start_time=$(stat -c %Y "$LOG_FILE" 2>/dev/null || stat -f %m "$LOG_FILE")
current_time=$(date +%s)
elapsed=$((current_time - start_time))
current_epoch=$(tail -100 "$LOG_FILE" | grep -oE "Epoch [0-9]+/" | tail -1 | grep -oE "[0-9]+")
if [ -n "$current_epoch" ] && [ $current_epoch -gt 0 ]; then
    time_per_epoch=$((elapsed / current_epoch))
    remaining_epochs=$((50 - current_epoch))
    eta_seconds=$((remaining_epochs * time_per_epoch))
    eta_hours=$((eta_seconds / 3600))
    eta_mins=$(( (eta_seconds % 3600) / 60 ))
    echo "Current: Epoch $current_epoch/50"
    echo "Time per epoch: ~$((time_per_epoch / 60)) minutes"
    echo "ETA: ~${eta_hours}h ${eta_mins}m"
else
    echo "Still initializing..."
fi
