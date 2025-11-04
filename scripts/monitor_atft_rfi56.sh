#!/bin/bash
# Monitor ATFT training and extract RFI-56 metrics
# Monitors both running training processes

set -e

OUTPUT_FILE="rfi_56_metrics.txt"
LOG_DIR="_logs/training"

echo "==========================================="
echo "ATFT Training Monitor - RFI-56 Extraction"
echo "==========================================="
echo "Monitoring: ml_training.log and Hydra logs"
echo "Output: $OUTPUT_FILE"
echo "Press Ctrl+C to stop"
echo "==========================================="
echo ""

# Find relevant log files
LOG_FILES=(
    "$LOG_DIR/ml_training.log"
    "$LOG_DIR/2025-11-03/07-46-36/ATFT-GAT-FAN.log"
    "$LOG_DIR/2025-11-03/07-21-*/ATFT-GAT-FAN.log"
)

# Check which files exist
ACTIVE_LOGS=()
for log in "${LOG_FILES[@]}"; do
    # Expand wildcard
    for expanded in $log; do
        if [ -f "$expanded" ]; then
            ACTIVE_LOGS+=("$expanded")
            echo "✓ Found: $expanded"
        fi
    done
done

if [ ${#ACTIVE_LOGS[@]} -eq 0 ]; then
    echo "⚠️  No log files found!"
    echo "Expected locations:"
    for log in "${LOG_FILES[@]}"; do
        echo "  - $log"
    done
    exit 1
fi

echo ""
echo "Monitoring ${#ACTIVE_LOGS[@]} log file(s)..."
echo ""

# Create empty output file
> "$OUTPUT_FILE"

# Monitor logs with tail -F (follow even if file rotates)
tail -F "${ACTIVE_LOGS[@]}" 2>/dev/null | \
    grep --line-buffered "RFI56 |" | \
    tee -a "$OUTPUT_FILE" | \
    while IFS= read -r line; do
        echo "[$(date '+%H:%M:%S')] $line"

        # Extract key metrics for quick evaluation
        if echo "$line" | grep -q "yhat_std"; then
            yhat_std=$(echo "$line" | grep -oP 'yhat_std=\K[0-9.e+-]+')
            rank_ic=$(echo "$line" | grep -oP 'RankIC=\K[0-9.e+-]+')
            gat_gate=$(echo "$line" | grep -oP 'gat_gate_mean=\K[0-9.e+-]+')

            echo "   ⚡ Quick check: yhat_std=$yhat_std, RankIC=$rank_ic, gat_gate=$gat_gate"

            # Check RFI-5/6 thresholds
            if (( $(echo "$yhat_std > 1e-3" | bc -l) )) && \
               (( $(echo "$rank_ic > 0" | bc -l) )) && \
               (( $(echo "$gat_gate >= 0.2" | bc -l) )) && \
               (( $(echo "$gat_gate <= 0.7" | bc -l) )); then
                echo "   ✅ PASS: RFI-5/6 thresholds met!"
                echo "   → Ready for coefficient tuning"
            fi
        fi
    done
