#!/bin/bash
# RFI-3: GPU Monitoring Setup
# Usage: scripts/rfi_3_gpu_monitor.sh [duration_seconds]

DURATION=${1:-3600}  # Default: 1 hour
OUTPUT_DIR="output/reports/diag_bundle"
OUTPUT_FILE="$OUTPUT_DIR/gpu_utilization.log"

mkdir -p "$OUTPUT_DIR"

echo "Starting GPU monitoring..."
echo "Duration: ${DURATION} seconds"
echo "Output: $OUTPUT_FILE"
echo ""
echo "Press Ctrl+C to stop early"
echo ""

# Calculate samples (1 sample per second)
SAMPLES=$((DURATION))

# Start monitoring
nvidia-smi dmon -s pucm -c $SAMPLES > "$OUTPUT_FILE" 2>&1 &
MONITOR_PID=$!

echo "GPU monitor started (PID: $MONITOR_PID)"
echo "Monitoring in background for ${DURATION}s..."
echo ""
echo "To check progress:"
echo "  tail -f $OUTPUT_FILE"
echo ""
echo "To stop early:"
echo "  kill $MONITOR_PID"
echo ""

# Wait for completion
wait $MONITOR_PID

echo "GPU monitoring complete: $OUTPUT_FILE"
