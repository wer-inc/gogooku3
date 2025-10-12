#!/usr/bin/env bash
#
# Wrapper script to run dataset generation with memory monitoring
#
# Usage:
#   bash scripts/monitoring/watch_dataset.sh [dataset-command] [args...]
#
# Examples:
#   bash scripts/monitoring/watch_dataset.sh make dataset-gpu
#   bash scripts/monitoring/watch_dataset.sh python scripts/data/dataset_generator_safe.py --start-date 2020-01-01 --end-date 2025-01-01
#

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default thresholds
WARN_THRESHOLD=${MEMORY_THRESHOLD_WARN:-80}
STOP_THRESHOLD=${MEMORY_THRESHOLD_STOP:-90}
CHECK_INTERVAL=5

echo -e "${BLUE}=== Dataset Generation with Memory Monitoring ===${NC}"
echo "Command: $*"
echo "Memory thresholds: warn=${WARN_THRESHOLD}%, stop=${STOP_THRESHOLD}%"
echo ""

# Check if memory monitor is available
if [[ ! -f "scripts/monitoring/memory_monitor.py" ]]; then
    echo -e "${RED}Error: memory_monitor.py not found${NC}"
    exit 1
fi

# Start the command in background
echo -e "${BLUE}Starting dataset generation...${NC}"
"$@" &
DATASET_PID=$!

echo "Dataset generation PID: $DATASET_PID"
echo ""

# Start memory monitor
echo -e "${BLUE}Starting memory monitor...${NC}"
python scripts/monitoring/memory_monitor.py \
    --pid "$DATASET_PID" \
    --warn-threshold "$WARN_THRESHOLD" \
    --stop-threshold "$STOP_THRESHOLD" \
    --interval "$CHECK_INTERVAL" \
    --gpu &
MONITOR_PID=$!

# Wait for dataset generation to complete
wait "$DATASET_PID"
DATASET_EXIT=$?

# Stop monitor
kill "$MONITOR_PID" 2>/dev/null || true

echo ""
echo -e "${BLUE}=== Generation Complete ===${NC}"

if [[ $DATASET_EXIT -eq 0 ]]; then
    echo -e "${GREEN}✓ Dataset generation completed successfully${NC}"
else
    echo -e "${RED}✗ Dataset generation failed with exit code $DATASET_EXIT${NC}"
fi

exit $DATASET_EXIT
