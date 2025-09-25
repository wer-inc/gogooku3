#!/bin/bash
#
# Run improved ATFT-GAT-FAN training with all performance optimizations
#
# Usage: ./run_improved_training.sh [options]
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🚀 PERFORMANCE-IMPROVED TRAINING${NC}"
echo -e "${BLUE}========================================${NC}"

# Parse arguments
VALIDATE_ONLY=0
SKIP_VALIDATION=0
USE_COMPILE=1

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --validate-only) VALIDATE_ONLY=1 ;;
        --skip-validation) SKIP_VALIDATION=1 ;;
        --no-compile) USE_COMPILE=0 ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --validate-only    Only run validation smoke test"
            echo "  --skip-validation  Skip validation and run full training"
            echo "  --no-compile       Disable torch.compile optimization"
            echo "  -h, --help         Show this help message"
            exit 0
            ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Set up improved environment variables
echo -e "\n${GREEN}📝 Setting up improved environment...${NC}"

# Critical performance improvements
export ALLOW_UNSAFE_DATALOADER=1
export NUM_WORKERS=8
export PERSISTENT_WORKERS=1
export PREFETCH_FACTOR=4
export PIN_MEMORY=1

# Loss function optimization
export USE_CS_IC=1
export CS_IC_WEIGHT=0.2
export USE_RANKIC=1
export SHARPE_WEIGHT=0.3

# PyTorch optimizations
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_BACKENDS_CUDNN_BENCHMARK=1
export TF32_ENABLED=1

# torch.compile settings
if [ "$USE_COMPILE" -eq 1 ]; then
    export ENABLE_TORCH_COMPILE=1
    export TORCH_COMPILE_MODE="max-autotune"
else
    export ENABLE_TORCH_COMPILE=0
fi

# Training settings
export USE_SWA=1
export SNAPSHOT_ENS=1

echo -e "${GREEN}✅ Environment configured${NC}"

# Display key settings
echo -e "\n${YELLOW}Key improvements enabled:${NC}"
echo "  • Multi-worker DataLoader: NUM_WORKERS=$NUM_WORKERS"
echo "  • Model capacity: hidden_size=256 (from 64)"
echo "  • IC/RankIC optimization: CS_IC_WEIGHT=$CS_IC_WEIGHT"
echo "  • torch.compile: $([[ $USE_COMPILE -eq 1 ]] && echo "ENABLED" || echo "DISABLED")"
echo "  • Learning rate scheduler: Plateau (adaptive)"

# Run validation if requested
if [ "$VALIDATE_ONLY" -eq 1 ]; then
    echo -e "\n${YELLOW}🔍 Running validation only...${NC}"
    python scripts/train_with_improvements.py --validate-only
    exit $?
fi

# Run validation unless skipped
if [ "$SKIP_VALIDATION" -eq 0 ]; then
    echo -e "\n${YELLOW}🔍 Running pre-training validation...${NC}"
    if ! python scripts/train_with_improvements.py --validate-only; then
        echo -e "${RED}⚠️ Validation failed${NC}"
        read -p "Continue anyway? [y/N]: " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

# Check for data availability
echo -e "\n${YELLOW}📊 Checking data availability...${NC}"
if [ -f "output/ml_dataset_latest_full.parquet" ]; then
    echo -e "${GREEN}✅ Dataset found: output/ml_dataset_latest_full.parquet${NC}"
    DATA_PATH="output/ml_dataset_latest_full.parquet"
elif [ -f "output/batch/ml_dataset_full.parquet" ]; then
    echo -e "${GREEN}✅ Dataset found: output/batch/ml_dataset_full.parquet${NC}"
    DATA_PATH="output/batch/ml_dataset_full.parquet"
else
    echo -e "${RED}❌ No dataset found. Please run data pipeline first.${NC}"
    echo "Run: make dataset-full START=2020-09-06 END=2025-09-06"
    exit 1
fi

# Run the improved training
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}🚀 Starting improved training...${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Choose the appropriate training script
if [ "$USE_COMPILE" -eq 1 ] && python -c "import torch; exit(0 if hasattr(torch, 'compile') else 1)" 2>/dev/null; then
    echo -e "${GREEN}Using torch.compile enhanced training${NC}"
    TRAIN_SCRIPT="scripts/train_with_torch_compile.py"
else
    echo -e "${YELLOW}Using standard training (torch.compile not available)${NC}"
    TRAIN_SCRIPT="scripts/train_atft.py"
fi

# Build the training command
CMD="python $TRAIN_SCRIPT"
CMD="$CMD --config-path configs/atft"
CMD="$CMD --config-name config"

# Add Hydra overrides for improvements
CMD="$CMD model=atft_gat_fan"
CMD="$CMD train=production_improved"
CMD="$CMD data.path=$DATA_PATH"
CMD="$CMD model.hidden_size=256"
CMD="$CMD train.batch.num_workers=$NUM_WORKERS"
CMD="$CMD train.scheduler.name=plateau"

# Log the command
echo -e "${YELLOW}Command:${NC}"
echo "$CMD"
echo

# Create log directory
LOG_DIR="logs/improved_training"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"

echo -e "${GREEN}Logging to: $LOG_FILE${NC}\n"

# Run training
if $CMD 2>&1 | tee "$LOG_FILE"; then
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}✅ Training completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"

    # Display summary if available
    if grep -q "val/rank_ic" "$LOG_FILE"; then
        echo -e "\n${YELLOW}Performance summary:${NC}"
        tail -n 50 "$LOG_FILE" | grep -E "(rank_ic|sharpe|loss)" | tail -n 10
    fi

    exit 0
else
    echo -e "\n${RED}========================================${NC}"
    echo -e "${RED}❌ Training failed${NC}"
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}Check log file: $LOG_FILE${NC}"
    exit 1
fi