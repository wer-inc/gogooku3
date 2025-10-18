#!/usr/bin/env bash
# ============================================================================
# Phase 1 Loss Weight Validation Script
# ============================================================================
#
# Purpose: Validate the effectiveness of Phase 1 loss weight optimization
# Duration: ~30-60 minutes (10 epochs)
# Expected improvement: RankIC 0.0014 → 0.020+
#
# ============================================================================

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
EPOCHS=10
BATCH_SIZE=2048
LR=2e-4
HIDDEN_SIZE=256
DATA_PATH="output/ml_dataset_latest_full.parquet"
LOG_DIR="_logs/phase1_validation"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/validate_${TIMESTAMP}.log"

# Create log directory
mkdir -p "$LOG_DIR"

# ============================================================================
# Functions
# ============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

print_header() {
    echo "" | tee -a "$LOG_FILE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG_FILE"
    echo "$1" | tee -a "$LOG_FILE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
}

# ============================================================================
# Pre-flight Checks
# ============================================================================

print_header "🔍 Phase 1 Validation: Loss Weight Optimization"

log_info "Timestamp: $TIMESTAMP"
log_info "Log file: $LOG_FILE"
log_info "Configuration:"
log_info "  - Epochs: $EPOCHS"
log_info "  - Batch size: $BATCH_SIZE"
log_info "  - Learning rate: $LR"
log_info "  - Hidden size: $HIDDEN_SIZE"
log_info "  - Dataset: $DATA_PATH"

# Check dataset exists
if [ ! -f "$DATA_PATH" ]; then
    log_error "Dataset not found: $DATA_PATH"
    log_info "Run 'make dataset-bg' to generate dataset first"
    exit 1
fi

# Check GPU availability
if ! nvidia-smi &> /dev/null; then
    log_warning "nvidia-smi not available - GPU may not be detected"
else
    log_success "GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG_FILE"
fi

# ============================================================================
# Phase 1 Loss Weight Settings
# ============================================================================

print_header "⚙️  Phase 1 Loss Weight Configuration"

export USE_RANKIC=1
export RANKIC_WEIGHT=0.5
export CS_IC_WEIGHT=0.3
export SHARPE_WEIGHT=0.1
export VAL_DEBUG_LOGGING=0

log_info "Loss weights:"
log_info "  ✅ USE_RANKIC=1 (Enabled)"
log_info "  ✅ RANKIC_WEIGHT=0.5 (Maximum RankIC focus)"
log_info "  ✅ CS_IC_WEIGHT=0.3 (2x from default 0.15)"
log_info "  ✅ SHARPE_WEIGHT=0.1 (Reduced from 0.3)"
log_info "  ✅ VAL_DEBUG_LOGGING=0 (Performance)"

# ============================================================================
# Baseline Metrics Recording
# ============================================================================

print_header "📊 Baseline Metrics (Current State)"

log_info "Expected baseline (from ISSUE.md):"
log_info "  - Val RankIC: 0.0014"
log_info "  - Val IC: 0.0082"
log_info "  - Val Sharpe: -0.007"

# ============================================================================
# Run Training
# ============================================================================

print_header "🚀 Starting Phase 1 Validation Training"

log_info "Running $EPOCHS epochs with optimized loss weights..."
log_info "This will take approximately 30-60 minutes"

# Run training with Phase 1 configuration
python scripts/train.py \
    --data-path "$DATA_PATH" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --hidden-size "$HIDDEN_SIZE" \
    --mode optimized \
    --no-background \
    2>&1 | tee -a "$LOG_FILE"

TRAIN_EXIT_CODE=$?

# ============================================================================
# Results Analysis
# ============================================================================

print_header "📈 Phase 1 Validation Results"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    log_success "Training completed successfully"

    # Extract final metrics from log
    FINAL_RANKIC=$(grep -oP "Val.*RankIC.*:\s*\K[0-9.-]+" "$LOG_FILE" | tail -1 || echo "N/A")
    FINAL_IC=$(grep -oP "Val.*IC.*:\s*\K[0-9.-]+" "$LOG_FILE" | tail -1 || echo "N/A")
    FINAL_SHARPE=$(grep -oP "Val.*Sharpe.*:\s*\K[0-9.-]+" "$LOG_FILE" | tail -1 || echo "N/A")

    log_info "Final Metrics:"
    log_info "  Val RankIC: $FINAL_RANKIC (target: >0.020)"
    log_info "  Val IC: $FINAL_IC (target: >0.015)"
    log_info "  Val Sharpe: $FINAL_SHARPE (target: >0)"

    # Success criteria evaluation
    SUCCESS=true

    if [ "$FINAL_RANKIC" != "N/A" ] && (( $(echo "$FINAL_RANKIC > 0.020" | bc -l) )); then
        log_success "✅ RankIC target achieved: $FINAL_RANKIC > 0.020"
    else
        log_warning "⚠️  RankIC below target: $FINAL_RANKIC <= 0.020"
        SUCCESS=false
    fi

    if [ "$FINAL_IC" != "N/A" ] && (( $(echo "$FINAL_IC > 0.015" | bc -l) )); then
        log_success "✅ IC target achieved: $FINAL_IC > 0.015"
    else
        log_warning "⚠️  IC below target: $FINAL_IC <= 0.015"
        SUCCESS=false
    fi

    if [ "$FINAL_SHARPE" != "N/A" ] && (( $(echo "$FINAL_SHARPE > 0" | bc -l) )); then
        log_success "✅ Sharpe target achieved: $FINAL_SHARPE > 0"
    else
        log_warning "⚠️  Sharpe below target: $FINAL_SHARPE <= 0"
        SUCCESS=false
    fi

    # ========================================================================
    # Next Steps Recommendation
    # ========================================================================

    print_header "🎯 Next Steps"

    if [ "$SUCCESS" = true ]; then
        log_success "🎉 Phase 1 PASSED - All targets achieved!"
        log_info "Recommendations:"
        log_info "  1. Proceed to Phase 2 (Feature enhancement + GAT)"
        log_info "  2. Run longer training (20 epochs) for stability confirmation"
        log_info "  3. Document Phase 1 results in PERFORMANCE_IMPROVEMENT_REPORT.md"
        log_info ""
        log_info "Commands:"
        log_info "  make train EPOCHS=20  # Longer validation"
        log_info "  # Then proceed to Phase 2 implementation"
    else
        log_warning "⚠️  Phase 1 targets not fully achieved"
        log_info "Recommendations:"
        log_info "  1. Review training logs for issues"
        log_info "  2. Try alternative weight configuration:"
        log_info "     - RANKIC_WEIGHT=0.3 (reduced from 0.5)"
        log_info "     - CS_IC_WEIGHT=0.25 (reduced from 0.3)"
        log_info "  3. Extend training to 15 epochs"
        log_info "  4. Check for data leakage or normalization issues"
        log_info ""
        log_info "Commands:"
        log_info "  RANKIC_WEIGHT=0.3 CS_IC_WEIGHT=0.25 make train EPOCHS=15"
    fi

else
    log_error "Training failed with exit code $TRAIN_EXIT_CODE"
    log_info "Check log file for details: $LOG_FILE"
    exit $TRAIN_EXIT_CODE
fi

# ============================================================================
# Summary Report
# ============================================================================

print_header "📝 Phase 1 Validation Summary"

cat >> "$LOG_DIR/phase1_summary_${TIMESTAMP}.md" << EOF
# Phase 1 Validation Summary

**Date**: $(date +%Y-%m-%d\ %H:%M:%S)
**Duration**: $EPOCHS epochs
**Log**: $LOG_FILE

## Configuration

- RANKIC_WEIGHT: 0.5
- CS_IC_WEIGHT: 0.3
- SHARPE_WEIGHT: 0.1
- Batch size: $BATCH_SIZE
- Learning rate: $LR

## Results

| Metric | Baseline | Target | Achieved | Status |
|--------|----------|--------|----------|--------|
| Val RankIC | 0.0014 | >0.020 | $FINAL_RANKIC | $([ "$SUCCESS" = true ] && echo "✅ PASS" || echo "⚠️ REVIEW") |
| Val IC | 0.0082 | >0.015 | $FINAL_IC | - |
| Val Sharpe | -0.007 | >0 | $FINAL_SHARPE | - |

## Next Steps

$(if [ "$SUCCESS" = true ]; then
    echo "- ✅ Proceed to Phase 2 (Feature + GAT enhancement)"
    echo "- Run longer training (20 epochs) for confirmation"
else
    echo "- ⚠️ Review and adjust loss weights"
    echo "- Extend training duration"
    echo "- Check for data/normalization issues"
fi)

---
*Generated by validate_loss_weights.sh*
EOF

log_success "Summary report saved: $LOG_DIR/phase1_summary_${TIMESTAMP}.md"

print_header "✅ Phase 1 Validation Complete"
log_info "Review full log: $LOG_FILE"
log_info "Review summary: $LOG_DIR/phase1_summary_${TIMESTAMP}.md"
