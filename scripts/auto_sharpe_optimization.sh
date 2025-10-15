#!/bin/bash
# Automated Sharpe Optimization Pipeline
# One-shot execution: Parallel sweep → Auto-selection → Production training → Backtest
#
# Estimated timeline:
# - Parallel sweep (5 epochs × 36 configs / 4 parallel): 45-60 min
# - Evaluation: 2-3 min
# - Production training (80 epochs): 5-7 hours
# - Backtest: 5-10 min
# Total: ~6-8 hours

set -e

# Configuration
SWEEP_DIR="output/sweep_results"
PROD_DIR="output/production_training"
BACKTEST_DIR="output/backtest_production"

# Execution flags
RUN_SWEEP=1
RUN_EVAL=1
RUN_TRAINING=1
RUN_BACKTEST=1
SKIP_CONFIRMATION=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-sweep)
            RUN_SWEEP=0
            shift
            ;;
        --skip-training)
            RUN_TRAINING=0
            shift
            ;;
        --skip-backtest)
            RUN_BACKTEST=0
            shift
            ;;
        --yes|-y)
            SKIP_CONFIRMATION=1
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --skip-sweep       Skip parallel sweep (use existing results)"
            echo "  --skip-training    Skip production training (only run sweep)"
            echo "  --skip-backtest    Skip backtest evaluation"
            echo "  --yes, -y          Skip confirmation prompts"
            echo "  --help, -h         Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "================================================================================"
echo "AUTOMATED SHARPE OPTIMIZATION PIPELINE"
echo "================================================================================"
echo "This pipeline will:"
echo "  1. Run parallel stability sweep (5 epochs × multiple configs)"
echo "  2. Automatically select top configuration(s)"
echo "  3. Run production training (80 epochs with 2-stage optimization)"
echo "  4. Evaluate with transaction-cost backtest"
echo ""
echo "Execution plan:"
echo "  Run parallel sweep:     $([ $RUN_SWEEP -eq 1 ] && echo 'YES' || echo 'NO (using existing)')"
echo "  Run production training: $([ $RUN_TRAINING -eq 1 ] && echo 'YES' || echo 'NO')"
echo "  Run backtest:           $([ $RUN_BACKTEST -eq 1 ] && echo 'YES' || echo 'NO')"
echo ""
echo "Estimated time: 6-8 hours (depends on hardware)"
echo "================================================================================"

if [ $SKIP_CONFIRMATION -eq 0 ]; then
    read -p "Continue? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# Record start time
start_time=$(date +%s)
pipeline_log="$PROD_DIR/pipeline_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$PROD_DIR"

echo "Pipeline log: $pipeline_log"
echo ""

# ============================================================================
# STEP 1: Parallel Stability Sweep
# ============================================================================
if [ $RUN_SWEEP -eq 1 ]; then
    echo "================================================================================"
    echo "STEP 1: Parallel Stability Sweep"
    echo "================================================================================"
    echo "Running parallel sweep to find variance-stable configurations..."
    echo "This will test ~36 hyperparameter combinations in parallel (4 at a time)"
    echo "Estimated time: 45-60 minutes"
    echo "================================================================================"

    bash scripts/parallel_sweep.sh 2>&1 | tee -a "$pipeline_log"

    if [ $? -ne 0 ]; then
        echo "❌ Parallel sweep failed!"
        exit 1
    fi

    echo "✅ Parallel sweep completed"
else
    echo "⏭️  Skipping parallel sweep (using existing results)"
fi

# ============================================================================
# STEP 2: Evaluate and Select Best Configuration
# ============================================================================
if [ $RUN_EVAL -eq 1 ]; then
    echo ""
    echo "================================================================================"
    echo "STEP 2: Evaluate and Select Best Configuration"
    echo "================================================================================"
    echo "Analyzing sweep results and ranking by composite score..."
    echo "Gate criteria:"
    echo "  - pred_std > 0.010 (no variance collapse)"
    echo "  - Val Sharpe > -0.01"
    echo "  - Val RankIC > 0.02"
    echo "================================================================================"

    python scripts/evaluate_sweep_results.py --sweep-dir "$SWEEP_DIR" 2>&1 | tee -a "$pipeline_log"

    if [ $? -ne 0 ]; then
        echo "❌ Evaluation failed!"
        exit 1
    fi

    echo "✅ Evaluation completed"

    # Check if top configs were found
    if [ ! -f "$SWEEP_DIR/top_config_ids.txt" ]; then
        echo "❌ No top configurations found! Check sweep results."
        exit 1
    fi

    best_config=$(head -n 1 "$SWEEP_DIR/top_config_ids.txt")
    echo "🏆 Best configuration: $best_config"
else
    echo "⏭️  Skipping evaluation"
fi

# ============================================================================
# STEP 3: Production Training with Best Configuration
# ============================================================================
if [ $RUN_TRAINING -eq 1 ]; then
    echo ""
    echo "================================================================================"
    echo "STEP 3: Production Training (80 epochs)"
    echo "================================================================================"
    echo "Running 2-stage training with best configuration:"
    echo "  Stage 1 (5 epochs):  Variance bootstrap"
    echo "  Stage 2 (75 epochs): Sharpe optimization with SWA/Snapshot Ensemble"
    echo "Estimated time: 5-7 hours"
    echo "================================================================================"

    if [ $SKIP_CONFIRMATION -eq 0 ]; then
        read -p "Start production training? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Skipping production training."
            RUN_TRAINING=0
        fi
    fi

    if [ $RUN_TRAINING -eq 1 ]; then
        bash scripts/run_best_config.sh "$SWEEP_DIR" 2>&1 | tee -a "$pipeline_log"

        if [ $? -ne 0 ]; then
            echo "❌ Production training failed!"
            exit 1
        fi

        echo "✅ Production training completed"
    fi
else
    echo "⏭️  Skipping production training"
fi

# ============================================================================
# STEP 4: Backtest Evaluation
# ============================================================================
if [ $RUN_BACKTEST -eq 1 ]; then
    echo ""
    echo "================================================================================"
    echo "STEP 4: Backtest Evaluation"
    echo "================================================================================"
    echo "Running realistic backtest with transaction costs..."
    echo "Estimated time: 5-10 minutes"
    echo "================================================================================"

    # Find best checkpoint
    best_checkpoint=$(ls -t output/checkpoints/*.pth 2>/dev/null | head -n 1)

    if [ -z "$best_checkpoint" ]; then
        echo "⚠️  No checkpoint found, skipping backtest"
    else
        echo "Using checkpoint: $best_checkpoint"

        python scripts/backtest_sharpe_model.py \
            --checkpoint "$best_checkpoint" \
            --data-path output/ml_dataset_latest_full.parquet \
            --output-dir "$BACKTEST_DIR" \
            2>&1 | tee -a "$pipeline_log"

        if [ $? -eq 0 ]; then
            echo "✅ Backtest completed"
            echo "Results: $BACKTEST_DIR/"
        else
            echo "⚠️  Backtest failed (non-critical)"
        fi
    fi
else
    echo "⏭️  Skipping backtest"
fi

# ============================================================================
# Pipeline Summary
# ============================================================================
end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(((duration % 3600) / 60))

echo ""
echo "================================================================================"
echo "PIPELINE COMPLETED"
echo "================================================================================"
echo "Total time: ${hours}h ${minutes}m"
echo ""
echo "Results:"
echo "  Sweep results:    $SWEEP_DIR/"
echo "  Training logs:    $PROD_DIR/logs/"
echo "  Checkpoints:      output/checkpoints/"
echo "  Backtest results: $BACKTEST_DIR/"
echo "  Pipeline log:     $pipeline_log"
echo ""
echo "Key metrics:"

# Extract final metrics if available
stage2_log="$PROD_DIR/logs/stage2_sharpe_optimization.log"
if [ -f "$stage2_log" ]; then
    final_sharpe=$(grep "Val Metrics.*Sharpe:" "$stage2_log" | tail -1 | grep -oP 'Sharpe:\s*\K[-\d.]+' || echo "N/A")
    final_ic=$(grep "Val Metrics.*IC:" "$stage2_log" | tail -1 | grep -oP 'IC:\s*\K[-\d.]+' | head -1 || echo "N/A")
    final_rankic=$(grep "Val Metrics.*RankIC:" "$stage2_log" | tail -1 | grep -oP 'RankIC:\s*\K[-\d.]+' || echo "N/A")

    echo "  Final Val Sharpe:  $final_sharpe"
    echo "  Final Val IC:      $final_ic"
    echo "  Final Val RankIC:  $final_rankic"
fi

# Extract backtest metrics if available
backtest_summary="$BACKTEST_DIR/summary_metrics.csv"
if [ -f "$backtest_summary" ]; then
    echo ""
    echo "Backtest results (with transaction costs):"
    python3 -c "
import pandas as pd
df = pd.read_csv('$backtest_summary')
for col in ['sharpe_ratio', 'annual_return', 'max_drawdown', 'transaction_cost_drag']:
    if col in df.columns:
        print(f'  {col:25s}: {df[col].iloc[0]:10.4f}')
" 2>/dev/null || echo "  (Parse error)"
fi

echo ""
echo "================================================================================"
echo "✅ All steps completed successfully!"
echo "================================================================================"

# Generate quick evaluation command
echo ""
echo "Quick evaluation commands:"
echo "  # View training summary"
echo "  python scripts/evaluate_trained_model.py --log-file $stage2_log"
echo ""
echo "  # Compare with baseline"
echo "  echo 'Baseline Sharpe: -0.0184, Target: 0.35+'"
echo ""
echo "  # View backtest details"
echo "  cat $BACKTEST_DIR/summary_metrics.csv"
echo ""
