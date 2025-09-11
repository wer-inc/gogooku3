#!/bin/bash

echo "============================================================"
echo "ATFT-GAT-FAN Automatic Hyperparameter Optimization"
echo "Phase-based optimization for maximum Sharpe Ratio"
echo "============================================================"

# 設定
DATA_PATH=${1:-"output/atft_data/train"}
TRIALS_PER_PHASE=${2:-10}
TRIAL_EPOCHS=${3:-5}
MAX_FILES=${4:-100}

echo "Configuration:"
echo "  Data Path: $DATA_PATH"
echo "  Trials per Phase: $TRIALS_PER_PHASE"
echo "  Trial Epochs: $TRIAL_EPOCHS"
echo "  Max Data Files: $MAX_FILES"
echo ""

# MLflow準備
export MLFLOW=1
export MLFLOW_TRACKING_URI="http://localhost:5000"
export MLFLOW_EXPERIMENT="ATFT-GAT-FAN-Optuna"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Phase 1: Basic Parameters
echo "🚀 Phase 1: Basic Parameter Optimization"
echo "  Optimizing: learning rate, batch size, weight decay, dropout"
python scripts/hyperparameter_tuning_real.py \
    --data-path "$DATA_PATH" \
    --phase 1 \
    --trials $TRIALS_PER_PHASE \
    --epochs $TRIAL_EPOCHS \
    --max-files $MAX_FILES

if [ $? -ne 0 ]; then
    echo "❌ Phase 1 failed"
    exit 1
fi

echo "✅ Phase 1 completed"
echo ""

# Phase 2: Graph Parameters  
echo "🚀 Phase 2: Graph Parameter Optimization"
echo "  Optimizing: k, edge_threshold, ewm_halflife, shrinkage_gamma"
python scripts/hyperparameter_tuning_real.py \
    --data-path "$DATA_PATH" \
    --phase 2 \
    --trials $TRIALS_PER_PHASE \
    --epochs $TRIAL_EPOCHS \
    --max-files $MAX_FILES

if [ $? -ne 0 ]; then
    echo "❌ Phase 2 failed"
    exit 1
fi

echo "✅ Phase 2 completed"
echo ""

# Phase 3: FAN/TFT Fusion
echo "🚀 Phase 3: FAN/TFT Fusion Optimization"  
echo "  Optimizing: gat_alpha, freq_dropout, edge_dropout, ema_decay"
python scripts/hyperparameter_tuning_real.py \
    --data-path "$DATA_PATH" \
    --phase 3 \
    --trials $TRIALS_PER_PHASE \
    --epochs $TRIAL_EPOCHS \
    --max-files $MAX_FILES

if [ $? -ne 0 ]; then
    echo "❌ Phase 3 failed"  
    exit 1
fi

echo "✅ Phase 3 completed"
echo ""

# 統合最適化
echo "🎯 Creating integrated optimized configuration..."
python scripts/hyperparameter_tuning_real.py \
    --data-path "$DATA_PATH" \
    --all-phases \
    --trials 0  # 実行せず結果統合のみ

echo ""
echo "============================================================"
echo "✅ ATFT-GAT-FAN Hyperparameter Optimization Completed!"
echo "============================================================"
echo "Next steps:"
echo "1. Check tuning_results/optimized_config.yaml"  
echo "2. Run full training with optimized parameters:"
echo "   python scripts/train_atft.py --config-path tuning_results --config-name optimized_config"
echo "3. Compare results with baseline in MLflow dashboard"
echo "============================================================"