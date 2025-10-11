#!/bin/bash
# GPU Training Script for Latest Dataset
# 最新データセットでGPU学習を実行するスクリプト

set -e

# GPU環境設定（永続化）
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export FORCE_GPU=1
# Ensure mini training path is disabled by default (can enable with FORCE_MINI_TRAIN=1)
export FORCE_MINI_TRAIN=${FORCE_MINI_TRAIN:-0}
# Require a real GPU by default; fail fast if not available
export REQUIRE_GPU=${REQUIRE_GPU:-1}
# Hint accelerator to downstream code
export ACCELERATOR=${ACCELERATOR:-gpu}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}
export TORCH_CUDNN_V8_API_ALLOWED=1
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_USE_RTLD_GLOBAL=YES
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-SYS}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-32}
export PYTHONUNBUFFERED=1
export WANDB_DISABLED=${WANDB_DISABLED:-1}
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_ENABLED=${WANDB_ENABLED:-0}
# Enable memory savers by default for stability on large feature sets
export GRAD_CHECKPOINT_VSN=${GRAD_CHECKPOINT_VSN:-1}

# カラー出力
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}🚀 GPU Training with Latest Dataset${NC}"

# 最新データセットを自動検出
LATEST_DATASET=$(ls -t output/datasets/ml_dataset_*_full.parquet 2>/dev/null | head -1)

if [ -z "$LATEST_DATASET" ]; then
    # output/にもチェック
    LATEST_DATASET=$(ls -t output/ml_dataset_*_full.parquet 2>/dev/null | head -1)
fi

if [ -z "$LATEST_DATASET" ]; then
    echo -e "${YELLOW}⚠️ No dataset found. Please run dataset generation first:${NC}"
    echo "make dataset-full-gpu START=2020-09-19 END=2025-09-19"
    exit 1
fi

echo -e "${GREEN}✓ Found latest dataset: $LATEST_DATASET${NC}"

# デフォルト引数（安全側の初期値に調整: OOM回避重視）
# 環境変数で上書き可能: TRAIN_BATCH_SIZE, TRAIN_VAL_BATCH_SIZE など
RUN_SAFE_PIPELINE=${1:-""}
ADV_GRAPH=${2:-"--adv-graph-train"}
LEARNING_RATE=${3:-"2e-4"}
MAX_EPOCHS=${4:-"75"}
# 大規模VSNでのピークVRAMと環境制約を考慮した安全デフォルト
BATCH_SIZE=${TRAIN_BATCH_SIZE:-1024}
VAL_BATCH_SIZE=${TRAIN_VAL_BATCH_SIZE:-1536}
# デフォルトの DataLoader 並列数を 8 に引き上げ（従来は 0 = 単一プロセスで極端に遅い）
NUM_WORKERS=${TRAIN_NUM_WORKERS:-8}
PREFETCH=${TRAIN_PREFETCH:-4}
# Optional overrides
# 勾配蓄積は train_atft.py の想定キー（train.batch.gradient_accumulation_steps）に合わせて渡す
GRAD_ACC=${TRAIN_ACCUMULATION:-4}
PRECISION=${TRAIN_PRECISION:-16-mixed}
VAL_INTERVAL=${TRAIN_VAL_INTERVAL:-1.0}

# 実行モード選択
if [ "$RUN_SAFE_PIPELINE" == "--safe" ]; then
    echo -e "${BLUE}Running with SafeTrainingPipeline validation...${NC}"
    python scripts/integrated_ml_training_pipeline.py \
        --data-path "$LATEST_DATASET" \
        --run-safe-pipeline \
        $ADV_GRAPH \
        train.batch.train_batch_size=$BATCH_SIZE \
        train.batch.val_batch_size=$VAL_BATCH_SIZE \
        train.batch.test_batch_size=$VAL_BATCH_SIZE \
        train.batch.num_workers=$NUM_WORKERS \
        train.batch.prefetch_factor=$PREFETCH \
        +train.batch.gradient_accumulation_steps=$GRAD_ACC \
        train.trainer.accumulate_grad_batches=$GRAD_ACC \
        train.trainer.precision=$PRECISION \
        train.trainer.val_check_interval=$VAL_INTERVAL \
        train.optimizer.lr=$LEARNING_RATE \
        train.trainer.max_epochs=$MAX_EPOCHS
    TRAIN_EXIT_CODE=$?
else
    echo -e "${BLUE}Running standard GPU training...${NC}"
    python scripts/integrated_ml_training_pipeline.py \
        --data-path "$LATEST_DATASET" \
        $ADV_GRAPH \
        train.batch.train_batch_size=$BATCH_SIZE \
        train.batch.val_batch_size=$VAL_BATCH_SIZE \
        train.batch.test_batch_size=$VAL_BATCH_SIZE \
        train.batch.num_workers=$NUM_WORKERS \
        train.batch.prefetch_factor=$PREFETCH \
        $( [ "$NUM_WORKERS" -gt 0 ] && echo train.batch.persistent_workers=true ) \
        +train.batch.gradient_accumulation_steps=$GRAD_ACC \
        train.trainer.accumulate_grad_batches=$GRAD_ACC \
        train.trainer.precision=$PRECISION \
        train.trainer.val_check_interval=$VAL_INTERVAL \
        train.optimizer.lr=$LEARNING_RATE \
        train.trainer.max_epochs=$MAX_EPOCHS
    TRAIN_EXIT_CODE=$?
fi

# Show success only if the previous command exited with 0
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ Training completed successfully${NC}"
else
    echo -e "${YELLOW}⚠️ Training finished with errors (see logs)${NC}"
    exit 1
fi
