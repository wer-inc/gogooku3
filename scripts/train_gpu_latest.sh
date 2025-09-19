#!/bin/bash
# GPU Training Script for Latest Dataset
# 最新データセットでGPU学習を実行するスクリプト

set -e

# GPU環境設定（永続化）
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export FORCE_GPU=1
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

# デフォルト引数
RUN_SAFE_PIPELINE=${1:-""}
ADV_GRAPH=${2:-"--adv-graph-train"}
LEARNING_RATE=${3:-"2e-4"}
MAX_EPOCHS=${4:-"75"}
BATCH_SIZE=${TRAIN_BATCH_SIZE:-4096}
VAL_BATCH_SIZE=${TRAIN_VAL_BATCH_SIZE:-6144}
NUM_WORKERS=${TRAIN_NUM_WORKERS:-16}
PREFETCH=${TRAIN_PREFETCH:-8}
# Optional overrides
GRAD_ACC=${TRAIN_ACCUMULATION:-1}
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
        train.trainer.accumulate_grad_batches=$GRAD_ACC \
        train.trainer.precision=$PRECISION \
        train.trainer.val_check_interval=$VAL_INTERVAL \
        train.optimizer.lr=$LEARNING_RATE \
        train.trainer.max_epochs=$MAX_EPOCHS
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
        train.trainer.accumulate_grad_batches=$GRAD_ACC \
        train.trainer.precision=$PRECISION \
        train.trainer.val_check_interval=$VAL_INTERVAL \
        train.optimizer.lr=$LEARNING_RATE \
        train.trainer.max_epochs=$MAX_EPOCHS
fi

echo -e "${GREEN}✅ Training completed successfully${NC}"
