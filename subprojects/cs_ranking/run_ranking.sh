#!/usr/bin/env bash
set -euo pipefail

# Simple wrapper to launch day‑ranking oriented training using existing pipeline.
#
# Usage:
#   ./run_ranking.sh --data output/ml_dataset_latest_full.parquet \
#                    --epochs 120 --batch-size 2048 --lr 2e-4

DATA="output/ml_dataset_latest_full.parquet"
EPOCHS=120
BATCH=2048
LR=2e-4

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data|--data-path)
      DATA="$2"; shift 2 ;;
    --epochs)
      EPOCHS="$2"; shift 2 ;;
    --batch-size)
      BATCH="$2"; shift 2 ;;
    --lr)
      LR="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ ! -f "$DATA" && ! -L "$DATA" ]]; then
  echo "❌ Dataset not found: $DATA" >&2
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export USE_AMP=1
export AMP_DTYPE=bf16

# Cross‑sectional ranking levers
export USE_DAY_BATCH=1
export USE_RANKIC=1
export RANKIC_WEIGHT=${RANKIC_WEIGHT:-0.5}
export CS_IC_WEIGHT=${CS_IC_WEIGHT:-0.3}
export SHARPE_WEIGHT=${SHARPE_WEIGHT:-0.1}

# Enable pairwise ranking loss (primary driver for RankIC)
export USE_PAIRWISE_RANK=${USE_PAIRWISE_RANK:-1}
export PAIRWISE_RANK_WEIGHT=${PAIRWISE_RANK_WEIGHT:-0.5}
export PAIRWISE_SAMPLE_RATIO=${PAIRWISE_SAMPLE_RATIO:-0.25}

# Throughput (A100 80GB defaults)
export ALLOW_UNSAFE_DATALOADER=1
export NUM_WORKERS=${NUM_WORKERS:-12}
export PERSISTENT_WORKERS=1
export PREFETCH_FACTOR=4
export PIN_MEMORY=1

echo "🚀 Cross‑sectional ranking run"
echo "  data    : $DATA"
echo "  epochs  : $EPOCHS"
echo "  batch   : $BATCH"
echo "  lr      : $LR"
echo "  workers : ${NUM_WORKERS}"
echo "  losses  : RankIC=$RANKIC_WEIGHT, CS_IC=$CS_IC_WEIGHT, Sharpe=$SHARPE_WEIGHT"

exec python scripts/train.py \
  --data-path "$DATA" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH" \
  --lr "$LR" \
  --hidden-size 256 \
  --mode optimized \
  --no-background
