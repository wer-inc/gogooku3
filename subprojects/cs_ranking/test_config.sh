#!/usr/bin/env bash
# Test script to verify cs_ranking configuration

echo "==================================================================="
echo "CS_RANKING CONFIGURATION TEST"
echo "==================================================================="
echo ""

# Source the environment setup from run_ranking.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export USE_AMP=1
export AMP_DTYPE=bf16

# Cross-sectional ranking levers
export USE_DAY_BATCH=1
export USE_RANKIC=1
export RANKIC_WEIGHT=${RANKIC_WEIGHT:-0.5}
export CS_IC_WEIGHT=${CS_IC_WEIGHT:-0.3}
export SHARPE_WEIGHT=${SHARPE_WEIGHT:-0.1}

# Enable pairwise ranking loss
export USE_PAIRWISE_RANK=${USE_PAIRWISE_RANK:-1}
export PAIRWISE_RANK_WEIGHT=${PAIRWISE_RANK_WEIGHT:-0.5}
export PAIRWISE_SAMPLE_RATIO=${PAIRWISE_SAMPLE_RATIO:-0.25}

# Throughput
export ALLOW_UNSAFE_DATALOADER=1
export NUM_WORKERS=${NUM_WORKERS:-12}
export PERSISTENT_WORKERS=1
export PREFETCH_FACTOR=4
export PIN_MEMORY=1

echo "✅ ENVIRONMENT VARIABLES:"
echo "   USE_DAY_BATCH           = $USE_DAY_BATCH"
echo "   USE_RANKIC              = $USE_RANKIC"
echo "   RANKIC_WEIGHT           = $RANKIC_WEIGHT"
echo "   CS_IC_WEIGHT            = $CS_IC_WEIGHT"
echo "   SHARPE_WEIGHT           = $SHARPE_WEIGHT"
echo "   USE_PAIRWISE_RANK       = $USE_PAIRWISE_RANK"
echo "   PAIRWISE_RANK_WEIGHT    = $PAIRWISE_RANK_WEIGHT"
echo "   PAIRWISE_SAMPLE_RATIO   = $PAIRWISE_SAMPLE_RATIO"
echo "   NUM_WORKERS             = $NUM_WORKERS"
echo "   PERSISTENT_WORKERS      = $PERSISTENT_WORKERS"
echo "   PREFETCH_FACTOR         = $PREFETCH_FACTOR"
echo "   PIN_MEMORY              = $PIN_MEMORY"
echo ""

echo "✅ LOSS FUNCTION CONFIGURATION:"
total=$(echo "$RANKIC_WEIGHT + $CS_IC_WEIGHT + $SHARPE_WEIGHT" | bc)
echo "   RankIC Weight  : $RANKIC_WEIGHT ($(echo "scale=1; $RANKIC_WEIGHT / $total * 100" | bc)%)"
echo "   CS_IC Weight   : $CS_IC_WEIGHT ($(echo "scale=1; $CS_IC_WEIGHT / $total * 100" | bc)%)"
echo "   Sharpe Weight  : $SHARPE_WEIGHT ($(echo "scale=1; $SHARPE_WEIGHT / $total * 100" | bc)%)"
echo "   Total          : $total"
echo ""

echo "✅ DATA PATHS:"
DATA="../../output/ml_dataset_latest_full.parquet"
if [[ -f "$DATA" || -L "$DATA" ]]; then
    echo "   Dataset: $DATA ✅"
    ls -lh "$DATA"
else
    echo "   Dataset: $DATA ❌ NOT FOUND"
fi
echo ""

echo "✅ SCRIPTS:"
if [[ -f "../../scripts/train.py" ]]; then
    echo "   train.py: ../../scripts/train.py ✅"
else
    echo "   train.py: ../../scripts/train.py ❌ NOT FOUND"
fi

if [[ -f "../../scripts/integrated_ml_training_pipeline.py" ]]; then
    echo "   integrated_ml_training_pipeline.py: ✅"
else
    echo "   integrated_ml_training_pipeline.py: ❌ NOT FOUND"
fi
echo ""

echo "✅ PREVIOUS RUN ARTIFACTS:"
if [[ -d "output/atft_data" ]]; then
    echo "   ATFT data: $(du -sh output/atft_data 2>/dev/null | cut -f1) ($(find output/atft_data -name "*.parquet" 2>/dev/null | wc -l) files)"
else
    echo "   ATFT data: NOT FOUND (will be generated)"
fi

if [[ -d "_logs/data_quality" ]]; then
    echo "   Quality logs: $(du -sh _logs/data_quality 2>/dev/null | cut -f1)"
else
    echo "   Quality logs: NOT FOUND"
fi
echo ""

echo "==================================================================="
echo "CONFIGURATION TEST COMPLETE"
echo "==================================================================="
echo ""
echo "To run quick experiment (3 epochs):"
echo "  cd /workspace/gogooku3/subprojects/cs_ranking"
echo "  make quick"
echo ""
echo "To run full experiment (120 epochs):"
echo "  cd /workspace/gogooku3/subprojects/cs_ranking"
echo "  make run"
