#!/usr/bin/env bash
set -euo pipefail

echo "[gogooku5] health-check"

# GPU status check
if command -v nvidia-smi &> /dev/null; then
    gpu_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -n "$gpu_info" ]; then
        echo "- GPU status: ✓ $gpu_info"
    else
        echo "- GPU status: ⚠ nvidia-smi available but no GPU detected"
    fi
else
    echo "- GPU status: ✗ nvidia-smi not found (CPU-only mode)"
fi

# Dataset builder check
if [ -f ../data/Makefile ]; then
    echo "- Dataset builder: ✓ ready"
else
    echo "- Dataset builder: ⚠ pending"
fi

# Models check
if [ -d ../models ]; then
    model_count=$(ls ../models 2>/dev/null | wc -l)
    echo "- Models: ✓ $model_count available"
else
    echo "- Models: ⚠ directory not found"
fi
