# Model Preservation & Recovery Playbook

**Created**: 2025-10-21
**Purpose**: Ensure best models are preserved, reproducible, and recoverable
**Current Best Model**: Tier 1 Experiment 1.1b (Sharpe 0.779)

---

## 🎯 Quick Reference

### Best Model Location (Tier 1)

```bash
# Model checkpoint (PyTorch state dict)
output/checkpoints/best_model_phase3.pth  # 19MB, saved 2025-10-21 06:33 JST

# Complete configuration and metrics
output/results/complete_training_result_20251021_063844.json  # 2.4MB

# Training logs
/tmp/experiment_1_1b_sharpe10.log  # 623KB

# Archived experiment bundle
archive/experiments_tier1_20251021.tar.gz  # 946KB (all Tier 1 experiments)
```

### Key Model Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Sharpe Ratio** | 0.779 | +33.8% over baseline (0.582) |
| **Val Loss** | -0.0319 | Negative = excellent performance |
| **Model Size** | 5.6M params | hidden_size=64 |
| **Loss Config** | Sharpe=1.0 | Pure Sharpe optimization |
| **Epochs** | 30 | Phase-based training |
| **Batch Size** | 1024 | GPU-optimized |
| **Learning Rate** | 2.0e-4 | AdamW optimizer |
| **Dataset** | 4.6M samples | 395 features, 2020-2025 |

---

## 📋 Model Preservation Checklist

### ✅ What's Already Preserved (Tier 1)

- [x] Model checkpoint (best_model_phase3.pth)
- [x] Complete training configuration (JSON)
- [x] Training logs (experiment_1_1b_sharpe10.log)
- [x] Loss weights (SHARPE_WEIGHT=1.0, others=0.0)
- [x] Model architecture (hidden_size=64, gat_heads=8)
- [x] Optimizer settings (AdamW, lr=2e-4, weight_decay=1e-4)
- [x] Dataset information (path, date range, feature count)
- [x] Phase configuration (4-phase training)
- [x] Final metrics (Sharpe, RankIC, IC, Val Loss)
- [x] Experiment archive (tar.gz with all logs/results)

### ⚠️ Missing/Recommended Additions

- [ ] Random seed documentation (not explicitly set)
- [ ] Exact Python/PyTorch/CUDA versions
- [ ] Data preprocessing steps (cross-sectional normalization details)
- [ ] Walk-forward split configuration
- [ ] Graph builder settings (if used)
- [ ] Model recovery test script

---

## 🔄 Model Recovery Procedure

### Method 1: Resume from Best Checkpoint (Recommended)

Use this to continue training or fine-tune the best model.

```bash
cd /workspace/gogooku3

# 1. Verify checkpoint exists
ls -lh output/checkpoints/best_model_phase3.pth
# Expected: 19MB file from 2025-10-21 06:33

# 2. Load in training script
python scripts/integrated_ml_training_pipeline.py \
  --resume-from-checkpoint output/checkpoints/best_model_phase3.pth \
  --max-epochs 30 \
  --data-path output/ml_dataset_latest_full.parquet \
  > /tmp/resume_training.log 2>&1 &

# 3. Monitor resumption
tail -f /tmp/resume_training.log | grep -E "Loaded checkpoint|Resuming|Epoch"
```

### Method 2: Exact Reproduction from Scratch

Use this to verify reproducibility or retrain from scratch.

```bash
cd /workspace/gogooku3

# 1. Set exact loss weights
export SHARPE_WEIGHT=1.0
export RANKIC_WEIGHT=0.0
export CS_IC_WEIGHT=0.0

# 2. Use same dataset
export DATASET_PATH="output/ml_dataset_latest_full.parquet"

# 3. Launch training with same configuration
python scripts/integrated_ml_training_pipeline.py \
  --max-epochs 30 \
  --data-path $DATASET_PATH \
  --batch-size 1024 \
  --lr 2.0e-4 \
  > /tmp/reproduce_tier1_best.log 2>&1 &

# 4. Monitor and compare
tail -f /tmp/reproduce_tier1_best.log | grep -E "Sharpe|Val Loss"
# Expected: Sharpe ~0.75-0.80, Val Loss ~-0.03
```

### Method 3: Inference Only (No Training)

Use this to generate predictions with the best model.

```bash
cd /workspace/gogooku3

# Create inference script
cat > scripts/inference_tier1_best.py << 'EOF'
#!/usr/bin/env python3
"""Inference with Tier 1 best model (Sharpe 0.779)"""

import torch
import polars as pl
from pathlib import Path

# Load checkpoint
checkpoint_path = Path("output/checkpoints/best_model_phase3.pth")
checkpoint = torch.load(checkpoint_path, map_location="cuda:0")

print(f"✅ Loaded checkpoint from Phase {checkpoint.get('phase', 'unknown')}")
print(f"   Epoch: {checkpoint.get('epoch', 'unknown')}")
print(f"   Val Loss: {checkpoint.get('best_val_loss', 'unknown')}")

# Load model (assuming ATFT-GAT-FAN)
# TODO: Import actual model class and instantiate
# model = ATFTModel(...)
# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()

# Load data and run inference
# df = pl.read_parquet("output/ml_dataset_latest_full.parquet")
# predictions = model(df)
# df.with_columns(pl.lit(predictions).alias("pred")).write_parquet("output/predictions.parquet")

print("⚠️  Inference script template created - implement model loading")
EOF

chmod +x scripts/inference_tier1_best.py
python scripts/inference_tier1_best.py
```

---

## 🔍 Model Validation Tests

### Test 1: Checkpoint Integrity

```bash
# Verify checkpoint can be loaded
python -c "
import torch
ckpt = torch.load('output/checkpoints/best_model_phase3.pth', map_location='cpu')
print(f'✅ Checkpoint keys: {list(ckpt.keys())}')
print(f'✅ Model state dict size: {len(ckpt.get(\"model_state_dict\", {}))} tensors')
print(f'✅ Optimizer state: {\"optimizer_state_dict\" in ckpt}')
print(f'✅ Best val loss: {ckpt.get(\"best_val_loss\", \"N/A\")}')
"
```

### Test 2: Configuration Consistency

```bash
# Extract and verify key configuration parameters
python -c "
import json
with open('output/results/complete_training_result_20251021_063844.json') as f:
    result = json.load(f)

print('✅ Loss Weights:')
print(f'   Sharpe: {result.get(\"config\", {}).get(\"SHARPE_WEIGHT\", \"N/A\")}')
print(f'   RankIC: {result.get(\"config\", {}).get(\"RANKIC_WEIGHT\", \"N/A\")}')
print(f'   CS-IC: {result.get(\"config\", {}).get(\"CS_IC_WEIGHT\", \"N/A\")}')

print('\\n✅ Final Metrics:')
print(f'   Test Sharpe: {result.get(\"final_metrics\", {}).get(\"test_sharpe_ratio\", \"N/A\")}')
print(f'   Val Loss: {result.get(\"best_val_loss\", \"N/A\")}')
"
```

### Test 3: Reproducibility Check

```bash
# Compare two training runs with same configuration
# (Run Method 2 twice and compare final Sharpe ratios)
# Expected: Within ±5% due to non-deterministic operations

# Note: For exact reproducibility, need to set:
# - PYTHONHASHSEED=0
# - torch.manual_seed(42)
# - torch.cuda.manual_seed(42)
# - torch.backends.cudnn.deterministic=True
# - torch.backends.cudnn.benchmark=False
```

---

## 📦 Backup and Archive Strategy

### Local Backups

```bash
# Create timestamped backup
BACKUP_DIR="archive/backups/tier1_best_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Copy essential files
cp output/checkpoints/best_model_phase3.pth "$BACKUP_DIR/"
cp output/results/complete_training_result_20251021_063844.json "$BACKUP_DIR/"
cp /tmp/experiment_1_1b_sharpe10.log "$BACKUP_DIR/"

# Create README
cat > "$BACKUP_DIR/README.md" << EOF
# Tier 1 Best Model Backup

**Created**: $(date)
**Sharpe Ratio**: 0.779
**Model**: ATFT-GAT-FAN (5.6M params, hidden_size=64)
**Loss Config**: Pure Sharpe (weight=1.0)

## Contents
- best_model_phase3.pth - Model checkpoint (19MB)
- complete_training_result_20251021_063844.json - Full config and metrics
- experiment_1_1b_sharpe10.log - Training logs

## Recovery
See /workspace/gogooku3/docs/playbooks/model_preservation_playbook.md
EOF

# Compress backup
tar -czf "${BACKUP_DIR}.tar.gz" -C archive/backups "$(basename $BACKUP_DIR)"
rm -rf "$BACKUP_DIR"

echo "✅ Backup created: ${BACKUP_DIR}.tar.gz"
```

### GCS Upload (Cloud Backup)

```bash
# Upload to Google Cloud Storage (assuming GCS configured)
export GOOGLE_APPLICATION_CREDENTIALS="/workspace/gogooku3/secrets/gogooku-b3b34bc07639.json"

# Upload checkpoint and results
gsutil -m cp -r output/checkpoints gs://gogooku3-models/tier1/
gsutil -m cp -r output/results gs://gogooku3-models/tier1/

# Upload archive
gsutil cp archive/experiments_tier1_20251021.tar.gz gs://gogooku3-models/tier1/

echo "✅ Uploaded to GCS: gs://gogooku3-models/tier1/"
```

---

## 🔧 Environment Recreation

### Python Environment

```bash
# Record current environment
pip freeze > requirements_tier1_best.txt

# Key dependencies for Tier 1 best model:
# torch==2.8.0+cu128
# pytorch-lightning==2.x
# polars==1.x
# hydra-core==1.3.x
```

### System Configuration

```bash
# GPU information
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# CUDA version
nvcc --version

# Python version
python --version

# Save to environment.txt
cat > environment_tier1_best.txt << EOF
GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
CUDA: $(nvcc --version | grep release | awk '{print $5}')
Python: $(python --version)
PyTorch: $(python -c "import torch; print(torch.__version__)")
Date: $(date)
EOF
```

---

## 📊 Model Performance Reference

### Tier 1 Best Model (Experiment 1.1b)

**Final Metrics**:
- **Test Sharpe Ratio**: 0.779 (+33.8% vs baseline 0.582)
- **Val Loss**: -0.0319 (negative = excellent)
- **RankIC@5d**: ~0.18-0.20 (expected from Tier 1 report)
- **Training Time**: 85 minutes (30 epochs)

**Training Progression**:
```
Phase 0 (Baseline):  5 epochs → Val Sharpe: -0.016 to 0.009
Phase 1 (AN):       10 epochs → High volatility (-0.720 to +0.786)
Phase 2 (GAT):       8 epochs → Stabilizing, Val Loss → -0.03
Phase 3 (Finetune):  6 epochs → Final optimization, Sharpe 0.779
```

**Key Success Factors**:
1. Pure Sharpe optimization (weight=1.0, no RankIC/CS-IC)
2. Phase-based training (gradual complexity increase)
3. Loss-metric alignment (loss directly measures Sharpe)
4. Stable convergence (no NaN, no divergence)

---

## 🚨 Common Recovery Issues

### Issue 1: Checkpoint Not Found

```bash
# Problem: FileNotFoundError: best_model_phase3.pth
# Solution: Check symlinks and actual file location
ls -la output/checkpoints/
find output -name "*best_model*.pth"

# If missing, restore from archive
tar -xzf archive/experiments_tier1_20251021.tar.gz -C /tmp/
cp /tmp/experiments_tier1_20251021/checkpoints/* output/checkpoints/
```

### Issue 2: Configuration Mismatch

```bash
# Problem: Model architecture doesn't match checkpoint
# Solution: Extract exact config from JSON result file

python -c "
import json
with open('output/results/complete_training_result_20251021_063844.json') as f:
    config = json.load(f)['config']
    print(f'hidden_size: {config.get(\"hidden_size\", 64)}')
    print(f'gat_heads: {config.get(\"gat_heads\", 8)}')
    print(f'num_layers: {config.get(\"num_layers\", 3)}')
"

# Use these values when instantiating model
```

### Issue 3: Dataset Version Mismatch

```bash
# Problem: Feature count mismatch (model expects 395, dataset has 307)
# Solution: Verify dataset used for Tier 1 training

# Check original dataset
python -c "
import polars as pl
df = pl.read_parquet('output/ml_dataset_latest_full.parquet')
print(f'Features: {len(df.columns)}')
print(f'Samples: {len(df)}')
print(f'Date range: {df[\"Date\"].min()} to {df[\"Date\"].max()}')
"

# If mismatch, regenerate dataset with same date range:
make dataset-gpu START=2020-09-06 END=2025-09-06
```

### Issue 4: Loss Weight Environment Variables

```bash
# Problem: Training with wrong loss weights
# Solution: Always verify environment before training

echo "Current loss weights:"
echo "  SHARPE_WEIGHT=${SHARPE_WEIGHT:-not set}"
echo "  RANKIC_WEIGHT=${RANKIC_WEIGHT:-not set}"
echo "  CS_IC_WEIGHT=${CS_IC_WEIGHT:-not set}"

# For Tier 1 best model, should be:
# SHARPE_WEIGHT=1.0
# RANKIC_WEIGHT=0.0
# CS_IC_WEIGHT=0.0
```

---

## 🎓 Lessons Learned (Tier 1)

### What Made This Model Successful

1. **Pure Sharpe Optimization**: weight=1.0 eliminated multi-objective conflicts
2. **No Intermediate Weights**: Avoided "middle trap" (0.8 weight performed worse)
3. **Loss-Metric Alignment**: Optimizing loss directly improved target metric
4. **Stable Training**: High batch-level volatility acceptable if convergence good
5. **Phase-Based Approach**: Gradual complexity increase prevented instability

### What to Avoid

1. **Intermediate Sharpe weights (0.7-0.9)**: Non-monotonic optimization landscape
2. **Premature optimization**: Test extremes (0.0, 0.5, 1.0) before fine-tuning
3. **Ignoring loss-metric divergence**: If loss improves but Sharpe degrades, stop
4. **Impatience**: Wait at least 5 epochs before diagnosing issues

---

## 🔜 Next Steps for Tier 2+

### Recommended Approach

**Use Tier 1 best model as foundation**:
1. **Experiment 2.1**: hidden_size 64→256 with Sharpe weight=1.0
2. **Experiment 3.1**: Add more features (220) with proven loss config
3. **HPO**: Fine-tune learning rate, batch size (not loss weights)

**Configuration to carry forward**:
```bash
# Proven settings from Tier 1
export SHARPE_WEIGHT=1.0
export RANKIC_WEIGHT=0.0
export CS_IC_WEIGHT=0.0

# New model capacity (Tier 2)
export HIDDEN_SIZE=256  # vs 64 in Tier 1

# Same training setup
export MAX_EPOCHS=30
export BATCH_SIZE=1024
export LEARNING_RATE=2.0e-4
```

**Expected Improvement**:
- Tier 1 (hidden=64): Sharpe 0.779
- Tier 2 (hidden=256): Expected 0.82-0.85 (+5-8%)
- **Target**: 0.849

---

## 📚 References

### Documentation

- **Experiment Report**: `docs/reports/experiments/tier1_sharpe_weight_optimization_20251021.md`
- **Experiment Status**: `EXPERIMENT_STATUS.md`
- **Loss Curriculum**: `scripts/utils/loss_curriculum.py`
- **HPO Config**: `configs/experiments/hpo_search_space.yaml`

### Command Reference

```bash
# View Tier 1 report
cat docs/reports/experiments/tier1_sharpe_weight_optimization_20251021.md | less

# Check experiment status
cat EXPERIMENT_STATUS.md | grep -A 10 "Current Best"

# View training logs
less /tmp/experiment_1_1b_sharpe10.log

# Extract Sharpe ratio from logs
grep "Achieved Sharpe" /tmp/experiment_1_1b_sharpe10.log
```

---

**Last Updated**: 2025-10-21
**Status**: Tier 1 Complete, Tier 2 Ready
**Next Experiment**: 2.1 (hidden_size=256) - Expected to reach target 0.849
