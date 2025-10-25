# Training Commands Reference

**Purpose**: Standardized commands for each training phase to ensure consistency and avoid parameter conflicts.

---

## 📋 Pre-Training Checklist

Before **every** training run:

```bash
# 1. Clear ATFT cache (if config changed)
./scripts/clean_atft_cache.sh --force

# 2. Run health check
./tools/project-health-check.sh

# 3. Verify training mode
export USE_MINI_TRAIN=0
export FORCE_MINI_TRAIN=0
echo "Training mode: Full (not mini)"

# 4. Check current training processes
ps aux | grep -E "train_atft|integrated_ml" | grep -v grep
```

---

## 🎯 Phase 0: Diagnostics

### Data Leakage Detection
```bash
python scripts/detect_data_leakage.py
```

**Expected output**:
- ✅ No suspicious feature names
- ✅ No perfect correlation with targets
- ✅ Normalization statistics reasonable

### Baseline Feature Analysis
```bash
python scripts/analyze_baseline_features.py \
  --data-path output/ml_dataset_latest_full.parquet \
  --output reports/feature_analysis_$(date +%Y%m%d).md
```

**Check for**:
- Feature importance distribution
- IC/RankIC per feature
- Redundant features (correlation > 0.95)

---

## 🚀 Phase 1: Baseline Training

### Short Test (5 epochs - Verify setup)

```bash
# Purpose: Verify configuration before long run
ATFT_TRAIN_CONFIG=adaptive_phase3_ext \
FORCE_SINGLE_PROCESS=1 \
DEGENERACY_GUARD_VERBOSE=1 \
  python scripts/integrated_ml_training_pipeline.py \
    --max-epochs 5 \
    --batch-size 1024 \
    --lr 2e-4 \
  2>&1 | tee _logs/training/phase1_test_$(date +%Y%m%d_%H%M%S).log
```

**Post-test verification**:
```bash
# Check degeneracy
grep -i "degeneracy\|pred.*std\|unique.*ratio" \
  _logs/training/phase1_test_*.log

# Check GPU usage
grep -i "gpu\|cuda" _logs/training/phase1_test_*.log

# Expected: No degeneracy warnings, GPU utilized
```

### Full Run (60 epochs - Production)

```bash
# Purpose: Establish baseline performance
ATFT_TRAIN_CONFIG=adaptive_phase3_ext \
FORCE_SINGLE_PROCESS=1 \
  python scripts/integrated_ml_training_pipeline.py \
    --max-epochs 60 \
    --batch-size 1024 \
    --lr 2e-4 \
  2>&1 | tee _logs/training/phase1_full_$(date +%Y%m%d_%H%M%S).log
```

**Monitor in separate terminal**:
```bash
# Real-time log monitoring
tail -f _logs/training/phase1_full_*.log | grep -E "Epoch|Sharpe|IC|Loss"

# Progress check
grep "Epoch.*/" _logs/training/phase1_full_*.log | tail -5
```

### With Time-Series Features (Optional)

```bash
# If Safe Mode baseline is stable, enable time-series
ATFT_TRAIN_CONFIG=adaptive_phase3_ext \
FORCE_SINGLE_PROCESS=1 \
  python scripts/integrated_ml_training_pipeline.py \
    --max-epochs 10 \
    --batch-size 1024 \
    --lr 2e-4 \
    model.input_dims.historical_features=373 \
    model.input_dims.basic_features=0 \
  2>&1 | tee _logs/training/phase1_timeseries_$(date +%Y%m%d_%H%M%S).log

# Check for NaN
grep -i "nan\|inf" _logs/training/phase1_timeseries_*.log
```

---

## ⚡ Phase 2: Scale-Up (Multi-worker)

### Stage 2.1: Larger Batch (Safe Mode)

```bash
# Increase batch size while still in Safe Mode
ATFT_TRAIN_CONFIG=adaptive_phase3_ext \
FORCE_SINGLE_PROCESS=1 \
  python scripts/integrated_ml_training_pipeline.py \
    --max-epochs 10 \
    --batch-size 2048 \
    --lr 2e-4 \
  2>&1 | tee _logs/training/phase2_stage1_$(date +%Y%m%d_%H%M%S).log
```

### Stage 2.2: 2 Workers (First Multi-process)

```bash
# CRITICAL: Test with 2 workers first
ATFT_TRAIN_CONFIG=adaptive_phase3_ext \
FORCE_SINGLE_PROCESS=0 \
NUM_WORKERS=2 \
  python scripts/integrated_ml_training_pipeline.py \
    --max-epochs 5 \
    --batch-size 1024 \
    --lr 2e-4 \
  2>&1 | tee _logs/training/phase2_stage2_$(date +%Y%m%d_%H%M%S).log

# Monitor for deadlock
ps aux | grep train_atft | awk '{print $2}' | \
  xargs -I{} ps -p {} -o pid,nlwp,stat,%cpu

# Expected: nlwp < 100, stat=Rl, %cpu > 50
```

### Stage 2.3: 4 Workers

```bash
# If Stage 2.2 stable, increase to 4
ATFT_TRAIN_CONFIG=adaptive_phase3_ext \
FORCE_SINGLE_PROCESS=0 \
NUM_WORKERS=4 \
  python scripts/integrated_ml_training_pipeline.py \
    --max-epochs 10 \
    --batch-size 1024 \
    --lr 2e-4 \
  2>&1 | tee _logs/training/phase2_stage3_$(date +%Y%m%d_%H%M%S).log
```

### Stage 2.4: Full Optimization (4 workers + large batch)

```bash
# Final configuration for throughput
ATFT_TRAIN_CONFIG=adaptive_phase3_ext \
FORCE_SINGLE_PROCESS=0 \
NUM_WORKERS=4 \
  python scripts/integrated_ml_training_pipeline.py \
    --max-epochs 60 \
    --batch-size 2048 \
    --lr 2e-4 \
  2>&1 | tee _logs/training/phase2_stage4_$(date +%Y%m%d_%H%M%S).log
```

---

## 🔬 Phase 3: Hyperparameter Optimization

### Prerequisites
```bash
# Verify baseline is stable and positive
python -c "
import re
with open('_logs/training/phase1_full_*.log') as f:
    logs = f.read()
    sharpes = re.findall(r'Val.*Sharpe[:\s]+([-\d.]+)', logs)
    sharpes = [float(s) for s in sharpes if s]
    if len(sharpes) > 10:
        recent = sharpes[-10:]
        if sum(s > 0 for s in recent) >= 8:
            print('✅ Ready for HPO (80%+ positive)')
        else:
            print('⏸️  Wait for stability')
"
```

### HPO Setup
```bash
# One-time setup
make hpo-setup
```

### Run HPO Study
```bash
# 20 trials, short epochs
make hpo-run \
  HPO_TRIALS=20 \
  HPO_STUDY=atft_phase3_tuning \
  HPO_EPOCHS=10

# Check progress
make hpo-status
```

### Apply Best Parameters
```bash
# After HPO completes
BEST_LR=$(optuna best-params atft_phase3_tuning --param lr)
BEST_SHARPE_WEIGHT=$(optuna best-params atft_phase3_tuning --param sharpe_weight)

# Create config with best params
cat > configs/train/phase3_optimized.yaml << EOF
defaults:
  - adaptive_phase3_ext

optimizer:
  lr: ${BEST_LR}

loss:
  auxiliary:
    sharpe_loss:
      weight: ${BEST_SHARPE_WEIGHT}
EOF

# Run with optimized config
ATFT_TRAIN_CONFIG=phase3_optimized \
  python scripts/integrated_ml_training_pipeline.py \
    --max-epochs 60 \
    --batch-size 2048 \
  2>&1 | tee _logs/training/phase3_optimized_$(date +%Y%m%d_%H%M%S).log
```

---

## 🏭 Phase 4: Production Training

### Final Run (120 epochs)

```bash
# Use all optimizations from Phase 2-3
ATFT_TRAIN_CONFIG=phase3_optimized \
FORCE_SINGLE_PROCESS=0 \
NUM_WORKERS=4 \
  python scripts/integrated_ml_training_pipeline.py \
    --max-epochs 120 \
    --batch-size 2048 \
    --lr ${BEST_LR:-2e-4} \
  2>&1 | tee _logs/training/phase4_production_$(date +%Y%m%d_%H%M%S).log
```

---

## 🛑 Emergency Stop

```bash
# Stop current training
make train-stop

# Or manual
ps aux | grep -E "train_atft|integrated_ml" | grep -v grep | awk '{print $2}' | xargs -r kill -15

# Wait 10 seconds, then force kill if needed
sleep 10
ps aux | grep -E "train_atft|integrated_ml" | grep -v grep | awk '{print $2}' | xargs -r kill -9
```

---

## 📊 Post-Training Analysis

### Extract Metrics
```bash
# Val Sharpe over time
grep "Val.*Sharpe" _logs/training/phase*_full_*.log | \
  sed 's/.*Sharpe[:\s]*\([-0-9.]*\)/\1/' > metrics/sharpe_history.txt

# Plot (if matplotlib available)
python -c "
import matplotlib.pyplot as plt
with open('metrics/sharpe_history.txt') as f:
    sharpes = [float(x.strip()) for x in f if x.strip()]
plt.plot(sharpes)
plt.xlabel('Epoch')
plt.ylabel('Val Sharpe')
plt.title('Sharpe Progression')
plt.savefig('metrics/sharpe_plot.png')
print('Saved to metrics/sharpe_plot.png')
"
```

### Evaluate Model
```bash
# Load best checkpoint and evaluate
python scripts/validate_production_model.py \
  --checkpoint outputs/best_model.pt \
  --data output/ml_dataset_latest_full.parquet \
  --output reports/model_eval_$(date +%Y%m%d).md
```

---

## ✅ Common Parameter Reference

| Parameter | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|-----------|---------|---------|---------|---------|
| **max_epochs** | 60 | 10-60 | 10 (HPO) | 120 |
| **batch_size** | 1024 | 1024-2048 | 1024 | 2048 |
| **lr** | 2e-4 | 2e-4 | 5e-5 - 5e-4 (search) | Best from HPO |
| **FORCE_SINGLE_PROCESS** | 1 | 0 | 0 | 0 |
| **NUM_WORKERS** | 0 | 2-4 | 4 | 4 |

---

## 🔍 Troubleshooting

### Issue: `--max-epochs` ignored

**Cause**: `integrated_ml_training_pipeline.py:675-678` may override

**Solution**: Always specify explicitly in CLI
```bash
--max-epochs 60  # CLI override takes precedence
```

### Issue: Deadlock after enabling multi-worker

**Cause**: PyTorch thread pool issue (CLAUDE.md:2025-10-14)

**Solution**:
```bash
# 1. Revert to Safe Mode
FORCE_SINGLE_PROCESS=1

# 2. Check thread fix is in place
grep "Limited PyTorch threads" scripts/train_atft.py
# Should find lines 9-18

# 3. Try with 2 workers first (not 8)
NUM_WORKERS=2
```

### Issue: NaN in predictions with time-series

**Cause**: `historical_features` configuration

**Solution**:
```bash
# 1. Clear cache
./scripts/clean_atft_cache.sh --force

# 2. Start with 0, gradually increase
model.input_dims.historical_features=0    # Test
model.input_dims.historical_features=100  # Test
model.input_dims.historical_features=200  # Test
model.input_dims.historical_features=373  # Final
```

---

**Last updated**: 2025-10-24
**Related**: [MODEL_INPUT_DIMS.md](MODEL_INPUT_DIMS.md), [EXPERIMENT_STATUS.md](../EXPERIMENT_STATUS.md)
