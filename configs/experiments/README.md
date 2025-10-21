# Experiment Configurations

Quick-launch experiments based on `experiment_design.md` from training_20251021_012605.

## Current Baseline
- **Sharpe**: 0.582
- **Loss weights**: Sharpe 0.6 / RankIC 0.25 / CS-IC 0.15
- **Target**: Sharpe 0.849 (45% remaining gap)

---

## Tier 1: Loss Schedule Optimization (Quick Wins, 1-2 hours each)

### Experiment 1.1a: Sharpe Weight 0.8
**Status**: ✅ RUNNING (PID 636126, started 2025-10-21 03:53)
**Expected**: Sharpe ~0.64-0.67 (+10-15% improvement)

```bash
# Launch command (already running)
cd /workspace/gogooku3
SHARPE_WEIGHT=0.8 RANKIC_WEIGHT=0.15 CS_IC_WEIGHT=0.05 \
python scripts/integrated_ml_training_pipeline.py \
  --max-epochs 30 \
  --data-path output/ml_dataset_latest_full.parquet \
  > /tmp/experiment_1_1a_sharpe08.log 2>&1
```

**Monitoring**:
```bash
tail -f /tmp/experiment_1_1a_sharpe08.log | grep -E "Epoch|Sharpe|Phase"
```

### Experiment 1.1b: Sharpe Weight 1.0 (Sharpe-only)
**Status**: ⏳ STAGED (ready to run after 1.1a)
**Expected**: Sharpe ~0.70-0.75 (+20-30% improvement)
**Risk**: Pure Sharpe may destabilize training (monitor closely)

```bash
cd /workspace/gogooku3
SHARPE_WEIGHT=1.0 RANKIC_WEIGHT=0.0 CS_IC_WEIGHT=0.0 \
python scripts/integrated_ml_training_pipeline.py \
  --max-epochs 30 \
  --data-path output/ml_dataset_latest_full.parquet \
  > /tmp/experiment_1_1b_sharpe10.log 2>&1 &

# Get PID and monitor
echo "Experiment 1.1b PID: $!"
tail -f /tmp/experiment_1_1b_sharpe10.log
```

### Experiment 1.2: Loss Schedule Curriculum
**Status**: ⚙️ IMPLEMENTATION READY
**Expected**: Sharpe ~0.70 (+15-20% improvement)

**Implementation**: See `scripts/utils/loss_curriculum.py` (to be created)

**Concept**:
- Phase 0-1: Balanced (Sharpe 0.5, RankIC 0.3, IC 0.2)
- Phase 2: Sharpe-focused (0.7, 0.2, 0.1)
- Phase 3: Sharpe-only (1.0, 0.0, 0.0)

```bash
cd /workspace/gogooku3
# Using phase-based loss schedule
python scripts/integrated_ml_training_pipeline.py \
  --max-epochs 30 \
  --data-path output/ml_dataset_latest_full.parquet \
  --use-loss-curriculum \
  > /tmp/experiment_1_2_curriculum.log 2>&1 &
```

---

## Tier 2: Model Capacity Expansion (2-4 hours each)

### Experiment 2.1: Hidden Size 256
**Status**: ⚙️ CONFIG READY
**Expected**: Sharpe ~0.70-0.73 (+20-25% improvement)

**Note**: `config_production_optimized.yaml` already has `hidden_size: 256`

```bash
cd /workspace/gogooku3
# Use production optimized config with Sharpe-focused weights
SHARPE_WEIGHT=0.8 RANKIC_WEIGHT=0.15 CS_IC_WEIGHT=0.05 \
python scripts/integrated_ml_training_pipeline.py \
  --max-epochs 30 \
  --data-path output/ml_dataset_latest_full.parquet \
  > /tmp/experiment_2_1_hidden256.log 2>&1 &
```

**Resource Requirements**:
- Model params: ~20M (vs 5.6M baseline)
- GPU memory: +2-3GB (still fits in 80GB A100)
- Training time: +30-50% per epoch

### Experiment 2.2: Multi-Head Attention Expansion
**Status**: 📝 CONFIG NEEDED

```bash
# Requires model config override
cd /workspace/gogooku3
python scripts/integrated_ml_training_pipeline.py \
  --max-epochs 30 \
  --data-path output/ml_dataset_latest_full.parquet \
  model.gat.architecture.heads=[16] \
  model.hidden_size=128 \
  > /tmp/experiment_2_2_multihead.log 2>&1 &
```

---

## Tier 3: Feature Expansion (2-4 hours each)

### Experiment 3.1: Feature Bundle 220
**Status**: 📝 CONFIG NEEDED
**Expected**: Sharpe ~0.73-0.76 (+25-30% improvement)

**Current**: 99 features (25% of 395 available)
**Target**: 220 features (55%)

**Feature Bundles**:
- **Bundle 150**: Core + Momentum (price, momentum, volume)
- **Bundle 220**: + Flow & Margin (daily margin interest, money flow)
- **Bundle 280**: + Sector & Correlation (sector short selling, cross-stock)

```bash
# Implementation: Modify feature selection in dataset builder
cd /workspace/gogooku3
# Regenerate dataset with more features
python scripts/pipelines/run_full_dataset.py \
  --start-date 2020-09-06 \
  --end-date 2025-09-06 \
  --enable-all-features \
  --max-features 220

# Then train
python scripts/integrated_ml_training_pipeline.py \
  --max-epochs 30 \
  --data-path output/ml_dataset_220features.parquet \
  > /tmp/experiment_3_1_features220.log 2>&1 &
```

---

## Tier 4: HPO (Hyperparameter Optimization, 4-6 hours)

### Experiment 4.1: Quick HPO Probe (12-16 trials, 2 epochs each)
**Status**: 📝 CONFIG NEEDED
**Expected**: +5-10% improvement → Sharpe ~0.61-0.64

**Search Space**:
```python
{
    'learning_rate': [1e-4, 2e-4, 3e-4, 5e-4],
    'weight_decay': [1e-5, 5e-5, 1e-4, 5e-4],
    'sharpe_weight': [0.6, 0.7, 0.8, 0.9, 1.0],
    'feature_dropout': [0.0, 0.1, 0.2, 0.3],
    'batch_size': [1024, 2048, 4096],
}
```

```bash
cd /workspace/gogooku3
make hpo-run \
    HPO_TRIALS=16 \
    HPO_STUDY=sharpe_optimization \
    HPO_MAX_EPOCHS=2 \
    HPO_PARAMS="lr,weight_decay,sharpe_weight,dropout,batch_size"
```

### Experiment 4.2: Full HPO (50-100 trials, 30 epochs each)
**Status**: ⏸️ ONLY IF Quick Probe shows promise
**Expected**: +10-20% improvement → Sharpe ~0.64-0.70

```bash
cd /workspace/gogooku3
make hpo-run \
    HPO_TRIALS=50 \
    HPO_STUDY=sharpe_optimization_full \
    HPO_MAX_EPOCHS=30
```

---

## Execution Timeline

### Phase 1: Quick Wins (1-2 days)
1. ✅ **Experiment 1.1a**: Sharpe 0.8 (RUNNING, ~1 hour)
2. ⏳ **Experiment 1.1b**: Sharpe 1.0 (~1 hour)
3. ⚙️ **Experiment 1.2**: Loss curriculum (~2 hours)
4. ⚙️ **Experiment 4.1**: Quick HPO probe (~6 hours)

**Expected Combined**: Sharpe ~0.67-0.76 (+15-30%)

### Phase 2: Capacity Expansion (2-3 days)
1. **Experiment 2.1**: Hidden size 256 (~4 hours)
2. **Experiment 3.1**: Feature bundle 220 (~3 hours)
3. Combine best from Phase 1 + Phase 2

**Expected Combined**: Sharpe ~0.70-0.81 (+20-40%)

### Phase 3: Refinement (3-5 days)
1. **Experiment 4.2**: Full HPO (24-48 hours)
2. Ensemble best 3-5 models
3. Production validation

**Target**: Sharpe 0.80-0.85 (95-100% of target 0.849)

---

## Monitoring Commands

### Check Running Experiments
```bash
# List all training processes
ps aux | grep -E "python.*integrated_ml_training_pipeline" | grep -v grep

# Check specific experiment
tail -f /tmp/experiment_*.log | grep -E "Epoch|Sharpe|Loss"
```

### Compare Results
```bash
# Extract final Sharpe from all experiments
for log in /tmp/experiment_*.log; do
    echo "=== $(basename $log) ==="
    grep -E "Achieved Sharpe|Final.*Sharpe" $log | tail -1
done
```

### Archive Successful Experiments
```bash
# After each experiment completes
RUN_ID="experiment_1_1a_$(date +%Y%m%d_%H%M%S)"
mkdir -p archive/$RUN_ID
cp /tmp/experiment_1_1a_sharpe08.log archive/$RUN_ID/
# Extract and save learning curves
python archive/training_20251021_012605/visualize_learning_curves.py --log $RUN_ID
```

---

## Quick Launch Summary

```bash
# Experiment 1.1a (RUNNING)
tail -f /tmp/experiment_1_1a_sharpe08.log

# Experiment 1.1b (ready to launch)
SHARPE_WEIGHT=1.0 RANKIC_WEIGHT=0.0 CS_IC_WEIGHT=0.0 \
python scripts/integrated_ml_training_pipeline.py --max-epochs 30 \
  --data-path output/ml_dataset_latest_full.parquet \
  > /tmp/experiment_1_1b_sharpe10.log 2>&1 &

# Experiment 1.2 (requires implementation)
# See scripts/utils/loss_curriculum.py

# Experiment 2.1 (ready to launch)
SHARPE_WEIGHT=0.8 RANKIC_WEIGHT=0.15 CS_IC_WEIGHT=0.05 \
python scripts/integrated_ml_training_pipeline.py --max-epochs 30 \
  --data-path output/ml_dataset_latest_full.parquet \
  > /tmp/experiment_2_1_hidden256.log 2>&1 &
```

---

**Created**: 2025-10-21 03:56 JST
**Based on**: `archive/training_20251021_012605/experiment_design.md`
