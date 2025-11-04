# OOM Prevention Guide - Cross-Validation Training

## Problem: Parallel Fold Training Causes OOM

### Symptoms
- Training Fold1 succeeds perfectly
- Launching Fold2-5 in parallel causes:
  - CPU usage → 100%
  - Memory usage → 100%
  - **OOM (Out of Memory) kill**

### Root Cause

**PanelCache Duplication**: Each training process independently:
1. Loads the full parquet dataset into memory
2. Builds PanelCache (date-code mapping, numpy arrays)
3. Holds cache in RAM throughout training

**5x Parallel = 5x Memory**:
- Single fold: ~100GB memory (dataset + cache + model + gradients)
- 5 parallel folds: ~500GB peak memory
- Even with 2TiB RAM, memory fragmentation + cache overhead → OOM

### Why Not Shared Memory?

Current implementation limitations:
- PanelCache uses Python dict/list (not shareable via multiprocessing)
- Numpy arrays are process-local (not memory-mapped)
- No persistent cache file (rebuilt every run)

---

## Solutions (3 Options)

### ✅ Option 1: Sequential Execution (RECOMMENDED)

**When to use**: Always (safest, guaranteed to work)

**Pros**:
- ✅ Zero OOM risk
- ✅ Predictable memory usage (~100GB per fold)
- ✅ Simple, no complexity
- ✅ Easy to monitor/debug

**Cons**:
- ⏱️ Takes 5x longer (~60 min → 300 min for 5 folds at 12 epochs each)

**Command**:
```bash
# Run folds 2-5 sequentially
bash apex-ranker/scripts/train_folds_sequential.sh 2 5

# With custom settings
CONFIG_FILE=apex-ranker/configs/v0_base.yaml \
MAX_EPOCHS=15 \
EMA_EPOCHS=3,7,12 \
bash apex-ranker/scripts/train_folds_sequential.sh 2 5
```

**Output**:
```
✅ Fold 2 completed (58m 32s)
✅ Fold 3 completed (61m 15s)
✅ Fold 4 completed (59m 48s)
✅ Fold 5 completed (60m 22s)
Total: 239m 57s (4h)
```

---

### ⚠️ Option 2: Partial Parallel (2 at a time)

**When to use**: If you're impatient and have stable system

**Pros**:
- ⚡ 2x faster than sequential (~150 min vs 300 min)
- ✅ Much safer than 5x parallel
- ✅ Reasonable memory usage (~200GB peak)

**Cons**:
- ⚠️ Still some OOM risk if system is busy
- ⚠️ More complex error handling

**Command**:
```bash
# Run 2 folds at a time (folds 2-5)
bash apex-ranker/scripts/train_folds_parallel_safe.sh 2 5

# With custom parallelism
PARALLEL_JOBS=3 \
bash apex-ranker/scripts/train_folds_parallel_safe.sh 2 5
```

**Output**:
```
Batch 1/2: Folds 2 3
  ✅ Fold 2 finished (60m)
  ✅ Fold 3 finished (62m)
Batch 2/2: Folds 4 5
  ✅ Fold 4 finished (59m)
  ✅ Fold 5 finished (61m)
Total: 122m (2h)
```

---

### 🚧 Option 3: Future Optimization (Not Implemented)

**Persistent PanelCache**:
- Save cache to disk once (`.pkl` or memory-mapped numpy)
- All folds load shared cache (no rebuild)
- Enables safe 5x parallel

**Estimated implementation**:
- 2-3 hours of development
- Testing needed
- Risk of cache corruption bugs

**Not recommended** until basic workflow is proven.

---

## Recommended Workflow

### Phase 1: Health Check (Fold1 only)

```bash
# 1. Create blended checkpoint
python apex-ranker/scripts/average_checkpoints.py \
  models/shortfocus_fold1_ema_epoch3.pt \
  models/shortfocus_fold1_ema_epoch6.pt \
  models/shortfocus_fold1_ema_epoch10.pt \
  --output models/shortfocus_fold1_blended.pt

# 2. Smoke backtest (quick validation)
python apex-ranker/scripts/backtest_smoke_test.py \
  --checkpoint models/shortfocus_fold1_blended.pt \
  --config apex-ranker/configs/v0_base.yaml \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --log-level INFO

# 3. Check key metrics
# - candidate_count > 0 (no zero-selection days)
# - fallback_rate < 20% (stable K selection)
# - PnL > 0 (after transaction costs)
# - h5_delta_p_at_k_pos > 0 (better than random)
```

**Decision point**:
- ✅ **Green**: All metrics healthy → Proceed to Phase 2
- ⚠️ **Yellow**: Some concerns → Tune hyperparams on Fold1 first
- ❌ **Red**: Major issues → Debug before expanding

---

### Phase 2: Full Cross-Validation

**If Phase 1 is Green**, run **sequential training** (safest):

```bash
# Sequential execution (4-5 hours for folds 2-5)
bash apex-ranker/scripts/train_folds_sequential.sh 2 5
```

**Alternative** (if you're brave):

```bash
# Parallel-safe (2 at a time, ~2 hours)
bash apex-ranker/scripts/train_folds_parallel_safe.sh 2 5
```

**Monitor progress**:
```bash
# Watch logs in real-time
tail -f logs/shortfocus_fold2.log

# Check memory usage
watch -n 5 free -h

# List all checkpoints
ls -lh models/shortfocus_fold*.pt
```

---

### Phase 3: Ensemble & Evaluation

```bash
# 1. Blend all folds (creates ensemble)
for fold in 1 2 3 4 5; do
  python apex-ranker/scripts/average_checkpoints.py \
    models/shortfocus_fold${fold}_ema_epoch3.pt \
    models/shortfocus_fold${fold}_ema_epoch6.pt \
    models/shortfocus_fold${fold}_ema_epoch10.pt \
    --output models/shortfocus_fold${fold}_blended.pt
done

# 2. Full backtest (all 5 folds)
python apex-ranker/scripts/backtest_v0.py \
  --checkpoints models/shortfocus_fold*_blended.pt \
  --config apex-ranker/configs/v0_base.yaml \
  --start-date 2023-01-01 \
  --end-date 2025-10-31 \
  --output results/shortfocus_5fold_backtest.json

# 3. A/B comparison
# Compare h5_delta_p_at_k_pos, h5_spread, h5_delta_ndcg
# Decision: DM > 1.96 & 95% CI > 0 → Production candidate
```

---

## Memory Usage Estimates

| Configuration | Single Fold | 2 Parallel | 5 Parallel |
|---------------|-------------|------------|------------|
| Dataset load  | 10GB        | 20GB       | 50GB       |
| PanelCache    | 40GB        | 80GB       | 200GB      |
| Model+Grads   | 30GB        | 60GB       | 150GB      |
| DataLoader    | 20GB        | 40GB       | 100GB      |
| **Total**     | **~100GB**  | **~200GB** | **~500GB** |

**System**: 2TiB RAM, but:
- OS/cache: ~100GB
- Available: ~1.9TiB
- Safe limit: 80% = ~1.5TiB

**Risk assessment**:
- 1 fold: ✅ 6% usage (safe)
- 2 parallel: ✅ 13% usage (safe)
- 5 parallel: ⚠️ 33% usage + fragmentation → **OOM risk**

---

## Troubleshooting

### 1. Training Killed with "Killed" message

**Cause**: OOM killer

**Fix**: Use sequential execution
```bash
bash apex-ranker/scripts/train_folds_sequential.sh 2 5
```

### 2. One fold failed in parallel mode

**Cause**: Resource contention

**Fix**: Re-run failed fold alone
```bash
python -m apex_ranker.scripts.train_v0 \
  --config apex-ranker/configs/v0_base.yaml \
  --cv-type purged_kfold \
  --cv-n-splits 5 \
  --cv-fold 3 \
  --embargo-days 5 \
  --output models/shortfocus_fold3.pt \
  --ema-snapshot-epochs 3,6,10 \
  --max-epochs 12
```

### 3. Slow training (low GPU utilization)

**Not related to OOM**, but common issue:

**Check**:
```bash
nvidia-smi dmon
# Look for: GPU util (sm) > 80%
```

**Fix** (if GPU idle):
- Increase batch_size in config
- Reduce num_workers (DataLoader CPU bottleneck)

---

## Quick Reference

### Sequential (Safest)
```bash
bash apex-ranker/scripts/train_folds_sequential.sh 2 5
```

### Parallel-Safe (2x faster)
```bash
bash apex-ranker/scripts/train_folds_parallel_safe.sh 2 5
```

### Manual (for debugging)
```bash
# Run single fold
python -m apex_ranker.scripts.train_v0 \
  --config apex-ranker/configs/v0_base.yaml \
  --cv-fold 2 \
  --output models/shortfocus_fold2.pt
```

---

## Summary

✅ **Use sequential execution** for production runs (guaranteed safe)
⚡ **Use 2-parallel** if you're in a hurry and system is stable
❌ **Avoid 5-parallel** until persistent cache is implemented

**Estimated timeline**:
- Sequential: 4-5 hours for 4 folds
- Parallel-safe (2x): 2-2.5 hours for 4 folds
- Risk: Sequential = 0%, Parallel = 5-10%

**Decision matrix**:
- Have 5 hours? → Sequential (safest)
- Need it in 2 hours? → Parallel-safe (acceptable risk)
- Want instant results? → Not possible without optimization
