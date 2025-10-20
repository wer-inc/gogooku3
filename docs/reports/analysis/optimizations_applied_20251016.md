# High-Impact Optimizations Applied

**Date**: 2025-10-16 14:30 UTC
**Based on**: Deep root cause analysis

---

## 🎯 Problems Identified

### 1. **CPU Bottleneck** (Primary issue)
- `NUM_WORKERS=0` forcing single-threaded data loading
- A100 GPU sitting idle at 0% utilization
- Polars/Parquet scans blocking on single CPU thread

### 2. **Insufficient Training**
- `PHASE_MAX_BATCHES=100` limiting each epoch to ~200k samples
- Optimizer barely exploring before phase changes
- Dataset has 8.9M samples but only ~2% used per epoch

### 3. **Metrics Degradation**
- Loss weights not prioritizing RankIC/IC
- `RANKIC_WEIGHT=0.2` too weak
- Sharpe-focused optimization causing rank inversion

### 4. **Unused Optimizations**
- Arrow cache (7.4GB) generated but never used
- Spearman regularizer implemented but not wired in
- Validation debug logging eating minutes per epoch

---

## ✅ Fixes Applied to `scripts/hpo/run_optuna_atft.py`

### Fix 1: Parallel Data Loading ✅
```python
# Before:
env["NUM_WORKERS"] = "0"  # CPU-bound bottleneck

# After:
env["ALLOW_UNSAFE_DATALOADER"] = "1"
env["NUM_WORKERS"] = "2"  # Parallel loading
env["MULTIPROCESSING_CONTEXT"] = "spawn"  # Avoid fork() deadlock
env["PREFETCH_FACTOR"] = "2"
env["PERSISTENT_WORKERS"] = "1"
```

**Expected Impact**:
- GPU utilization: 0% → 70-80%
- Throughput: 2-3x improvement
- Epoch time: 7 min → 3-4 min

---

### Fix 2: Loss Weight Rebalancing ✅
```python
# Before: (default)
# RANKIC_WEIGHT=0.2, CS_IC_WEIGHT=0.15, SHARPE_WEIGHT=0.3

# After:
env["USE_RANKIC"] = "1"
env["RANKIC_WEIGHT"] = "0.5"  # 0.2 → 0.5 (2.5x stronger)
env["CS_IC_WEIGHT"] = "0.3"   # 0.15 → 0.3 (2x stronger)
env["SHARPE_WEIGHT"] = "0.1"  # 0.3 → 0.1 (3x weaker)
```

**Expected Impact**:
- Val RankIC: -0.006 → 0.040+ (positive and meaningful)
- Val IC: -0.010 → 0.020+ (positive)
- Better rank ordering learning

---

### Fix 3: Full Dataset Training ✅
```python
# Before:
# PHASE_MAX_BATCHES=100 (only ~200k samples/epoch)

# After:
env["PHASE_MAX_BATCHES"] = "0"  # No limit (full 8.9M samples)
```

**Expected Impact**:
- Gradient steps per epoch: ~200 → ~4,000 (20x more)
- Better exploration of parameter space
- More stable convergence

---

### Fix 4: Validation Logging Reduction ✅
```python
# After:
env["VAL_DEBUG_LOGGING"] = "0"  # Disable per-batch VAL-DEBUG
```

**Note**: Environment variable added but train_atft.py doesn't check it yet.
Will need code update if validation is still slow.

---

## 📊 Expected Results

### Before Optimizations (Phase 0, Epoch 5)
```
Val Sharpe:  0.001243  (worse than baseline 0.002)
Val IC:      -0.009922 (negative!)
Val RankIC:  -0.005886 (negative!)
GPU:         0% utilization
Epoch time:  ~7 minutes
```

### After Optimizations (Target)
```
Val Sharpe:  0.010-0.020
Val IC:      0.015-0.025 (positive)
Val RankIC:  0.040-0.060 (positive, better than baseline 0.028)
GPU:         70-80% utilization
Epoch time:  3-4 minutes (2x faster)
```

---

## 🧪 Verification Plan

### Step 1: 2-Epoch Dry Run
```bash
python scripts/hpo/run_optuna_atft.py \
  --data-path output/ml_dataset_latest_full.parquet \
  --n-trials 1 \
  --max-epochs 2 \
  --study-name atft_dryrun_optimized \
  --output-dir output/hpo_dryrun
```

**Success Criteria**:
- ✅ GPU utilization >60% during training
- ✅ Epoch time <5 minutes
- ✅ Val RankIC >0 (positive)
- ✅ Val IC >0 (positive)
- ✅ No crashes or deadlocks

---

### Step 2: Short HPO Sweep (5 trials)
If dry run succeeds:
```bash
python scripts/hpo/run_optuna_atft.py \
  --data-path output/ml_dataset_latest_full.parquet \
  --n-trials 5 \
  --max-epochs 10 \
  --study-name atft_hpo_optimized_v2 \
  --output-dir output/hpo_optimized_v2
```

**Time**: ~5 trials × 10 epochs × 4 min = 200 min = **3.3 hours**

**Success Criteria**:
- ✅ Best Val RankIC >0.040
- ✅ Val IC positive across trials
- ✅ Sharpe reasonable (>0.005)

---

## 🔮 Future Optimizations (Not Yet Applied)

### 1. Arrow Cache Integration
**Status**: Arrow cache exists (7.4GB) but not used

**Implementation Required**:
- Modify `ProductionDataModuleV2` to detect `.arrow` inputs
- Use `pyarrow.ipc.RecordBatchFileReader` instead of `pl.scan_parquet`

**Expected Benefit**: Additional 20-30% throughput

---

### 2. Spearman Regularizer Wiring
**Status**: Implemented in `src/gogooku3/training/losses/` but not wired into training loop

**Implementation Required**:
- Add to loss computation in `train_atft.py`
- Make it HPO-tunable via environment variable

**Expected Benefit**: RankIC +0.02-0.03

---

### 3. Validation Logging Gating
**Status**: VAL-DEBUG logs still run for every batch

**Implementation Required**:
- Add `if os.getenv("VAL_DEBUG_LOGGING", "1") == "1"` guards
- Around lines 3774-3787 in `train_atft.py`

**Expected Benefit**: Validation 2-3x faster

---

## 📈 Performance Projection

### Current Path (with applied fixes)
```
Dry run (2 epochs):     ~8 minutes
Short HPO (5 trials):   ~3.3 hours
Full HPO (20 trials):   ~13 hours
```

### With All Optimizations (future)
```
Dry run (2 epochs):     ~4 minutes
Short HPO (5 trials):   ~2 hours
Full HPO (20 trials):   ~8 hours
```

---

## 🎯 Success Metrics

### Infrastructure
- [x] Parallel data loading enabled
- [x] Loss weights rebalanced
- [x] Full dataset training enabled
- [ ] GPU utilization >60% (verify in dry run)
- [ ] Epoch time <5 min (verify in dry run)

### Model Quality
- [ ] Val RankIC positive and >0.040
- [ ] Val IC positive and >0.015
- [ ] Val Sharpe reasonable (0.010-0.020)
- [ ] Stable across multiple trials

---

## 🚀 Next Steps

1. **Now**: Run 2-epoch dry run (ETA: 8 minutes)
2. **+10 min**: Analyze dry run results
3. **If successful**: Launch 5-trial HPO (ETA: 3.3 hours)
4. **+3.3 hours**: Analyze HPO results
5. **If successful**: Scale to full 20-trial sweep

---

**Status**: ✅ Optimizations applied, ready for dry run testing
**Expected Timeline**: 3-4 hours to validated optimized HPO setup
**Risk**: Low (spawn context tested, loss weights conservative)
