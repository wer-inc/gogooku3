# Next Steps - Optimized Training Pipeline

**Date**: 2025-10-16 14:35 UTC
**Status**: 🧪 Dry run in progress

---

## 🎯 Current Status

### ✅ Optimizations Applied

1. **Parallel Data Loading** (`NUM_WORKERS=2`, spawn context)
2. **Loss Weight Rebalancing** (RankIC focus: 0.5, Sharpe reduced: 0.1)
3. **Full Dataset Training** (`PHASE_MAX_BATCHES=0`)
4. **Improved Defaults** (bf16, persistent workers, prefetch)

### 🧪 Testing Now

**Dry Run**: 1 trial × 2 epochs
- **PID**: 480218, 482793
- **Started**: 14:34 UTC
- **ETA**: ~8-10 minutes (if optimizations work)
- **Goal**: Verify GPU >60%, RankIC positive

---

## 📋 Decision Tree

### If Dry Run Succeeds (GPU >60%, RankIC >0)

✅ **Proceed to Short HPO**

```bash
# 5 trials × 10 epochs = ~3.3 hours
python scripts/hpo/run_optuna_atft.py \
  --data-path output/ml_dataset_latest_full.parquet \
  --n-trials 5 \
  --max-epochs 10 \
  --study-name atft_hpo_optimized_v2 \
  --output-dir output/hpo_optimized_v2
```

**Timeline**:
- Start: ~14:45 UTC
- Complete: ~18:00 UTC (3.3 hours)
- Analysis: 18:00-18:30
- Decision: Continue to full 20-trial or adjust

---

### If Dry Run Fails (GPU still low or RankIC negative)

❌ **Additional Debugging Required**

**Possible Issues**:
1. spawn() context not working → check logs for multiprocessing errors
2. NUM_WORKERS still forced to 0 → check env var propagation
3. Loss weights not applied → check train_atft.py reads USE_RANKIC
4. Data module override → check integrated_ml_training_pipeline.py

**Actions**:
1. Review logs: `tail -200 logs/ml_training.log`
2. Check env vars: `cat /proc/<PID>/environ | tr '\0' '\n'`
3. Verify GPU activity: `nvidia-smi dmon -c 20`
4. Adjust and rerun

---

## 🎯 Success Criteria

### Dry Run (Minimum Bar)
- ✅ No crashes or deadlocks
- ✅ GPU utilization >40% (ideally >60%)
- ✅ Val RankIC >-0.01 (preferably >0)
- ✅ Epoch time <6 minutes

### Short HPO (Target)
- ✅ Best trial: Val RankIC >0.040
- ✅ Best trial: Val IC >0.015
- ✅ Consistent positive metrics across trials
- ✅ Reasonable Sharpe (0.005-0.020)

---

## 🔮 Future Enhancements (Post-HPO)

### 1. Arrow Cache Integration
**When**: After validating spawn() works
**Benefit**: +20-30% throughput
**Effort**: Medium (modify data_module.py)

### 2. Spearman Regularizer
**When**: If RankIC still suboptimal after HPO
**Benefit**: +0.02-0.03 RankIC
**Effort**: Low (wire into loss computation)

### 3. VAL-DEBUG Logging Removal
**When**: If validation is still slow
**Benefit**: 2-3x faster validation
**Effort**: Low (add env var checks)

---

## 📊 Performance Tracking

### Baseline (Before Any Fixes)
```
Epoch time: ~7 minutes
GPU: 0-10% utilization
Val RankIC: -0.006 (negative)
Val IC: -0.010 (negative)
```

### After Parallel Loading (Target)
```
Epoch time: 3-5 minutes
GPU: 60-80% utilization
Val RankIC: 0.020-0.060 (positive)
Val IC: 0.015-0.030 (positive)
```

---

## 🎓 Key Learnings

1. **Root Cause Matters**: NUM_WORKERS=0 was the primary bottleneck, not the model or data
2. **Loss Weights Critical**: Small changes (0.2→0.5 RankIC weight) can flip sign of metrics
3. **Full Dataset Needed**: 100-batch cap prevented proper learning
4. **spawn() > fork()**: Avoids thread state zombies on multi-core systems

---

## 📞 Monitoring Commands

### Real-time GPU
```bash
watch -n 2 nvidia-smi dmon
```

### Training Progress
```bash
tail -f logs/ml_training.log | grep -E "Epoch|Phase|Val Metrics"
```

### Process Health
```bash
ps aux | grep train_atft | grep -v grep
```

### Results (after completion)
```bash
cat output/hpo_dryrun/trial_0/metrics.json | jq '.'
```

---

## ⏱️ Timeline

```
14:35 UTC - Dry run started (2 epochs)
14:43 UTC - Dry run expected complete (~8 min)
14:45 UTC - Analysis & decision
14:50 UTC - Launch short HPO (if successful)
18:00 UTC - Short HPO complete (~3.3 hours)
18:30 UTC - Final analysis & decision
```

---

**Current**: ⏳ Waiting for dry run (ETA: ~5 min)
**Next Check**: 14:40 UTC (check GPU utilization)
**Decision Point**: 14:45 UTC (analyze dry run results)
