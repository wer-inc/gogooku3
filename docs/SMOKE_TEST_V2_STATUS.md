# Smoke Test V2 - Monitoring Status

**Launch Time**: 2025-10-31 04:21:51 UTC
**PID**: 959590
**Status**: ✅ **ACTIVE** - Initializing data loaders
**Log**: `_logs/training/smoke_test_v2_20251031_042151.log`

---

## 🎯 Test Objective

Validate structural fixes to prediction head:
1. **Per-day loss threading** (group_day, sid, exposure metadata)
2. **Relaxed prediction head** (dropout from config, LayerScale=1.0, LayerNorm optional, output init std=0.05)
3. **base_dropout fix** (properly initialized in MultiHorizonPredictionHeads)

**Success Criteria**:
- ✅ Training completes without crashes (10 epochs)
- 🎯 **SCALE(yhat/y) > 0.00** (prediction variance restored)
- 🎯 **IC values** (check if still negative or improved)
- 🎯 **Sharpe > 0.08** (any improvement vs baseline 0.0818)

---

## 📊 Current Status (04:24 UTC)

**Initialization Phase**:
- ✅ GPU detected: NVIDIA A100-SXM4-80GB (85GB memory)
- ✅ Model hidden_size=256 configured
- ⚠️ DataLoader forced to single-process mode (num_workers=0) - safety guard active
- ✅ Found 100 train files, 100 val files
- ✅ OnlineRobustScaler fitted on 8,192 samples
- ✅ Global date filter: 2016-01-01 (ensures sufficient history)
- ⏳ Creating data loaders...

**Runtime**: ~3 minutes elapsed (still initializing)

---

## 🔍 Key Changes vs Previous Experiments

### What's Different (Structural Fixes)

**1. Per-Day Loss Threading** (scripts/train_atft.py:2070-2280):
- Sharpe/RankIC/CS-IC now computed **per trading day** (not across mixed batches)
- Exposure neutralization and turnover penalties operate per day
- Gradients aligned with APEX's ListNet/RankNet optimization

**2. Relaxed Prediction Head** (atft_gat_fan.py:1526-1584):
- Dropout from config/env (not hardcoded)
- LayerScale defaults to 1.0 (was constrained)
- LayerNorm optional (off by default to preserve variance)
- Output weights init std=0.05 (was smaller)
- Shared encoder drops hard LayerNorm

**3. base_dropout Fix** (atft_gat_fan.py:1610-1617):
- Properly initialized in `MultiHorizonPredictionHeads.__init__`
- Matches `QuantilePredictionHead` initialization logic
- Supports env override via `PRED_HEAD_DROPOUT`

### What's Same (For Comparison)

**Loss Weights**:
```bash
SHARPE_WEIGHT=0.5     # Same as Experiment 1
RANKIC_WEIGHT=0.2     # Same as Experiment 1
CS_IC_WEIGHT=0.1      # Same as Experiment 1
HUBER_WEIGHT=0.1      # Same as Experiment 1
```

**Training Config**:
- Max epochs: 10 (smoke test)
- Batch size: 2048
- Precision: bf16-mixed
- Grad monitoring: ENABLE_GRAD_MONITOR=1, EVERY=200

---

## 📋 Expected Timeline

```
04:21 UTC : Launch
04:24 UTC : Data loader initialization (current)
04:27 UTC : First epoch training start (estimated)
04:30 UTC : First validation metrics (estimated)
04:35 UTC : Epoch 5 validation (critical checkpoint)
04:55 UTC : Epoch 10 complete (final evaluation)
```

**Total Runtime**: ~30-35 minutes estimated

---

## 🚨 Critical Checkpoints

### Epoch 1 Validation (~04:30 UTC)
**Look for**:
- `SCALE(yhat/y)` value (hoping for > 0.00)
- IC values (hoping for improvement vs -0.018, -0.045, -0.043, -0.060)
- Sharpe value (baseline: 0.0818)

**Decision**:
- If SCALE > 0.00 → ✅ Variance restored! Continue monitoring
- If SCALE = 0.00 → ⚠️ Structural fixes insufficient, needs deeper investigation

### Epoch 5 Validation (~04:35 UTC)
**Look for**:
- Sharpe trend (increasing, stable, or decreasing)
- IC trend (moving toward positive)
- Prediction variance stability

**Decision**:
- If Sharpe > 0.09 → ✅ Improvement! Approve 25-epoch run
- If Sharpe 0.08-0.09 → ⚠️ Marginal, continue to epoch 10
- If Sharpe < 0.08 → ❌ No improvement, investigate further

### Epoch 10 Final (~04:55 UTC)
**Evaluation**:
- Compare all metrics vs Experiment 1/2 baseline
- If SCALE > 0.00 and Sharpe > 0.10 → **Launch 25-epoch full run**
- If SCALE > 0.00 but Sharpe < 0.10 → **Try different config**
- If SCALE = 0.00 → **Architecture investigation required**

---

## 📊 Baseline Comparison

**Previous Experiments** (WITHOUT structural fixes):

| Experiment | Config | Sharpe | IC (h20) | SCALE | Status |
|------------|--------|--------|----------|-------|--------|
| Baseline (50ep) | Default | 0.0818 | -0.0603 | 0.00 | ❌ Plateau |
| Exp1 (25ep) | Sharpe=0.5 | 0.0818 | -0.0603 | 0.00 | ❌ No change |
| Exp2 (25ep) | Cosine LR | 0.0818 | -0.0603 | 0.00 | ❌ No change |

**Current Smoke Test V2** (WITH structural fixes):

| Metric | Target | Status |
|--------|--------|--------|
| **SCALE(yhat/y)** | > 0.00 | ⏳ Pending (epoch 1) |
| **IC (h20)** | > -0.06 | ⏳ Pending (epoch 1) |
| **Sharpe** | > 0.10 | ⏳ Pending (epoch 5) |

---

## 🔧 Monitoring Commands

### Quick Status
```bash
# Process status
ps -p 959590 -o pid,stat,%cpu,etime --no-headers

# Latest log output
tail -30 _logs/training/smoke_test_v2_20251031_042151.log

# Find epoch completions
grep -E "Epoch [0-9]+/10" _logs/training/smoke_test_v2_20251031_042151.log | tail -5
```

### Real-Time Monitoring
```bash
# Live progress
tail -f _logs/training/smoke_test_v2_20251031_042151.log | grep -E "Epoch|train/total_loss|val/loss|SCALE|Sharpe"

# Validation metrics only
tail -f _logs/training/smoke_test_v2_20251031_042151.log | grep "Val metrics"
```

### Critical Metrics Extraction
```bash
# After completion, extract all validation SCALE values
grep "SCALE(yhat/y)" _logs/training/smoke_test_v2_20251031_042151.log

# Extract all Sharpe values
grep "Sharpe:" _logs/training/smoke_test_v2_20251031_042151.log

# Extract all IC values
grep "IC=" _logs/training/smoke_test_v2_20251031_042151.log | grep "h=20"
```

---

## 🎯 Next Actions Based on Results

### If SCALE > 0.00 (Variance Restored) ✅

**Immediate**:
1. Document variance restoration in diagnostic report
2. Compare IC/Sharpe vs baseline
3. Launch 25-epoch full experiment with same config

**Command** (25-epoch full run):
```bash
ENABLE_GRAD_MONITOR=1 GRAD_MONITOR_EVERY=200 \
SHARPE_WEIGHT=0.5 RANKIC_WEIGHT=0.2 CS_IC_WEIGHT=0.1 HUBER_WEIGHT=0.1 \
nohup python scripts/integrated_ml_training_pipeline.py \
  --data-path output/datasets/ml_dataset_latest_full.parquet \
  --max-epochs 25 \
  train.batch.train_batch_size=2048 \
  train.batch.num_workers=8 \
  train.trainer.precision=bf16-mixed \
  > _logs/training/exp3_full_perday_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### If SCALE = 0.00 (Variance Still Collapsed) ❌

**Investigation Required**:
1. Check if per-day loss is actually active (grep for `[LOSS-PER-DAY]` logs)
2. Verify prediction head initialization (grep for `[PRED-HEAD-INIT]` logs)
3. Examine output layer gradients (grep for `prediction_head` in grad monitors)
4. Consider explicit variance flooring or de-normalization in heads

**Possible Next Steps**:
- Increase output init std (0.05 → 0.10)
- Add explicit variance floor in loss function
- Test with different LayerScale values (1.0 → 0.1 or 10.0)
- Investigate target scaling in data preprocessing

---

## 📁 Files

**Log**: `_logs/training/smoke_test_v2_20251031_042151.log`
**PID**: `_logs/training/smoke_test_v2.pid` (contains: 959590)
**Fix Doc**: `docs/URGENT_FIX_base_dropout.md`
**Diagnostic**: `docs/ATFT_DIAGNOSTIC_REPORT_20251031.md`

---

**Last Updated**: 2025-10-31 04:24 UTC (Initializing data loaders)
**Status**: ✅ Training process active, no crashes
**Next Checkpoint**: Epoch 1 validation (~04:30 UTC)

---

*Smoke test running autonomously. Monitoring for SCALE > 0.00 as primary success criterion.*
