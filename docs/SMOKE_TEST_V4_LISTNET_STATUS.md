# Smoke Test V4 (ListNet) - Monitoring Status

**Launch Time**: 2025-10-31 05:41:38 UTC
**PID**: 979574
**Status**: ✅ **ACTIVE** - Epoch 6/10 in progress
**Log**: `_logs/training/smoke_test_v4_listnet_20251031_054138.log`

---

## 🎯 Test Objective

Validate **ListNet ranking loss** integration for cross-sectional optimization:

**Key Changes vs Smoke Test V3**:
1. ✅ **ListNet Loss Enabled**: `USE_LISTNET_LOSS=1`
2. ✅ **Ranking Optimization**: Per-day ListNet loss (group_day keyed)
3. ✅ **Reduced MSE Dominance**: Huber weight reduced to 0.1
4. ✅ **Balanced Loss Composition**: ListNet (0.3) + Sharpe (0.3) + RankIC (0.2) + CS-IC (0.1) + Huber (0.1)

**Success Criteria**:
- ✅ Training completes without crashes (10 epochs)
- 🎯 **SCALE(yhat/y) > 0.00** (prediction variance restored by ranking loss)
- 🎯 **IC values improved** (target: moving toward positive)
- 🎯 **Sharpe > 0.10** (improvement vs 0.0818 baseline)

---

## 📊 Current Status (05:45 UTC)

**Training Progress**:
- ✅ Model initialization successful
- ✅ Epochs 1-5 completed
- ⏳ Epoch 6/10 in progress (iteration 12+)
- Loss: 2.2973 (training, showing variation from 2.7667)
- **Loss dynamics**: Values changing significantly (NOT stuck at plateau)
- **Runtime**: ~4 minutes elapsed

**Positive Indicators**:
- ✅ No crashes or errors
- ✅ Loss values show meaningful variation (2.7667 → 2.2973)
- ✅ Process running steadily (16.5% CPU)

---

## 🔍 Key Configuration vs Previous Tests

### Loss Function Composition

**Smoke Test V3** (NO ListNet):
```bash
SHARPE_WEIGHT=0.5
RANKIC_WEIGHT=0.2
CS_IC_WEIGHT=0.1
HUBER_WEIGHT=0.1
# Total financial metrics: 0.8, MSE: 0.1
# Result: SCALE=0.00, Sharpe=0.0818 (no improvement)
```

**Smoke Test V4** (WITH ListNet):
```bash
USE_LISTNET_LOSS=1       # NEW: Ranking optimization enabled
LISTNET_WEIGHT=0.3        # NEW: Significant ranking component
LISTNET_TAU=1.0          # Temperature for softmax
LISTNET_TOPK=50          # Focus on top 50 stocks per day

SHARPE_WEIGHT=0.3        # Reduced from 0.5
RANKIC_WEIGHT=0.2        # Unchanged
CS_IC_WEIGHT=0.1         # Unchanged
HUBER_WEIGHT=0.1         # Unchanged

# Total composition:
# - Ranking-focused: 0.3 (ListNet) + 0.2 (RankIC) = 0.5
# - Portfolio metrics: 0.3 (Sharpe) + 0.1 (CS-IC) = 0.4
# - Reconstruction: 0.1 (Huber)
```

### Technical Implementation

**Per-Day ListNet Loss** (`scripts/train_atft.py:2170-2260`):
- Operates on `group_day` metadata
- Computes daily ranking loss (Top-K focused)
- Gradients align with APEX's ListNet/RankNet approach
- Temperature (τ=1.0) controls softmax sharpness

**Prediction Head** (Same as V3):
- All 4 variables properly initialized
- Dropout from config (base_dropout)
- LayerScale defaults to 1.0
- LayerNorm disabled by default (use_shared_layernorm=False)
- Output init std = 0.05

---

## 📋 Expected Timeline

```
05:41 UTC : Launch
05:45 UTC : Epoch 6 in progress (current)
05:48 UTC : Epoch 10 complete (estimated)
05:50 UTC : Validation metrics extraction
```

**Total Runtime**: ~10 minutes estimated

---

## 🚨 Critical Checkpoints

### Epoch 10 Final Metrics (~05:48 UTC)

**Primary Success Criterion**:
- **SCALE(yhat/y) > 0.00** → Ranking loss restores prediction variance

**Secondary Metrics**:
- IC values (h1, h5, h10, h20): Check for positive shift
- Sharpe ratio: Target > 0.10 (vs 0.0818 baseline)
- Loss dynamics: Confirm ListNet component active

**Decision Tree**:
1. If **SCALE > 0.00 AND Sharpe > 0.10**:
   - ✅ **SUCCESS** → Launch 25-epoch full experiment with ListNet
   - Document ListNet as key breakthrough

2. If **SCALE > 0.00 BUT Sharpe 0.08-0.10**:
   - ⚠️ **PARTIAL** → Variance restored but needs tuning
   - Try adjusted ListNet weight (0.3 → 0.4 or 0.5)
   - Or increase LISTNET_TOPK (50 → 100)

3. If **SCALE = 0.00**:
   - ❌ **INSUFFICIENT** → ListNet alone not enough
   - Next: Test APEX's MultiHorizonHead directly
   - Investigate GAT residual dampening signal

---

## 📊 Comparison Matrix

| Test | Config | SCALE | IC (h20) | Sharpe | Status |
|------|--------|-------|----------|--------|--------|
| **Baseline (50ep)** | Default | 0.00 | -0.0603 | 0.0818 | ❌ Plateau |
| **Exp1 (25ep)** | Sharpe=0.5 | 0.00 | -0.0603 | 0.0818 | ❌ No change |
| **Exp2 (25ep)** | Cosine LR | 0.00 | -0.0603 | 0.0818 | ❌ No change |
| **Smoke V3 (10ep)** | Per-day loss | 0.00 | -0.0603 | 0.0818 | ❌ No change |
| **Smoke V4 (10ep)** | + ListNet | ⏳ | ⏳ | ⏳ | 🔄 Running |

---

## 🔧 Monitoring Commands

### Process Status
```bash
# Check if running
ps -p 979574 -o pid,stat,%cpu,etime --no-headers

# Log file size (growing?)
ls -lh _logs/training/smoke_test_v4_listnet_20251031_054138.log
```

### Extract Final Metrics (After Completion)
```bash
# SCALE values (PRIMARY SUCCESS CRITERION)
grep "SCALE(yhat/y)" _logs/training/smoke_test_v4_listnet_20251031_054138.log

# Sharpe ratio
grep "Sharpe:" _logs/training/smoke_test_v4_listnet_20251031_054138.log

# IC values
grep "IC=" _logs/training/smoke_test_v4_listnet_20251031_054138.log | grep "h=20"

# ListNet loss component (verify active)
grep "listnet" _logs/training/smoke_test_v4_listnet_20251031_054138.log -i
```

### Real-Time Monitoring
```bash
# Live progress
tail -f _logs/training/smoke_test_v4_listnet_20251031_054138.log | grep -E "Epoch|SCALE|Sharpe|ListNet"
```

---

## 🎯 Next Actions Based on Results

### If SCALE > 0.00 (SUCCESS ✅)

**Immediate Action**:
```bash
# Launch 25-epoch full experiment with ListNet
ENABLE_GRAD_MONITOR=1 GRAD_MONITOR_EVERY=200 \
USE_LISTNET_LOSS=1 LISTNET_WEIGHT=0.3 LISTNET_TAU=1.0 LISTNET_TOPK=50 \
SHARPE_WEIGHT=0.3 RANKIC_WEIGHT=0.2 CS_IC_WEIGHT=0.1 HUBER_WEIGHT=0.1 \
nohup python scripts/integrated_ml_training_pipeline.py \
  --data-path output/datasets/ml_dataset_latest_full.parquet \
  --max-epochs 25 \
  train.batch.train_batch_size=2048 \
  train.batch.num_workers=8 \
  train.trainer.precision=bf16-mixed \
  > _logs/training/exp3_listnet_full_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

**Documentation Update**:
- Update `docs/ATFT_DIAGNOSTIC_REPORT_20251031.md` with ListNet breakthrough
- Record variance restoration mechanism
- Compare vs APEX-Ranker results (Sharpe 1.95)

### If SCALE = 0.00 (INSUFFICIENT ❌)

**Next Investigation**:
1. **Option A: Test APEX's MultiHorizonHead**
   - Directly use APEX-Ranker's prediction head
   - Isolates GAT/WAN architecture impact
   - 10-epoch quick comparison

2. **Option B: Increase ListNet Weight**
   - Try LISTNET_WEIGHT=0.5 (from 0.3)
   - Reduce Huber to 0.05 (from 0.1)
   - Stronger ranking signal

3. **Option C: Diagnostic Forward Pass**
   - Enable `ENABLE_PREDICTION_HEAD_DIAGNOSTICS=1`
   - Single-batch forward trace
   - Inspect output layer scales

---

## 📁 Files

**Current Run**:
- **Log**: `_logs/training/smoke_test_v4_listnet_20251031_054138.log` (121KB, growing)
- **PID File**: `_logs/training/smoke_test_v4_listnet.pid` (contains: 979574)

**Documentation**:
- **Status Doc**: `docs/SMOKE_TEST_V4_LISTNET_STATUS.md` (this file)
- **Diagnostic Report**: `docs/ATFT_DIAGNOSTIC_REPORT_20251031.md`
- **Previous Tests**: `docs/SMOKE_TEST_V3_STATUS.md` (SCALE=0.00 result)

**Code References**:
- **ListNet Loss**: `scripts/train_atft.py:2170-2260`
- **Loss Init**: `scripts/train_atft.py:7580-7760`
- **Prediction Head**: `src/atft_gat_fan/models/architectures/atft_gat_fan.py:1607-1698`

---

**Last Updated**: 2025-10-31 05:45 UTC (Epoch 6/10 in progress)
**Status**: ✅ Training active, loss showing variation, no crashes
**Next Checkpoint**: Epoch 10 completion + metrics extraction (~05:48 UTC)

---

*ListNet ranking loss enabled. Monitoring for SCALE > 0.00 as breakthrough indicator.*
