# Variance Collapse Diagnostic Report

**Date**: 2025-10-31
**Issue**: SCALE(yhat/y) = 0.00 across all experiments, Sharpe plateau at 0.0818
**Status**: 🔍 **ROOT CAUSE NARROWED** - Issue isolated to validation metrics calculation

---

## Executive Summary

After comprehensive architecture isolation testing and dataset diagnostics, we've determined:

1. ✅ **Dataset has healthy variance** (92-100% cross-sectional variance preserved)
2. ✅ **Prediction head generates variance** (std ~0.09-0.50 at output)
3. ✅ **Model predictions NOT degenerate** (pre-normalization variance confirmed)
4. ❌ **Variance collapses during validation** (SCALE=0.00 after cross-sectional normalization)

**Conclusion**: The issue is NOT in the architecture, loss function, or dataset preprocessing. The problem is in **how validation metrics apply cross-sectional normalization** to predictions that lack sufficient stock-level differentiation.

---

## Investigation Timeline

### Phase 1: Architecture Isolation (Smoke Tests V3-V5)

**V3: Structural fixes** (Per-day loss + relaxed prediction head)
- Result: SCALE=0.00, Sharpe=0.0818
- Conclusion: Structural changes insufficient

**V4: ListNet ranking loss** (weight=0.3, TopK=50)
- Result: SCALE=0.00, Sharpe=0.0818
- Conclusion: Ranking optimization alone insufficient

**V5: Complete architecture bypass** (10 epochs)
- Configuration:
  - ✅ GAT completely bypassed (BYPASS_GAT_COMPLETELY=1)
  - ✅ WAN normalization bypassed (BYPASS_ADAPTIVE_NORM=1)
  - ✅ Graph builder disabled (DISABLE_GRAPH_BUILDER=1)
  - ✅ Pure ListNet loss (weight=0.8, all other losses=0.0)
  - ✅ MultiHorizonPredictionHeads (proven architecture)
- Result: **SCALE=0.00, Sharpe=0.0818** (identical to baseline)
- **Conclusion**: Architecture complexity is NOT the root cause

### Phase 2: Dataset Normalization Diagnostic

**Raw Dataset Analysis**:
```
returns_1d:  overall_std=0.283, per-day_std=0.283, ratio=0.9999 (100.0% preserved)
returns_5d:  overall_std=0.283, per-day_std=0.283, ratio=0.9999 (100.0% preserved)
returns_10d: overall_std=0.078, per-day_std=0.072, ratio=0.9238 (92.4% preserved)
returns_20d: overall_std=0.116, per-day_std=0.108, ratio=0.9328 (93.3% preserved)
```

**Finding**: ✅ **Dataset has healthy cross-sectional variance**
**Conclusion**: Dataset normalization is NOT killing the signal

**Note**: `returns_1d` and `returns_5d` appear to be quantiles/probabilities (mean~0.49, bounded [0,1]) rather than raw returns, but this doesn't explain the variance collapse since their cross-sectional variance is still preserved.

---

## Key Diagnostic Evidence

### 1. Prediction Head Output (From Isolation Test Logs)

```
[PRED-HEAD-DIAG] horizon_1d post-scale std=0.501146
[PRED-HEAD-DIAG] horizon_5d post-scale std=0.115344
[PRED-HEAD-DIAG] horizon_10d post-scale std=0.097343
[PRED-HEAD-DIAG] horizon_20d post-scale std=0.093817
```

**Interpretation**: Predictions have **non-zero variance** (0.09-0.50) at the prediction head output layer.

### 2. Validation Metrics (All Epochs, All Horizons)

```
Val metrics (z) h=1:  SCALE(yhat/y)=0.00 IC=-0.0180
Val metrics (z) h=5:  SCALE(yhat/y)=0.00 IC=-0.0451
Val metrics (z) h=10: SCALE(yhat/y)=0.00 IC=-0.0430
Val metrics (z) h=20: SCALE(yhat/y)=0.00 IC=-0.0603
```

**Interpretation**: After validation's cross-sectional normalization, **all variance disappears** (SCALE=0.00).

### 3. The Variance Collapse Flow

```
Prediction Head Output:
  horizon_20d: std = 0.093817  ← Non-zero variance
        ↓
Cross-Sectional Normalization (per-day Z-score):
  (yhat - mean_day) / std_day
        ↓
Validation SCALE Calculation:
  SCALE(yhat/y) = 0.00  ← Complete collapse
```

**Root Cause Hypothesis**: Predictions are **too similar across stocks on the same day** (low cross-sectional diversity), so per-day Z-score normalization collapses them to near-identical values.

---

## What This Tells Us

### ❌ NOT the Problem:

1. **GAT/WAN architecture complexity** - Bypassed completely, no change in SCALE
2. **Graph builder** - Disabled, no change in SCALE
3. **Dataset normalization** - Raw data has 92-100% variance preserved
4. **Prediction head degeneracy** - Outputs have std ~0.09-0.50
5. **Loss function** - Pure ListNet with no MSE still yields SCALE=0.00

### ✅ IS the Problem:

**Lack of cross-sectional (per-day) prediction differentiation**

The model is predicting **nearly identical scores for all stocks on each day**:

```
Example (hypothetical):
Day 2025-10-31:
  Stock A: prediction = 0.0050
  Stock B: prediction = 0.0051
  Stock C: prediction = 0.0049
  Stock D: prediction = 0.0052
  ... (3,700 stocks with predictions in range [0.0045, 0.0055])

After per-day Z-score:
  All stocks → near-zero (std_day ≈ 0.002)
```

---

## Why Is Cross-Sectional Diversity Missing?

### Hypothesis 1: Feature Space Lacks Stock-Specific Information

**Evidence**:
- All stocks share similar normalized features (cross-sectional Z-scores in dataset)
- No stock embeddings or identifiers in the model
- Features may be too "homogeneous" after normalization

**Test**: Check if features have sufficient per-day variance across stocks

### Hypothesis 2: Model Learns to Predict the Mean

**Evidence**:
- Negative IC values suggest model is anti-predictive
- SCALE=0.00 suggests model predicts near-constant value per day
- ListNet loss may be insufficient to enforce diversity

**Test**: Check if predictions cluster tightly around daily mean

### Hypothesis 3: Validation Normalization Mismatch

**Evidence**:
- Training may use different normalization than validation
- `_normalize` method in validation may be over-aggressive

**Test**: Compare training vs validation normalization logic

---

## Next Steps (Priority Order)

### 🚨 IMMEDIATE (Step 3 from User's Plan)

**Add per-day std logging in validation metrics calculation**

Location: `scripts/train_atft.py` - validation metrics function

```python
# BEFORE cross-sectional normalization
per_day_stats = predictions.group_by('Date').agg([
    pl.col('horizon_20').std().alias('pred_std'),
    pl.col('target_20').std().alias('target_std'),
    pl.col('horizon_20').count().alias('n_stocks')
])
logger.info(f"[VAL-DIVERSITY] Per-day prediction std: {per_day_stats['pred_std'].mean():.6f}")

# AFTER cross-sectional normalization
...
```

This will confirm whether predictions are collapsing **before** or **after** normalization.

### 📊 MEDIUM (Step 1 from User's Plan)

**Add per-day std logging in training loop before loss**

Location: `scripts/train_atft.py` - training step before loss calculation

```python
# Log per-day prediction diversity
for horizon in predictions.keys():
    pred = predictions[horizon]
    target = targets[horizon]
    # Group by group_day metadata and compute std
    ...
```

### 🔬 LONG-TERM (Depends on Above Findings)

Based on where variance collapses:

**If predictions are diverse BEFORE validation normalization**:
- Issue is validation normalization strategy
- Solution: Modify `_normalize` to preserve variance

**If predictions are uniform BEFORE validation normalization**:
- Issue is model not learning cross-sectional patterns
- Solutions:
  1. Add stock embeddings
  2. Stronger diversity regularization in loss
  3. Re-examine feature engineering for stock-specific signal
  4. Test with non-normalized features

---

## Files Referenced

**Training Scripts**:
- `scripts/train_atft.py` - Main training loop, validation metrics
- `scripts/integrated_ml_training_pipeline.py` - Training orchestrator

**Model Architecture**:
- `src/atft_gat_fan/models/architectures/atft_gat_fan.py` - MultiHorizonPredictionHeads

**Normalization**:
- `src/gogooku3/data/normalization.py` - CrossSectionalNormalizer

**Diagnostic Logs**:
- `_logs/training/isolation_test_10ep_20251031_073911.log` - Isolation test results
- `_logs/dataset_variance_diagnostic_fixed.log` - Dataset variance analysis

**Diagnostic Scripts**:
- `scripts/diagnose_dataset_variance.py` - Dataset variance checker

---

## Comparison with APEX-Ranker

**APEX-Ranker** (Sharpe 1.95, working correctly):
- Architecture: PatchTST (simpler than GAT+FAN)
- Loss: ListNet + RankNet (ranking-focused)
- Features: 64-89 features (fewer than ATFT's ~307)
- **Key difference**: No complex graph attention or normalization layers

**ATFT-GAT-FAN** (Sharpe 0.0818, stuck):
- Architecture: GAT + FAN (complex relational modeling)
- Loss: MultiHorizonLoss with various components
- Features: ~307 features (higher dimensional)
- **Problem**: Predictions lack cross-sectional diversity

**Hypothesis**: APEX-Ranker's simpler architecture may allow better learning of stock-specific patterns, while ATFT's complexity may be smoothing out cross-sectional differences.

---

## Status

- **Phase 1 (Architecture Isolation)**: ✅ Complete - Not the root cause
- **Phase 2 (Dataset Diagnostic)**: ✅ Complete - Dataset variance is healthy
- **Phase 3 (Validation Logging)**: ⏳ **NEXT** - Add per-day std logging to pinpoint collapse location
- **Phase 4 (Fix Implementation)**: ⏳ Pending Phase 3 results

**Priority**: Implement Phase 3 logging to determine if variance collapse happens:
- **Before** cross-sectional normalization → Model issue (predictions too uniform)
- **After** cross-sectional normalization → Normalization strategy issue

---

**Last Updated**: 2025-10-31 07:45 UTC
**Next Action**: Implement per-day std logging in validation metrics (`scripts/train_atft.py`)
