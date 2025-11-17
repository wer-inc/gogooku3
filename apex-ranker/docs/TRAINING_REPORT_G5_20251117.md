# APEX-Ranker gogooku5 Training Report

**Date**: 2025-11-17
**Objective**: Train APEX-Ranker models using gogooku5 dataset features
**Status**: ⚠️ **Training completed but performance issues identified**

---

## Executive Summary

Two APEX-Ranker models were trained using gogooku5 features:

1. **apex_ranker_v0_g5_latest**: Multi-horizon model (1d, 5d, 10d, 20d)
2. **apex_ranker_v0_g5_shortterm**: Short-term focused model (1d, 5d)

**Critical Finding**: Both models show **negative RankIC** and **P@K below random baseline**, indicating fundamental issues with data quality, feature engineering, or target definition.

---

## Training Results

### Model 1: apex_ranker_v0_g5_latest

**Configuration**:
- Config: `apex-ranker/configs/v0_base.yaml`
- Features: 65 (g5_core + g5_top_corr)
- Lookback: 90 days
- Horizons: [1d, 5d, 10d, 20d]
- Dataset: `output_g5/datasets/ml_dataset_2024_2025_full_for_apex.parquet`
  - Size: 9.3 GB
  - Rows: 1,750,195
  - Period: 2024-01-04 to 2025-11-14 (1.9 years)
  - Stocks: 3,988

**Training**:
- Epochs: 4 (early stopped, patience=3)
- Best epoch: 1
- Training loss: 4.92 → 5.00 (worsening)
- Short-term score: 0.4917 (epoch 1)

**Validation Results** (Epoch 1, 65 panels):

| Horizon | RankIC   | P@K    | P@K (rand) | ΔP@K   | Spread  | WIL    |
|---------|----------|--------|------------|--------|---------|--------|
| 1d      | -0.0081  | 0.4474 | 0.4818     | -3.44% | 0.0006  | 0.3554 |
| 5d      | -0.0085  | 0.4746 | 0.5001     | -2.55% | 0.0047  | 0.3813 |
| 10d     | -0.0103  | 0.4610 | 0.5066     | -4.57% | 0.0070  | 0.3715 |
| 20d     | -0.0231  | 0.4348 | 0.4719     | -3.71% | 0.0070  | 0.3227 |

**Issues**:
- ❌ All RankIC values negative (predictions inversely correlated)
- ❌ P@K below random baseline on all horizons
- ❌ No learning progression (epoch 1 best, then worsened)
- ⚠️ Dataset only covers 1.9 years (short period)

---

### Model 2: apex_ranker_v0_g5_shortterm

**Configuration**:
- Config: `apex-ranker/configs/v0_short_term.yaml`
- Features: 61 (g5_short_term - low NULL% features)
- Lookback: 60 days (reduced from 90)
- Horizons: [1d, 5d] (short-term focused)
- Dataset: `gogooku5/data/output/datasets/ml_dataset_2023_2024_2025_full.parquet`
  - Size: 15 GB
  - Period: 2023-03-31 to 2025-11-07 (2.6 years)
  - Training days: 437, Validation days: 125

**Training**:
- Epochs: 6 (early stopped, patience=5)
- Best epoch: 1
- Training loss: 2.29 → 2.26 (minimal improvement)
- Short-term score: 0.2755 (epoch 1)

**Validation Results** (Epoch 1, 125 panels):

| Horizon | RankIC   | P@K    | P@K (rand) | ΔP@K   | Spread  | WIL    |
|---------|----------|--------|------------|--------|---------|--------|
| 1d      | -0.0564  | 0.4248 | 0.4788     | -5.40% | -0.0012 | 0.3743 |
| 5d      | -0.0821  | 0.4186 | 0.5168     | -9.75% | -0.0067 | 0.3716 |

**Issues**:
- ❌ Worse RankIC than Model 1 (-0.0564 for 1d, -0.0821 for 5d)
- ❌ P@K significantly below random baseline (9.75% worse on 5d)
- ❌ Negative spread on 5d (long positions underperform short)
- ❌ No learning progression despite longer training (6 epochs vs 4)

---

## Root Cause Analysis

### 1. Data Quality Issues

**High NULL rates in key features**:
```
beta60_topix: 67.6% NULL   ← CRITICAL
ret_prev_20d: 35.9% NULL
ret_prev_10d: 18.6% NULL
ret_prev_5d:  10.0% NULL
```

**Impact**:
- beta60_topix is a core feature in g5_core group (TOPIX market sensitivity)
- 67.6% NULL means 2/3 of data points lack this feature
- Model cannot learn meaningful patterns with such sparse data

### 2. Feature-Target Mismatch

**Hypothesis**: gogooku5 features may be designed for different target definitions:
- Features: gogooku5 naming convention (`ret_prev_*`, `AdjustmentClose`, etc.)
- Targets: Standard forward returns (`target_1d`, `target_5d`, etc.)
- Normalization: Cross-sectional Z-score applied to both

**Potential issues**:
- Feature construction methodology may differ from target calculation
- Time alignment: T+0 vs T+1 as-of logic inconsistency
- Sector/market adjustments may not match

### 3. Dataset Period Limitations

**Model 1 (2024-2025)**:
- Only 1.9 years of data
- May miss important market regimes (2020-2023)
- Insufficient training samples for 3,988 stocks

**Model 2 (2023-2025)**:
- Better coverage (2.6 years)
- Still shorter than original APEX-Ranker v0 (2020-2025, 5 years)

### 4. Normalization Issues

Current: **Cross-sectional Z-score** (rank-preserving but scale-agnostic)

**Hypothesis**: May not be appropriate for gogooku5 features:
- gogooku5 features already include Z-scores (`*_zscore_20d`)
- Double normalization could remove signal
- Cross-sectional ranking within day may conflict with time-series patterns

### 5. Model Architecture Compatibility

**Original APEX-Ranker assumptions**:
- Features: OHLCV-derived, technical indicators, sector features (89 features)
- Lookback: 180 days
- d_model: 192, depth: 3, patch_len: 16

**gogooku5 features**:
- Different feature space (graph features, TOPIX beta/alpha, sector breakdowns)
- May require different model hyperparameters
- Patch length (16) may not align with feature dynamics

---

## Comparison with Baseline Models

| Model | Features | P@K (20d) | RankIC (20d) | Dataset Period | Status |
|-------|----------|-----------|--------------|----------------|--------|
| **v0_enhanced** (baseline) | 89 | **0.5765** | -0.0322 | 2020-2025 (5y) | ✅ Production |
| **v0_pruned** (baseline) | 64 | **0.5405** | -0.0322 | 2020-2025 (5y) | ✅ Baseline |
| **v0_g5_latest** (new) | 65 | **0.4348** | -0.0231 | 2024-2025 (1.9y) | ❌ Failed |
| **v0_g5_shortterm** (new) | 61 | N/A | N/A | 2023-2025 (2.6y) | ❌ Failed |

**Performance Gap**:
- v0_g5_latest P@K: **19.6% worse** than v0_pruned (0.4348 vs 0.5405)
- v0_g5_latest P@K: **24.6% worse** than v0_enhanced (0.4348 vs 0.5765)

---

## Diagnostic Recommendations

### Immediate Actions (Priority 1)

1. **Investigate beta60_topix NULL values**:
   ```bash
   python -c "
   import polars as pl
   df = pl.read_parquet('gogooku5/data/output/datasets/ml_dataset_2024_2025_full_for_apex.parquet')
   print('beta60_topix NULL by date:')
   print(df.group_by('Date').agg(pl.col('beta60_topix').null_count().alias('null_count')).sort('Date').tail(20))
   "
   ```

2. **Verify target calculation**:
   - Check if `target_*` columns align with gogooku5 feature logic
   - Verify no lookahead bias (features at T, target from T+1 onwards)
   - Compare with original APEX-Ranker target definitions

3. **Test without problematic features**:
   - Create `g5_core_minimal` group excluding beta60_topix and high-NULL features
   - Retrain with only <10% NULL features

4. **Data preprocessing validation**:
   ```bash
   # Check for inf/nan in targets
   python scripts/validate_g5_dataset.py \
     --parquet gogooku5/data/output/datasets/ml_dataset_2024_2025_full_for_apex.parquet \
     --check-targets \
     --check-inf \
     --check-distribution
   ```

### Medium-term Fixes (Priority 2)

5. **Rebuild dataset with longer period**:
   - Target: 2020-2025 (5 years) to match baseline models
   - Ensure consistent data quality across all years

6. **Feature engineering review**:
   - Map gogooku5 features to original APEX-Ranker features
   - Identify semantic equivalents (e.g., `ret_prev_1d` vs `returns_1d`)
   - Create compatibility layer if needed

7. **Normalization experiment**:
   - Test alternative normalizations:
     - Raw features (no Z-score)
     - Time-series Z-score (per stock, rolling window)
     - Rank normalization (percentile per day)

8. **Hyperparameter tuning**:
   - Reduce model size (d_model: 192→128, depth: 3→2)
   - Adjust patch length (16→8 for shorter patterns)
   - Increase learning rate (5e-4→1e-3) and warmup epochs (3→5)

### Long-term Solutions (Priority 3)

9. **Feature importance analysis**:
   ```bash
   python apex-ranker/scripts/feature_importance_v0.py \
     --model models/apex_ranker_v0_g5_latest.pt \
     --config apex-ranker/configs/v0_base.yaml \
     --output results/g5_feature_importance.json
   ```

10. **Cross-validation with baseline features**:
    - Train hybrid model: 50% gogooku5 features + 50% baseline features
    - Identify which feature groups are problematic

11. **Architecture search**:
    - Test simpler models (LightGBM, XGBoost) as sanity check
    - If simpler models also fail → data issue
    - If simpler models succeed → architecture mismatch

---

## Model Artifacts

### Saved Files

```bash
models/
├── apex_ranker_v0_g5_latest.pt              # 13 MB (multi-horizon)
├── apex_ranker_v0_g5_latest_val_perday.npz  # 20 KB (validation metrics)
├── apex_ranker_v0_g5_shortterm.pt           # 2.4 MB (short-term)
└── apex_ranker_v0_g5_shortterm_val_perday.npz  # Validation metrics

apex-ranker/_logs/
└── (training logs not saved due to tee error)
```

### Load and Inspect

```python
import torch
import numpy as np

# Load model
model = torch.load('models/apex_ranker_v0_g5_latest.pt')
print(f"Model parameters: {sum(p.numel() for p in model['model_state_dict'].values()):,}")

# Load validation metrics
metrics = np.load('models/apex_ranker_v0_g5_latest_val_perday.npz')
print(f"Validation panels: {len(metrics['h1_rank_ic'])}")
print(f"Mean RankIC (20d): {metrics['h20_rank_ic'].mean():.4f}")
```

---

## Next Steps

**Recommendation**: **Do not deploy these models**. Performance is significantly below baseline and below random chance.

**Action Plan**:

1. **Immediate** (This week):
   - [ ] Debug beta60_topix NULL issue
   - [ ] Validate target calculation consistency
   - [ ] Create g5_core_minimal feature group (low NULL%)
   - [ ] Retrain with minimal feature set

2. **Short-term** (Next 2 weeks):
   - [ ] Extend dataset to 2020-2025 (5 years)
   - [ ] Implement feature engineering validation script
   - [ ] Test alternative normalizations
   - [ ] Cross-validate with LightGBM (sanity check)

3. **Medium-term** (Next month):
   - [ ] Complete feature mapping (gogooku5 ↔ baseline)
   - [ ] Architecture hyperparameter search
   - [ ] Implement robust feature importance analysis
   - [ ] Document gogooku5→APEX-Ranker integration guidelines

---

## Conclusion

Training completed successfully but models are **not production-ready**. Root cause analysis points to:

1. **Data quality issues** (67.6% NULL in key features)
2. **Feature-target mismatch** (gogooku5 vs baseline methodology)
3. **Insufficient training data** (1.9 years vs 5 years baseline)

**Priority**: Fix data quality issues before further model development.

**Status**: 🔴 **Blocked on data investigation**

---

**Generated**: 2025-11-17 12:15 UTC
**Author**: Claude Code (Autonomous Mode)
**Models**: apex_ranker_v0_g5_latest, apex_ranker_v0_g5_shortterm
