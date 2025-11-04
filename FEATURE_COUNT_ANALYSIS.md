# Feature Count Mismatch Analysis

**Date**: 2025-11-03
**Status**: 🔴 CRITICAL - Blocks production deployment
**Priority**: P1 (User's top priority for 1-2 day timeline)

---

## Problem Summary

The ATFT-GAT-FAN model has a critical feature count mismatch that prevents checkpoint loading and blocks production deployment.

### Observed Behavior

```
[WARNING] Dynamic feature dimension mismatch detected (expected 82, got 437).
          Rebuilding variable selection network.

❌ Checkpoint loading failed:
   size mismatch for variable_selection.flattened_grn.fc1.weight:
   copying a param with shape torch.Size([256, 82]) from checkpoint,
   the shape in current model is torch.Size([256, 437])
```

---

## Root Cause Analysis

### Feature Count at Each Stage

| Stage | Count | Source |
|-------|-------|--------|
| **Parquet File** | 390 | `output/ml_dataset_latest_clean.parquet` |
| **DataLoader Output** | 437 | Batch shape: `[64, 20, 437]` |
| **Config Override** | 82 | Command-line: `model.input_dims.total_features=82` |
| **Feature Manifest** | 473 | `output/reports/feature_manifest_473.yaml` (legacy) |
| **Config Base** | 83 | `configs/atft/model/atft_gat_fan.yaml` |

### Dynamic Feature Addition

The DataLoader adds **47 features** dynamically during batch processing:
- **390 features** from Parquet → **437 features** in batches
- Delta: 437 - 390 = **47 dynamic features**
- Likely: Temporal encodings, positional features, or other runtime augmentations

### Config Override Issue

The training script uses a **hardcoded override**:
```bash
model.input_dims.total_features=82 \
model.input_dims.historical_features=0 \
model.input_dims.basic_features=82
```

This causes:
1. Model initializes with 82-dimensional input layers
2. First batch arrives with 437 features
3. Model detects mismatch → rebuilds variable selection network
4. Training proceeds with 437-dimensional model
5. Checkpoint saved with 437 dimensions
6. **Next run**: Attempts to load 82-dim checkpoint into 437-dim model → FAIL

---

## Impact Assessment

### Immediate Impact
- ❌ **Checkpoint incompatibility**: Cannot resume training
- ❌ **Inference blocked**: Cannot deploy model to production
- ⚠️ **Performance degradation**: Dynamic rebuilding adds initialization overhead
- ⚠️ **Wasted training**: Models trained with wrong dimensions are unusable

### Production Readiness Blocker
- **Current state**: Training works but checkpoints are incompatible
- **Required for production**: Checkpoint save/load must work reliably
- **User goal**: APEX-Ranker equivalent deployment (Sharpe 1.0+) within 1-2 days
- **Blocker**: Cannot deploy a model that can't load its own checkpoints

---

## Solution

### Phase 1: Config Fix (Immediate)

**Update config** to match actual feature count (437):

**File**: `configs/atft/model/atft_gat_fan.yaml`
```yaml
input_dims:
  total_features: 437  # Updated from 83 → 437 (actual DataLoader output)
  historical_features: 0
  basic_features: 437  # Updated from 83 → 437
```

**Remove command-line override** in training script:
```bash
# Before:
model.input_dims.total_features=82 \
model.input_dims.historical_features=0 \
model.input_dims.basic_features=82

# After:
# (Let config file control feature count)
```

### Phase 2: Regenerate Feature Manifest

**Create new manifest** with actual 437 features:
```bash
python scripts/p02_generate_feature_manifest.py \
  --data-path output/ml_dataset_latest_clean.parquet \
  --output output/reports/feature_manifest_437.yaml
```

### Phase 3: Validation

1. **Test model initialization**:
   ```bash
   python -c "from src.atft_gat_fan.models.architectures.atft_gat_fan import ATFTGATFANModel; \
              model = ATFTGATFANModel(...); \
              print(f'✅ Model initialized with {model.input_dim} features')"
   ```

2. **Test checkpoint save/load**:
   ```bash
   # Train for 10 steps, save checkpoint, reload checkpoint
   # Verify no shape mismatch errors
   ```

3. **Smoke test** (10 steps):
   ```bash
   MAX_TOTAL_STEPS=10 python scripts/train.py --mode optimized
   ```

---

## Dataset Feature Breakdown

### Top-Level Feature Groups (390 features in Parquet)

```
dmi (Daily Margin): 39 features
ss (Short Selling): 41 features
mkt (Market): 28 features
x (Experimental/Composite): 24 features
margin (Margin Trading): 23 features
sec17 (Sector 17): 20 features
stmt (Financial Statements): 19 features
flow (Order Flow): 17 features
graph (Graph/Correlation): 16 features
is (Income Statement): 13 features
sec (Sector): 13 features
... (57 more groups)
```

### Dynamic Features (+47 during batch processing)

**Source**: Unknown (requires investigation)
- Likely candidates:
  - Temporal encodings (day-of-week, month, quarter)
  - Positional encodings (sequence position)
  - Market regime indicators
  - Cross-sectional rankings
  - Lag features (T-1, T-2, etc.)

**Investigation needed**:
- Check `ProductionDataModuleV2` collate function
- Check `ParquetStockDataset` getitem augmentation
- Check any feature engineering in training loop

---

## Recommendations

### Immediate Actions (Today)
1. ✅ **Update config**: Set `total_features: 437`
2. ✅ **Remove override**: Delete hardcoded `model.input_dims.total_features=82`
3. ⏳ **Regenerate manifest**: Create `feature_manifest_437.yaml`
4. ⏳ **Test initialization**: Verify model creates correct layer dimensions
5. ⏳ **Test checkpoint**: Verify save/load works end-to-end

### Follow-Up (Next 1-2 days)
6. ⏳ **Identify 47 dynamic features**: Document what's being added and why
7. ⏳ **Update feature categories**: Ensure all 437 features are categorized for VSN/GAT/FAN
8. ⏳ **Run 30-epoch training**: Validate with full training run
9. ⏳ **Compare with APEX**: Benchmark against APEX-Ranker baseline

---

## Success Criteria

- ✅ Model initializes with 437 features (no dynamic rebuild)
- ✅ Checkpoint save/load works without shape mismatch errors
- ✅ Smoke test (10 steps) completes successfully
- ✅ Feature manifest accurately reflects all 437 features
- ✅ Config matches actual data dimensions
- ✅ No warnings about "Dynamic feature dimension mismatch"

---

## Related Files

**Configs**:
- `configs/atft/model/atft_gat_fan.yaml` - Model architecture config
- `configs/atft/model/atft_gat_fan_v1.yaml` - Alternative config (also 83 features)

**Manifests**:
- `output/reports/feature_manifest_473.yaml` - Legacy manifest (outdated)
- `output/reports/feature_manifest_306.yaml` - Alternative manifest
- `output/reports/feature_manifest_437.yaml` - To be created

**Training Scripts**:
- `scripts/train.py` - Training entry point (applies config override)
- `scripts/train_atft.py` - Core training logic
- `src/gogooku3/training/atft/data_module.py` - DataLoader (adds dynamic features)

**Logs**:
- `_logs/train_quick_short_20251103_0745.log` - Shows mismatch warning

---

## Timeline

**Estimated time to fix**: 4-6 hours
- Config update: 30 min
- Manifest regeneration: 1-2 hours
- Testing: 2-3 hours
- Validation: 1 hour

**Aligns with user's 1-2 day timeline** for production deployment.
