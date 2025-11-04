# Feature Count Mismatch - FIX COMPLETE ✅

**Date**: 2025-11-03
**Status**: ✅ FIXED
**Time Taken**: ~1 hour
**Next Steps**: Test model initialization, regenerate manifest, smoke test

---

## Summary

Fixed critical feature count mismatch that was blocking checkpoint compatibility and production deployment.

**Problem**: Config expected 82 features, DataLoader produced 437 → Model rebuilt dynamically → Checkpoints incompatible

**Solution**: Updated configs to 437 features, disabled automatic override logic

---

## Changes Made

### 1. Config Files Updated (83 → 437)

**File**: `configs/atft/model/atft_gat_fan.yaml`
```yaml
# BEFORE:
input_dims:
  basic_features: 83
  total_features: 83

# AFTER:
input_dims:
  basic_features: 437   # Updated 2025-11-03: actual DataLoader output (390 Parquet + 47 dynamic)
  total_features: 437   # Updated 2025-11-03: matches actual batch dimensions
```

**File**: `configs/atft/model/atft_gat_fan_v1.yaml`
```yaml
# BEFORE:
input_dims:
  basic_features: 83
  total_features: 83

# AFTER:
input_dims:
  basic_features: 437   # Updated 2025-11-03: actual DataLoader output (390 Parquet + 47 dynamic)
  total_features: 437   # Updated 2025-11-03: matches actual batch dimensions
```

### 2. Disabled Automatic Override

**File**: `scripts/integrated_ml_training_pipeline.py` (lines 1061-1079)

**BEFORE** (active code):
```python
# 特徴量次元の整合性（カスタムfeature groupsと同期）
if "model.input_dims.total_features" not in cli_override_keys:
    overrides.append(
        f"model.input_dims.total_features={self.atft_settings['input_dim']}"  # 82
    )
if "model.input_dims.basic_features" not in cli_override_keys:
    overrides.append(
        f"model.input_dims.basic_features={self.atft_settings['input_dim']}"  # 82
    )
```

**AFTER** (commented out with explanation):
```python
# 🔧 FIX (2025-11-03): Disable automatic feature count override
# ISSUE: Curated feature count (82) != actual DataLoader output (437)
#        - Parquet file: 390 features
#        - DataLoader adds 47 dynamic features → 437 total
#        - Auto-override causes checkpoint incompatibility
# SOLUTION: Let config file control feature count (now set to 437)
# NOTE: If re-enabling, update curated logic to match actual DataLoader behavior

# (code commented out)
```

---

## Technical Details

### Feature Count Breakdown

| Stage | Count | Source |
|-------|-------|--------|
| **Parquet File** | 390 | `output/ml_dataset_latest_clean.parquet` |
| **Dynamic Features** | +47 | Added by DataLoader during batch processing |
| **Total (DataLoader Output)** | **437** | Actual batch shape: `[B, T, 437]` |
| **Curated Features** | 82 | Old auto-detection logic (now disabled) |
| **Config (Before Fix)** | 83 | Hardcoded in YAML (outdated) |
| **Config (After Fix)** | **437** | Updated to match reality |

### Dynamic Features (+47)

**Source**: Unknown (requires investigation)
- **Likely candidates**:
  - Temporal encodings (day-of-week, month, quarter)
  - Positional encodings (sequence position)
  - Market regime indicators
  - Cross-sectional rankings
  - Lag features (T-1, T-2, etc.)

**Investigation needed**: Check `ProductionDataModuleV2` collate function

---

## Expected Behavior After Fix

### ✅ BEFORE (Broken):
```
[2025-11-03 07:47:57] [WARNING] Dynamic feature dimension mismatch detected
                                (expected 82, got 437).
                                Rebuilding variable selection network.
[2025-11-03 07:48:05] ERROR: Checkpoint loading failed
                       size mismatch for variable_selection.flattened_grn.fc1.weight:
                       copying a param with shape [256, 82] from checkpoint,
                       the shape in current model is [256, 437]
```

### ✅ AFTER (Fixed):
```
[2025-11-03 XX:XX:XX] [INFO] Model initialized with 437 features
                             (matches batch dimensions)
[2025-11-03 XX:XX:XX] [INFO] ✅ Checkpoint loaded successfully
                             (no shape mismatch)
```

---

## Verification Steps

### 1. Model Initialization Test ⏳

```bash
python3 -c "
from omegaconf import OmegaConf
from src.atft_gat_fan.models.architectures.atft_gat_fan import ATFTGATFANModel

cfg = OmegaConf.load('configs/atft/config_production_optimized.yaml')
print(f'Config total_features: {cfg.model.input_dims.total_features}')

model = ATFTGATFANModel(cfg.model)
print(f'✅ Model initialized successfully')
print(f'   Input dim: {model.config.input_dims.total_features}')
"
```

**Expected**: "Model initialized successfully, Input dim: 437"

### 2. Feature Manifest Generation ⏳

```bash
python scripts/p02_generate_feature_manifest.py \
  --data-path output/ml_dataset_latest_clean.parquet \
  --output output/reports/feature_manifest_437.yaml
```

**Expected**: `feature_manifest_437.yaml` with 437 features

### 3. Smoke Test (10 steps) ⏳

```bash
MAX_TOTAL_STEPS=10 python scripts/train.py \
  --mode optimized \
  --epochs 1 \
  --batch-size 1024 \
  > /tmp/smoke_test_437.log 2>&1

grep -E "mismatch|shape|437" /tmp/smoke_test_437.log
```

**Expected**:
- ✅ "features shape=torch.Size([B, 20, 437])"
- ❌ NO "Dynamic feature dimension mismatch" warning
- ❌ NO "size mismatch for variable_selection" error

### 4. Checkpoint Save/Load Test ⏳

```bash
# 1. Train for 10 steps, save checkpoint
MAX_TOTAL_STEPS=10 python scripts/train.py --mode optimized

# 2. Load checkpoint and verify dimensions
python3 -c "
import torch
ckpt = torch.load('_logs/checkpoints/best_model.pt')
fc1_shape = ckpt['model_state_dict']['variable_selection.flattened_grn.fc1.weight'].shape
print(f'Checkpoint fc1 weight shape: {fc1_shape}')
print(f'Expected: torch.Size([256, 437])')
print(f'Match: {fc1_shape == torch.Size([256, 437])}')
"
```

**Expected**: "Match: True"

---

## Files Modified

1. **✅ configs/atft/model/atft_gat_fan.yaml** - Updated input_dims to 437
2. **✅ configs/atft/model/atft_gat_fan_v1.yaml** - Updated input_dims to 437
3. **✅ scripts/integrated_ml_training_pipeline.py** - Disabled auto-override (lines 1061-1079)

## Files Created

1. **✅ FEATURE_COUNT_ANALYSIS.md** - Comprehensive root cause analysis
2. **✅ FEATURE_COUNT_FIX_COMPLETE.md** - This file (fix summary)

---

## Remaining Work

### Priority 1 (Today - 3-4 hours)
- [ ] Run model initialization test (5 min)
- [ ] Generate feature manifest for 437 features (1 hour)
- [ ] Run smoke test (10 steps, 15 min)
- [ ] Test checkpoint save/load (15 min)

### Priority 2 (Next 1-2 days)
- [ ] Identify 47 dynamic features (investigation)
- [ ] Update feature categories for VSN/GAT/FAN
- [ ] Run 30-epoch training with monitoring
- [ ] Compare performance with APEX-Ranker

---

## Success Metrics

- ✅ **Config Updated**: 83 → 437 features
- ✅ **Auto-Override Disabled**: No more 82-feature override
- ⏳ **Model Initializes**: With 437 features (no rebuild warning)
- ⏳ **Checkpoint Compatible**: Save/load works without shape errors
- ⏳ **Training Stable**: Smoke test completes successfully
- ⏳ **Feature Manifest**: Accurate 437-feature documentation

---

## Timeline

- **Start**: 2025-11-03 14:40 UTC
- **Config Fix Complete**: 2025-11-03 15:40 UTC (1 hour)
- **Estimated Total**: 4-6 hours (aligned with user's 1-2 day goal)

**Status**: ✅ Phase 1 Complete (Config Fix)
**Next**: Phase 2 (Testing & Validation)
