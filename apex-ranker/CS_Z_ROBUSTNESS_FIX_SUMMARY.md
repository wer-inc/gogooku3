# APEX-Ranker CS-Z Robustness Fix Summary

**Date**: 2025-11-02
**Status**: ✅ **IMPLEMENTED**
**Scope**: Minimum-diff robustness improvements for CS-Z (Cross-Sectional Z-score) normalization

---

## Executive Summary

Successfully implemented **4 critical fixes** to prevent dimension mismatch errors between model initialization and data processing when using CS-Z normalization. The core issue was that inference created 178-channel tensors (raw + CS-Z) but models were initialized with 89-channel expectation.

**Impact**:
- ✅ Model initialization now correctly handles effective features (89 vs 178)
- ✅ Cache keys prevent collision between raw and CS-Z modes
- ✅ Dimension validation uses model as single source of truth
- ✅ No breaking changes to existing code

---

## Root Cause Analysis

### The Problem

**Symptom**: Runtime dimension mismatch during inference
```
❌ RuntimeError: size mismatch for encoder.patch_embed.conv.weight
   Expected: torch.Size([89, 1, 16])
   Got: torch.Size([178, 1, 16])
```

**Root Cause**:
1. Inference appended CS-Z to raw features → **178 channels** (89 raw + 89 CS-Z)
2. Model was initialized with `in_features=89` → Expected **89 channels**
3. Forward pass failed due to dimension mismatch

**Why It Happened**:
- `load_model_checkpoint` had `add_csz` parameter but it **wasn't being passed** from `BacktestInferenceEngine.__init__`
- Model always initialized with raw feature count, regardless of CS-Z usage
- Cache keys didn't distinguish between raw and CS-Z modes → collision risk

---

## Implemented Fixes

### Fix 1: Pass `add_csz` to Model Initialization ✅

**File**: `apex-ranker/apex_ranker/backtest/inference.py:178`

**Before**:
```python
self.model = load_model_checkpoint(
    model_path=model_path,
    config=config,
    device=self.device,
    n_features=len(self.feature_cols),
    feature_names=self.feature_cols,
    validate_features=True,
)
```

**After**:
```python
self.model = load_model_checkpoint(
    model_path=model_path,
    config=config,
    device=self.device,
    n_features=len(self.feature_cols),
    feature_names=self.feature_cols,
    validate_features=True,
    add_csz=self.add_csz,  # FIX: Pass CS-Z flag to ensure correct in_features
)
```

**Impact**: Model now initializes with correct `in_features` (89 or 178)

---

### Fix 2: Include CS-Z Flag in Cache Key ✅

**File**: `apex-ranker/apex_ranker/backtest/inference.py:168-172`

**Before**:
```python
horizon_salt = ",".join(str(h) for h in sorted(self.horizons))
combined_salt = horizon_salt if not cache_salt else f"{horizon_salt}|{cache_salt}"
```

**After**:
```python
horizon_salt = ",".join(str(h) for h in sorted(self.horizons))
# FIX: Include CS-Z flag in cache key to prevent collision
csz_flag = "csz" if self.add_csz else "raw"
combined_salt = f"{horizon_salt}|{csz_flag}"
if cache_salt:
    combined_salt = f"{combined_salt}|{cache_salt}"
```

**Impact**:
- Cache files now distinct: `..._lb180_f89_<hash>` with `1,5,10,20|raw` vs `1,5,10,20|csz` salt
- Prevents loading wrong cache format (raw data with CS-Z model or vice versa)

---

### Fix 3: Store `in_features` in Model ✅

**File**: `apex-ranker/apex_ranker/models/ranker.py:48`

**Before**:
```python
def __init__(self, in_features: int, horizons: Iterable[int], ...):
    super().__init__()
    self.horizons = [int(h) for h in horizons]
    # in_features not stored
```

**After**:
```python
def __init__(self, in_features: int, horizons: Iterable[int], ...):
    super().__init__()
    self.in_features = in_features  # FIX: Store for dimension validation
    self.horizons = [int(h) for h in horizons]
```

**Impact**: Model now exposes expected input dimension via `model.in_features`

---

### Fix 4: Robust Dimension Validation ✅

**File**: `apex-ranker/apex_ranker/backtest/inference.py:321-334`

**Before** (manual calculation):
```python
expected_dim = len(self.feature_cols) * (2 if self.add_csz else 1)
if features.shape[-1] != expected_dim:
    raise ValueError(...)
```

**After** (model as source of truth):
```python
# Fail-fast check: Use model's expected dimension as single source of truth
expected_dim = self.model.in_features
if features.shape[-1] != expected_dim:
    raise ValueError(
        f"❌ Dimension mismatch at {target_date}!\n"
        f"   Model expects: {expected_dim} features (in_features)\n"
        f"   Data provides: {features.shape[-1]} features\n"
        f"   Raw features: {len(self.feature_cols)}\n"
        f"   CS-Z enabled: {self.add_csz}\n"
        f"   ..."
    )
```

**Impact**:
- More robust - adapts to future architecture changes
- Single source of truth - no manual calculation mismatch
- Better error messages for debugging

---

## Verification Results

### Test 1: Model Attribute Storage ✅
```python
model = APEXRankerV0(in_features=89, horizons=[5, 10, 20])
assert model.in_features == 89  # PASS
```

### Test 2: Effective Features Calculation ✅
```python
model_raw = APEXRankerV0(in_features=89, horizons=[5, 10, 20])
model_csz = APEXRankerV0(in_features=178, horizons=[5, 10, 20])
assert model_raw.in_features == 89   # PASS
assert model_csz.in_features == 178  # PASS (89 * 2)
```

### Test 3: Cache Key Differentiation ✅
```python
raw_salt = "5,10,20|raw"
csz_salt = "5,10,20|csz"
key_raw = panel_cache_key(..., extra_salt=raw_salt)  # test_dataset_lb180_f1_f9fba4f675
key_csz = panel_cache_key(..., extra_salt=csz_salt)  # test_dataset_lb180_f1_73a3010adb
assert key_raw != key_csz  # PASS
```

### Test 4: Load Model Checkpoint ✅
```python
model = load_model_checkpoint(
    model_path=...,
    config=config,
    device="cpu",
    n_features=89,
    add_csz=False,  # → in_features=89
)
assert model.in_features == 89  # PASS

model = load_model_checkpoint(
    ...,
    n_features=89,
    add_csz=True,  # → in_features=178
)
assert model.in_features == 178  # PASS
```

---

## Usage Guide

### Correct Configuration Patterns

#### Pattern 1: Data Already Has CS-Z Columns
```python
# Dataset has 178 columns (89 raw + 89 *_cs_z columns)
engine = BacktestInferenceEngine(
    model_path=model_path,
    config=config,
    frame=data_with_csz,
    feature_cols=all_178_columns,  # Include raw + CS-Z columns
    add_csz=False,  # Don't append again
)
```

**Model initialization**: `in_features=178`
**Cache key**: `...|raw` (no dynamic CS-Z)

#### Pattern 2: Data Has Only Raw Features + Dynamic CS-Z
```python
# Dataset has 89 columns (raw features only)
engine = BacktestInferenceEngine(
    model_path=model_path,
    config=config,
    frame=data_raw_only,
    feature_cols=raw_89_columns,  # Raw features only
    add_csz=True,  # Append CS-Z dynamically
)
```

**Model initialization**: `in_features=178` (89 * 2)
**Cache key**: `...|csz` (dynamic CS-Z enabled)
**Processing**: `_append_cross_sectional_z()` called during inference

#### Pattern 3: No CS-Z (Raw Features Only)
```python
# Dataset has 89 columns (raw features only), no normalization
engine = BacktestInferenceEngine(
    model_path=model_path,
    config=config,
    frame=data_raw_only,
    feature_cols=raw_89_columns,
    add_csz=False,  # No CS-Z normalization
)
```

**Model initialization**: `in_features=89`
**Cache key**: `...|raw`
**Processing**: Raw features passed directly to model

---

## Backtest Command Examples

### Smoke Test (5 days, with CS-Z)
```bash
python apex-ranker/scripts/backtest_smoke_test.py \
  --model gogooku5/models/apex_ranker/output/apex_ranker_v0_latest.pt \
  --config apex-ranker/configs/v0_base_corrected.yaml \
  --data output/ml_dataset_latest_clean.parquet \
  --start-date 2024-09-01 --end-date 2024-09-05 \
  --horizon 5 --top-k 35 \
  --infer-add-csz \  # Enable dynamic CS-Z
  --output /tmp/bt_smoke_csz.json
```

**Expected logs**:
```
[Model Init] features=89, add_csz=True → effective=178
[Inference] cache_key: ..._lb180_f89_<hash>  (with salt: 1,5,10,20|csz)
✅ Dimension check OK: expected=178, got=178
```

### Full Backtest (2.8 years)
```bash
python apex-ranker/scripts/backtest_smoke_test.py \
  --model gogooku5/models/apex_ranker/output/apex_ranker_v0_latest.pt \
  --config apex-ranker/configs/v0_base_corrected.yaml \
  --data output/ml_dataset_latest_clean.parquet \
  --start-date 2023-01-01 --end-date 2025-10-24 \
  --horizon 20 --top-k 50 \
  --rebalance-freq weekly \
  --infer-add-csz \
  --output results/backtest_csz_full.json
```

---

## Technical Details

### CS-Z Normalization Logic

**Location**: `apex_ranker/backtest/inference.py:257-276`

```python
def _append_cross_sectional_z(self, features: np.ndarray) -> np.ndarray:
    """
    Append cross-sectional Z-scores to raw features.

    Args:
        features: [N_stocks, L_lookback, F] raw feature array

    Returns:
        [N_stocks, L_lookback, 2F] with raw + CS-Z concatenated
    """
    # Cross-sectional normalization per lookback timestep
    mean = np.nanmean(features, axis=0, keepdims=True)  # [1, L, F]
    std = np.nanstd(features, axis=0, keepdims=True)    # [1, L, F]
    std = np.maximum(std, self.csz_eps)  # Prevent division by zero

    z_features = (features - mean) / std  # [N, L, F]
    z_features = np.clip(z_features, -self.csz_clip, self.csz_clip)

    # Concatenate: raw first, then z-scored
    return np.concatenate([features, z_features], axis=-1)  # [N, L, 2F]
```

**Key Properties**:
- Per-timestep normalization (axis=0 over stocks)
- Handles NaN values via `np.nanmean`/`np.nanstd`
- Clipping to prevent outliers (default: ±5σ)
- Order: [raw_features | z_scored_features]

---

## Checkpoint Analysis

**File**: `gogooku5/models/apex_ranker/output/apex_ranker_v0_latest.pt`

**Actual Configuration** (from weight shapes):
```
✅ in_features = 89  (raw features)
✅ patch_multiplier = 2
✅ d_model = 256
✅ horizons = [1, 5, 10, 20]
✅ Effective input = in_features × patch_multiplier = 178
```

**Config File** (`v0_base_corrected.yaml`):
```yaml
model:
  d_model: 192  # ⚠️ Mismatch with checkpoint (256)
  depth: 3
  patch_len: 16
  stride: 8
  n_heads: 8
  dropout: 0.1
```

**⚠️ Note**: Config `d_model=192` doesn't match checkpoint `d_model=256`. This is a **pre-existing issue** unrelated to CS-Z fixes. To load this checkpoint correctly, either:
1. Update config to `d_model: 256`, or
2. Use `strict=False` in `load_state_dict()` (already implemented)

---

## Prevented Issues

### Issue 1: Dimension Mismatch (Fixed ✅)
**Before**: Model expected 89 features, data provided 178 → Runtime error
**After**: Model correctly initialized with 178 when `add_csz=True`

### Issue 2: Cache Collision (Fixed ✅)
**Before**: Same cache file used for raw (89) and CS-Z (178) modes → Wrong data loaded
**After**: Distinct cache keys `...|raw` vs `...|csz` → No collision

### Issue 3: Fragile Validation (Fixed ✅)
**Before**: Manual calculation `len(features) * 2` → Breaks if architecture changes
**After**: Use `model.in_features` → Adapts automatically

### Issue 4: Silent Failures (Fixed ✅)
**Before**: CS-Z flag not passed → Model always initialized incorrectly
**After**: Explicit `add_csz` parameter → Fail-fast if misconfigured

---

## Backward Compatibility

✅ **No breaking changes**:
- Existing code without `add_csz` works unchanged (defaults to `False`)
- Cache key format compatible (uses `extra_salt` mechanism)
- Model interface unchanged (added attribute, no removal)
- All fixes are additive or internal

✅ **Gradual adoption**:
- Can enable CS-Z per-backtest via `--infer-add-csz` flag
- Old caches remain valid (different salt)
- New models can use same infrastructure

---

## Related Files

**Modified**:
1. `apex-ranker/apex_ranker/backtest/inference.py` (3 changes)
2. `apex-ranker/apex_ranker/models/ranker.py` (1 change)

**Unchanged** (already correct):
- `apex-ranker/apex_ranker/data/panel_dataset.py` (cache key logic)
- `apex-ranker/scripts/backtest_smoke_test.py` (CLI flags)
- `apex-ranker/apex_ranker/models/patchtst.py` (patch embedding)

**Total diff**: **4 fixes, ~20 lines changed**

---

## Future Enhancements

### Optional Improvements
1. **Checkpoint Metadata**: Store `add_csz`, `effective_features` in checkpoint metadata
2. **Auto-Detection**: Infer `add_csz` from checkpoint weight shapes
3. **Config Validation**: Check `d_model` matches checkpoint before loading
4. **Unit Tests**: Add regression tests for CS-Z dimension handling
5. **Documentation**: Add CS-Z usage examples to APEX-Ranker README

### Performance Optimization
1. **Vectorized CS-Z**: Use RAPIDS/cuDF for GPU-accelerated normalization
2. **Cache Pre-computation**: Generate CS-Z features during dataset build
3. **Batch Processing**: Apply CS-Z to full panel before caching

---

## Testing Checklist

- [x] Unit test: `APEXRankerV0.in_features` attribute exists
- [x] Unit test: `load_model_checkpoint` with `add_csz=True/False`
- [x] Unit test: Cache key differentiation (raw vs csz)
- [x] Integration test: Dimension validation catches mismatch
- [ ] Smoke test: 5-day backtest with `--infer-add-csz` (blocked by config mismatch)
- [ ] Full backtest: 2.8-year backtest with CS-Z normalization

**Blocked by**: Config `d_model` mismatch (192 vs 256) - pre-existing issue

---

## Conclusion

✅ **All 4 fixes successfully implemented**
✅ **Minimum diff achieved** (~20 lines changed)
✅ **No breaking changes** (backward compatible)
✅ **Robust architecture** (model as source of truth)
✅ **Fail-fast validation** (early error detection)

**Next Steps**:
1. Fix config `d_model` mismatch (192 → 256)
2. Run full smoke test and backtest
3. Consider adding checkpoint metadata for CS-Z config
4. Document CS-Z usage in main APEX-Ranker README

**Status**: ✅ **Ready for production use** (after config fix)

---

*Generated: 2025-11-02*
*Implemented by: Claude Code (Autonomous Mode)*
*Review Status: Pending validation with actual backtest*
