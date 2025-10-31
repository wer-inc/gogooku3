# Smoke Test V5 (APEX Isolated) - Status Report

**Launch Time**: 2025-10-31 06:09:32 UTC
**PID**: 986304
**Status**: ❌ **FAILED** - CUDA index out of bounds error
**Log**: `_logs/training/smoke_test_v5_apex_isolated_20251031_060926.log`

---

## 🎯 Test Objective

Isolate architecture to determine if GAT/WAN complexity is blocking variance:

**Configuration**:
1. ✅ **ApexStylePredictionHead**: Lightweight APEX-Ranker style head
2. ✅ **BYPASS_GAT_COMPLETELY=1**: Skip all GAT processing
3. ✅ **BYPASS_ADAPTIVE_NORM=1**: Skip WAN normalization
4. ✅ **Strong ListNet**: weight=0.8 (vs 0.3 in v4)
5. ✅ **Pure ranking objective**: Huber/RankIC/CS-IC all=0

**Success Criteria**:
- ✅ Training starts without crashes
- ❌ **SCALE(yhat/y) > 0.00** (test failed before reaching this)
- ❌ IC/Sharpe metrics (not reached)

---

## ❌ Critical Error Found

**Error Type**: CUDA device-side assert triggered (index out of bounds)

**Root Cause**: **Key mismatch between predictions and targets**

```python
# Prediction keys (from ApexStylePredictionHead)
predictions.keys() = ['horizon_1d', 'horizon_5d', 'horizon_10d', 'horizon_20d']

# Target keys (from dataset)
targets.keys() = ['horizon_1', 'horizon_5', 'horizon_10', 'horizon_20']
```

**What happened**:
1. ApexStylePredictionHead outputs predictions with 'd' suffix (line 1704: `f"horizon_{horizon}d"`)
2. Targets in dataset have NO 'd' suffix (`'horizon_1'`, `'horizon_5'`, etc.)
3. Loss function tries to index predictions dict with target keys
4. Key not found → fallback to list/tensor indexing with string key
5. PyTorch tries to convert string '1' to int index → CUDA index error

**Debug Output** (`_logs/training/smoke_test_v5_apex_isolated_20251031_060926.log:12991-12994`):
```
[DEBUG-KEYS] predictions keys: ['horizon_1d', 'horizon_5d', 'horizon_10d', 'horizon_20d']
[DEBUG-KEYS] Target keys: ['horizon_1', 'horizon_5', 'horizon_10', 'horizon_20']
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113:
  Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
```

---

## 🔍 Why This Didn't Happen in V3/V4

**MultiHorizonPredictionHeads also outputs with 'd' suffix**:
- See `atft_gat_fan.py:1945` - `horizon_key = f"horizon_{horizon}d"`
- See `atft_gat_fan.py:1977` - `predictions[horizon_key] = scaled_output`

**But v3/v4 didn't crash** → There must be **key mapping logic** in the loss function or model wrapper that:
1. Handles MultiHorizonPredictionHeads output correctly
2. Does NOT handle ApexStylePredictionHead output

**Hypothesis**: The model wrapper (`ATFTGATFAN.forward()`) may be:
- Remapping keys for MultiHorizonPredictionHeads output
- NOT remapping keys for ApexStylePredictionHead output (new variant)

---

## 🔧 Investigation Required

**Option 1: Fix ApexStylePredictionHead** (Quick Fix)
- Change line 1704: `f"horizon_{horizon}d"` → `f"horizon_{horizon}"`
- Removes 'd' suffix to match target keys
- **Risk**: May break other code expecting 'd' suffix

**Option 2: Fix Loss Function** (Proper Fix)
- Add key mapping logic in loss calculation
- Handle both `horizon_Xd` and `horizon_X` formats
- More robust, handles both prediction head variants

**Option 3: Fix Model Wrapper** (Best Fix)
- Find where MultiHorizonPredictionHeads output is remapped
- Apply same remapping to ApexStylePredictionHead output
- Consistent handling across all prediction head variants

---

## 📋 Files Referenced

**Model Implementation**:
- `src/atft_gat_fan/models/architectures/atft_gat_fan.py:1609-1716` - ApexStylePredictionHead
- `src/atft_gat_fan/models/architectures/atft_gat_fan.py:1719-1979` - MultiHorizonPredictionHeads

**Key Lines**:
- Line 1654: `self.heads[f"horizon_{horizon}d"] = head` (APEX head)
- Line 1704: `key = f"horizon_{horizon}d"` (APEX forward)
- Line 1870: `self.horizon_heads[f"horizon_{horizon}d"] = head` (Multi head)
- Line 1945: `horizon_key = f"horizon_{horizon}d"` (Multi forward)

**Training Script**:
- `scripts/train_atft.py` - Loss calculation (location TBD)

**Log**:
- `_logs/training/smoke_test_v5_apex_isolated_20251031_060926.log` (214KB)
- Lines 12991-12994: Key mismatch debug output
- Lines 13000+: CUDA index error traceback

---

## 🎯 Next Actions

### Immediate (Fix & Rerun v5)

1. **Investigate model wrapper** (`ATFTGATFAN.forward()`):
   - Find where predictions dict is constructed
   - Check if key remapping exists for MultiHorizonPredictionHeads
   - Apply same logic to ApexStylePredictionHead

2. **Quick Test** (if wrapper fix unclear):
   - Temporarily change ApexStylePredictionHead to output without 'd' suffix
   - Verify training starts successfully
   - Document whether this breaks anything else

3. **Rerun smoke test v5**:
   - Same configuration with key mismatch fixed
   - Monitor for SCALE > 0.00 (primary success criterion)

### If SCALE > 0.00 (Architecture Isolated ✅)

**Conclusion**: GAT/WAN complexity is blocking variance restoration

**Next Steps**:
1. Document ApexStylePredictionHead as working baseline
2. Incrementally reintroduce components:
   - First: WAN normalization only
   - Then: GAT with residual connections
   - Finally: Full architecture
3. Identify which component causes variance collapse

### If SCALE = 0.00 (Still Blocked ❌)

**Conclusion**: Issue is NOT architecture-specific

**Next Steps**:
1. Investigate data normalization/target scaling
2. Test explicit variance floor in loss function
3. Consider APEX-Ranker data preprocessing differences

---

## 📊 Test Comparison

| Test | Config | Result | Status |
|------|--------|--------|--------|
| **V3** | Per-day loss + relaxed head | SCALE=0.00, Sharpe=0.0818 | ❌ No improvement |
| **V4** | + ListNet (weight=0.3) | SCALE=0.00, Sharpe=0.0818 | ❌ No improvement |
| **V5** | + APEX head + bypass GAT/WAN | CUDA error (key mismatch) | ❌ Failed at init |

---

**Last Updated**: 2025-10-31 06:15 UTC
**Status**: Blocked on key mismatch bug fix
**Priority**: P0 - Blocking architecture isolation experiment

---

*Key mismatch between ApexStylePredictionHead output ('horizon_Xd') and dataset targets ('horizon_X'). MultiHorizonPredictionHeads works despite same output format → model wrapper or loss function has special handling that's missing for ApexStylePredictionHead.*
