# URGENT: Multiple Undefined Variables in MultiHorizonPredictionHeads

**Issue**: Smoke test v2 failing with `NameError: name 'use_shared_layernorm' is not defined`

**Root Cause**: In the "relaxed prediction head" structural fixes, the user added new configuration logic but forgot to define **THREE additional variables**:
1. ✅ `base_dropout` (line 1655) - **USER FIXED**
2. ❌ `use_shared_layernorm` (line 1657) - **NEEDS FIX**
3. ❌ `output_init_std` (line 1707) - **NEEDS FIX**
4. ❌ `layer_scale_val` (line 1718) - **NEEDS FIX**

---

## ❌ Problem Locations

**File**: `src/atft_gat_fan/models/architectures/atft_gat_fan.py`

**Line 1657** (uses `use_shared_layernorm` - undefined):
```python
if use_shared_layernorm:  # ❌ NameError
    shared_layers.append(nn.LayerNorm(hidden_size // 2))
```

**Line 1707** (uses `output_init_std` - undefined):
```python
std = output_init_std * (0.5 if horizon <= 5 else 1.0)  # ❌ NameError
```

**Line 1718** (uses `layer_scale_val` - undefined):
```python
scale = layer_scale_val * (0.8 if horizon <= 5 else 1.2)  # ❌ NameError
```

**Pattern**: Function signature has `output_std` and `layer_scale_gamma` (lines 1607-1608) but code uses different variable names (`output_init_std`, `layer_scale_val`). These need to be derived from the function parameters, just like in `QuantilePredictionHead` (lines 1537-1560).

---

## ✅ Fix Required

All three variables need to be initialized after `base_dropout` (line 1624). They should follow the same pattern as `QuantilePredictionHead` (lines 1537-1560):

**Insert after line 1624** (after base_dropout initialization):

```python
# Output initialization std (from function parameter and config/env)
output_init_std = float(output_std)
if arch_cfg is not None and hasattr(arch_cfg, "output_init_std"):
    try:
        output_init_std = float(getattr(arch_cfg, "output_init_std"))
    except Exception:
        pass
env_std = os.getenv("PRED_HEAD_INIT_STD")
if env_std:
    try:
        output_init_std = float(env_std)
    except Exception:
        pass

# LayerScale value (from function parameter and config/env)
layer_scale_val = float(layer_scale_gamma)
if arch_cfg is not None and hasattr(arch_cfg, "layer_scale_gamma"):
    try:
        layer_scale_val = float(getattr(arch_cfg, "layer_scale_gamma"))
    except Exception:
        pass
env_layer_scale = os.getenv("PRED_HEAD_LAYER_SCALE")
if env_layer_scale:
    try:
        layer_scale_val = float(env_layer_scale)
    except Exception:
        pass

# LayerNorm configuration (disabled by default per relaxed prediction head design)
use_shared_layernorm = False
if arch_cfg is not None and hasattr(arch_cfg, "use_shared_layernorm"):
    try:
        use_shared_layernorm = bool(getattr(arch_cfg, "use_shared_layernorm"))
    except Exception:
        pass
env_use_layernorm = os.getenv("PRED_HEAD_USE_LAYERNORM")
if env_use_layernorm:
    try:
        use_shared_layernorm = env_use_layernorm.lower() in ("1", "true", "yes")
    except Exception:
        pass
```

---

## 📍 Exact Edit Location

**File**: `src/atft_gat_fan/models/architectures/atft_gat_fan.py`

**Current code** (lines 1616-1625):
```python
# Dropout設定 (QuantilePredictionHead と同一ロジック)
arch_cfg = getattr(config, "architecture", None)
base_dropout = float(getattr(arch_cfg, "dropout", 0.0)) if arch_cfg else 0.0
env_dropout = os.getenv("PRED_HEAD_DROPOUT")
if env_dropout:
    try:
        base_dropout = float(env_dropout)
    except Exception:
        pass

def _extract_horizons(cfg: DictConfig | None) -> list[int] | None:
```

**After line 1624** (before `def _extract_horizons`), add:

```python
# Dropout設定 (QuantilePredictionHead と同一ロジック)
arch_cfg = getattr(config, "architecture", None)
base_dropout = float(getattr(arch_cfg, "dropout", 0.0)) if arch_cfg else 0.0
env_dropout = os.getenv("PRED_HEAD_DROPOUT")
if env_dropout:
    try:
        base_dropout = float(env_dropout)
    except Exception:
        pass

# ⬇️ INSERT use_shared_layernorm DEFINITION HERE ⬇️
# LayerNorm configuration (disabled by default per relaxed prediction head design)
use_shared_layernorm = False
if arch_cfg is not None and hasattr(arch_cfg, "use_shared_layernorm"):
    try:
        use_shared_layernorm = bool(getattr(arch_cfg, "use_shared_layernorm"))
    except Exception:
        pass

# Environment variable override
env_use_layernorm = os.getenv("PRED_HEAD_USE_LAYERNORM")
if env_use_layernorm:
    try:
        use_shared_layernorm = env_use_layernorm.lower() in ("1", "true", "yes")
    except Exception:
        pass

def _extract_horizons(cfg: DictConfig | None) -> list[int] | None:
```

---

## 🔧 Complete Fix Snippet (Copy-Paste Ready)

Add this **after line 1624** (after `base_dropout` initialization, before `def _extract_horizons`):

```python
# Output initialization std (from function parameter and config/env)
output_init_std = float(output_std)
if arch_cfg is not None and hasattr(arch_cfg, "output_init_std"):
    try:
        output_init_std = float(getattr(arch_cfg, "output_init_std"))
    except Exception:
        pass
env_std = os.getenv("PRED_HEAD_INIT_STD")
if env_std:
    try:
        output_init_std = float(env_std)
    except Exception:
        pass

# LayerScale value (from function parameter and config/env)
layer_scale_val = float(layer_scale_gamma)
if arch_cfg is not None and hasattr(arch_cfg, "layer_scale_gamma"):
    try:
        layer_scale_val = float(getattr(arch_cfg, "layer_scale_gamma"))
    except Exception:
        pass
env_layer_scale = os.getenv("PRED_HEAD_LAYER_SCALE")
if env_layer_scale:
    try:
        layer_scale_val = float(env_layer_scale)
    except Exception:
        pass

# LayerNorm configuration (disabled by default per relaxed prediction head design)
use_shared_layernorm = False
if arch_cfg is not None and hasattr(arch_cfg, "use_shared_layernorm"):
    try:
        use_shared_layernorm = bool(getattr(arch_cfg, "use_shared_layernorm"))
    except Exception:
        pass
env_use_layernorm = os.getenv("PRED_HEAD_USE_LAYERNORM")
if env_use_layernorm:
    try:
        use_shared_layernorm = env_use_layernorm.lower() in ("1", "true", "yes")
    except Exception:
        pass
```

---

## ✅ Verification

After applying the fix:

```bash
# Test compilation
python -m compileall src/atft_gat_fan/models/architectures/atft_gat_fan.py

# Rerun smoke test v3
ENABLE_GRAD_MONITOR=1 GRAD_MONITOR_EVERY=200 \
SHARPE_WEIGHT=0.5 RANKIC_WEIGHT=0.2 CS_IC_WEIGHT=0.1 HUBER_WEIGHT=0.1 \
nohup python scripts/integrated_ml_training_pipeline.py \
  --data-path output/datasets/ml_dataset_latest_full.parquet \
  --max-epochs 10 \
  train.batch.train_batch_size=2048 \
  train.batch.num_workers=8 \
  train.trainer.precision=bf16-mixed \
  > _logs/training/smoke_test_v3_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo "Smoke test v3 started with PID: $!"
echo $! > _logs/training/smoke_test_v3.pid
```

---

## 📊 Impact

**Severity**: 🔴 **CRITICAL** - Blocks all training with new prediction head code

**Affected**:
- Smoke test v2 (crashed immediately at model initialization)
- All training runs using MultiHorizonPredictionHeads

**Root Cause Pattern**:
Similar to the `base_dropout` issue - structural fixes added new configuration logic but forgot to initialize the control variables. This is the **second undefined variable** in the same `__init__` method.

---

## 🎯 Why This Happened

The user implemented "relaxed prediction head" changes which made LayerNorm optional:
- Added conditional logic `if use_shared_layernorm:` at line 1657
- Forgot to define the variable that controls this condition
- Same pattern as the previous `base_dropout` error

**Lesson**: When adding conditional logic, ALWAYS define the condition variable first.

---

## 🚨 Additional Variables to Check

Given two undefined variables in a row, let's check for other potentially missing variables in MultiHorizonPredictionHeads:

```bash
# Check for undefined variables in the initialization section
grep -n "output_std\|layer_scale_gamma\|output_init_std\|layer_scale_val" \
  src/atft_gat_fan/models/architectures/atft_gat_fan.py | grep -A5 -B5 1602
```

These variables are used in QuantilePredictionHead (lines 1530-1560) and may also be needed in MultiHorizonPredictionHeads if the relaxed head design uses them.

---

**Status**: ⚠️ **Awaiting fix**
**Smoke test v2**: Crashed at model initialization (line 1657)
**Next action**: Apply fix above, verify compilation, rerun smoke test v3

---

*Created*: 2025-10-31 04:30 UTC (estimated)
*Priority*: P0 (blocks all progress - second critical bug in same method)
*Related*: URGENT_FIX_base_dropout.md (similar undefined variable issue)
