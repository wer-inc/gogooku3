# URGENT: base_dropout Undefined in MultiHorizonPredictionHeads

**Issue**: Smoke test failing with `NameError: name 'base_dropout' is not defined`

**Root Cause**: In your recent edits to `src/atft_gat_fan/models/architectures/atft_gat_fan.py`, `base_dropout` is defined in `QuantilePredictionHead.__init__` (line 1536) but not in `MultiHorizonPredictionHeads.__init__` (starting line 1602).

---

## ❌ Problem Locations

**File**: `src/atft_gat_fan/models/architectures/atft_gat_fan.py`

**Lines using `base_dropout` without definition**:
- Line 1645: `nn.Dropout(base_dropout)` in shared_encoder
- Line 1668: `base_dropout * 0.5` for short-term horizons
- Line 1679: `base_dropout * 1.5` for long-term horizons

**Current `MultiHorizonPredictionHeads.__init__`** (line 1602):
```python
def __init__(
    self,
    hidden_size: int,
    config: DictConfig,
    training_cfg: DictConfig | None = None,
    output_std: float = 0.05,
    layer_scale_gamma: float = 1.0,
):
    super().__init__()
    self.config = config

    # ... horizon extraction code ...

    # Line 1645: base_dropout used HERE but never defined!
    shared_layers: list[nn.Module] = [
        nn.Linear(hidden_size, hidden_size // 2),
        nn.ReLU(),
        nn.Dropout(base_dropout),  # ❌ NameError
    ]
```

---

## ✅ Fix Required

Add the same `base_dropout` initialization logic that you added to `QuantilePredictionHead.__init__` (line 1536).

**Insert after line 1611** (`self.config = config`):

```python
# Extract dropout configuration
arch_cfg = getattr(config, "architecture", None)
base_dropout = float(getattr(arch_cfg, "dropout", 0.0)) if arch_cfg else 0.0

# Optional: Add environment variable override (match QuantilePredictionHead)
env_dropout = os.getenv("PRED_HEAD_DROPOUT")
if env_dropout:
    try:
        base_dropout = float(env_dropout)
    except Exception:
        pass
```

---

## 📍 Exact Edit Location

**File**: `src/atft_gat_fan/models/architectures/atft_gat_fan.py`

**After line 1611**:
```python
class MultiHorizonPredictionHeads(nn.Module):
    """Multi-horizon prediction heads - 各予測期間専用の出力層"""

    def __init__(
        self,
        hidden_size: int,
        config: DictConfig,
        training_cfg: DictConfig | None = None,
        output_std: float = 0.05,
        layer_scale_gamma: float = 1.0,
    ):
        super().__init__()
        self.config = config

        # ⬇️ INSERT base_dropout DEFINITION HERE ⬇️
        arch_cfg = getattr(config, "architecture", None)
        base_dropout = float(getattr(arch_cfg, "dropout", 0.0)) if arch_cfg else 0.0

        # 予測対象期間の設定 (新しいconfig構造をサポート)
        self._training_cfg = training_cfg or getattr(config, "training", None)
        # ... rest of __init__ ...
```

---

## 🔧 Complete Fix Snippet

```python
# Add after line 1611 (self.config = config)

# Extract dropout configuration (matches QuantilePredictionHead)
arch_cfg = getattr(config, "architecture", None)
base_dropout = float(getattr(arch_cfg, "dropout", 0.0)) if arch_cfg else 0.0

# Environment variable override for PRED_HEAD_DROPOUT
env_dropout = os.getenv("PRED_HEAD_DROPOUT")
if env_dropout:
    try:
        base_dropout = float(env_dropout)
    except Exception:
        pass
```

---

## ✅ Verification

After applying the fix:

```bash
# Test compilation
python -m compileall src/atft_gat_fan/models/architectures/atft_gat_fan.py

# Rerun smoke test
ENABLE_GRAD_MONITOR=1 GRAD_MONITOR_EVERY=200 SHARPE_WEIGHT=0.5 RANKIC_WEIGHT=0.2 \
CS_IC_WEIGHT=0.1 HUBER_WEIGHT=0.1 \
python scripts/integrated_ml_training_pipeline.py \
  --data-path output/datasets/ml_dataset_latest_full.parquet \
  --max-epochs 10 \
  train.batch.train_batch_size=2048 \
  train.batch.num_workers=8 \
  train.trainer.precision=bf16-mixed
```

---

## 📊 Impact

**Severity**: 🔴 **CRITICAL** - Blocks all training with new prediction head code

**Affected**:
- All training runs using the updated prediction head
- Smoke test validation of per-day loss fixes

**Not Affected**:
- Previous experiments (Exp1/Exp2) that completed before this edit
- APEX-Ranker (separate codebase)

---

## 🎯 Why This Happened

You added `base_dropout` configuration to `QuantilePredictionHead` but forgot to add the same initialization to `MultiHorizonPredictionHeads`. The two classes are independent and each needs its own variable definitions.

**Lesson**: When adding initialization logic to one class, check if sibling classes in the same file need the same logic.

---

**Status**: ⚠️ **Awaiting fix**
**Smoke test**: Currently failing at initialization
**Next action**: Apply fix above, then rerun smoke test

---

*Created*: 2025-10-31 04:17 UTC
*Priority*: P0 (blocks all progress)
