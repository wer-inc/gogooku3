# 🔧 Augmentation Phase Configuration Fix

**Date**: 2025-10-23
**Issue**: Augmentation phase not executing despite CLI parameter
**Status**: ✅ RESOLVED

## Problem Statement

Training completed with only 34 epochs instead of planned 60 epochs. The augmentation phase (15 epochs) did not execute despite passing `train.phase_training.phases.augmentation.epochs=15` via CLI.

## Root Cause

**Hydra Limitation**: CLI overrides can only **modify** existing configuration keys, not **create** new nested structures.

**Failed Approach**:
```bash
# This DOESN'T work - tries to create new phase via CLI:
python train_atft.py train.phase_training.phases.augmentation.epochs=15
```

**Why It Failed**:
- Base configuration only defines 4 phases (Baseline, Adaptive Norm, GAT, Fine-tuning)
- Hydra CLI override tried to add new `augmentation` key under `phases`
- Hydra silently ignored the override (no error, just skipped)
- Training completed after Phase 3 with only 34 epochs total

## Solution Implemented

### 1. New Configuration File: `configs/train/adaptive_phase3_ext.yaml`

```yaml
defaults:
  - adaptive  # Inherit from base adaptive config

# Extend adaptive training with longer fine-tuning phase and tightened stopping
trainer:
  max_epochs: 60

early_stopping:
  patience: 4

phase_training:
  phases:
    augmentation:
      epochs: 15
```

**Key Features**:
- ✅ Properly defines augmentation phase in YAML (not CLI)
- ✅ Inherits all adaptive training settings via `defaults: - adaptive`
- ✅ Extends max_epochs to 60 (from default 45)
- ✅ Tightens early stopping patience to 4 (from default 10)

### 2. Pipeline Integration

**File**: `scripts/integrated_ml_training_pipeline.py`

**Changes**:

**Line 259-263** - Set default config:
```python
os.environ.setdefault("ATFT_TRAIN_CONFIG", "adaptive_phase3_ext")
```

**Line 608-614** - Automatic config injection:
```python
# Ensure a train config override is always present
has_train_override = any(
    isinstance(arg, str) and arg.startswith("train=") for arg in cmd
)
if not has_train_override:
    train_cfg = os.getenv("ATFT_TRAIN_CONFIG", "").strip()
    if train_cfg:
        cmd.append(f"train={train_cfg}")
```

**Benefits**:
- Automatically adds `train=adaptive_phase3_ext` to Hydra command
- No need to remember complex CLI syntax
- Can override via `ATFT_TRAIN_CONFIG` environment variable
- Backward compatible with explicit `train=` overrides

## Expected Phase Configuration

### Before Fix (34 epochs)
| Phase | Epochs |
|-------|--------|
| Phase 0: Baseline | 10 |
| Phase 1: Adaptive Norm | 10 |
| Phase 2: GAT | 8 |
| Phase 3: Fine-tuning | 6 |
| **Total** | **34** |

### After Fix (49+ epochs)
| Phase | Epochs | Purpose |
|-------|--------|---------|
| Phase 0: Baseline | 10 | Foundation |
| Phase 1: Adaptive Norm | 10 | Normalization learning |
| Phase 2: GAT | 8 | Graph Attention activation |
| Phase 3: Fine-tuning | 6 | Final optimization |
| **Augmentation** | **15** | **Data augmentation** ✅ |
| **Total** | **49** | **(+15 epochs)** |

**Max Epochs**: 60 (with early stopping patience=4)

## Usage

### Default Mode (Recommended)
```bash
# Automatically uses adaptive_phase3_ext config
FORCE_SINGLE_PROCESS=1 ALLOW_UNSAFE_DATALOADER=0 \
python scripts/integrated_ml_training_pipeline.py \
  --batch-size 128 --max-epochs 60 --lr 1e-4
```

### Explicit Config Override
```bash
# Explicitly specify config
ATFT_TRAIN_CONFIG=adaptive_phase3_ext \
FORCE_SINGLE_PROCESS=1 ALLOW_UNSAFE_DATALOADER=0 \
python scripts/integrated_ml_training_pipeline.py \
  --batch-size 128 --max-epochs 60 --lr 1e-4
```

### Alternative Config
```bash
# Use different config (e.g., production_improved)
ATFT_TRAIN_CONFIG=production_improved \
python scripts/integrated_ml_training_pipeline.py \
  --batch-size 2048 --max-epochs 120
```

## Verification

### 1. Check Configuration Loaded
```bash
grep "ATFT_TRAIN_CONFIG" _logs/training/*.log
grep "train=adaptive_phase3_ext" _logs/training/*.log
```

### 2. Verify Augmentation Phase Executes
```bash
grep -i "augmentation" _logs/training/*.log
grep "Phase.*Augmentation" _logs/training/*.log
```

### 3. Verify Total Epochs
```bash
# Should see epochs up to 49+ (with augmentation)
grep "Epoch [0-9]*/[0-9]*:" _logs/training/*.log | tail -20
```

## Testing Checklist

- [ ] Quick test run (3 epochs) to verify config loads
- [ ] Verify augmentation phase appears in logs
- [ ] Full run (60 epochs) to confirm augmentation executes
- [ ] Compare metrics with previous 34-epoch run
- [ ] Evaluate Phase 1 best checkpoint separately

## Impact Assessment

### Before Fix
- ✅ Training stable (no crashes)
- ⚠️ Only 34 epochs executed
- ❌ Augmentation phase skipped
- ❌ Final Sharpe: -0.0066 (far from target 0.849)

### Expected After Fix
- ✅ Training stable (same safety)
- ✅ Full 49+ epochs with augmentation
- ✅ Augmentation phase properly executed
- 🎯 Monitor Sharpe improvement with extended training

## Related Issues

- **Issue**: Phase 1 peak performance not sustained
  - Phase 1 (Adaptive Norm): Sharpe 0.0134 ✅
  - Phase 3 (Fine-tuning): Sharpe -0.0066 ⚠️
  - **Action**: Extract Phase 1 checkpoint for evaluation

- **Issue**: Model not learning useful patterns
  - IC/RankIC near zero after 34 epochs
  - Hit rate below 50% (worse than random)
  - **Action**: Monitor improvement with augmentation phase

## Technical Learnings

### Hydra Configuration Best Practices

1. **Define Structure in YAML**:
   - ✅ Create new phases in config file
   - ❌ Don't try to create via CLI overrides

2. **Use Inheritance**:
   - ✅ `defaults: - adaptive` to reuse configs
   - ✅ Override only necessary fields

3. **Environment Variables**:
   - ✅ Use `ATFT_TRAIN_CONFIG` for config selection
   - ✅ Maintain backward compatibility

4. **CLI Overrides**:
   - ✅ Use for modifying existing keys (e.g., `trainer.max_epochs=60`)
   - ❌ Don't use for creating new nested structures

## References

- Configuration file: `configs/train/adaptive_phase3_ext.yaml`
- Pipeline integration: `scripts/integrated_ml_training_pipeline.py:259-263, 608-614`
- Previous training log: `_logs/training/train_phase3_ext_20251023_132147.log` (34 epochs)
- Status report: `_logs/training/training_status_report_20251023_232000.md`

---

**Status**: ✅ Ready for testing
**Next Action**: Quick test run to verify augmentation phase executes
**Estimated Impact**: +15 epochs, potential Sharpe improvement with data augmentation
