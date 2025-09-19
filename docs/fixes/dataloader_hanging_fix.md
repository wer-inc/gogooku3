# DataLoader Hanging Issue Fix

**Date**: 2025-09-19
**Issue**: Training process hanging at DataLoader creation
**Root Cause**: Missing import statement causing undefined class reference

## Problem Description

The training script `scripts/train_atft.py` was hanging indefinitely when attempting to create DataLoaders. The process would stop responding after logging "Creating data loaders..." with no error message.

## Root Cause Analysis

1. **Missing Import**: The script used `ProductionDataModuleV2` class but never imported it
2. **Silent Failure**: Python's behavior when encountering an undefined class name in certain contexts caused the process to hang rather than raise an ImportError
3. **Unnecessary Code Block**: The script contained code attempting to rebuild validation datasets using an undefined `ProductionDatasetV2` class

## Solution Implemented

### 1. Added Explicit Import
```python
# scripts/train_atft.py line 41
from src.gogooku3.training.atft.data_module import ProductionDataModuleV2
```

### 2. Removed Unnecessary Validation Dataset Rebuild
- **Lines Removed**: 4217-4284 in `scripts/train_atft.py`
- **Replacement**: Direct use of `data_module.val_dataloader()` which already handles feature column alignment
- **Reason**: The data module internally manages feature consistency between train and validation datasets

### 3. Created Regression Test
- **File**: `tests/unit/test_dataloader_regression.py`
- **Purpose**: Ensures DataLoader creation completes within timeout (5 seconds)
- **Coverage**: Tests normal creation, iteration, and empty dataset handling

## Testing

### Regression Test
```bash
pytest tests/unit/test_dataloader_regression.py -v
```

### Manual Verification
```bash
# Test with minimal dataset
python scripts/train_atft.py \
    data.source.data_dir=output/atft_data_minimal_v2 \
    data.use_day_batch_sampler=false \
    train.batch.train_batch_size=2 \
    train.trainer.max_epochs=1
```

## Prevention Measures

1. **Explicit Imports**: Always import classes explicitly at module level for fail-fast behavior
2. **Timeout Tests**: Regression tests with timeouts to catch hanging issues
3. **Code Review**: Check for undefined references during code review

## Impact

- **Fixed**: Training scripts can now initialize DataLoaders successfully
- **Performance**: No performance impact, removed unnecessary code
- **Reliability**: Added regression test prevents reoccurrence

## Related Files Changed

1. `scripts/train_atft.py` - Added import, removed problematic code block
2. `src/gogooku3/training/atft/data_module.py` - No changes needed, already correct
3. `tests/unit/test_dataloader_regression.py` - New regression test

## Lessons Learned

1. **Import Dependencies Explicitly**: Missing imports should fail immediately with ImportError
2. **Test for Timeouts**: Include timeout tests for operations that interact with I/O or complex initialization
3. **Simplify When Possible**: The removed validation dataset rebuild was unnecessary complexity