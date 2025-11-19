# Schema Validation Categorical Bug Fix

**Date**: 2025-11-14
**Component**: `gogooku5/data/src/builder/pipelines/dataset_builder.py`
**Issue**: Schema validation failed with hash mismatch (f3c70013d8b91c4b != 89f27cdf7eb9c285)

## Problem

### Symptoms
```
❌ Schema validation failed (hash: f3c70013d8b91c4b != 89f27cdf7eb9c285)
   Type mismatches (2):
      SecId: expected Categorical, got Int32
      Code: expected Categorical, got String
```

### Root Cause

**Python variable scoping issue in `save_with_cache()` + `validate_dataframe()` interaction**

**Flow**:
1. `dataset_builder.py:1078` - Calls `save_with_cache(df, ..., categorical_columns=["Code", "SecId"])`
2. `lazy_io.py:265` - **Local variable reassignment**: `df = df.with_columns([pl.col(col).cast(pl.Categorical) ...])`
3. `lazy_io.py:281` - Writes **modified** df to parquet (with Categorical types) ✅
4. `dataset_builder.py:1106` - Validates **original** df (without Categorical types) ❌

**Issue**: In Python, parameter reassignment (`df = ...`) creates a new local variable. The caller's `df` remains unchanged.

**Result**:
- Saved parquet file: `Code=Categorical, SecId=Categorical, hash=89f27cdf7eb9c285` ✅
- Validation sees: `Code=String, SecId=Int32, hash=f3c70013d8b91c4b` ❌
- Schema validation fails with hash mismatch

### Verification

**Proof that saved file is correct**:
```python
import polars as pl
df = pl.scan_parquet('/workspace/gogooku3/output/chunks/2020Q1/ml_dataset.parquet')
print(df.schema['Code'])    # Categorical ✅
print(df.schema['SecId'])   # Categorical ✅

# Manual hash calculation
actual_columns = {name: str(dtype) for name, dtype in df.collect_schema().items()}
ordered = [f"{name}:{dtype}" for name, dtype in sorted(actual_columns.items())]
hash_input = ";".join(ordered)
actual_hash = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:16]
print(actual_hash)  # 89f27cdf7eb9c285 ✅ (matches manifest)
```

## Solution

**Change**: Validate the **saved parquet file** instead of the in-memory DataFrame

**File**: `gogooku5/data/src/builder/pipelines/dataset_builder.py:1106`

**Before**:
```python
validation_result = self.schema_validator.validate_dataframe(df)
```

**After**:
```python
# FIX: Validate the saved parquet file (with categorical types applied)
# instead of the in-memory df (before categorical conversion)
# Issue: save_with_cache() modifies df locally, caller's df remains unchanged
validation_result = self.schema_validator.validate_parquet(parquet_path)
```

**Benefits**:
1. ✅ Validates the **actual saved data** (what gets read back later)
2. ✅ Reflects all transformations applied during save (categorical, compression, etc.)
3. ✅ Safer - catches any I/O-related schema changes
4. ✅ No changes needed to `save_with_cache()` signature

### Verification

```python
from pathlib import Path
from gogooku5.data.src.builder.utils.schema_validator import SchemaValidator

chunk_path = Path("/workspace/gogooku3/output/chunks/2020Q1/ml_dataset.parquet")
validator = SchemaValidator()

result = validator.validate_parquet(chunk_path)
print(result)  # ✅ Schema valid (hash: 89f27cdf7eb9c285)
```

## Impact

**Fixed chunks**: All chunks built after 2025-11-14 11:45 JST
**Broken chunks**: 2020Q1 from previous build (already fixed when re-read)
**Action needed**: None - fix is automatic on next rebuild

## Related Files

- `gogooku5/data/src/builder/pipelines/dataset_builder.py:1106-1109` - Schema validation call
- `gogooku5/data/src/builder/utils/lazy_io.py:265-267` - Categorical conversion
- `gogooku5/data/src/builder/utils/schema_validator.py:160-164` - validate_parquet() method
- `gogooku5/data/schema/feature_schema_manifest.json` - Expected schema (v1.3.0, hash=89f27cdf7eb9c285)

## Lessons Learned

1. **Validate saved data, not intermediate data** - Always validate what's actually written to disk
2. **Python scoping gotcha** - Parameter reassignment doesn't affect caller's variable
3. **Test with actual I/O** - In-memory tests miss I/O-related transformations
4. **Use parquet file validation** - `validate_parquet()` > `validate_dataframe()` for saved data
