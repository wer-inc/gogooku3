# gogooku5 Chunk Build Fixes (2025-11-14)

## Summary

Fixed two critical issues preventing reliable quarterly chunk-based dataset building in gogooku5:

1. **Polars Date Append Error** - Arrow IPC schema consistency issue
2. **start_time Parameter Missing** - Build duration tracking failure

**Commit**: `b680ff8` (fix(gogooku5): resolve Polars date append error and start_time parameter issue)

---

## Issue 1: Polars Date Append Error in Arrow IPC Writes

### Error Message
```
ERROR: could not append value: 2020-03-31 of type: date to the builder
it might also be that a value overflows the data-type's capacity
```

### Root Cause

Arrow IPC (Inter-Process Communication) format requires strict schema consistency across all rows. The error occurred when:

1. Small table data (e.g., `listed` snapshots) was loaded from dict/JSON sources
2. Polars inferred mixed types for date columns (some as `pl.Date`, some as `pl.Utf8`)
3. Arrow IPC write failed due to inconsistent schema

This happened specifically in the `_load_small_table_cached()` method when writing to `.arrow` cache files.

### Technical Details

**Location**: `gogooku5/data/src/builder/pipelines/dataset_builder.py`

**Affected Code**:
```python
# Line 1163 (before fix)
df.write_ipc(cache_path, compression="zstd")
```

**Problem**: No schema normalization before IPC write, leading to type inconsistencies.

### Solution

**1. Added `_normalize_schema_for_ipc()` method** (Lines 1178-1245):

```python
def _normalize_schema_for_ipc(self, df: pl.DataFrame) -> pl.DataFrame:
    """Normalize DataFrame schema for Arrow IPC write compatibility.

    Arrow IPC requires strict schema consistency. This method ensures:
    - Date columns are properly typed (not mixed String/Date)
    - All columns have consistent types across all rows

    Strategy:
    - For Utf8 columns: Detect date-like strings (YYYY-MM-DD pattern)
      and explicitly cast to pl.Date
    - For Object/Struct types: Convert to Utf8 for IPC compatibility
    """
```

**Key Features**:
- **Date Detection**: Samples non-null values and checks for YYYY-MM-DD pattern
- **Safe Casting**: Tests cast on sample before applying to full column
- **Graceful Fallback**: Keeps as string if cast fails
- **Object Handling**: Converts complex types (Object, Struct) to Utf8

**2. Updated `_load_small_table_cached()`** (Lines 1163-1167):

```python
# FIX: Normalize schema before writing to IPC
# Arrow IPC requires consistent types - convert any mixed-type columns explicitly
# This prevents "could not append value: YYYY-MM-DD of type: date" errors
# when Polars infers inconsistent types from dict data
df = self._normalize_schema_for_ipc(df)
df.write_ipc(cache_path, compression="zstd")
```

### Verification

**Test**: 2020Q1 chunk build (2020-01-01 to 2020-03-31)

```bash
python gogooku5/data/tools/merge_chunks.py build-chunk \
  --start-date 2020-01-01 --end-date 2020-03-31 \
  --lookback-days 180 --warmup-days 180 \
  --output-dir output_g5/chunks
```

**Results**:
- ✅ **Status**: Completed successfully
- ✅ **Rows**: 211,593
- ✅ **Columns**: 2,784
- ✅ **Build Duration**: 321.22 seconds
- ✅ **Cache**: `[SMALL TABLE] Cache hit for listed` (no more date append errors)

---

## Issue 2: start_time Parameter Missing in _persist_chunk_dataset

### Error Message
```
ERROR: name 'start_time' is not defined
```

### Root Cause

The `_persist_chunk_dataset()` method used `start_time` variable to calculate build duration:

```python
# Lines 1001, 1019 (before fix)
build_duration_seconds = time.time() - start_time
```

However, the function signature didn't include `start_time` as a parameter, and the call site didn't pass it.

### Technical Details

**Location**: `gogooku5/data/src/builder/pipelines/dataset_builder.py`

**Affected Code**:
```python
# Line 944 (before fix)
def _persist_chunk_dataset(self, df: pl.DataFrame, chunk_spec: ChunkSpec) -> Path:
    # ...
    build_duration_seconds = time.time() - start_time  # ❌ start_time not defined
```

```python
# Line 743 (before fix)
return self._persist_chunk_dataset(finalized, chunk_spec)  # ❌ Not passing start_time
```

### Solution

**1. Updated function signature** (Line 944):

```python
def _persist_chunk_dataset(
    self,
    df: pl.DataFrame,
    chunk_spec: ChunkSpec,
    *,
    start_time: float  # ✅ Added as keyword-only parameter
) -> Path:
```

**2. Updated call site** (Line 743):

```python
return self._persist_chunk_dataset(finalized, chunk_spec, start_time=start_time)
```

### Verification

**Test**: Check `status.json` in completed chunk

```json
{
  "build_duration_seconds": 321.22,
  "chunk_id": "2020Q1",
  "feature_schema_hash": "8dc473f22cbf4f0b",
  "feature_schema_version": "1.1.0",
  "output_end": "2020-03-31",
  "output_start": "2020-01-01",
  "rows": 211593,
  "state": "completed",
  "timestamp": "2025-11-14T04:45:58"
}
```

✅ **Result**: Build duration properly tracked and saved

---

## Impact

### Before Fixes
- ❌ Chunk builds failed with Polars date append error
- ❌ Build duration not tracked (NameError on start_time)
- ❌ Cannot proceed with gogooku5 normalization and deployment

### After Fixes
- ✅ Chunk builds complete successfully (2020Q1 verified)
- ✅ Build duration properly tracked in status.json
- ✅ Schema normalization ensures IPC compatibility
- ✅ Can proceed with full chunk rebuild (2020Q1→2025Q4)

---

## Related Work

**gogooku5 Normalization Plan**:

1. ✅ **Step 1**: Output root stabilization (`OUTPUT_DIR=/workspace/gogooku3/output_g5`)
2. ✅ **Step 2**: Fix chunk build prerequisites
   - ✅ `raw_manifest.json` regenerated (38KB, 7 sources)
   - ✅ Polars date bug fixed (this document)
   - ✅ start_time parameter fixed (this document)
3. 🔄 **Step 3**: Full chunk rebuild → merge → GCS deployment (in progress)

**Current Status** (as of 2025-11-14):
- Chunk build running in background (PID: 84347)
- Processing 2020Q1 → 2025Q4 (24 quarterly chunks)
- Using `--resume` mode to skip completed chunks
- Expected completion: 2-3 hours

---

## Files Changed

**Modified**:
- `gogooku5/data/src/builder/pipelines/dataset_builder.py` (+78 lines, -2 lines)
  - Added `_normalize_schema_for_ipc()` method (67 lines)
  - Updated `_load_small_table_cached()` to call normalization (5 lines)
  - Added `start_time` parameter to `_persist_chunk_dataset()` (2 lines)
  - Updated call site in `build_chunk()` (1 line)

**Created**:
- `docs/fixes/gogooku5_chunk_build_fixes_20251114.md` (this document)

---

## Testing Checklist

- [x] 2020Q1 chunk builds successfully (211,593 rows, 2,784 cols)
- [x] No Polars date append errors in logs
- [x] Build duration properly saved to status.json
- [x] Small table cache hit messages appear in logs
- [x] Schema normalization logs show date column detection
- [ ] Full chunk rebuild (2020Q1→2025Q4) completes successfully
- [ ] Merged dataset validated (row count, date range, schema)
- [ ] GCS upload successful

---

## Lessons Learned

### 1. Arrow IPC Strict Typing
**Problem**: Arrow IPC format is stricter than Parquet or CSV.

**Solution**: Always normalize schemas before IPC writes, especially when:
- Loading from dict/JSON sources
- Mixing data from multiple sources
- Working with date/datetime columns

**Best Practice**:
```python
# Always normalize before IPC write
df = self._normalize_schema_for_ipc(df)
df.write_ipc(path, compression="zstd")
```

### 2. Function Parameter Discipline
**Problem**: Using variables without passing them as parameters.

**Root Cause**: Code evolved over time, `start_time` tracking added later.

**Solution**:
- Use keyword-only parameters (`*, param: type`) for clarity
- Pass all required context through function signatures
- Avoid relying on closure/scope for critical data

**Best Practice**:
```python
def _persist_chunk_dataset(
    self,
    df: pl.DataFrame,
    chunk_spec: ChunkSpec,
    *,
    start_time: float  # Keyword-only - explicit and clear
) -> Path:
    ...
```

### 3. Incremental Testing
**Approach**: Test single chunk (2020Q1) before full rebuild.

**Benefits**:
- Faster iteration (5 min vs 2-3 hours)
- Early error detection
- Confidence in full rebuild

**Recommendation**: Always test chunk build logic on single quarter first.

---

## Future Improvements

1. **Schema Validation**: Add explicit schema validation before all IPC writes
2. **Type Hints**: Strengthen type hints in `_persist_chunk_dataset()` and related methods
3. **Unit Tests**: Add tests for `_normalize_schema_for_ipc()` with various edge cases
4. **Monitoring**: Add build duration tracking to logs and monitoring dashboard

---

## References

- **Commit**: `b680ff8` (fix(gogooku5): resolve Polars date append error and start_time parameter issue)
- **Modified File**: `gogooku5/data/src/builder/pipelines/dataset_builder.py`
- **Test Log**: `_logs/chunk_rebuild_fixed.log`
- **Verification**: `output_g5/chunks/2020Q1/status.json`
