# Disk Quota Exceeded Issue - Root Cause & Solution

**Date**: 2025-11-14
**Status**: ✅ RESOLVED
**Severity**: Critical (Build failure)

## 📊 Problem Summary

Chunk build process repeatedly crashed at 2025Q1 Parquet save with corrupted files ("File out of specification: The file must end with PAR1").

## 🔍 Root Cause Analysis (User Diagnosis)

**Incorrect diagnosis** (Claude):
- ❌ Parquet file corruption from process crash
- ❌ Memory or OOM killer issues
- ❌ Code bugs in save logic

**Correct diagnosis** (User):
- ✅ **Disk quota exceeded** from writing to wrong directory
- ✅ `output/chunks` (47GB) instead of `output_g5/chunks` (512 bytes)
- ✅ Parquet save fails mid-write → incomplete file → missing PAR1 magic bytes

## 🎯 Key Evidence

1. **Output path mismatch**:
   ```
   Log: "Saving Parquet: output/chunks/2025Q1/ml_dataset.parquet"
   Expected: "Saving Parquet: output_g5/chunks/2025Q1/ml_dataset.parquet"
   ```

2. **Disk usage**:
   ```
   output/chunks:    47GB (old builds - quota exhausted)
   output_g5/chunks: 512 bytes (correct destination - almost empty)
   ```

3. **Crash pattern**: 3 independent processes all failed at identical point (Parquet save)

4. **Previous error**: "Disk quota exceeded" when updating `.gitignore`

## ✅ Solution

### 1. Stop all processes
```bash
ps aux | grep "python.*build_chunks" | grep -v grep | awk '{print $2}' | xargs -r kill -9
```

### 2. Delete old output/chunks (47GB)
```bash
rm -rf output/chunks
# Result: output/ 61GB → 15GB (47GB freed)
```

### 3. Set correct environment variable
```bash
export DATA_OUTPUT_DIR=/workspace/gogooku3/output_g5  # ← CRITICAL
export DIM_SECURITY_PATH=/workspace/gogooku3/output_g5/dim_security.parquet
export RAW_MANIFEST_PATH=/workspace/gogooku3/output_g5/raw_manifest.json
export CATEGORICAL_COLUMNS="Code,SectorCode"
export MAX_CONCURRENT_FETCH=200
export MAX_PARALLEL_WORKERS=128
export QUOTES_PARALLEL_WORKERS=200

python3 gogooku5/data/scripts/build_chunks.py \
  --start 2020-01-01 \
  --end 2025-11-07 \
  --resume
```

## 🧠 Technical Details

**ChunkPlanner default path**:
```python
# gogooku5/data/src/builder/chunks/planner.py:57
self.output_root = output_root or (self.settings.data_output_dir / "chunks")
```

**Settings default** (from config):
```python
# Without DATA_OUTPUT_DIR env var:
data_output_dir = Path("/workspace/gogooku3/output")  # ← OLD PATH

# With DATA_OUTPUT_DIR env var:
data_output_dir = Path(os.getenv("DATA_OUTPUT_DIR", "/workspace/gogooku3/output"))
```

## 📈 Results

- ✅ **Disk space freed**: 47GB
- ✅ **Correct output**: `output_g5/chunks/` being populated
- ✅ **Build progress**: 2020Q1 created successfully
- ✅ **No crashes**: Parquet saves completing normally

## 🎓 Lessons Learned

1. **Always verify output paths** in logs, not just environment variables
2. **Disk quota errors** can manifest as file corruption (incomplete writes)
3. **Multiple independent failures at identical point** → Infrastructure issue, not code bug
4. **Trust user diagnosis** when backed by concrete evidence

## 🔧 Prevention

1. **Add to environment setup docs**:
   ```bash
   export DATA_OUTPUT_DIR=/workspace/gogooku3/output_g5
   ```

2. **Add validation**:
   ```python
   # In build_chunks.py main():
   actual_output = planner.output_root
   logger.info(f"📁 Output directory: {actual_output}")
   if "output/" in str(actual_output) and "output_g5/" not in str(actual_output):
       logger.warning("⚠️  Using legacy output/ path - set DATA_OUTPUT_DIR=output_g5")
   ```

3. **Update .gitignore**:
   ```
   output_g5/
   _logs/
   ```

4. **Regular git gc**:
   ```bash
   git gc --aggressive --prune=now
   ```

## 📝 Related Issues

- SecId migration (Phase 3)
- Chunk rebuild after bug fixes
- Parallel process coordination

## 🙏 Credits

**Diagnosis**: User (complete and accurate)
**Root cause**: Output path misconfiguration
**Fix**: Environment variable correction
