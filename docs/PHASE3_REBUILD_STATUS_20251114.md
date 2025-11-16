# Phase 3 Chunk Rebuild Status - 2025-11-14

**Status**: 🔄 **IN PROGRESS** - Awaiting completion (2-3 hours)
**Started**: 2025-11-14 10:19 JST
**Expected Completion**: 2025-11-14 12:00-13:00 JST

---

## Quick Reference

### Process Details
- **PID**: `222348`
- **Command**: `./venv/bin/python gogooku5/data/scripts/build_chunks.py --start 2020-01-01 --end 2025-11-07 --resume`
- **Log File**: `_logs/chunk_rebuild_phase3_20251114_101930.log`
- **PID File**: `_logs/chunk_rebuild.pid`

### Target Configuration
- **Date Range**: 2020-01-01 → 2025-11-07 (5 years)
- **Expected Chunks**: ~24 quarterly chunks
- **Schema Version**: v1.2.0
- **Target Hash**: `f077a15d37e1157a`
- **Features**: Phase 3 (SecId + Categorical + Futures + Index Options)

---

## Monitoring Commands

### Check Build Progress
```bash
# Quick status check
python3 scripts/check_chunk_status.py

# Live log monitoring
tail -f _logs/chunk_rebuild_phase3_20251114_101930.log

# Process status
ps -p 222348 -o pid,stat,etime,%cpu,%mem

# Completed chunks count
ls -lh output_g5/chunks/ | grep "^d" | wc -l
```

### Periodic Monitoring (Every 30 minutes)
```bash
# Set up periodic check
watch -n 1800 'python3 scripts/check_chunk_status.py'
```

---

## Completion Workflow

When all chunks are completed (check with `check_chunk_status.py`), proceed with:

### Step 1: Validate All Chunks ✅

```bash
# Run comprehensive validation
python3 scripts/check_chunk_status.py

# Expected output:
# ✅ All 24 chunks completed successfully!
# ✅ 24/24 chunks match target schema (Phase 3: f077a15d37e1157a)
```

**Acceptance Criteria**:
- All chunks in `state: "completed"`
- All chunks have `feature_schema_hash: "f077a15d37e1157a"`
- No failed chunks
- Row counts are reasonable (>100K per chunk)

---

### Step 2: Merge Chunks into Full Dataset 📦

```bash
cd /workspace/gogooku3/gogooku5/data

python3 tools/merge_chunks.py \
  --chunks-dir /workspace/gogooku3/output_g5/chunks \
  --output /workspace/gogooku3/output_g5/ml_dataset_full.parquet \
  --validate
```

**Expected Output**:
```
✅ Merged 24 chunks successfully
   Total rows: 5,000,000-6,000,000
   Columns: 2,727
   Schema hash: f077a15d37e1157a
   Date range: 2020-01-01 → 2025-11-07
```

**Verification**:
```bash
# Check merged dataset
python3 << 'EOF'
import polars as pl
df = pl.scan_parquet("/workspace/gogooku3/output_g5/ml_dataset_full.parquet")
schema = df.collect_schema()
print(f"Total columns: {len(schema)}")
print(f"Total rows: {df.select(pl.count()).collect().item():,}")
print(f"Date range: {df.select(pl.min('Date')).collect().item()} to {df.select(pl.max('Date')).collect().item()}")
print(f"\nKey columns:")
print(f"  Code: {schema.get('Code')}")
print(f"  SecId: {schema.get('SecId')}")
print(f"  fut_topix_front_ret_1d: {'✅ Present' if 'fut_topix_front_ret_1d' in schema else '❌ Missing'}")
print(f"  date_idxopt: {'✅ Present' if 'date_idxopt' in schema else '❌ Missing'}")
EOF
```

---

### Step 3: Deploy to GCS ☁️

```bash
cd /workspace/gogooku3

# Sync output_g5 directory to GCS
python3 scripts/sync_multi_dirs_to_gcs.py \
  --dirs output_g5 \
  --bucket gogooku-datasets

# Or use direct GCS upload
gsutil -m rsync -r output_g5/ gs://gogooku-datasets/output_g5/
```

**Expected Output**:
```
✅ Uploaded ml_dataset_full.parquet (X GB)
✅ Uploaded dim_security.parquet (49 KB)
✅ Uploaded 24 chunk directories
✅ Total: X GB uploaded
```

---

## Progress Checkpoints

### Milestone 1: First Chunk Complete (~10 minutes)
- ✅ 2020Q1 chunk completes
- Verify: `ls output_g5/chunks/2020Q1/status.json`
- Expected: `"state": "completed"`, `rows: ~210,000`

### Milestone 2: 25% Complete (~45 minutes)
- ✅ 6 chunks completed (2020Q1-2021Q2)
- Verify: `python3 scripts/check_chunk_status.py`

### Milestone 3: 50% Complete (~1.5 hours)
- ✅ 12 chunks completed (2020Q1-2022Q4)

### Milestone 4: 75% Complete (~2 hours)
- ✅ 18 chunks completed (2020Q1-2024Q2)

### Milestone 5: 100% Complete (~2.5-3 hours)
- ✅ All 24 chunks completed (2020Q1-2025Q4)
- Ready for merge

---

## Troubleshooting

### If Process Stops Unexpectedly

```bash
# Check if process is still running
ps -p 222348

# If stopped, check exit status in logs
tail -50 _logs/chunk_rebuild_phase3_20251114_101930.log

# Identify last completed chunk
python3 scripts/check_chunk_status.py

# Resume from last checkpoint
DIM_SECURITY_PATH=/workspace/gogooku3/output_g5/dim_security.parquet \
RAW_MANIFEST_PATH=/workspace/gogooku3/output_g5/raw_manifest.json \
CATEGORICAL_COLUMNS="Code,SecId,SectorCode" \
./venv/bin/python gogooku5/data/scripts/build_chunks.py \
  --start 2020-01-01 \
  --end 2025-11-07 \
  --resume \
  > _logs/chunk_rebuild_resume_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### If Chunks Fail Schema Validation

```bash
# Identify failed chunks
python3 scripts/check_chunk_status.py | grep "❌"

# Check error details
cat output_g5/chunks/<FAILED_CHUNK>/status.json | python3 -m json.tool

# Common issues:
# 1. Missing futures columns → Check futures_topix.py fix was applied
# 2. Type mismatches → Verify categorical conversion in lazy_io.py
# 3. Missing index option columns → Check API availability for early dates
```

---

## Key Files Modified

### Code Fixes Applied
1. **`gogooku5/data/src/builder/features/macro/futures_topix.py:167-170`**
   - Added `.with_columns(pl.col("date").cast(pl.Date, strict=False))`
   - Fixes: Object → Date type mismatch in futures join

2. **`output/cache/dim_security.parquet`** (symlink)
   - Points to `/workspace/gogooku3/output_g5/dim_security.parquet`
   - Fixes: dim_security path resolution

### Backup Created
- **Location**: `/workspace/gogooku3/output_g5/chunks_backup_phase2_20251114_101737`
- **Size**: 13 GB
- **Contents**: 7 Phase 2 chunks (2015Q1, 2020Q1-Q4, 2023Q4, 2024Q1)
- **Status**: Mixed schemas (Phase 2 and incomplete Phase 3)

---

## Success Criteria

**All chunks must meet these criteria**:
- ✅ `state: "completed"`
- ✅ `feature_schema_hash: "f077a15d37e1157a"`
- ✅ `feature_schema_version: "1.2.0"`
- ✅ `rows > 100,000` (typical quarterly chunk)
- ✅ Columns: 2,727
- ✅ Key columns present:
  - `Code` (Categorical)
  - `SecId` (Categorical)
  - `fut_topix_front_ret_1d` (Float64)
  - `date_idxopt` (Date)
  - `idxopt_underlying_price` (Float64)

---

## Next Session Actions

When you return to check status:

1. **Quick Health Check**:
   ```bash
   python3 scripts/check_chunk_status.py
   ```

2. **If Complete** → Proceed to Step 1 (Validation)

3. **If In Progress** → Check estimated time remaining:
   - Count completed chunks
   - Multiply remaining chunks × 7 minutes average
   - Continue waiting

4. **If Failed** → Review troubleshooting section above

---

**Status Last Updated**: 2025-11-14 10:22 JST
**Next Review**: 2025-11-14 12:00 JST (estimated completion)
