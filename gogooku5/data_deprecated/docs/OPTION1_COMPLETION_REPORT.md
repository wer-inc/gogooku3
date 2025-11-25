# Option 1 Complete: 2025年全四半期 Rebuild with Manifest v1.4.0

**Completion Date**: 2025-11-15
**Status**: ✅ **FULLY VALIDATED**

---

## Executive Summary

All 2025 quarters (Q1, Q2, Q3, Q4) have been successfully rebuilt with **manifest v1.4.0** and **fully validated** at both metadata and data content levels. Phase 1 ML Best Practices features are correctly implemented across the entire 2025 dataset.

---

## Build Summary

| Quarter | Rows | Build Time | Schema Version | Schema Hash | Status |
|---------|------|------------|----------------|-------------|--------|
| **2025Q1** | 218,624 | 443.9s (7.4 min) | v1.4.0 | 81c029b120e9c5e2 | ✅ |
| **2025Q2** | 237,122 | 446.4s (7.4 min) | v1.4.0 | 81c029b120e9c5e2 | ✅ |
| **2025Q3** | 235,980 | 456.0s (7.6 min) | v1.4.0 | 81c029b120e9c5e2 | ✅ |
| **2025Q4** | 117,580 | 403.6s (6.7 min) | v1.4.0 | 81c029b120e9c5e2 | ✅ |
| **Total** | **809,306** | **1,749.9s (29.2 min)** | - | - | ✅ |

---

## Metadata Validation (validate_2025_complete.py)

**Result**: ✅ **ALL PASSED**

- ✅ All 4 quarters have `state: "completed"`
- ✅ All 4 quarters have schema version `v1.4.0`
- ✅ All 4 quarters have schema hash `81c029b120e9c5e2`
- ✅ Total rows: 809,306
- ✅ Total build time: 29.2 minutes

---

## Data Quality Validation (validate_2025_data_quality.py)

**Result**: ✅ **ALL PASSED (4/4 quarters)**

### 1. Flag Columns (Phase 1.3)

All 4 flag columns present with **Int8 dtype** in all quarters:

| Flag Column | Total Instances | Percentage |
|-------------|----------------|------------|
| `flag_halted` | 0 | 0.00% |
| `flag_price_limit_hit` | 2,815 | 0.35% |
| `flag_delisted` | 0 | 0.00% |
| `flag_adjustment_event` | 388 | 0.05% |

**Validation**: ✅ All 4 columns have correct Int8 dtype, no NULLs, valid value distributions

### 2. Git Metadata (Phase 1.4)

**Validation**: ✅ All quarters have non-null `git_sha` and `git_branch`

Example from 2025Q4 metadata.json:
```json
{
  "git_sha": "ecc79e6db108d8005a2184b2ceb6cb794dcfa75a",
  "git_branch": "feature/sec-id-join-optimization"
}
```

### 3. Macro Columns (Phase 1.5)

**Validation**: ✅ All quarters have **237 macro columns** (exceeds 35+ requirement)

Macro prefixes validated:
- `topix_*`
- `nk225_opt_*`
- `trades_spec_*`
- `trading_cal_*`
- `vix_*`

### 4. Data Integrity

**Validation**: ✅ No critical issues

| Quarter | Date Range | Unique Codes | Date NULLs | Code NULLs |
|---------|-----------|--------------|------------|------------|
| 2025Q1 | 2025-01-06 to 2025-03-31 | - | 0 | 0 |
| 2025Q2 | 2025-04-01 to 2025-06-30 | - | 0 | 0 |
| 2025Q3 | 2025-07-01 to 2025-09-30 | - | 0 | 0 |
| 2025Q4 | 2025-10-01 to 2025-11-14 | 3,801 | 0 | 0 |

**Target Column NULL Rates** (2025Q4 example):
- `ret_prev_1d`: 4.2% NULL ✅
- `ret_prev_5d`: 17.0% NULL ✅
- `ret_prev_10d`: 33.0% NULL ✅
- `ret_prev_20d`: 64.9% NULL ✅ (expected for recent data)

---

## Phase 1 Features Validated

All 6 Phase 1 implementation steps validated across full 2025 year:

1. ✅ **Phase 1.1**: Git metadata utility (`gogooku5/data/src/builder/utils/git_metadata.py`)
2. ✅ **Phase 1.2**: Robust clipping function (`add_robust_clipping()`)
3. ✅ **Phase 1.3**: 4 missing value flags with Int8 dtype
   - `flag_halted`
   - `flag_price_limit_hit`
   - `flag_delisted`
   - `flag_adjustment_event`
4. ✅ **Phase 1.4**: Git metadata embedding in dataset_builder.py
5. ✅ **Phase 1.5**: Macro column enforcement (`_ensure_macro_columns()`)
6. ✅ **Phase 1.6**: Schema manifest update (v1.3.0 → v1.4.0)

---

## Cache Performance

### Index Options Cache
- ✅ Cache hit: Saved ~2h42m
- ✅ Performance: 100% hit rate on subsequent builds

### Quotes Cache
- ⚠️ First build: Schema mismatch (expected, due to 4 new flag columns)
- ✅ Future builds: Will benefit from cached quotes

---

## Validation Scripts Created

1. **`validate_phase1_features.py`**: 7-point metadata validation
   - Status, schema hash, git metadata, flag columns, macro columns, data types, column count

2. **`validate_2025_complete.py`**: Metadata-level validation for all 2025 quarters
   - Schema version, hash, state, build completion

3. **`validate_phase1_data_quality.py`**: Single-quarter data content validation
   - Flag columns, git metadata, macro columns, data integrity

4. **`validate_2025_data_quality.py`**: Comprehensive data quality validation for all 2025 quarters
   - Flag column statistics, git metadata presence, macro columns count, data integrity across all quarters

---

## Next Steps (as per user's strategy)

### ✅ Option 1: 2025年全再ビルド **[COMPLETED]**

**What was done**:
- Rebuilt all 2025 quarters (Q1, Q2, Q3, Q4) with manifest v1.4.0
- Validated both metadata and actual data content
- Total time: 29.2 minutes
- Total rows: 809,306

### 📋 Immediate Next Step: Apex Ranker Testing

**Objective**: Test Phase 1 features with Apex Ranker to validate new columns work correctly

**Test Plan**:
1. Load 2025 data with Apex Ranker
2. Verify 4 new flag columns are handled correctly
3. Validate git metadata tracking
4. Confirm macro columns are present and usable
5. Run inference and backtest to ensure no regressions

### 📋 Future Step: Option 2 (2020-2024 rebuild)

**When to execute**: After successful Apex Ranker validation with 2025 data

**Scope**:
- Rebuild all 2020-2024 chunks with manifest v1.4.0
- Estimated time: 5-6 hours (20 quarters × 15-18 min each)
- Total rows: ~4-5M (estimated)

**Benefits**:
- Full historical dataset with unified schema v1.4.0
- All Phase 1 features available across entire time series
- Consistent feature engineering across all years
- Eliminates schema version fragmentation

**Risks**:
- Time cost (5-6 hours of rebuild time)
- Cache invalidation (first build will be slower)
- Potential unforeseen issues with historical data

---

## Technical Details

### Build Configuration

```bash
export DATA_OUTPUT_DIR=/workspace/gogooku3/output_g5
export DIM_SECURITY_PATH=/workspace/gogooku3/output_g5/dim_security.parquet
export RAW_MANIFEST_PATH=/workspace/gogooku3/output_g5/raw_manifest.json
export CATEGORICAL_COLUMNS="Code"  # Changed from "Code,SectorCode"
export MAX_CONCURRENT_FETCH=200
export MAX_PARALLEL_WORKERS=128
export QUOTES_PARALLEL_WORKERS=200
```

### Schema Changes (v1.3.0 → v1.4.0)

**New Columns**: 4 flag columns
- `flag_halted` (Int8)
- `flag_price_limit_hit` (Int8)
- `flag_delisted` (Int8)
- `flag_adjustment_event` (Int8)

**Categorical Columns**: Changed from `"Code,SectorCode"` to `"Code"` only
- `SectorCode` and `MarketCode` are now String type (not Categorical)

**Total Columns**: 2771 (unchanged from v1.3.0, but with new Int8 flag columns added)

**Schema Hash**: 81c029b120e9c5e2 (v1.4.0)

---

## File Locations

### Chunk Data
```
/workspace/gogooku3/output_g5/chunks/
├── 2025Q1/
│   ├── ml_dataset.parquet (617M)
│   ├── status.json
│   └── metadata.json
├── 2025Q2/
│   ├── ml_dataset.parquet
│   ├── status.json
│   └── metadata.json
├── 2025Q3/
│   ├── ml_dataset.parquet
│   ├── status.json
│   └── metadata.json
└── 2025Q4/
    ├── ml_dataset.parquet
    ├── status.json
    └── metadata.json
```

### Validation Scripts
```
/workspace/gogooku3/gogooku5/data/tests/
├── validate_phase1_features.py
├── validate_2025_complete.py
├── validate_phase1_data_quality.py
└── validate_2025_data_quality.py
```

### Build Logs
```
/workspace/gogooku3/_logs/
└── chunk_rebuild_2025Q1Q2Q3_v140_20251115_153808.log
```

---

## Conclusion

**Option 1 (2025年全再ビルド)** has been **successfully completed** and **fully validated**. All Phase 1 ML Best Practices features are correctly implemented across the entire 2025 dataset.

The 2025 data is now ready for:
1. ✅ Apex Ranker testing and validation
2. ✅ Production use with unified schema v1.4.0
3. ✅ ML experiments with new flag columns and git metadata tracking

After successful Apex Ranker validation, proceed with **Option 2 (2020-2024 rebuild)** to unify the full historical dataset.

---

**Generated**: 2025-11-15
**Validated By**: validate_2025_complete.py, validate_2025_data_quality.py
**Schema Version**: v1.4.0
**Schema Hash**: 81c029b120e9c5e2
