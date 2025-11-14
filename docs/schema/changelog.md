# Feature Schema Manifest - Changelog

This document tracks changes to the feature schema manifest used for chunk validation.

---

## v1.1.0 (2025-11-12)

**Status**: ✅ Active

**Schema Hash**: `2ea3ac61abd9af19`

**Total Columns**: 2784

**Reference Dataset**: `gogooku5/data/output/chunks/2020Q1/ml_dataset.parquet`

### Changes from v1.0.0

#### Removed Columns (4)
- `target_1d` (Float64)
- `target_5d` (Float64)
- `target_10d` (Float64)
- `target_20d` (Float64)

**Reason**: These columns represent future returns and are added **after** chunk merge, not during chunk creation. The v1.0.0 manifest was incorrectly generated from a post-merge dataset (`ml_dataset_2024H1_merged_final.parquet`), which already contained these columns.

### Validation Impact

**Before v1.1.0** (with v1.0.0 manifest):
- All 15 chunks failed validation (100% failure rate)
- False positives: target_* columns marked as "missing"
- 2023Q1-Q2: DisclosedDate genuinely missing

**After v1.1.0** (expected):
- 2020Q1-2022Q4: ✓ Pass (12/15 chunks, 80%)
- 2023Q1-Q2: ✗ Fail (DisclosedDate missing, type mismatches)
- 2025Q1.skip: ✗ Fail (TBD)

### Schema Definition

**Core Columns**:
- `Date` (Date): Trading date
- `Code` (String): Stock ticker
- `DisclosedDate` (Date): Financial statement disclosure date

**Feature Groups** (2781 columns):
- Price features: OHLCV, returns, volatility
- Volume features: Daily, weekly, rolling averages
- Financial statements: 48 columns (fs_*)
- Sector features: Sector codes, industry classification
- Index features: TOPIX, NK225 relationships
- Technical indicators: Moving averages, RSI, MACD
- Graph features: Correlation networks, PageRank
- Options features: Index options data

**Metadata Columns**:
- `fs_observation_count` (Int64): Number of financial statements observed
- `fs_lag_days` (Int64): Days since last financial statement

### Known Issues

#### 2023Q1-Q2 Schema Problems

**Missing Column**:
- `DisclosedDate` (Date) - Absent in 2023Q1 and 2023Q2 chunks

**Type Mismatches**:
- `fs_observation_count`: Expected Int64, got Int16
- `fs_lag_days`: Expected Int64, got Int32

**Cause**: Likely due to:
1. Data source API changes
2. Feature generation logic modifications
3. Memory optimization attempts (downcast to smaller types)

**Resolution**: Rebuild 2023Q1-Q2 chunks with correct schema

### Migration Guide

#### For Existing Datasets

**If using post-merge datasets** (with target_* columns):
- No action needed for training pipelines
- Continue using existing merged datasets
- New chunks will align with v1.1.0 (no target_*)

**If building new chunks**:
- Use v1.1.0 manifest for validation
- Chunks will pass validation without target_* columns
- Add target_* columns after merge using post-processing

#### For CI/CD Pipelines

Update validation commands:
```bash
# Old (v1.0.0)
python tools/check_chunks.py --validate-schema --schema-manifest schema/feature_schema_manifest_v1.0.0_backup.json

# New (v1.1.0)
python tools/check_chunks.py --validate-schema  # Uses v1.1.0 by default
```

### Backward Compatibility

**v1.0.0 Manifest**: Backed up to `schema/feature_schema_manifest_v1.0.0_backup.json`

**Restoration Command** (if rollback needed):
```bash
cd /workspace/gogooku3/gogooku5/data
cp schema/feature_schema_manifest_v1.0.0_backup.json schema/feature_schema_manifest.json
```

**When to Rollback**:
- If v1.1.0 validation fails unexpectedly
- If new chunks show different schema issues
- If merge process depends on post-merge schema

---

## v1.0.0 (2025-11-12)

**Status**: ⚠️ Deprecated

**Schema Hash**: `2875957eecefb206`

**Total Columns**: 2788

**Reference Dataset**: `output/ml_dataset_2024H1_merged_final.parquet` (POST-MERGE)

### Issues Discovered

1. **Incorrect Reference**: Used post-merge dataset instead of chunk-time dataset
2. **False Failures**: All chunks failed validation due to missing target_* columns
3. **Misleading Errors**: Real issues (DisclosedDate, type mismatches) hidden by false positives

### Lessons Learned

1. **Schema Versioning**: Always version manifests with clear reference sources
2. **Chunk vs Merge Schema**: Distinguish between chunk-time and merge-time schemas
3. **Validation Context**: Validate chunks against chunk-time schema, not merge-time schema
4. **Traceability**: Document source datasets and generation procedures

---

## Future Versions

### v1.2.0 (Planned)

**Scope**: 2025Q1+ data with potential schema evolution

**Considerations**:
- New data source APIs
- Additional feature engineering
- Type optimizations (with explicit documentation)

### v2.0.0 (Future)

**Breaking Changes**:
- Major schema refactoring
- Column renames or removals
- Type changes with business logic impact

**Migration Strategy**:
- Multi-version support period
- Automated migration tools
- Clear deprecation timeline

---

**Last Updated**: 2025-11-12
**Maintainer**: gogooku3 Team
**Related Docs**: `DAGSTER_OPERATIONS_GUIDE.md`, `schema_validation_implementation_20251112.md`
