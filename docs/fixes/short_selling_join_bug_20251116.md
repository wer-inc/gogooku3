# Short Selling Data Loss Bug - Root Cause Analysis

**Date**: 2025-11-16
**Severity**: Critical
**Impact**: 100% data loss for all 21 short_selling features
**Status**: Root cause identified, fix pending

---

## Executive Summary

All 21 short_selling features have 100% NULL values in final dataset despite successful API fetch (4,862 records) and logged "successful" join. Root cause: **Timestamp mismatch in regular join** causing zero matches.

---

## Evidence

### Build Logs (/tmp/build_2024Q1_clean.log)

```
[2025-11-16 02:43:26,557] INFO - Fetching short selling data from 2023-08-29 to 2024-03-31
Retrieved 4862 short selling records
[2025-11-16 02:44:51,480] INFO - Prepared snapshot data: 4862 rows, T+1 availability at 09:00 JST
[2025-11-16 02:44:51,489] INFO - [PATCH D] Joined short selling market features with T+1 as-of join
```

### Dataset Status

```
✅ Build completed: 222,774 rows, 2,771 columns
✅ Schema validation: PASSED (hash: 81c029b120e9c5e2)
❌ Data quality: All 21 short_selling columns 100% NULL
```

---

## Root Cause Analysis

### The Bug (dataset_builder.py:6726-6731)

```python
# WRONG: Regular join requires EXACT timestamp match
result = df.join(
    market.sort("available_ts"),
    left_on="asof_ts",         # ← 15:00 JST
    right_on="available_ts",   # ← 09:00 JST
    how="left",
).drop(["available_ts"], strict=False)
```

### Timestamp Mismatch

| Timestamp | Source | Time | Example |
|-----------|--------|------|---------|
| `asof_ts` | `df` (main dataframe) | **15:00 JST** | 2024-01-05 15:00:00+09:00 |
| `available_ts` | `market` (short selling) | **09:00 JST** | 2024-01-05 09:00:00+09:00 |

Since **15:00 ≠ 09:00**, the regular `join()` returns NULL for all rows (zero exact matches).

### Why This Happened

1. **Line 546**: `asof_ts` added to main dataframe at **15:00 JST**
   ```python
   combined_df = add_asof_timestamp(aligned_quotes, date_col="date")
   # Default: time_jst=time(15, 0)
   ```

2. **Line 6667**: `available_ts` prepared for short selling at **09:00 JST**
   ```python
   short_prepared = prepare_snapshot_pl(
       short_df,
       availability_hour=9,  # ← 9am JST
       availability_minute=0,
   )
   ```

3. **Lines 6726-6731**: Regular `join()` used instead of `join_asof()`
   - Regular join requires **exact** timestamp match
   - As-of join would match **nearest past** timestamp
   - Result: 4,862 records fetched, **0 records joined**

---

## The Fix

### Option 1: Use join_asof() (Recommended)

Replace regular `join()` with Polars `join_asof()`:

```python
# CORRECT: As-of join matches nearest past timestamp
result = df.join_asof(
    market.sort("available_ts"),
    left_on="asof_ts",
    right_on="available_ts",
    strategy="backward",  # Use latest past data
).drop(["available_ts"], strict=False)
```

**Benefits**:
- Matches 09:00 data to 15:00 backbone automatically
- Consistent with other as-of joins in codebase
- Handles T+1 availability correctly

**Reference**: See `interval_join_pl()` in `/workspace/gogooku3/gogooku5/data/src/builder/utils/asof.py:160-237`

### Option 2: Align Timestamps (Alternative)

Change short selling to use 15:00 availability:

```python
short_prepared = prepare_snapshot_pl(
    short_df,
    availability_hour=15,  # ← Match backbone timestamp
    availability_minute=0,
)
```

**Drawbacks**:
- Incorrect business logic (short selling data available at 09:00, not 15:00)
- Violates T+1 temporal safety principle

---

## Impact Assessment

### Data Loss

- **Records fetched**: 4,862 (API successful)
- **Records joined**: 0 (timestamp mismatch)
- **Data loss**: 100%

### Affected Features (21 columns)

```
short_selling_ratio_market
short_selling_ratio_market_outlier_flag
short_selling_ratio_market_ewm_30
short_selling_ratio_market_ewm_30_outlier_flag
short_selling_with_restrictions_ratio
short_selling_with_restrictions_ratio_outlier_flag
short_selling_with_restrictions_ratio_ewm_30
short_selling_with_restrictions_ratio_ewm_30_outlier_flag
short_selling_without_restrictions_ratio
short_selling_without_restrictions_ratio_outlier_flag
short_selling_without_restrictions_ratio_ewm_30
short_selling_without_restrictions_ratio_ewm_30_outlier_flag
short_selling_sector_ratio
short_selling_sector_ratio_outlier_flag
short_selling_sector_ratio_ewm_30
short_selling_sector_ratio_ewm_30_outlier_flag
short_selling_sector_rel
short_selling_sector_rel_outlier_flag
short_selling_sector_rel_ewm_30
short_selling_sector_rel_ewm_30_outlier_flag
short_positions_ratio
```

### Historical Scope

- **All periods affected**: 2024Q1, 2024Q2-Q4, 2025Q1-Q4
- **Why 2025 seemed OK**: Used corrupted cache (also NULL), bypassing buggy join
- **Why 2024Q1 exposed it**: No cache, forced fresh API fetch → hit buggy join

---

## Code Locations

| File | Lines | Description |
|------|-------|-------------|
| `dataset_builder.py` | 6629-6830 | `_add_short_selling_features_asof()` method |
| `dataset_builder.py` | 6726-6731 | ❌ **Buggy regular join** |
| `dataset_builder.py` | 546 | `asof_ts` creation (15:00 JST) |
| `asof.py` | 160-237 | ✅ Correct `interval_join_pl()` reference implementation |

---

## Similar Bugs (Potential)

Search codebase for other instances of this pattern:

```bash
# Find all regular joins on timestamp columns
grep -n "\.join(" dataset_builder.py | grep -i "ts\|time\|date"

# Expected: Should use join_asof() for temporal data
```

---

## Next Steps

1. ✅ **Root cause identified** (this document)
2. 📋 **Implement fix**: Change `join()` to `join_asof()` at line 6726
3. 📋 **Add unit test**: Verify timestamp mismatch handling
4. 📋 **Delete all short_selling cache**: Corrupted with NULL data
5. 📋 **Rebuild 2024Q1**: Verify fix with real data
6. 📋 **Rebuild all 2024-2025 periods**: Full dataset regeneration

---

## Prevention

### Code Review Checklist

- [ ] All timestamp joins use `join_asof()` not `join()`
- [ ] Temporal data has explicit availability time documentation
- [ ] T+1 availability logic verified with unit tests
- [ ] Join result validation (check for unexpected NULL rates)

### Logging Enhancement

Add post-join validation:

```python
result = df.join_asof(market, ...)

# Validate join success
null_rate = result.select(pl.col("short_selling_ratio_market").is_null().mean()).item()
if null_rate > 0.95:
    LOGGER.error(f"High NULL rate ({null_rate:.1%}) after short_selling join - possible timestamp mismatch")
```

---

## References

- User requirement: "データはnullは良くないです。問題です。根本原因を調査してください"
- As-of join utility: `/workspace/gogooku3/gogooku5/data/src/builder/utils/asof.py`
- Feature schema: `/workspace/gogooku3/gogooku5/data/schema/feature_schema_manifest.json` (v1.4.0)
- Build log: `/tmp/build_2024Q1_clean.log`

---

**Documented by**: Claude Code
**Investigation duration**: ~30 minutes
**Key insight**: "Logs say successful join, but regular join requires EXACT timestamp match (15:00 ≠ 09:00)"
