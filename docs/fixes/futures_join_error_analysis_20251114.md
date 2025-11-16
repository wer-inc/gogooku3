# TOPIX Futures Features Join Error - Root Cause Analysis

**Date**: 2025-11-14  
**Status**: IDENTIFIED - Type mismatch in date columns during futures feature join  
**Severity**: HIGH - Causes complete failure to attach TOPIX futures features

---

## Error Location

**File**: `/workspace/gogooku3/gogooku5/data/src/builder/features/macro/futures_topix.py`  
**Line**: 259  
**Method**: `build_futures_features()`

```python
# Line 259 - PROBLEMATIC JOIN
fn_df = fn_df.join(topix_clean, on="date", how="left")
```

---

## Root Cause: Date Type Mismatch

### Left DataFrame (fn_df)

**Source**: Concatenation of results from `select_front_next()` (line 236)

```python
# Line 161-167 in select_front_next()
result = pl.DataFrame(
    {
        "date": [date_val],  # date_val extracted from Date column (pl.Date type)
        "front_settle": [front_settle],
        "next_settle": [next_settle],
        # ... other columns
    }
)
```

**Issue**: When `date_val` is a Python `datetime.date` object extracted from a Polars `pl.Date` column, Polars infers it as `Object` type instead of `pl.Date` type.

**Actual Schema**:
```
date: Object (should be Date)
front_settle: Float64
next_settle: Float64
front_oi: Float64
front_last_trading_day: String
front_day_open: Float64
front_day_close: Float64
front_night_close: Float64
```

### Right DataFrame (topix_clean)

**Source**: Lines 252-257

```python
topix_clean = topix_df.select(
    [
        pl.col(date_col).cast(pl.Date, strict=False).alias("date"),  # Explicitly cast to pl.Date
        pl.col(close_col).cast(pl.Float64, strict=False).alias("topix_close"),
    ]
).filter(pl.col("topix_close").is_not_null() & pl.col("date").is_not_null())
```

**Schema**:
```
date: Date (explicit cast)
topix_close: Float64
```

### Join Failure

When attempting to join on `"date"`:
```
Left:  date: Object
Right: date: Date
→ ERROR: datatypes of join keys don't match
```

Polars cannot automatically cast `Object` to `Date` because the Object type is ambiguous.

---

## Why This Happens

1. **Line 110**: Date extracted from DataFrame column using `.head(1).item()`
   ```python
   date_val = df["Date"].head(1).item() if "Date" in df.columns else None
   ```
   This returns a Python `datetime.date` object

2. **Line 161**: Placed in a list and passed to DataFrame constructor
   ```python
   "date": [date_val],
   ```
   Polars cannot infer the type correctly and uses `Object` instead of `Date`

3. **Line 259**: Join fails because left and right schemas don't match
   ```
   Object ≠ Date
   ```

---

## Call Stack

1. **dataset_builder.py:746** - Calls `add_futures_topix(combined_df, futures_features)`
2. **engineer.py:111** - Simple `df.join(merge_select, on=self.date_column, how="left")`
3. **futures_topix.py:259** - The problematic join between `fn_df` (Object type date) and `topix_clean` (Date type date)

---

## Recommended Fix

### Option A: Explicit Cast in select_front_next() (RECOMMENDED)

**File**: `/workspace/gogooku3/gogooku5/data/src/builder/features/macro/futures_topix.py`  
**Lines**: 159-167

**Current Code**:
```python
result = pl.DataFrame(
    {
        "date": [date_val],
        "front_settle": [front_settle],
        "next_settle": [next_settle],
        "front_oi": [front_oi],
        "front_last_trading_day": [front_last_trading_day],
    }
)
```

**Fixed Code**:
```python
result = pl.DataFrame(
    {
        "date": [date_val],
        "front_settle": [front_settle],
        "next_settle": [next_settle],
        "front_oi": [front_oi],
        "front_last_trading_day": [front_last_trading_day],
    }
).with_columns(
    pl.col("date").cast(pl.Date, strict=False)  # Explicitly cast to pl.Date
)
```

**Advantages**:
- Fixes the problem at the source
- Ensures `fn_df` always has correct type
- Simple, one-line fix
- Type is correct immediately after creation

---

### Option B: Cast Before Join (Defensive)

**File**: `/workspace/gogooku3/gogooku5/data/src/builder/features/macro/futures_topix.py`  
**Lines**: 246-259

**Current Code**:
```python
if topix_df is not None and not topix_df.is_empty():
    # ... prepare topix_clean ...
    fn_df = fn_df.join(topix_clean, on="date", how="left")
```

**Fixed Code**:
```python
if topix_df is not None and not topix_df.is_empty():
    # ... prepare topix_clean ...
    # Ensure date columns have matching types
    fn_df = fn_df.with_columns(pl.col("date").cast(pl.Date, strict=False))
    fn_df = fn_df.join(topix_clean, on="date", how="left")
```

**Advantages**:
- Defensive programming approach
- Works even if Option A isn't applied
- Catches type mismatches before join
- Clear intent in join section

---

### Option C: Consistent Type from Start (BEST)

**File**: `/workspace/gogooku3/gogooku5/data/src/builder/features/macro/futures_topix.py`  
**Lines**: 108-112

Instead of extracting raw date value and inferring type, explicitly convert:

**Current Code**:
```python
date_val = df["Date"].head(1).item() if "Date" in df.columns else None
```

**Fixed Code**:
```python
# Extract and convert to Date type immediately
date_df = df.select(pl.col("Date").cast(pl.Date, strict=False).alias("date"))
if not date_df.is_empty():
    date_val = date_df["date"].head(1).item()
else:
    date_val = None
```

**Advantages**:
- Most robust approach
- Ensures type consistency throughout
- Prevents future similar issues
- Clear data transformation chain

---

## Test Case

**File**: To be created at `/workspace/gogooku3/gogooku5/data/tests/test_futures_join_fix.py`

```python
import polars as pl
import pytest
from gogooku5.data.src.builder.features.macro.futures_topix import (
    select_front_next,
    build_futures_features,
)

def test_select_front_next_date_type():
    """Verify select_front_next produces Date type, not Object."""
    # Create sample futures data with Date column
    fut_df = pl.DataFrame({
        "Date": [pl.date(2025, 1, 6)],
        "ProductCategory": ["TOPIXF"],
        "CentralContractMonthFlag": ["1"],
        "ContractMonth": ["202501"],
        "SettlementPrice": [28500.0],
        "OpenInterest": [100000.0],
        "DaySessionOpen": [28400.0],
        "DaySessionClose": [28500.0],
        "NightSessionClose": [28450.0],
    }).with_columns(pl.col("Date").cast(pl.Date, strict=False))
    
    result = select_front_next(fut_df)
    
    # CRITICAL: date column must be pl.Date, not Object
    assert result.schema["date"] == pl.Date, (
        f"Expected date column type pl.Date, got {result.schema['date']}"
    )

def test_build_futures_features_join_types():
    """Verify build_futures_features can join with topix_df."""
    # Create futures data
    fut_df = pl.DataFrame({
        "Date": [pl.date(2025, 1, 6), pl.date(2025, 1, 7)],
        "ProductCategory": ["TOPIXF", "TOPIXF"],
        "DerivativesProductCategory": ["TOPIXF", "TOPIXF"],
        "CentralContractMonthFlag": ["1", "1"],
        "ContractMonth": ["202501", "202501"],
        "SettlementPrice": [28500.0, 28550.0],
        "OpenInterest": [100000.0, 105000.0],
        "DaySessionOpen": [28400.0, 28450.0],
        "DaySessionClose": [28500.0, 28550.0],
        "NightSessionClose": [28450.0, 28500.0],
    })
    
    # Create TOPIX spot data
    topix_df = pl.DataFrame({
        "Date": [pl.date(2025, 1, 6), pl.date(2025, 1, 7)],
        "Close": [2500.0, 2505.0],
    })
    
    # This should not raise TypeError
    result = build_futures_features(fut_df, topix_df=topix_df)
    
    # Result should have date column
    assert "date" in result.columns
    assert result.schema["date"] == pl.Date
    
    # topix_close should exist (join successful)
    assert "topix_close" in result.columns or result.is_empty()
```

---

## Impact Assessment

**Severity**: HIGH  
**Frequency**: ALWAYS - Occurs on every dataset build that includes TOPIX futures  
**Silent Failure**: YES - Error is caught and logged as warning, but futures features are silently skipped

**Current Behavior**:
```
LOGGER.warning("Failed to attach TOPIX futures features: %s", exc)
```

**Impact on Dataset**:
- Futures-based features (`fut_topix_*`) are completely missing
- No basis calculations, carry approximation, or futures regime indicators
- Dataset is functionally incomplete despite no error visible in main logs
- Only visible in detailed logs/exceptions

---

## Implementation Steps

1. **Apply Fix** (Recommended: Option A + Option B for robustness)
   - Add cast in `select_front_next()` return (line 167)
   - Add defensive cast before join (line 258)

2. **Add Tests**
   - Unit test for `select_front_next()` date type
   - Integration test for `build_futures_features()` join

3. **Verify**
   ```bash
   pytest gogooku5/data/tests/test_futures_join_fix.py -v
   ```

4. **Test Full Pipeline**
   ```bash
   make dataset-bg START=2025-01-01 END=2025-01-31
   # Check logs for: "TOPIX futures features attached: X features"
   ```

5. **Validate Output**
   - Confirm futures features in final dataset
   - Check `fut_topix_*` columns present
   - Verify no NULL values for valid dates

---

## References

- **Polars Type Inference**: https://docs.pola-rs.com/user-guide/concepts/data-types/
- **Join Operations**: https://docs.pola-rs.com/api/python/stable/reference/dataframe/api/polars.DataFrame.join.html
- **Cast Behavior**: https://docs.pola-rs.com/api/python/stable/reference/expressions/api/polars.Expr.cast.html

---

## Related Issues

- Similar type mismatches possible in:
  - `add_options_features()` (line 164 in engineer.py)
  - Any date-based joins without explicit type alignment

---

## Phase Information

**Migration Phase**: Phase 3.2 - SEC_ID Join Optimization  
**Related Fixes**: Short selling data normalization (2025-10-20), Polars date append error (2025-10-14)
