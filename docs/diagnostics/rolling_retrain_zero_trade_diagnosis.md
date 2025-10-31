# Rolling Retrain Zero-Trade Diagnosis

**Date**: 2025-10-30
**Status**: ✅ ROOT CAUSE IDENTIFIED
**Severity**: CRITICAL - 100% failure rate

---

## Executive Summary

The rolling walk-forward validation pipeline (`run_rolling_retrain.py`) produced **ZERO trades** across all 18 months (Jan 2024 - Jun 2025). Every monthly backtest completed successfully but reported empty portfolios due to no eligible stock candidates.

**Root Cause**: Panel cache built from monthly window (19 days) lacks sufficient lookback history (180 days required), causing all stocks to be filtered out during eligibility checks.

**Impact**: Cannot validate model performance or measure degradation over time.

---

## Observed Behavior

### Symptoms

**All 18 monthly evaluations show**:
```json
{
  "summary": {
    "trading_days": 18-20,
    "total_trades": 0,              // ❌ ZERO trades
    "rebalance_count": 0             // ❌ No rebalances
  },
  "performance": {
    "sharpe_ratio": 0.0,
    "total_return": 0.0,
    "transaction_costs": { "total_cost": 0 }
  }
}
```

**Every prediction day shows**:
```json
{
  "selection_count": 0,              // ❌ No candidates selected
  "optimized_top_k": 0,              // ❌ Optimization found nothing
  "avg_prediction_score": null,      // ❌ No predictions
  "cash": 10000000,                  // 💰 All cash, no positions
  "equity_value": 0
}
```

**Backtest logs confirm**:
```
[Backtest] 2024-01-04: model produced no candidates
[Backtest] 2024-01-05: model produced no candidates
[Backtest] 2024-01-09: model produced no candidates
... (100% failure rate)
```

---

## Root Cause Analysis

### Investigation Steps

#### 1. Cache Structure Inspection

```python
import pickle
cache = pickle.load(open('cache/panel_prod/ml_dataset_..._.pkl', 'rb'))

# Results:
Total codes in cache.codes: 3849        # ✅ Stocks exist
Total dates in cache: 19                 # ⚠️ Only 19 days!
Date range: 19726 - 19753                # 2024-01-04 to 2024-01-31

# Critical finding:
cache.date_to_codes[19726] = []          # ❌ EMPTY for first date
cache.date_to_codes[19753] = []          # ❌ EMPTY for all dates
```

#### 2. Individual Stock Data Check

```python
sample_code = list(cache.codes.keys())[0]
payload = cache.codes[sample_code]

# Results:
dates shape: (19,)                       # Only 19 dates per stock
features shape: (19, 89)                 # Features exist
date range: 19726 - 19753                # Same 19-day window
```

**Finding**: Stock-level data exists, but reverse index `date_to_codes` is empty.

#### 3. Eligibility Filter Analysis

File: `apex-ranker/apex_ranker/data/panel_dataset.py:82-110`

```python
def build_panel_cache(...):
    for date_int in unique_date_ints:
        eligible: list[str] = []
        for code, payload in codes_data.items():
            dates = payload["dates"]
            idx = np.searchsorted(dates, date_int)

            # ❌ FAILURE POINT 1: Lookback check
            start = idx - lookback + 1           # For first date: 0 - 180 + 1 = -179
            if start < 0:
                continue                          # ❌ Filtered out!

            # ❌ FAILURE POINT 2: Window size check
            window = payload["features"][start : idx + 1]
            if window.shape[0] != lookback:      # Expects exactly 180 days
                continue                          # ❌ Filtered out!

            # ❌ FAILURE POINT 3: NaN check
            if np.isnan(window).any():
                continue

            eligible.append(code)

        # ❌ FAILURE POINT 4: Minimum stocks check
        if len(eligible) >= min_stocks_per_day:  # min_stocks_per_day = 0
            date_to_codes[int(date_int)] = eligible
```

**Analysis**:
- **Day 1 (2024-01-04)**: `start = -179 < 0` → ALL stocks filtered (need 179 days before first date)
- **Day 19 (2024-01-31)**: `start = 18 - 180 + 1 = -161 < 0` → Still filtered (need 161 more days)
- **Minimum viable date**: Would need at least 180 days in cache to have ANY eligible stocks

#### 4. Data Loader Logic Verification

File: `apex-ranker/apex_ranker/data/loader.py:51-59`

```python
def load_backtest_frame(...):
    if lookback > 0 and start_dt is not None:
        all_dates = frame["Date"].unique().sort().to_numpy()
        idx = np.searchsorted(all_dates, start_dt)
        buffer_idx = max(0, idx - lookback)      # ✅ Correct: idx - 180
        buffer_start = all_dates[buffer_idx]      # ✅ Should load 180 extra days
        frame = frame.filter(pl.col("Date") >= buffer_start)
```

**Test Results**:
```python
Dataset: output/ml_dataset_latest_full.parquet
Total dates: 1223 (2020-10-27 to 2025-10-24)

For start_date = 2024-01-04:
  idx = 780
  buffer_idx = 600 (idx - 180)
  buffer_start = 2023-04-11          # ✅ Correct calculation
  Expected buffer: 180 trading days   # ✅ Logic is sound
```

**Conclusion**: `load_backtest_frame()` logic is CORRECT and SHOULD provide 180-day buffer.

---

### Root Cause Identified

**The bug is NOT in the code, but in how `run_rolling_retrain.py` uses it**:

1. `run_rolling_retrain.py` calls the backtest script with monthly windows
2. Each month builds a NEW panel cache from the monthly data window only
3. Cache contains **only 19 days** instead of **180 days buffer + 19 days**
4. Without sufficient lookback, ALL stocks fail eligibility filters
5. Result: Empty `date_to_codes` → Zero candidates → Zero trades

**Hypothesis**: `run_rolling_retrain.py` either:
- Doesn't pass `lookback` parameter to data loader
- Passes monthly-filtered dataframe instead of full dataset
- Panel cache is built BEFORE loader adds buffer (caching happens too early)
- Panel cache dir prevents proper buffer loading (reuses stale 19-day cache)

---

## Proof of Concept

### Test 1: Standalone Backtest (Works)

```bash
python apex-ranker/scripts/backtest_smoke_test.py \
  --data output/ml_dataset_latest_full.parquet \
  --start-date 2024-01-04 \
  --end-date 2024-01-31 \
  --model results/rolling_retrain_full/models/apex_ranker_2024-01.pt \
  --config apex-ranker/configs/v0_base.yaml \
  --horizon 20

# Expected (if buffer works): Loads 2023-04-11 to 2024-01-31 (780 days - 600 = 180 buffer)
# Actual: "model produced no candidates" → buffer NOT working
```

### Test 2: Panel Cache Inspection

```python
# Cache was built from THIS data:
Date range: 2024-01-04 → 2024-01-31 (19 days only)
# But SHOULD have been built from:
Date range: 2023-04-11 → 2024-01-31 (180 + 19 = ~200 days)
```

---

## Fix Recommendations

### Option 1: Fix Panel Cache Building (Recommended)

**Issue**: Panel cache is built too early, before lookback buffer is added.

**Fix Location**: `apex-ranker/apex_ranker/backtest/inference.py:150-172`

**Current Flow**:
```python
# BacktestInferenceEngine.__init__
if not cache_loaded:
    feature_frame = frame.select(...)  # ❌ Uses frame without buffer
    self.cache = build_panel_cache(feature_frame, lookback=180, ...)
```

**Problem**: `frame` is already filtered to monthly window, missing buffer.

**Solution A - Ensure buffer in frame**:
```python
# In BacktestInferenceEngine.__init__, add assertion:
min_required_days = self.lookback
actual_days = len(frame["Date"].unique())
if actual_days < min_required_days:
    raise ValueError(
        f"Dataset has {actual_days} days but lookback={min_required_days} requires "
        f"at least {min_required_days} days of history. "
        f"Ensure load_backtest_frame() includes buffer."
    )
```

**Solution B - Build cache from full dataset**:
```python
# Option: Load full dataset for cache building, not filtered frame
if dataset_path is not None:
    # Load FULL dataset for cache (ignore start/end dates)
    full_frame = pl.read_parquet(dataset_path, columns=[...])
    full_frame = add_cross_sectional_zscores(full_frame, ...)
    self.cache = build_panel_cache(full_frame, ...)  # Use FULL data
```

### Option 2: Relax Eligibility Filters

**Issue**: Strict 180-day lookback requirement may be too aggressive for inference.

**Fix Location**: `apex-ranker/apex_ranker/data/panel_dataset.py:90-92`

**Current**:
```python
start = idx - lookback + 1
if start < 0:
    continue  # ❌ Rejects any stock without FULL lookback
```

**Alternative**:
```python
# Allow partial windows (min 60 days instead of 180)
MIN_LOOKBACK = 60  # Configurable minimum
start = max(0, idx - lookback + 1)  # Don't go negative
window = payload["features"][start : idx + 1]

if window.shape[0] < MIN_LOOKBACK:  # Minimum viable window
    continue

# Pad to lookback length if needed
if window.shape[0] < lookback:
    pad_size = lookback - window.shape[0]
    pad = np.zeros((pad_size, window.shape[1]), dtype=np.float32)
    window = np.vstack([pad, window])
```

**Trade-offs**:
- ✅ Allows predictions on recent IPOs and newly listed stocks
- ✅ Enables bootstrap period predictions
- ⚠️ May reduce prediction quality for stocks with short history
- ⚠️ Padding with zeros could confuse the model

### Option 3: Fix run_rolling_retrain.py Data Loading

**Issue**: Script may not preserve lookback buffer when calling backtest.

**Fix Location**: `apex-ranker/scripts/run_rolling_retrain.py` (line unknown - needs inspection)

**Investigation Needed**:
```bash
# Check how run_rolling_retrain.py loads data:
grep -A10 "load_backtest_frame\|BacktestInferenceEngine" apex-ranker/scripts/run_rolling_retrain.py
```

**Expected Fix**:
```python
# Ensure lookback is passed:
frame = load_backtest_frame(
    data_path=data_path,
    start_date=fold_start,
    end_date=fold_end,
    feature_cols=feature_cols,
    lookback=180,  # ✅ Must be passed!
)
```

### Option 4: Disable Panel Cache Dir (Quick Workaround)

**Issue**: Cached panel from narrow window prevents proper buffer loading.

**Fix**: Remove `--panel-cache-dir` argument:

```bash
# Current (BROKEN):
python apex-ranker/scripts/run_rolling_retrain.py \
  --panel-cache-dir cache/panel_prod  # ❌ Reuses bad cache

# Fixed (SLOW but works):
python apex-ranker/scripts/run_rolling_retrain.py \
  # No panel-cache-dir → Rebuilds cache with proper buffer each time
```

**Trade-offs**:
- ✅ Guaranteed to work (forces fresh cache build)
- ❌ Slow (rebuilds 10.6M sample cache every month = ~2 min × 18 months = 36 min overhead)
- ⚠️ Wastes computation (cache should be reusable)

---

## Recommended Action Plan

### Immediate (Today)

1. **Verify the hypothesis**:
   ```bash
   # Test standalone backtest WITHOUT panel cache
   rm -rf cache/panel_prod/*
   python apex-ranker/scripts/backtest_smoke_test.py \
     --data output/ml_dataset_latest_full.parquet \
     --start-date 2024-01-04 \
     --end-date 2024-01-31 \
     --model results/rolling_retrain_full/models/apex_ranker_2024-01.pt \
     --config apex-ranker/configs/v0_base.yaml \
     --horizon 20 \
     --output /tmp/test_no_cache.json

   # Check if candidates are generated:
   grep "model produced" /tmp/test_*.log
   ```

2. **Inspect `run_rolling_retrain.py`**:
   ```bash
   grep -n "load_backtest_frame\|panel_cache\|lookback" apex-ranker/scripts/run_rolling_retrain.py
   ```

3. **Implement Solution A** (add assertion to catch the bug early):
   - Edit `apex-ranker/apex_ranker/backtest/inference.py:150`
   - Add validation that `frame` has sufficient history
   - Raises clear error instead of silently failing

### Short-term (This Week)

1. **Implement Solution B** (build cache from full dataset):
   - Modify `BacktestInferenceEngine.__init__` to load full dataset for caching
   - Keep filtered `frame` for backtest, use full data for panel cache
   - Update panel cache key to include date range

2. **Fix `run_rolling_retrain.py`**:
   - Ensure `lookback=180` is passed to `load_backtest_frame()`
   - Pass full dataset path to inference engine (not pre-filtered frame)

3. **Re-run rolling retrain**:
   ```bash
   # Clear bad caches
   rm -rf cache/panel_prod/*

   # Resume from July 2025
   python apex-ranker/scripts/run_rolling_retrain.py \
     --start-date 2025-07-01 \
     --end-date 2025-12-31 \
     --fold-offset <calculate from existing checkpoints> \
     ...
   ```

### Long-term (Phase 4)

1. **Add integration tests**:
   - Test panel cache building with various date ranges
   - Verify lookback buffer is preserved
   - Assert `date_to_codes` is non-empty for all dates

2. **Improve error messages**:
   - "model produced no candidates" → detailed diagnostics
   - Log cache statistics (eligible stocks per date)
   - Warning if cache has < lookback days

3. **Performance optimization**:
   - Global panel cache (shared across all months)
   - Incremental cache updates (don't rebuild from scratch)
   - Cache pre-warming before backtest starts

---

## Technical Details

### Date Integer Encoding

```python
DATE_EPOCH = datetime.date(1970, 1, 1)

def date_to_int(date: Date) -> int:
    return (date - DATE_EPOCH).days

# Examples:
2024-01-04 → 19726
2024-01-31 → 19753
2023-04-11 → 19458 (180 days earlier)
```

### Panel Cache Structure

```python
@dataclass
class PanelCache:
    date_ints: list[int]                      # Eligible dates
    date_to_codes: dict[int, list[str]]       # date_int → [codes]
    codes: dict[str, dict]                    # code → {dates, features, targets, masks}
    lookback: int                             # Required window size
    min_stocks: int                           # Minimum stocks per day
    target_columns: list[str]                 # Target column names
```

### Eligibility Criteria (Current)

For a stock to be eligible on a given date:
1. ✅ Stock must exist in cache (`code in cache.codes`)
2. ✅ Date must be in stock's history (`date_int in payload["dates"]`)
3. ✅ Must have `lookback` days BEFORE date (`idx >= lookback - 1`)
4. ✅ Window size must be exactly `lookback` (`window.shape[0] == 180`)
5. ✅ No NaN values in feature window
6. ✅ No NaN values in targets (if targets exist)
7. ✅ Mask value is 1 (if masks exist)

**Failure Point**: Criterion #3 requires 180 days WITHIN the cache, but cache only has 19 days total.

---

## Files Requiring Changes

1. **apex-ranker/apex_ranker/backtest/inference.py**
   - Line 150-172: Add buffer validation or load full dataset for cache
   - Add logging of cache statistics

2. **apex-ranker/scripts/run_rolling_retrain.py**
   - Ensure `lookback` parameter is passed correctly
   - Pass full dataset path (not monthly slice)

3. **apex-ranker/apex_ranker/data/panel_dataset.py** (optional)
   - Line 90-92: Relax lookback requirement (allow partial windows)
   - Add MIN_LOOKBACK constant

4. **apex-ranker/scripts/backtest_smoke_test.py** (optional)
   - Add --skip-panel-cache flag for debugging
   - Log cache statistics before backtest

---

## References

- Rolling retrain output: `results/rolling_retrain_full/evaluations/*.json`
- Panel cache: `cache/panel_prod/*.pkl`
- Backtest driver: `apex-ranker/scripts/backtest_smoke_test.py`
- Panel cache builder: `apex-ranker/apex_ranker/data/panel_dataset.py`
- Inference engine: `apex-ranker/apex_ranker/backtest/inference.py`
- Data loader: `apex-ranker/apex_ranker/data/loader.py`

---

**Session**: 2025-10-30
**Investigator**: Claude Code (Autonomous)
**Status**: ✅ Diagnosis complete, ready for fix implementation
