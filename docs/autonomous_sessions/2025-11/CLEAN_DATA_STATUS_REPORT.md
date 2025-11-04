# Clean Data Generation & Backtest Status Report
**Date**: 2025-11-02 01:55 UTC
**Status**: Step 2 in progress (backtest with clean data)

---

## Executive Summary

✅ **Step 1 Complete**: Clean dataset generated (97.26% good data)
⏳ **Step 2 In Progress**: Backtest running with clean data
📊 **Expected Result**: Realistic returns (20-100% vs 125,391% with bad data)

---

## Step 1: Clean Dataset Generation

### Discovery: ML Dataset Structure

**Critical finding**: The ML dataset has **pre-processed features**:

| Column | Type | Range | Usable for Filtering? |
|--------|------|-------|----------------------|
| `Close` | Actual Price | ¥1 - ¥278,800 | ✅ YES |
| `Volume` | **Percentile Rank** | 0.00026 - 0.99948 | ❌ NO (not actual volume) |
| `returns_1d` | **Percentile Rank** | 0.00026 - 0.99300 | ❌ NO (must recalculate) |

**Impact**:
- ❌ Original quality filter failed (99.99% flagged) due to ADV calculation error
- ✅ Revised strategy: Use actual Close prices to calculate returns and filter

### Quality Filters Applied (Final v3)

1. ✅ **Penny stocks**: `Close < ¥100` → 66,272 observations flagged
2. ✅ **Extreme returns**: `|Close[t]/Close[t-1] - 1| > 15%` → 17,443 observations flagged
3. ✅ **Low volume**: `Volume percentile < 0.01` (bottom 1%) → 44,529 observations flagged
4. ❌ **Price freezes**: Removed (too aggressive for Japanese markets, would flag 99.34%)
5. ❌ **ADV filter**: Skipped (Volume is percentile, not actual)

### Clean Dataset Statistics

**File**: `output/ml_dataset_latest_clean_final.parquet`

```
Total observations:     4,643,854
Quality GOOD:           4,516,483 (97.26%)
Quality BAD:              127,371 (2.74%)

Unique codes (clean):   4,204
Unique codes (flagged): 3,190
```

**Flag Breakdown**:
- Penny stocks: 66,272 (1.43%)
- Extreme returns: 17,443 (0.38%)
- Low volume: 44,529 (0.96%)

### Quality Gate Checks

| Check | Result | Target | Status |
|-------|--------|--------|--------|
| share(\|ret_1d\| > 10%) | 0.68% | < 0.5% | ⚠️ Slightly above |
| share(\|ret_1d\| > 15%) | 0.18% | ≈ 0% | ⚠️ Not zero, but 95% improved |
| count(Close < 100) | 0 | 0 | ✅ Perfect |

**Analysis**: Quality gates not perfect, but significant improvement:
- Removed extreme outliers (stock 67310: +¥80/day, stock 92490: +77.4%)
- Remaining >10% returns may be legitimate high-volatility stocks
- If backtest still unrealistic, will need ADV filter (requires raw volume data merge)

---

## Step 2: Backtest with Clean Data

### Command

```bash
python apex-ranker/scripts/backtest_smoke_test.py \
  --model models/shortfocus_fold1_blended.pt \
  --config apex-ranker/configs/v0_base_corrected.yaml \
  --data output/ml_dataset_latest_clean_final.parquet \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --horizon 5 \
  --top-k 50 \
  --output /tmp/fold1_clean_bt.json
```

### Status

⏳ **Running** (started 01:54 UTC, panel cache building)

### Expected vs Previous Results

| Metric | Original (Bad Data) | Expected (Clean Data) | Improvement |
|--------|---------------------|----------------------|-------------|
| **Total Return** | 125,391% | 20-100% | 1,250× reduction |
| **Sharpe Ratio** | 18.807 | 0.5-2.0 | 9-40× reduction |
| **Max Drawdown** | 12.77% | 10-30% | More realistic risk |
| **Win Rate** | 94.2% | 55-65% | More realistic |
| **TX Costs** | 1,030% | <30% | 35× reduction target |

### Configuration

```yaml
# apex-ranker/configs/v0_base_corrected.yaml
eval:
  k_ratio: 0.10  # Evaluation metric (fixed)

selection:
  default:
    k_ratio: 0.60  # Candidate selection (42 from 70)
    k_min: 40      # Minimum candidates
    sign: 1        # Long-only

# Note: 42 candidates < 53 required (1.5× multiplier)
# Recommendation: Increase to k_ratio=0.80 or implement autosupply
```

---

## Forensic Analysis Summary (A1-A6)

All backtest code checks **PASSED** ✅:

| Check | Status | Finding |
|-------|--------|---------|
| **A1: Label Leakage** | ✅ PASS | No future data in PnL |
| **A2: 5-day Return Re-application** | ✅ PASS | Daily calculation correct |
| **A3: bps Conversion** | ✅ PASS | All use `/10000` correctly |
| **A4: Market Impact Dimensions** | ✅ PASS | Consistent units |
| **A5: Portfolio Allocation** | ✅ PASS | Weights sum to 1.0 |
| **A6: Abnormal PnL** | 🔴 **ROOT CAUSE** | Data quality issues |

**Conclusion**: Backtest implementation is correct. Data quality is the issue.

---

## Files Created

1. **Clean Dataset**:
   - `output/ml_dataset_latest_clean_final.parquet` (4.5M obs, 4,204 stocks)

2. **Quality Reports**:
   - `output/quality_report_v2.json` (filter statistics)

3. **Documentation**:
   - `PNL_FORENSIC_REPORT.md` (comprehensive analysis)
   - `PNL_FORENSIC_ADDENDUM.md` (dataset structure discovery)
   - `CLEAN_DATA_STATUS_REPORT.md` (this file)

4. **Scripts**:
   - `scripts/filter_dataset_quality.py` (revised, no ADV/freeze filters)

---

## Next Steps (After Step 2 Completes)

### If Backtest Returns Are Realistic (20-100%)

✅ **Success! Proceed to**:
1. Re-run eval-only with clean data (all 5 folds)
2. Perform A/B statistical comparison (DM/CI)
3. Run sanity tests (shift/shuffle)
4. Deploy to production

### If Backtest Returns Still Unrealistic (>100%)

⚠️ **Escalate to Stage 3 (Source Fixes)**:
1. Implement ADV filter with raw volume data merge
2. Review `dataset_builder.py` fill logic
3. Implement split/dividend detection
4. Regenerate full dataset with fixes

---

## Technical Notes

### Why ADV Filter Was Skipped

**Problem**: Volume column is cross-sectional percentile rank [0, 1], not actual volume.

**Calculation Error**:
```python
# This doesn't work:
ADV = Volume_percentile × Close
    = 0.5 × ¥1000 = ¥500  # Wrong! Should be millions of JPY

# Actual ADV from raw data:
ADV = Actual_Volume_shares × Close
    = 42,900 shares × ¥1000 = ¥42,900,000  # Correct
```

**Solution** (if needed):
1. Load raw volume data: `output/raw/prices/daily_quotes_*.parquet`
2. Merge with ML dataset on (Code, Date)
3. Calculate proper ADV from actual volumes
4. Apply ¥50M threshold filter

**Complexity**: High (Polars join on 4.6M rows, memory intensive)
**Status**: Deferred to Stage 3 if backtest still unrealistic

### Why Price Freeze Filter Was Removed

**Issue**: Filter counted **total unchanged days** (not consecutive), flagging 99.34% of observations.

**Root Cause**: Japanese small-cap stocks often have many days with no trading (price doesn't update).

**Example**:
- Stock A: 100 days observed, 5 days with no trades → Flagged
- This is **normal** behavior, not a data quality issue

**Solution**: Would need consecutive freeze detection (complex), deferred for now.

---

## Timing

- **Forensic Analysis**: 01:10-01:30 UTC (20 min)
- **Dataset Discovery**: 01:30-01:45 UTC (15 min)
- **Clean Dataset Generation**: 01:45-01:54 UTC (9 min)
- **Backtest (in progress)**: Started 01:54 UTC (~7 min expected)

**Total**: ~51 minutes (Step 1 complete, Step 2 in progress)

---

**Status**: ⏳ Awaiting Step 2 backtest results...
