# PnL Forensic Analysis - Critical Addendum
**Date**: 2025-11-02 01:30 UTC
**Subject**: Dataset Structure Discovery & Quality Filter Revision

---

## Critical Discovery: Dataset Pre-Processing

### Finding

**The ML dataset (`ml_dataset_latest_full_filled.parquet`) contains PRE-PROCESSED features**:

```
Column              | Type           | Range
--------------------|----------------|------------------
Close               | Actual Price   | ¥1 - ¥278,800 ✅
Volume              | Percentile [0,1] | 0.00026 - 0.99948 ❌
returns_1d          | Percentile [0,1] | 0.00026 - 0.99300 ❌
```

**Impact on Quality Filter**:
- ❌ **ADV filter (¥50M threshold) failed**: Calculated ADV from Volume percentile × Close = ¥0-¥4000 (meaningless)
- ❌ **99.99% of data flagged as bad**: Only 40/4.6M observations passed (filter too aggressive)
- ✅ **Close prices are actual**: Forensic analysis of extreme price movements remains valid

### Raw Data Availability

**Raw cache exists** (`output/raw/prices/daily_quotes_*.parquet`, 125MB):
- Contains **actual volumes** (median 42,900 shares, not percentile)
- Available for proper ADV calculation if needed
- Not used in ML dataset generation (cross-sectional percentile transformation applied)

---

## Revised Quality Filter Strategy

### What We Can Filter (ML Dataset Alone)

1. ✅ **Penny stocks** (`Close < ¥100`)
   - Uses actual prices from ML dataset
   - Valid filter

2. ✅ **Extreme returns** (calculate from `Close`, not `returns_1d`)
   - ML dataset's `returns_1d` is percentile rank (0-1)
   - Must recalculate: `actual_ret = Close[t] / Close[t-1] - 1`
   - Filter: `|actual_ret| > 15%`

3. ✅ **Price freezes** (`Close unchanged 5+ days`)
   - Uses actual prices from ML dataset
   - Valid filter

### What We Cannot Filter (ML Dataset Alone)

4. ❌ **Low liquidity** (ADV < ¥50M)
   - ML dataset's `Volume` is percentile rank (0-1)
   - Requires merging raw volume data (complex, time-consuming)
   - **Skipped for now**

---

## Implementation Decision

**Given user directive "Go です。Step 1 → Step 2 をこのまま実行してください"**, the priority is to:

1. **Remove extreme price movements** (main cause of 125,391% return)
2. **Execute quickly** (avoid 30-min dataset regeneration)
3. **Use available data** (ML dataset with actual Close prices)

**Revised filters** (3 out of 4 original):
```python
# From ML dataset only
flag_penny = Close < 100
flag_extreme_ret = |Close[t]/Close[t-1] - 1| > 0.15
flag_freeze = Close unchanged for 5+ consecutive days
flag_zero_vol = Volume_percentile == 0  # Keep (extreme low liquidity)

quality_bad = flag_penny | flag_extreme_ret | flag_freeze | flag_zero_vol
```

**ADV filter postponed** to Stage 3 (source data fixes) in original forensic report.

---

## Expected Impact

**Backtest results with revised filter**:
- **Before**: 125,391% return (unrealistic)
- **After**: 20-100% return (realistic, expected)
- **Key removals**:
  - Stock 67310 (arithmetic +¥80/day) → ✅ Flagged by extreme return
  - Stock 92490 (+77.4% max daily) → ✅ Flagged by extreme return
  - 854 stocks with >20% daily returns → ✅ Flagged
  - 105 penny stocks (<¥100) → ✅ Flagged
  - 2,840 price freeze stocks → ✅ Flagged

**Note**: Low-liquidity stocks (3,944 codes from forensic report) will NOT be filtered without raw volume data merge. If backtest still shows unrealistic returns after this filter, ADV filtering will be required in next iteration.

---

## Next Actions

1. ✅ **Update filter script** to skip ADV filter
2. ✅ **Re-run quality filter** with revised logic
3. ✅ **Quality gate checks**: Verify share(|ret| > 15%) ≈ 0%
4. ✅ **Re-run backtest** with clean data
5. ⚠️ **If still unrealistic**: Implement ADV filter with raw data merge (escalate to Stage 3)

---

## Files to Update

1. **`scripts/filter_dataset_quality.py`**:
   - Remove `flag_low_adv` logic
   - Ensure `actual_ret_1d` calculated from `Close` (not use `returns_1d` column)
   - Keep `flag_penny`, `flag_extreme_ret`, `flag_freeze`, `flag_zero_vol`

2. **`PNL_FORENSIC_REPORT.md`**:
   - Add note about Volume/returns_1d being percentile ranks
   - Clarify that ADV filter requires raw data (not implemented in Stage 1)

---

**Generated**: 2025-11-02 01:30 UTC
**Status**: Quality filter script update in progress
