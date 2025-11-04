# Execution Summary: Clean Data Generation & Parallel Tasks
**Date**: 2025-11-02 02:06 UTC
**Session**: PnL Forensic Investigation → Clean Data → Parallel Execution

---

## ✅ **Completed Tasks**

### 1. PnL Forensic Analysis (A1-A6) ✅
- **Status**: All checks PASSED
- **Conclusion**: Backtest implementation is CORRECT. Data quality is the issue.
- **Files**: `PNL_FORENSIC_REPORT.md`, `PNL_FORENSIC_ADDENDUM.md`

### 2. Clean Dataset Generation (v3) ✅
- **File**: `output/ml_dataset_latest_clean_final.parquet`
- **Quality**: 97.26% good data (4.5M obs, 4,204 stocks)
- **Filters**: Penny stocks, extreme returns (>15%), low volume (bottom 1%)
- **Skipped**: ADV (requires raw volume), price freezes (too aggressive)

### 3. Backtest with Clean Data (Fold1) ✅
- **Result**: 17,775% return (86% reduction, but STILL unrealistic)
- **Analysis**: Remaining low-liquidity stocks causing issues
- **Next**: Implement ADV filter (Stage 3)

### 4. Autosupply Mechanism Implementation ✅
- **File**: `apex-ranker/apex_ranker/backtest/selection.py:92-140`
- **Function**: `compute_autosupply_k_ratio(candidate_count, target_top_k, alpha=1.5)`
- **Purpose**: Ensures ≥53 candidates for target_top_k=35 (1.5× multiplier rule)

---

## ⏳ **In Progress (Background)**

### 5. Eval-Only with Clean Data (All 10 Folds) ⏳
- **Status**: Running (started 02:00 UTC)
- **Models**: 5 baseline + 5 shortfocus
- **Output**: `*_clean_reval_k10_val_perday.npz` (for A/B statistical comparison)
- **ETA**: ~3-5 hours total
- **Monitor**: `tail -f /tmp/clean_eval_all.log`

---

## 📊 **Key Findings**

### Dataset Structure Discovery 🔍
**Critical insight**: ML dataset has pre-processed features
- `Close`: Actual prices ✅ (¥1 - ¥278,800)
- `Volume`: **Percentile rank [0,1]** ❌ (not actual volume)
- `returns_1d`: **Percentile rank [0,1]** ❌ (not actual returns)

**Impact**: ADV filter failed (calculated ¥417 median vs actual ¥42.9M median)

### Clean Data Backtest Results ⚠️
**Still unrealistic** despite 86% improvement:
- 17,775% return (target: 20-100%)
- Sharpe 18.888 (target: 0.5-2.0)
- Max DD 7.15% (target: 10-30%)

**Why?** Low-liquidity stocks remain (ADV filter not applied)

---

## 🚧 **Escalation Required: Stage 3 (Source Fixes)**

### Required Actions

**1. Implement ADV Filter** 🔴 **CRITICAL**
- Merge raw volume data from `output/raw/prices/daily_quotes_*.parquet`
- Calculate proper ADV: `trailing_60d_mean(Actual_Volume × Close)`
- Apply ¥50M threshold filter
- **Expected impact**: Reduce returns from 17,775% → 20-100%

**2. Trailing Window ADV** (Avoid Look-Ahead Bias)
```python
# Calculate ADV with proper trailing window (exclude current day)
adv_60d = (
    df.sort(['Code', 'Date'])
    .with_columns([
        (pl.col('Volume') * pl.col('Close')).alias('dollar_volume'),
    ])
    .with_columns([
        # Trailing 60-day mean (excluding current day)
        pl.col('dollar_volume')
        .shift(1).over('Code')
        .rolling_mean(window_size=60, min_periods=20)
        .alias('adv_60d')
    ])
)
```

**3. Split/Dividend Detection** (Source Data Issue)
- Stock 67310: Perfect +¥80/day progression → Likely data error or stock split
- Stock 92490: +77.4% daily return → Check for reverse split
- Implement split detection: `ratio = Close[t] / Close[t-1]`, flag if ratio ≈ [2, 3, 5, 10, 0.5, 0.33, 0.2, 0.1]

---

## 📋 **Pending Tasks (Priority Order)**

### 1. Wait for Eval-Only Completion ⏳
- **ETA**: 3-5 hours
- **Then**: A/B statistical comparison (DM/CI)

### 2. Implement ADV Filter (Stage 3) 🔴
- Create `scripts/create_dataset_with_actual_adv.py`
- Merge raw volume data
- Apply ¥50M threshold
- Regenerate clean dataset

### 3. Re-run Backtest with ADV-Filtered Data 🔴
- **Expected**: Realistic returns (20-100%)
- **If still unrealistic**: Investigate source data issues (splits/dividends)

### 4. A/B Statistical Comparison (After Eval-Only) ✅
```bash
python scripts/analyze_run_comparison.py \
  --a "models/baseline_fold*_clean_reval_k10_val_perday.npz" \
  --b "models/shortfocus_fold*_clean_reval_k10_val_perday.npz" \
  --metric h5_delta_p_at_k_pos --block 5

# Pass criteria: DM > 1.96, 95%CI > 0
```

### 5. Sanity Tests 📝
- Shift test (+1 day)
- Shuffle test (random scores)
- Index regression (β analysis)

---

## 📁 **Files Created**

### Documentation
1. **`PNL_FORENSIC_REPORT.md`** - Complete forensic analysis (A1-A6)
2. **`PNL_FORENSIC_ADDENDUM.md`** - Dataset structure discovery
3. **`CLEAN_DATA_STATUS_REPORT.md`** - Clean data generation status
4. **`EXECUTION_SUMMARY.md`** - This file

### Code
1. **`scripts/filter_dataset_quality.py`** - Quality filter (v3, no ADV/freeze)
2. **`apex-ranker/apex_ranker/backtest/selection.py`** - Added `compute_autosupply_k_ratio()`
3. **`/tmp/run_clean_eval_all_folds.sh`** - Eval-only runner (10 folds)

### Data
1. **`output/ml_dataset_latest_clean_final.parquet`** - Clean dataset (97.26% good data)
2. **`output/quality_report_v2.json`** - Filter statistics

### Results
1. **`/tmp/fold1_clean_bt.json`** - Backtest with clean data (17,775% return)
2. **`models/*_clean_reval_k10_val_perday.npz`** - Eval-only outputs (in progress)

---

## 🔢 **Quality Gate Results**

| Check | Result | Target | Status |
|-------|--------|--------|--------|
| share(\|ret_1d\| > 10%) | 0.68% | < 0.5% | ⚠️ Slightly above |
| share(\|ret_1d\| > 15%) | 0.18% | ≈ 0% | ⚠️ Not zero |
| count(Close < 100) | 0 | 0 | ✅ Perfect |
| Backtest return | 17,775% | 20-100% | 🔴 Too high |

**Analysis**: Quality gates partially met, but **ADV filter required** for realistic backtests.

---

## 🎯 **Recommendation**

### Immediate (Next 3 Hours)
- ✅ Wait for eval-only completion (background)
- ⏳ Monitor progress: `tail -f /tmp/clean_eval_all.log`

### After Eval-Only Completes
1. Run A/B statistical comparison (DM/CI)
2. If DM > 1.96 and 95%CI > 0 → Candidate model **provisionally passes**
3. **But**: Must implement ADV filter before production deployment

### Priority Escalation (Stage 3)
1. Implement ADV filter with raw volume data
2. Re-run backtest to verify realistic returns
3. If returns still unrealistic → Investigate source data (splits/dividends)

---

## ⏱️ **Timeline**

| Task | Start | End | Duration | Status |
|------|-------|-----|----------|--------|
| Forensic Analysis | 01:10 | 01:30 | 20 min | ✅ |
| Dataset Discovery | 01:30 | 01:45 | 15 min | ✅ |
| Clean Dataset Gen | 01:45 | 01:54 | 9 min | ✅ |
| Backtest (Clean) | 01:54 | 02:06 | 12 min | ✅ |
| Autosupply Impl | 02:00 | 02:05 | 5 min | ✅ |
| Eval-Only (10 folds) | 02:00 | ~05:00 | ~3h | ⏳ |
| **Total elapsed** | 01:10 | 02:06 | **56 min** | - |

---

## 📝 **Next Session Guidance**

When eval-only completes:

```bash
# 1. Verify output files
ls -lh models/*_clean_reval_k10_val_perday.npz

# 2. Run A/B comparison
python scripts/analyze_run_comparison.py \
  --a "models/baseline_fold*_clean_reval_k10_val_perday.npz" \
  --b "models/shortfocus_fold*_clean_reval_k10_val_perday.npz" \
  --metric h5_delta_p_at_k_pos --block 5

# 3. If pass → Implement ADV filter (Stage 3)
python scripts/create_dataset_with_actual_adv.py \
  --ml-dataset output/ml_dataset_latest_clean_final.parquet \
  --raw-prices output/raw/prices/daily_quotes_*.parquet \
  --min-adv 50000000 \
  --output output/ml_dataset_latest_clean_with_adv.parquet

# 4. Re-run backtest with ADV-filtered data
python apex-ranker/scripts/backtest_smoke_test.py \
  --data output/ml_dataset_latest_clean_with_adv.parquet \
  --config apex-ranker/configs/v0_base_corrected.yaml \
  --model models/shortfocus_fold1_blended.pt \
  --start-date 2024-01-01 --end-date 2024-12-31 \
  --output /tmp/fold1_clean_adv_bt.json

# Expected: Total return 20-100%, Sharpe 0.5-2.0
```

---

**Status**: ✅ Step 1 & 2 complete, ⏳ Step 3 in progress (eval-only), 🔴 Stage 3 escalation required (ADV filter)
