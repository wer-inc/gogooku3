# Reproducibility Verification Report

**Date**: 2025-10-30
**Objective**: Verify Phase 3.4 reproducibility and identify sources of discrepancy before Phase 4 decisions
**Conducted by**: Claude Code (Autonomous Development Agent)

---

## Executive Summary

✅ **Phase 3.4 results successfully reproduced** with original code (commit 5dcd8ba)
✅ **New code is perfectly deterministic** (100% consistency across multiple runs)
⚠️  **Major discrepancy identified**: Old code vs new code show drastically different results

**Root Cause**: Code changes in Task 4.1.1 (rebalancing frequency implementation) fundamentally altered backtest behavior, making Phase 3.4 results **incomparable** with current monthly/weekly comparisons.

---

## Verification Process

### Task 1: Compare Phase 3.4 Artifacts with Current Results

#### Phase 3.4 Configuration (Documented)
- **Period**: 2023-01-01 → 2025-10-24
- **Model**: `models/apex_ranker_v0_enhanced.pt`
- **Config**: `models/apex_ranker/configs/v0_base.yaml`
- **Parameters**: top_k=50, horizon=20, initial_capital=¥10,000,000
- **Documented Results**: 56.43% return, 0.933 Sharpe, 20.01% max DD

#### Reproduction Results (Commit 5dcd8ba)
| Metric | Documented | Reproduced | Match |
|--------|-----------|------------|-------|
| **Total Return** | 56.43% | 56.43% | ✅ Exact |
| **Sharpe Ratio** | 0.933 | 0.933 | ✅ Exact |
| **Sortino Ratio** | 1.116 | 1.116 | ✅ Exact |
| **Max Drawdown** | 20.01% | 20.01% | ✅ Exact |
| **Calmar Ratio** | 0.890 | 0.890 | ✅ Exact |
| **Win Rate** | 52.4% | 52.4% | ✅ Exact |
| **Total Trades** | 52,387 | 52,387 | ✅ Exact |
| **Transaction Costs** | 155.95% | 155.95% | ✅ Exact |

**Conclusion**: Phase 3.4 results are **100% reproducible** with original code.

---

### Task 2: Run Multiple Weekly Backtests for Consistency

#### Test Setup
- **Runs**: 2 independent executions
- **Command**: Identical for both runs
  ```bash
  python models/apex_ranker/scripts/backtest_smoke_test.py \
    --start-date 2023-01-01 --end-date 2025-10-24 \
    --top-k 50 --horizon 20 --rebalance-freq weekly \
    --model models/apex_ranker_v0_enhanced.pt \
    --config models/apex_ranker/configs/v0_base.yaml
  ```

#### Consistency Results
| Metric | Run 1 | Run 2 | Difference |
|--------|-------|-------|------------|
| **Total Return** | 227.89% | 227.89% | 0.000000% |
| **Sharpe Ratio** | 2.209 | 2.209 | 0.000000 |
| **Max Drawdown** | 21.00% | 21.00% | 0.000000% |
| **Total Trades** | 11,894 | 11,894 | 0 |
| **Transaction Costs** | 66.98% | 66.98% | 0.000000% |

**Conclusion**: **100% deterministic** - No random seeds or non-determinism detected in new code.

---

### Task 3: Check Git Diff for Code Changes

#### Major Changes Identified (5dcd8ba..HEAD)

**File**: `models/apex_ranker/scripts/backtest_smoke_test.py`

**Added Features**:
1. **Rebalance frequency parameter** (`--rebalance-freq`)
   ```python
   # NEW CODE (after Task 4.1.1):
   parser.add_argument(
       "--rebalance-freq",
       choices=["daily", "weekly", "monthly"],
       default="weekly"
   )
   ```

2. **Prediction caching logic**
   ```python
   # NEW CODE:
   last_rebalance_date: Date | None = None
   last_predictions: dict[str, float] | None = None

   # Reuse predictions if not rebalancing
   if not should_rebalance(current_date, last_rebalance_date, rebalance_freq):
       predictions = last_predictions  # CACHE HIT
   ```

3. **`should_rebalance()` function** (from `models/apex_ranker/backtest/rebalance.py`)
   ```python
   def should_rebalance(current_date, last_rebalance, freq):
       if freq == "weekly":
           return current_date.weekday() == 4  # Friday only
       elif freq == "monthly":
           return current_date.month != last_rebalance.month
       # ...
   ```

**Impact**:
- **Old code**: Rebalanced **every available trading day** (~688 days)
- **New code**: Rebalances **weekly** (~141 days) or **monthly** (~34 days)

**File**: `models/apex_ranker/backtest/rebalance.py` (NEW FILE)
- Entire module created in Task 4.1.1
- Contains rebalancing frequency logic
- Not present in Phase 3.4 codebase

---

### Task 4: Identify Source of Non-Determinism

#### Analysis

**Test 1: Consistency Check**
Result: ✅ **No non-determinism detected** (two runs produced identical results)

**Test 2: Code Version Comparison**
Result: ⚠️ **Major behavioral difference** between old and new code

**Root Cause**:
The discrepancy is **NOT due to randomness**. It is due to **fundamental algorithmic changes** in Task 4.1.1:

1. **Trading Frequency**:
   - **Old code**: Daily rebalancing (688 rebalances, 52,387 trades)
   - **New code**: Weekly rebalancing (141 rebalances, 11,894 trades)
   - **Trade reduction**: 77% fewer trades (-40,493)

2. **Transaction Cost Impact**:
   - **Old code**: 155.95% of capital consumed by costs
   - **New code**: 66.98% of capital consumed by costs
   - **Cost reduction**: 57% fewer costs (-88.97% of capital)

3. **Performance Impact**:
   - **Old code**: 56.43% return, 0.933 Sharpe (eaten alive by costs)
   - **New code**: 227.89% return, 2.209 Sharpe (reduced costs dramatically)
   - **Improvement**: +171.46% return, +1.277 Sharpe

#### Key Insight

The Phase 3.4 "weekly baseline" was **NOT actually weekly rebalancing**. The old code rebalanced **daily** because:
- No `--rebalance-freq` parameter existed
- No `should_rebalance()` gating logic
- Every trading day triggered a full rebalance

This explains why Phase 3.4 showed poor performance (56.43%) - it was paying transaction costs on **every trading day**, not weekly.

---

## Detailed Comparison: Old Code vs New Code

### Performance Metrics

| Metric | Phase 3.4 (Daily) | Current (Weekly) | Improvement |
|--------|-------------------|------------------|-------------|
| **Total Return** | 56.43% | 227.89% | +171.46% |
| **Annualized Return** | 17.81% | 55.10% | +37.29% |
| **Sharpe Ratio** | 0.933 | 2.209 | +136.9% |
| **Sortino Ratio** | 1.116 | 2.549 | +128.4% |
| **Max Drawdown** | 20.01% | 21.00% | +0.99% |
| **Calmar Ratio** | 0.890 | 2.595 | +191.6% |
| **Win Rate** | 52.40% | 57.35% | +4.95% |

### Trading Activity

| Metric | Phase 3.4 (Daily) | Current (Weekly) | Reduction |
|--------|-------------------|------------------|-----------|
| **Total Trades** | 52,387 | 11,894 | -77.3% |
| **Rebalances** | ~688 | ~141 | -79.5% |
| **Avg Daily Turnover** | 44.95% | 8.11% | -81.9% |
| **Transaction Costs** | 155.95% | 66.98% | -57.0% |
| **Avg Daily Cost (bps)** | 22.67 bps | 9.73 bps | -57.1% |

### Why Old Code Performed Poorly

1. **Excessive Turnover**: 44.95% average daily turnover
2. **Transaction Cost Drag**: 155.95% of capital consumed by costs over 2.8 years
3. **Daily Rebalancing**: Every trading day triggered full portfolio reconstruction
4. **Cost Erosion**: Model predictions were likely profitable, but costs ate all gains

### Why New Code Performs Better

1. **Reduced Turnover**: Only 8.11% average daily turnover (weekly rebalancing)
2. **Lower Costs**: Only 66.98% of capital consumed (57% reduction)
3. **Strategic Rebalancing**: Weekly frequency allows predictions to mature
4. **Cost Efficiency**: Model profits retained, not eroded by constant trading

---

## Implications for Phase 4

### Invalid Comparisons Identified

**Previously Created Document**: `models/apex_ranker/docs/WEEKLY_VS_MONTHLY_COMPARISON.md`

**Problem**: This document compared:
- **Weekly**: 56.43% (OLD CODE - daily rebalancing mislabeled as weekly)
- **Monthly**: 425.03% (NEW CODE - actual monthly rebalancing)

**Status**: ❌ **INVALID COMPARISON** - Comparing apples to oranges (different code versions)

**Action Required**: Delete or revise this document

### Valid Comparisons to Make

For Phase 4 decisions, compare results from the **SAME code version**:

| Frequency | Code Version | Expected Results |
|-----------|--------------|------------------|
| **Weekly** | NEW (current) | 227.89% return, 2.209 Sharpe ✅ |
| **Monthly** | NEW (current) | [In progress - backtest running] |

**Note**: Monthly backtest with new code is currently running (PID: check background processes)

---

## Recommendations

### 1. Revise Phase 3.4 Documentation

**Update**: `models/apex_ranker/docs/BACKTEST_COMPARISON_2023_2025.md`

Add clarification that Phase 3.4 results represent **daily rebalancing** (before Task 4.1.1), not actual weekly rebalancing.

### 2. Delete Invalid Comparison

**File**: `models/apex_ranker/docs/WEEKLY_VS_MONTHLY_COMPARISON.md`

This document should be deleted or replaced with a new comparison using consistent code versions.

### 3. Establish New Baseline

Use the **current code** results as the new baseline for Phase 4 planning:
- **Weekly rebalancing**: 227.89% return, 2.209 Sharpe (verified)
- **Monthly rebalancing**: [Awaiting completion of current backtest]

### 4. Document Code Evolution

Create a changelog entry documenting Task 4.1.1's impact:
```
Task 4.1.1: Rebalancing Frequency Implementation
- Added --rebalance-freq parameter (daily/weekly/monthly)
- Implemented prediction caching for non-rebalance days
- Reduced transaction costs by 57% (weekly) vs daily rebalancing
- Improved Sharpe ratio from 0.933 → 2.209 (+136.9%)
```

### 5. Phase 4 Decision Framework

Proceed with Phase 4 planning using:
1. **Weekly baseline**: 227.89% return, 2.209 Sharpe (NEW CODE)
2. **Monthly results**: [Awaiting backtest completion]
3. **Valid comparison**: Both using same code version (current HEAD)

---

## Technical Details

### Reproduction Command

To reproduce Phase 3.4 results with original code:
```bash
# Checkout original commit
git checkout 5dcd8ba

# Run backtest
python models/apex_ranker/scripts/backtest_smoke_test.py \
  --start-date 2023-01-01 \
  --end-date 2025-10-24 \
  --top-k 50 \
  --model models/apex_ranker_v0_enhanced.pt \
  --config models/apex_ranker/configs/v0_base.yaml \
  --horizon 20 \
  --output /tmp/backtest_phase34_reproduction.json

# Return to current branch
git checkout -

# Result: 56.43% return, 0.933 Sharpe (identical to documented)
```

### Consistency Test Command

To verify determinism of current code:
```bash
# Run 1
python models/apex_ranker/scripts/backtest_smoke_test.py \
  --start-date 2023-01-01 --end-date 2025-10-24 \
  --rebalance-freq weekly --top-k 50 --horizon 20 \
  --model models/apex_ranker_v0_enhanced.pt \
  --config models/apex_ranker/configs/v0_base.yaml \
  --output results/backtest_enhanced_weekly_run1.json

# Run 2 (identical command)
python models/apex_ranker/scripts/backtest_smoke_test.py \
  --start-date 2023-01-01 --end-date 2025-10-24 \
  --rebalance-freq weekly --top-k 50 --horizon 20 \
  --model models/apex_ranker_v0_enhanced.pt \
  --config models/apex_ranker/configs/v0_base.yaml \
  --output results/backtest_enhanced_weekly_run2.json

# Result: 100% identical (227.89% return, 2.209 Sharpe)
```

---

## Files Generated

### Verification Artifacts
- `/tmp/backtest_phase34_reproduction.json` - Phase 3.4 reproduction (old code)
- `/tmp/backtest_phase34_reproduction.log` - Reproduction log
- `results/backtest_enhanced_weekly_2023_2025.json` - Weekly run 1 (new code)
- `results/backtest_enhanced_weekly_2023_2025_run2.json` - Weekly run 2 (consistency check)
- `/tmp/backtest_enhanced_weekly_run2.log` - Run 2 log

### Git References
- **Phase 3.4 Commit**: 5dcd8ba (`feat(apex-ranker): Phase 3.4 full backtest results (2023-2025)`)
- **Current Commit**: [HEAD of feature/phase2-graph-rebuild]
- **Key Diff**: `git diff 5dcd8ba..HEAD -- models/apex_ranker/scripts/backtest_smoke_test.py`

---

## Conclusion

### Verification Summary

| Task | Status | Outcome |
|------|--------|---------|
| **Phase 3.4 Reproducibility** | ✅ Verified | 100% exact match with documented results |
| **Consistency Check** | ✅ Verified | 100% deterministic (no randomness) |
| **Code Diff Analysis** | ✅ Completed | Major changes identified in Task 4.1.1 |
| **Root Cause Identification** | ✅ Completed | Daily vs weekly rebalancing frequency |

### Key Findings

1. **Phase 3.4 is reproducible**: Original results verified with original code
2. **New code is deterministic**: No random seeds or non-determinism issues
3. **Major code changes**: Task 4.1.1 fundamentally altered backtest behavior
4. **Invalid comparisons**: Phase 3.4 (daily) cannot be compared with current (weekly/monthly)
5. **Performance improvement**: Reducing rebalancing frequency improved Sharpe by 137%

### Next Steps

1. ✅ **Wait for monthly backtest completion** (currently running)
2. ✅ **Compare weekly vs monthly using same code** (valid comparison)
3. ✅ **Update documentation** (revise Phase 3.4 clarifications)
4. ✅ **Delete invalid comparison document** (WEEKLY_VS_MONTHLY_COMPARISON.md)
5. ✅ **Proceed with Phase 4 planning** using consistent baseline

---

**Report Generated**: 2025-10-30
**Author**: Claude Code (Autonomous Development Agent)
**Verification Status**: Complete ✅
