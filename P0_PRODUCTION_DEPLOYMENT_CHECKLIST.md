# P0 Production Deployment Checklist

## Executive Summary

**Test Phase Complete**: P0-1 through P0-3 ✅
**Verdict**: BASELINE configuration (EI OFF, A.3 OFF, A.4 OFF) approved for production
**Performance**: Sharpe 1.439, Return 44.85%, MaxDD 16.40% (2024-01-01 to 2025-10-31)

---

## Test Results Summary

### Pattern B Confirmed: A.3 Hysteresis is the Culprit

| Test | Configuration | Sharpe | Return | MaxDD | vs BASE | Status |
|------|---------------|--------|--------|-------|---------|--------|
| **T0 (BASELINE)** | EI OFF, A.3 OFF, A.4 OFF | 1.439 | 44.85% | 16.40% | — | ✅ **PRODUCTION** |
| **T1 (EI only)** | EI ON, A.3 OFF, A.4 OFF | 1.439 | 44.85% | 16.40% | 0.0% | ✅ PASS |
| **T2 (EI+A.3)** | EI ON, A.3 ON (exit_k=60) | 1.158 | 33.62% | 16.63% | -19.5% | ❌ NO-GO |

**Key Insights**:
1. ✅ **EI module has ZERO impact** (T0 = T1, perfect match)
2. ❌ **A.3 Hysteresis causes -19.5% Sharpe degradation** (T1 → T2)
3. ✅ **Supply stability confirmed** (k_min=53 autosupply working correctly)

---

## Production Configuration

### Model Specifications

```yaml
Model: models/apex_ranker_v0_enhanced.pt
Config: apex-ranker/configs/v0_base_89_cleanADV.yaml
Dataset: output/ml_dataset_latest_clean_with_adv.parquet
Features: 89 (Clean+ADV dataset with actual_adv)
```

### Trading Parameters

```yaml
Rebalance Frequency: monthly
Horizon: 20 days
Top-K: 35 holdings
Min Weight: 2.0% (0.0200)
Turnover Limit: 35% (0.35)
Cost Penalty: 1.00
```

### Selection Gate

```yaml
k_ratio: 0.6 (percentile-based candidate filter)
k_min: 53 (autosupply: ceil(1.5 × 35) ensures supply stability)
gate_ratio: 0.6
Fallback: Enabled (deterministic Top-53 when percentile < k_min)
```

### Enhanced Inference (EI) Module

```yaml
--use-enhanced-inference: DISABLED (no performance benefit confirmed)
--ei-hysteresis-entry-k: N/A
--ei-hysteresis-exit-k: N/A
--ei-neutralize-risk: DISABLED (A.4 showed NO-GO results)
```

### Features Mode

```yaml
--features-mode: fill-zero
```

---

## Pre-Deployment Checklist

### Critical Verifications

- [ ] **1. Model Checkpoint Integrity**
  ```bash
  # Verify checkpoint exists and matches expected hash
  ls -lh models/apex_ranker_v0_enhanced.pt
  md5sum models/apex_ranker_v0_enhanced.pt
  ```

- [ ] **2. Config Integrity**
  ```bash
  # Verify config file and feature count
  grep -E "in_features|patch_multiplier|add_csz" apex-ranker/configs/v0_base_89_cleanADV.yaml
  # Expected: in_features: 89, patch_multiplier: 1 (or auto), add_csz: false
  ```

- [ ] **3. Dataset Integrity**
  ```bash
  # Verify dataset exists and check row count
  python -c "import polars as pl; df = pl.read_parquet('output/ml_dataset_latest_clean_with_adv.parquet'); print(f'Rows: {len(df):,}, Columns: {len(df.columns)}, Date range: {df[\"Date\"].min()} to {df[\"Date\"].max()}')"
  # Expected: ~2.3M rows, 90+ columns, dates through 2025-10-24
  ```

- [ ] **4. Feature Compatibility**
  ```bash
  # Verify feature names match between dataset and config
  python scripts/diagnose_feature_health.py \
    --config apex-ranker/configs/v0_base_89_cleanADV.yaml \
    --data output/ml_dataset_latest_clean_with_adv.parquet
  ```

- [ ] **5. Clear Python Bytecode Cache**
  ```bash
  # Remove all __pycache__ to prevent stale code issues
  find apex-ranker -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
  find apex-ranker -name "*.pyc" -delete
  ```

- [ ] **6. Dependency Verification**
  ```bash
  # Verify critical packages
  python -c "import torch; import polars; import cvxpy; print('✅ All dependencies OK')"
  ```

### Performance Validation

- [ ] **7. Smoke Test (5 days)**
  ```bash
  python apex-ranker/scripts/backtest_smoke_test.py \
    --model models/apex_ranker_v0_enhanced.pt \
    --config apex-ranker/configs/v0_base_89_cleanADV.yaml \
    --data output/ml_dataset_latest_clean_with_adv.parquet \
    --start-date 2025-09-01 --end-date 2025-09-05 \
    --rebalance-freq monthly --horizon 20 --top-k 35 \
    --features-mode fill-zero \
    --output /tmp/production_smoke_test.json

  # Expected: No errors, JSON output created
  ```

- [ ] **8. Reproducibility Test (Top-5 stocks)**
  ```bash
  # Run twice and verify Top-5 stocks match
  # First run
  python apex-ranker/scripts/backtest_smoke_test.py \
    --model models/apex_ranker_v0_enhanced.pt \
    --config apex-ranker/configs/v0_base_89_cleanADV.yaml \
    --data output/ml_dataset_latest_clean_with_adv.parquet \
    --start-date 2024-01-01 --end-date 2024-01-31 \
    --rebalance-freq monthly --horizon 20 --top-k 35 \
    --features-mode fill-zero \
    --output /tmp/repro_test_1.json

  # Second run
  python apex-ranker/scripts/backtest_smoke_test.py \
    --model models/apex_ranker_v0_enhanced.pt \
    --config apex-ranker/configs/v0_base_89_cleanADV.yaml \
    --data output/ml_dataset_latest_clean_with_adv.parquet \
    --start-date 2024-01-01 --end-date 2024-01-31 \
    --rebalance-freq monthly --horizon 20 --top-k 35 \
    --features-mode fill-zero \
    --output /tmp/repro_test_2.json

  # Compare Top-5 holdings (should be identical)
  python - <<'PY'
  import json
  with open('/tmp/repro_test_1.json') as f: h1 = json.load(f)['daily'][0]['holdings'][:5]
  with open('/tmp/repro_test_2.json') as f: h2 = json.load(f)['daily'][0]['holdings'][:5]
  assert h1 == h2, f"Holdings mismatch!\nRun1: {h1}\nRun2: {h2}"
  print(f"✅ Reproducibility confirmed: {h1}")
  PY
  ```

- [ ] **9. Cost Model Validation**
  ```bash
  # Verify transaction costs are reasonable (3-5% of capital)
  grep "Total transaction costs" /tmp/production_smoke_test.json
  # Expected: ~¥370K (3.7%) for 22 monthly rebalances over ~22 months
  ```

### Supply Stability Validation

- [ ] **10. Supply Metrics Check**
  ```bash
  # Verify candidate_kept = 53 consistently
  grep "candidate_kept=" /tmp/production_smoke_test.json | awk -F'=' '{print $NF}' | sort | uniq
  # Expected: 53 (all dates)

  # Verify fallback mechanism
  grep "fallback=" /tmp/production_smoke_test.json | awk -F'=' '{print $NF}' | sort | uniq -c
  # Expected: All dates show fallback=1 (k_min=53 autosupply working)
  ```

---

## Production Command

**Final Production Backtest Command** (for record-keeping):

```bash
python apex-ranker/scripts/backtest_smoke_test.py \
  --model models/apex_ranker_v0_enhanced.pt \
  --config apex-ranker/configs/v0_base_89_cleanADV.yaml \
  --data output/ml_dataset_latest_clean_with_adv.parquet \
  --start-date 2024-01-01 \
  --end-date 2025-10-31 \
  --rebalance-freq monthly \
  --horizon 20 \
  --top-k 35 \
  --features-mode fill-zero \
  --output results/BASELINE_FINAL.json \
  > /tmp/production_baseline.log 2>&1
```

**Expected Results** (for validation):
```
Sharpe ratio: 1.439
Total return: 44.85%
Annualized return: 23.52%
Max drawdown: 16.40%
Sortino ratio: 1.810
Calmar ratio: 1.434
Win rate: 56.9%
Total transaction costs: ¥370,429 (3.70% of capital)
Rebalances: 22
```

---

## Deployment Metadata

**Freeze Date**: 2025-11-03
**Test Period**: 2024-01-01 to 2025-10-31 (22 months, 442 trading days)
**Model Version**: v0_enhanced (89 features, Clean+ADV dataset)
**Config Hash**: `md5sum apex-ranker/configs/v0_base_89_cleanADV.yaml`
**Dataset Hash**: `md5sum output/ml_dataset_latest_clean_with_adv.parquet`

---

## Known Issues & Mitigation

### 1. A.3 Hysteresis Degradation
**Issue**: A.3 hysteresis (entry≠exit thresholds) causes -19.5% Sharpe degradation
**Mitigation**: Disabled in production (entry_k = exit_k = default, no explicit flags)
**Status**: ✅ Resolved (confirmed T1 test: EI with A.3 OFF = BASELINE)

### 2. A.4 Risk Neutralization Ineffective
**Issue**: A.4 risk neutralization showed NO-GO results in previous tests
**Mitigation**: Disabled in production (no `--ei-neutralize-risk` flag)
**Status**: ✅ Resolved

### 3. Supply Fallback 100%
**Issue**: `fallback=1` on all rebalance dates (may appear as warning)
**Root Cause**: `k_min=53` autosupply safeguard > percentile filter output (~42)
**Mitigation**: This is EXPECTED and CORRECT behavior (supply stability guarantee)
**Status**: ✅ By design (not a bug)

---

## Post-Deployment Monitoring

### Daily Checks

1. **Prediction Stability**
   - Top-10 holdings should be relatively stable month-over-month
   - Check for sudden 100% turnover (indicates model instability)

2. **Cost Monitoring**
   - Monthly transaction costs should be 10-20 bps (0.10-0.20%)
   - Annual costs should be <4% of capital

3. **Supply Metrics**
   - `candidate_kept` should remain at 53
   - `fallback=1` is expected (k_min autosupply)

### Monthly Reviews

1. **Performance Attribution**
   - Compare realized returns vs predicted scores
   - Monitor Sharpe ratio (target: >1.4)
   - Track max drawdown (alert if >20%)

2. **Feature Health**
   - Check for missing features or NaN spikes
   - Monitor feature distribution shifts

### Quarterly Re-validation

1. **Backtest Update**
   - Re-run backtest with latest 6 months of data
   - Verify Sharpe ratio remains >1.4

2. **Model Drift Detection**
   - Compare prediction distribution vs training period
   - Check IC/RankIC degradation

---

## Rollback Procedure

If production performance degrades (Sharpe <1.0 for 3+ months):

1. **Immediate Actions**
   ```bash
   # Stop production deployment
   # Revert to last known good configuration
   git checkout <last-good-commit>
   ```

2. **Root Cause Analysis**
   - Check for dataset issues (missing features, data quality)
   - Verify model checkpoint integrity
   - Review recent market regime changes

3. **Re-validation**
   - Re-run P0-3 supply stability check
   - Smoke test with recent data
   - Compare with historical BASELINE results

---

## Approval Sign-off

- [ ] **Technical Lead**: _______________  Date: _____
- [ ] **Risk Manager**: _______________  Date: _____
- [ ] **Production Engineer**: _______________  Date: _____

---

## Change Log

| Date | Version | Change | Approved By |
|------|---------|--------|-------------|
| 2025-11-03 | 1.0.0 | Initial BASELINE deployment checklist | — |

---

## References

- Test Results: `results/BASELINE_FINAL.json`
- T1 Test (EI only): `results/bt_T1_EI_only_A3_OFF.json`
- A.3 Sweep: `results/bt_A3only_exit{45,55,60,70}.json`
- Logs: `/tmp/p0_1a_baseline.log`, `/tmp/p0_2b_t1_ei_only.log`
- Code: `apex-ranker/scripts/backtest_smoke_test.py`
- Selection Logic: `apex-ranker/apex_ranker/backtest/selection.py`
