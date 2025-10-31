# Phase 4.3.2 → 4.3.3 Regime Threshold Tuning Plan

**Status:** ready to execute
**Author:** Phase 4 task force
**Date:** 2025-10-30

## 1. Goal

Improve the real-time regime detector so that it adapts to Japanese market dynamics without
misclassifying high-momentum bull periods. Target improvements vs. the Phase 4.3.1b baseline:

- Sharpe ratio (median, monthly rebalanced walk-forward): +5–10%
- Negative-fold rate (<0 Sharpe): reduce from 18.2% to <12%
- Preserve >95% exposure during strong uptrends while enforcing protective cuts in stress periods

## 2. Data Snapshot (2020-10-27 → 2025-10-24)

All statistics pulled from `output/ml_dataset_latest_full.parquet` and aggregated to one
observation per trading day.

| Quantile | 20d annualised volatility | 20d momentum |
|----------|---------------------------|--------------|
| 50%      | 0.148                     | 1.3%         |
| 75%      | 0.183                     | 3.6%         |
| 90%      | 0.213                     | 5.9%         |
| 95%      | 0.260                     | —            |
| 99%      | 0.646                     | —            |

Yearly averages (vol = annualised, mom = 20d cumulative return):

| Year | vol_mean | vol_median | vol_p90 | vol_max | mom_mean | mom_median |
|------|----------|------------|---------|---------|----------|------------|
|2020|0.138|0.139|0.171|0.187|4.0%|4.0%|
|2021|0.160|0.160|0.195|0.211|0.9%|0.9%|
|2022|0.177|0.176|0.223|0.284|-0.2%|-0.2%|
|2023|0.137|0.131|0.192|0.222|2.2%|2.2%|
|2024|0.195|0.147|0.264|0.662|1.6%|1.6%|
|2025|0.175|0.131|0.368|0.554|1.7%|1.7%|

## 3. Proposed Threshold Updates

### 3.1 Volatility & Momentum Bands

| Regime | Annualised vol (20d) | 20d momentum | Additional rule |
|--------|----------------------|--------------|-----------------|
| **Crisis** | >0.30 OR (>0.26 and momentum < -3%) | any | drawdown > 10% → exposure capped 20% |
| **Bear** | 0.18–0.30 | < -1.5% | if vol >0.22 and momentum <-2.5% → exposure 35% |
| **Bull** | <0.18 | > +2.0% | exposure 100%; if confidence <0.75 → clamp 80% |
| **Sideways** | others | — | exposure 60–80% depending on drawdown |

### 3.2 Lookback & Confidence

- **Lookback window:** evaluate 15d, 20d (current), 30d, 40d
- **Confidence threshold:** start at 0.75 (was 0.80), examine 0.70–0.85 range
- **Momentum smoothing:** test EMA(half-life 5 days) to reduce regime flaps

### 3.3 Exposure Curve (baseline)

| Regime | Drawdown <5% | Drawdown 5–10% | Drawdown >10% |
|--------|---------------|----------------|----------------|
| Crisis | 20% | 15% | 10% |
| Bear   | 50% | 40% | 25% |
| Sideways | 75% | 65% | 50% |
| Bull   | 100% | 85% | 70% |

## 4. Experiment Plan

1. **Historical profiling (Day 1)**
   - Confirm above statistics for sub-periods (2021, 2022 crisis, 2024 rally).
   - Produce plots (volatility time series, momentum distribution) for documentation.

2. **Grid search (Days 1-2)**
   - Evaluate combinations: lookback {15,20,30}, vol thresholds {0.24,0.26,0.30}, momentum thresholds {-1%, -2%, -3%}.
   - Run regime-adaptive backtest on crisis window and 12-fold sample (every fourth fold) to prune weak configs.

3. **Full walk-forward validation (Day 3)**
   - Run 44-fold regime-adaptive walk-forward with top 2 parameter sets.
   - Collect: Sharpe median/mean, exposure distribution, negative-fold count.

4. **Selection criteria**
   - Sharpe median ≥ static Sharpe (2.227) + 5%.
   - Negative folds ≤ 6 of 44.
   - Average exposure within 60–85% range.

5. **Deliverables**
   - Updated `RegimeDetector` configuration (YAML/JSON) for production.
   - Analysis report (plots + metrics table).
   - PR with code + documentation.

## 5. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Overfitting crisis window | Use multi-period sampling in grid search; reserve folds for out-of-sample check |
| Increased churn due to regime flipping | Apply confidence smoothing & minimum holding period (e.g., 5 trading days) |
| Runtime overhead | Cache daily volatility/momentum series; reuse for multiple parameter sets |

## 6. Timeline (Estimate)

| Day | Task |
|-----|------|
| 1 | Historical profiling + grid search setup |
| 2 | Run grid search + analyze sample folds |
| 3 | Full walk-forward with finalists + decision |

## 7. Dependencies

- `realtime_regime.py` (scale fix applied)
- `backtest_regime_adaptive.py` (regime-aware backtest driver)
- Walk-forward dataset `output/ml_dataset_latest_full.parquet`
- Compute time for 44-fold evaluation (~30–60 minutes per run)
