# Phase 4.2: Walk-Forward Validation Plan

**Status**: 📋 Planning
**Target Start**: 2025-10-30
**Estimated Duration**: 2-3 weeks
**Objective**: Validate model stability and performance degradation over time

---

## 🎯 Objectives

### Primary Goals
1. **Verify Temporal Stability**: Confirm monthly rebalancing Sharpe >2.5 holds across different market regimes
2. **Detect Model Decay**: Identify if model performance degrades without retraining
3. **Validate Out-of-Sample**: Ensure results aren't due to overfitting to 2023-2025 period
4. **Optimize Retraining Schedule**: Determine if monthly retraining improves performance

### Success Criteria
- ✅ **Median Sharpe Ratio** across folds: >2.0 (target: >2.5)
- ✅ **Consistency**: <30% variance in Sharpe across folds
- ✅ **No significant decay**: Performance stable over 12+ month test periods
- ⚠️ **Early warning**: If any fold drops below 1.5 Sharpe, investigate

---

## 📐 Methodology

### Walk-Forward Splits

**Configuration**:
```yaml
Training Window: 252 trading days (~1 year)
Test Window: 63 trading days (~3 months)
Step Size: 21 trading days (~1 month)
Total Folds: 10-12 (depending on data availability)
Period: 2020-01-01 → 2025-10-24
```

**Visual Representation**:
```
Fold 1:  [Train: 2020-01 → 2021-01] [Test: 2021-01 → 2021-04]
Fold 2:  [Train: 2020-02 → 2021-02] [Test: 2021-02 → 2021-05]
...
Fold 10: [Train: 2023-10 → 2024-10] [Test: 2024-10 → 2025-01]
Fold 11: [Train: 2023-11 → 2024-11] [Test: 2024-11 → 2025-02]
Fold 12: [Train: 2023-12 → 2024-12] [Test: 2024-12 → 2025-03]
```

### Retraining Strategy

**Option A: Static Model** (Baseline)
- Use same `apex_ranker_v0_enhanced.pt` for all folds
- No retraining between folds
- Tests model robustness without adaptation

**Option B: Rolling Retraining** (Adaptive)
- Retrain model on each fold's training window
- Tests benefit of continuous adaptation
- Higher computational cost (~12 training runs)

**Recommendation**: Run both, compare performance delta

---

## 🛠️ Implementation Plan

### Phase 4.2.1: Infrastructure Setup (Week 1, Days 1-2)

#### Task 4.2.1.1: Walk-Forward Splitter
**File**: `apex-ranker/backtest/walk_forward.py`

```python
class WalkForwardSplitter:
    """
    Generate walk-forward train/test splits

    Parameters:
        train_days: Training window size (default: 252)
        test_days: Test window size (default: 63)
        step_days: Step size between folds (default: 21)
    """

    def split(self, dates: list[date]) -> list[tuple[list[date], list[date]]]:
        """Returns [(train_dates, test_dates), ...]"""
        pass
```

**Deliverables**:
- [ ] `walk_forward.py` implementation
- [ ] Unit tests for edge cases (insufficient data, gaps)
- [ ] Validation against manual split verification

#### Task 4.2.1.2: Backtest Runner
**File**: `apex-ranker/scripts/run_walk_forward_backtest.py`

```bash
python apex-ranker/scripts/run_walk_forward_backtest.py \
  --model models/apex_ranker_v0_enhanced.pt \
  --config apex-ranker/configs/v0_base.yaml \
  --data output/ml_dataset_latest_full.parquet \
  --train-days 252 \
  --test-days 63 \
  --step-days 21 \
  --rebalance-freq monthly \
  --top-k 50 \
  --horizon 20 \
  --output results/walk_forward_static.json
```

**Features**:
- Automatic fold generation
- Per-fold backtest execution
- Aggregate metrics calculation
- Progress tracking with ETA

**Deliverables**:
- [ ] Script implementation
- [ ] CLI argument validation
- [ ] JSON output format design
- [ ] Error handling for failed folds

### Phase 4.2.2: Static Model Validation (Week 1, Days 3-5)

#### Task 4.2.2.1: Run Static Walk-Forward
**Command**:
```bash
nohup python apex-ranker/scripts/run_walk_forward_backtest.py \
  --model models/apex_ranker_v0_enhanced.pt \
  --config apex-ranker/configs/v0_base.yaml \
  --data output/ml_dataset_latest_full.parquet \
  --train-days 252 --test-days 63 --step-days 21 \
  --rebalance-freq monthly --top-k 50 --horizon 20 \
  --output results/walk_forward_static_monthly.json \
  > /tmp/walk_forward_static.log 2>&1 &
```

**Expected Runtime**: ~3-4 hours (12 folds × 15-20 min/fold)

**Deliverables**:
- [ ] Walk-forward results JSON
- [ ] Per-fold performance metrics
- [ ] Aggregate statistics (mean, median, std)

#### Task 4.2.2.2: Analysis & Visualization
**Metrics to Track**:
- Sharpe ratio distribution across folds
- Return consistency (mean, std, min, max)
- Max drawdown per fold
- Transaction cost stability
- Performance by market regime (bull/bear/neutral)

**Visualizations**:
1. **Sharpe Evolution**: Line plot of Sharpe by fold
2. **Return Distribution**: Box plot of returns across folds
3. **Drawdown Analysis**: Max DD by fold
4. **Cost Stability**: Transaction costs by fold

**Tool**: Jupyter notebook `apex-ranker/notebooks/walk_forward_analysis.ipynb`

**Deliverables**:
- [ ] Analysis notebook
- [ ] Performance summary report
- [ ] Identified weak folds (if any)

### Phase 4.2.3: Rolling Retraining (Week 2, Optional)

**Note**: Only execute if static model shows significant decay (>20% Sharpe variance)

#### Task 4.2.3.1: Training Automation
**File**: `apex-ranker/scripts/train_fold.py`

```bash
python apex-ranker/scripts/train_fold.py \
  --data output/ml_dataset_latest_full.parquet \
  --train-start 2020-01-01 \
  --train-end 2021-01-01 \
  --output models/fold_1.pt \
  --config apex-ranker/configs/v0_base.yaml \
  --max-epochs 50
```

**Deliverables**:
- [ ] Fold-specific training script
- [ ] Model checkpoint management
- [ ] Training convergence validation

#### Task 4.2.3.2: Run Rolling Walk-Forward
**Command**:
```bash
nohup python apex-ranker/scripts/run_walk_forward_backtest.py \
  --config apex-ranker/configs/v0_base.yaml \
  --data output/ml_dataset_latest_full.parquet \
  --train-days 252 --test-days 63 --step-days 21 \
  --rebalance-freq monthly --top-k 50 --horizon 20 \
  --retrain-per-fold \
  --output results/walk_forward_rolling_monthly.json \
  > /tmp/walk_forward_rolling.log 2>&1 &
```

**Expected Runtime**: ~50-60 hours (12 folds × 4-5 hours training/fold)

**Deliverables**:
- [ ] Rolling retrain results
- [ ] Comparison: Static vs Rolling
- [ ] ROI analysis (retraining cost vs performance gain)

### Phase 4.2.4: Reporting & Decision (Week 2-3)

#### Task 4.2.4.1: Comprehensive Report
**File**: `apex-ranker/docs/WALK_FORWARD_VALIDATION_REPORT.md`

**Sections**:
1. **Executive Summary**: Key findings and recommendations
2. **Methodology**: Walk-forward configuration details
3. **Static Model Results**: Performance across folds
4. **Rolling Model Results**: Performance with retraining (if applicable)
5. **Regime Analysis**: Performance by market condition
6. **Recommendations**: Deployment strategy and retraining schedule

**Deliverables**:
- [ ] Comprehensive report
- [ ] Performance visualizations
- [ ] Risk assessment

#### Task 4.2.4.2: Deployment Decision
**Decision Framework**:

| Scenario | Median Sharpe | Recommendation |
|----------|---------------|----------------|
| **Excellent** | >2.5 | Deploy immediately with monthly rebalancing |
| **Good** | 2.0-2.5 | Deploy with conservative sizing |
| **Acceptable** | 1.5-2.0 | Deploy with monitoring, plan improvements |
| **Poor** | <1.5 | Delay deployment, investigate model issues |

**Deliverables**:
- [ ] Go/No-go deployment decision
- [ ] Risk mitigation plan (if deploying with caveats)
- [ ] Phase 4.3 kickoff (API development)

---

## 📊 Expected Outputs

### Per-Fold Metrics
```json
{
  "fold_id": 1,
  "train_period": {"start": "2020-01-01", "end": "2021-01-01"},
  "test_period": {"start": "2021-01-01", "end": "2021-04-01"},
  "performance": {
    "total_return_pct": 35.42,
    "sharpe_ratio": 2.31,
    "sortino_ratio": 2.89,
    "max_drawdown_pct": 18.45,
    "win_rate": 0.585,
    "total_trades": 312,
    "transaction_costs_pct": 8.12
  },
  "market_regime": "bull"  // bull/bear/neutral (based on TOPIX)
}
```

### Aggregate Statistics
```json
{
  "total_folds": 12,
  "sharpe_ratio": {
    "mean": 2.48,
    "median": 2.51,
    "std": 0.32,
    "min": 1.87,
    "max": 2.95
  },
  "returns": {
    "mean": 38.21,
    "median": 36.78,
    "std": 12.45,
    "min": 18.92,
    "max": 54.31
  },
  "weak_folds": [3, 7],  // Folds with Sharpe < 2.0
  "regime_performance": {
    "bull": {"median_sharpe": 2.78, "folds": [1, 2, 5, 9]},
    "bear": {"median_sharpe": 1.95, "folds": [3, 7]},
    "neutral": {"median_sharpe": 2.42, "folds": [4, 6, 8, 10, 11, 12]}
  }
}
```

---

## 🚨 Risk Management

### Potential Issues

#### Issue 1: High Variance Across Folds
**Symptom**: Sharpe std > 0.5 (e.g., 1.2 to 3.5 range)
**Cause**: Model overfits to specific market regimes
**Mitigation**:
1. Investigate weak folds (market conditions, outliers)
2. Consider regime-specific model ensembles
3. Add robustness constraints to training

#### Issue 2: Performance Decay Over Time
**Symptom**: Later folds consistently worse than early folds
**Cause**: Distribution shift, model staleness
**Mitigation**:
1. Implement rolling retraining (Phase 4.2.3)
2. Add drift detection monitoring
3. Schedule monthly/quarterly model refresh

#### Issue 3: Single Fold Failure
**Symptom**: One fold shows <1.0 Sharpe while others >2.0
**Cause**: Extreme market event (e.g., COVID crash)
**Mitigation**:
1. Identify event (check market news, TOPIX movement)
2. Exclude outlier fold from statistics (with justification)
3. Add event-based risk controls to production

---

## 📅 Timeline

### Week 1: Infrastructure & Static Validation
- **Day 1-2**: Implement walk-forward splitter and runner
- **Day 3**: Run static walk-forward backtest (overnight)
- **Day 4-5**: Analyze results, create visualizations

**Milestone**: Static walk-forward validation complete

### Week 2: Rolling Retraining (Conditional)
- **Day 1-2**: Implement fold-specific training automation
- **Day 3-7**: Run rolling walk-forward (48+ hours runtime)

**Milestone**: Rolling validation complete (if needed)

### Week 3: Reporting & Decision
- **Day 1-2**: Compile comprehensive report
- **Day 3**: Team review and deployment decision
- **Day 4-5**: Phase 4.3 planning (API development)

**Milestone**: Deployment go/no-go decision

---

## ✅ Success Checkpoints

### Checkpoint 1: Infrastructure Ready
- [ ] Walk-forward splitter tested and verified
- [ ] Backtest runner handles all edge cases
- [ ] JSON output format finalized

### Checkpoint 2: Static Validation Complete
- [ ] All 12 folds executed successfully
- [ ] Performance metrics aggregated
- [ ] Weak folds (if any) identified and analyzed

### Checkpoint 3: Decision Made
- [ ] Walk-forward report completed
- [ ] Team reviewed findings
- [ ] Go/no-go deployment decision documented
- [ ] Phase 4.3 plan approved (if deploying)

---

## 🔗 Dependencies

### Data Dependencies
- `output/ml_dataset_latest_full.parquet` (2020-2025 data)
- TOPIX index for market regime classification

### Model Dependencies
- `models/apex_ranker_v0_enhanced.pt` (current best model)
- `apex-ranker/configs/v0_base.yaml` (configuration)

### Code Dependencies
- Existing backtest infrastructure (`backtest_smoke_test.py`)
- Panel dataset builder (`apex_ranker/data/panel_dataset.py`)
- Inference pipeline (`scripts/inference_v0.py`)

---

## 📖 References

### Academic Literature
- **"Advances in Financial Machine Learning"** (Marcos López de Prado)
  - Chapter 7: Cross-Validation in Finance
  - Chapter 12: Backtesting through Cross-Validation
- **"Machine Learning for Asset Managers"** (Marcos López de Prado)
  - Walk-forward optimization pitfalls

### Industry Standards
- **CFM Research**: Walk-forward validation best practices
- **AQR**: Out-of-sample testing protocols
- **Renaissance Technologies**: Rolling window methodology (public talks)

---

**Status**: Planning Complete, Ready to Execute
**Next Action**: Implement Task 4.2.1.1 (Walk-Forward Splitter)
**Owner**: Development Team
**Reviewer**: Technical Lead
