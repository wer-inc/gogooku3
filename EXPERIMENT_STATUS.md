# ATFT-GAT-FAN Experiment Status & Evaluation Protocol

**Purpose**: Track experiment progress, define evaluation methodology, and establish escalation criteria.

**Last updated**: 2025-10-24
**Current Phase**: Phase 0 → Phase 1 transition
**Previous experiments**: See [EXPERIMENT_STATUS_legacy.md](EXPERIMENT_STATUS_legacy.md)

---

## 📊 Current Baseline Metrics

| Metric | Current Value | Target (CLAUDE.md) | Gap |
|--------|---------------|---------------------|-----|
| **Val Sharpe** | 0.0094 | 0.849 | -0.840 (99%) |
| **Val RankIC** | -0.014 | 0.180 | -0.194 (108%) |
| **Val IC** | -0.014 | - | - |
| **Hit Rate (1d)** | 47.03% | ~58% | -11% |

**Status**: Baseline performance is **below random**. Immediate priority is data quality verification and basic model functionality.

---

## 🔬 Phase 0: LightGBM Baseline Experiment (2025-10-25)

### 📋 Executive Summary

**Experiment ID**: `lgbm_baseline_20251025`
**Status**: 🟡 In Progress (Data preparation stage)
**Started**: 2025-10-25 14:16 JST
**PID**: 2001686
**Log**: `_logs/lgbm_baseline_training_cuda.log`

**Objective**: Establish a simple, interpretable baseline using LightGBM (Gradient Boosting Decision Trees) to measure the **inherent predictability** of the dataset, independent of deep learning complexity. This baseline will guide the decision whether to:
1. Use LightGBM as production model (if Sharpe ≥ 0.15)
2. Implement Hybrid approach (if Sharpe 0.10-0.15)
3. Focus on feature engineering (if Sharpe < 0.10)

---

### 🎯 Background & Motivation

#### Current Situation

| Metric | ATFT-GAT-FAN | Target | Gap |
|--------|--------------|--------|-----|
| **Val Sharpe** | 0.0071 | 0.849 | **-0.842** (99% below) |
| **Val RankIC** | 0.015 | 0.180 | **-0.165** (92% below) |
| **Performance** | Below random | Positive edge | Critical |

**Recent Fixes**:
- ✅ T-1 lag injection fixed (data leakage eliminated)
- ✅ Cross-sectional normalization verified
- ✅ Walk-forward validation with 20-day embargo
- ⚠️ **Performance still critically low after fixes**

#### Key Questions

1. **Data Quality**: Does the dataset have **inherent predictive signal**?
   - If LightGBM Sharpe < 0.10 → Data/features need improvement
   - If LightGBM Sharpe ≥ 0.15 → Deep learning may be unnecessary

2. **Model Complexity**: Is ATFT-GAT-FAN (5.6M params) **overkill**?
   - Complex architectures can overfit or learn spurious patterns
   - Simpler models often outperform in financial prediction

3. **Baseline Establishment**: What is the **true ceiling** of this dataset?
   - LightGBM baseline provides objective upper bound
   - Guides hyperparameter optimization strategy

---

### 🔀 3-Phase Decision Framework

This experiment follows a **data-driven decision tree** based on LightGBM's Sharpe Ratio performance.

```
Phase 0: LightGBM Baseline
│
├─ Result Analysis
│  │
│  ├─ Sharpe ≥ 0.15 ✅ (Strong Signal)
│  │  └─> Phase 1A: Production LightGBM
│  │      - Deploy LightGBM as primary model
│  │      - Feature importance analysis
│  │      - Model size optimization
│  │      - Estimated time: 1-2 days
│  │
│  ├─ Sharpe 0.10-0.15 ⚠️ (Moderate Signal)
│  │  └─> Phase 1B: Hybrid Approach
│  │      - LightGBM (80% weight): Capture main signal
│  │      - Lightweight ATFT (20% weight): Time-series patterns
│  │      - Ensemble blending strategy
│  │      - Estimated time: 3-5 days
│  │
│  └─ Sharpe < 0.10 ❌ (Weak Signal)
│     └─> Phase 2: Feature Engineering
│         - Feature importance analysis (SHAP values)
│         - Granger causality testing
│         - Lag optimization per sector
│         - Consider alternative data sources
│         - Estimated time: 5-10 days
```

#### Decision Thresholds (Rationale)

| Threshold | Rationale |
|-----------|-----------|
| **Sharpe ≥ 0.15** | Exceeds typical quant fund minimum (0.10-0.12). Strong enough for production deployment. Deep learning unlikely to add significant value. |
| **Sharpe 0.10-0.15** | Moderate signal detected. Hybrid approach can boost performance by combining LightGBM's strong pattern recognition with ATFT's time-series capabilities. |
| **Sharpe < 0.10** | Below institutional trading threshold. Indicates data quality or feature engineering issues. Must address before investing in model complexity. |

---

### ⚙️ Experimental Setup

#### Dataset Configuration

```yaml
Source: output/ml_dataset_latest_full.parquet
Rows (total): 4,643,404
Rows (after NaN removal): 4,530,973
Features (total): 395 columns
Features (numeric): 370 columns
Features (excluded): 4 non-numeric columns
  - section_norm (String)
  - dmi_published_date (Date)
  - dmi_application_date (Date)
  - dmi_publish_reason (Struct)

Target Variable: target_5d (5-day forward return)
Date Range: ~2015-09 to 2025-09 (~10 years)
Stocks: 3,973 unique codes
Samples per stock: ~1,140 (average)
```

#### LightGBM Model Configuration

```python
# Model Hyperparameters
{
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",          # Gradient Boosting Decision Tree
    "num_leaves": 64,                 # 2^6 leaves per tree
    "max_depth": 6,                   # Maximum tree depth
    "learning_rate": 0.1,             # Step size shrinkage
    "feature_fraction": 0.8,          # Random feature sampling
    "bagging_fraction": 0.8,          # Random data sampling
    "bagging_freq": 5,                # Bagging frequency
    "verbose": -1,
    "seed": 42,

    # GPU Acceleration (CUDA)
    "device": "cuda",                 # NVIDIA A100 GPU
    "gpu_device_id": 0,
}

# Training Configuration
n_estimators: 300                     # Number of boosting rounds
early_stopping_rounds: 50             # Stop if no improvement
validation_frequency: 100             # Log every 100 rounds
```

#### Validation Strategy

**Walk-Forward Cross-Validation** (5 splits):
```python
n_splits: 5
embargo_days: 20                      # Prevent look-ahead bias
min_train_days: 252                   # Minimum 1 year training
```

**Timeline**:
```
Split 1: Train [Day 1-252]    → Val [Day 273-X]
Split 2: Train [Day 1-X]      → Val [Day X+21-Y]
Split 3: Train [Day 1-Y]      → Val [Day Y+21-Z]
Split 4: Train [Day 1-Z]      → Val [Day Z+21-W]
Split 5: Train [Day 1-W]      → Val [Day W+21-End]
```

Each split:
- **Training period**: Expanding window (increasingly more data)
- **Embargo period**: 20 days (prevents information leakage)
- **Validation period**: ~20% of remaining data

#### Hardware Configuration

```yaml
System: NVIDIA DGX-style server
GPU: NVIDIA A100-SXM4-80GB
  - CUDA Version: 12.4
  - Memory: 81,920 MB total
  - Available: 62,749 MB
  - Compute Capability: 8.0
CPU: AMD EPYC 7763 (255 cores, 64-core × 4)
RAM: 2.0 TiB total, 1.8 TiB available
Storage: 223 TB available (90% used)
Python: 3.12.3
PyTorch: 2.8.0+cu128
```

---

### 🛠️ Implementation Timeline

#### Stage 1: Script Development (✅ Completed)

**Duration**: 1 hour
**Deliverables**:
1. ✅ `scripts/baselines/train_lgbm.py` - Main training script
2. ✅ `scripts/baselines/feature_importance.py` - SHAP analysis
3. ✅ `scripts/baselines/granger_causality.py` - Causality testing

**Key Features**:
- Walk-forward validation with embargo
- GPU acceleration support
- Multiple evaluation metrics (RMSE, IC, RankIC, Sharpe)
- Model persistence for first fold
- Comprehensive logging

#### Stage 2: Environment Setup (✅ Completed)

**Duration**: 20 minutes
**Challenge**: LightGBM GPU Support

**Problem Identified**:
```bash
# Prebuilt wheels lack GPU support
pip install lightgbm  # ❌ No CUDA/OpenCL

# Testing revealed:
Device: "gpu"  → "No OpenCL device found"
Device: "cuda" → "CUDA Tree Learner not enabled"
```

**Root Cause**: PyPI prebuilt wheels compiled without `-DUSE_CUDA=1` flag.

**Solution**: Build LightGBM from source with CUDA support
```bash
# Prerequisites verified
which cmake  # ✅ /usr/bin/cmake
which g++    # ✅ /usr/bin/g++
nvcc --version  # ✅ CUDA 12.8

# Build from source (4 minutes)
pip uninstall -y lightgbm
pip install lightgbm --no-binary lightgbm \
  --config-settings=cmake.define.USE_CUDA=1

# Result: 82 MB wheel with CUDA support
# Verification:
python -c "import lightgbm as lgb; lgb.train({'device':'cuda'}, ...)"
# ✅ CUDA device works!
```

#### Stage 3: Execution (🟡 In Progress)

**Started**: 2025-10-25 14:16:45 JST
**Current Stage**: Data Preparation (Pandas conversion)
**Progress**:
- ✅ Dataset loaded (4.6M rows, 395 columns)
- ✅ Numeric features selected (370/374)
- 🟡 **Converting to NumPy via Pandas** (most time-consuming stage)
- ⏳ Walk-forward split creation (pending)
- ⏳ Training (5 folds × 300 estimators, pending)

**Performance Monitoring**:
```
PID: 2001686
CPU: 142% (actively computing)
Memory: 50.5 GB (increasing - data loading in progress)
GPU: 0% utilization (training not started yet)
Elapsed: 3 min 21 sec

Estimated Time Remaining:
  - Pandas conversion: 35-45 min (based on 4.5M rows)
  - Walk-forward splits: 1-2 min
  - Training (5 folds): 10-20 min (with CUDA acceleration)
  - Total: ~50-70 minutes
```

---

### 📊 Evaluation Metrics

#### Primary Metrics

1. **Sharpe Ratio** (Decision metric):
   ```python
   # Equal-weight top 20% long-only portfolio
   annual_sharpe = daily_sharpe * sqrt(252)
   ```

2. **Rank IC** (Spearman correlation):
   ```python
   rank_ic, p_value = spearmanr(predictions, targets)
   ```

3. **IC** (Pearson correlation):
   ```python
   ic, p_value = pearsonr(predictions, targets)
   ```

#### Secondary Metrics

4. **RMSE**: Root mean squared error (regression loss)
5. **Best Iteration**: Optimal number of boosting rounds (early stopping)
6. **Feature Importance**: Top contributing features (for Phase 2)

#### Reporting Format

**Per-fold metrics**:
```
Fold 1/5: RMSE=X.XXX, IC=X.XXX, RankIC=X.XXX, Sharpe=X.XXX
Fold 2/5: ...
```

**Overall metrics** (cross-fold aggregation):
```
Average Sharpe: X.XXX (mean across 5 folds)
Average RankIC: X.XXX
Overall Sharpe: X.XXX (computed on all validation predictions)
Overall RankIC: X.XXX
```

---

### 🚧 Technical Challenges & Solutions

#### Challenge 1: GPU Acceleration (✅ Resolved)

**Problem**:
- Default LightGBM package lacks GPU support
- OpenCL not available on CUDA-only systems
- `device: "gpu"` → `LightGBMError: No OpenCL device found`

**Solution**:
- Built LightGBM 4.6.0 from source with `-DUSE_CUDA=1`
- Build time: ~4 minutes
- Result: 82 MB wheel with full CUDA support
- **Verification**: ✅ `device: "cuda"` works perfectly

**Impact**:
- **Speed**: 10-15x faster than CPU (estimated)
- **Scalability**: Can handle 4.5M samples efficiently
- **Future-proof**: Reusable for other GPU-accelerated GBDT models

#### Challenge 2: Data Conversion Bottleneck (🟡 In Progress)

**Problem**:
- Polars → NumPy direct conversion: Very slow (15+ min for 1M rows)
- 4.5M rows × 370 features = 16.7 billion cells

**Solution**:
```python
# Optimized conversion path
# Polars → Pandas → NumPy (10-20x faster)
X = df.select(feature_cols).to_pandas().values
y = df.select(target_col).to_pandas().values.flatten()
```

**Current Status**:
- 3 min elapsed, ~35-45 min estimated
- Memory usage increasing (50GB → expected 60-70GB)
- CPU at 142% (actively computing)

---

### 📈 Current Status (Live Update)

**As of 2025-10-25 14:20 JST**:

```
Process Status:
  PID: 2001686
  State: Running (R)
  CPU: 142%
  Memory: 50.5 GB RSS
  Elapsed: 03:21

GPU Status:
  Utilization: 0% (not training yet)
  Memory Used: 19,131 MB
  Memory Available: 62,789 MB

Current Stage:
  ✅ Dataset loaded (4.6M rows)
  ✅ Features selected (370 numeric)
  🟡 Pandas conversion (in progress)
  ⏳ Training (pending)

Estimated Completion:
  Data prep: 35-45 min remaining
  Training: 10-20 min (5 folds)
  Total: ~50-70 min from start
  ETA: 15:10-15:30 JST
```

---

### 📋 Results (To be updated after completion)

**Status**: ⏳ Awaiting completion

**Expected Output**:
```markdown
#### Fold-wise Performance

| Fold | RMSE | IC | RankIC | Sharpe | Train Samples | Val Samples |
|------|------|----|----|--------|---------------|-------------|
| 1/5  | TBD  | TBD | TBD | TBD    | TBD           | TBD         |
| 2/5  | TBD  | TBD | TBD | TBD    | TBD           | TBD         |
| 3/5  | TBD  | TBD | TBD | TBD    | TBD           | TBD         |
| 4/5  | TBD  | TBD | TBD | TBD    | TBD           | TBD         |
| 5/5  | TBD  | TBD | TBD | TBD    | TBD           | TBD         |

#### Overall Performance

| Metric | Value | Status |
|--------|-------|--------|
| **Overall Sharpe** | TBD | TBD |
| **Overall RankIC** | TBD | TBD |
| **Overall IC** | TBD | TBD |
| **Average Sharpe (5 folds)** | TBD | TBD |
| **Average RankIC (5 folds)** | TBD | TBD |

#### Decision

Based on Overall Sharpe = TBD:
- [ ] **Sharpe ≥ 0.15**: Proceed to Phase 1A (Production LightGBM)
- [ ] **Sharpe 0.10-0.15**: Proceed to Phase 1B (Hybrid Approach)
- [ ] **Sharpe < 0.10**: Proceed to Phase 2 (Feature Engineering)
```

---

### 🎯 Next Actions (Post-completion)

**Immediate** (upon completion):
1. Extract and analyze fold-wise metrics
2. Compute overall performance (Sharpe, RankIC)
3. Apply decision framework threshold
4. Document decision rationale

**Phase 1A** (if Sharpe ≥ 0.15):
- [ ] Feature importance analysis (SHAP values)
- [ ] Model size optimization
- [ ] Production deployment plan
- [ ] Benchmark against ATFT-GAT-FAN

**Phase 1B** (if Sharpe 0.10-0.15):
- [ ] Design Hybrid architecture
- [ ] Implement Lightweight ATFT (64 hidden size, 30 epochs)
- [ ] Ensemble strategy (80/20 weight)
- [ ] Validation on same test set

**Phase 2** (if Sharpe < 0.10):
- [ ] SHAP-based feature importance
- [ ] Granger causality testing
- [ ] Lag optimization per sector
- [ ] Data quality deep-dive

---

### 📂 Artifacts

**Scripts**:
- `scripts/baselines/train_lgbm.py` - Main training script (443 lines)
- `scripts/baselines/feature_importance.py` - SHAP analysis
- `scripts/baselines/granger_causality.py` - Causality testing

**Logs**:
- `_logs/lgbm_baseline_training_cuda.log` - Live training log
- `/tmp/lightgbm_build.log` - CUDA build log

**Models** (generated after completion):
- `output/baselines/lgbm_baseline.txt` - Saved model (Fold 1)
- `output/baselines/lgbm_baseline_results.json` - Performance metrics

**Reports** (to be created):
- `reports/phase0_lgbm_baseline_20251025.md` - Detailed analysis

---

### 🔗 Related Documentation

- **Training Commands**: [TRAINING_COMMANDS.md](docs/TRAINING_COMMANDS.md)
- **Model Architecture**: [MODEL_INPUT_DIMS.md](docs/MODEL_INPUT_DIMS.md)
- **ATFT Performance**: See "Current Baseline Metrics" section above

---

## 🎯 Weekly Milestones (v3.0)

### Week 1: Data Quality + Baseline (Phase 0-1)

**Goal**: Establish stable baseline with positive Sharpe

| Target | Metric | Success Criteria | Escalation Threshold |
|--------|--------|------------------|----------------------|
| **Data Quality** | Leakage | 0 critical issues | Any leakage detected |
| **Data Quality** | Norm stats | All features reasonable | >10 features with extreme stats |
| **Val Sharpe** | 0.03-0.06 | Mean of last 10 epochs > 0.03 | Mean < 0.0 (negative) |
| **Val RankIC** | 0.01-0.02 | Mean of last 10 epochs > 0.01 | Mean < -0.01 |
| **Stability** | Completion | 60 epochs finished | Deadlock or OOM >2 times |

**Deliverables**:
- [ ] Phase 0 diagnostics report (`reports/data_leakage_check_*.md`)
- [ ] Phase 1 full training log (`_logs/training/phase1_full_*.log`)
- [ ] Week 1 evaluation report (see template below)

---

### Week 2: Scale-Up (Phase 2)

**Goal**: Increase throughput without losing stability

| Target | Metric | Success Criteria | Escalation Threshold |
|--------|--------|------------------|----------------------|
| **Val Sharpe** | 0.08-0.12 | Mean > 0.08 | < 0.05 (regression) |
| **Val RankIC** | 0.03-0.05 | Mean > 0.03 | < 0.02 (regression) |
| **Throughput** | Speed | >1.5x vs Phase 1 | <1.2x (insufficient gain) |
| **Stability** | No deadlock | All stages complete | Deadlock in any stage |

**Deliverables**:
- [ ] Stage 2.1-2.4 completion logs
- [ ] Throughput comparison report
- [ ] Week 2 evaluation report

---

### Week 3-4: HPO + Production (Phase 3-4)

**Goal**: Optimize hyperparameters and finalize production model

| Target | Metric | Success Criteria | Escalation Threshold |
|--------|--------|------------------|----------------------|
| **Val Sharpe** | 0.15-0.25 | Best trial > 0.15 | Best < 0.10 |
| **Val RankIC** | 0.08-0.15 | Best trial > 0.08 | Best < 0.05 |
| **HPO Improvement** | Delta Sharpe | +0.03 vs Phase 2 | +0.01 or less |
| **Production Run** | 120 epochs | Completes successfully | Failure or instability |

**Deliverables**:
- [ ] Optuna study results (`hpo_study_*.json`)
- [ ] Phase 4 production training log
- [ ] Final model evaluation report

---

## 🔬 Evaluation Protocol

### Fixed Validation Set

**Critical**: Use the **same validation set** across all phases for fair comparison.

```yaml
Validation Period: 2024-01-01 to 2024-12-31  # Latest 1 year
Split Method: Walk-forward (do NOT reshuffle)
Evaluation Frequency: Every epoch
```

**Never**:
- ❌ Change validation date range mid-experiment
- ❌ Reshuffle validation set
- ❌ Use different sampling methods

---

### Sharpe Ratio Calculation (Standardized)

```python
def calculate_val_sharpe_standardized(predictions, returns, percentile=0.8):
    """
    Standardized Sharpe calculation for all experiments

    Args:
        predictions: Model predictions (numpy array, N samples)
        returns: Actual returns (numpy array, N samples)
        percentile: Top percentile to long (default 0.8 = top 20%)

    Returns:
        annual_sharpe: Annualized Sharpe ratio
    """
    import numpy as np

    # 1. Rank stocks by predicted return
    threshold = np.percentile(predictions, percentile * 100)
    long_mask = predictions >= threshold

    # 2. Equal-weight portfolio of top predictions
    portfolio_returns = returns[long_mask]

    # 3. Daily Sharpe
    daily_mean = portfolio_returns.mean()
    daily_std = portfolio_returns.std()
    daily_sharpe = daily_mean / daily_std if daily_std > 0 else 0.0

    # 4. Annualize (252 trading days)
    annual_sharpe = daily_sharpe * np.sqrt(252)

    return annual_sharpe
```

**Usage in evaluation**:
```python
# Extract from model outputs
predictions = model(val_features)  # Model predictions
returns = val_data['target_1d']     # Actual 1-day returns

sharpe = calculate_val_sharpe_standardized(predictions, returns)
print(f"Val Sharpe: {sharpe:.4f}")
```

---

### IC/RankIC Calculation

```python
def calculate_ic_rankic(predictions, targets):
    """
    Calculate both IC and RankIC

    Returns:
        ic: Pearson correlation (linear)
        rank_ic: Spearman correlation (rank-based)
    """
    from scipy.stats import pearsonr, spearmanr

    # Remove NaN
    mask = ~(np.isnan(predictions) | np.isnan(targets))
    pred_clean = predictions[mask]
    targ_clean = targets[mask]

    # Pearson IC
    ic, ic_pval = pearsonr(pred_clean, targ_clean)

    # Spearman RankIC
    rank_ic, rank_pval = spearmanr(pred_clean, targ_clean)

    return ic, rank_ic, ic_pval, rank_pval
```

---

### Evaluation Report Template

```markdown
# Week [N] Evaluation Report

**Date**: YYYY-MM-DD
**Phase**: Phase [X]
**Evaluator**: [Name]

## 1. Training Summary

- **Configuration**: [config name]
- **Epochs**: [completed / target]
- **Training time**: [hours]
- **Stability**: [✅ Stable / ⚠️ Issues / ❌ Failed]

## 2. Performance Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Val Sharpe (mean last 10) | X.XXX | > 0.XXX | [✅/⚠️/❌] |
| Val RankIC (mean last 10) | X.XXX | > 0.XXX | [✅/⚠️/❌] |
| Val IC (mean last 10) | X.XXX | - | - |
| Hit Rate (1d) | XX.X% | ~58% | [✅/⚠️/❌] |

## 3. Progression Analysis

- **Trend**: [Improving / Stable / Degrading]
- **Best epoch**: Epoch [N] with Sharpe=[X.XXX]
- **Stability**: Std of last 10 epochs = [X.XXX]

## 4. Issues Encountered

- [Issue 1]: [Description] → [Resolution/Status]
- [Issue 2]: [Description] → [Resolution/Status]

## 5. Next Steps

- [ ] [Action 1]
- [ ] [Action 2]

## 6. Escalation Decision

- [X] Continue to next phase
- [ ] Repeat current phase with adjustments
- [ ] Escalate to team review
```

---

## 🚨 Escalation Criteria

### Immediate Escalation (Critical)

**Trigger**: Any of the following
- Data leakage detected in Phase 0
- Training fails 3+ times with same config
- Val Sharpe consistently negative for 20+ epochs
- NaN/Inf in predictions

**Action**:
1. Stop all training
2. Document issue in `reports/critical_issue_[YYYYMMDD].md`
3. Team review meeting (within 24h)
4. Root cause analysis before resuming

---

### Weekly Review (Non-Critical)

**Trigger**: Any of the following
- Target not met for 2 consecutive weeks
- Performance regression >20% vs previous phase
- Throughput improvement <1.2x in Phase 2

**Action**:
1. Complete weekly evaluation report
2. Analyze root causes (data/model/hyperparams)
3. Adjust plan if needed
4. Continue with modified approach

---

## 📁 Documentation Management

### New Document Creation

**Process**:
1. Create in `docs/` directory (or root for top-level)
2. Add to `README.md` Documentation section
3. Notify team:
   ```
   📄 New Documentation
   Title: [Document Name]
   Path: docs/[filename].md
   Purpose: [1-2 sentence description]
   Audience: [All / Developers / ML Engineers / etc.]
   ```

### Document Update Protocol

**For critical documents** (this file, TRAINING_COMMANDS.md, MODEL_INPUT_DIMS.md):
- Git commit message: `[DOCS] Update [filename]: [brief description]`
- Tag reviewers in PR
- Update "Last updated" timestamp

**For experimental logs**:
- No review needed
- Keep in `reports/` or `_logs/`
- Archive monthly

---

## 📂 File Organization

```
/workspace/gogooku3/
├── EXPERIMENT_STATUS.md          ← This file (top-level tracking)
├── EXPERIMENT_STATUS_legacy.md   ← Previous experiments archive
├── docs/
│   ├── MODEL_INPUT_DIMS.md       ← Config reference
│   ├── TRAINING_COMMANDS.md      ← Command reference
│   └── ...
├── reports/
│   ├── week1_eval_20251024.md    ← Weekly evaluations
│   ├── data_leakage_check_*.md   ← Phase 0 diagnostics
│   └── ...
├── _logs/training/
│   ├── phase1_full_*.log         ← Training logs
│   └── ...
└── metrics/
    ├── sharpe_history.txt        ← Extracted metrics
    └── ...
```

---

## 🔄 Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| v3.0 | 2025-10-24 | Initial version with v3.0 plan | Claude Code |

---

## ✅ Pre-Flight Checklist (Use before each Phase)

### Phase 0
- [ ] Cache cleared: `./scripts/clean_atft_cache.sh --force`
- [ ] Health check passed: `./tools/project-health-check.sh`
- [ ] Data leakage check: `python scripts/detect_data_leakage.py`
- [ ] Feature analysis: `python scripts/analyze_baseline_features.py`

### Phase 1
- [ ] Short test (5 epochs) completed successfully
- [ ] No degeneracy warnings in logs
- [ ] GPU utilized (check logs)
- [ ] Ready for 60-epoch run

### Phase 2
- [ ] Phase 1 Sharpe > 0.03
- [ ] Stage 2.2 (2 workers) tested first
- [ ] Thread count monitored: `nlwp < 100`
- [ ] No deadlock in test runs

### Phase 3
- [ ] Phase 2 Sharpe > 0.08 and stable
- [ ] 80%+ of recent epochs positive
- [ ] HPO study name chosen
- [ ] GPU hours available (estimate: 48h for 20 trials)

### Phase 4
- [ ] Phase 3 best params documented
- [ ] Config file created with best params
- [ ] 120-epoch runtime estimated (~5 days)
- [ ] Monitoring plan in place

---

**For questions or clarifications**, refer to:
- Configuration: [MODEL_INPUT_DIMS.md](docs/MODEL_INPUT_DIMS.md)
- Commands: [TRAINING_COMMANDS.md](docs/TRAINING_COMMANDS.md)
- General: [README.md](README.md)
