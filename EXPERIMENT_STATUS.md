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
