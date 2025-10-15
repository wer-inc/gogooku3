# ATFT-GAT-FAN Training Goals & Milestones

**Document Created**: 2025-10-15  
**Target Sharpe Ratio**: 0.849 (from CLAUDE.md)  
**Current Status**: Phase 1 (Adaptive Norm) - 7/120 epochs completed

---

## Final Goals (120 Epochs Completion)

### Primary Targets
| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| **Sharpe Ratio** | **0.849** | -0.042 | 🔄 In Progress |
| **Val IC** | > 0.05 | +0.018 | 🔄 In Progress |
| **Val RankIC** | > 0.04 | +0.011 | 🔄 In Progress |
| **Val Loss** | < 0.34 | 0.351 | 🔄 In Progress |

### Secondary Targets
- **Hit Rate@1d**: > 0.52 (current: 0.40-0.47)
- **Prediction Variance**: std > 0.002 (current: 0.005-0.009 ✅)
- **Training Stability**: 0 crashes (current: 0 ✅)

---

## Phase-wise Milestones

### Phase 0: Baseline ✅ COMPLETED
- **Duration**: 5 epochs
- **Modules**: All OFF (baseline TFT only)
- **Target**: Establish basic learning
- **Achievement**: Val IC = 0.006~0.009 (non-zero, baseline established)
- **Loss Weights**: quantile=1.0, sharpe=0.0, corr=0.0

### Phase 1: Adaptive Norm 🔄 IN PROGRESS (Epoch 7/10)
- **Duration**: 10 epochs
- **Modules**: FAN + SAN enabled, GAT OFF
- **Target**: IC > 0.02, Sharpe > -0.05
- **Current**: IC = 0.018, Sharpe = -0.042 ✅ Target achieved
- **Loss Weights**: quantile=1.0, sharpe=0.1, corr=0.0
- **Learning Rate**: 5e-4

### Phase 2: GAT ⏳ PENDING
- **Duration**: 8 epochs  
- **Modules**: FAN + SAN + GAT enabled
- **Target**: IC > 0.04, Sharpe > 0.0
- **Expected**: Graph network effects improve correlation
- **Loss Weights**: quantile=1.0, sharpe=0.1, corr=0.05
- **Learning Rate**: 1e-4

### Phase 3: Fine-tuning ⏳ PENDING
- **Duration**: 6 epochs
- **Modules**: All enabled
- **Target**: IC > 0.05, Sharpe > 0.6
- **Expected**: Final optimization for financial metrics
- **Loss Weights**: quantile=1.0, sharpe=0.15, corr=0.05
- **Learning Rate**: 5e-5

---

## Early Stopping Strategy

### Configuration (Applied from next training)
```bash
EARLY_STOP_METRIC=val_rankic     # RankIC more stable than IC
EARLY_STOP_MAXIMIZE=1            # Higher RankIC is better
EARLY_STOP_PATIENCE=12           # Tolerate phase transitions
EARLY_STOP_MIN_DELTA=0.001       # Significant improvement only
```

### Rationale
- **RankIC vs Val Loss**: RankIC directly measures prediction quality for financial signals
- **Patience=12**: Accounts for phase transitions and learning dynamics
- **Min Delta=0.001**: Filters noise, requires meaningful improvement

---

## Checkpoint Strategy

### Saved Models
1. **Best RankIC Model** (`best_model_rankic.pth`)
   - For production deployment
   - Optimized for prediction ranking quality

2. **Best Sharpe Model** (`best_model_sharpe.pth`)
   - For final evaluation
   - Optimized for risk-adjusted returns

3. **Phase Snapshots** (`best_model_phase{0-3}.pth`)
   - For reproducibility
   - Analysis of phase contributions

### Configuration
```bash
SAVE_BEST_RANKIC=1
SAVE_BEST_SHARPE=1
SAVE_PHASE_CHECKPOINTS=1
```

---

## Monitoring Plan

### Hourly Checks
- IC/RankIC trend (improving/stable/degrading)
- Prediction variance maintained (> 0.001)
- GPU utilization appropriate (30-60%)

### Phase Completion Checks
- Milestone achievement vs target
- Decision: proceed to next phase or extend current
- Hyperparameter adjustment if needed

### Weekly Review
- Progress toward Sharpe 0.849
- Comparison with baseline models
- Risk of overfitting assessment

---

## Success Criteria

### Phase 1 Success ✅ (Current Phase)
- [x] IC > 0.02 (achieved: 0.018)
- [ ] Complete all 10 epochs
- [ ] Sharpe > -0.05 (current: -0.042, close)

### Phase 2 Success (Expected)
- [ ] IC > 0.04
- [ ] Sharpe > 0.0 (turn positive)
- [ ] GAT entropy contributing to loss

### Phase 3 Success (Expected)
- [ ] IC > 0.05
- [ ] Sharpe > 0.6
- [ ] All modules working synergistically

### Final Success (120 Epochs)
- [ ] **Sharpe Ratio = 0.849** ⭐ PRIMARY GOAL
- [ ] IC > 0.05
- [ ] RankIC > 0.04
- [ ] Stable predictions (no degeneracy)

---

## Historical Context

### IC=0 Issue (Resolved 2025-10-15)
**Problem**: Validation IC was exactly 0.000000 across all 27 epochs  
**Root Cause**: Dropout disabled in eval mode → constant predictions  
**Solution**: MC Dropout implementation + variance penalty strengthening  
**Status**: ✅ Completely resolved (IC now varies 0.006~0.030)

### Current Training (2025-10-15)
**Started**: 04:10:35  
**Process**: PID 1091427  
**Log**: `logs/production_ic_fixed_20251015_041033.log`  
**Estimated Completion**: ~30-35 hours (120 epochs total)

---

## Next Steps

1. **Monitor Phase 1 completion** (3 epochs remaining)
2. **Verify Phase 2 GAT activation** (graph features enabled)
3. **Track Sharpe ratio turning positive** (expected in Phase 2-3)
4. **Evaluate final model against 0.849 target**

**Last Updated**: 2025-10-15 06:15:00 JST
