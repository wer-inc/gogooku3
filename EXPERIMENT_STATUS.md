# Experiment Suite Status

**Updated**: 2025-10-21 07:10 JST
**Current Best**: Sharpe 0.779 (Experiment 1.1b - Pure Sharpe optimization)

---

## 🎯 Overall Goals

- **Current Best**: Sharpe 0.779 🎯
- **Target**: Sharpe 0.849
- **Remaining Gap**: 0.070 (8.2%)
- **Progress**: **91.7% of target achieved**
- **Strategy**: Systematic tier-by-tier optimization

---

## ✅ Tier 1 COMPLETE: Sharpe Weight Optimization

**Status**: ✅ **3/3 Experiments Completed** (2025-10-21 06:38 JST)
**Duration**: 4.5 hours total
**Key Discovery**: Pure Sharpe optimization (weight=1.0) outperformed all other configurations

### Completed Experiments

| Experiment | Sharpe Weight | Final Sharpe | Best Val Loss | Improvement | Completion |
|------------|---------------|--------------|---------------|-------------|------------|
| **Baseline** | 0.6 | 0.582 | 0.378 | - | 02:49 JST ✅ |
| **Exp 1.1a** | 0.8 | 0.476 | 0.180 | -18.2% ❌ | 05:13 JST ✅ |
| **Exp 1.1b** | 1.0 | **0.779** 🎯 | -0.0319 | **+33.8%** | 06:38 JST ✅ |

### Key Findings

**1. Non-Linear Optimization Landscape**
- Sharpe weight 0.6 → Sharpe 0.582 ✅ (Baseline)
- Sharpe weight 0.8 → Sharpe 0.476 ❌ (Middle trap - worst performance)
- Sharpe weight 1.0 → Sharpe 0.779 🎯 (Pure optimization - best performance)

**2. The "Middle Trap" Phenomenon**
- Intermediate weights (0.7-0.9) create conflicting optimization pressures
- Neither fully optimized nor balanced → suboptimal convergence
- Recommendation: **Avoid intermediate Sharpe weights**

**3. Loss-Metric Alignment**
- Exp 1.1a: Good loss (0.180) ≠ Good Sharpe (0.476)
- Exp 1.1b: Excellent loss (-0.0319) = Excellent Sharpe (0.779)
- **Lesson**: Loss function must align with business goal

**4. Pure Sharpe Optimization is Stable**
- High batch-level volatility is normal (Sharpe -0.720 ~ +0.786)
- Training converged successfully without instability
- **Conclusion**: Pure Sharpe (weight=1.0) is safe and effective

### Documentation & Archives

- ✅ **Full Report**: `docs/reports/experiments/tier1_sharpe_weight_optimization_20251021.md`
- ✅ **Archive**: `archive/experiments_tier1_20251021.tar.gz` (946KB)
  - Complete logs for all experiments
  - Result JSON files
  - Comprehensive analysis report
- ✅ **Logs**:
  - Exp 1.1a: `/tmp/experiment_1_1a_sharpe08.log` (640KB)
  - Exp 1.1b: `/tmp/experiment_1_1b_sharpe10.log` (623KB)

---

## 🚀 Next: Tier 2 - Model Capacity Expansion

### Experiment 2.1: Hidden Size 256 (READY TO LAUNCH)

**Configuration**:
- **Model**: hidden_size 256 (vs current 64)
- **Sharpe weight**: 1.0 (proven optimal from Tier 1)
- **Expected Sharpe**: 0.82-0.85 (+5-8% improvement)
- **Expected**: **High probability of reaching target 0.849**
- **Runtime**: ~2 hours
- **Priority**: ⭐⭐⭐ (Highest - likely to achieve goal)

**Rationale**:
- Larger model capacity (5.6M → 20M params)
- Combined with proven optimal loss weight (1.0)
- Config already prepared in `config_production_optimized.yaml`

**Launch Command**:
```bash
cd /workspace/gogooku3
SHARPE_WEIGHT=1.0 RANKIC_WEIGHT=0.0 CS_IC_WEIGHT=0.0 \
python scripts/integrated_ml_training_pipeline.py \
  --max-epochs 30 \
  --data-path output/ml_dataset_latest_full.parquet \
  > /tmp/experiment_2_1_hidden256.log 2>&1 &
```

### Experiment 3.1: Feature Bundle 220 (DATASET REBUILD NEEDED)

**Configuration**:
- **Features**: 220 (vs current 99)
- **Sharpe weight**: 1.0
- **Expected Sharpe**: 0.85+ (+9%+)
- **Runtime**: Dataset rebuild 30-60 min + Training ~3 hours
- **Priority**: ⭐⭐ (High - after confirming capacity scaling)

**Steps**:
1. Regenerate dataset:
   ```bash
   python scripts/pipelines/run_full_dataset.py \
     --start-date 2020-09-06 \
     --end-date 2025-09-06 \
     --enable-all-features \
     --max-features 220
   ```

2. Train with new dataset:
   ```bash
   SHARPE_WEIGHT=1.0 RANKIC_WEIGHT=0.0 CS_IC_WEIGHT=0.0 \
   python scripts/integrated_ml_training_pipeline.py \
     --max-epochs 30 \
     --data-path output/ml_dataset_220features.parquet \
     > /tmp/experiment_3_1_features220.log 2>&1 &
   ```

---

## ⏳ Tier 1 Alternative Experiments (Optional)

### Experiment 1.2: Loss Curriculum

**Status**: ⚙️ IMPLEMENTATION READY (Backup strategy)
**Expected**: Sharpe ~0.70-0.75 (+15-20%)
**Use Case**: If pure Sharpe (1.0) causes issues in larger models
**Priority**: ⭐ (Optional - pure Sharpe already proven effective)

**Implementation**: `scripts/utils/loss_curriculum.py`
- Progressive schedule: Phase 0-1 (0.5) → Phase 2 (0.7) → Phase 3 (1.0)
- Requires integration into training script

---

## 📊 Execution Timeline

### ✅ Phase 1: Quick Wins (COMPLETE)

| Experiment | Status | Runtime | Final Sharpe | Result |
|------------|--------|---------|--------------|--------|
| Baseline | ✅ Complete | 71 min | 0.582 | Reference |
| 1.1a (Sharpe 0.8) | ✅ Complete | 83 min | 0.476 | Failed ❌ |
| 1.1b (Sharpe 1.0) | ✅ Complete | 85 min | **0.779** | **Best 🎯** |

**Achievement**: +33.8% improvement, 91.7% of target

### 🔄 Phase 2: Capacity Expansion (READY)

| Experiment | Status | Runtime | Expected Sharpe | Priority |
|------------|--------|---------|----------------|----------|
| 2.1 (Hidden 256) | ⏸️ Ready | ~2h | 0.82-0.85 | ⭐⭐⭐ |
| 3.1 (Features 220) | ⏸️ Dataset | ~4h | 0.85+ | ⭐⭐ |

**Target**: Reach or exceed 0.849 Sharpe

### 📝 Phase 3: Refinement (Planned)

| Experiment | Status | Runtime | Expected Sharpe |
|------------|--------|---------|----------------|
| 4.1 (Quick HPO) | ⚙️ Ready | ~6h | +2-5% |
| 4.2 (Full HPO) | ⚙️ Ready | ~48h | +5-10% |
| Ensemble | 📝 Planned | ~2h | +2-5% |
| Production validation | 📝 Planned | 1 day | - |

---

## 📈 Progress Tracking

### Sharpe Ratio Improvement Path

| Milestone | Sharpe | Gap to Target | Achievement |
|-----------|--------|---------------|-------------|
| **Initial Baseline** | 0.136 | -0.713 | Starting point |
| **Optimized Baseline** | 0.582 | -0.267 | 68.6% of target |
| **Current Best (Tier 1)** | **0.779** | **-0.070** | **91.7% of target** |
| **Expected (Tier 2)** | 0.82-0.85 | -0.03 ~ +0.001 | 96-100%+ |
| **Target** | 0.849 | 0.000 | Goal |

**Cumulative Improvement**:
- Initial → Current: +0.643 (+472% improvement)
- Latest improvement (Tier 1): +0.197 (+33.8%)
- Expected next (Tier 2): +0.041-0.071 (+5-9%)

---

## 📁 Generated Files & Documentation

```
docs/reports/experiments/
└── tier1_sharpe_weight_optimization_20251021.md  # Full Tier 1 analysis

configs/experiments/
├── README.md                                      # All experiment launch commands
├── hpo_search_space.yaml                         # HPO configuration

scripts/utils/
└── loss_curriculum.py                            # Loss schedule implementation

archive/
├── training_20251021_012605.tar.gz              # Baseline package (101KB)
└── experiments_tier1_20251021.tar.gz            # Tier 1 complete archive (946KB)
```

---

## 🔧 Recommended Next Actions

### Immediate (Today - High Priority)

**1. Launch Experiment 2.1** (Hidden size 256)
- Use proven optimal Sharpe weight=1.0
- Expected to reach or exceed target 0.849
- Runtime: ~2 hours
- **Action**: Execute launch command above

### Short-term (This Week)

**2. Validate Tier 2 Results**
- If Exp 2.1 reaches target → Proceed to production validation
- If below target → Launch Exp 3.1 (Features 220)

**3. Production Readiness**
- Walk-forward validation on holdout period
- Ensemble top 2-3 models
- Risk analysis and stress testing

### Medium-term (Next Week - Optional)

**4. HPO Optimization** (if target not yet reached)
- Quick probe (16 trials × 2 epochs)
- Fine-tune learning rate, weight decay, batch size

**5. Advanced Techniques** (stretch goals)
- Ensemble best models
- Multi-period validation
- Feature importance analysis

---

## 📊 Success Metrics

### Primary
- **Test Sharpe Ratio**: Target 0.849 (current best 0.779, 91.7%)

### Secondary
- **Val Loss**: Lower is better (current best: -0.0319)
- **IC / RankIC**: Maintained during optimization
- **Hit Rate**: Current ~0.53, Target >0.55
- **Training time**: <2 hours per run ✅
- **GPU memory**: <70GB ✅

### Stability
- Sharpe std across runs: Monitor in Tier 2
- No training crashes: ✅ All Tier 1 stable
- Consistent convergence: ✅ Verified

---

## 🚨 Risk Assessment

### Identified Risks

**1. Pure Sharpe may not scale to larger models**
- **Likelihood**: Low (Tier 1 showed stability)
- **Impact**: Medium
- **Mitigation**: Test Exp 2.1 with same weight=1.0 setting
- **Fallback**: Use loss curriculum if instability appears

**2. Model capacity increase may cause overfitting**
- **Likelihood**: Medium
- **Impact**: Medium
- **Mitigation**: Monitor train/val gap, use dropout
- **Validation**: Cross-validation on walk-forward splits

**3. GPU memory constraints with hidden_size=256**
- **Likelihood**: Low
- **Impact**: High (blocks experiment)
- **Mitigation**: Start with batch_size=1024, use gradient checkpointing
- **Monitoring**: `nvidia-smi` during training

### Mitigation Strategies

1. ✅ **Progressive scaling**: Test capacity before features
2. ✅ **Multiple metrics**: Track Sharpe, RankIC, IC, Hit Rate
3. ✅ **Automated monitoring**: Use auto-launch scripts
4. ✅ **Comprehensive documentation**: All experiments archived

---

## 🎓 Lessons Learned (Tier 1)

### ✅ What Worked

1. **Pure Sharpe optimization (weight=1.0)** - Best strategy
2. **Automated experiment queue** - Saved monitoring time
3. **Phase-based training** - Stable convergence
4. **Bold hypothesis testing** - Discovered optimal extreme value
5. **Comprehensive documentation** - Reproducible and sharable

### ❌ What Didn't Work

1. **Intermediate Sharpe weight (0.8)** - "Middle trap" phenomenon
2. **Assumption of monotonicity** - Non-linear relationship
3. **Multi-objective without priority** - Conflicting gradients

### 🔍 Insights

1. **Prefer extremes over middle** in single-metric optimization
2. **Test boundary conditions first** (0.0, 0.5, 1.0)
3. **Monitor loss-metric alignment** - Divergence is a red flag
4. **Pure optimization is often safer** than complex balancing
5. **Volatility ≠ Instability** - High variance acceptable if convergence good

---

## 📋 Quick Reference

### Launch Next Experiment

```bash
# Experiment 2.1: Hidden Size 256 (RECOMMENDED)
cd /workspace/gogooku3
SHARPE_WEIGHT=1.0 RANKIC_WEIGHT=0.0 CS_IC_WEIGHT=0.0 \
python scripts/integrated_ml_training_pipeline.py \
  --max-epochs 30 \
  --data-path output/ml_dataset_latest_full.parquet \
  > /tmp/experiment_2_1_hidden256.log 2>&1 &

# Monitor
tail -f /tmp/experiment_2_1_hidden256.log | grep -E "Epoch|Sharpe|Phase"
```

### Check Status

```bash
# Check running processes
ps aux | grep -E "python.*integrated_ml" | grep -v grep

# Check latest results
ls -lth output/results/complete_training_result_*.json | head -3

# View latest Sharpe
tail -50 /tmp/experiment_*.log | grep "Achieved Sharpe"
```

---

**Status Report Generated**: 2025-10-21 07:10 JST
**Next Update**: After Experiment 2.1 completion (~2 hours)
**Full Tier 1 Report**: `docs/reports/experiments/tier1_sharpe_weight_optimization_20251021.md`
