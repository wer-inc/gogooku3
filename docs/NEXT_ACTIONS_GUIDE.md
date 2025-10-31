# Next Actions Quick-Start Guide

**Date**: 2025-10-31
**Status**: Systems validated, ready for next phase
**Purpose**: Quick reference for resuming work on either APEX or ATFT

---

## 🚀 Path 1: Deploy APEX-Ranker (Production-Ready)

### Prerequisites Met ✅
- [x] 22-month validation complete (Sharpe 1.95)
- [x] Panel cache fix verified (100% trade consistency)
- [x] Parquet data ready for ingestion
- [x] Documentation complete

### Quick Deploy (15 Minutes)

```bash
# 1. Verify data files
ls -lh results/rolling_retrain_fixed/rolling_metrics_summary.parquet
ls -lh cache/panel_prod/*.pkl

# 2. Load data for analysis
python3 << 'EOF'
import polars as pl
df = pl.read_parquet('results/rolling_retrain_fixed/rolling_metrics_summary.parquet')
print(f"✅ Loaded {len(df)} months of data")
print(f"Coverage: {df['month'].min()} to {df['month'].max()}")
print(f"Avg Sharpe: {df['sharpe'].mean():.2f}")
EOF

# 3. Review documentation
cat docs/ROLLING_RETRAIN_ANALYSIS.md | less
cat docs/PROJECT_STATUS_SUMMARY.md | less
```

### Optional: Create Monitoring Dashboard

**If you want automated dashboards**, I can help create:
1. **Performance Dashboard** - Monthly returns, Sharpe, drawdowns
2. **Alert System** - Email/Slack on Sharpe < 0 or TX cost > 0.6%
3. **Weekly Reports** - Automated PDF/email summaries

**Time Required**: ~2 hours to set up, then automated

**Just ask**: "Create APEX monitoring dashboard" when ready

---

## 🔬 Path 2: ATFT Diagnostic Investigation (30-45 Minutes)

### Prerequisites Met ✅
- [x] Gradient fix validated (encoder active)
- [x] 50-epoch plateau confirmed (Sharpe 0.0818)
- [x] Diagnostic plan documented

### Quick Diagnostic (30-45 Minutes)

```bash
# Step 1: Extract train/val loss curves (10 min)
grep -E "Epoch [0-9]+/50.*Loss" _logs/training/prod_validation_50ep_*.log > /tmp/loss_curves.txt

# Analyze for:
# - Val loss plateaus early → Overfitting
# - Both plateau → Capacity issue
# - Losses decrease → Loss function misalignment

# Step 2: Check learning rate progression (5 min)
grep "LR=" _logs/training/prod_validation_50ep_*.log | awk -F'LR=' '{print $2}' | cut -d' ' -f1 > /tmp/lr_progression.txt

# Analyze for:
# - LR < 1e-5 by epoch 20 → Too aggressive decay
# - LR constant → Scheduler not working

# Step 3: Review loss function configuration (10 min)
grep -E "PHASE_LOSS_WEIGHTS|SHARPE_WEIGHT|IC_WEIGHT|RANKIC_WEIGHT" _logs/training/prod_validation_50ep_*.log

# Check if Sharpe component is active and weighted appropriately

# Step 4: Gradient attribution analysis (10 min)
grep "GRAD-MONITOR.*temporal_encoder" _logs/training/prod_validation_50ep_*.log | head -20 > /tmp/grads_early.txt
grep "GRAD-MONITOR.*temporal_encoder" _logs/training/prod_validation_50ep_*.log | tail -20 > /tmp/grads_late.txt

# Compare early vs late epoch gradients:
# - Decreasing → Encoder contribution fading
# - Stable but low → Weak contribution
```

### Automated Diagnostic Script

**If you want automation**, I can create a comprehensive diagnostic script that:
1. Extracts all metrics automatically
2. Generates visual plots (loss curves, LR schedule, gradient norms)
3. Produces a diagnosis report with recommendations

**Just ask**: "Create ATFT diagnostic script" when ready

---

## 🔧 Path 3: ATFT Configuration Experiments (2-4 Hours)

### Based on Diagnostic Results

#### Experiment A: Adjust Learning Rate Schedule
```bash
# If diagnostics show LR decaying too fast
PLATEAU_PATIENCE=10 \
PLATEAU_FACTOR=0.5 \
python scripts/integrated_ml_training_pipeline.py \
  --data-path output/datasets/ml_dataset_latest_full.parquet \
  --max-epochs 30 \
  train.batch.train_batch_size=2048 \
  train.trainer.precision=bf16-mixed

# Success criterion: Sharpe > 0.10 by epoch 20
```

#### Experiment B: Tune Loss Weights
```bash
# If diagnostics show Sharpe component underweighted
SHARPE_WEIGHT=0.5 \
CS_IC_WEIGHT=0.1 \
RANKIC_WEIGHT=0.1 \
python scripts/integrated_ml_training_pipeline.py \
  --data-path output/datasets/ml_dataset_latest_full.parquet \
  --max-epochs 30 \
  train.batch.train_batch_size=2048 \
  train.trainer.precision=bf16-mixed

# Success criterion: Sharpe > 0.10 by epoch 20
```

#### Experiment C: Increase Model Capacity
```bash
# If diagnostics show capacity limitation
python scripts/integrated_ml_training_pipeline.py \
  --data-path output/datasets/ml_dataset_latest_full.parquet \
  --max-epochs 30 \
  model.hidden_size=512 \
  train.batch.train_batch_size=1024 \
  train.trainer.precision=bf16-mixed

# Note: Reduced batch size to fit larger model in GPU memory
# Success criterion: Sharpe > 0.10 by epoch 20
```

#### Experiment D: Different Optimizer Settings
```bash
# If diagnostics show optimizer issue
python scripts/integrated_ml_training_pipeline.py \
  --data-path output/datasets/ml_dataset_latest_full.parquet \
  --max-epochs 30 \
  train.optimizer.lr=1e-3 \
  train.batch.train_batch_size=2048 \
  train.trainer.precision=bf16-mixed

# Success criterion: Sharpe > 0.10 by epoch 20
```

---

## 📊 Path 4: Transaction Cost Optimization (1 Week)

### APEX-Ranker Cost Reduction Experiment

**Current**: 5.46%/year (weekly rebalancing, 5x/month)
**Target**: <2%/year (monthly rebalancing)

```bash
# Step 1: Backtest monthly rebalancing (6 months: 2024-07 to 2024-12)
# This would require modifying the rebalancing frequency in the rolling retrain script
# Location: apex-ranker/scripts/rolling_retrain.py

# Step 2: Compare results
# Weekly (current):  Net return after 5.46% annual TX cost
# Monthly (proposed): Net return after ~1% annual TX cost

# Step 3: Decision
# If net return (monthly) > net return (weekly) → Deploy monthly
# If net return (monthly) < net return (weekly) → Keep weekly
```

**If you want this experiment**, I can:
1. Modify the rebalancing frequency parameter
2. Run backtest comparison (6 months)
3. Generate side-by-side comparison report

**Just ask**: "Run APEX TX cost experiment" when ready

---

## 🎯 Decision Matrix

### Which Path Should You Choose?

| Goal | Recommended Path | Time Required | Expected Outcome |
|------|-----------------|---------------|------------------|
| **Deploy working system now** | Path 1: APEX Deploy | 15 min | Production trading system |
| **Understand ATFT plateau** | Path 2: ATFT Diagnostics | 30-45 min | Root cause identified |
| **Improve ATFT performance** | Path 3: ATFT Experiments | 2-4 hours | Sharpe > 0.10 validation |
| **Reduce APEX costs** | Path 4: TX Optimization | 1 week | 3-4% annual savings |
| **Do everything** | Paths 1→2→3→4 | 2-3 weeks | Optimized dual system |

### My Recommendation (Priority Order)

1. **Path 1 (15 min)**: Deploy APEX-Ranker
   - Proven performance (Sharpe 1.95)
   - 22-month validation
   - Generate returns while working on ATFT

2. **Path 2 (30-45 min)**: ATFT Diagnostics
   - Identify root cause of plateau
   - Low time investment
   - Informs next steps

3. **Path 3 (2-4 hours)**: ATFT Experiments
   - Only if diagnostics show clear fix
   - Success: Sharpe > 0.10 → Proceed to 120 epochs
   - Failure: Consider architecture redesign

4. **Path 4 (1 week)**: TX Cost Optimization
   - Optimize working APEX system
   - Potential 3-4% annual improvement
   - Low risk (just frequency change)

---

## 🚦 Quick Commands Reference

### Check System Status
```bash
# APEX: Verify data files
ls -lh results/rolling_retrain_fixed/*.parquet
ls -lh cache/panel_prod/*.pkl

# ATFT: Check training logs
ls -lh _logs/training/prod_validation_*.log
grep "Achieved Sharpe" _logs/training/prod_validation_50ep_*.log

# Review documentation
ls -lh docs/*.md
```

### Monitor Running Training (if any)
```bash
# Check if ATFT training still running
ps aux | grep integrated_ml_training_pipeline | grep -v grep

# Quick status
./scripts/training_status.sh

# Live monitoring
./scripts/monitor_training.sh
```

### Review Results
```bash
# APEX summary statistics
python3 << 'EOF'
import polars as pl
df = pl.read_parquet('results/rolling_retrain_fixed/rolling_metrics_summary.parquet')
print(f"Months: {len(df)}")
print(f"Avg Sharpe: {df['sharpe'].mean():.2f}")
print(f"Avg Return: {df['total_return_pct'].mean():.2f}%")
print(f"Win Rate: {(df['total_return_pct'] > 0).sum() / len(df) * 100:.1f}%")
EOF

# ATFT validation results
cat docs/VALIDATION_RESULTS_50EP.md | grep -A 10 "Key Finding"
```

---

## 📚 Documentation Quick Reference

### APEX-Ranker
- **Performance Analysis**: `docs/ROLLING_RETRAIN_ANALYSIS.md`
- **Fix Verification**: `docs/fixes/rolling_retrain_zero_trade_fix_verification.md`
- **Diagnostics**: `docs/diagnostics/rolling_retrain_zero_trade_diagnosis.md`
- **Operations**: `docs/operations/rolling_retrain_restart_plan.md`

### ATFT-GAT-FAN
- **50-Epoch Analysis**: `docs/VALIDATION_RESULTS_50EP.md` ⚠️ READ THIS FIRST
- **Gradient Fix Guide**: `docs/GRADIENT_FIX_SUMMARY.md`
- **Quick Reference**: `docs/QUICK_REFERENCE.txt`
- **Session Summary**: `docs/VALIDATION_SESSION_20251031.md`

### Cross-Project
- **Strategic Overview**: `docs/PROJECT_STATUS_SUMMARY.md` ⭐ START HERE
- **This Guide**: `docs/NEXT_ACTIONS_GUIDE.md`

---

## 🎬 How to Resume

### Simple Commands to Get Started

**Option 1: Deploy APEX Now**
```bash
cat docs/ROLLING_RETRAIN_ANALYSIS.md
# Review results, then ask: "Help me deploy APEX-Ranker pipeline"
```

**Option 2: Fix ATFT Plateau**
```bash
cat docs/VALIDATION_RESULTS_50EP.md
# Review diagnostics plan, then ask: "Run ATFT diagnostic investigation"
```

**Option 3: Optimize APEX Costs**
```bash
# Ask: "Run APEX transaction cost optimization experiment"
```

**Option 4: Get Automation**
```bash
# Ask: "Create monitoring dashboard for APEX-Ranker"
# Or: "Create ATFT diagnostic automation script"
```

---

## ✅ What's Already Done

### APEX-Ranker ✅
- [x] Panel cache fix implemented and verified
- [x] 22-month rolling retrain completed
- [x] Results exported to parquet
- [x] Documentation complete
- [x] Production-ready

### ATFT-GAT-FAN ✅
- [x] Gradient fix implemented and validated
- [x] 5/20/50 epoch validations completed
- [x] Plateau identified and documented
- [x] Diagnostic plan created
- [x] Monitoring scripts created

### Infrastructure ✅
- [x] Comprehensive documentation (10+ files)
- [x] Monitoring scripts (training_status.sh, monitor_training.sh)
- [x] Data export (CSV + Parquet)
- [x] Cross-project strategic analysis

---

## 🎯 Success Criteria Recap

### APEX-Ranker (Already Met ✅)
- [x] 100% trade consistency (22/22 months)
- [x] Sharpe > 1.0 (achieved: 1.95)
- [x] Positive returns (achieved: +2.83%/mo)
- [x] Win rate > 60% (achieved: 68%)
- [ ] TX cost < 2%/year (current: 5.46%) ← Optimization opportunity

### ATFT-GAT-FAN (Partially Met ⚠️)
- [x] Gradient flow active
- [x] Zero degeneracy
- [x] Training stability
- [ ] Sharpe progression (stuck at 0.0818) ← Needs investigation
- [ ] Target Sharpe 0.849 (gap: 10.4x) ← Needs configuration tuning

---

## 💡 Just Ask When Ready

I'm ready to help with any of these paths. Just say:

- **"Deploy APEX"** → I'll help set up production pipeline
- **"Diagnose ATFT"** → I'll run the 30-45 min investigation
- **"Create dashboard"** → I'll build monitoring automation
- **"Optimize TX costs"** → I'll run the rebalancing experiment
- **"Run experiment [A/B/C/D]"** → I'll execute configuration tests

All systems validated, documented, and ready to go! 🚀

---

**Generated**: 2025-10-31 01:45 UTC
**Status**: Ready for next phase
**Quick Start**: Read `docs/PROJECT_STATUS_SUMMARY.md` first, then choose your path
