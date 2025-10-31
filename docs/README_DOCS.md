# Documentation Index

**Last Updated**: 2025-10-31
**Purpose**: Navigate all project documentation

---

## 🚀 Start Here

**New to the project?** Read these in order:
1. **`PROJECT_STATUS_SUMMARY.md`** - Strategic overview of both systems
2. **`NEXT_ACTIONS_GUIDE.md`** - Quick-start guide for next steps
3. Choose your focus: APEX (deploy) or ATFT (optimize)

---

## 📊 APEX-Ranker Documentation

### Production Results ✅
- **`ROLLING_RETRAIN_ANALYSIS.md`** - Full 22-month performance analysis
  - Coverage: 2024-01 to 2025-10
  - Performance: Sharpe 1.95, Return +2.83%/mo
  - Status: Production-ready

### Technical Details
- **`diagnostics/rolling_retrain_zero_trade_diagnosis.md`** - Root cause analysis
- **`fixes/rolling_retrain_zero_trade_fix_verification.md`** - Fix validation
- **`operations/rolling_retrain_restart_plan.md`** - Operations playbook

### APEX-Ranker Core Docs (apex-ranker/)
- **`EXPERIMENT_STATUS.md`** - Model status and experiment history
- **`INFERENCE_GUIDE.md`** - Production usage guide
- **`README.md`** - Project overview
- **`docs/BACKTEST_COMPARISON_2023_2025.md`** - Enhanced vs pruned model comparison

---

## 🔬 ATFT-GAT-FAN Documentation

### Validation Results
- **`VALIDATION_RESULTS_50EP.md`** ⚠️ **CRITICAL** - Plateau analysis + diagnostics
  - Sharpe plateau at 0.0818
  - Root cause investigation plan
  - Configuration experiments

- **`VALIDATION_RESULTS_20EP.md`** - 20-epoch validation
  - Baseline results
  - Gradient flow confirmation

- **`VALIDATION_SESSION_20251031.md`** - Session summary
  - Quick overview of validation runs

### Production Guides
- **`GRADIENT_FIX_SUMMARY.md`** - Production deployment guide
  - Core fixes implementation
  - Environment variable tuning
  - Monitoring and troubleshooting

- **`QUICK_REFERENCE.txt`** - One-page cheat sheet
  - Commands and parameters
  - Warning signs and troubleshooting
  - Production deployment checklist

### Training Status (archived)
- **`PROD_VALIDATION_STATUS.md`** - Real-time monitoring guide
  - Process/GPU status tracking
  - Expected behavior reference

---

## 🔧 Scripts & Tools

### Monitoring Scripts
- **`scripts/monitor_training.sh`** - Live training dashboard
  - Auto-refresh every 30 seconds
  - Tracks progress, gradients, degeneracy, GPU

- **`scripts/training_status.sh`** - Quick status snapshot
  - One-time execution
  - Process, metrics, GPU status

### APEX-Ranker Scripts (apex-ranker/scripts/)
- **`inference_v0.py`** - Production inference CLI
- **`train_v0.py`** - Model training
- **`backtest_smoke_test.py`** - Backtest driver (Phase 3.3+)
- **`monitor_predictions.py`** - Prediction logging & monitoring

---

## 📁 Data Files

### APEX-Ranker Results
```
results/rolling_retrain_fixed/
├── rolling_metrics_summary.parquet  # 22 months, 4.3KB
├── rolling_metrics_summary.csv      # Same data, 2.5KB
├── rolling_metrics_summary.json     # Full JSON
└── [monthly results]                # 2024-01/ to 2025-10/
```

### APEX-Ranker Cache
```
cache/panel_prod/
└── panel_*.pkl                      # Reusable panel cache files
```

### ATFT Training Logs
```
_logs/training/
├── prod_validation_20ep_*.log       # 20-epoch validation
├── prod_validation_50ep_*.log       # 50-epoch validation
└── *.pid                            # Process ID files
```

### ATFT Results
```
output/results/
└── complete_training_result_*.json  # Training results
```

---

## 🎯 Quick Reference by Task

### I want to...

**Deploy APEX-Ranker**
→ Read: `ROLLING_RETRAIN_ANALYSIS.md` + `NEXT_ACTIONS_GUIDE.md` (Path 1)

**Understand ATFT plateau**
→ Read: `VALIDATION_RESULTS_50EP.md` (Section: Plateau Analysis)

**Fix ATFT performance**
→ Read: `VALIDATION_RESULTS_50EP.md` (Section: Required Investigations)
→ Follow: `NEXT_ACTIONS_GUIDE.md` (Path 2 → Path 3)

**Monitor training**
→ Use: `scripts/training_status.sh` or `scripts/monitor_training.sh`

**Check gradient fix**
→ Read: `GRADIENT_FIX_SUMMARY.md` + `VALIDATION_RESULTS_20EP.md`

**Compare both systems**
→ Read: `PROJECT_STATUS_SUMMARY.md`

**Get production commands**
→ Read: `QUICK_REFERENCE.txt` (ATFT) or `NEXT_ACTIONS_GUIDE.md` (both)

**Review operations**
→ Read: `operations/rolling_retrain_restart_plan.md` (APEX)

---

## 📊 Performance Summary (Quick Reference)

### APEX-Ranker (22 Months: 2024-01 to 2025-10)
```
✅ Production-Ready

Trades: 5,097 total (232/month avg)
Return: +2.83%/month (2025: +4.03%)
Sharpe: 1.95 (2025: 2.67)
Win Rate: 68% (2025: 80%)
TX Cost: 5.46%/year (optimize to <2%)
```

### ATFT-GAT-FAN (50 Epochs Validated)
```
⚠️ Needs Optimization

Gradient Fix: ✅ Validated (encoder active)
Degeneracy: ✅ 0 resets (excellent)
Training: ✅ Stable (no crashes)
Sharpe: ⚠️ Plateau at 0.0818 (vs target 0.849)
Next: Diagnostic investigation required
```

---

## 🔄 Document Status

### Recently Created (2025-10-31)
- ✅ `ROLLING_RETRAIN_ANALYSIS.md` - APEX full analysis
- ✅ `VALIDATION_RESULTS_50EP.md` - ATFT plateau analysis
- ✅ `PROJECT_STATUS_SUMMARY.md` - Cross-project overview
- ✅ `NEXT_ACTIONS_GUIDE.md` - Quick-start guide
- ✅ `README_DOCS.md` - This index

### Previously Created (2025-10-30)
- ✅ `GRADIENT_FIX_SUMMARY.md` - ATFT production guide
- ✅ `VALIDATION_RESULTS_20EP.md` - 20-epoch validation
- ✅ `QUICK_REFERENCE.txt` - ATFT cheat sheet
- ✅ `VALIDATION_SESSION_20251031.md` - Session summary
- ✅ `PROD_VALIDATION_STATUS.md` - Monitoring guide

### Legacy (Archived)
- `SESSION_SUMMARY.md` (2024-10-16) - Old session notes
- Various experimental docs in `docs/diagnostics/`, `docs/fixes/`

---

## 🎓 Reading Paths

### Path 1: Quick Overview (15 Minutes)
1. `PROJECT_STATUS_SUMMARY.md` (10 min)
2. `NEXT_ACTIONS_GUIDE.md` (5 min)
3. **Decision**: Deploy APEX or optimize ATFT?

### Path 2: APEX Deep Dive (1 Hour)
1. `ROLLING_RETRAIN_ANALYSIS.md` (30 min)
2. `diagnostics/rolling_retrain_zero_trade_diagnosis.md` (15 min)
3. `fixes/rolling_retrain_zero_trade_fix_verification.md` (15 min)
4. **Result**: Full understanding of APEX system

### Path 3: ATFT Deep Dive (1 Hour)
1. `VALIDATION_RESULTS_50EP.md` (30 min)
2. `GRADIENT_FIX_SUMMARY.md` (20 min)
3. `QUICK_REFERENCE.txt` (10 min)
4. **Result**: Ready to run diagnostics/experiments

### Path 4: Production Deployment (2 Hours)
1. `PROJECT_STATUS_SUMMARY.md` (10 min)
2. `ROLLING_RETRAIN_ANALYSIS.md` (30 min) - APEX
3. `GRADIENT_FIX_SUMMARY.md` (20 min) - ATFT
4. `NEXT_ACTIONS_GUIDE.md` (10 min)
5. Execution: Deploy + configure monitoring (50 min)
6. **Result**: Production system running

---

## 💡 Common Questions

**Q: Which system should I deploy first?**
A: APEX-Ranker. It's validated over 22 months with Sharpe 1.95. ATFT needs optimization first.

**Q: What's wrong with ATFT?**
A: Gradient fix works perfectly, but performance plateaus at Sharpe 0.0818 (vs target 0.849). Needs configuration tuning.

**Q: How do I fix the ATFT plateau?**
A: Follow diagnostics in `VALIDATION_RESULTS_50EP.md`, then run experiments in `NEXT_ACTIONS_GUIDE.md` (Path 3).

**Q: Can I use both systems together?**
A: Yes, eventually. Deploy APEX now, optimize ATFT in parallel, then combine for portfolio diversification.

**Q: Where are the actual data files?**
A: `results/rolling_retrain_fixed/*.parquet` (APEX), `_logs/training/*.log` (ATFT), `output/results/*.json` (ATFT results)

**Q: How do I monitor training?**
A: Use `scripts/training_status.sh` for quick check or `scripts/monitor_training.sh` for live dashboard.

**Q: What documentation should I share with my team?**
A: Start with `PROJECT_STATUS_SUMMARY.md` and `QUICK_REFERENCE.txt`. Full details in respective analysis docs.

---

## 🚦 Status Indicators

### ✅ Production-Ready
- APEX-Ranker panel cache fix
- ATFT gradient flow restoration
- Monitoring infrastructure
- Documentation complete

### ⚠️ Needs Attention
- ATFT Sharpe plateau (requires diagnostics)
- APEX transaction costs (5.46%/year → optimize to <2%)

### 📋 Planned
- ATFT configuration experiments (after diagnostics)
- APEX monthly rebalancing test
- Monitoring dashboard automation
- Ensemble strategy (APEX + ATFT)

---

## 📞 Support

**If you need help**:
1. Check `NEXT_ACTIONS_GUIDE.md` for quick-start commands
2. Review relevant analysis doc for detailed context
3. Ask specific questions with context

**Common requests**:
- "Deploy APEX" → Uses `NEXT_ACTIONS_GUIDE.md` Path 1
- "Diagnose ATFT" → Uses `NEXT_ACTIONS_GUIDE.md` Path 2
- "Create dashboard" → Custom automation setup
- "Run experiment" → Uses `NEXT_ACTIONS_GUIDE.md` Path 3

---

**Last Updated**: 2025-10-31 01:50 UTC
**Total Documents**: 15+ (including subdirectories)
**Coverage**: Full dual-system documentation
**Status**: Complete and ready for production
