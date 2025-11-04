# Quick Start: Cross-Validation Training (OOM-Safe)

## 🎯 TL;DR - What You Need to Do Now

Fold1が完走したので、次は**健全性チェック → Fold2-5の安全な実行**です。

### ⚡ 3-Step Workflow (推奨)

```bash
# Step 1: Fold1のブレンド＆健全性チェック（5分）
python apex-ranker/scripts/average_checkpoints.py \
  models/shortfocus_fold1_ema_epoch3.pt \
  models/shortfocus_fold1_ema_epoch6.pt \
  models/shortfocus_fold1_ema_epoch10.pt \
  --output models/shortfocus_fold1_blended.pt

python apex-ranker/scripts/backtest_smoke_test.py \
  --checkpoint models/shortfocus_fold1_blended.pt \
  --config apex-ranker/configs/v0_base.yaml \
  --log-level INFO

# Step 2: 健全性がGreenなら、Fold2-5を安全実行（4-5時間）
bash apex-ranker/scripts/train_folds_sequential.sh 2 5

# Step 3: 全Foldブレンド＆バックテスト
# (自動化スクリプトは下記参照)
```

---

## 📋 Step-by-Step Instructions

### Step 1: Fold1 Health Check (5-10 minutes)

#### 1a. Create blended checkpoint
```bash
cd /workspace/gogooku3

python apex-ranker/scripts/average_checkpoints.py \
  models/shortfocus_fold1_ema_epoch3.pt \
  models/shortfocus_fold1_ema_epoch6.pt \
  models/shortfocus_fold1_ema_epoch10.pt \
  --output models/shortfocus_fold1_blended.pt
```

**Expected output**:
```
Loading 3 checkpoints...
  ✓ models/shortfocus_fold1_ema_epoch3.pt
  ✓ models/shortfocus_fold1_ema_epoch6.pt
  ✓ models/shortfocus_fold1_ema_epoch10.pt
Averaged 3 checkpoints
Saved to: models/shortfocus_fold1_blended.pt
```

#### 1b. Run smoke backtest
```bash
python apex-ranker/scripts/backtest_smoke_test.py \
  --checkpoint models/shortfocus_fold1_blended.pt \
  --config apex-ranker/configs/v0_base.yaml \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --top-k 50 \
  --horizon 5 \
  --log-level INFO
```

#### 1c. Check key metrics

**Green (健全) indicators**:
- ✅ `candidate_count > 0` on all days (選定株数が常に存在)
- ✅ `fallback_rate < 20%` (Kの選定が安定)
- ✅ `total_return > 0` (コスト込みでプラス)
- ✅ `h5_delta_p_at_k_pos > 0` (ランダムより良い)

**Yellow (要注意) indicators**:
- ⚠️ `fallback_rate 20-40%` → K選定が不安定
- ⚠️ `h5_delta_p_at_k_pos` near 0 → ランダムと同等

**Red (要修正) indicators**:
- ❌ `candidate_count = 0` on multiple days → ゼロ選定問題
- ❌ `total_return < 0` → コスト負け
- ❌ `h5_delta_p_at_k_pos < 0` → ランダムより悪い

**Decision**:
- **Green** → Proceed to Step 2
- **Yellow** → Tune `k_ratio` (0.05 → 0.03) or `tau` (60 → 90) on Fold1, then re-check
- **Red** → Debug model/labels before expanding

---

### Step 2: Train Folds 2-5 (OOM-Safe)

**Option A: Sequential (推奨 - 100% safe)**

```bash
# Default: Folds 2-5, 12 epochs, EMA at 3/6/10
bash apex-ranker/scripts/train_folds_sequential.sh 2 5
```

**Custom configuration**:
```bash
# 15 epochs, different EMA snapshots
MAX_EPOCHS=15 \
EMA_EPOCHS=3,7,12 \
bash apex-ranker/scripts/train_folds_sequential.sh 2 5

# Different output location
OUTPUT_PREFIX=models/experiment_v2 \
LOG_DIR=logs/experiment_v2 \
bash apex-ranker/scripts/train_folds_sequential.sh 2 5
```

**Expected output**:
```
========================================
Sequential Fold Training (OOM-Safe)
========================================
Config: apex-ranker/configs/v0_base.yaml
Folds: 2 to 5
Output: models/shortfocus_foldN.pt
Logs: logs/shortfocus_foldN.log
========================================

[2025-11-01 14:10:00] Starting Fold 2/5
...
✅ Fold 2 completed successfully
   Duration: 58m 32s
   Output: models/shortfocus_fold2.pt
   Log: logs/shortfocus_fold2.log
...
========================================
Sequential Training Complete!
========================================
Folds completed: 2 to 5
Total duration: 239m 57s
Average per fold: 59m
```

**Timeline**: ~4-5 hours for 4 folds (at 12 epochs each)

---

**Option B: Parallel-Safe (2x faster, small OOM risk)**

```bash
# Run 2 folds at a time (default)
bash apex-ranker/scripts/train_folds_parallel_safe.sh 2 5

# Or 3 at a time (higher risk)
PARALLEL_JOBS=3 \
bash apex-ranker/scripts/train_folds_parallel_safe.sh 2 5
```

**Expected output**:
```
========================================
Batch 1/2: Folds 2 3
========================================
  Launched Fold 2 (PID: 12345)
  Launched Fold 3 (PID: 12346)
Waiting for batch 1 to complete...
  ✅ Fold 2 finished (PID: 12345)
  ✅ Fold 3 finished (PID: 12346)
...
========================================
Parallel Training Complete!
========================================
Total duration: 122m
Status: ✅ All folds completed successfully
```

**Timeline**: ~2-2.5 hours for 4 folds (2 at a time)

---

### Step 3: Blend All Folds & Full Backtest

#### 3a. Create blended checkpoints for all folds

```bash
# Automated blending for all 5 folds
for fold in 1 2 3 4 5; do
  echo "Blending Fold ${fold}..."
  python apex-ranker/scripts/average_checkpoints.py \
    models/shortfocus_fold${fold}_ema_epoch3.pt \
    models/shortfocus_fold${fold}_ema_epoch6.pt \
    models/shortfocus_fold${fold}_ema_epoch10.pt \
    --output models/shortfocus_fold${fold}_blended.pt
done
```

#### 3b. Run full backtest (all periods)

```bash
# Full period backtest (2023-2025)
python apex-ranker/scripts/backtest_v0.py \
  --checkpoints models/shortfocus_fold*_blended.pt \
  --config apex-ranker/configs/v0_base.yaml \
  --start-date 2023-01-01 \
  --end-date 2025-10-31 \
  --top-k 50 \
  --horizon 5 \
  --output results/shortfocus_5fold_backtest.json \
  --daily-csv results/shortfocus_daily.csv
```

#### 3c. A/B Comparison

**Key metrics to check**:
1. **h5_delta_p_at_k_pos**: DM > 1.96 & 95% CI > 0 (統計的有意)
2. **h5_spread**: Top-Bottom spread が正・有意
3. **h5_delta_ndcg**: NDCG改善が正・有意
4. **Operational guards**:
   - Zero selection rate < 1%
   - Fallback rate < 20%
   - Beta/sector exposure within bands

**Decision criteria**:
- ✅ **All Green** → Production candidate (A/B test approved)
- ⚠️ **Some Yellow** → Conditional approval (monitor closely)
- ❌ **Any Red** → Back to optimization

---

## 🔧 Monitoring & Troubleshooting

### Real-time monitoring

```bash
# Watch training progress
tail -f logs/shortfocus_fold2.log

# Monitor memory usage
watch -n 5 'free -h | grep Mem'

# Check GPU utilization
watch -n 2 nvidia-smi

# List checkpoints
ls -lh models/shortfocus_fold*.pt
```

### Common issues

**Issue 1: Training killed with "Killed"**
```
Cause: OOM (Out of Memory)
Fix: Use sequential execution (guaranteed safe)
```

**Issue 2: Slow GPU utilization (<50%)**
```
Cause: CPU bottleneck in DataLoader
Fix: Already optimized (batch_size=1, num_workers=0)
      This is expected for daily batching
```

**Issue 3: One fold failed in parallel mode**
```
Cause: Resource contention
Fix: Re-run that fold alone:
     python -m apex_ranker.scripts.train_v0 \
       --config apex-ranker/configs/v0_base.yaml \
       --cv-fold 3 \
       --output models/shortfocus_fold3.pt
```

---

## 📊 Memory Usage Summary

| Mode | Folds/Time | Memory Peak | OOM Risk | Recommended |
|------|------------|-------------|----------|-------------|
| Sequential | 1 at a time, 4-5h | ~100GB | 0% | ✅ **YES** |
| Parallel-Safe (2x) | 2 at a time, 2-2.5h | ~200GB | 5% | ⚡ OK if urgent |
| Parallel (5x) | 5 at a time, 1h | ~500GB | **90%** | ❌ **NO** |

**System**: 2TiB RAM, 1.9TiB available

---

## 📁 Output Structure

After completion, you'll have:

```
models/
├── shortfocus_fold1.pt              # Original checkpoint
├── shortfocus_fold1_ema_epoch3.pt   # EMA snapshot at epoch 3
├── shortfocus_fold1_ema_epoch6.pt   # EMA snapshot at epoch 6
├── shortfocus_fold1_ema_epoch10.pt  # EMA snapshot at epoch 10
├── shortfocus_fold1_blended.pt      # Averaged (3+6+10)
├── shortfocus_fold2.pt
├── shortfocus_fold2_ema_epoch3.pt
...
└── shortfocus_fold5_blended.pt

logs/
├── shortfocus_fold1.log
├── shortfocus_fold2.log
...
└── shortfocus_fold5.log

results/
├── shortfocus_5fold_backtest.json
└── shortfocus_daily.csv
```

---

## ✅ Final Checklist

Before moving to production:

- [ ] Fold1 health check passed (Green metrics)
- [ ] All 5 folds trained successfully
- [ ] All folds blended (EMA snapshots averaged)
- [ ] Full backtest completed (2023-2025)
- [ ] A/B metrics are Green (DM > 1.96, CI > 0)
- [ ] Operational guards passed (<1% zero selection, <20% fallback)
- [ ] Logs saved and reviewed

---

## 🚀 Next Steps After Completion

1. **Create production ensemble**:
   ```bash
   # Average all 5 fold-blended checkpoints
   python apex-ranker/scripts/average_checkpoints.py \
     models/shortfocus_fold*_blended.pt \
     --output models/shortfocus_production_ensemble.pt
   ```

2. **Deploy to inference API** (if approved):
   ```bash
   python apex-ranker/apex_ranker/api/server.py \
     --checkpoint models/shortfocus_production_ensemble.pt \
     --config apex-ranker/configs/v0_base.yaml \
     --port 8000
   ```

3. **Set up monitoring** (Grafana, alerts, daily logs)

---

## 📚 Reference Documentation

- **OOM Prevention Guide**: `apex-ranker/docs/OOM_PREVENTION_GUIDE.md`
- **Experiment Status**: `apex-ranker/EXPERIMENT_STATUS.md`
- **Training Script**: `apex-ranker/scripts/train_v0.py`
- **Config Reference**: `apex-ranker/configs/v0_base.yaml`

---

**Status**: Ready to execute 🚀
**Estimated time**: 4-5 hours (sequential) or 2-2.5 hours (parallel-safe)
**Risk**: 0% OOM (sequential), 5% OOM (parallel-safe)
