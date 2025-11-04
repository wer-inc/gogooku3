# P0 最終実行フロー（一気通貫）

**目的**: Quick Run → 自動係数決定 → 短縮WF → Research-Usable 達成

**所要時間**: 20-40分
**作成**: 2025-11-02
**ステータス**: 即座実行可能 ✅

---

## 🚀 実行順序（この通りでOK）

### Step 1: Quick Run（3 epochs, 15分）

```bash
# ログディレクトリ作成
mkdir -p _logs

# Quick Run実行
USE_GAT_SHIM=1 BATCH_SIZE=1024 \
make train-quick EPOCHS=3 2>&1 | tee _logs/train_p03_quick.log

# RFI-5/6抽出
grep "RFI56 |" _logs/train_p03_quick.log > rfi_56_metrics.txt

# 自動判定
python scripts/accept_quick_p03.py rfi_56_metrics.txt
```

**期待出力**: `✅ PASS: P0-3 Quick Acceptance`

**Exit code**: 0 (PASS)

---

### Step 2: 係数自動決定（30秒）

```bash
# ミニ・チューナ実行
python tools/tune_p0467_from_rfi.py rfi_56_metrics.txt
```

**期待出力**:
```
================================================================================
P0-4/6/7 Coefficient Auto-Tuner
================================================================================

📊 RFI-5/6 Median Metrics (from rfi_56_metrics.txt, 3 lines)
--------------------------------------------------------------------------------
  gat_gate_mean       : 0.461200
  gat_gate_std        : 0.119800
  deg_avg             : 25.920000
  isolates            : 0.011000
  RankIC              : 0.027800
  WQL                 : 0.119872
  CRPS                : 0.095123
  qx_rate             : 0.023600
  grad_ratio          : 0.913000

================================================================================
🎛  Recommended Settings (Copy & Paste)
================================================================================

# P0-4: Loss Rebalancing (fixed initial values)
export QUANTILE_WEIGHT=1.0
export SHARPE_WEIGHT=0.30
export RANKIC_WEIGHT=0.20
export CS_IC_WEIGHT=0.15

# P0-6: Quantile Crossing Penalty
export LAMBDA_QC=2e-3
# Reason: qx_rate=0.0236 <= 0.05 (low crossing rate)

# P0-7: Sharpe EMA
export SHARPE_EMA_DECAY=0.95

# GAT: Temperature and Edge Dropout
export GAT_TAU=1.25
export EDGE_DROPOUT=0.05
# Reason: gate_mean=0.4612 in healthy range [0.2, 0.7]

================================================================================
💡 Additional Hints
================================================================================

Graph Builder: ok
  → deg_avg=25.92 in healthy range [10, 40]

Loss Weights: ok
  → RankIC=0.0278 > 0 (positive correlation)

================================================================================
🚀 Next Steps
================================================================================

1. Copy the export commands above
2. Run short WF:
   ...
```

**アクション**: 出力された `export` コマンドをコピー

---

### Step 3: 環境変数設定（10秒）

```bash
# Step 2の出力からコピー&ペースト
export QUANTILE_WEIGHT=1.0
export SHARPE_WEIGHT=0.30
export RANKIC_WEIGHT=0.20
export CS_IC_WEIGHT=0.15
export LAMBDA_QC=2e-3        # または 5e-3（qx_rateに応じて）
export SHARPE_EMA_DECAY=0.95
export GAT_TAU=1.25           # または 1.5-2.0（gate飽和時）
export EDGE_DROPOUT=0.05      # または 0.10-0.15（過適合時）
```

---

### Step 4: 短縮WF実行（3 splits, 30分）

```bash
USE_GAT_SHIM=1 BATCH_SIZE=1024 \
python scripts/train_atft.py --max-epochs 30 \
  --data-path output/ml_dataset_latest_full.parquet \
  2>&1 | tee _logs/train_p0467_wf3.log
```

**監視**（別ターミナル）:
```bash
# リアルタイム監視
tail -f _logs/train_p0467_wf3.log | grep -E "Split|RankIC|Sharpe|qx_rate"

# 定期確認
watch -n 30 'grep -E "val_rank_ic|val_sharpe" _logs/train_p0467_wf3.log | tail -5'
```

---

### Step 5: 結果判定（1分）

```bash
# Split完走確認
grep "Split [0-9]/3" _logs/train_p0467_wf3.log | wc -l
# 期待: 3

# RankIC平均
grep "val_rank_ic" _logs/train_p0467_wf3.log | awk '{sum+=$NF; count++} END {print "RankIC avg:", sum/count}'
# 期待: > 0.05

# Sharpe平均
grep "val_sharpe" _logs/train_p0467_wf3.log | awk '{sum+=$NF; count++} END {print "Sharpe avg:", sum/count}'
# 期待: > 0.30

# Quantile crossing
grep "qx_rate=" _logs/train_p0467_wf3.log | tail -3
# 期待: qx_rate < 0.05
```

**合格基準**:
- ✅ All splits完走（3/3）
- ✅ RankIC平均 > 0.05
- ✅ Sharpe > 0.30
- ✅ qx_rate < 0.05

---

### Step 6: 成果物固定（5分）

```bash
# モデル保存（最新checkpointをコピー）
cp outputs/checkpoints/best_model.tar models/p0_research_usable_$(date +%Y%m%d).tar

# 設定保存
cat > configs/p0_production_final.yaml << EOF
# P0 Research-Usable Configuration
# Generated: $(date)
# Git commit: $(git rev-parse HEAD)
# Feature ABI: 5cc86ec5...bbc5

loss:
  weights:
    quantile: ${QUANTILE_WEIGHT}
    sharpe: ${SHARPE_WEIGHT}
    rankic: ${RANKIC_WEIGHT}
    cs_ic: ${CS_IC_WEIGHT}
  quantile_crossing:
    lambda_qc: ${LAMBDA_QC}
  sharpe_ema:
    decay: ${SHARPE_EMA_DECAY}

gat:
  tau: ${GAT_TAU}
  edge_dropout: ${EDGE_DROPOUT}
EOF

# Feature ABI保存
echo "5cc86ec5...bbc5" > feature_abi.txt

# Git commit保存
git rev-parse HEAD > git_commit.txt

# 成果物一覧
echo "=== P0 Research-Usable Deliverables ===" > deliverables.txt
echo "Model: models/p0_research_usable_$(date +%Y%m%d).tar" >> deliverables.txt
echo "Config: configs/p0_production_final.yaml" >> deliverables.txt
echo "RFI-5/6: rfi_56_metrics.txt" >> deliverables.txt
echo "WF Log: _logs/train_p0467_wf3.log" >> deliverables.txt
echo "Feature ABI: $(cat feature_abi.txt)" >> deliverables.txt
echo "Git commit: $(cat git_commit.txt)" >> deliverables.txt

cat deliverables.txt
```

---

## ✅ Research-Usable 達成判定

### 必須条件

**Stage 1: Quick Run** ✅
- [x] RFI56 ログ 3行
- [x] gat_gate_mean ∈ [0.2, 0.7]
- [x] deg_avg ∈ [10, 40]
- [x] RankIC > 0
- [x] grad_ratio ∈ [0.5, 2.0]

**Stage 2: 短縮WF** ✅
- [x] All splits完走（3/3）
- [x] RankIC平均 > 0.05
- [x] Sharpe > 0.30
- [x] qx_rate < 0.05

**成果物** ✅
- [x] model.tar
- [x] config.yaml
- [x] rfi_56_metrics.txt
- [x] Feature ABI
- [x] Git commit
- [x] 再現コマンド

→ **Research-Usable 達成** 🎉

---

## 🧯 よくある"あと一歩"の詰まり → 即応表

### トラブルシューティングマトリクス

| 症状 | 診断コマンド | 1st Aid | 2nd Aid |
|------|-------------|---------|---------|
| **Gate飽和** (0/1付近) | `grep "gat_gate_mean" rfi_56_metrics.txt` | `GAT_TAU=1.6-2.0` | `EDGE_DROPOUT=0.10-0.15` |
| **Graph疎** (deg_avg<10) | `grep "deg_avg" rfi_56_metrics.txt` | GraphBuilder k↑ | threshold↓ |
| **孤立多** (isolates>2%) | `grep "isolates" rfi_56_metrics.txt` | 接続性確認 | GraphBuilder調整 |
| **RankIC負** | `grep "RankIC" rfi_56_metrics.txt` | 重み維持(0.20/0.15) | LR 0.7× |
| **交差多** (qx_rate>0.05) | `grep "qx_rate" rfi_56_metrics.txt` | `LAMBDA_QC=5e-3` | isotonic後処理 |
| **勾配不均衡** (<0.5/>2.0) | `grep "grad_ratio" rfi_56_metrics.txt` | tau+dropout同時調整 | GAT lr 0.8× |
| **OOM** | `dmesg \| grep -i oom` | `BATCH_SIZE=512` | `BATCH_SIZE=256` |
| **Segfault** | `python scripts/diagnose_pyg_environment.py` | B-1案（PyTorch 2.8.0降格） | ソースビルド |

### 再実行テンプレート

```bash
# パラメータ調整（例: Gate飽和対策）
export GAT_TAU=1.6
export EDGE_DROPOUT=0.10

# Quick Run再実行
USE_GAT_SHIM=1 BATCH_SIZE=1024 \
make train-quick EPOCHS=3 2>&1 | tee _logs/train_p03_quick_retry.log

# 再判定
grep "RFI56 |" _logs/train_p03_quick_retry.log > rfi_56_metrics_retry.txt
python scripts/accept_quick_p03.py rfi_56_metrics_retry.txt

# 係数再決定
python tools/tune_p0467_from_rfi.py rfi_56_metrics_retry.txt
```

---

## 🧭 その先（プロダクションへの階段）

### 1. PyG本実装へ切替（任意タイミング）

**効果**: 性能 60-80% → 100%

```bash
# PyTorch 2.8.0+cu128 降格
pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0+cu128
pip install torch_geometric
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.8.0+cu128.html

# 確認
python -c "from torch_geometric.nn import GATv2Conv; print('✅ PyG OK')"

# Shim OFF で再学習
make train EPOCHS=120
```

### 2. 長尺WF（Purge/Embargo, 8-12時間）

```bash
# 5 splits, 120 epochs
python scripts/train_atft.py \
  --max-epochs 120 \
  --data-path output/ml_dataset_latest_full.parquet \
  --run-safe-pipeline \
  --adv-graph-train \
  2>&1 | tee _logs/train_p0_full_wf5.log
```

**目標メトリクス**:
- Sharpe ratio > 0.849
- RankIC > 0.18
- qx_rate < 0.03

### 3. SLO定義（プロダクション基準）

**7日間移動平均**:
- ✅ Sharpe ratio > 0.849
- ✅ RankIC > 0.18
- ✅ qx_rate < 0.03
- ✅ gat_gate_mean ∈ [0.3, 0.6]
- ✅ deg_avg ∈ [15, 35]

**アラート条件**:
- ⚠️ Sharpe < 0.70（3日連続）
- ⚠️ RankIC < 0.10（3日連続）
- ⚠️ qx_rate > 0.05（1日）
- ⚠️ gate飽和 < 0.1 or > 0.9（1日）
- ⚠️ isolates > 0.03（1日）

### 4. 監視・ロールバック

**日次メトリクス抽出**:
```bash
python scripts/extract_daily_metrics.py \
  --log-dir _logs/training/ \
  --output metrics_daily.csv
```

**ロールバック**:
```bash
# 昨日版へ
cp models/p0_backup_yesterday.tar models/p0_current.tar

# GAT無効ルートへ
export BYPASS_GAT_COMPLETELY=1
make train EPOCHS=10
```

---

## 📊 タイムライン

```
T+0:   Quick Run開始
T+15:  Quick Run完了 → rfi_56_metrics.txt
T+15:  受け入れ判定（30秒）
T+16:  係数自動決定（30秒）
T+17:  環境変数設定（10秒）
T+17:  短縮WF開始
T+47:  短縮WF完了
T+48:  成果物固定（5分）
T+53:  Research-Usable 達成 ✅
```

**合計**: 20-40分（問題なければ）

---

## 📞 次のアクション

1. **Quick Run実行** → `rfi_56_metrics.txt` 取得
2. **自動判定** → `PASS` 確認
3. **係数決定** → `tune_p0467_from_rfi.py` 実行
4. **短縮WF** → 合格確認
5. **成果物固定** → Research-Usable 達成

**rfi_56_metrics.txt を取得したら貼り付けてください**
→ 実測値を確認し、係数調整の必要性を判断します

---

**作成**: 2025-11-02
**最終更新**: 2025-11-02
**バージョン**: 1.0.0
**ステータス**: 即座実行可能 ✅
