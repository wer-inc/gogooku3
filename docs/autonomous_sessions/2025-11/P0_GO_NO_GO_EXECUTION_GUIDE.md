# P0 Go/No-Go 実行ガイド（20-40分で完了）

**目的**: P0-3完了 → RFI-5/6取得 → P0-4/6/7有効化 → 短縮WF → 本番学習

**作成**: 2025-11-02
**ステータス**: 即座実行可能 ✅

---

## 📋 実行フロー（3ステップ）

```
Step 1: Quick Run (3 epochs, 15分)
   ↓
Step 2: 受け入れ判定 (30秒)
   ↓
Step 3a: PASS → P0-4/6/7有効化 → 短縮WF (3 splits, 30分)
   or
Step 3b: Borderline/FAIL → トリアージ → 再実行
```

---

## 🚀 Step 1: Quick Run（RFI-5/6取得）

### 前提条件確認

```bash
# 1. Dataset存在確認
ls -lh output/ml_dataset_latest_full.parquet
# 期待: 1-5GB程度のファイル

# 2. train_atft.py パッチ確認
grep "log_rfi_56_metrics" scripts/train_atft.py
# 期待: 2マッチ（import + 呼び出し）

# 3. FAN/SAN有効確認
echo $BYPASS_ADAPTIVE_NORM
# 期待: （空）または 0

# 4. GAT有効確認
echo $BYPASS_GAT_COMPLETELY
# 期待: （空）または 0
```

### 実行（3 epoch）

```bash
# ログディレクトリ作成
mkdir -p _logs

# Quick Run実行
USE_GAT_SHIM=1 BATCH_SIZE=1024 \
make train-quick EPOCHS=3 2>&1 | tee _logs/train_p03_quick.log
```

**監視ポイント**（別ターミナル）:
```bash
# リアルタイム監視
tail -f _logs/train_p03_quick.log

# 起動確認（最初の1分）
# 期待ログ:
# [P0-3 GAT-FALLBACK] Using GraphConvShim (PyG-free mode)
# Feature ABI: 5cc86ec5...bbc5
# Phase 0: Baseline

# Epoch 1開始確認（2-5分）
# 期待ログ:
# Epoch 1/3: Train Loss=...
# [VAL-DEBUG] batch0 metrics - Sharpe: ..., IC: ..., RankIC: ...

# RFI-5/6ログ確認（各epoch終了時）
# 期待ログ:
# RFI56 | epoch=1 gat_gate_mean=... deg_avg=... RankIC=...
```

**想定所要時間**: 15分（A100 80GB, batch_size=1024）

### RFI-5/6抽出

```bash
# メトリクス抽出
grep "RFI56 |" _logs/train_p03_quick.log > rfi_56_metrics.txt

# 確認
cat rfi_56_metrics.txt

# 期待される出力例:
# RFI56 | epoch=1 gat_gate_mean=0.4523 gat_gate_std=0.1234 deg_avg=25.67 isolates=0.012 corr_mean=0.345 corr_std=0.234 RankIC=0.0234 WQL=0.123456 CRPS=0.098765 qx_rate=0.0234 grad_ratio=0.87
# RFI56 | epoch=2 gat_gate_mean=0.4612 gat_gate_std=0.1198 deg_avg=26.12 isolates=0.011 corr_mean=0.351 corr_std=0.228 RankIC=0.0289 WQL=0.119872 CRPS=0.095123 qx_rate=0.0198 grad_ratio=0.92
# RFI56 | epoch=3 gat_gate_mean=0.4701 gat_gate_std=0.1167 deg_avg=25.98 isolates=0.010 corr_mean=0.348 corr_std=0.231 RankIC=0.0312 WQL=0.116543 CRPS=0.091234 qx_rate=0.0176 grad_ratio=0.95
```

---

## ✅ Step 2: 受け入れ判定（Go/No-Go）

### 自動判定スクリプト実行

```bash
python scripts/accept_quick_p03.py rfi_56_metrics.txt
```

**期待される出力（PASS時）**:
```
================================================================================
P0-3 Quick Acceptance Test (Go/No-Go)
================================================================================

📊 Parsed 33 metrics from 3 epochs

✅ GAT gate_mean: 0.4612 (healthy range)
✅ Graph deg_avg: 25.92 (healthy connectivity)
✅ Graph isolates: 0.0110 (minimal isolation)
✅ RankIC: 0.0278 (positive correlation)
   ℹ️  Low but acceptable for initial epochs
✅ Gradient ratio: 0.913 (balanced)
✅ Quantile crossing: 0.0236 (low)
✅ WQL trend: 0.123456 → 0.116543 (improving)
✅ CRPS trend: 0.098765 → 0.091234 (improving)

================================================================================
✅ PASS: P0-3 Quick Acceptance

Next steps:
1. Enable P0-4/6/7 coefficients
2. Run short WF validation (3 splits)
3. Monitor full training (120 epochs)
================================================================================
```

**Exit codes**:
- `0`: PASS - すべてのチェック合格 → Step 3aへ
- `1`: FAIL - 複数の重大な問題 → トリアージへ
- `2`: WARN - ボーダーライン → 手動レビュー推奨

### 手動確認（スクリプトなし）

```bash
# Gate統計
# 期待: gat_gate_mean ∈ [0.2, 0.7], gat_gate_std ∈ [0.05, 0.30]

# Graph統計
# 期待: deg_avg ∈ [10, 40], isolates < 0.02

# RankIC
# 期待: RankIC > 0（初期は 0.02-0.10でOK）

# Gradient ratio
# 期待: grad_ratio ∈ [0.5, 2.0]

# Quantile crossing
# 期待: qx_rate < 0.05（超える場合は P0-6 ペナルティ強化）
```

---

## 🎛 Step 3a: PASS → P0-4/6/7有効化 + 短縮WF

### 係数確定（rfi_56_metrics.txt から）

```bash
# qx_rate の中央値を確認
grep "qx_rate=" rfi_56_metrics.txt | awk -F'qx_rate=' '{print $2}' | awk '{print $1}' | sort -n | awk 'NR==2'

# 判定:
# qx_rate < 0.05 → LAMBDA_QC=2e-3 (デフォルト)
# qx_rate > 0.05 → LAMBDA_QC=5e-3 (ペナルティ強化)
```

### 環境変数設定

```bash
# P0-4: Loss Rebalancing
export QUANTILE_WEIGHT=1.0
export SHARPE_WEIGHT=0.30
export RANKIC_WEIGHT=0.20
export CS_IC_WEIGHT=0.15

# P0-6: Quantile Crossing (qx_rate < 0.05の場合)
export LAMBDA_QC=2e-3

# P0-7: Sharpe EMA
export SHARPE_EMA_DECAY=0.95
export SHARPE_EMA_WARMUP=10

# GAT安定化（必要時のみ）
# export GAT_TAU=1.25  # デフォルト値のまま
# export GAT_EDGE_DROPOUT=0.05
```

### 短縮WF実行（3 splits）

```bash
# P0-4/6/7有効 + 短縮WF
USE_GAT_SHIM=1 BATCH_SIZE=1024 \
python scripts/train_atft.py \
  --max-epochs 30 \
  --data-path output/ml_dataset_latest_full.parquet \
  --run-safe-pipeline \
  --adv-graph-train \
  2>&1 | tee _logs/train_p0467_wf3.log
```

**注**: Walk-Forward splitsは `scripts/train_atft.py` 内で自動設定（通常5 splits、ここでは環境変数で3に調整可能）

**想定所要時間**: 30分（3 splits × 10分/split）

### 成功判定（短縮WF）

```bash
# 全splits完走確認
grep "Split [0-9]/3" _logs/train_p0467_wf3.log | wc -l
# 期待: 3

# RankIC平均値
grep "val_rank_ic" _logs/train_p0467_wf3.log | awk '{sum+=$NF; count++} END {print "RankIC avg:", sum/count}'
# 期待: > 0.05

# Sharpe ratio
grep "val_sharpe" _logs/train_p0467_wf3.log | awk '{sum+=$NF; count++} END {print "Sharpe avg:", sum/count}'
# 期待: > 0.3

# Quantile crossing
grep "qx_rate=" _logs/train_p0467_wf3.log | tail -3
# 期待: qx_rate < 0.05（ペナルティが効いている）
```

**基準**:
- ✅ All splits完走（exit code 0）
- ✅ RankIC平均 > 0.05
- ✅ Sharpe ratio > 0.3
- ✅ qx_rate < 0.05

→ **合格なら本番学習（120 epochs）へ**

---

## 🔴 Step 3b: Borderline/FAIL → トリアージ

### トラブルシューティングマトリクス

| 症状 | 原因 | 手当 |
|------|------|------|
| `gat_gate_mean ≈ 0/1` | Gate飽和 | `tau` を 1.5-2.0 に増加 |
| `deg_avg < 10` | グラフ疎 | GraphBuilder `k ↑` または `threshold ↓` |
| `isolates > 2%` | 孤立ノード多 | 接続性確認、k-NN増加 |
| `RankIC ≤ 0` | 初期学習不安定 | RankIC/CS-IC重みを一時的に `0.05` に下げる |
| `qx_rate > 0.05` | 交差多発 | `LAMBDA_QC=5e-3` に増加 |
| `grad_ratio < 0.5 or > 2.0` | 勾配不均衡 | `tau` と `edge_dropout` 同時調整 |
| OOM | メモリ不足 | `BATCH_SIZE=512` に削減 |
| Segfault | PyG環境問題 | 即座にB-1案（PyTorch 2.8.0降格）実施 |

### 再実行（修正後）

```bash
# 環境変数で調整パラメータを設定
export GAT_TAU=1.5  # Gate飽和対策
export LAMBDA_QC=5e-3  # qx_rate高い場合

# Quick Run再実行
USE_GAT_SHIM=1 BATCH_SIZE=1024 \
make train-quick EPOCHS=3 2>&1 | tee _logs/train_p03_quick_retry.log

# 再判定
grep "RFI56 |" _logs/train_p03_quick_retry.log > rfi_56_metrics_retry.txt
python scripts/accept_quick_p03.py rfi_56_metrics_retry.txt
```

---

## 🏁 本番学習（P0完了後）

### 前提条件

- ✅ Quick Run PASS
- ✅ 短縮WF PASS
- ✅ P0-4/6/7係数確定

### 実行コマンド

```bash
# 環境変数確認（係数が設定されていること）
env | grep -E "QUANTILE_WEIGHT|SHARPE_WEIGHT|RANKIC_WEIGHT|CS_IC_WEIGHT|LAMBDA_QC|SHARPE_EMA_DECAY"

# 本番学習（120 epochs, 5 splits）
USE_GAT_SHIM=1 BATCH_SIZE=2048 \
make train EPOCHS=120 2>&1 | tee _logs/train_p0_production.log
```

**想定所要時間**: 8-12時間（A100 80GB, 120 epochs）

### 監視（別ターミナル）

```bash
# リアルタイム監視
tail -f _logs/train_p0_production.log | grep -E "Epoch|RFI56|val_rank_ic|val_sharpe"

# メトリクス抽出（定期実行）
watch -n 60 'grep "RFI56 |" _logs/train_p0_production.log | tail -5'
```

### 目標メトリクス（120 epochs完了時）

```
Sharpe ratio: > 0.849
RankIC: > 0.18
qx_rate: < 0.03
gat_gate_mean: 0.3-0.6（安定）
deg_avg: 15-35（安定）
```

---

## 📊 成果物提出フォーマット

### P0完了報告テンプレート

```markdown
## P0 Complete - Production Ready

### Environment
- PyTorch: 2.9.0+cu128
- CUDA: 12.8
- GAT mode: Shim (GraphConvShim)
- Feature ABI: 5cc86ec5...bbc5
- Git commit: <commit_hash>

### RFI-5/6 Quick Run (3 epochs)
- gat_gate_mean: 0.4612
- deg_avg: 25.92
- RankIC: 0.0278
- qx_rate: 0.0236
- Acceptance: ✅ PASS

### P0-4/6/7 Coefficients (confirmed)
- QUANTILE_WEIGHT: 1.0
- SHARPE_WEIGHT: 0.30
- RANKIC_WEIGHT: 0.20
- CS_IC_WEIGHT: 0.15
- LAMBDA_QC: 2e-3
- SHARPE_EMA_DECAY: 0.95

### Short WF (3 splits, 30 epochs)
- RankIC avg: 0.067
- Sharpe avg: 0.412
- qx_rate: 0.023
- Validation: ✅ PASS

### Full Training (120 epochs, 5 splits)
- Sharpe ratio: 0.873
- RankIC: 0.192
- qx_rate: 0.027
- Status: ✅ Production Ready

### Deliverables
- Model: models/p0_complete_YYYYMMDD.tar
- Config: configs/p0_production_final.yaml
- Predictions: outputs/predictions_daily.csv
- Reproduce: `make reproduce --run-id <ID>`
```

---

## 🧭 次のステップ（P0完了後）

### 即座実施

1. **PyG本実装へ切替**（任意・時間があれば）
   ```bash
   # PyTorch 2.8.0+cu128降格
   pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0+cu128
   pip install torch_geometric
   pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
     -f https://data.pyg.org/whl/torch-2.8.0+cu128.html

   # Shim OFF で再学習
   make train EPOCHS=120
   ```

2. **Hyperparameter Tuning**（Optuna）
   - tau: 1.0-2.0
   - edge_dropout: 0.03-0.15
   - SHARPE_EMA_DECAY: 0.92-0.97
   - Loss weights: Grid search

3. **Production Deployment**
   - FastAPI endpoint
   - Daily prediction pipeline
   - Monitoring dashboard

### 研究課題

- **P1**: Attention entropy regularization（`attn_entropy_coef`）
- **P2**: Multi-scale graph（複数時間窓）
- **P3**: Adaptive loss scheduling（メトリクスベース）
- **P4**: Ensemble（複数checkpoint平均）

---

## 📞 サポート情報

### 失敗時の報告項目

1. エラーメッセージ全文
2. `rfi_56_metrics.txt` 全文
3. 最後の100行ログ（`tail -100 _logs/train_*.log`）
4. 環境情報（`python scripts/diagnose_pyg_environment.py`）
5. 実行コマンド

### 成功時の報告項目

1. `rfi_56_metrics.txt` 全文
2. 受け入れテスト結果（`accept_quick_p03.py`出力）
3. 短縮WF結果（RankIC, Sharpe, qx_rate平均）
4. 次ステップ希望

---

**作成**: 2025-11-02
**最終更新**: 2025-11-02
**バージョン**: 1.0.0
**ステータス**: 即座実行可能 ✅
