# P0 クイックリファレンス（1枚チートシート）

**即座実行コマンド集** - コピペで完了

---

## 🚀 3コマンドで完了（20分）

```bash
# 1. Quick Run (15分)
USE_GAT_SHIM=1 BATCH_SIZE=1024 make train-quick EPOCHS=3 2>&1 | tee _logs/train_p03_quick.log

# 2. RFI-5/6抽出 (10秒)
grep "RFI56 |" _logs/train_p03_quick.log > rfi_56_metrics.txt

# 3. 受け入れ判定 (10秒)
python scripts/accept_quick_p03.py rfi_56_metrics.txt
```

**期待結果**: `✅ PASS: P0-3 Quick Acceptance`

---

## ✅ PASS後: P0-4/6/7有効化 + 短縮WF（30分）

```bash
# 係数設定
export QUANTILE_WEIGHT=1.0
export SHARPE_WEIGHT=0.30
export RANKIC_WEIGHT=0.20
export CS_IC_WEIGHT=0.15
export LAMBDA_QC=2e-3        # qx_rate < 0.05の場合
# export LAMBDA_QC=5e-3      # qx_rate > 0.05の場合（コメント外す）
export SHARPE_EMA_DECAY=0.95

# 短縮WF実行
USE_GAT_SHIM=1 BATCH_SIZE=1024 \
python scripts/train_atft.py \
  --max-epochs 30 \
  --data-path output/ml_dataset_latest_full.parquet \
  2>&1 | tee _logs/train_p0467_wf3.log
```

**期待結果**: RankIC > 0.05, Sharpe > 0.3, qx_rate < 0.05

---

## 🏁 本番学習（120 epochs, 8-12時間）

```bash
# 係数確認
env | grep -E "QUANTILE_WEIGHT|SHARPE_WEIGHT|RANKIC_WEIGHT|CS_IC_WEIGHT"

# 実行
USE_GAT_SHIM=1 BATCH_SIZE=2048 make train EPOCHS=120 2>&1 | tee _logs/train_p0_production.log
```

**目標**: Sharpe > 0.849, RankIC > 0.18

---

## 🔴 トラブルシューティング

### Segfault → B-1案（5分）

```bash
pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0+cu128
pip install torch_geometric
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
make train-quick EPOCHS=3
```

### OOM → Batch size削減

```bash
USE_GAT_SHIM=1 BATCH_SIZE=512 make train-quick EPOCHS=3
```

### GAT skip → グラフビルド

```bash
python scripts/build_graph_cache.py --start-date 2024-01-01 --end-date 2025-01-31
```

### Gate飽和 (gate_mean ≈ 0/1)

```bash
export GAT_TAU=1.5  # または 2.0
```

### 交差多発 (qx_rate > 0.05)

```bash
export LAMBDA_QC=5e-3
```

---

## 📊 成功判定基準

### Quick Run (3 epochs)
- ✅ `RFI56 |` ログ 3行
- ✅ `gat_gate_mean` ∈ [0.2, 0.7]
- ✅ `deg_avg` ∈ [10, 40]
- ✅ `RankIC > 0`
- ✅ `grad_ratio` ∈ [0.5, 2.0]

### 短縮WF (3 splits)
- ✅ All splits完走
- ✅ RankIC平均 > 0.05
- ✅ Sharpe > 0.3
- ✅ qx_rate < 0.05

### 本番学習 (120 epochs)
- ✅ Sharpe ratio > 0.849
- ✅ RankIC > 0.18
- ✅ qx_rate < 0.03

---

## 📁 ドキュメント索引

| ファイル | 用途 | 優先度 |
|---------|------|--------|
| `P0_GO_NO_GO_EXECUTION_GUIDE.md` | 完全実行ガイド | ⭐⭐⭐⭐⭐ |
| `P0_3_EXECUTION_RECIPE.md` | P0-3実行レシピ | ⭐⭐⭐⭐⭐ |
| `P0_3_TRAIN_ATFT_PATCH.md` | ログ統合パッチ | ⭐⭐⭐⭐⭐ |
| `P0_4_6_7_COEFFICIENTS.md` | 係数設定詳細 | ⭐⭐⭐⭐ |
| `P0_3_FINAL_DELIVERABLES.md` | 成果物一覧 | ⭐⭐⭐⭐ |
| `P0_QUICK_REFERENCE.md` | 本ファイル | ⭐⭐⭐⭐ |

---

## 🔧 デバッグコマンド

```bash
# 環境診断
python scripts/diagnose_pyg_environment.py

# パッチ確認
grep "log_rfi_56_metrics" scripts/train_atft.py

# 最新ログ確認
tail -100 _logs/train_p03_quick.log

# RFI-5/6確認
cat rfi_56_metrics.txt

# プロセス確認
ps aux | grep train_atft
```

---

## 📞 クイックヘルプ

**問題**: RFI56ログが出ない
**解決**: `P0_3_TRAIN_ATFT_PATCH.md` を参照してパッチ適用

**問題**: gat_gate_mean が NaN
**解決**: edge_index供給確認 → グラフビルダー実行

**問題**: RankIC が負
**解決**: 正常（初期10 epoch以内）、継続監視

**問題**: 速度が遅い
**解決**: Shim性能は60-80%、PyG移行で100%

---

**作成**: 2025-11-02
**バージョン**: 1.0.0
**ステータス**: コピペ可能 ✅
