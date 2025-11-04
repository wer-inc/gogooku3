# P0-3: 実行レシピ（RFI-5/6確実回収）

**目的**: `USE_GAT_SHIM=1 make train-quick EPOCHS=3` を確実に成功させ、RFI-5/6データを回収

**作成**: 2025-11-02
**ステータス**: 即座実行可能

---

## 🎯 Go/No-Go判定基準

### ✅ Success (Go)
- 3 epoch完走
- `RFI56 |` ログ出力あり
- `gat_gate_mean` が 0.2-0.7 の範囲
- `deg_avg` が 10-40 の範囲
- segfault/OOM なし

### ❌ Failure (No-Go)
- Segfault発生 → **即座にB-1案（PyTorch 2.8.0降格）実施**
- OOM発生 → `BATCH_SIZE=512` に変更して再実行
- GAT skip → グラフビルダー確認

---

## 📋 実行手順（5ステップ）

### Step 1: 環境確認（1分）

```bash
# PyTorch/CUDA確認
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, GPU: {torch.cuda.is_available()}')"

# 期待: PyTorch: 2.9.0+cu128, CUDA: 12.8, GPU: True

# Dataset確認
ls -lh output/ml_dataset_latest_full.parquet

# 期待: 1-5GB程度のファイル
```

### Step 2: Shim mode学習開始（5-15分）

```bash
# 実行（ログファイル出力付き）
USE_GAT_SHIM=1 BATCH_SIZE=1024 make train-quick EPOCHS=3 2>&1 | tee _logs/train_p03_quick.log

# 期待されるログ（最初の数行）:
# [P0-3 GAT-FALLBACK] Using GraphConvShim (PyG-free mode).
# Reason: GATv2Conv unavailable
# Performance: ~60-80% of PyG, suitable for RFI-5/6 collection.
```

**監視ポイント**:
- 最初の1分: モデルロード・初期化（segfault注意）
- 2-5分: Epoch 1開始（OOM注意）
- 5-15分: Epoch 1-3完走

**途中で止まったら**:
```bash
# 別ターミナルで確認
tail -f _logs/train_p03_quick.log

# プロセス確認
ps aux | grep python | grep train

# GPU確認
nvidia-smi
```

### Step 3: RFI-5/6メトリクス抽出（30秒）

```bash
# RFI-5/6ログ抽出
grep -E "RFI56 \|" _logs/train_p03_quick.log | tail -n 5

# 期待される出力例:
# RFI56 | epoch=1 gat_gate_mean=0.4523 gat_gate_std=0.1234 deg_avg=25.67 isolates=0.012 corr_mean=0.345 corr_std=0.234 RankIC=0.0234 WQL=0.123456 CRPS=0.098765 qx_rate=0.0234 grad_ratio=0.87
# RFI56 | epoch=2 gat_gate_mean=0.4612 gat_gate_std=0.1198 deg_avg=26.12 isolates=0.011 corr_mean=0.351 corr_std=0.228 RankIC=0.0289 WQL=0.119872 CRPS=0.095123 qx_rate=0.0198 grad_ratio=0.92
# RFI56 | epoch=3 gat_gate_mean=0.4701 gat_gate_std=0.1167 deg_avg=25.98 isolates=0.010 corr_mean=0.348 corr_std=0.231 RankIC=0.0312 WQL=0.116543 CRPS=0.091234 qx_rate=0.0176 grad_ratio=0.95
```

**健全レンジチェック**:
```bash
# Gate統計
gat_gate_mean: 0.2-0.7 ✅ (0.0/1.0に張り付いていない)
gat_gate_std: 0.05-0.30 ✅ (学習中で分散がある)

# Graph統計
deg_avg: 10-40 ✅ (適度な接続)
isolates: < 0.02 ✅ (孤立ノードが少ない)

# Loss統計
RankIC: > 0 ✅ (初期は0.01-0.05程度でもOK)
qx_rate: < 0.05 ✅ (分位点交差が少ない)

# Gradient統計
grad_ratio: 0.5-2.0 ✅ (Base/GAT勾配バランス良好)
```

### Step 4: 詳細ログ確認（1分）

```bash
# GAT初期化確認
grep "P0-3 GAT" _logs/train_p03_quick.log | head -5

# 期待:
# [P0-3 GAT-FALLBACK] Using GraphConvShim (PyG-free mode).
# [P0-3 FUSION-INIT] GatedCrossSectionFusion: hidden=256, tau=1.25, ...

# GAT実行確認
grep "GAT-EXEC\|FUSION" _logs/train_p03_quick.log | head -10

# 期待:
# [P0-3 GAT-EXEC] edge_index.shape=torch.Size([2, 1234]), ...
# [P0-3 FUSION] z_base.shape=torch.Size([64, 256]), z_gat.shape=torch.Size([64, 256]), ...

# エラー確認（ないことを確認）
grep -i "error\|exception\|fail\|segfault\|oom" _logs/train_p03_quick.log

# 期待: （マッチなし）
```

### Step 5: データ提出（共有用）

```bash
# RFI-5/6抽出（JSONフォーマット）
grep "RFI56 |" _logs/train_p03_quick.log > rfi_56_metrics.txt

# 代表バッチの詳細統計
grep "Graph stats" _logs/train_p03_quick.log | head -1 > graph_stats_sample.txt

# 完了報告テンプレート
cat << 'EOF' > P03_RFI_SUBMISSION.md
# P0-3 RFI-5/6 提出

## 実行環境
- PyTorch: 2.9.0+cu128
- CUDA: 12.8
- GAT mode: Shim (GraphConvShim)
- Epochs: 3
- Batch size: 1024

## RFI-5/6 メトリクス

\`\`\`
$(cat rfi_56_metrics.txt)
\`\`\`

## 健全性チェック

- [x] 3 epoch完走
- [x] gat_gate_mean: 0.2-0.7
- [x] deg_avg: 10-40
- [x] RankIC > 0
- [x] qx_rate < 0.05

## 観察された問題

（なし / あれば記述）

## 次のステップ

P0-4/6/7実装を依頼
EOF

echo "✅ P03_RFI_SUBMISSION.md を作成しました"
```

---

## 🔴 トラブルシューティング

### Issue 1: Segfault（最優先対応）

**症状**:
```
Segmentation fault (core dumped)
```

**原因**: PyG環境問題（PyTorch 2.9.0+cu128 vs PyG不整合）

**対処**: **即座にB-1案実施**
```bash
# PyTorch 2.8.0+cu128 へ降格
pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0+cu128

# PyG + 拡張
pip install torch_geometric
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.8.0+cu128.html

# 確認
python -c "from torch_geometric.nn import GATv2Conv; print('✅ PyG OK')"

# 再実行（Shim不要）
make train-quick EPOCHS=3
```

### Issue 2: OOM (Out of Memory)

**症状**:
```
CUDA out of memory. Tried to allocate X MiB
```

**対処**:
```bash
# Batch sizeを半減
USE_GAT_SHIM=1 BATCH_SIZE=512 make train-quick EPOCHS=3 2>&1 | tee _logs/train_p03_quick_bs512.log

# それでもOOMなら
USE_GAT_SHIM=1 BATCH_SIZE=256 make train-quick EPOCHS=3 2>&1 | tee _logs/train_p03_quick_bs256.log
```

### Issue 3: GAT skip（グラフ未実行）

**症状**:
```
# RFI56ログにて
deg_avg=0.0 isolates=1.0
gat_gate_mean=nan
```

**原因**: edge_index/edge_attrが供給されていない

**対処**:
```bash
# グラフビルダー確認
grep "graph_builder\|edge_index" _logs/train_p03_quick.log

# グラフキャッシュ確認
ls -lh output/graph_cache/

# 手動でグラフビルド
python scripts/build_graph_cache.py --start-date 2024-01-01 --end-date 2025-01-31
```

### Issue 4: RFI56ログが出ない

**原因**: train_atft.py へのログ統合未実施

**対処**: 次セクション参照（train_atft.py修正）

---

## 📝 次のステップ（成功後）

### A. RFI-5/6データ共有

以下を提出:
```
P03_RFI_SUBMISSION.md
rfi_56_metrics.txt
graph_stats_sample.txt
```

### B. P0-4/6/7実装依頼

RFI-5/6データに基づいて:
- **P0-4**: Loss rebalancing (Sharpe/RankIC/CS_IC weights)
- **P0-6**: Quantile crossing penalty (qx_rate > 0.05の場合)
- **P0-7**: Sharpe EMA decay tuning (バッチノイズ抑制)

### C. 環境安定化（後日）

時間を見てB-1案実施:
- PyTorch 2.8.0+cu128 降格
- PyG実装（GATv2Conv）使用
- 性能向上（60-80% → 100%）

---

## 🎯 成功判定基準（再掲）

**Minimum viable success**:
- [x] 3 epoch完走（segfault/OOM なし）
- [x] `RFI56 |` ログ出力（3行）
- [x] `gat_gate_mean` 範囲内（0.2-0.7）
- [x] `deg_avg` 範囲内（10-40）

**これだけでP0-4/6/7に進めます！**

---

**作成**: 2025-11-02
**最終更新**: 2025-11-02
**想定所要時間**: 15-20分（成功時）/ 60分（B-1案必要時）
