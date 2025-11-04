# P0-3: クイックスタートガイド

**P0-3実装**: ✅ 完了（GAT勾配フロー復旧）
**環境問題**: ⚠️ PyG segfault（PyTorch 2.9.0+cu128）
**解決策**: ✅ 実装済み（A案: 安全シム / B案: 環境修正）

---

## 🚀 今すぐ実行（A案: 安全シム）

### 1. RFI-5/6データ収集（推奨）

```bash
# Shim mode で3-epoch学習
USE_GAT_SHIM=1 make train-quick EPOCHS=3
```

**期待されるログ**:
```
[P0-3 GAT-FALLBACK] Using GraphConvShim (PyG-free mode).
Performance: ~60-80% of PyG, suitable for RFI-5/6 collection.
```

### 2. メトリクス抽出

```bash
# Gate統計（P0-3特有）
grep "gat_gate_mean" _logs/training/train_*.log
# 期待: gat_gate_mean=0.2-0.7, gat_gate_std=0.05-0.30

# グラフ統計（RFI-5）
grep -E "deg_avg|isolates" _logs/training/train_*.log
# 期待: deg_avg=10-40, isolates < 2%

# Loss統計（RFI-6）
grep -E "Sharpe_EMA|RankIC|CRPS|WQL" _logs/training/train_*.log
# 期待: Sharpe_EMA > 0.8, RankIC > 0.15
```

### 3. RFI-5/6報告

以下をご共有ください:
- `gat_gate_mean/std` (1-3 epoch平均)
- `deg_avg/isolates` (任意1バッチ)
- `Sharpe_EMA / RankIC / CRPS / quantile_crossing_rate` (各epoch)

---

## 🔧 環境修正（B-1案: PyTorch降格）

### いつやる？
- RFI-5/6収集後、時間がある時
- 本番学習前（PyG実装で最高性能）

### 手順

```bash
# 1. PyTorch 2.8.0+cu128 に降格
pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0+cu128

# 2. PyG + 拡張
pip install torch_geometric
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.8.0+cu128.html

# 3. 確認
python -c "from torch_geometric.nn import GATv2Conv; print('✅ PyG OK')"

# 4. PyG実装で学習（USE_GAT_SHIM未設定）
make train-quick EPOCHS=3

# 5. ログで確認
grep "P0-3 GAT-INIT" _logs/training/train_*.log
# 期待: "Using PyG GATv2Conv (full GAT implementation)"
```

---

## 📊 性能比較

| モード | コマンド | 速度 | 精度 | 用途 |
|--------|----------|------|------|------|
| **Shim** | `USE_GAT_SHIM=1 make train-quick` | 60-80% | 良好 | RFI収集 |
| **PyG** | `make train-quick` (B-1実施後) | 100% | 最高 | 本番学習 |

**注**: どちらもゲート付き残差融合（P0-3の核心）は完全機能

---

## 🆘 トラブルシューティング

### Q1: `USE_GAT_SHIM=1`でもエラーが出る

```bash
# 診断実行
python scripts/diagnose_pyg_environment.py

# エラーログ確認
tail -100 _logs/training/train_*.log
```

### Q2: ゲート統計が出ない

**原因**: GAT未実行（グラフデータなし）
**対処**:
```bash
# グラフビルダー確認
grep "graph_builder" _logs/training/train_*.log
# edge_indexが渡されているか確認
```

### Q3: PyG降格後もsegfault

```bash
# torch/PyG バージョン確認
python -c "import torch, torch_geometric; print(f'torch={torch.__version__}, PyG={torch_geometric.__version__}')"

# 拡張の再インストール
pip uninstall -y pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
```

---

## 📁 関連ファイル

**実装**:
- `src/atft_gat_fan/models/components/gat_shim.py` - GraphConvShim実装
- `src/atft_gat_fan/models/components/gat_fuse.py` - フォールバック機能
- `configs/atft/gat/default.yaml` - GAT設定

**ドキュメント**:
- `P0_3_COMPLETION_REPORT.md` - 完了報告（詳細）
- `P0_3_PyG_ENVIRONMENT_SOLUTIONS.md` - 環境問題解決策（詳細）
- `P0_3_QUICK_START.md` - このファイル（簡潔）

**診断**:
- `scripts/diagnose_pyg_environment.py` - 環境診断
- `scripts/test_gat_shim_mode.py` - Shimモードテスト

---

## ✅ チェックリスト

- [ ] `USE_GAT_SHIM=1 make train-quick EPOCHS=3` 実行
- [ ] ログでShimモード確認（`GAT-FALLBACK`メッセージ）
- [ ] RFI-5/6メトリクス抽出
- [ ] RFI-5/6データ報告
- [ ] （後日）PyTorch 2.8.0降格（B-1案）
- [ ] （後日）PyG実装で学習確認

---

**作成**: 2025-11-02
**最終更新**: 2025-11-02
**ステータス**: 即座実行可能
