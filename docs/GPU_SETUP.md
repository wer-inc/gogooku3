# GPU環境セットアップガイド

## 概要

`scripts/setup_gpu_env.sh`は、ATFT-GAT-FANのdataset生成とtraining両方に必要なGPU環境を1コマンドでセットアップするスクリプトです。

## 実行方法

### オプション1: Makefileターゲット（推奨）

```bash
# GPU環境セットアップ実行
make setup-gpu

# 確認のみ（dry-run）
make check-gpu
```

### オプション2: 直接実行

```bash
# GPU環境セットアップ実行
bash scripts/setup_gpu_env.sh

# 確認のみ（dry-run）
bash scripts/setup_gpu_env.sh --dry-run
```

## セットアップ内容

### Step 1: 環境チェック
- ✅ Python 3.10+ 確認
- ✅ CUDA/GPU の確認
- ✅ CUDA compiler (nvcc) 確認
- ✅ .env ファイルの確認（JQuants認証情報）

### Step 2: 競合パッケージの削除
- ✅ RAPIDS残骸の削除（cudf/cugraph/rmm）
  - PyTorchとの競合を回避

### Step 3: GPU依存パッケージのインストール
- ✅ CuPy (CUDA 12.x用) インストール
- ✅ typing_extensions 更新（pydantic互換性）
- ✅ PyTorch CUDA サポート確認

### Step 4: コード修正 - graph_builder_gpu.py
- ✅ `src/data/utils/graph_builder_gpu.py` のimport文修正
  - cudf/cuGraphをオプショナルimportに変更
  - CuPyのみでGPU高速化を実現

### Step 5: コード修正 - Pipeline cuGraph依存関係
- ✅ `scripts/pipelines/run_full_dataset.py` のpreflight check修正
  - `import cugraph` を削除（CuPyのみチェック）
- ✅ `src/pipeline/full_dataset.py` のgraph builder選択修正
  - `import cugraph` を削除（CuPyのみチェック）

### Step 6: 動作確認テスト
- ✅ PyTorch CUDA 動作確認
- ✅ CuPy GPU 動作確認
- ✅ graph_builder_gpu 動作確認
- ✅ peer特徴量計算の確認

### Step 7: サマリー表示
- ✅ システム構成の表示
- ✅ 次のステップの案内

## ログ出力

セットアップの詳細ログは以下に保存されます：

```
_logs/setup_gpu_env.log
```

## 次のステップ

### 1. Dataset生成（5年分、GPU加速）

```bash
make dataset-bg START=2020-10-11 END=2025-10-10
```

**所要時間**: 30-60分（GPU版）

### 2. Training（GPU使用）

```bash
make train-optimized
```

### 3. クイックテスト

```bash
make smoke
```

## トラブルシューティング

### エラー: "CUDA not available"

```bash
# GPU確認
nvidia-smi

# CUDA確認
nvcc --version
```

### エラー: "cudf/cugraph import error"

→ **正常です**。このスクリプトはcudf/cuGraphなしで動作するように設計されています。

### エラー: "JQuants credentials not found"

```bash
# .envファイルを作成
cp .env.example .env

# 認証情報を設定
vim .env
```

以下を設定：
```
JQUANTS_AUTH_EMAIL=your_email@example.com
JQUANTS_AUTH_PASSWORD=your_password
```

## 環境情報

セットアップ後の環境：

| 項目 | 内容 |
|------|------|
| GPU | NVIDIA H100 PCIe (80GB) |
| Python | 3.10.12 |
| PyTorch | 2.8.0+cu128 |
| CuPy | 13.6.0 |
| CUDA | 12.8/12.9 |

## 備考

### CuPyのみでGPU加速

このセットアップでは**CuPyのみ**を使用してGPU加速を実現します：

- ✅ **メリット**：
  - PyTorchとの競合なし
  - インストールが簡単
  - メモリ効率が良い

- ⚠️ **制約**：
  - cuGraph機能は使用不可（ただし、実際には不要）

### GPU版peer特徴量

GPU版graph_builderは以下の特徴量を計算します：

- `peer_mean_return`: peer群の平均リターン
- `peer_var_return`: peer群のリターン分散
- `peer_count`: peer数
- `peer_correlation_mean`: peer群との平均相関（絶対値）

これらは**CPU版と完全互換**です。

## FAQ

### Q: 何度実行しても安全ですか？

**A**: はい。スクリプトは冪等性を持ち、何度実行しても安全です。

### Q: 既存の環境を壊しませんか？

**A**: いいえ。以下の安全策があります：
- dry-run機能（`--dry-run`）
- バックアップ作成（修正前に`.backup`作成）
- 既存パッケージの確認

### Q: RAPIDSは必要ですか？

**A**: いいえ。このプロジェクトではCuPyのみでGPU加速を実現します。RAPIDSは不要です。

### Q: セットアップにどれくらい時間がかかりますか？

**A**: 通常1-2分です（パッケージが既にインストール済みの場合は10秒程度）。

## 関連ドキュメント

- [Dataset Generation Guide](../Makefile.dataset)
- [Training Guide](../README.md)
- [CLAUDE.md](../CLAUDE.md)
