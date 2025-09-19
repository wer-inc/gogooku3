# GPU学習の永続化設定

## 概要
最新のデータセットで自動的にGPU学習を実行する設定が完了しました。

## 🚀 使用方法

### 基本コマンド（推奨）
```bash
# 最新データセットで自動GPU学習
make train-gpu-latest

# SafeTrainingPipeline検証付き
make train-gpu-latest-safe
```

### 直接スクリプト実行
```bash
# 標準GPU学習
./scripts/train_gpu_latest.sh

# 検証付き
./scripts/train_gpu_latest.sh --safe

# カスタムパラメータ
./scripts/train_gpu_latest.sh "" --adv-graph-train 2e-4 75
```

### CLIコマンド（詳細制御）
```bash
# GPU環境設定（自動適用済み）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# 統合パイプライン
python scripts/integrated_ml_training_pipeline.py \
  --data-path output/datasets/ml_dataset_20250319_20250919_20250919_223415_full.parquet \
  --adv-graph-train \
  train.optimizer.lr=2e-4 \
  train.trainer.max_epochs=75
```

## ⚙️ 永続化設定

### 1. 環境変数（.env）
```bash
# GPU設定（永続化済み）
FORCE_GPU=1
REQUIRE_GPU=1
USE_GPU_ETL=1
RMM_POOL_SIZE=70GB
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# GPU学習デフォルト設定
GPU_TRAINING_ENABLED=1
DEFAULT_LEARNING_RATE=2e-4
DEFAULT_MAX_EPOCHS=75
ADV_GRAPH_TRAIN=1
```

### 2. 自動データセット検出
スクリプトは以下の順で最新データセットを自動検出：
1. `output/datasets/ml_dataset_*_full.parquet`
2. `output/ml_dataset_*_full.parquet`

### 3. Makefileターゲット
```makefile
train-gpu-latest         # 標準GPU学習
train-gpu-latest-safe    # SafeTrainingPipeline検証付き
```

## 📊 パフォーマンス設定

### GPU最適化
- **メモリ拡張**: `expandable_segments:True`でOOM回避
- **バッチサイズ**: 自動調整（OOM時に縮小）
- **混合精度**: bf16使用でメモリ効率向上

### 学習パラメータ
- **学習率**: 2e-4（デフォルト）
- **エポック数**: 75（デフォルト）
- **グラフ学習**: Advanced Graph Training有効

## 🔍 確認方法

### GPU使用状況
```bash
# リアルタイムモニタリング
watch -n 1 nvidia-smi

# 学習ログ確認
tail -f logs/ml_training.log
```

### データセット確認
```bash
# 最新データセット表示
ls -lht output/datasets/ml_dataset_*_full.parquet | head -1
```

## トラブルシューティング

### データセットが見つからない場合
```bash
# データセット生成（GPU-ETL使用）
make dataset-full-gpu START=2020-09-19 END=2025-09-19
```

### CUDA Out of Memory
```bash
# バッチサイズを減らす
python scripts/integrated_ml_training_pipeline.py \
  --data-path <dataset> \
  data.batch.batch_size=256
```

### GPU が検出されない
```bash
# CUDA確認
python -c "import torch; print(torch.cuda.is_available())"
```

## 📈 期待される性能
- **データセット**: 480,973行 × 395列（6ヶ月）
- **学習時間**: A100で約2-3時間（75エポック）
- **目標Sharpe**: 0.849
- **RankIC@1d**: 0.180以上