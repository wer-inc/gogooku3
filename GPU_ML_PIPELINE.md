# GPU ML Pipeline - 完全ガイド

## 📋 概要

本ドキュメントは、gogooku3-standaloneにおけるGPU環境での機械学習パイプライン（データセット作成から学習まで）の完全ガイドです。
すべての設定は永続化済みで、簡単なコマンドで実行可能です。

## 🚀 クイックスタート

```bash
# 1. データセット作成（GPU-ETL使用）
make dataset-full-gpu START=2020-09-19 END=2025-09-19

# 2. GPU学習（最新データセット自動検出）
make train-gpu-latest

# 3. モニタリング
watch -n 1 nvidia-smi
```

## 📊 データセット作成（GPU-ETL）

### デフォルト設定（v2.1.0以降）

**GPU-ETLはデフォルトで有効になっています。** 特別な設定は不要です。

### 実行方法

#### 1. 標準コマンド（GPU-ETL自動有効）
```bash
# Makefileターゲット
make dataset-full START=2020-09-19 END=2025-09-19

# 直接実行
python scripts/pipelines/run_full_dataset.py \
  --jquants --start-date 2020-09-19 --end-date 2025-09-19
```

#### 2. 明示的GPU版コマンド
```bash
# GPU-ETL強制有効
make dataset-full-gpu START=2020-09-19 END=2025-09-19

# ラッパースクリプト（デフォルト6ヶ月）
./scripts/run_dataset_gpu.sh

# カスタム期間指定
./scripts/run_dataset_gpu.sh --start-date 2020-09-19 --end-date 2025-09-19
```

#### 3. 研究用設定（インデックス付き）
```bash
make dataset-full-research START=2020-09-19 END=2025-09-19
```

### GPU-ETLで高速化される処理

1. **Cross-sectional normalization**: 日次断面での正規化
2. **Rank computation**: ランク計算
3. **Z-score calculation**: Z-スコア計算
4. **Correlation matrix**: 相関行列計算

### パフォーマンス目安

- **小規模（1-2日）**: CPUの方が高速な場合あり
- **中規模（1週間-1ヶ月）**: GPU-ETLで1.5-3倍高速化
- **大規模（6ヶ月以上）**: GPU-ETLで3-5倍高速化

## 🧠 GPU学習

### 最新データセット自動検出機能

`train_gpu_latest.sh`は以下の順で最新データセットを自動検出：
1. `output/datasets/ml_dataset_*_full.parquet`
2. `output/ml_dataset_*_full.parquet`

### 実行方法

#### 1. 推奨コマンド ✨
```bash
# 基本GPU学習
make train-gpu-latest

# SafeTrainingPipeline検証付き
make train-gpu-latest-safe
```

#### 2. スクリプト直接実行
```bash
# 標準実行
./scripts/train_gpu_latest.sh

# 検証付き
./scripts/train_gpu_latest.sh --safe

# カスタムパラメータ
./scripts/train_gpu_latest.sh "" --adv-graph-train 2e-4 75
```

#### 3. 環境変数によるカスタマイズ
```bash
# バッチサイズ調整
TRAIN_BATCH_SIZE=4096 \
TRAIN_VAL_BATCH_SIZE=6144 \
./scripts/train_gpu_latest.sh

# ワーカー数調整
TRAIN_NUM_WORKERS=16 \
TRAIN_PREFETCH=8 \
./scripts/train_gpu_latest.sh

# 精度設定
TRAIN_PRECISION=16-mixed \
./scripts/train_gpu_latest.sh
```

#### 4. 詳細CLI（フル制御）
```bash
python scripts/integrated_ml_training_pipeline.py \
  --data-path output/datasets/ml_dataset_20250319_20250919_20250919_223415_full.parquet \
  --adv-graph-train \
  train.batch.train_batch_size=4096 \
  train.batch.val_batch_size=6144 \
  train.batch.num_workers=16 \
  train.batch.prefetch_factor=8 \
  train.trainer.accumulate_grad_batches=1 \
  train.trainer.precision=16-mixed \
  train.optimizer.lr=2e-4 \
  train.trainer.max_epochs=75
```

## ⚙️ GPU最適化設定

### 環境変数（永続化済み）

#### .envファイル
```bash
# GPU基本設定
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

#### train_gpu_latest.sh内の最適化設定（改良版）
```bash
# メモリ最適化
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

# cuDNN最適化
TORCH_CUDNN_V8_API_ENABLED=1
TORCH_CUDNN_V8_API_ALLOWED=1

# 並列処理最適化
OMP_NUM_THREADS=16
CUDA_DEVICE_MAX_CONNECTIONS=32
NCCL_P2P_LEVEL=SYS

# バッチサイズ（大幅増加）
TRAIN_BATCH_SIZE=4096
TRAIN_VAL_BATCH_SIZE=6144
TRAIN_NUM_WORKERS=16
TRAIN_PREFETCH=8
```

### デフォルトパラメータ

| パラメータ | デフォルト値 | 説明 |
|-----------|------------|------|
| batch_size | 4096 | 訓練バッチサイズ |
| val_batch_size | 6144 | 検証バッチサイズ |
| num_workers | 16 | データローダーワーカー数 |
| prefetch_factor | 8 | プリフェッチ係数 |
| precision | 16-mixed | 混合精度学習 |
| learning_rate | 2e-4 | 学習率 |
| max_epochs | 75 | 最大エポック数 |
| accumulate_grad | 1 | 勾配累積ステップ |

## 📈 パフォーマンス指標

### データセット作成（GPU-ETL）
- **処理速度**: CPU比 3-5倍高速
- **メモリ使用**: RMM pool 70GB
- **A100 80GB**: 6ヶ月データを約30分で処理

### ML学習（GPU）
- **学習時間**: A100で2-3時間（75エポック）
- **スループット**: 5130 samples/sec
- **メモリ使用**: 最大60GB（モデル+データ）

### 期待される精度
- **Target Sharpe**: 0.849
- **RankIC@1d**: 0.180以上
- **Model Parameters**: 5.6M

## 🔄 完全ワークフロー例

### 本番環境向け（6ヶ月データ）
```bash
# 1. データセット作成
make dataset-full-gpu START=2020-09-19 END=2025-09-19

# 2. データ確認
ls -lht output/datasets/ml_dataset_*_full.parquet | head -1

# 3. GPU学習開始
make train-gpu-latest

# 4. モニタリング
# 別ターミナルで
watch -n 1 nvidia-smi
tensorboard --logdir logs/
```

### 研究・実験向け
```bash
# 1. 研究用データセット（インデックス付き）
make dataset-full-research START=2020-09-19 END=2025-09-19

# 2. SafeTrainingPipeline検証付き学習
make train-gpu-latest-safe

# 3. 結果分析
make research-plus DATASET=output/ml_dataset_latest_full.parquet
```

### HPO（ハイパーパラメータ最適化）併用
```bash
# 1. データセット準備
make dataset-full-gpu START=2020-09-19 END=2025-09-19

# 2. HPO実行
make train-integrated-hpo

# 3. 最適パラメータで再学習
make train-gpu-latest
```

## 🔍 モニタリング

### GPU使用状況
```bash
# リアルタイムモニタリング
watch -n 1 nvidia-smi

# GPUメモリのみ
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1

# プロセス確認
nvidia-smi pmon -i 0
```

### 学習進捗
```bash
# ログ監視
tail -f logs/ml_training.log

# TensorBoard
tensorboard --logdir logs/ --port 6006
```

### データセット確認
```bash
# 最新データセット
ls -lht output/datasets/ml_dataset_*_full.parquet | head -1

# メタデータ確認
python -c "import polars as pl; df=pl.scan_parquet('output/datasets/*.parquet'); print(df.collect().shape)"
```

## 🛠️ トラブルシューティング

### データセットが見つからない
```bash
# データセット生成
make dataset-full-gpu START=2020-09-19 END=2025-09-19

# 確認
find output -name "*.parquet" -type f
```

### CUDA Out of Memory
```bash
# バッチサイズを減らす
TRAIN_BATCH_SIZE=2048 \
TRAIN_VAL_BATCH_SIZE=3072 \
./scripts/train_gpu_latest.sh

# またはCLIで
python scripts/integrated_ml_training_pipeline.py \
  --data-path <dataset> \
  train.batch.train_batch_size=2048
```

### GPU が検出されない
```bash
# CUDA確認
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'Devices: {torch.cuda.device_count()}')"

# cuDF確認
python -c "import cudf; print('cuDF OK')"
```

### RMM初期化警告
```
RMM init failed: module 'rmm' has no attribute 'rmm_cupy_allocator'
```
これは正常です。pool allocatorなしでもGPU-ETLは動作します。

## 📊 作成済みデータセット情報

### 最新データセット（2025-09-19時点）
- **ファイル**: `ml_dataset_20250319_20250919_20250919_223415_full.parquet`
- **サイズ**: 480,973行 × 395列
- **期間**: 2025-03-19 ～ 2025-09-19（6ヶ月）
- **銘柄数**: 3,850
- **特徴量**: 359 features + metadata

### 特徴量カテゴリ
- **価格/出来高**: ~70列
- **テクニカル**: ~20列
- **市場（TOPIX）**: ~30列
- **フロー**: ~37列（拡張版）
- **マージン**: ~86列（週次+日次）
- **財務**: ~20列
- **その他**: ~146列

## 📁 関連ファイル

### 設定ファイル
- `.env`: 環境変数設定
- `configs/atft/train/production.yaml`: 本番学習設定
- `configs/pipeline/full_dataset.yaml`: データセット生成設定

### スクリプト
- `scripts/train_gpu_latest.sh`: GPU学習自動実行
- `scripts/run_dataset_gpu.sh`: GPU-ETLデータセット生成
- `scripts/integrated_ml_training_pipeline.py`: 統合MLパイプライン

### ドキュメント
- `GPU_ETL_USAGE.md`: GPU-ETL使用ガイド
- `GPU_TRAINING.md`: GPU学習ガイド
- `docs/ml/dataset_new.md`: データセット仕様（395列）

## 🎯 次のステップ

1. **より長期間のデータ**: 1年以上のデータセット作成
2. **アンサンブル学習**: 複数モデルの組み合わせ
3. **リアルタイム推論**: 学習済みモデルでの予測システム構築
4. **AutoML統合**: Optuna等によるさらなる最適化

---
最終更新: 2025-09-19
バージョン: v2.1.0