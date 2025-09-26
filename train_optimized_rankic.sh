#!/bin/bash

# RankIC改善のための最適化学習スクリプト
# 目標: Val RankIC 0.075+

echo "============================================================"
echo "🚀 Optimized Training for RankIC Improvement"
echo "Target: Val RankIC > 0.075 (Baseline: 0.0813)"
echo "============================================================"

# 環境変数設定
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0

# データパス（future returnsを使用）
DATA_PATH="/home/ubuntu/gogooku3-standalone/output/ml_dataset_future_returns.parquet"
ATFT_DIR="/home/ubuntu/gogooku3-standalone/output/atft_data"

# 主要改善点：
# 1. 学習率を高く（5e-4）
# 2. RankIC損失の重み増加（0.5）
# 3. バッチサイズ最適化（2048）
# 4. hidden_size増加（256）
# 5. Feature clipping有効化

# ハイパーパラメータ
export HIDDEN_SIZE=256              # モデル容量増加
export BATCH_SIZE=2048              # バッチサイズ最適化
export LEARNING_RATE=5e-4           # 学習率を高く
export MAX_EPOCHS=150               # エポック数増加

# 損失関数の重み（RankIC重視）
export USE_RANKIC=1
export RANKIC_WEIGHT=0.5            # RankIC重み増加（0.3→0.5）
export SHARPE_WEIGHT=0.2            # Sharpe重み減少
export CS_IC_WEIGHT=0.2             # Cross-sectional IC
export USE_HUBER=1                  # Huber loss for stability
export HUBER_WEIGHT=0.1

# データ処理
export FEATURE_CLIP_VALUE=10.0      # 異常値クリッピング
export MIN_TRAINING_DATE="2018-01-01"
export OUTPUT_NOISE_STD=0.01        # ノイズ追加（過学習防止）

# DataLoader最適化
export NUM_WORKERS=4                # 並列化
export ALLOW_UNSAFE_DATALOADER=1    # マルチワーカー許可
export PERSISTENT_WORKERS=1         # ワーカー永続化

# 安定性設定
export DEGENERACY_GUARD=1           # Degeneracy防止
export GRADIENT_CLIP_VAL=1.0        # 勾配クリッピング

# Phase Training設定（段階的学習）
export PHASE_TRAINING=1
export PHASE_LOSS_WEIGHTS="{'baseline': {'mse': 0.8, 'rank_ic': 0.2}, 'adaptive': {'mse': 0.5, 'rank_ic': 0.5}, 'gat': {'mse': 0.3, 'rank_ic': 0.7}, 'finetune': {'mse': 0.2, 'rank_ic': 0.8}}"

echo "Configuration:"
echo "  DATA: $DATA_PATH"
echo "  HIDDEN_SIZE: $HIDDEN_SIZE"
echo "  BATCH_SIZE: $BATCH_SIZE"
echo "  LEARNING_RATE: $LEARNING_RATE"
echo "  RANKIC_WEIGHT: $RANKIC_WEIGHT"
echo "  NUM_WORKERS: $NUM_WORKERS"

# 学習実行
python scripts/train_atft.py \
    data.source.data_dir=$ATFT_DIR \
    model.hidden_size=$HIDDEN_SIZE \
    train.batch.train_batch_size=$BATCH_SIZE \
    train.optimizer.lr=$LEARNING_RATE \
    train.trainer.max_epochs=$MAX_EPOCHS \
    train.trainer.precision=bf16-mixed \
    train.trainer.gradient_clip_val=$GRADIENT_CLIP_VAL \
    train.trainer.check_val_every_n_epoch=1 \
    train.trainer.enable_progress_bar=true \
    train.early_stopping.monitor="val/rank_ic_5d" \
    train.early_stopping.mode="max" \
    train.early_stopping.patience=20 \
    train.scheduler.type="plateau" \
    train.scheduler.factor=0.5 \
    train.scheduler.patience=5 \
    data.loader.num_workers=$NUM_WORKERS \
    data.loader.persistent_workers=true \
    2>&1 | tee training_optimized_rankic.log

echo "============================================================"
echo "✅ Training completed"
echo "Check training_optimized_rankic.log for results"
echo "============================================================"