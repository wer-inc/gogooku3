#!/bin/bash
# 最適化された設定で実行するスクリプト

# 1. 予測符号の反転（最重要！）
export INVERT_PREDICTION_SIGN=1

# 2. ターゲット変数の正規化
export TARGET_VOL_NORM=1

# 3. 高度なグラフビルダー
export USE_ADV_GRAPH_TRAIN=1
export GRAPH_K=20  # 15→20に増やす
export GRAPH_EDGE_THR=0.2  # 0.25→0.2に緩和
export EWM_HALFLIFE=20  # 30→20に短縮
export SHRINKAGE_GAMMA=0.15  # 0.1→0.15に増加

# 4. 学習率とスケジューラ
export LR_INITIAL=3e-4  # 2e-4→3e-4に増加
export LR_MIN=1e-5
export LR_PATIENCE=8  # 早期停止を少し遅らせる

# 5. 正則化の強化
export DROPOUT_RATE=0.2  # ドロップアウト率
export WEIGHT_DECAY=5e-4  # L2正則化

# 6. バッチサイズ（大きくして安定化）
export BATCH_SIZE=1024  # 512→1024

# 7. Mixed Precisionの最適化
export PRECISION="bf16-mixed"

# 8. 早期停止の調整
export EARLY_STOP_PATIENCE=12  # 9→12
export EARLY_STOP_MIN_DELTA=5e-4  # 1e-4→5e-4

# 9. 損失関数の重み調整
export LOSS_WEIGHT_HUBER=0.3
export LOSS_WEIGHT_RANKIC=0.35
export LOSS_WEIGHT_SHARPE=0.25
export LOSS_WEIGHT_MAE=0.1

# 10. データ拡張
export DATA_AUGMENT=1
export NOISE_LEVEL=0.01
export MIXUP_ALPHA=0.2

echo "=========================================="
echo "最適化された設定でトレーニングを開始"
echo "=========================================="
echo "主な改善点:"
echo "- 予測符号の反転: ON"
echo "- ターゲット正規化: ON"
echo "- 学習率: 3e-4"
echo "- バッチサイズ: 1024"
echo "- 高度グラフ: K=20"
echo "=========================================="

# 実行（Hydraオーバーライドで有効パラメータを明示指定）
python scripts/train_atft.py \
    data.source.data_dir=output/atft_data/train \
    train.batch.train_batch_size=$BATCH_SIZE \
    train.optimizer.lr=$LR_INITIAL \
    train.optimizer.weight_decay=5e-4 \
    train.trainer.max_epochs=100 \
    train.trainer.precision=$PRECISION \
    early_stopping.monitor=val/sharpe_ratio \
    early_stopping.mode=max \
    checkpoint.monitor=val/sharpe_ratio \
    checkpoint.mode=max \
    model.gat.layer_config.dropout=${DROPOUT_RATE:-0.2} \
    model.prediction_head.architecture.dropout=${DROPOUT_RATE:-0.2}
