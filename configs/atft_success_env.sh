#!/bin/bash

# ATFT-GAT-FAN 成功環境変数設定
# Sharpe比 0.849 を達成した設定

# 🔴 最重要環境変数
export DEGENERACY_GUARD=1          # 退行ガード有効【必須】
export DEGENERACY_ABORT=0          # 警告のみ（中断しない）
export DEGENERACY_WARMUP_STEPS=500 # ウォームアップステップ数
export DEGENERACY_CHECK_EVERY=100  # チェック頻度
export DEGENERACY_MIN_RATIO=0.10   # 最小分散比率

# 予測分散制御
export PRED_VAR_MIN=0.01           # 最小分散【重要】
export PRED_VAR_WEIGHT=1.0         # 分散ペナルティ重み
export PRED_STD_MIN=0.1            # 標準偏差最小値

# ヘッドノイズ
export HEAD_NOISE_STD=0.02         # ウォームアップ時のノイズ
export HEAD_NOISE_WARMUP_EPOCHS=2  # ウォームアップエポック数
export OUTPUT_NOISE_STD=0.02       # 出力ノイズ

# 🟡 性能最適化環境変数
export NUM_WORKERS=16              # ワーカー数【重要】
export PREFETCH_FACTOR=4           # プリフェッチファクター
export PIN_MEMORY=1                # GPUメモリピンニング
export PERSISTENT_WORKERS=1        # ワーカー永続化

# Mixed Precision設定
export USE_AMP=1                   # AMP有効
export AMP_DTYPE=bf16              # BF16使用（A100最適）

# GAT融合制御
export GAT_ALPHA_INIT=0.5          # 初期α値
export GAT_ALPHA_MIN=0.1           # 最小α値（GAT寄与下限）
export GAT_ALPHA_PENALTY=1e-2      # α正則化
export SPARSITY_LAMBDA=0.001       # グラフスパース化

# 🟢 学習戦略環境変数
export FUSE_FORCE_MODE=tft_only    # TFTのみ使用（初期）
export FORCE_MODE_EPOCHS=2         # 強制モードエポック数
export EDGE_DROPOUT_INPUT_P=0.0    # エッジドロップアウトなし

# 損失関数設定
export USE_T_NLL=0                 # Student-t NLL（初期は無効）
export USE_PINBALL=1               # Pinball loss有効
export NLL_WEIGHT=0.7              # NLL重み
export PINBALL_WEIGHT=0.3          # Pinball重み
export HWEIGHTS="1:0.6,2:0.15,3:0.1,5:0.1,10:0.05"  # ホライズン重み

# バッチサイズとラベル処理
export BATCH_SIZE=256              # 小さめバッチで安定性確保
export GRAD_ACCUM_STEPS=1          # 勾配累積なし
export LABEL_CLIP_BPS_MAP="1:2000,2:2000,3:2000,5:2000,10:5000"  # ラベルクリッピング

# 評価設定
export EVAL_MAX_BATCHES=16         # 評価バッチ数制限
export EVAL_MIN_VALID_RATIO=0.6    # 最小有効データ率

echo "✅ ATFT-GAT-FAN 成功環境変数設定完了"
