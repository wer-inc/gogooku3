# ATFT-GAT-FAN 重要学習環境変数

## 概要
ATFT-GAT-FANモデルで**Sharpe比 0.849**を達成した際の環境変数設定。
これらの設定は学習の安定性と性能に直接影響する極めて重要な要素です。

## 成功実績
- **達成Sharpe比**: 0.849
- **学習日時**: 2025-08-25
- **ログファイル**: `runs/last/train_fixed_v2.log`
- **使用スクリプト**: `scripts/train_gat_fixed.sh`

## 🔴 最重要環境変数

### 1. 退行防止（Degeneracy Prevention）
モデルが単一値に収束する「崩壊」を防ぐ設定。**これがないと学習が失敗します。**

```bash
export DEGENERACY_GUARD=1          # 退行ガード有効【必須】
export DEGENERACY_ABORT=0          # 警告のみ（中断しない）
export DEGENERACY_WARMUP_STEPS=500 # ウォームアップステップ数
export DEGENERACY_CHECK_EVERY=100  # チェック頻度
export DEGENERACY_MIN_RATIO=0.10   # 最小分散比率
```

### 2. 予測分散制御（Prediction Variance Control）
予測の多様性を維持し、リスク管理に必要な予測幅を確保。

```bash
export PRED_VAR_MIN=0.01           # 最小分散【重要】
export PRED_VAR_WEIGHT=1.0         # 分散ペナルティ重み
export PRED_STD_MIN=0.1            # 標準偏差最小値
```

### 3. ヘッドノイズ（Head Noise Injection）
初期学習の安定化のためのノイズ注入。

```bash
export HEAD_NOISE_STD=0.02         # ウォームアップ時のノイズ
export HEAD_NOISE_WARMUP_EPOCHS=2  # ウォームアップエポック数
export OUTPUT_NOISE_STD=0.02       # 出力ノイズ
```

## 🟡 性能最適化環境変数

### 4. データローダー最適化
GPU使用率85-90%を実現し、学習速度を2-3倍に向上。

```bash
export NUM_WORKERS=16              # ワーカー数【重要】
export PREFETCH_FACTOR=4           # プリフェッチファクター
export PIN_MEMORY=1                # GPUメモリピンニング
export PERSISTENT_WORKERS=1        # ワーカー永続化
```

### 5. Mixed Precision設定
BF16混合精度による高速化と数値安定性の両立。

```bash
export USE_AMP=1                   # AMP有効
export AMP_DTYPE=bf16              # BF16使用（A100最適）
```

### 6. GAT融合制御
Graph Attention Networkの段階的統合。

```bash
export GAT_ALPHA_INIT=0.5          # 初期α値
export GAT_ALPHA_MIN=0.1           # 最小α値（GAT寄与下限）
export GAT_ALPHA_PENALTY=1e-2      # α正則化
export SPARSITY_LAMBDA=0.001       # グラフスパース化
```

## 🟢 学習戦略環境変数

### 7. 段階的学習（Phased Training）
安定した学習のための段階的アプローチ。

```bash
# Phase 1: 初期ウォームアップ（最初の2エポック）
export FUSE_FORCE_MODE=tft_only    # TFTのみ使用
export FORCE_MODE_EPOCHS=2         # 強制モードエポック数
export EDGE_DROPOUT_INPUT_P=0.0    # エッジドロップアウトなし

# Phase 2: GAT統合（3エポック目以降）
unset FUSE_FORCE_MODE              # 融合モード解除
export EDGE_DROPOUT_INPUT_P=0.1    # エッジドロップアウト開始

# Phase 3: 完全学習（安定後）
export HEAD_NOISE_STD=0.0          # ノイズ除去
export USE_T_NLL=1                 # Student-t分布有効化
```

### 8. 損失関数設定
マルチホライズン予測のための損失関数調整。

```bash
export USE_T_NLL=0                 # Student-t NLL（初期は無効）
export USE_PINBALL=1               # Pinball loss有効
export NLL_WEIGHT=0.7              # NLL重み
export PINBALL_WEIGHT=0.3          # Pinball重み
export HWEIGHTS="1:0.6,2:0.15,3:0.1,5:0.1,10:0.05"  # ホライズン重み
```

### 9. バッチサイズとラベル処理
安定性を重視した設定。

```bash
export BATCH_SIZE=256              # 小さめバッチで安定性確保
export GRAD_ACCUM_STEPS=1          # 勾配累積なし
export LABEL_CLIP_BPS_MAP="1:2000,2:2000,3:2000,5:2000,10:5000"  # ラベルクリッピング
```

### 10. 評価設定
高速な評価のための設定。

```bash
export EVAL_MAX_BATCHES=16         # 評価バッチ数制限
export EVAL_MIN_VALID_RATIO=0.6    # 最小有効データ率
```

## 使用方法

### gogooku3での適用方法

1. **環境変数スクリプトの実行**
```bash
source /home/ubuntu/gogooku2/apps/gogooku3/configs/atft_success_env.sh
```

2. **Pythonコード内での設定**
```python
import os

# 重要な環境変数を設定
os.environ["DEGENERACY_GUARD"] = "1"
os.environ["PRED_VAR_MIN"] = "0.01"
os.environ["NUM_WORKERS"] = "16"

# モデル初期化（環境変数が自動適用される）
from scripts.models.atft_inference import ATFTInference
atft = ATFTInference()
```

3. **学習実行時の適用**
```bash
# 環境変数を設定してから実行
export DEGENERACY_GUARD=1
export PRED_VAR_MIN=0.01
python train.py
```

## チェックリスト

学習開始前に以下を確認：

- [ ] `DEGENERACY_GUARD=1` が設定されている
- [ ] `PRED_VAR_MIN` が0.01以上に設定されている
- [ ] `NUM_WORKERS` がCPUコア数に応じて設定されている
- [ ] `USE_AMP=1` でMixed Precisionが有効
- [ ] Phase 1では `FUSE_FORCE_MODE=tft_only`

## トラブルシューティング

### 問題: 予測が単一値に収束
- **解決**: `DEGENERACY_GUARD=1`、`PRED_VAR_MIN`を増やす

### 問題: GPU使用率が低い
- **解決**: `NUM_WORKERS`を増やす、`PREFETCH_FACTOR`を調整

### 問題: 学習が不安定
- **解決**: `BATCH_SIZE`を減らす、`HEAD_NOISE_STD`を増やす

### 問題: メモリ不足
- **解決**: `BATCH_SIZE`を減らす、`NUM_WORKERS`を減らす

## 参考資料

- 元の成功ログ: `/home/ubuntu/gogooku2/apps/ATFT-GAT-FAN/runs/last/train_fixed_v2.log`
- 設定スクリプト: `/home/ubuntu/gogooku2/apps/ATFT-GAT-FAN/scripts/train_gat_fixed.sh`
- CLAUDE.md: `/home/ubuntu/gogooku2/apps/ATFT-GAT-FAN/docs/CLAUDE.md`

## 重要度レベル

- 🔴 **必須**: これがないと学習が失敗する
- 🟡 **推奨**: 性能と速度に大きく影響
- 🟢 **オプション**: 状況に応じて調整

最も重要なのは**退行防止**と**予測分散制御**です。これらの設定により、安定した学習と高い性能（Sharpe比 0.849）を実現できます。
