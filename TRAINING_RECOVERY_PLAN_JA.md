# ATFT-GAT-FAN 学習復旧プラン

## 🚨 現状の問題

### 主要な問題点
1. **予測退化**: モデルが定数予測（yhat_std=0.0）
2. **NaN値の発生**: Validation targetsに大量のNaN
3. **GATバイパス**: BYPASS_GAT_COMPLETELY=1 が設定されている
4. **古い実行**: 最終実行から3日経過（2025-10-31）

### 修正済み（2025-11-03）
- ✅ ゼロロスバグ（`train_epoch`で予測辞書の抽出ミス）
- ✅ ミニトレーニングOOM（GPUキャッシュクリア不足）

---

## 📝 復旧手順

### Step 1: 環境変数の確認と修正

```bash
# 現在の設定を確認
grep -E "BYPASS_GAT|DISABLE_GRAPH" .env

# GATバイパスを無効化（推奨）
# .env を編集:
# BYPASS_GAT_COMPLETELY=0  # または削除
# DISABLE_GRAPH_BUILDER=0  # または削除
```

**理由**: GATは重要な特徴抽出器なので、バイパスすると性能が大幅に低下します。

---

### Step 2: データセットの健全性チェック

```bash
# NaN値の原因を特定
python -c "
import polars as pl
df = pl.read_parquet('output/ml_dataset_latest_full.parquet')
print('Dataset shape:', df.shape)
print('\\nNaN counts per column:')
for col in ['horizon_1', 'horizon_5', 'horizon_10', 'horizon_20']:
    if col in df.columns:
        null_count = df[col].null_count()
        print(f'{col}: {null_count:,} NaNs ({null_count/len(df)*100:.2f}%)')
"
```

**期待される結果**: NaN率は5%以下が理想

---

### Step 3: クイックスモークテスト（修正版で）

```bash
# 私の修正が適用された状態で3エポック実行
make train-quick EPOCHS=3

# または直接実行:
BYPASS_GAT_COMPLETELY=0 \
DISABLE_GRAPH_BUILDER=0 \
python scripts/integrated_ml_training_pipeline.py \
  --max-epochs 3 \
  --batch-size 1024 \
  --lr 2e-4 \
  --data-path output/ml_dataset_latest_full.parquet
```

**チェックポイント**:
- ✅ Loss > 0 （ゼロでない）
- ✅ `yhat_std > 0.001` （予測に分散がある）
- ✅ 「No matching horizons found」エラーが出ない
- ✅ GPU使用率 > 50%

---

### Step 4: ログの確認

```bash
# 最新のログをリアルタイムで確認
tail -f _logs/training/train_*.log | grep -E "loss|Loss|yhat_std|SCALE"

# 期待される出力例:
# train/total_loss: 0.345  # ← ゼロでない
# yhat_std=0.012          # ← ゼロでない
# SCALE(yhat/y)=0.85      # ← ゼロでない
```

---

### Step 5: 問題が続く場合の診断

#### Option A: セーフモードでの実行

```bash
# 安定したシングルワーカーモード
FORCE_SINGLE_PROCESS=1 \
BYPASS_GAT_COMPLETELY=0 \
make train-safe EPOCHS=10
```

#### Option B: 予測ヘッドの診断

```python
# scripts/diagnose_prediction_head.py
import torch
from src.atft_gat_fan.models.architectures.atft_gat_fan import ATFT_GAT_FAN
from omegaconf import OmegaConf

# モデルをロード
config = OmegaConf.load('configs/atft/config_production_optimized.yaml')
model = ATFT_GAT_FAN(config)

# ダミー入力で予測を確認
dummy_input = torch.randn(32, 20, 83)  # [batch, time, features]
output = model({"features": dummy_input})

print("Output keys:", output.keys())
if "predictions" in output:
    preds = output["predictions"]
    for k, v in preds.items():
        print(f"{k}: shape={v.shape}, mean={v.mean():.6f}, std={v.std():.6f}")
```

**期待される結果**: 各horizonでstd > 0.001

---

## 🎯 成功の指標

### 最小限の成功基準（3エポック後）
- ✅ Train loss: 0.1 ~ 0.5 範囲
- ✅ Val loss: 0.1 ~ 0.5 範囲
- ✅ `yhat_std > 0.001` (全horizon)
- ✅ IC (Information Coefficient) > 0.01
- ✅ GPU使用率 > 50%

### 良好な学習の指標（10エポック後）
- ✅ Train loss: 減少傾向
- ✅ Val loss: 減少傾向（過学習なし）
- ✅ IC@1d > 0.05
- ✅ Sharpe > 0.1

---

## 🔧 トラブルシューティング

### 問題: 予測分散がまだゼロ

**原因候補**:
1. 予測ヘッドの初期化が不適切
2. 学習率が低すぎる（勾配がほぼゼロ）
3. 正規化層が予測を潰している

**対策**:
```bash
# 学習率を上げてテスト
make train-quick EPOCHS=3 LR=5e-4

# または予測ヘッドの初期化を確認
python scripts/diagnose_prediction_head.py
```

### 問題: NaN値が大量発生

**原因候補**:
1. データセットにNaNが含まれている
2. 正規化で除算ゼロが発生
3. 極端な外れ値

**対策**:
```bash
# データセットを再生成（NaN除去）
make dataset-rebuild START=2020-01-01 END=2025-01-01

# またはNaN値をフィルタリング
python scripts/filter_dataset_quality.py \
  --input output/ml_dataset_latest_full.parquet \
  --output output/ml_dataset_cleaned.parquet \
  --max-nan-ratio 0.05
```

### 問題: CUDA OOM

**対策**:
```bash
# バッチサイズを減らす
make train-quick EPOCHS=3 BATCH_SIZE=512

# またはセーフモード
make train-safe EPOCHS=10
```

---

## 📊 期待される学習曲線

### 正常な学習パターン
```
Epoch 1: Train Loss=0.450, Val Loss=0.420, IC@1d=0.015
Epoch 3: Train Loss=0.320, Val Loss=0.310, IC@1d=0.035
Epoch 5: Train Loss=0.250, Val Loss=0.260, IC@1d=0.055
Epoch 10: Train Loss=0.180, Val Loss=0.200, IC@1d=0.080
```

### 異常なパターン
```
❌ Epoch 1-10: Loss=0.0005 (変化なし) → 予測退化
❌ Epoch 1-10: IC=0.0000 → 予測に情報なし
❌ Epoch 5: Train Loss=0.1, Val Loss=0.5 → 過学習
```

---

## 🚀 次のステップ

1. **即座に**: Step 1-3を実行してスモークテスト
2. **24時間以内**: 問題が続く場合はStep 5の診断
3. **48時間以内**: データセット再生成を検討

---

**作成日**: 2025-11-03
**ステータス**: 修正済み（テスト待ち）
**優先度**: P0（最重要）
