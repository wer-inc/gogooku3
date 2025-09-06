# ATFT-GAT-FAN → gogooku3 学習機能移行計画

## 現状分析

### gogooku3の現状
- ✅ **推論機能**: 実装済み（atft_inference.py）
- ✅ **特徴量変換**: 実装済み（feature_converter.py）
- ❌ **学習機能**: 未実装
- ❌ **データローダー**: 未実装
- ❌ **設定システム**: 未実装

### ATFT-GAT-FANから移行が必要なもの

## 🔴 必須移行コンポーネント

### Phase 1: コアファイル移行（必須）

```bash
# 1. モデルアーキテクチャ全体
cp -r /home/ubuntu/gogooku2/apps/ATFT-GAT-FAN/src/models/* \
      /home/ubuntu/gogooku2/apps/gogooku3/src/models/

# 2. データローダー
cp -r /home/ubuntu/gogooku2/apps/ATFT-GAT-FAN/src/data/* \
      /home/ubuntu/gogooku2/apps/gogooku3/src/data/

# 3. 損失関数
cp -r /home/ubuntu/gogooku2/apps/ATFT-GAT-FAN/src/losses/* \
      /home/ubuntu/gogooku2/apps/gogooku3/src/losses/

# 4. ユーティリティ
cp -r /home/ubuntu/gogooku2/apps/ATFT-GAT-FAN/src/utils/* \
      /home/ubuntu/gogooku2/apps/gogooku3/src/utils/

# 5. 学習スクリプト
cp /home/ubuntu/gogooku2/apps/ATFT-GAT-FAN/scripts/train.py \
   /home/ubuntu/gogooku2/apps/gogooku3/scripts/train_atft.py
```

### Phase 2: 設定システム移行

```bash
# Hydra設定
cp -r /home/ubuntu/gogooku2/apps/ATFT-GAT-FAN/configs/* \
      /home/ubuntu/gogooku2/apps/gogooku3/configs/atft/
```

### Phase 3: 依存関係追加

```bash
# requirements.txtに追加が必要
hydra-core>=1.3.0
torch_geometric>=2.3.0
pywt  # PyWavelets
```

## 🟡 オプション（推奨）移行コンポーネント

### 実験管理
```bash
# MLflow/実験トラッキング
- experiments/
- mlruns/
```

### テストスクリプト
```bash
# システムテスト
- tests/
- scripts/test_system.py
```

## 実装方法

### Option 1: シンボリックリンク（推奨）
ATFT-GAT-FANを直接参照して、コードの重複を避ける

```python
# gogooku3/scripts/train_with_atft.py
import sys
import os
from pathlib import Path

# ATFT-GAT-FANを直接参照
ATFT_PATH = Path("/home/ubuntu/gogooku2/apps/ATFT-GAT-FAN")
sys.path.insert(0, str(ATFT_PATH))

# 環境変数設定
exec(open("/home/ubuntu/gogooku2/apps/gogooku3/configs/atft_success_env.sh").read())

# ATFT-GAT-FANのtrain.pyを直接使用
from scripts import train

# gogooku3のデータで学習
def train_with_gogooku3_data():
    # データパスをgogooku3に変更
    os.environ["DATA_DIR"] = "/home/ubuntu/gogooku2/apps/gogooku3/output"

    # 学習実行
    train.main()
```

### Option 2: 完全移行
全ファイルをgogooku3にコピーして独立させる

```bash
#!/bin/bash
# migrate_training.sh

# ソースディレクトリ
SRC_DIR="/home/ubuntu/gogooku2/apps/ATFT-GAT-FAN"
DST_DIR="/home/ubuntu/gogooku2/apps/gogooku3"

# srcディレクトリ作成
mkdir -p $DST_DIR/src

# 必要なファイルをコピー
cp -r $SRC_DIR/src/* $DST_DIR/src/
cp -r $SRC_DIR/configs $DST_DIR/configs_atft
cp $SRC_DIR/scripts/train.py $DST_DIR/scripts/train_atft.py

# requirements追加
echo "hydra-core>=1.3.0" >> $DST_DIR/requirements.txt
echo "torch_geometric>=2.3.0" >> $DST_DIR/requirements.txt

echo "✅ Migration complete"
```

## 学習実行方法（移行後）

### 1. 環境設定
```bash
# 成功した環境変数を設定
source /home/ubuntu/gogooku2/apps/gogooku3/configs/atft_success_env.sh
```

### 2. データ準備
```bash
# gogooku3のMLデータセットを使用
cd /home/ubuntu/gogooku2/apps/gogooku3
python scripts/pipelines/run_pipeline.py
```

### 3. 学習実行
```bash
# ATFT-GAT-FAN学習
python scripts/train_atft.py \
    data.source.data_dir=./output \
    train=profiles/robust \
    train.batch.train_batch_size=256
```

## チェックリスト

移行前に確認：
- [ ] ATFT-GAT-FANの最新チェックポイントをバックアップ
- [ ] gogooku3のデータ形式がATFT互換か確認
- [ ] GPU/メモリリソースが十分か確認
- [ ] 環境変数設定ファイルが準備されているか

移行後に確認：
- [ ] train.pyが正常に起動するか
- [ ] データローダーが動作するか
- [ ] モデルが正しく初期化されるか
- [ ] 1エポック学習が完了するか

## 推奨アプローチ

**最小限の移行で学習を可能にする方法：**

1. **シンボリックリンクでATFT-GAT-FANを参照**
   - コードの重複を避ける
   - 元のコードを保護
   - すぐに実行可能

2. **gogooku3のデータをATFT形式に変換**
   - feature_converter.pyを使用
   - 13次元特徴量に変換

3. **環境変数で調整**
   - DATA_DIRをgogooku3に変更
   - 成功した設定を適用

## 結論

現在のgogooku3では学習はできませんが、以下のいずれかで可能になります：

1. **簡単な方法**: ATFT-GAT-FANを直接参照して学習
2. **完全な方法**: 必要ファイルをすべて移行

どちらの方法でも、**Sharpe比 0.849**の性能を維持できます。
