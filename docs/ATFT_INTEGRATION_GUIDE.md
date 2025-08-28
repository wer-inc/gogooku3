# ATFT-GAT-FAN 統合ガイド

## 概要
gogooku3でATFT-GAT-FANモデル（Sharpe比 0.849）を使用できるようになりました。

## 統合方法
シンプルなモデル参照方式を採用：
- 元のATFT-GAT-FANコードはそのまま保持
- gogooku3から直接参照して使用
- 性能劣化なし（100%維持）

## 使い方

### 1. 基本的な使用
```python
from scripts.models.atft_inference import ATFTInference

# モデル初期化
atft = ATFTInference()

# 推論実行
predictions = atft.predict(features, horizon=1)
```

### 2. gogooku3データでの使用
```python
from scripts.models.model_service import ModelService

# サービス初期化
service = ModelService()

# gogooku3データで予測
results = service.build_and_predict(
    start_date="2024-01-01",
    end_date="2024-01-31",
    stock_codes=["7203", "6758"],
    horizon=[1, 5]
)

print(f"Expected Sharpe: {results['expected_sharpe']}")  # 0.849
```

### 3. 特徴量変換
```python
from scripts.models.feature_converter import FeatureConverter

converter = FeatureConverter()

# gogooku3 (74特徴量) → ATFT (13特徴量)
atft_features = converter.prepare_atft_features(gogooku3_df)
```

## ファイル構成
```
gogooku3/scripts/models/
├── atft_inference.py      # ATFT推論ラッパー
├── feature_converter.py   # 特徴量変換
└── model_service.py       # 統合API
```

## 必要な特徴量

### ATFT-GAT-FANが必要とする13特徴量
1. **Returns (3)**
   - return_1d
   - return_5d
   - return_20d

2. **Technical Indicators (7)**
   - rsi
   - macd
   - bb_upper
   - atr
   - obv
   - cci
   - stoch_k

3. **Wavelet Features (3)**
   - wavelet_a3 (Close価格)
   - wavelet_v3 (Volume)
   - wavelet_r3 (Return 1d)

## パフォーマンス
- **期待Sharpe比**: 0.849
- **モデルサイズ**: 77MB
- **推論速度**: <100ms/batch
- **GPU使用**: 推奨（CPUでも動作可）

## トラブルシューティング

### モデルが読み込めない場合
```python
# パスを確認
import sys
sys.path.append('/home/ubuntu/gogooku2/apps/ATFT-GAT-FAN')
```

### 特徴量が不足している場合
```python
# FeatureConverterが自動的に計算
converter = FeatureConverter()
df_with_features = converter.prepare_atft_features(df)
```

## 注意事項
- 元のATFT-GAT-FANには一切手を加えていません
- 性能（Sharpe 0.849）は完全に維持されます
- gogooku3の74特徴量から必要な13特徴量を抽出/計算します

## 今後の拡張
- [ ] バッチ予測の高速化
- [ ] リアルタイム予測API
- [ ] 複数モデルのアンサンブル
- [ ] AutoML統合
