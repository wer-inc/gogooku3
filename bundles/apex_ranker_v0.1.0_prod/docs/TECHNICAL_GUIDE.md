# APEX Ranker v0.1.0 技術ガイド

**対象**: モデル再訓練、トラブルシューティング、システム拡張を行う技術者

**最終更新**: 2025-11-04

---

## 📋 目次

1. [重要な前提知識](#重要な前提知識)
2. [はまりやすいポイント TOP 5](#はまりやすいポイント-top-5)
3. [モデル訓練の完全ガイド](#モデル訓練の完全ガイド)
4. [データセット互換性](#データセット互換性)
5. [Feature-ABI (特徴量互換性)](#feature-abi-特徴量互換性)
6. [Cross-Sectional Standardization (CS-Z)](#cross-sectional-standardization-cs-z)
7. [トラブルシューティング](#トラブルシューティング)
8. [チェックリスト](#チェックリスト)

---

## 重要な前提知識

### モデルとデータセットの関係

```
データセット (395特徴量)
    ↓
特徴量フィルタリング (89特徴量選択)  ← Feature-ABI
    ↓
CS-Z適用 (89 → 178チャネル)  ← Cross-Sectional Standardization
    ↓
モデル訓練 (178次元で学習)
    ↓
チェックポイント保存 (178次元のweight)
```

**重要**:
- モデルは**178次元**で訓練されている（89特徴量 × 2）
- データセットは**395特徴量**を含むが、モデルは**89特徴量のみ使用**
- 推論時は必ず**特徴量フィルタリング + CS-Z検出**が必要

---

## はまりやすいポイント TOP 5

### 🔴 1. CS-Z設定ミス（最も重大）

**症状**:
```
RuntimeError: size mismatch for encoder.patch_embed.conv.weight:
  copying a param with shape torch.Size([178, 1, 16]) from checkpoint,
  the shape in current model is torch.Size([89, 1, 16]).
```

**原因**:
```yaml
# ❌ 間違い: 明示的にCS-Z OFFを指定
model:
  patch_multiplier: 1  # これを書くとエラー

# ✅ 正解: 何も書かない（auto検出）
model:
  d_model: 256
  # patch_multiplierは書かない
```

**なぜ起こるか**:
- モデルは**CS-Z ONで訓練**されている（178次元）
- `patch_multiplier: 1`を指定すると**CS-Z OFFで初期化**される（89次元）
- 次元が合わずload_state_dict()が失敗

**対策**:
1. **絶対に`patch_multiplier`を明示指定しない**
2. デフォルト値`auto`で自動検出させる
3. チェックポイントから178次元を読み取り、内部で`add_csz=False`を設定

**検証方法**:
```bash
# ログで確認
grep "Model Init" /tmp/backtest.log
# 期待: [Model Init] in_features=89, patch_multiplier=auto, add_csz=False
#                                                           ^^^^^^^^^^^^^^
#                    CS-Z OFF（特徴はすでにチェックポイント内で2倍化済み）
```

---

### 🔴 2. Feature-ABI不一致

**症状**:
```
[Model Init] in_features=354, patch_multiplier=auto
RuntimeError: size mismatch ... torch.Size([708, 1, 16]) ...
```

**原因**:
- データセットは**395特徴量**
- モデルは**89特徴量**で訓練
- 特徴量フィルタリングがスキップされた

**なぜ起こるか**:
```python
# ❌ 間違い: 全特徴量をロード
dataset = pl.read_parquet("ml_dataset_latest_clean_with_adv.parquet")
# → 395特徴量 × 2 (CS-Z) = 790次元 ≠ 178次元

# ✅ 正解: 89特徴量のみ選択
feature_names = load_feature_names("configs/feature_names_v0_latest_89.json")
dataset = dataset.select(["Date", "Code"] + feature_names + target_cols)
# → 89特徴量 × 2 (CS-Z) = 178次元 ✓
```

**対策**:
1. `backtest_smoke_test.py`の特徴量ロジックを参照
2. `feature_names_v0_latest_89.json`を必ず使用
3. Feature-ABIメタデータをチェックポイントに埋め込む（将来改善）

**検証方法**:
```bash
# データセット特徴量数
python -c "import polars as pl; df = pl.read_parquet('output/ml_dataset_latest_clean_with_adv.parquet'); print(len([c for c in df.columns if not c.startswith('target_') and c not in ['Date', 'Code']]))"
# 出力: 395

# モデル期待特徴量数
grep "base_features" bundles/apex_ranker_v0.1.0_prod/MANIFEST.lock
# 出力: "base_features": 89
```

---

### 🟡 3. データセット日付範囲ミスマッチ

**症状**:
```
[Backtest] Date span: 2024-12-24 → 2025-10-03
⚠️  WARNING: Lookback period (180 days) exceeds available data
```

**原因**:
- モデルは**180日のlookback**が必要
- データセット開始日が遅すぎる（2024-12-24など）
- 最初の180日間は推論不可

**対策**:
```python
# 推論開始日の計算
推論開始日 = データセット開始日 + 180日

# 例:
# データセット開始: 2024-01-01
# 推論開始: 2024-06-30 (180日後)
# → 2024-01-01 ~ 2024-06-29は学習データとして使用
```

**推奨データセット期間**:
```
訓練用: 2020-01-01 ~ 2024-12-31 (5年間)
検証用: 2025-01-01 ~ 2025-10-31 (10ヶ月)
```

---

### 🟡 4. Panel Cacheの非互換性

**症状**:
```
FileNotFoundError: cache/panel/ml_dataset_..._lb180_f89_a015bb2ee3.pkl
```

**原因**:
- Panel cacheは**データセットハッシュ + lookback + 特徴量数**で一意化
- データセットを更新するとキャッシュが無効化される
- 古いキャッシュが残っていると混乱

**対策**:
```bash
# キャッシュクリア
rm -rf cache/panel/*.pkl

# 自動再生成（初回のみ遅い）
# 2回目以降は高速化
```

**キャッシュ命名規則**:
```
ml_dataset_latest_clean_with_adv_lb180_f89_a015bb2ee3.pkl
                                 ^^^  ^^^ ^^^^^^^^^^
                              lookback 特徴量数 データハッシュ
```

---

### 🟡 5. 月次リバランスのタイミング

**症状**:
```
[Backtest] 2025-09-20: Rebalanced (monthly)
[Backtest] 2025-10-01: Rebalanced (monthly)
# ← なぜか11日後にリバランス？
```

**原因**:
- `rebalance_freq=monthly`は**月初営業日**を基準
- 9/20がたまたま月初営業日（9/1~9/19が休場）
- 次は10/1が月初営業日

**対策**:
```python
# 営業日カレンダーを確認
from apex_ranker.utils import get_business_day_calendar

cal = get_business_day_calendar()
print(cal.get_month_start("2025-09"))
# → 2025-09-20 (9/1~9/19が祝日・休場)
```

**注意**:
- 日本市場の祝日カレンダーを使用
- ゴールデンウィーク、年末年始は要注意
- 月初が3連休の場合、4日目がリバランス日

---

## モデル訓練の完全ガイド

### 前提条件

1. **データセット準備**:
   ```bash
   # 89特徴量のデータセットを生成
   # （現在は395特徴量データセットから手動フィルタリング）

   # 将来: 専用ビルダーで89特徴量のみ生成
   python scripts/build_dataset_89feat.py \
     --start-date 2020-01-01 \
     --end-date 2024-12-31 \
     --output output/ml_dataset_20200101_20241231_89feat.parquet
   ```

2. **設定ファイル**:
   ```yaml
   # configs/v0_base_89.yaml
   model:
     d_model: 256
     depth: 4
     patch_len: 16
     stride: 8
     n_heads: 8
     dropout: 0.2
     # patch_multiplierは書かない（autoで自動検出）

   data:
     lookback: 180
     horizons: [1, 5, 10, 20]

   training:
     batch_size: 256
     learning_rate: 0.0001
     max_epochs: 50
   ```

3. **特徴量リスト**:
   ```bash
   # 89特徴量の定義ファイル
   ls configs/feature_names_v0_latest_89.json
   ```

### 訓練手順

#### Step 1: データセット検証

```bash
# 特徴量数確認
python -c "
import polars as pl
import json

# データセット
df = pl.read_parquet('output/ml_dataset_20200101_20241231_89feat.parquet')
dataset_features = [c for c in df.columns if not c.startswith('target_') and c not in ['Date', 'Code']]
print(f'Dataset features: {len(dataset_features)}')

# 設定ファイル
with open('configs/feature_names_v0_latest_89.json') as f:
    config_features = json.load(f)
print(f'Config features: {len(config_features)}')

# 一致確認
assert set(dataset_features) == set(config_features), 'Feature mismatch!'
print('✅ Feature lists match')
"
```

#### Step 2: 訓練実行

```bash
# Purged K-Fold Cross-Validation
python apex-ranker/scripts/train_v0.py \
  --config apex-ranker/configs/v0_base_89.yaml \
  --data output/ml_dataset_20200101_20241231_89feat.parquet \
  --output models/apex_ranker_v0_new.pt \
  --max-epochs 50 \
  --cv-folds 5 \
  --embargo-days 5
```

**訓練時間**: 約10-12時間（A100 80GB GPU）

#### Step 3: チェックポイント検証

```bash
# 次元確認
python -c "
import torch
ckpt = torch.load('models/apex_ranker_v0_new.pt', map_location='cpu')
conv_weight = ckpt['encoder.patch_embed.conv.weight']
effective_dim = conv_weight.shape[0]
print(f'Effective dimension: {effective_dim}')
assert effective_dim == 178, f'Expected 178, got {effective_dim}'
print('✅ Checkpoint dimension correct (89×2 = 178)')
"
```

#### Step 4: スモークテスト

```bash
# 5日間の動作確認
python apex-ranker/scripts/backtest_smoke_test.py \
  --model models/apex_ranker_v0_new.pt \
  --config apex-ranker/configs/v0_base_89.yaml \
  --data output/ml_dataset_20200101_20241231_89feat.parquet \
  --start-date 2024-09-01 \
  --end-date 2024-09-05 \
  --top-k 10 \
  --horizon 20 \
  --output /tmp/smoke_test.json

# ログ確認
grep "Model Init" /tmp/smoke_test.log
# 期待: in_features=89, patch_multiplier=auto, add_csz=False
```

---

## データセット互換性

### データセット形式

```
ml_dataset_YYYYMMDD_YYYYMMDD_89feat.parquet
    ├── Date (pl.Date)
    ├── Code (pl.Utf8)
    ├── feature_0 (pl.Float32)
    ├── feature_1 (pl.Float32)
    │   ...
    ├── feature_88 (pl.Float32)
    ├── target_ret_1d (pl.Float32)
    ├── target_ret_5d (pl.Float32)
    ├── target_ret_10d (pl.Float32)
    └── target_ret_20d (pl.Float32)
```

### 互換性チェック

```python
def validate_dataset_compatibility(dataset_path, model_config_path):
    """データセットとモデルの互換性検証"""
    import polars as pl
    import yaml
    import json

    # 1. データセット特徴量
    df = pl.read_parquet(dataset_path)
    dataset_features = [
        c for c in df.columns
        if not c.startswith('target_') and c not in ['Date', 'Code']
    ]

    # 2. モデル設定
    with open(model_config_path) as f:
        config = yaml.safe_load(f)

    # 3. 特徴量リスト
    feature_names_path = "configs/feature_names_v0_latest_89.json"
    with open(feature_names_path) as f:
        expected_features = json.load(f)

    # 4. チェック
    print(f"Dataset features: {len(dataset_features)}")
    print(f"Expected features: {len(expected_features)}")

    missing = set(expected_features) - set(dataset_features)
    extra = set(dataset_features) - set(expected_features)

    if missing:
        print(f"❌ Missing features: {missing}")
        return False

    if extra:
        print(f"⚠️  Extra features (will be ignored): {extra}")

    print("✅ Dataset compatible")
    return True

# 使用例
validate_dataset_compatibility(
    "output/ml_dataset_latest_clean_with_adv.parquet",
    "apex-ranker/configs/v0_base_89_cleanADV.yaml"
)
```

---

## Feature-ABI (特徴量互換性)

### Feature-ABIとは

**Application Binary Interface for Features**

モデルチェックポイントと特徴量定義の互換性を保証する仕組み。

### 現在の実装（v0.1.0）

**チェックポイント**:
- ✅ 重みデータ（torch.Size([178, 1, 16])）
- ❌ 特徴量メタデータ**なし**（将来改善予定）

**特徴量定義**:
- ✅ `configs/feature_names_v0_latest_89.json` （外部ファイル）
- ❌ チェックポイント内に埋め込まれていない

### はまりポイント

```python
# ❌ 間違い: 特徴量リストを忘れる
model = load_model(checkpoint)
dataset = pl.read_parquet(dataset_path)  # 395特徴量
predictions = model(dataset)  # エラー！

# ✅ 正解: 特徴量リストを明示的にロード
feature_names = json.load(open("configs/feature_names_v0_latest_89.json"))
dataset = dataset.select(["Date", "Code"] + feature_names + targets)  # 89特徴量
predictions = model(dataset)  # OK
```

### 将来の改善（v0.2.0予定）

```python
# チェックポイントにメタデータを埋め込む
checkpoint = {
    'state_dict': model.state_dict(),
    'feature_abi': {
        'version': '1.0',
        'feature_names': feature_names,
        'feature_hash': 'a015bb2ee3',
        'base_features': 89,
        'effective_dim': 178,
    }
}

# 自動検証
if ckpt['feature_abi']['feature_hash'] != dataset_hash:
    raise ValueError("Feature-ABI mismatch!")
```

---

## Cross-Sectional Standardization (CS-Z)

### CS-Zとは

**日次クロスセクション標準化**: 各営業日ごとに全銘柄の特徴量をZ-score標準化

```python
# 日ごとに標準化
for date in dates:
    stocks_today = df.filter(pl.col("Date") == date)
    for feature in features:
        mean = stocks_today[feature].mean()
        std = stocks_today[feature].std()
        df[feature] = (df[feature] - mean) / std
```

### なぜCS-Zが必要か

1. **時系列ドリフト対策**: マクロ環境変化に頑健
2. **銘柄間比較**: 相対的な強さを評価
3. **スケール統一**: 異なる単位の特徴量を揃える

### CS-Zの適用タイミング

```
訓練時:
  raw features (89) → CS-Z → standardized (89) → stack → 178ch → モデル訓練

推論時（現在のv0.1.0）:
  raw features (89) → （CS-Zなし） → 89ch → チェックポイント(178ch)からロード
                                               → 内部でadd_csz=Falseを設定
```

### 重要な注意点

**モデルは既にCS-Z適用済みの178次元で訓練されている**

```python
# チェックポイント保存時（訓練時）
# CS-Z ON: 89 × 2 = 178次元で訓練
checkpoint = {
    'encoder.patch_embed.conv.weight': torch.randn(178, 1, 16)  # ← 178次元
}

# 推論時
# patch_multiplier=autoで自動検出
# → add_csz=False（特徴は既にチェックポイント内で2倍化済み）
model.load_state_dict(checkpoint)  # 178次元をロード
```

### よくある誤解

❌ **誤解1**: 「推論時にCS-Zを手動適用する必要がある」
- ✅ **正解**: チェックポイントが既にCS-Z適用済み。手動適用は不要。

❌ **誤解2**: 「`patch_multiplier: 2`を指定すべき」
- ✅ **正解**: `auto`で自動検出。明示指定すると次元ミスマッチが起こる。

❌ **誤解3**: 「89特徴量と178特徴量のモデルは別物」
- ✅ **正解**: 同じモデル。CS-Zで89→178に拡張しているだけ。

---

## トラブルシューティング

### 問題1: 次元ミスマッチ (178 vs 89)

**症状**:
```
RuntimeError: size mismatch for encoder.patch_embed.conv.weight:
  copying a param with shape torch.Size([178, 1, 16]) from checkpoint,
  the shape in current model is torch.Size([89, 1, 16]).
```

**診断**:
```bash
# 設定ファイル確認
grep "patch_multiplier" apex-ranker/configs/v0_base_89_cleanADV.yaml
```

**修正**:
```yaml
# patch_multiplier行を削除
model:
  d_model: 256
  # patch_multiplier: 1  ← この行を削除
```

---

### 問題2: 次元ミスマッチ (178 vs 708)

**症状**:
```
RuntimeError: size mismatch for encoder.patch_embed.conv.weight:
  copying a param with shape torch.Size([178, 1, 16]) from checkpoint,
  the shape in current model is torch.Size([708, 1, 16]).
```

**診断**:
```bash
# データセット特徴量数確認
python -c "import polars as pl; df = pl.read_parquet('output/ml_dataset_latest_clean_with_adv.parquet'); print(len([c for c in df.columns if not c.startswith('target_') and c not in ['Date', 'Code']]))"
# 出力: 395 ← 多すぎる！
```

**修正**:
```python
# 特徴量フィルタリングを追加
import json

# 89特徴量のみ選択
with open("configs/feature_names_v0_latest_89.json") as f:
    feature_names = json.load(f)

dataset = pl.read_parquet(dataset_path)
dataset = dataset.select(["Date", "Code"] + feature_names + target_cols)
```

---

### 問題3: Panel Cache読み込みエラー

**症状**:
```
FileNotFoundError: cache/panel/ml_dataset_..._lb180_f89_a015bb2ee3.pkl
```

**診断**:
```bash
ls -lh cache/panel/*.pkl
# 古いキャッシュが残っている
```

**修正**:
```bash
# キャッシュクリア
rm -rf cache/panel/*.pkl

# 再実行（自動再生成）
python apex-ranker/scripts/backtest_smoke_test.py ...
```

---

### 問題4: Supply不足 (candidate_kept < 53)

**症状**:
```
[Backtest] 2025-XX-XX: candidate_kept=48 sign=1
⚠️  WARNING: Supply below target (k_min=53)
```

**診断**:
```bash
# 選択ゲート設定確認
grep "k_ratio" apex-ranker/configs/v0_base_89_cleanADV.yaml
# 出力: k_ratio: 0.60
```

**修正**:
```yaml
# k_ratioを緩和
selection:
  k_ratio: 0.70  # 0.60 → 0.70
  k_min: 53
```

---

### 問題5: APIサーバー起動失敗

**症状**:
```
[Model Init] in_features=354
RuntimeError: size mismatch ...
```

**診断**:
APIサーバーは特徴量フィルタリングを実装していない（v0.1.0の既知問題）

**回避策**:
```bash
# 手動実行モードを使用
python apex-ranker/scripts/backtest_smoke_test.py \
  --model bundles/apex_ranker_v0.1.0_prod/models/apex_ranker_v0_enhanced.pt \
  --config bundles/apex_ranker_v0.1.0_prod/configs/v0_base_89_cleanADV.yaml \
  --data output/ml_dataset_latest_clean_with_adv.parquet \
  --start-date 2025-10-01 \
  --end-date 2025-10-01 \
  --top-k 35 \
  --horizon 20 \
  --output predictions_today.json
```

**将来修正** (v0.2.0):
`apex_ranker/api/server.py`に特徴量フィルタリングを追加

---

## チェックリスト

### モデル訓練前

- [ ] データセット期間: 5年以上（推奨: 2020-2024）
- [ ] データセット特徴量: 89特徴量（`feature_names_v0_latest_89.json`）
- [ ] lookback: 180日分のデータ確保
- [ ] 設定ファイル: `patch_multiplier`を明示指定**しない**
- [ ] GPU: A100 80GB推奨（訓練時間: 10-12時間）

### 訓練後

- [ ] チェックポイント次元確認: `effective_dim == 178`
- [ ] スモークテスト: 5日間の推論成功
- [ ] ログ確認: `[Model Init] in_features=89, patch_multiplier=auto, add_csz=False`
- [ ] Feature-ABI: 特徴量リストをバンドルに含める

### デプロイ前

- [ ] validate_bundle.py実行: PASSED
- [ ] MANIFEST.lockチェック: SHA256一致
- [ ] 本番データセット準備: 最新日付まで
- [ ] Panel cache初期化: 初回推論で自動生成

### 運用中

- [ ] 月次リバランス日確認: 営業日カレンダー
- [ ] Supply安定性: `candidate_kept == 53`
- [ ] Transaction cost: <30 bps/day
- [ ] Performance: Sharpe > 1.4

---

## 参考資料

### 関連ドキュメント

- `DEPLOYMENT_STATUS.md` - デプロイ手順とCS-Z問題の詳細
- `MANIFEST.lock` - 本番バンドルのメタデータ
- `P0_PRODUCTION_DEPLOYMENT_CHECKLIST.md` - 運用チェックリスト

### 重要な設定ファイル

```
apex-ranker/
├── configs/
│   ├── v0_base_89_cleanADV.yaml          # 本番設定
│   ├── feature_names_v0_latest_89.json   # 89特徴量リスト
│   └── feature_aliases_compat.yaml       # 特徴量エイリアス
├── scripts/
│   ├── train_v0.py                       # 訓練スクリプト
│   └── backtest_smoke_test.py            # 推論スクリプト
└── apex_ranker/
    ├── models/ranker.py                  # APEXRankerV0モデル
    └── data/loader.py                    # データローダー
```

### コマンドクイックリファレンス

```bash
# 訓練
python apex-ranker/scripts/train_v0.py \
  --config configs/v0_base_89.yaml \
  --data output/ml_dataset_89feat.parquet \
  --output models/new_model.pt

# 検証
python scripts/validate_bundle.py \
  --bundle bundles/apex_ranker_v0.1.0_prod

# 推論
python apex-ranker/scripts/backtest_smoke_test.py \
  --model models/apex_ranker_v0_enhanced.pt \
  --config configs/v0_base_89_cleanADV.yaml \
  --start-date 2025-11-01 \
  --end-date 2025-11-01 \
  --top-k 35
```

---

**最終更新**: 2025-11-04
**バージョン**: v0.1.0-prod
**問い合わせ**: 技術ドキュメントの改善提案は Issue へ
