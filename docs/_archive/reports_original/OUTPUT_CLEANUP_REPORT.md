# Output ディレクトリ整理レポート

## 📋 実施日時
2025年1月27日 17:13 JST

## 🎯 整理目的
- テストファイルの蓄積による容量圧迫を解消
- 最新ファイルへの簡単なアクセスを提供
- ディレクトリ構造を整理

## 📊 整理前の状態

### 容量とファイル数
- **総容量**: 89MB
- **ファイル数**: 51個
- **問題点**:
  - タイムスタンプ付きテストファイルが大量に蓄積
  - 同じデータの重複保存（CSV、Parquet、メタデータ）
  - 不要な中間ファイル（flow_features.csv等）

## 🔧 実施内容

### 1. アーカイブディレクトリの作成
```bash
mkdir -p output/archive
```

### 2. 古いテストファイルの移動
- `ml_dataset_20250827_*` 形式のファイルを全てarchiveへ移動
- 移動ファイル数: 約40個
- 移動容量: 約75MB

### 3. 不要ファイルの削除
削除したファイル:
- `dataset_info.json`
- `flow_features.csv`
- `price_data.csv`
- `technical_features.csv`
- `trades_spec.csv`
- `ml_dataset.csv`
- 重複するシンボリックリンク

### 4. 最新ファイルへのシンボリックリンク作成
```bash
latest.parquet -> ml_dataset_20250827_170500.parquet
latest_metadata.json -> ml_dataset_20250827_170500_metadata.json
```

## 📁 整理後の構造

```
output/
├── archive/                     # 過去のテストファイル（75MB）
│   ├── ml_dataset_20250827_002051.*
│   ├── ml_dataset_20250827_002242.*
│   └── ... (約40ファイル)
│
├── latest.parquet               # 最新データへのリンク
├── latest_metadata.json         # 最新メタデータへのリンク
└── ml_dataset_test_topix_*      # 最近のテスト結果（保持）
```

## 📈 整理結果

### Before
- **容量**: 89MB
- **ファイル数**: 51個
- **アクセス性**: 困難（どれが最新か不明）

### After
- **容量**: 76MB（archive含む）
- **メインディレクトリ**: 168KB
- **ファイル数**: 11個（メインディレクトリ）
- **アクセス性**: 良好（latest.*で最新データアクセス可能）

## 💡 改善効果

1. **容量節約**: 約13MB削減
2. **視認性向上**: ファイル数を51個→11個に削減
3. **アクセス性改善**: `latest.parquet`で常に最新データにアクセス可能
4. **履歴保持**: archiveディレクトリに過去データを保存

## 📝 使用方法

### 最新データへのアクセス
```python
import polars as pl

# 常に最新データを読み込み
df = pl.read_parquet("output/latest.parquet")
```

### 過去データへのアクセス
```python
# 特定の過去データが必要な場合
df = pl.read_parquet("output/archive/ml_dataset_20250827_002051.parquet")
```

## 🔄 今後の運用指針

### 自動クリーンアップスクリプト
```bash
#!/bin/bash
# 7日以上前のファイルをアーカイブ
find output -name "ml_dataset_*.parquet" -mtime +7 -exec mv {} output/archive/ \;

# 30日以上前のアーカイブを削除
find output/archive -name "*.parquet" -mtime +30 -delete
```

### 命名規則
- **テスト実行**: `ml_dataset_YYYYMMDD_HHMMSS.*`
- **本番実行**: `ml_dataset_production_YYYYMMDD.*`
- **シンボリックリンク**: `latest.*`

## ✅ 確認事項

- [x] 最新データファイルの保持
- [x] シンボリックリンクの作成
- [x] 不要ファイルの削除
- [x] アーカイブディレクトリの整理
- [x] 容量削減の確認

---
*実施日: 2025年1月27日*
*実施者: Claude*
