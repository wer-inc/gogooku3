# J-Quants データ契約ベースバリデーション

このディレクトリには、J-Quants由来の全データブロックの「データ契約」を定義するYAMLファイルが含まれています。

## ファイル構成

- `jqx_all.yaml`: 全データソースの契約定義

## 使い方

### 基本的な使い方

```bash
# 契約ベースのバリデータを実行
python tools/validator_jqx_all.py \
  --dataset gogooku5/data/output/ml_dataset_latest_full.parquet \
  --contract tools/contracts/jqx_all.yaml \
  --debug-meta-dir gogooku5/data/output/cache/snapshots \
  --report /tmp/jqx_validation_report.json \
  --summary /tmp/jqx_validation_summary.md
```

### 既存バリデータとの統合

既存の`validator_1to9.py`に`--contract`オプションを追加しました：

```bash
python tools/validator_1to9.py \
  --dataset gogooku5/data/output/ml_dataset_latest_full.parquet \
  --start 2024-01-01 \
  --end 2025-01-31 \
  --snapshots-dir gogooku5/data/output/cache/snapshots \
  --contract tools/contracts/jqx_all.yaml
```

## 契約ファイルの構造

### データソース定義

各データソース（`daily_quotes`, `margin_daily`, `earnings`など）に対して以下を定義：

- **required**: 必須列（存在しない場合はFAIL）
- **derived**: 派生列（存在しない場合はWARN、optionalブロックの場合は部分的な欠落のみWARN）
- **rules**:
  - **asof**: As-of規則（例: `T+1_09:00`, `same_day_15:00`）
  - **null_max**: 各列の欠損率上限（超過時はFAIL/WARN）
  - **invariants**: 整合性チェック（例: `(1+ret_prev_1d) ≈ (1+ret_overnight)*(1+ret_intraday)`）

### 特殊ブロック

- **denylist**: 学習禁則列のパターン（存在する場合はFAIL）
- **canonical_ohlc**: カノニカルOHLCチェック（raw/Adj Close削除確認）

## レポート出力

### JSONレポート

```json
{
  "status": "PASS|WARN|FAIL",
  "blocks": [
    {
      "block": "daily_quotes",
      "status": "PASS",
      "issues": []
    }
  ]
}
```

### Markdownサマリー

人間が読める形式のサマリー（`--summary`で指定したパスに出力）

## 実装漏れチェック

契約ファイルを更新することで、新しいデータソースの追加や既存ソースの変更を自動的に検証できます。

### 新しいデータソースを追加する場合

1. `tools/contracts/jqx_all.yaml`に新しいソース定義を追加
2. バリデータを実行して実装状況を確認

### 既存ソースの変更を検証する場合

1. 契約ファイルの`required`/`derived`を更新
2. バリデータを実行して差分を確認
