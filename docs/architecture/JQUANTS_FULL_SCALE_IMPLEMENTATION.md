# J-Quants API フルスケール実装ガイド

## 概要

営業日ごとの動的な銘柄リスト取得とMarket Codeフィルタリングを実装した、完全な本番対応版J-Quants APIインテグレーションです。

## 主な特徴

### 1. Trading Calendar API統合
- 営業日・休日・半休日の正確な取得
- カレンダーベースのデータ取得で無駄なAPI呼び出しを削減
- キャッシュ機能により高速化

### 2. Market Codeフィルタリング
- 8つのターゲット市場のみを対象（プライム、スタンダード、グロース等）
- TOKYO PRO MARKET (0105) と その他 (0109) を除外
- 約4,000銘柄を効率的に取得

### 3. 営業日ごとの動的銘柄管理
- IPO（新規上場）を自動検出して追加
- 上場廃止銘柄を自動除外
- 各営業日の正確な銘柄リストを維持

### 4. 有料プラン対応の最適化
- 同時接続数: 75（有料プラン向け）
- バッチ処理による効率化
- メモリ管理と進捗表示

## 実行方法

### 基本的な実行

```bash
# J-Quants APIを使用して4年分のデータを取得
python scripts/pipelines/run_pipeline.py --jquants

# 期間を指定して取得
python scripts/pipelines/run_pipeline.py --jquants \
  --start-date 2021-01-01 \
  --end-date 2025-01-31
```

### テスト実行

```bash
# 統合テスト
python tests/test_jquants_integration.py

# 短期間でのテスト（1週間分）
python scripts/pipelines/run_pipeline.py --jquants \
  --start-date 2025-01-01 \
  --end-date 2025-01-07
```

## 処理フロー

```
1. Trading Calendar API
   └─> 営業日リスト取得（4年間で約980日）

2. 営業日ごとの処理ループ
   ├─> Listed Info API（その日の銘柄リスト）
   ├─> Market Codeフィルタリング（8市場のみ）
   └─> Daily Quotes API（価格データ取得）

3. データ統合
   ├─> 全期間のデータ結合
   ├─> 営業日フィルタリング
   └─> TOPIX指数データマージ
```

## パフォーマンス目安

### データ規模
- 営業日数: 約980日（4年間）
- 銘柄数: 約4,000銘柄/日
- 総レコード数: 約400万レコード

### 処理時間（有料プラン）
- 営業日取得: 1-2秒
- 銘柄リスト取得: 1-2分（全期間）
- 価格データ取得: 30-60分（全データ）
- 合計: 約1時間

### メモリ使用量
- ピーク時: 8-16GB
- 推奨: 32GB以上のシステム

## 設定ファイル

### .env
```bash
# J-Quants認証情報（必須）
JQUANTS_AUTH_EMAIL=your_email@example.com
JQUANTS_AUTH_PASSWORD=your_password

# API設定（有料プラン向け）
MAX_CONCURRENT_FETCH=75
```

### configs/jquants_api_config.yaml
```yaml
# レート制限
rate_limits:
  max_concurrent: 75  # 有料プラン

# バッチ処理
batch_processing:
  chunk_size: 500      # 銘柄数/バッチ
  daily_batch_size: 10 # 並列処理日数
  memory_limit_gb: 16.0
```

## エラー処理

### リトライ機構
- 最大3回の自動リトライ
- 指数バックオフ（2, 4, 8秒）
- 429エラー（レート制限）に対応

### キャッシュ
- Trading Calendar: 1週間保持
- Listed Info: 24時間保持
- 価格データ: キャッシュなし（最新データ取得）

## トラブルシューティング

### メモリ不足エラー
```bash
# メモリ使用量を制限
export MEMORY_LIMIT_GB=8
python scripts/pipelines/run_pipeline.py --jquants
```

### API認証エラー
```bash
# 認証情報を確認
cat .env | grep JQUANTS

# 手動で認証テスト
python tests/test_jquants_integration.py
```

### データ取得エラー
```bash
# キャッシュをクリア
rm -rf cache/

# デバッグモードで実行
LOG_LEVEL=DEBUG python scripts/pipelines/run_pipeline.py --jquants
```

## 統計情報の確認

実行後、以下の情報が表示されます：

```
✅ 営業日数: 980日
✅ 期間中のユニーク銘柄数: 4,523
✅ 日次平均銘柄数: 3,987
✅ 新規上場: 234銘柄
✅ 上場廃止: 189銘柄
✅ 総レコード数: 3,907,260
```

## 出力ファイル

### データファイル
- `output/ml_dataset_{timestamp}.parquet` - Parquet形式（推奨）
- `output/ml_dataset_{timestamp}.csv` - CSV形式（互換性用）

### メタデータ
- `output/ml_dataset_{timestamp}_metadata.json` - データセット情報

## 次のステップ

1. **機械学習パイプライン実行**
   ```bash
   python scripts/integrated_ml_training_pipeline.py
   ```

2. **データ品質検証**
   ```bash
   python scripts/validate_improvements.py --detailed
   ```

3. **ハイパーパラメータ調整**
   ```bash
   python scripts/hyperparameter_tuning.py --trials 50
   ```

## 更新履歴

- 2025-08-31: フルスケール実装完了
  - テスト制限（30日、100銘柄）を削除
  - バッチ処理の最適化
  - 進捗表示の改善
  - メモリ管理の強化