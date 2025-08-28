# gogooku3 実装計画書 - 既存システム統合版

## 概要
既存の `batch` (データ取得・特徴量計算) と `ATFT-GAT-FAN` (ML学習) を統合し、OSS中心の堅牢な金融MLパイプラインを構築します。

## 現状分析

### 既存資産
1. **apps/batch**
   - JQuants API統合済み（150並列）
   - 713技術指標計算済み（pandas-ta）
   - BigQueryローダー実装済み
   - Docker化済み

2. **apps/ATFT-GAT-FAN**
   - PyTorch GAT/Transformer実装済み
   - Hydra設定管理済み
   - MLflow実験追跡済み
   - Variance Watchdog実装済み

### 課題
- 手動実行が必要（`pnpm batch:docker:run` → `make ml-link` → `make train`）
- エラー時の全体再実行が必要
- コスト可視化なし
- BigQuery依存によるコスト増

## 実装フェーズ

### Phase 0: 準備（1週間）
**目的**: 開発環境構築とOSSスタック導入

#### タスク
```yaml
week_1:
  infrastructure:
    - [ ] MinIO導入（Docker Compose）
    - [ ] ClickHouse導入（Docker Compose）
    - [ ] Redis導入（Docker Compose）
    - [ ] Dagster OSS導入

  monitoring:
    - [ ] Prometheus + Grafana導入
    - [ ] Loki導入
    - [ ] Alertmanager設定

  security:
    - [ ] SOPS + age セットアップ
    - [ ] JQuants認証情報移行
```

#### 成果物
- `docker-compose.oss.yml` - 全OSSスタック定義
- `secrets/` - SOPS暗号化設定
- `monitoring/` - Grafanaダッシュボード定義

### Phase 1: データパイプライン移行（2週間）
**目的**: batchをDagsterアセット化し、BigQuery→ClickHouse移行

#### タスク
```yaml
week_2:
  contracts:
    - [ ] TSEカレンダー実装
    - [ ] Data Contract定義（price/volume/corporate_actions）
    - [ ] pandera検証実装

  dagster_assets:
    - [ ] raw_price_data アセット（JQuants取得）
    - [ ] corporate_actions アセット
    - [ ] technical_features アセット（713→127指標削減）

week_3:
  storage:
    - [ ] Parquet→MinIO書き込み実装
    - [ ] ClickHouseテーブル作成
    - [ ] バルクロード実装

  migration:
    - [ ] 既存BigQueryデータ移行スクリプト
    - [ ] 特徴量計算の並列化維持（200ワーカー）
```

#### 移行スクリプト例
```python
# migration/batch_to_dagster.py
from apps.batch.core import AsyncJQuantsDataFetcher, ParallelProcessor
from dagster import asset, Output
import pandas as pd

@asset(partitions_def=date_ticker_partitions)
def jquants_price_data(context) -> Output[pd.DataFrame]:
    """既存batch取得ロジックをアセット化"""
    fetcher = AsyncJQuantsDataFetcher()
    date = context.partition_key.keys_by_dimension["date"]
    ticker = context.partition_key.keys_by_dimension["ticker"]

    # 既存の取得ロジックを流用
    df = fetcher.fetch_daily_quotes(ticker, date)

    # Data Contract検証追加
    PriceDataValidator.schema.validate(df)
    PriceDataValidator.ohlc(df)

    return Output(df, metadata={"rows": len(df)})

@asset
def technical_features_optimized(price_data: pd.DataFrame) -> pd.DataFrame:
    """713→127指標に削減"""
    processor = ParallelProcessor(max_workers=200)

    # Feature Manifestに基づく選択
    active_features = load_active_features()  # 127指標のみ

    # 既存の並列計算ロジックを流用
    features = processor.compute_features(price_data, active_features)

    # MinIOに保存
    write_parquet_minio(features, "s3://gogooku/features/technical.parquet")

    return features
```

### Phase 2: ML学習パイプライン統合（2週間）
**目的**: ATFT-GAT-FANをDagster化し、Feast Feature Store導入

#### タスク
```yaml
week_4:
  feast:
    - [ ] Feature Store設定（offline: MinIO, online: Redis）
    - [ ] price_features FeatureView定義
    - [ ] flow_features FeatureView定義（投資部門別）

  ml_integration:
    - [ ] training アセット実装（既存train.sh流用）
    - [ ] Variance Watchdog統合
    - [ ] PurgedGroupKFold統合

week_5:
  optimization:
    - [ ] Optuna統合（ハイパーパラメータ探索）
    - [ ] MLflow→MinIOアーティファクト保存
    - [ ] Champion/Challenger実装
```

#### 統合スクリプト例
```python
# ml/training_asset.py
from apps.ATFT_GAT_FAN.src.models import ATFT_GAT_FAN
from apps.ATFT_GAT_FAN.src.data.loaders import ProductionLoaderV2
from dagster import asset
import mlflow

@asset(metadata={"compute": "gpu"})
def ml_training(technical_features: pd.DataFrame):
    """既存ATFT-GAT-FAN学習をアセット化"""

    # 既存のデータローダー流用
    loader = ProductionLoaderV2(
        data_dir="/tmp/features",
        sequence_length=20,
        batch_size=512
    )

    # 既存のモデル流用
    model = ATFT_GAT_FAN(
        input_size=127,  # 削減後の特徴量数
        hidden_size=512,
        num_heads=16
    )

    # Variance Watchdog適用
    watchdog = VarianceWatchdog(min_std_ratio=0.1)

    # 学習実行（既存のtrain関数流用）
    with mlflow.start_run():
        train_with_watchdog(model, loader, watchdog)

        # MinIOにモデル保存
        mlflow.pytorch.log_model(model, "model")

    return model
```

### Phase 3: 監視・自動化（2週間）
**目的**: SLO監視とエンドツーエンド自動化

#### タスク
```yaml
week_6:
  monitoring:
    - [ ] Prometheusメトリクス実装
    - [ ] Grafanaダッシュボード作成
    - [ ] SLOアラート設定（Alertmanager）

  automation:
    - [ ] TSEカレンダーセンサー実装
    - [ ] 15:00自動起動設定
    - [ ] エラーリトライ実装（ticker/date粒度）

week_7:
  testing:
    - [ ] ゴールデンデータセット作成
    - [ ] 回帰テスト実装（IC/RMSE/Sharpe）
    - [ ] E2Eテスト実装
```

#### 監視設定例
```yaml
# monitoring/prometheus/rules.yml
groups:
  - name: slo_alerts
    rules:
      - alert: DataFreshnessViolation
        expr: histogram_quantile(0.95, data_freshness_minutes_bucket) > 5
        for: 5m
        annotations:
          summary: "データ新鮮度SLO違反: p95 > 5分"

      - alert: LowGPUUtilization
        expr: avg_over_time(gpu_utilization[1h]) < 0.85
        for: 10m
        annotations:
          summary: "GPU利用率低下: < 85%"
```

### Phase 4: 本番移行（1週間）
**目的**: 段階的な本番切り替え

#### タスク
```yaml
week_8:
  migration:
    - [ ] 並行稼働期間設定（既存BigQuery + 新ClickHouse）
    - [ ] データ整合性検証
    - [ ] パフォーマンス比較

  cutover:
    - [ ] BigQuery→ClickHouse完全移行
    - [ ] コスト削減効果測定
    - [ ] ドキュメント更新
```

## コスト削減試算

### 現状（月額）
- BigQuery: ¥50,000/日 × 30日 = ¥1,500,000
- 手動運用工数: 1時間/日 × ¥5,000 × 30日 = ¥150,000
- **合計: ¥1,650,000/月**

### 移行後（月額）
- ClickHouse（自己ホスト）: サーバー代 ¥50,000
- MinIO（自己ホスト）: 含む
- 自動化による工数削減: ¥10,000（監視のみ）
- **合計: ¥60,000/月**

**削減効果: 96.4%減（¥1,590,000/月の削減）**

## リスクと対策

### リスク
1. **データ移行時の不整合**
   - 対策: 並行稼働期間で検証

2. **GPU利用効率低下**
   - 対策: バッチサイズ自動調整

3. **ClickHouse運用経験不足**
   - 対策: DuckDBでの開発環境構築

## 成功指標

### 定量指標
- [ ] EOD後5分以内のデータ利用可能（p95）
- [ ] 日次パイプライン成功率 ≥ 99.5%
- [ ] 学習時間 ≤ 120分
- [ ] GPU利用率 ≥ 85%
- [ ] コスト削減 ≥ 90%

### 定性指標
- [ ] 完全自動化（手動介入不要）
- [ ] ticker/date粒度での再実行可能
- [ ] 監視ダッシュボードでの可視化

## 実装優先順位

1. **必須（MVP）**
   - Dagster導入
   - MinIO/ClickHouse導入
   - 既存ロジックのアセット化

2. **重要**
   - Feast Feature Store
   - Prometheus監視
   - 自動リトライ

3. **あると良い**
   - Optuna最適化
   - Tritonサービング
   - A/Bテスト

## 次のステップ

### 即座に開始可能なタスク
```bash
# 1. OSSスタック起動
cd /home/ubuntu/gogooku2/apps/gogooku3
docker-compose -f docker-compose.oss.yml up -d

# 2. 既存コードのDagster化開始
mkdir -p dagster/{assets,jobs,sensors}
cp -r ../batch/core dagster/legacy/
cp -r ../ATFT-GAT-FAN/src dagster/ml/

# 3. 最初のアセット実装
python dagster/assets/price_data.py
```

この計画により、既存資産を最大限活用しながら、段階的に堅牢なMLパイプラインを構築できます。
