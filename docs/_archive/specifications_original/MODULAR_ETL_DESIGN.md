# Modular ETL Design for JQuants Data Pipeline

## Overview
モジュール化されたETLアーキテクチャにより、各APIエンドポイントを独立して更新可能。

## Architecture

```
┌─────────────────────────────────────────────────┐
│                JQuants API                      │
├──────────┬──────────┬────────────┬─────────────┤
│  Prices  │  TOPIX   │ TradesSpec │ ListedInfo  │
└────┬─────┴────┬─────┴─────┬──────┴──────┬──────┘
     │          │           │             │
     ▼          ▼           ▼             ▼
┌─────────────────────────────────────────────────┐
│           Component Fetchers                    │
│  - Async/Concurrent                             │
│  - Pagination Support                           │
│  - Error Handling                               │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│           Data Processors                       │
│  - Type Conversion                              │
│  - Feature Engineering                          │
│  - Validation                                   │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│           Dataset Manager                       │
│  - Merge/Join                                   │
│  - Deduplication                                │
│  - Versioning                                   │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│              Storage Layer                      │
│  - Parquet (Primary)                            │
│  - CSV (Compatibility)                          │
│  - Metadata (JSON)                              │
└─────────────────────────────────────────────────┘
```

## Components

### 1. Price Data Component (`prices`)
- **Endpoint**: `/prices/daily_quotes`
- **Features**: OHLCV data
- **Update Frequency**: Daily

### 2. TOPIX Component (`topix`)
- **Endpoint**: `/indices/topix`
- **Features**:
  - Alpha (超過リターン)
  - Relative Strength
  - Market Regime
- **Update Frequency**: Daily

### 3. Trades Specification Component (`trades_spec`)
- **Endpoint**: `/markets/trades_spec`
- **Features**:
  - Trading Unit
  - Market Section
  - Unit Change Flag
- **Update Frequency**: When changes occur

### 4. Listed Info Component (`listed_info`)
- **Endpoint**: `/listed/info`
- **Features**:
  - Market Type
  - Sector Code
  - TSE Prime Flag
- **Update Frequency**: Monthly

## Usage Examples

### Update Single Component
```bash
# Update only TOPIX data
python scripts/modular_updater.py \
  --dataset output/ml_dataset_latest.parquet \
  --update topix \
  --days 30 \
  --tag topix_update

# Update only trades specification
python scripts/modular_updater.py \
  --dataset output/ml_dataset_latest.parquet \
  --update trades_spec \
  --from-date 2024-01-01 \
  --to-date 2024-12-31
```

### Update Multiple Components
```bash
# Update TOPIX and trades spec together
python scripts/modular_updater.py \
  --dataset output/ml_dataset_latest.parquet \
  --update topix trades_spec \
  --days 90 \
  --tag multi_update
```

### Incremental Updates
```bash
# Daily update job
python scripts/modular_updater.py \
  --dataset output/ml_dataset_latest.parquet \
  --update prices topix \
  --days 1 \
  --tag daily

# Weekly update job
python scripts/modular_updater.py \
  --dataset output/ml_dataset_latest.parquet \
  --update trades_spec listed_info \
  --days 7 \
  --tag weekly
```

## Dagster Integration

```python
from dagster import asset, AssetIn, Output
from pathlib import Path
import asyncio
from modular_updater import ModularDataUpdater

@asset
def fetch_topix_data(context):
    """Fetch TOPIX index data"""
    updater = ModularDataUpdater()
    params = {
        "from": context.op_config["from_date"],
        "to": context.op_config["to_date"]
    }
    topix_df = asyncio.run(
        updater.fetch_component("topix", params)
    )
    return Output(topix_df, metadata={"rows": len(topix_df)})

@asset(ins={"base_data": AssetIn(), "topix_data": AssetIn()})
def update_with_topix(context, base_data, topix_data):
    """Update dataset with TOPIX features"""
    updater = ModularDataUpdater()
    updated_df = updater.update_dataset(
        base_data,
        {"topix": topix_data}
    )
    return Output(updated_df)

@asset
def ml_dataset_with_topix(update_with_topix):
    """Final ML dataset with TOPIX features"""
    return update_with_topix
```

## MinIO/S3 Storage

```python
# Save to MinIO
import boto3
from io import BytesIO

def save_to_minio(df: pl.DataFrame, bucket: str, key: str):
    """Save DataFrame to MinIO/S3"""
    s3_client = boto3.client(
        's3',
        endpoint_url='http://minio:9000',
        aws_access_key_id='minioadmin',
        aws_secret_access_key='minioadmin'
    )

    # Convert to parquet bytes
    buffer = BytesIO()
    df.write_parquet(buffer)
    buffer.seek(0)

    # Upload to MinIO
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=buffer.getvalue()
    )
```

## Benefits

1. **Flexibility**: 各コンポーネントを独立して更新
2. **Efficiency**: 必要なデータのみ取得
3. **Scalability**: 新しいAPIエンドポイントを簡単に追加
4. **Maintainability**: コンポーネントごとのテストとデバッグ
5. **Cost-effective**: API呼び出しの最小化

## Future Extensions

1. **Financial Statements** (`/fins/statements`)
2. **Dividend Data** (`/fins/dividend`)
3. **Index Components** (`/indices/components`)
4. **Options Data** (`/option/index_option`)
5. **Corporate Actions** (`/markets/corporate_actions`)

## Monitoring

```yaml
# prometheus metrics
jquants_api_calls_total{component="topix", status="success"} 145
jquants_api_calls_total{component="prices", status="error"} 2
jquants_data_rows_fetched{component="topix"} 252
jquants_processing_duration_seconds{component="topix"} 1.23
```
