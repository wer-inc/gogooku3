# gogooku3/batch 新規設計計画書

## 概要
既存の `/apps/batch` の機能を参考にしつつ、gogooku3の設計思想（OSS、Dagster Asset、Data Contract）に基づいて完全に再設計します。

## 設計方針

### 基本原則
1. **Dagster Asset中心** - すべての処理をアセットとして定義
2. **Data Contract First** - 契約駆動開発で品質保証
3. **OSS Only** - MinIO/ClickHouse/DuckDBを活用
4. **失敗の局所化** - ticker/date粒度での再実行
5. **観測可能性** - メトリクス/ログ/トレースの完備

### 既存batchからの改善点
- ✅ モノリシック処理 → **マイクロアセット**
- ✅ BigQuery依存 → **ClickHouse/MinIO**
- ✅ 全体再実行 → **パーティション単位再実行**
- ✅ 手動実行 → **センサー駆動自動実行**
- ✅ 713指標全計算 → **127指標に最適化**

## ディレクトリ構造

```
/home/ubuntu/gogooku2/apps/gogooku3/
├── batch/                          # 新設計バッチ処理
│   ├── contracts/                  # データ契約定義
│   │   ├── __init__.py
│   │   ├── price_data.yaml        # 価格データ契約
│   │   ├── corporate_actions.yaml # CA契約
│   │   └── features.yaml          # 特徴量契約
│   │
│   ├── assets/                     # Dagsterアセット
│   │   ├── __init__.py
│   │   ├── raw/                   # 生データ取得
│   │   │   ├── jquants_price.py
│   │   │   ├── jquants_breakdown.py
│   │   │   └── corporate_actions.py
│   │   ├── cleaned/               # クレンジング済み
│   │   │   ├── adjusted_price.py
│   │   │   └── validated_data.py
│   │   └── features/              # 特徴量
│   │       ├── technical.py
│   │       ├── investor_flow.py
│   │       └── composite.py
│   │
│   ├── validators/                # バリデーター
│   │   ├── __init__.py
│   │   ├── price_validator.py    # pandera schemas
│   │   ├── ohlc_validator.py
│   │   └── business_day.py
│   │
│   ├── calendar/                  # TSEカレンダー
│   │   ├── __init__.py
│   │   ├── tse_calendar.py
│   │   └── holidays.yaml
│   │
│   ├── storage/                   # ストレージI/O
│   │   ├── __init__.py
│   │   ├── minio_client.py       # MinIO操作
│   │   ├── clickhouse_client.py  # ClickHouse操作
│   │   └── parquet_io.py         # Parquet最適化
│   │
│   ├── metrics/                   # メトリクス収集
│   │   ├── __init__.py
│   │   ├── prometheus.py         # Prometheusエクスポーター
│   │   └── collectors.py         # カスタムメトリクス
│   │
│   ├── tests/                     # テスト
│   │   ├── __init__.py
│   │   ├── golden/               # ゴールデンデータ
│   │   ├── property/            # プロパティテスト
│   │   └── integration/         # 統合テスト
│   │
│   ├── config/                   # 設定
│   │   ├── __init__.py
│   │   ├── settings.py          # 環境変数管理
│   │   └── features_manifest.yaml # 特徴量定義
│   │
│   └── dagster/                  # Dagster設定
│       ├── __init__.py
│       ├── repository.py         # リポジトリ定義
│       ├── jobs.py              # ジョブ定義
│       ├── sensors.py           # センサー定義
│       └── schedules.py         # スケジュール定義
│
├── docker/                       # Docker設定
│   ├── batch.Dockerfile
│   └── docker-compose.batch.yml
│
└── docs/
    └── BATCH_REDESIGN_PLAN.md   # この文書
```

## 実装フェーズ

### Phase 1: 基盤構築（3日）

#### Day 1: プロジェクト初期化
```python
# batch/__init__.py
"""
gogooku3 batch processing module
OSS-based financial data pipeline with Dagster orchestration
"""
__version__ = "3.0.0"
```

```python
# batch/config/settings.py
from pydantic import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # JQuants
    JQUANTS_EMAIL: str
    JQUANTS_PASSWORD: str
    JQUANTS_MAX_CONCURRENT: int = 150

    # Storage
    MINIO_ENDPOINT: str = "http://localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "gogooku3"

    # ClickHouse
    CLICKHOUSE_HOST: str = "localhost"
    CLICKHOUSE_PORT: int = 9000
    CLICKHOUSE_DATABASE: str = "gogooku3"

    # Processing
    MAX_WORKERS: int = 200
    BATCH_SIZE: int = 1000

    # Monitoring
    PROMETHEUS_PORT: int = 9090

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
```

#### Day 2: Data Contracts実装
```yaml
# batch/contracts/price_data.yaml
name: price_data
version: 1.0.0
description: "TSE日次価格データ契約"

schema:
  fields:
    - name: ticker
      type: string
      constraints:
        pattern: "^[0-9]{4}$"
        nullable: false
        description: "4桁銘柄コード"

    - name: date
      type: datetime
      constraints:
        timezone: "Asia/Tokyo"
        nullable: false
        business_day: true
        description: "取引日（JST）"

    - name: open
      type: float32
      constraints:
        nullable: false
        gt: 0
        description: "始値"

    - name: high
      type: float32
      constraints:
        nullable: false
        gte: ["open", "low", "close"]
        description: "高値"

    - name: low
      type: float32
      constraints:
        nullable: false
        lte: ["open", "high", "close"]
        gt: 0
        description: "安値"

    - name: close
      type: float32
      constraints:
        nullable: false
        gt: 0
        description: "終値"

    - name: volume
      type: uint64
      constraints:
        nullable: false
        ge: 0
        description: "出来高"

    - name: adjustment_factor
      type: float32
      constraints:
        nullable: true
        gt: 0
        description: "調整係数"

quality_checks:
  - name: ohlc_consistency
    type: custom
    function: validate_ohlc

  - name: business_day_check
    type: custom
    function: validate_business_day

  - name: volume_sanity
    type: range
    min: 0
    max: 1e12

metadata:
  granularity: daily
  primary_key: [ticker, date]
  partition_key: date
  cluster_key: ticker
  ttl_days: 730  # 2年
```

```python
# batch/validators/price_validator.py
import pandera as pa
from pandera import Check, DataFrameSchema
import pandas as pd
from typing import Dict, Any
import yaml

class ContractValidator:
    """Data Contract準拠のバリデーター"""

    def __init__(self, contract_path: str):
        with open(contract_path) as f:
            self.contract = yaml.safe_load(f)
        self.schema = self._build_schema()

    def _build_schema(self) -> DataFrameSchema:
        """契約からpanderaスキーマを生成"""
        columns = {}

        for field in self.contract['schema']['fields']:
            checks = []
            constraints = field.get('constraints', {})

            # 基本制約
            if 'gt' in constraints:
                checks.append(Check.gt(constraints['gt']))
            if 'gte' in constraints:
                checks.append(Check.ge(constraints['gte']))
            if 'lt' in constraints:
                checks.append(Check.lt(constraints['lt']))
            if 'lte' in constraints:
                checks.append(Check.le(constraints['lte']))
            if 'pattern' in constraints:
                checks.append(Check.str_matches(constraints['pattern']))

            # 型マッピング
            dtype_map = {
                'string': str,
                'float32': 'float32',
                'uint64': 'uint64',
                'datetime': 'datetime64[ns, Asia/Tokyo]'
            }

            columns[field['name']] = pa.Column(
                dtype_map.get(field['type']),
                checks=checks,
                nullable=constraints.get('nullable', True)
            )

        return DataFrameSchema(columns, coerce=True)

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """データフレームを契約に対して検証"""
        # スキーマ検証
        df = self.schema.validate(df)

        # カスタム品質チェック
        for check in self.contract.get('quality_checks', []):
            if check['type'] == 'custom':
                getattr(self, check['function'])(df)

        return df

    def validate_ohlc(self, df: pd.DataFrame):
        """OHLC整合性チェック"""
        assert (df['high'] >= df[['open', 'low', 'close']].max(axis=1)).all()
        assert (df['low'] <= df[['open', 'high', 'close']].min(axis=1)).all()

    def validate_business_day(self, df: pd.DataFrame):
        """営業日チェック"""
        from batch.calendar import TSECalendar
        cal = TSECalendar()
        invalid = df[~df['date'].dt.date.apply(cal.is_business_day)]
        assert invalid.empty, f"非営業日データ: {invalid['date'].unique()}"
```

#### Day 3: ストレージ層実装
```python
# batch/storage/minio_client.py
import s3fs
import pyarrow.parquet as pq
import pandas as pd
from batch.config import settings
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class MinIOClient:
    """MinIO S3互換ストレージクライアント"""

    def __init__(self):
        self.fs = s3fs.S3FileSystem(
            key=settings.MINIO_ACCESS_KEY,
            secret=settings.MINIO_SECRET_KEY,
            client_kwargs={'endpoint_url': settings.MINIO_ENDPOINT}
        )
        self.bucket = settings.MINIO_BUCKET

    def write_parquet(
        self,
        df: pd.DataFrame,
        path: str,
        partition_cols: Optional[list] = None,
        compression: str = 'snappy'
    ):
        """Parquet形式で書き込み（型最適化付き）"""
        # データ型最適化
        df = self._optimize_dtypes(df)

        full_path = f"s3://{self.bucket}/{path}"

        if partition_cols:
            # パーティション書き込み
            df.to_parquet(
                full_path,
                engine='pyarrow',
                compression=compression,
                partition_cols=partition_cols,
                filesystem=self.fs
            )
        else:
            # 単一ファイル書き込み
            with self.fs.open(full_path, 'wb') as f:
                df.to_parquet(f, compression=compression)

        logger.info(f"Written {len(df)} rows to {full_path}")
        return full_path

    def read_parquet(
        self,
        path: str,
        columns: Optional[list] = None,
        filters: Optional[list] = None
    ) -> pd.DataFrame:
        """Parquet読み込み（プッシュダウン付き）"""
        full_path = f"s3://{self.bucket}/{path}"

        df = pd.read_parquet(
            full_path,
            columns=columns,
            filters=filters,
            filesystem=self.fs
        )

        logger.info(f"Read {len(df)} rows from {full_path}")
        return df

    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """データ型最適化でストレージ削減"""
        for col in df.columns:
            col_type = df[col].dtype

            if col_type != 'object':
                c_min = df[col].min()
                c_max = df[col].max()

                # 整数最適化
                if str(col_type)[:3] == 'int':
                    if c_min > -128 and c_max < 127:
                        df[col] = df[col].astype('int8')
                    elif c_min > -32768 and c_max < 32767:
                        df[col] = df[col].astype('int16')
                    elif c_min > -2147483648 and c_max < 2147483647:
                        df[col] = df[col].astype('int32')

                # 浮動小数点最適化
                else:
                    if c_min > -3.4e38 and c_max < 3.4e38:
                        df[col] = df[col].astype('float32')

        return df
```

### Phase 2: データ取得アセット実装（3日）

#### Day 4: JQuantsアセット
```python
# batch/assets/raw/jquants_price.py
from dagster import asset, Output, AssetMaterialization
from dagster import DailyPartitionsDefinition, DynamicPartitionsDefinition, MultiPartitionsDefinition
import pandas as pd
import aiohttp
import asyncio
from typing import List, Dict
import logging
from batch.config import settings
from batch.validators import ContractValidator
from batch.storage import MinIOClient

logger = logging.getLogger(__name__)

# パーティション定義
date_partitions = DailyPartitionsDefinition(
    start_date="2020-01-01",
    timezone="Asia/Tokyo"
)

ticker_partitions = DynamicPartitionsDefinition(name="ticker")

date_ticker_partitions = MultiPartitionsDefinition({
    "date": date_partitions,
    "ticker": ticker_partitions
})

@asset(
    partitions_def=date_ticker_partitions,
    metadata={
        "owner": "data-team",
        "sla_minutes": 5,
        "description": "JQuantsから日次価格データを取得"
    }
)
def jquants_price_raw(context) -> Output[pd.DataFrame]:
    """JQuants日次価格データ取得アセット"""

    # パーティションキー取得
    date = context.partition_key.keys_by_dimension["date"]
    ticker = context.partition_key.keys_by_dimension["ticker"]

    context.log.info(f"Fetching {ticker} for {date}")

    # JQuants APIクライアント（既存ロジック流用可能）
    fetcher = JQuantsFetcher()
    df = asyncio.run(fetcher.fetch_daily(ticker, date))

    # Data Contract検証
    validator = ContractValidator("batch/contracts/price_data.yaml")
    df = validator.validate(df)

    # MinIOに保存
    client = MinIOClient()
    path = f"raw/price/{date}/{ticker}.parquet"
    client.write_parquet(df, path)

    # メタデータ記録
    context.add_output_metadata({
        "num_rows": len(df),
        "ticker": ticker,
        "date": date,
        "path": path
    })

    return Output(df)

class JQuantsFetcher:
    """非同期JQuants APIクライアント"""

    def __init__(self):
        self.email = settings.JQUANTS_EMAIL
        self.password = settings.JQUANTS_PASSWORD
        self.max_concurrent = settings.JQUANTS_MAX_CONCURRENT
        self.token = None

    async def fetch_daily(self, ticker: str, date: str) -> pd.DataFrame:
        """日次データ取得"""
        async with aiohttp.ClientSession() as session:
            # 認証
            if not self.token:
                self.token = await self._authenticate(session)

            # データ取得
            url = f"https://api.jquants.com/v1/prices/daily"
            params = {
                "code": ticker,
                "date": date
            }
            headers = {"Authorization": f"Bearer {self.token}"}

            async with session.get(url, params=params, headers=headers) as resp:
                data = await resp.json()

        return pd.DataFrame(data['daily_quotes'])

    async def _authenticate(self, session: aiohttp.ClientSession) -> str:
        """認証トークン取得"""
        url = "https://api.jquants.com/v1/token/auth_user"
        data = {
            "mailaddress": self.email,
            "password": self.password
        }

        async with session.post(url, json=data) as resp:
            result = await resp.json()
            return result['refreshToken']
```

#### Day 5: コーポレートアクション・調整
```python
# batch/assets/cleaned/adjusted_price.py
from dagster import asset, AssetIn
import pandas as pd
from batch.storage import MinIOClient

@asset(
    ins={
        "price": AssetIn("jquants_price_raw"),
        "corporate_actions": AssetIn("corporate_actions_master")
    },
    metadata={
        "owner": "data-team",
        "description": "CA調整済み価格データ"
    }
)
def adjusted_price_data(context, price: pd.DataFrame, corporate_actions: pd.DataFrame) -> pd.DataFrame:
    """コーポレートアクション調整済み価格"""

    df = price.copy()

    # 分割・併合調整
    for _, ca in corporate_actions.iterrows():
        if ca['action_type'] == 'split':
            mask = (df['ticker'] == ca['ticker']) & (df['date'] < ca['action_date'])
            df.loc[mask, ['open', 'high', 'low', 'close']] /= ca['ratio']
            df.loc[mask, 'volume'] *= ca['ratio']

        elif ca['action_type'] == 'reverse_split':
            mask = (df['ticker'] == ca['ticker']) & (df['date'] < ca['action_date'])
            df.loc[mask, ['open', 'high', 'low', 'close']] *= ca['ratio']
            df.loc[mask, 'volume'] /= ca['ratio']

    # 権利落ち調整
    if 'dividend' in corporate_actions.columns:
        for _, div in corporate_actions[corporate_actions['action_type'] == 'dividend'].iterrows():
            mask = (df['ticker'] == div['ticker']) & (df['date'] < div['ex_date'])
            adjustment = 1 - (div['amount'] / df.loc[mask, 'close'].shift(1))
            df.loc[mask, ['open', 'high', 'low', 'close']] *= adjustment

    # MinIOに保存
    client = MinIOClient()
    date = context.partition_key.keys_by_dimension["date"]
    ticker = context.partition_key.keys_by_dimension["ticker"]
    path = f"cleaned/adjusted_price/{date}/{ticker}.parquet"
    client.write_parquet(df, path)

    context.log.info(f"Adjusted {len(df)} rows for {ticker}")

    return df

@asset(
    metadata={
        "owner": "data-team",
        "description": "コーポレートアクションマスタ"
    }
)
def corporate_actions_master(context) -> pd.DataFrame:
    """CA マスターデータ"""

    # 実際はJQuantsまたは外部ソースから取得
    ca_data = pd.DataFrame({
        'ticker': ['7203', '6758', '9984'],
        'action_type': ['split', 'split', 'dividend'],
        'action_date': pd.to_datetime(['2024-10-01', '2024-07-01', '2024-06-30']),
        'ratio': [5.0, 10.0, None],
        'amount': [None, None, 50.0],
        'ex_date': [None, None, pd.Timestamp('2024-06-28')]
    })

    # MinIOに保存
    client = MinIOClient()
    path = "master/corporate_actions.parquet"
    client.write_parquet(ca_data, path)

    return ca_data
```

#### Day 6: 投資部門別データ
```python
# batch/assets/raw/jquants_breakdown.py
from dagster import asset, Output
import pandas as pd
import numpy as np
from batch.storage import MinIOClient

@asset(
    partitions_def=date_ticker_partitions,
    metadata={
        "owner": "data-team",
        "description": "投資部門別売買データ"
    }
)
def investor_breakdown_raw(context) -> Output[pd.DataFrame]:
    """JQuants投資部門別データ取得"""

    date = context.partition_key.keys_by_dimension["date"]
    ticker = context.partition_key.keys_by_dimension["ticker"]

    # JQuants markets/breakdown エンドポイントから取得
    fetcher = JQuantsFetcher()
    df = asyncio.run(fetcher.fetch_breakdown(ticker, date))

    # 基本的な変換
    df['foreign_net'] = df['sell_value_foreign'] - df['buy_value_foreign']
    df['individual_net'] = df['sell_value_individual'] - df['buy_value_individual']
    df['institution_net'] = df['sell_value_institution'] - df['buy_value_institution']

    # MinIOに保存
    client = MinIOClient()
    path = f"raw/breakdown/{date}/{ticker}.parquet"
    client.write_parquet(df, path)

    return Output(df)
```

### Phase 3: 特徴量計算アセット（3日）

#### Day 7: テクニカル指標（最適化版）
```python
# batch/assets/features/technical.py
from dagster import asset, AssetIn
import pandas as pd
import pandas_ta as ta
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from batch.config import settings
from batch.storage import MinIOClient
import yaml

@asset(
    ins={"price": AssetIn("adjusted_price_data")},
    metadata={
        "owner": "ml-team",
        "compute": "cpu-intensive",
        "description": "テクニカル特徴量（127指標）"
    }
)
def technical_features(context, price: pd.DataFrame) -> pd.DataFrame:
    """最適化されたテクニカル特徴量計算"""

    # Feature Manifest読み込み
    with open("batch/config/features_manifest.yaml") as f:
        manifest = yaml.safe_load(f)

    # アクティブな特徴量のみ計算（713→127）
    active_features = [f for f in manifest['features'] if f['status'] == 'active']

    # 並列処理で高速計算
    with ProcessPoolExecutor(max_workers=settings.MAX_WORKERS) as executor:
        futures = []

        for feature in active_features:
            future = executor.submit(
                calculate_single_feature,
                price,
                feature
            )
            futures.append((feature['name'], future))

        # 結果収集
        results = {}
        for name, future in futures:
            results[name] = future.result()

    # DataFrame統合
    df = pd.concat([price[['ticker', 'date', 'close']], pd.DataFrame(results)], axis=1)

    # MinIOに保存
    client = MinIOClient()
    date = context.partition_key.keys_by_dimension["date"]
    path = f"features/technical/{date}.parquet"
    client.write_parquet(df, path, partition_cols=['ticker'])

    context.log.info(f"Calculated {len(active_features)} features")

    return df

def calculate_single_feature(df: pd.DataFrame, feature: dict) -> pd.Series:
    """単一特徴量計算"""

    name = feature['name']
    params = feature.get('params', {})

    # 主要な指標のみ実装（例）
    if name == 'returns_1d':
        return df['close'].pct_change(1)

    elif name == 'rsi_14':
        return ta.rsi(df['close'], length=14)

    elif name == 'sma_20':
        return ta.sma(df['close'], length=20)

    elif name == 'volume_ratio_20':
        return df['volume'] / df['volume'].rolling(20).mean()

    elif name == 'volatility_20':
        returns = df['close'].pct_change()
        return returns.rolling(20).std() * np.sqrt(252)

    elif name == 'atr_14':
        return ta.atr(df['high'], df['low'], df['close'], length=14)

    # 他の必要な指標を追加
    else:
        return pd.Series(index=df.index)
```

#### Day 8: 投資部門別フロー特徴量
```python
# batch/assets/features/investor_flow.py
from dagster import asset, AssetIn
import pandas as pd
import numpy as np
from scipy import stats
from batch.storage import MinIOClient

@asset(
    ins={
        "breakdown": AssetIn("investor_breakdown_raw"),
        "price": AssetIn("adjusted_price_data")
    },
    metadata={
        "owner": "ml-team",
        "description": "投資部門別フロー特徴量"
    }
)
def investor_flow_features(context, breakdown: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    """投資部門別フロー特徴量生成"""

    features = pd.DataFrame(index=breakdown.index)

    # 1. フローZスコア（5日/20日）
    for window in [5, 20]:
        foreign_net = breakdown['foreign_net']
        features[f'foreign_flow_z_{window}d'] = (
            (foreign_net - foreign_net.rolling(window).mean()) /
            foreign_net.rolling(window).std()
        )

        individual_net = breakdown['individual_net']
        features[f'individual_flow_z_{window}d'] = (
            (individual_net - individual_net.rolling(window).mean()) /
            individual_net.rolling(window).std()
        )

    # 2. フロー持続性
    features['foreign_persistence'] = (
        np.sign(breakdown['foreign_net'])
        .rolling(10)
        .sum()
    )

    # 3. 個人vs海外の拮抗度
    features['retail_vs_foreign'] = (
        (breakdown['individual_net'] - breakdown['foreign_net']) /
        (np.abs(breakdown['individual_net']) + np.abs(breakdown['foreign_net']) + 1e-8)
    )

    # 4. フローとリターンの相関
    returns = price['close'].pct_change()
    features['flow_return_corr_20d'] = (
        breakdown['foreign_net']
        .rolling(20)
        .corr(returns)
    )

    # 5. フロー反転検出
    features['flow_reversal'] = (
        (np.sign(breakdown['foreign_net']) !=
         np.sign(breakdown['foreign_net'].shift(1)))
        .astype(int)
    )

    # 6. 出来高正規化フロー
    volume_value = price['volume'] * price['close']
    features['normalized_foreign_flow'] = (
        breakdown['foreign_net'] / (volume_value + 1e-8)
    )

    # 7. セクター集約フロー（将来実装）
    # features['sector_flow_beta'] = calculate_sector_beta(...)

    # MinIOに保存
    client = MinIOClient()
    date = context.partition_key.keys_by_dimension["date"]
    path = f"features/investor_flow/{date}.parquet"
    client.write_parquet(features, path)

    context.log.info(f"Generated {len(features.columns)} flow features")

    return features
```

#### Day 9: 統合特徴量
```python
# batch/assets/features/composite.py
from dagster import asset, AssetIn, Output
import pandas as pd
from batch.storage import MinIOClient, ClickHouseClient

@asset(
    ins={
        "price": AssetIn("adjusted_price_data"),
        "technical": AssetIn("technical_features"),
        "flow": AssetIn("investor_flow_features")
    },
    metadata={
        "owner": "ml-team",
        "description": "統合ML特徴量"
    }
)
def ml_ready_features(
    context,
    price: pd.DataFrame,
    technical: pd.DataFrame,
    flow: pd.DataFrame
) -> Output[pd.DataFrame]:
    """ML用統合特徴量"""

    # 結合
    df = price[['ticker', 'date', 'close']].copy()
    df = df.merge(technical, on=['ticker', 'date'], how='left')
    df = df.merge(flow, on=['ticker', 'date'], how='left')

    # 欠損値処理
    df = df.fillna(method='ffill', limit=3)

    # 型最適化
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')

    # MinIOに保存（Parquet）
    minio_client = MinIOClient()
    date = context.partition_key.keys_by_dimension["date"]
    path = f"features/ml_ready/{date}.parquet"
    minio_client.write_parquet(df, path)

    # ClickHouseにも保存（高速クエリ用）
    ch_client = ClickHouseClient()
    ch_client.insert_dataframe(
        df,
        table="features.ml_ready",
        database="gogooku3"
    )

    # メタデータ
    context.add_output_metadata({
        "num_rows": len(df),
        "num_features": len(df.columns) - 3,  # ticker, date, close除く
        "date": str(date),
        "null_ratio": df.isnull().sum().sum() / df.size
    })

    return Output(df)
```

### Phase 4: オーケストレーション実装（2日）

#### Day 10: Dagsterジョブ・センサー
```python
# batch/dagster/jobs.py
from dagster import job, op, sensor, RunRequest, SkipReason
from batch.calendar import TSECalendar
import datetime
import pytz

@job(
    metadata={
        "owner": "data-team",
        "description": "日次バッチ処理パイプライン"
    }
)
def daily_batch_pipeline():
    """日次バッチ処理ジョブ"""
    # アセットの依存関係に基づいて自動実行
    pass

@sensor(
    job=daily_batch_pipeline,
    minimum_interval_seconds=300  # 5分間隔でチェック
)
def market_close_sensor(context):
    """東証クローズ後センサー"""

    jst = pytz.timezone('Asia/Tokyo')
    now = datetime.datetime.now(jst)

    # 15:00以降チェック
    if now.hour < 15:
        return SkipReason("Market still open")

    # 営業日チェック
    calendar = TSECalendar()
    if not calendar.is_business_day(now.date()):
        return SkipReason("Not a business day")

    # 本日分未実行なら実行
    run_key = f"daily_{now.date()}"

    if context.instance.has_run(run_key):
        return SkipReason(f"Already executed for {now.date()}")

    return RunRequest(
        run_key=run_key,
        tags={
            "date": str(now.date()),
            "triggered_by": "market_close_sensor"
        }
    )

@sensor(
    job=daily_batch_pipeline,
    minimum_interval_seconds=3600  # 1時間間隔
)
def retry_failed_partitions_sensor(context):
    """失敗パーティション再実行センサー"""

    # 失敗したパーティションを検出
    failed_partitions = context.instance.get_failed_partitions(
        asset_key="jquants_price_raw",
        after_timestamp=context.last_tick_completion_time
    )

    if not failed_partitions:
        return SkipReason("No failed partitions")

    # 再実行リクエスト生成
    run_requests = []
    for partition_key in failed_partitions[:10]:  # 一度に最大10個
        run_requests.append(
            RunRequest(
                run_key=f"retry_{partition_key}",
                partition_key=partition_key,
                tags={
                    "retry": "true",
                    "original_failure": str(partition_key)
                }
            )
        )

    return run_requests
```

#### Day 11: 監視・メトリクス
```python
# batch/metrics/prometheus.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
from functools import wraps

# メトリクス定義
data_fetch_counter = Counter(
    'batch_data_fetch_total',
    'Total data fetch attempts',
    ['source', 'status']
)

data_freshness_histogram = Histogram(
    'batch_data_freshness_minutes',
    'Data freshness in minutes',
    buckets=[1, 2, 3, 5, 10, 15, 30, 60]
)

feature_calculation_duration = Histogram(
    'batch_feature_calc_seconds',
    'Feature calculation duration',
    ['feature_type']
)

active_workers_gauge = Gauge(
    'batch_active_workers',
    'Number of active workers'
)

def track_metrics(metric_type: str):
    """メトリクス計測デコレーター"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()

            try:
                result = func(*args, **kwargs)
                data_fetch_counter.labels(
                    source=metric_type,
                    status='success'
                ).inc()
                return result

            except Exception as e:
                data_fetch_counter.labels(
                    source=metric_type,
                    status='failure'
                ).inc()
                raise e

            finally:
                duration = time.time() - start
                feature_calculation_duration.labels(
                    feature_type=metric_type
                ).observe(duration)

        return wrapper
    return decorator

# Prometheusサーバー起動
def start_metrics_server(port: int = 9090):
    """メトリクスサーバー起動"""
    start_http_server(port)
```

### Phase 5: テスト実装（2日）

#### Day 12: ゴールデンデータ・テスト
```python
# batch/tests/golden/test_golden_data.py
import pytest
import pandas as pd
import numpy as np
from batch.validators import ContractValidator
from batch.assets.cleaned import adjust_for_actions

class TestGoldenData:
    """ゴールデンデータテスト"""

    @pytest.fixture
    def golden_price_data(self):
        """テスト用ゴールデンデータ"""
        dates = pd.date_range('2024-01-01', '2024-01-31', freq='B')

        data = []
        for date in dates:
            for ticker in ['7203', '6758', '9984']:
                data.append({
                    'ticker': ticker,
                    'date': date,
                    'open': 1000 + np.random.randn() * 10,
                    'high': 1010 + np.random.randn() * 10,
                    'low': 990 + np.random.randn() * 10,
                    'close': 1000 + np.random.randn() * 10,
                    'volume': 1000000 + np.random.randint(-100000, 100000)
                })

        return pd.DataFrame(data)

    def test_contract_validation(self, golden_price_data):
        """契約検証テスト"""
        validator = ContractValidator("batch/contracts/price_data.yaml")
        validated = validator.validate(golden_price_data)
        assert len(validated) == len(golden_price_data)

    def test_corporate_action_adjustment(self, golden_price_data):
        """CA調整テスト"""
        ca_data = pd.DataFrame({
            'ticker': ['7203'],
            'action_type': ['split'],
            'action_date': pd.Timestamp('2024-01-15'),
            'ratio': [2.0]
        })

        adjusted = adjust_for_actions(golden_price_data, ca_data)

        # 分割前後の価格チェック
        pre_split = adjusted[
            (adjusted['ticker'] == '7203') &
            (adjusted['date'] < '2024-01-15')
        ]['close'].mean()

        post_split = adjusted[
            (adjusted['ticker'] == '7203') &
            (adjusted['date'] >= '2024-01-15')
        ]['close'].mean()

        # 約2倍の差があるはず
        assert abs(pre_split / post_split - 0.5) < 0.1
```

#### Day 13: プロパティテスト
```python
# batch/tests/property/test_features.py
import hypothesis
from hypothesis import strategies as st
import pandas as pd
import numpy as np
from batch.assets.features import calculate_single_feature

class TestFeatureProperties:
    """特徴量プロパティテスト"""

    @hypothesis.given(
        prices=st.lists(
            st.floats(min_value=1, max_value=10000),
            min_size=20,
            max_size=100
        )
    )
    def test_rsi_bounds(self, prices):
        """RSIが0-100の範囲内"""
        df = pd.DataFrame({'close': prices})
        feature = {'name': 'rsi_14', 'params': {'length': 14}}

        rsi = calculate_single_feature(df, feature)

        assert (rsi.dropna() >= 0).all()
        assert (rsi.dropna() <= 100).all()

    def test_returns_calculation(self):
        """リターン計算の正確性"""
        df = pd.DataFrame({
            'close': [100, 110, 121, 133.1]
        })

        feature = {'name': 'returns_1d'}
        returns = calculate_single_feature(df, feature)

        expected = [np.nan, 0.1, 0.1, 0.1]
        np.testing.assert_array_almost_equal(
            returns.values,
            expected,
            decimal=5
        )
```

## docker-compose設定

```yaml
# docker/docker-compose.batch.yml
version: '3.8'

services:
  minio:
    image: minio/minio:latest
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - minio_data:/data
    command: server /data --console-address ":9001"

  clickhouse:
    image: clickhouse/clickhouse-server:latest
    ports:
      - "8123:8123"
      - "9000:9000"
    volumes:
      - clickhouse_data:/var/lib/clickhouse
      - ./clickhouse/init.sql:/docker-entrypoint-initdb.d/init.sql

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"

  dagster:
    build:
      context: ../batch
      dockerfile: ../docker/batch.Dockerfile
    ports:
      - "3000:3000"  # Dagit UI
    environment:
      - DAGSTER_HOME=/opt/dagster/dagster_home
    volumes:
      - ../batch:/opt/dagster/app
      - dagster_home:/opt/dagster/dagster_home
    depends_on:
      - minio
      - clickhouse
      - redis

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  minio_data:
  clickhouse_data:
  dagster_home:
  grafana_data:
```

## Makefile

```makefile
# batch/Makefile
.PHONY: help setup test run clean

help:
	@echo "gogooku3 batch processing"
	@echo "========================"
	@echo "make setup    - 環境セットアップ"
	@echo "make test     - テスト実行"
	@echo "make run      - Dagster起動"
	@echo "make clean    - クリーンアップ"

setup:
	python -m venv venv
	./venv/bin/pip install -U pip
	./venv/bin/pip install -r requirements.txt
	docker-compose -f docker/docker-compose.batch.yml up -d

test:
	./venv/bin/pytest tests/ -v --cov=batch

run:
	./venv/bin/dagster dev -f batch/dagster/repository.py

clean:
	docker-compose -f docker/docker-compose.batch.yml down
	rm -rf venv __pycache__ .pytest_cache
```

## requirements.txt

```text
# batch/requirements.txt
# Core
dagster==1.5.0
dagster-webserver==1.5.0
pandas==2.1.0
numpy==1.24.0
pyarrow==14.0.0

# Data Sources
aiohttp==3.9.0
requests==2.31.0
jpholiday==0.1.9

# Storage
s3fs==2023.12.0
clickhouse-driver==0.2.6
redis==5.0.0

# Validation
pandera==0.17.0
great-expectations==0.18.0
pydantic==2.5.0

# Features
pandas-ta==0.3.14b0
scipy==1.11.0

# Testing
pytest==7.4.0
pytest-cov==4.1.0
hypothesis==6.90.0

# Monitoring
prometheus-client==0.19.0

# Utilities
python-dotenv==1.0.0
PyYAML==6.0.1
```

## 実装開始手順

```bash
# 1. ディレクトリ作成
cd /home/ubuntu/gogooku2/apps/gogooku3
mkdir -p batch/{assets,contracts,validators,storage,calendar,metrics,tests,config,dagster}
mkdir -p docker

# 2. 初期ファイル作成
touch batch/__init__.py
touch batch/config/settings.py
touch batch/contracts/price_data.yaml

# 3. 環境セットアップ
make setup

# 4. 最初のアセット実装
python batch/assets/raw/jquants_price.py

# 5. Dagster起動
make run
```

これで、既存batchの良い部分（並列処理、JQuants統合）を活かしながら、gogooku3の設計思想に基づいた堅牢なシステムが構築できます。
