# gogooku3 – Financial ML Pipeline Specification (OSS / Self‑Hosted GPU)

## 1. 目的・要件

* 目的: 金融MLパイプラインの **再現性・堅牢性・低コスト** を最大化（OSSのみ）。
* 非機能要件:

  * **EOD後データ新鮮度 p95 < 5分**, **日次成功率 ≥ 99.5%**, **学習 ≤ 120分**, **GPU利用率 ≥ 85%**。
  * 出力データは **フラットな配列のオブジェクト**（ネスト/`undefined`無し）。
  * すべての処理を **Dagster Asset** として管理、**契約（Data Contract）** と **チェック** を内蔵。

## 2. コンポーネント（OSS）

```yaml
orchestration:
  engine: Dagster OSS
  scheduler: Dagster Scheduler + TSE Calendar Sensor

storage_dwh:
  object_store: MinIO (S3 互換, Versioning 有効)
  file_format: Parquet/Arrow
  dev_dwh: DuckDB
  prod_dwh: ClickHouse (MergeTree, Partition: month, Order: ticker,date, TTL)

feature_store:
  offline: Feast (Parquet on MinIO)
  online: Feast + Redis

ml_platform:
  training: PyTorch + Hydra
  tracking: MLflow (backend: SQLite/Postgres, artifacts: MinIO)
  hpo: Optuna
  serving: Triton (TorchScript/ONNX), Champion/Challenger

quality_testing:
  data_contracts: pandera + Great Expectations (Dagster asset checks)
  tests: pytest + Hypothesis + 回帰テスト（IC/RMSE/Sharpe）

observability_security:
  metrics: Prometheus（OTEL Exporter）
  logging: Loki
  dashboard: Grafana
  alerting: Alertmanager（Slack/PagerDuty Webhook）
  secrets: SOPS + age
```

## 3. アーキテクチャ

```mermaid
graph TB
  subgraph Sources
    JQ[J-Quants]
    CA[Corporate Actions]
    CAL[TSE Calendar]
  end
  subgraph Orchestration (Dagster)
    S[Calendar Sensor]
    A[Assets (raw→features→training→registry)]
    Q[Asset Checks]
    RC[Run Coordination]
  end
  subgraph Storage/DWH
    M[MinIO (Parquet)]
    CH[ClickHouse]
    R[Redis (Online Features)]
  end
  subgraph ML
    TR[Training (PyTorch/Hydra/Optuna)]
    MF[MLflow Tracking/Registry]
    SRV[Triton (Champion/Challenger)]
  end
  subgraph Observability
    P[Prometheus]
    L[Loki]
    G[Grafana]
    AM[Alertmanager]
  end
  JQ-->S; CA-->S; CAL-->S
  S-->A; A-->Q; A-->M; M-->CH; CH-->TR; TR-->MF; MF-->SRV
  P-->G; L-->G; AM-->G
```

## 4. データ契約（Data Contracts）

### 4.1 価格データ契約（YAML）

```yaml
# contracts/price_data.yaml
name: price_data
version: 1.0.0
granularity: daily
primary_key: [ticker, date]
time_definition: { event_time: date, processing_time: _ingested_at, watermark: 1h }
duplicate_policy: reject
missing_policy: { strategy: forward_fill, max_gap: 3 }
schema:
  fields:
    - { name: ticker, type: string, constraints: { pattern: "^[0-9]{4}$", nullable: false } }
    - { name: date,   type: datetime, constraints: { tz: "Asia/Tokyo", nullable: false, business_day: true } }
    - { name: open,   type: float, constraints: { nullable: false, gt: 0 } }
    - { name: high,   type: float, constraints: { nullable: false, gte: open } }
    - { name: low,    type: float, constraints: { nullable: false, lte: open } }
    - { name: close,  type: float, constraints: { nullable: false, gt: 0 } }
    - { name: volume, type: int,   constraints: { nullable: false, ge: 0 } }
```

### 4.2 バリデーション（pandera + 追加規則）

```python
# quality/price_checks.py
import pandas as pd, pandera as pa
from pandera import Check, DataFrameSchema

class PriceDataValidator:
    schema = DataFrameSchema({
        "ticker": pa.Column(str, Check.str_matches(r"^\d{4}$")),
        "date":   pa.Column(pd.DatetimeTZDtype(tz="Asia/Tokyo")),
        "open":   pa.Column(float, Check.gt(0)),
        "high":   pa.Column(float),
        "low":    pa.Column(float),
        "close":  pa.Column(float, Check.gt(0)),
        "volume": pa.Column(int, Check.ge(0)),
    }, coerce=True)

    @staticmethod
    def ohlc(df: pd.DataFrame) -> bool:
        hi_ok = (df["high"] >= df[["open","low","close"]].max(axis=1)).all()
        lo_ok = (df["low"]  <= df[["open","high","close"]].min(axis=1)).all()
        assert hi_ok and lo_ok, "OHLC consistency failed"
        return True
```

### 4.3 TSEカレンダー（営業日）

```python
# calendar/tse_calendar.py
import datetime as dt, jpholiday

class TSECalendar:
    def __init__(self, start=2020, end=2035):
        self.holidays = {d for y in range(start, end+1) for d,_ in jpholiday.year_holidays(y)}
        for y in range(start, end+1):  # 12/31-1/3
            self.holidays.update({dt.date(y,12,31), dt.date(y+1,1,1), dt.date(y+1,1,2), dt.date(y+1,1,3)})
        self.special_closes = set()

    def is_business_day(self, d: dt.date) -> bool:
        return d.weekday() < 5 and d not in self.holidays and d not in self.special_closes
```

### 4.4 コーポレートアクション調整

```python
# corporate_actions/adjust.py
import pandas as pd

def adjust_for_actions(df: pd.DataFrame, ca: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for _, r in ca.sort_values("action_date").iterrows():
        mask = (out["ticker"] == r["ticker"]) & (out["date"] < pd.to_datetime(r["action_date"]))
        if r["action_type"] == "split":
            out.loc[mask, ["open","high","low","close"]] /= float(r["ratio"]); out.loc[mask, "volume"] *= float(r["ratio"])
        elif r["action_type"] == "reverse_split":
            out.loc[mask, ["open","high","low","close"]] *= float(r["ratio"]); out.loc[mask, "volume"] /= float(r["ratio"])
    return out
```

## 5. 特徴量

### 5.1 Feature Manifest

```yaml
# features/features.yaml
version: 1.0.0
features:
  active:
    - { name: returns_1d,     formula: "close/lag(close,1)-1",        dtype: float32, importance: high }
    - { name: rsi_14,         formula: "ta.rsi(close,14)",            dtype: float32, importance: high }
    - { name: vol_ratio_20,   formula: "volume/mean(volume,20)",      dtype: float32, importance: medium }
    - { name: foreign_z_5,    formula: "zscore(foreign_net,5)",       dtype: float32, importance: high, source: jquants_breakdown }
  experimental:
    - { name: flow_persist_10, formula: "sum(sign(foreign_net),10)",  dtype: int8,    importance: low }
  deprecated:
    - { name: bbands_upper_200, reason: "low value/high cost", deprecated_date: "2025-01-15" }
totals: { all: 713, active: 127, compute_savings_pct: 82.2 }
```

### 5.2 Feast 定義（Offline: Parquet on MinIO / Online: Redis）

```python
# feature_store/defs.py
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32
from datetime import timedelta

ticker = Entity(name="ticker", join_keys=["ticker"])

price_source = FileSource(
    path="s3://gogooku/features/price_features.parquet",
    timestamp_field="date",
    created_timestamp_column="created_at"
)

price_features = FeatureView(
    name="price_features",
    entities=[ticker],
    ttl=timedelta(days=1),
    schema=[
        Field(name="returns_1d",   dtype=Float32),
        Field(name="rsi_14",       dtype=Float32),
        Field(name="vol_ratio_20", dtype=Float32),
    ],
    source=price_source,
    online=True,
)
```

## 6. オーケストレーション（Dagster）

### 6.1 パーティション

```python
# dagster/partitions.py
from dagster import DailyPartitionsDefinition, DynamicPartitionsDefinition, MultiPartitionsDefinition
date_p = DailyPartitionsDefinition(start_date="2020-01-01", timezone="Asia/Tokyo")
ticker_p = DynamicPartitionsDefinition(name="ticker")
date_ticker = MultiPartitionsDefinition({"date": date_p, "ticker": ticker_p})
```

### 6.2 アセット

```python
# dagster/assets.py
from dagster import asset, AssetIn, Output
import pandas as pd
from calendar.tse_calendar import TSECalendar
from quality.price_checks import PriceDataValidator
from corporate_actions.adjust import adjust_for_actions

@asset(partitions_def=date_ticker, metadata={"owner":"data","sla_minutes":5})
def raw_price_data(context) -> Output[pd.DataFrame]:
    k = context.partition_key
    date = k.keys_by_dimension["date"]; ticker = k.keys_by_dimension["ticker"]
    df = fetch_from_jquants(ticker, date)                 # 既存取得関数を利用
    cal = TSECalendar()
    PriceDataValidator.schema.validate(df)
    PriceDataValidator.ohlc(df)
    assert df["date"].dt.date.map(cal.is_business_day).all()
    return Output(df, metadata={"rows": len(df), "ticker": ticker, "date": date})

@asset(partitions_def=date_ticker, ins={"price": AssetIn("raw_price_data")})
def technical_features(price: pd.DataFrame) -> pd.DataFrame:
    feats = compute_selected_features(price)              # Manifest準拠
    return feats

@asset(partitions_def=date_p, ins={"features": AssetIn("technical_features")}, metadata={"compute":"gpu"})
def ml_training(features: pd.DataFrame):
    # Hydra設定, Optuna, MLflow記録, Variance Watchdog, PurgedGroupKFold
    model = train_with_watchdog_and_cv(features)
    log_to_mlflow(model)
    return model
```

### 6.3 実行制御

* GPUジョブに `tags: {"compute":"gpu"}` を付与し、Dagsterの同時実行上限で **GPU並列数** を制御。
* I/O系は「ticker/date」の粒度で **小さな失敗・再実行** を可能に。

## 7. ML

### 7.1 Variance Watchdog（要点）

* 予測標準偏差EMA / 目的変数標準偏差 **< 0.1** で退避・早期停止。
* `std_ratio` を MLflow に記録、閾値割れ時にタグ `early_stop_reason=variance_collapse`。

### 7.2 クロスバリデーション

* **Purged Group K-Fold + Embargo**（銘柄リーク防止、イベント近傍除外）。

### 7.3 サービング

* **Triton** で Champion/Challenger（初期 10%→25%→100%）。
* p95 レイテンシ違反時は自動で Challenger の比率を縮小。

## 8. 監視・SLO

### 8.1 メトリクス（Prometheus 名称例）

* `data_freshness_minutes_bucket`（ヒストグラム）
* `pipeline_runs_total{status, pipeline}`（カウンタ）
* `training_duration_seconds`（ヒストグラム）
* `gpu_utilization{job="training"}`（ゲージ）

### 8.2 SLO（しきい値）

```yaml
slos:
  data_freshness_p95_min: 5
  pipeline_success_rate_pct: 99.5
  training_time_max_min: 120
  gpu_util_avg_pct: 85
```

### 8.3 代表 PromQL

* p95 新鮮度:
  `histogram_quantile(0.95, sum(rate(data_freshness_minutes_bucket[30d])) by (le))`
* 成功率:
  `sum(increase(pipeline_runs_total{status="success"}[30d])) / sum(increase(pipeline_runs_total[30d])) * 100`
* GPU利用率:
  `avg_over_time(gpu_utilization{job="training"}[1d])`

## 9. DWH（ClickHouse）

### 9.1 テーブル

```sql
CREATE TABLE features.price_features
(
  ticker LowCardinality(String),
  date   Date,
  sector LowCardinality(String),
  open Float32, high Float32, low Float32, close Float32,
  volume UInt64,
  returns_1d Float32, rsi_14 Float32, vol_ratio_20 Float32,
  created_at DateTime
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(date)
ORDER BY (ticker, date)
TTL date + INTERVAL 2 YEAR DELETE;
```

### 9.2 マテビュー（例）

```sql
CREATE MATERIALIZED VIEW features.daily_returns_mv
ENGINE = MergeTree
PARTITION BY toYYYYMM(date)
ORDER BY (sector, date)
AS
SELECT
  date, ticker, sector,
  close / lagInFrame(close) OVER (PARTITION BY ticker ORDER BY date) - 1 AS returns,
  stddevPopState(returns) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS volatility_20d
FROM features.price_features;
```

## 10. ストレージ I/O（Parquet→MinIO）

```python
# storage/parquet_io.py
import pyarrow as pa, pyarrow.parquet as pq, s3fs, pandas as pd

def write_parquet_minio(df: pd.DataFrame, url: str, endpoint="http://localhost:9000"):
    fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": endpoint})
    table = pa.Table.from_pandas(df, preserve_index=False)
    with fs.open(url, "wb") as f:
        pq.write_table(table, f, compression="snappy", data_page_size=1<<20)
```

## 11. シークレット管理（SOPS）

```yaml
# .sops.yaml
creation_rules:
  - path_regex: secrets/.*\.enc\.yaml
    age: ["age1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"]
```

## 12. テスト

* **ゴールデンデータ**：分割・権利落ち・臨時休場・訂正を含む固定小規模セット。
* **プロパティテスト**：RSI範囲\[0,100]、SMA(w=1)=恒等、BB上下関係、NaN伝播。
* **回帰テスト**：`featureset@version` + IC/RMSE/Sharpe を ±ε 管理。
* **オンライン一致**：Feast Offline vs Online（PSI < 0.2）。

## 13. プロモーション（dev→staging→prod）

* Dagster ジョブで **品質チェック合格時のみ昇格**。
* 特徴テーブルは **原子的スワップ**（ClickHouse: 一時テーブル→`RENAME TABLE`）。
* モデルは **MLflow Registry** のステージング/プロダクション段階を使用。

## 14. 実装受け入れ基準（Acceptance）

* **SLO達成**（§8.2）。
* **再現性**：同一 `featureset@version` でメトリクス差分 ±ε。
* **耐障害性**：任意の `(date,ticker)` 粒度の失敗が **局所再実行** で復旧可能。
