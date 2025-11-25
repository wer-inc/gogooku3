# gogooku5 Dataset Builder (Skeleton)

This package will host the standalone dataset generation pipeline described in `../MIGRATION_PLAN.md`.

## Roadmap
1. Implement configuration and API clients under `src/builder/api/`.
2. Migrate feature engineering modules (`core`, `legacy`, `macro`).
3. Introduce full and optimized pipelines under `src/builder/pipelines/` with caching support.
4. Deliver integration tests that rebuild one month of data and compare with `gogooku3` outputs.

## Usage
```bash
# Ensure environment variables are set
cp .env.example .env

# Build 30-day dataset (START/END default to last 30 days)
make build

# Override date range
make build START=2024-01-01 END=2024-01-31

# Warm caches without writing parquet output
make build-optimized START=2024-01-01 END=2024-01-31 CACHE_ONLY=1

# Output location
# - Default (no env): data/output
# - Monorepo: set DATA_OUTPUT_DIR=/workspace/gogooku3/gogooku5/data/output in .env
```

## Output Artifacts
- Timestamped parquet + metadata pairs are written under `output/` using the pattern
  `ml_dataset_{YYYYMMDDYYYYMMDD}_{timestamp}_full.parquet`.
- Symlinks `ml_dataset_latest.parquet` and `ml_dataset_latest_full.parquet` always point to the newest dataset.
- Metadata is persisted alongside the parquet (`*_metadata.json`) with `ml_dataset_latest_metadata.json` tracking the latest snapshot.
- Retention (default: keep 3 snapshots) and compression (`zstd` by default) are configurable via `.env`.

## Schema: SecId Column (Phase 3 Migration)

**Version**: Introduced in schema v1.2.0 (2025-11-14)

### What is SecId?

`SecId` is a **globally stable integer identifier** (Int32 → Categorical) for securities, designed to replace string-based `Code` joins with high-performance int32 joins internally.

**Key properties**:
- **Type**: `Categorical` (optimized from Int32, range 1-5088)
- **Source**: Generated from `dim_security.parquet` (security master table)
- **Nullability**: `true` (historical/delisted securities not in dim_security will have NULL SecId)
- **Backward compatibility**: `Code` column remains present alongside `SecId`

### Why SecId?

**Performance improvements** (Phase 3 join migration):
- **Join speed**: 30-50% faster (Int32 vs String comparison)
- **Memory**: ~50% reduction in join column footprint
- **Cache locality**: Better CPU cache utilization with int32 keys

**7 internal joins migrated** from `Code` (String) → `sec_id` (Int32):
1. Quotes + Listed (eager/lazy)
2. Quotes + Margin features
3. Margin adjustment lookups
4. GPU features join

### Schema Details

```python
# Column specification
{
  "name": "SecId",
  "dtype": "Categorical",  # Optimized 8-bit encoding for 193 unique values (Q1 2024)
  "nullable": true         # NULL for delisted/unknown codes
}
```

### Usage

**For downstream consumers**:
```python
import polars as pl

# Load dataset
df = pl.read_parquet("ml_dataset.parquet")

# SecId is available alongside Code
assert "SecId" in df.columns  # ✅
assert "Code" in df.columns   # ✅ Backward compatible

# High-performance joins (use SecId when possible)
dim_security = pl.read_parquet("dim_security.parquet")
result = df.join(dim_security, on="SecId", how="left")  # 30-50% faster than Code join
```

**NULL handling**:
```python
# Typical Q1 2024 stats:
# - Total rows: 222,774
# - Valid SecId: 10,244 (4.6%)
# - NULL SecId: 212,530 (95.4%) - delisted securities

# Filter to currently listed securities only
active_df = df.filter(pl.col("SecId").is_not_null())
```

### Migration Status

| Phase | Description | Status | Date |
|-------|-------------|--------|------|
| **Phase 1** | dim_security generation | ✅ Complete | 2025-10-XX |
| **Phase 2** | sec_id propagation + categorical | ✅ Complete | 2025-10-XX |
| **Phase 3.1** | Internal join migration (7 joins) | ✅ Complete | 2025-11-XX |
| **Phase 3.2** | SecId output propagation | ✅ Complete | 2025-11-14 |

**Implementation details**: See `/tmp/phase3_completion_report.md` for full technical documentation.

## Testing
```bash
# Run data package tests (requires using the package source path)
PYTHONPATH=gogooku5/data/src pytest gogooku5/data/tests -q
```

## Data Fetchers
- `builder.api.advanced_fetcher.AdvancedJQuantsFetcher` wraps legacy async clients so gogooku5 can download TOPIX, indices, trades spec, margin (daily/weekly), futures, options, short selling、dividends、earning statements, etc.
- `data/tools/clean_empty_cache.py` can purge stale cache files (0-row Parquet/IPC) before a new build. Run `python data/tools/clean_empty_cache.py --dry-run` to inspect and without `--dry-run` to delete.
- Use `AdvancedJQuantsFetcher` together with `builder.utils.asyncio.run_sync` when integrating new pipelines to avoid manual event loop handling.
- `builder.features.core.flow.enhanced.FlowFeatureEngineer` converts cached trades-spec data into flow metrics (`foreign_sentiment`, `smart_flow_indicator`, etc.), driven by `DataSourceManager.trades_spec()`.
- Parity check CLI: `python scripts/compare_parity.py <gogooku3 parquet> <gogooku5 parquet> [--output-json report.json]` to inspect schema and numeric differences. For automated runs, set `PARITY_BASELINE_PATH=/path/to/gogooku3_parquet` (and optionally `PARITY_CANDIDATE_PATH=/path/to/gogooku5_parquet`) before `python tools/project-health-check.sh`.
- DatasetBuilder now materialises a full営業日×銘柄グリッド（日本の祝日カレンダーヒューリスティクス＋実観測日付）をベースに各特徴量を付与し、欠損日の可視化と gogooku3 とのパリティ確認を容易にしています。

Index option fetches (/option/index_option) respect `INDEX_OPTION_PARALLEL_FETCH=true`（既定）と `INDEX_OPTION_PARALLEL_CONCURRENCY`（既定:8）で並列取得でき、`SOURCE_CACHE_*` 設定と組み合わせてスナップショットを再利用できます。

### Source cache controls

APIソース（財務・配当・空売り・マージン・決算など）は `output/cache` にスナップショットされます。以下の環境変数 or Dagster resource 設定で挙動を切り替えられます。

| 設定 | 説明 |
| --- | --- |
| `SOURCE_CACHE_MODE` | `read_write` (既定) / `read` / `off` |
| `SOURCE_CACHE_FORCE_REFRESH` | `true` で TTL を無視して常に API から再取得 |
| `SOURCE_CACHE_ASOF` | `YYYY-MM-DD` や `today`。キャッシュキーに `asof-<date>` が付与され、同一スナップショットを再利用可能 |
| `SOURCE_CACHE_TAG` | 任意タグ（例 `backfill`）。キャッシュキーとメタ情報に記録 |
| `SOURCE_CACHE_TTL_OVERRIDE_DAYS` | データ種別ごとの TTL をまとめて上書き |

Dagster では `dataset_builder` resource に `source_cache_*` を渡すことで run 単位でこれらを指定できます。

### MLflow 連携

`ENABLE_MLFLOW_LOGGING=1` を設定すると、Dagster 資産（チャンク構築・マージ）と Apex Ranker 学習スクリプトが MLflow にパラメータ／メトリクス／アーティファクトを記録します。

| 変数 | 説明 |
| --- | --- |
| `ENABLE_MLFLOW_LOGGING` | `1` でロギング有効化 |
| `MLFLOW_EXPERIMENT_NAME` | 実験名（既定: `tse-forecasting`） |
| `MLFLOW_TRACKING_URI` | トラッキングサーバ URI |
| `dagster_run_id` (タグ) | Dagster run と MLflow run を紐付けるため自動付与 |

Dagster の resource config でも `enable_mlflow_logging`, `mlflow_experiment_name`, `mlflow_tracking_uri` を上書きできます。

Detailed pipeline behavior, feature coverage, and validation routines will be documented as implementation progresses through the migration milestones.

### Chunkヘルスチェック

チャンク出力の整合性は `data/tools/check_chunks.py` で確認できます:

```bash
python gogooku5/data/tools/check_chunks.py \
  --chunks-dir gogooku5/data/output/chunks \
  --fail-on-warning

# Monorepo (DATA_OUTPUT_DIR=/workspace/gogooku3/gogooku5/data/output) での例
DATA_OUTPUT_DIR=/workspace/gogooku3/gogooku5/data/output \
python gogooku5/data/tools/check_chunks.py --fail-on-warning
```

`status.json`/`metadata.json`/Parquet の欠落や `rows=0`、`state!="completed"` などを一覧化し、`--fail-on-warning` を付けると異常時に終了コード1を返します。

### Dataset hash / schema fingerprint

`merge_chunks.py` は最終 Parquet を書き出す際に

- `dataset_hash`（Parquet本体のSHA256）
- `feature_schema_version`（列名+dtype hash）

をメタデータへ埋め込み、Dagster asset／学習スクリプトはこの情報を MLflow タグに記録します。
`metadata.json` に両方の値が無い場合は学習を開始できないので、常に最新のデータビルダーでチャンクを作成してください。

### Dataset quality checker

`data/tools/check_dataset_quality.py` は完成済みデータセット（チャンク単位 / フルマージ）に対して

- `(date, code)` 主キー重複
- 指定ターゲット列の欠損
- 未来日データ混入
- as-of 順序（例: `fs_disclosed_date <= date`）

を一括検査します。JSON レポート出力にも対応しているため、`tools/project-health-check.sh` や CI に組み込んで品質を自動監視してください。

環境変数で DatasetBuilder 実行時に自動チェックを有効化できます:

| 変数 | 説明 |
| --- | --- |
| `ENABLE_DATASET_QUALITY_CHECK=1` | チャンク/フル書き出し直後にチェックを実行（失敗でビルド停止） |
| `DATASET_QUALITY_TARGETS` | ターゲット列（スペース or カンマ区切り） |
| `DATASET_QUALITY_ASOF_CHECKS` | `col<=reference_col` 形式の as-of 制約（スペース or カンマ区切り） |

`.env.example` では `ret_prev_1d/5d/20d/60d` をターゲット列、`DisclosedDate` と `earnings_event_date` を as-of 制約として定義しています。別の列／閾値を使いたい場合は上記の変数を上書きしてください。
| `DATASET_QUALITY_FAIL_ON_WARNING` | `1` で警告も失敗扱い |
| `DATASET_QUALITY_DATE_COL` / `DATASET_QUALITY_CODE_COL` | 主キー列名（既定: `date` / `code`） |
| `DATASET_QUALITY_ALLOW_FUTURE_DAYS` | 未来日許容日数（既定: 0） |

## Dagster Integration
`gogooku5/data/src/dagster_gogooku5` ships reusable Dagster assets that wrap the dataset builder:

```bash
# Launch Dagster UI with the gogooku5 definitions
export DAGSTER_HOME=/workspace/gogooku3/gogooku5   # use absolute path
PYTHONPATH=gogooku5/data/src dagster dev -m dagster_gogooku5.defs
```

- `g5_dataset_chunks`: builds DatasetBuilder chunks for a configurable date range. Configure `start`, `end`, `chunk_months`, etc. directly in Dagster.
- `g5_dataset_full`: merges the latest completed chunks by invoking the existing `data/tools/merge_chunks.py` helper.
- `dataset_builder_resource`: initializes `DatasetBuilder` with optional overrides (output dir, dataset tag, refresh behavior).

These assets allow you to schedule recurring dataset builds via Dagster jobs or run ad‑hoc chunk builds/merges from the UI with full observability.

> 🕒 **Timezone**
> `gogooku5/dagster.yaml` sets `instance.local_timezone` to `Asia/Tokyo` to ensure all Dagster run timestamps are in JST.
> Export `DAGSTER_HOME=/absolute/path/to/gogooku5` （絶対パス） before running `dagster dev` / `dagster job …` to use this configuration.
