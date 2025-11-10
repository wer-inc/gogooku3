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
```

## Output Artifacts
- Timestamped parquet + metadata pairs are written under `output/` using the pattern
  `ml_dataset_{YYYYMMDDYYYYMMDD}_{timestamp}_full.parquet`.
- Symlinks `ml_dataset_latest.parquet` and `ml_dataset_latest_full.parquet` always point to the newest dataset.
- Metadata is persisted alongside the parquet (`*_metadata.json`) with `ml_dataset_latest_metadata.json` tracking the latest snapshot.
- Retention (default: keep 3 snapshots) and compression (`zstd` by default) are configurable via `.env`.

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

Detailed pipeline behavior, feature coverage, and validation routines will be documented as implementation progresses through the migration milestones.

## Dagster Integration
`gogooku5/data/src/dagster_gogooku5` ships reusable Dagster assets that wrap the dataset builder:

```bash
# Launch Dagster UI with the gogooku5 definitions
DAGSTER_HOME=$PWD/gogooku5 PYTHONPATH=gogooku5/data/src dagster dev -f gogooku5/data/src/dagster_gogooku5/defs.py
```

- `g5_dataset_chunks`: builds DatasetBuilder chunks for a configurable date range. Configure `start`, `end`, `chunk_months`, etc. directly in Dagster.
- `g5_dataset_full`: merges the latest completed chunks by invoking the existing `data/tools/merge_chunks.py` helper.
- `dataset_builder_resource`: initializes `DatasetBuilder` with optional overrides (output dir, dataset tag, refresh behavior).

These assets allow you to schedule recurring dataset builds via Dagster jobs or run ad‑hoc chunk builds/merges from the UI with full observability.

> 🕒 **Timezone**
> `gogooku5/dagster.yaml` sets `instance.local_timezone` to `Asia/Tokyo`.
> Export `DAGSTER_HOME=$PWD/gogooku5` (or copy dagster.yaml into your DAGSTER_HOME) before running `dagster dev` / `dagster job …` to keep all Dagster timestamps in JST.
