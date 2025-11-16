# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

**gogooku5** is a modular refactoring of the gogooku3 monolithic ML system for Japanese stock market prediction. The architecture separates dataset generation (`data/`) from model training (`models/`) with a shared dataset artifact consumed by multiple models (ATFT-GAT-FAN, APEX-Ranker).

**Key Design Principles**:
- Complete separation of dataset generation and model training
- Multi-model support through shared versioned datasets
- Independent package management with per-package `pyproject.toml`
- GPU-accelerated ETL using RAPIDS/cuDF
- Schema-validated feature engineering with SecId optimization

## Essential Commands

### Dataset Generation

```bash
# Build dataset for last 30 days (default)
make dataset

# Build with custom date range (from repository root)
make -C data build START=2024-01-01 END=2024-12-31

# Using CLI directly (chunk-based approach)
PYTHONPATH=data/src python -m cli.main build \
  --start 2024-01-01 --end 2024-12-31 \
  --chunk-months 1 --output-dir data/output

# Merge completed chunks into final dataset
python data/tools/merge_chunks.py \
  --chunks-dir data/output/chunks \
  --output data/output/ml_dataset_latest_full.parquet

# Validate dataset quality
python data/tools/check_dataset_quality.py \
  --dataset data/output/ml_dataset_latest_full.parquet \
  --targets ret_prev_1d,ret_prev_5d \
  --asof-checks "DisclosedDate<=Date,earnings_event_date<=Date"
```

### Model Training

```bash
# APEX-Ranker (PatchTST-based stock ranking)
make -C models/apex_ranker train               # Full training
make -C models/apex_ranker train-quick         # Smoke test
make -C models/apex_ranker inference \
  APEX_MODEL=output/models/apex_ranker_v0.pt

# ATFT-GAT-FAN (not yet migrated - placeholder)
make -C models/atft_gat_fan train
```

### Dagster Orchestration

```bash
# Launch Dagster UI (absolute path required for DAGSTER_HOME)
export DAGSTER_HOME=/workspace/gogooku3/gogooku5
PYTHONPATH=data/src dagster dev -m dagster_gogooku5.defs

# Materialize assets from CLI
dagster asset materialize -m dagster_gogooku5.defs \
  --select g5_dataset_chunks g5_dataset_full
```

### Testing & Validation

```bash
# Data package tests
PYTHONPATH=data/src pytest data/tests -v

# Compare gogooku3 vs gogooku5 parity
python data/scripts/compare_parity.py \
  /path/to/gogooku3.parquet \
  /path/to/gogooku5.parquet \
  --output-json parity_report.json

# Validate dependencies (detect import cycles, missing packages)
python data/scripts/validate_dependencies.py

# Check chunk integrity
python data/tools/check_chunks.py \
  --chunks-dir data/output/chunks \
  --fail-on-warning
```

### Linting & Formatting

```bash
# Data package
make -C data lint          # Run ruff + mypy
make -C data format        # Format with ruff

# Model packages (same pattern)
make -C models/apex_ranker lint
```

## High-Level Architecture

### Package Structure

```
gogooku5/
├── data/                      # Dataset generation (standalone package)
│   ├── src/builder/           # Core dataset builder
│   │   ├── api/               # JQuants API clients, fetchers
│   │   ├── features/          # Feature engineering modules
│   │   │   ├── core/          # Technical, volatility, graph, sector
│   │   │   ├── macro/         # TOPIX, VIX, indices, options
│   │   │   ├── fundamentals/  # Financial statements, dividends
│   │   │   └── events/        # Limit flags, corporate actions
│   │   ├── pipelines/         # DatasetBuilder orchestration
│   │   ├── chunks/            # Chunk planning and metadata
│   │   ├── validation/        # Quality checks, parity validation
│   │   └── utils/             # Storage, cache, async, logging
│   ├── src/cli/               # CLI interface (build, merge commands)
│   ├── src/dagster_gogooku5/  # Dagster assets & resources
│   ├── scripts/               # build.py, build_chunks.py, compare_parity.py
│   ├── tools/                 # check_chunks.py, merge_chunks.py, manage_raw_sources.py
│   ├── schema/                # feature_schema_manifest.json (versioned)
│   ├── output/                # Generated datasets, chunks, cache
│   ├── pyproject.toml         # Data package dependencies
│   └── Makefile               # build, lint, test targets
│
├── models/                    # Model-specific packages
│   ├── apex_ranker/           # PatchTST stock ranking model
│   │   ├── src/apex_ranker/   # models, data, backtest, utils
│   │   ├── scripts/           # train_v0.py, inference_v0.py, backtest_v0.py
│   │   ├── configs/           # v0_base.yaml, feature_groups.yaml
│   │   ├── pyproject.toml     # Model dependencies
│   │   └── Makefile.train     # train, inference targets
│   └── atft_gat_fan/          # (Planned migration from gogooku3)
│
├── common/                    # Shared utilities (optional, minimal)
│   ├── src/common/
│   │   ├── data/              # Dataset loader abstractions
│   │   ├── metrics/           # Shared evaluation metrics
│   │   └── utils/             # Logging, storage helpers
│   └── pyproject.toml
│
├── tools/                     # Cross-package tools
│   ├── claude-code.sh         # Enhanced Claude Code launcher
│   └── health-check.sh        # Project health diagnostics
│
├── Makefile                   # Top-level delegation (dataset, train-atft, train-apex)
├── MIGRATION_PLAN.md          # Migration roadmap and milestones
└── dagster.yaml               # Dagster instance config (JST timezone)
```

### Data Flow

```
JQuants API → DatasetBuilder → Chunks → Merge → ml_dataset_latest_full.parquet
                                                          ↓
                                            APEX-Ranker / ATFT-GAT-FAN
                                                          ↓
                                                   Model Training
```

### Critical Components

**DatasetBuilder** (`data/src/builder/pipelines/dataset_builder.py`):
- Orchestrates 30+ feature engineering modules
- Manages API fetching, caching, and prefetching
- Implements SecId-based joins (30-50% faster than Code)
- Outputs schema-validated Parquet with metadata

**Feature Engineering Modules**:
- `features/core/`: Technical indicators, volatility, graph networks, sector aggregation
- `features/macro/`: TOPIX, VIX, global indices, NK225 options, trading calendar
- `features/fundamentals/`: Financial statements (as-of T+1), dividends, breakdowns
- `features/events/`: Limit flags, corporate actions

**SecId Column** (Schema v1.2.0):
- Categorical int32 identifier (1-5088 range, optimized to 8-bit encoding)
- Replaces string `Code` for high-performance joins
- 7 internal joins migrated: quotes+listed, quotes+margin, GPU features
- Performance: 30-50% faster joins, ~50% memory reduction

**Source Cache System**:
- `output/cache/` stores API snapshots (TTL configurable)
- Environment variables: `SOURCE_CACHE_MODE`, `SOURCE_CACHE_ASOF`, `SOURCE_CACHE_TAG`
- Arrow IPC format for 3-5x faster reads vs Parquet

**Dataset Quality Checks**:
- Validates: (date, code) uniqueness, target column nulls, future data leaks, as-of ordering
- Configured via `.env`: `DATASET_QUALITY_TARGETS`, `DATASET_QUALITY_ASOF_CHECKS`
- Auto-runs when `ENABLE_DATASET_QUALITY_CHECK=1`

**MLflow Integration**:
- Tracks dataset builds (chunks, merges) and model training
- Environment: `ENABLE_MLFLOW_LOGGING=1`, `MLFLOW_EXPERIMENT_NAME`, `MLFLOW_TRACKING_URI`
- Dagster runs auto-tagged with `dagster_run_id`

## Important Design Patterns

### As-of Joins (Time-Series Safety)

Financial statements and dividends use T+1 as-of logic to prevent future peeking:

```python
# DisclosedDate (publication) must be <= Date (trading date)
# Example: Statement disclosed on 2024-01-15 is available from 2024-01-16 onwards
prepare_fs_snapshot(df_fs, settings)  # Adds DisclosedDate column
interval_join_pl(quotes, fs_snapshot, "Date", "DisclosedDate", None)
```

**Critical as-of constraints**:
- `fs_disclosed_date <= date` (financial statements)
- `earnings_event_date <= date` (earnings announcements)
- `dividend_ex_date >= date` (dividend ex-dates are forward-looking)

### Chunk-based Processing

Large datasets (10+ years) are split into monthly chunks for parallelization and resumability:

```python
# CLI automatically chunks by month
python -m cli.main build --start 2020-01-01 --end 2024-12-31 --chunk-months 1

# Output: data/output/chunks/20200101_20200131/, 20200201_20200229/, ...
# Each chunk: chunk.parquet, status.json, metadata.json

# Merge completed chunks
python data/tools/merge_chunks.py --chunks-dir data/output/chunks
```

**Dagster multiprocess executor** builds 3 chunks concurrently (255 CPU cores, 1.8TB RAM).

### GPU-Accelerated ETL

When RAPIDS/cuDF is available, use GPU for:
- Large joins (quotes + listed info)
- Rolling window calculations (adv60, volatility)
- Correlation graph construction

**Fallback to CPU** automatically if GPU unavailable or memory insufficient.

### Schema Validation

All datasets validated against `data/schema/feature_schema_manifest.json`:

```python
from builder.utils.schema_validator import SchemaValidator

validator = SchemaValidator("data/schema/feature_schema_manifest.json")
validator.validate(df)  # Raises if columns missing or dtype mismatch
```

**Manifest versioning**: v1.0.0 (baseline), v1.1.0 (macro features), v1.2.0 (SecId column)

### Dataset Artifacts

```python
from builder.utils import DatasetArtifact

artifact = DatasetArtifact.save(
    df,
    output_dir="data/output",
    tag="full",  # Creates ml_dataset_<daterange>_<timestamp>_full.parquet
    metadata={"git_commit": "abc123", "schema_version": "1.2.0"}
)
# Auto-creates symlinks: ml_dataset_latest_full.parquet, ml_dataset_latest_metadata.json
```

## Configuration & Environment

### Required Environment Variables

```bash
# J-Quants API (required)
JQUANTS_AUTH_EMAIL=your_email@example.com
JQUANTS_AUTH_PASSWORD=your_password
JQUANTS_PLAN_TIER=standard  # or 'premium'

# Dataset output (optional, defaults to data/output_g5)
DATA_OUTPUT_DIR=output_g5

# Quality checks (recommended for production)
ENABLE_DATASET_QUALITY_CHECK=1
DATASET_QUALITY_TARGETS=ret_prev_1d,ret_prev_5d,ret_prev_20d,ret_prev_60d
DATASET_QUALITY_ASOF_CHECKS=DisclosedDate<=Date,earnings_event_date<=Date
```

### Cache Configuration

```bash
# Cache format: 'parquet' (compatible), 'ipc' (3-5x faster), 'auto' (hybrid)
CACHE_FORMAT=auto

# Source cache controls
SOURCE_CACHE_MODE=read_write  # read_write, read, off
SOURCE_CACHE_ASOF=2024-12-31  # Snapshot date (enables cache reuse)
SOURCE_CACHE_TAG=backfill     # Custom tag for cache namespace

# TTL overrides (days, 0=no expiry)
CACHE_TTL_DAYS_MARGIN_DAILY=1
CACHE_TTL_DAYS_TOPIX=3
CACHE_TTL_DAYS_TRADES_SPEC=7
```

### Dagster Configuration

```bash
# Must use absolute path for dagster.yaml location
export DAGSTER_HOME=/workspace/gogooku3/gogooku5

# Instance timezone set to JST in dagster.yaml
# instance.local_timezone: Asia/Tokyo
```

## Migration Status (Phase 1)

**Completed**:
- ✅ Data package structure with independent `pyproject.toml`
- ✅ DatasetBuilder with 30+ feature modules
- ✅ SecId column optimization (Phase 1-3.2 complete)
- ✅ Chunk-based processing with CLI
- ✅ Dagster integration (assets, resources, jobs)
- ✅ Dataset quality validation framework
- ✅ Source cache system with TTL and as-of snapshots
- ✅ MLflow tracking integration
- ✅ APEX-Ranker model package (training, inference, backtest)

**In Progress**:
- 🚧 ATFT-GAT-FAN model migration
- 🚧 Common package utilities (minimal, add only when >1 consumer)

**Planned**:
- 📋 Additional model packages
- 📋 CI/CD pipeline integration
- 📋 Production deployment tooling

## Common Pitfalls

### 1. PYTHONPATH Setup

When running scripts directly (not via Makefile), set `PYTHONPATH`:

```bash
# ❌ Wrong
python data/src/cli/main.py build

# ✅ Correct
PYTHONPATH=data/src python -m cli.main build
```

### 2. Dagster Absolute Paths

Dagster requires absolute `DAGSTER_HOME`:

```bash
# ❌ Wrong
export DAGSTER_HOME=gogooku5

# ✅ Correct
export DAGSTER_HOME=/workspace/gogooku3/gogooku5
```

### 3. As-of Join Direction

Financial statements are disclosed BEFORE they apply to trading dates:

```python
# ❌ Wrong: Future peeking
df.join(fs, on="Date", how="left")

# ✅ Correct: As-of join with disclosed date
interval_join_pl(df, fs, "Date", "DisclosedDate", None)
```

### 4. SecId NULL Handling

SecId is NULL for delisted/unknown securities:

```python
# Filter to active securities only
active_df = df.filter(pl.col("SecId").is_not_null())

# Total Q1 2024: 222,774 rows, valid SecId: 10,244 (4.6%)
```

### 5. Chunk Status Validation

Always check chunk status before merging:

```bash
# ❌ Wrong: Merge without validation
python data/tools/merge_chunks.py --chunks-dir data/output/chunks

# ✅ Correct: Validate first
python data/tools/check_chunks.py --chunks-dir data/output/chunks --fail-on-warning
python data/tools/merge_chunks.py --chunks-dir data/output/chunks
```

## Dataset Schema Highlights

**Core Columns**:
- `Date`: Trading date (Polars Date type)
- `Code`: Stock code (String, e.g. "13010")
- `SecId`: Security ID (Categorical, 1-5088, NULL for delisted)

**Target Labels** (forward returns):
- `ret_prev_1d`, `ret_prev_5d`, `ret_prev_10d`, `ret_prev_20d`, `ret_prev_60d`

**Feature Groups** (300+ columns):
- Technical: OHLCV, volume, returns, momentum
- Volatility: Rolling std, parkinson, garman-klass
- Graph: Correlation networks, centrality measures
- Sector: Peer aggregations, sector momentum
- Macro: TOPIX, VIX, NK225 options, trading calendar
- Fundamentals: P/E, P/B, ROE, dividend yield (as-of T+1)
- Events: Limit flags, earnings announcements

**Schema Versioning**: See `data/schema/feature_schema_manifest.json`

## Tools & Utilities

**Data Tools** (`data/tools/`):
- `check_chunks.py`: Validate chunk integrity (status, metadata, parquet)
- `check_dataset_quality.py`: Validate final dataset (duplicates, nulls, as-of)
- `merge_chunks.py`: Merge completed chunks with schema validation
- `manage_raw_sources.py`: Inspect/export raw API snapshots
- `clean_empty_cache.py`: Purge 0-row cache files
- `materialize_asset.py`: CLI wrapper for Dagster asset materialization

**Data Scripts** (`data/scripts/`):
- `build_chunks.py`: CLI for chunk-based dataset generation
- `build_dim_security.py`: Generate dim_security.parquet (SecId master table)
- `compare_parity.py`: Compare gogooku3 vs gogooku5 datasets
- `validate_dependencies.py`: Detect import cycles and missing packages
- `warm_macro_cache.py`: Pre-populate macro feature caches

**Model Scripts** (`models/apex_ranker/scripts/`):
- `train_v0.py`: APEX-Ranker training with composite ranking loss
- `inference_v0.py`: Batch inference for top-K stock selection
- `backtest_v0.py`: Equal-weight portfolio backtest

## Key Files to Read First

When working on specific areas:

**Dataset Generation**:
1. `data/src/builder/pipelines/dataset_builder.py` - Main orchestration
2. `data/src/builder/api/data_sources.py` - API fetching abstraction
3. `data/src/builder/features/core/technical.py` - Example feature module
4. `data/schema/feature_schema_manifest.json` - Schema reference

**CLI & Orchestration**:
1. `data/src/cli/main.py` - CLI entry point
2. `data/src/cli/commands/build.py` - Build command implementation
3. `data/src/dagster_gogooku5/assets.py` - Dagster asset definitions
4. `data/src/dagster_gogooku5/resources.py` - DatasetBuilder resource

**Quality & Validation**:
1. `data/src/builder/validation/quality.py` - Quality check framework
2. `data/src/builder/validation/parity.py` - Parity comparison logic
3. `data/tools/check_chunks.py` - Chunk validation tool
4. `data/tools/check_dataset_quality.py` - Dataset quality CLI

**Model Training (APEX-Ranker)**:
1. `models/apex_ranker/src/apex_ranker/models/apex_ranker_v0.py` - Model architecture
2. `models/apex_ranker/src/apex_ranker/data/dataset_loader.py` - Dataset loading
3. `models/apex_ranker/scripts/train_v0.py` - Training script
4. `models/apex_ranker/configs/v0_base.yaml` - Hyperparameter config

## Development Workflow

**Adding New Features**:
1. Create feature module in `data/src/builder/features/<category>/`
2. Implement feature engineer class with `generate()` method
3. Register in `DatasetBuilder._run_feature_engineering()`
4. Update `feature_schema_manifest.json` with new columns
5. Test with small date range: `make -C data build START=2024-01-01 END=2024-01-31`
6. Validate schema: `SchemaValidator().validate(df)`

**Dataset Rebuild**:
1. Ensure `.env` configured with credentials
2. Run chunk build: `python -m cli.main build --chunk-months 1`
3. Validate chunks: `python data/tools/check_chunks.py`
4. Merge: `python data/tools/merge_chunks.py`
5. Quality check: `python data/tools/check_dataset_quality.py`

**Model Training**:
1. Ensure dataset exists: `data/output/ml_dataset_latest_full.parquet`
2. Configure training: Edit `models/apex_ranker/configs/v0_base.yaml`
3. Train: `make -C models/apex_ranker train`
4. Inference: `make -C models/apex_ranker inference APEX_MODEL=<path>`
5. Backtest: `python models/apex_ranker/scripts/backtest_v0.py`

## References

- Migration roadmap: `MIGRATION_PLAN.md`
- Data package README: `data/README.md`
- APEX-Ranker README: `models/apex_ranker/README.md`
- Data package CLAUDE.md: `data/CLAUDE.md` (package-specific guidance)
