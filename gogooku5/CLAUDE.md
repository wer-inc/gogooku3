# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

**gogooku5** is a modular refactoring of the gogooku3 monolithic ML system for Japanese stock market prediction. The architecture separates dataset generation (`data_v2/`) from model training (`models/`) with a shared dataset artifact consumed by multiple models (ATFT-GAT-FAN, APEX-Ranker).

**Current Infrastructure**: `data_v2/` uses **DuckDB** as the primary database backend with **Polars** for data transformations.

**Key Design Principles**:
- Complete separation of dataset generation and model training
- Multi-model support through shared versioned datasets
- DuckDB as system of record (single source of truth)
- 150+ financial features with Polars-native computation
- Independent package management with per-package `pyproject.toml`

## Essential Commands

### Dataset Generation (data_v2 - Current)

```bash
# Full dataset build (DuckDB-based)
make dataset

# Quick test (last 30 days)
make dataset-quick

# Or step-by-step:
make -C data_v2 duckdb-init                    # Initialize DuckDB
make -C data_v2 duckdb-load-calendar           # Fetch trading calendar
make -C data_v2 duckdb-fetch-listed START=2015-01-01 END=2025-12-31
make -C data_v2 duckdb-fetch-daily-quotes START=2015-01-01 END=2025-12-31
make -C data_v2 duckdb-fetch-breakdown START=2015-01-01 END=2025-12-31
make -C data_v2 duckdb-compute-features        # Compute 150+ features

# Export to Parquet (optional)
PYTHONPATH=data_v2 python -c "
import duckdb
con = duckdb.connect('data_v2/output/jquants.duckdb')
con.execute('COPY features_daily TO \"data_v2/output/features_daily.parquet\"')
"

# Check project status
make status
make health-check
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
# Launch Dagster UI
export DAGSTER_HOME=/workspace/gogooku3/gogooku5
PYTHONPATH=data_v2 dagster dev -m data_v2.dagster_project

# Execute jobs from CLI
make -C data_v2 dagster-job-duckdb-calendar
make -C data_v2 dagster-job-duckdb-listed START=2024-01-01 END=2024-12-31
make -C data_v2 dagster-job-duckdb-daily-quotes
make -C data_v2 dagster-job-duckdb-breakdown
make -C data_v2 dagster-job-duckdb-features
```

### Linting & Formatting

```bash
# Model packages
make -C models/apex_ranker lint
```

## High-Level Architecture

### Package Structure

```
gogooku5/
├── data_v2/                   # Current data infrastructure (DuckDB-based)
│   ├── etl/                   # Core ETL modules
│   │   ├── auth.py            # J-Quants token management
│   │   ├── client.py          # HTTP client for J-Quants API
│   │   ├── config.py          # Configuration (URLs, timeouts, paths)
│   │   ├── duckdb.py          # DuckDB connection management
│   │   ├── schemas.py         # SQL table schemas (KEY FILE)
│   │   ├── yfinance_tickers.py# Macro ticker definitions
│   │   ├── feature/           # Feature computation modules
│   │   │   └── prices.py      # Core price/breakdown features (900+ lines)
│   │   ├── normalize/         # API response normalization
│   │   └── upsert/            # DuckDB data insertion
│   ├── dagster_project/       # Dagster orchestration
│   ├── scripts/               # CLI entry points
│   │   └── duckdb_loader.py   # Main CLI tool
│   ├── output/                # Data outputs
│   │   └── jquants.duckdb     # DuckDB database file
│   ├── Makefile               # Build/run commands
│   ├── FEATURES.md            # Feature design document (150+ features)
│   ├── IMPLEMENTATION_GUIDE.md# Implementation roadmap
│   └── DUCKDB_QUICKSTART.md   # Quick reference
│
├── data_deprecated/           # Legacy Polars-based pipeline (archived)
│   ├── src/builder/           # Core dataset builder
│   │   ├── api/               # JQuants API clients, fetchers
│   │   ├── features/          # Feature engineering modules
│   │   ├── pipelines/         # DatasetBuilder orchestration
│   │   └── utils/             # Storage, cache, async, logging
│   ├── src/cli/               # CLI interface
│   └── src/dagster_gogooku5/  # Dagster assets & resources
│
├── models/                    # Model-specific packages
│   ├── apex_ranker/           # PatchTST stock ranking model
│   │   ├── src/apex_ranker/   # models, data, backtest, utils
│   │   ├── scripts/           # train_v0.py, inference_v0.py, backtest_v0.py
│   │   ├── configs/           # v0_base.yaml, feature_groups.yaml
│   │   └── Makefile.train     # train, inference targets
│   └── atft_gat_fan/          # (Planned migration from gogooku3)
│
├── common/                    # Shared utilities (optional, minimal)
│
├── tools/                     # Cross-package tools
│   ├── claude-code.sh         # Enhanced Claude Code launcher
│   └── codex.sh               # Codex CLI launcher
│
├── Makefile                   # Top-level delegation
├── MIGRATION_PLAN.md          # Migration roadmap and milestones
├── DATA_V2_EXPLORATION_REPORT.md  # Architecture overview
└── dagster.yaml               # Dagster instance config (JST timezone)
```

### Data Flow (data_v2)

```
J-Quants API → Normalize → DuckDB Tables → Feature Computation → features_daily
                               ↓
                    (daily_quotes, breakdown, listed_info, trading_calendar)
                               ↓
                         APEX-Ranker / ATFT-GAT-FAN
                               ↓
                          Model Training
```

### DuckDB Tables

| Table | Description | Primary Key |
|-------|-------------|-------------|
| `trading_calendar` | Trading days with holiday division | date |
| `listed_info` | Security metadata (sector, market) | (date, code) |
| `daily_quotes` | OHLCV + adjusted prices | (date, code) |
| `breakdown` | Buy/sell value/volume breakdown | (date, code) |
| `yf_prices` | Macro indices (SPY, VIX, etc.) | (date, ticker) |
| `features_daily` | 150+ computed features | (date, code) |

### Feature Categories (150+ features)

**Returns & Momentum** (13 features):
- `ret_1d, ret_5d, ret_20d, ret_60d` - multi-horizon log returns
- `mom_5, mom_10, mom_20, mom_chg_5_20` - momentum metrics
- `fwd_ret_1d/5d/10d/20d` - forward returns (targets)

**Volatility & Range** (11 features):
- `rv_5, rv_20, rv_ratio` - realized volatility
- `parkinson_20, gk_20` - High/Low-based volatility
- `atr_14, adx_14` - Average True Range, ADX

**Volume & Liquidity** (10 features):
- `vol_ratio_5/20, vol_z_20` - volume anomalies
- `amihud_1d, amihud_20` - price impact metrics

**Technical Indicators** (20+ features):
- `rsi_14, macd, macd_signal, macd_hist` - momentum indicators
- `bb_mid/upper/lower_20_2, bb_percent_b_20_2` - Bollinger Bands
- `ema_12, ema_26` - Exponential moving averages

**Breakdown/Demand-Supply** (50+ features):
- `sell_short_ratio, buy_margin_new_ratio` - composition ratios
- `net_flow_value, flow_imbalance` - net demand
- `short_squeeze_risk, long_liquidation_risk` - risk metrics
- `cs_pct_*` - cross-sectional percentiles

**Session Features** (Premium plan):
- `ret_morning/afternoon, session_divergence`
- `morning_win_rate_20, afternoon_win_rate_20`

For full feature list, see `data_v2/FEATURES.md`.

## Configuration & Environment

### Required Environment Variables

```bash
# J-Quants API (required)
JQUANTS_AUTH_EMAIL=your_email@example.com
JQUANTS_AUTH_PASSWORD=your_password
JQUANTS_PLAN_TIER=standard  # or 'premium'

# Copy .env.example to .env
cp data_v2/.env.example data_v2/.env
```

### data_v2/.env Configuration

```bash
# J-Quants credentials
JQUANTS_AUTH_EMAIL=xxx
JQUANTS_AUTH_PASSWORD=xxx

# Date range for feature computation
START=2015-01-01
END=2025-12-31

# Parallel workers
WORKERS=10
SLEEP_SEC=0.0

# Feature computation warmup
WARMUP_DAYS=260
```

## Migration Status

**data_v2 Infrastructure** (Current):
- ✅ DuckDB as system of record
- ✅ All 4 J-Quants endpoints (calendar, listed, quotes, breakdown)
- ✅ yfinance macro data integration
- ✅ 150+ feature computation implemented
- ✅ Dagster orchestration (jobs, ops)
- ✅ CLI tool (duckdb_loader.py)
- ⚠️ Cross-sectional features (partial)
- ⚠️ Run-length metrics (implemented)
- ❌ RAW/CORE layer architecture (designed, not implemented)

**Models**:
- ✅ APEX-Ranker model package (training, inference, backtest)
- 🚧 ATFT-GAT-FAN model migration

## Common Pitfalls

### 1. PYTHONPATH Setup

When running scripts directly (not via Makefile), set `PYTHONPATH`:

```bash
# ❌ Wrong
python data_v2/scripts/duckdb_loader.py

# ✅ Correct
PYTHONPATH=data_v2 python scripts/duckdb_loader.py --db output/jquants.duckdb init
```

### 2. DuckDB Connection

DuckDB defaults to `data_v2/output/jquants.duckdb`:

```bash
# Initialize tables
make -C data_v2 duckdb-init

# Or with custom path
PYTHONPATH=data_v2 python scripts/duckdb_loader.py --db /custom/path.duckdb init
```

### 3. Missing Calendar Data

Feature computation requires trading calendar:

```bash
# Fetch calendar first
make -C data_v2 duckdb-load-calendar

# Then fetch quotes/breakdown with --auto-fetch-calendar
make -C data_v2 duckdb-fetch-daily-quotes --auto-fetch-calendar
```

### 4. Warmup Period

Feature computation needs warmup period for rolling calculations:

```bash
# Default: 260 days warmup
# Requesting 2025 data? Need quotes from 2024 too.
make -C data_v2 duckdb-fetch-daily-quotes START=2024-01-01 END=2025-12-31
```

## Key Files to Read First

**Data Infrastructure (data_v2)**:
1. `data_v2/etl/schemas.py` - Table schemas and feature column definitions
2. `data_v2/etl/feature/prices.py` - Core feature computation (900+ lines)
3. `data_v2/scripts/duckdb_loader.py` - CLI entry point
4. `data_v2/Makefile` - Available targets

**Documentation**:
1. `data_v2/FEATURES.md` - 150+ feature definitions by category
2. `data_v2/DUCKDB_QUICKSTART.md` - Quick start guide
3. `DATA_V2_EXPLORATION_REPORT.md` - Full architecture analysis

**Model Training (APEX-Ranker)**:
1. `models/apex_ranker/src/apex_ranker/models/apex_ranker_v0.py` - Model architecture
2. `models/apex_ranker/scripts/train_v0.py` - Training script
3. `models/apex_ranker/configs/v0_base.yaml` - Hyperparameter config

## Development Workflow

**Building a Dataset**:
```bash
# 1. Initialize DuckDB
make -C data_v2 duckdb-init

# 2. Fetch calendar (required first)
make -C data_v2 duckdb-load-calendar

# 3. Fetch data
make -C data_v2 duckdb-fetch-listed START=2024-01-01 END=2024-12-31
make -C data_v2 duckdb-fetch-daily-quotes START=2024-01-01 END=2024-12-31
make -C data_v2 duckdb-fetch-breakdown START=2024-01-01 END=2024-12-31

# 4. Compute features
START=2024-01-01 END=2024-12-31 make -C data_v2 duckdb-compute-features
```

**Model Training**:
```bash
# 1. Ensure features are computed
make -C data_v2 duckdb-compute-features

# 2. Export to Parquet (if model expects file input)
# Or use DuckDB directly in model code

# 3. Train
make -C models/apex_ranker train
```

## References

- Migration roadmap: `MIGRATION_PLAN.md`
- Architecture analysis: `DATA_V2_EXPLORATION_REPORT.md`
- Feature definitions: `data_v2/FEATURES.md`
- APEX-Ranker README: `models/apex_ranker/README.md`
