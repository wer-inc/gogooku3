# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

ATFT-GAT-FAN: Advanced financial ML system for Japanese stock market prediction. Multi-horizon forecasting using Graph Attention Networks with production-grade time-series safety protocols. Target Sharpe ratio: 0.849.

### Hardware Environment
- **GPU**: NVIDIA A100 80GB PCIe
- **CPU**: 24-core AMD EPYC 7V13
- **Memory**: 216GB RAM
- **Storage**: 291GB SSD

## Essential Commands

### Quick Start
```bash
# Install and verify
pip install -e .
cp .env.example .env  # Configure JQUANTS_AUTH_EMAIL, JQUANTS_AUTH_PASSWORD
python -c "import gogooku3; print(f'✅ v{gogooku3.__version__}')"

# Pre-commit setup
pre-commit install
pre-commit install -t commit-msg
```

### Primary Training Commands
```bash
# RECOMMENDED: Production optimized training (PDF analysis-based improvements)
make train-optimized              # All improvements applied
make train-optimized-dry          # Check configuration
make train-optimized-report       # Show optimization report

# Alternative training methods
make train-integrated              # Standard integrated pipeline
make train-integrated-safe         # With SafeTrainingPipeline validation
make train-atft                   # Direct ATFT training
make smoke                         # Quick 1-epoch test

# GPU-specific training
REQUIRE_GPU=1 make train-optimized  # Force GPU usage
make train-gpu-latest              # Use latest dataset with GPU
```

### Data Pipeline

**Note**: Dataset commands are now organized in `Makefile.dataset` with a clean 3-layer structure.

```bash
# Layer 1: User-Friendly (RECOMMENDED)
make dataset-bg                    # SSH-safe background build (5 years, GPU)
make go                            # Alias for dataset-bg
make dataset                       # Interactive all-in-one build

# Layer 2: Detailed Control
make dataset-gpu START=2015-09-27 END=2025-09-26  # GPU-ETL with RAPIDS/cuDF
make dataset-cpu START=2020-09-06 END=2025-09-06  # CPU-only fallback
make dataset-prod START=2020-09-06 END=2025-09-06 # Production config
make dataset-research START=2020-09-06 END=2025-09-06  # Research features

# Layer 3: Utilities
make dataset-check                 # Environment check (relaxed)
make dataset-check-strict          # Environment check (strict GPU)
make dataset-clean                 # Clean artifacts (keep raw/cache)
make dataset-rebuild               # Clean + rebuild with defaults
make cache-stats                   # Show cache statistics
make cache-prune                   # Prune old cache (120d)

# Legacy commands (still supported)
make dataset-full-gpu START=... END=...  # Direct GPU generation
make fetch-all START=... END=...         # Raw data only
python scripts/data/ml_dataset_builder.py  # Build from existing
```

### Testing & Validation
```bash
# Run specific test
pytest tests/unit/test_file.py::test_function -v

# Test suites
pytest tests/unit/ -v          # Unit tests
pytest tests/integration/ -v   # Integration tests
pytest -m "not slow"           # Skip slow tests

# Validation scripts
python scripts/smoke_test.py --max-epochs 1
python scripts/validate_improvements.py --detailed
python scripts/test_phase1_features.py  # J-Quants Phase 1
python scripts/test_phase2_features.py  # J-Quants Phase 2
```

### Research & Analysis
```bash
make research-plus              # Complete research bundle
make research-baseline DATASET=output/ml_dataset_latest_full.parquet
make research-lags PATTERN="output/*.parquet"
make research-report FACTORS=returns_5d,ret_1d_vs_sec HORIZONS=1,5,10,20
```

### HPO (Hyperparameter Optimization)
```bash
make hpo-setup
make hpo-run HPO_TRIALS=20 HPO_STUDY=atft_hpo_production
make hpo-status
make hpo-resume
```

### Code Quality
```bash
ruff check src/ --fix          # Auto-fix linting
ruff format src/               # Format code
mypy src/gogooku3             # Type checking
pre-commit run --all-files     # Full quality check
```

### Cache Management
```bash
# Verify cache configuration and integrity
make cache-verify              # Comprehensive 3-step verification
                              # 1. Check USE_CACHE=1 in .env
                              # 2. Verify cache directories exist
                              # 3. List cache files with sizes

# Quick cache overview
make cache-status              # Show cache state summary

# Clean cache files
make cache-clean               # Interactive cache cleanup (with confirmation)

# View cache documentation
make cache-info                # Display cache help and documentation

# Daily cache updates (automatic pre-population)
make update-cache              # Update caches manually (verbose mode)
make update-cache-silent       # Silent mode for cron execution

# Setup automatic daily cache updates (recommended for production)
# Add to crontab (crontab -e):
# 0 8 * * * cd /root/gogooku3 && make update-cache-silent >> /var/log/gogooku3_cache.log 2>&1
#
# This ensures:
# - 100% cache hit rate during daytime (8am-midnight)
# - No API waste from 1-day date differences
# - Caches updated with latest day's data every morning
# - Silent execution (only errors logged)

# Expected cache structure after first dataset build:
# output/raw/prices/          (~2-3GB for 5-10 years of OHLCV data)
# output/raw/indices/          (~5-10MB for TOPIX index data)
# output/raw/statements/       (~50-200MB for financial statements)
# output/graph_cache/          (~500MB-1GB for correlation graphs)

# Cache performance impact:
# - Daily Quotes Fetch: 45-60s → 2-3s (95% faster)
# - API Calls (10 years): ~2,520 → 0 (100% saved)
# - Total Build Time: 10-15 min → 5-8 min (40% faster)
# - With daily updates: 100% cache hit (no misses after 8am)
```

## High-Level Architecture

### Training Pipeline Flow

```
JQuants API → Raw Data → Feature Engineering → ML Dataset → Training Pipeline → Model
                                                      ↓
                                            SafeTrainingPipeline (optional)
                                                      ↓
                                              Walk-Forward Splits
                                                      ↓
                                              ATFT-GAT-FAN Model
```

### Core Components

**`scripts/pipelines/run_full_dataset.py`** - Dataset generation orchestrator:
- Coordinates dataset creation (up to 395 features with all data sources; currently ~303-307 features)
- All feature modules enabled by default (GPU-ETL, graphs, options, etc.)
- **Note**: Futures features (88-92 columns) disabled due to API unavailability
- Manages JQuants API fetching with up to 75 concurrent requests
- Integrates run_pipeline_v4_optimized.py for base data

**`scripts/pipelines/run_pipeline_v4_optimized.py`** - Optimized data pipeline:
- JQuantsAsyncFetcher with semaphore control (MAX_CONCURRENT_FETCH=75)
- Axis-based optimization (by-date vs by-code fetching)
- Batch processing with asyncio.gather()
- Event detection and incremental updates

**`scripts/integrated_ml_training_pipeline.py`** - Training orchestrator:
1. Loads ML dataset (auto-detects or uses `--data-path`)
2. Optionally runs SafeTrainingPipeline (`--run-safe-pipeline`)
3. Builds correlation graphs (`--adv-graph-train`)
4. Executes `train_atft.py` with Hydra configs
5. Handles OOM retry, GPU detection, parameter passthrough

**`scripts/train_atft.py`** - Core training script:
- Hydra-based configuration management
- Phase-based training (baseline → GAT → FAN → finetune)
- Multi-horizon prediction [1d, 5d, 10d, 20d]
- Loss weights controlled via `PHASE_LOSS_WEIGHTS` env var

**`src/gogooku3/`** - Modern package structure (v2.0.0):
- `data/`: ProductionDatasetV3, CrossSectionalNormalizerV2, WalkForwardSplitterV2
- `features/`: QualityFinancialFeaturesGenerator, up to 395 features (~307 active; 88-92 futures disabled)
- `models/`: ATFT-GAT-FAN (~5.6M params), LightGBM baseline
- `training/`: SafeTrainingPipeline (7-step validation)
- `graph/`: FinancialGraphBuilder (correlation networks)

### Critical Safety Mechanisms

**Data Leakage Prevention**:
- Walk-Forward validation with 20-day embargo
- Cross-sectional normalization (daily Z-score)
- No BatchNorm in time-series (prevents cross-sample leakage)
- T+1 as-of joins for statements (15:00 JST cutoff)

**Training Stability**:
- OOM auto-retry with batch halving
- Gradient clipping (norm=1.0)
- Mixed precision (bf16)
- EMA teacher model
- Snapshot ensembling

## Recent Production Optimizations

### PDF Analysis-Based Improvements (Implemented)

**1. Multi-Worker DataLoader**:
- `ALLOW_UNSAFE_DATALOADER=1` overrides safety guard
- `NUM_WORKERS=8`, `PERSISTENT_WORKERS=1`
- GPU utilization improved 2-3x

**2. Model Capacity**:
- `hidden_size: 64 → 256` (~20M params)
- Better pattern learning capacity

**3. Loss Function Optimization**:
- `USE_RANKIC=1`, `RANKIC_WEIGHT=0.2`
- `CS_IC_WEIGHT=0.15` (increased from 0.05)
- `SHARPE_WEIGHT=0.3`
- Phase-based gradual shift to financial metrics

**4. torch.compile**:
- `improvements.compile_model=true`
- `TORCH_COMPILE_MODE=max-autotune`
- 10-30% speed improvement on PyTorch 2.x

**5. Scheduler**:
- Plateau scheduler instead of Cosine
- Monitor `val/rank_ic_5d`
- Adaptive learning rate reduction

### Phase 2: Smart Partial Match Cache (Implemented)

Phase 2 implements three advanced cache optimization strategies to maximize cache hit rates and minimize unnecessary API calls:

**1. Coverage Threshold Fallback** (`run_pipeline_v4_optimized.py:160-206`):
- Rejects partial cache files with insufficient coverage (<30% by default)
- Falls back to full API fetch when cache provides minimal benefit
- Prevents inefficient "worst of both worlds" scenario (cache load + most data from API)
- **Environment variable**: `MIN_CACHE_COVERAGE=0.3` (range: 0.0-1.0)
- **Example**: Requesting 10 days but only 2 days cached → Full API fetch (more efficient)

**2. Multi-Cache File Merging** (`run_pipeline_v4_optimized.py:222-397, 494-583`):
- Automatically combines up to 3 cache files to maximize coverage
- Greedy algorithm selects files that fill date gaps
- Unified loader handles both single-file and multi-file cases
- **Environment variable**: `ENABLE_MULTI_CACHE=1` (1=enabled, 0=disabled)
- **Example**: Cache1 (days 1-5) + Cache2 (days 6-10) → 100% coverage, no API needed
- **Performance**: Can improve cache hit rate from 50% → 100% in overlapping scenarios

**3. Contiguous Range Merging** (`run_pipeline_v4_optimized.py:444-491, 1376-1378`):
- Merges adjacent date ranges into single API calls
- Reduces API request count and improves efficiency
- Applied to TOPIX, statements, and price data fetching
- **Example**: Fetching (2025-01-01, 2025-01-05) + (2025-01-06, 2025-01-10) → Single call for (2025-01-01, 2025-01-10)
- **Performance**: Can reduce API calls by 50-70% in typical scenarios

**Production Results**:
- Total pipeline time: 5.13 seconds
- Cache hit rate: 100% (3/3 sources)
- Time saved: ~78 seconds per run
- Speedup: 1529% faster vs cold cache
- Zero API calls with warm cache

**Key Technical Details**:
- Handles Polars Date vs String type conversions (`_load_cache_data()`)
- Extended cache saving: Merged data saved with full date range for future efficiency
- Age-based validation: Respects `CACHE_MAX_AGE_DAYS` for all cache files
- Graceful degradation: Falls back to API fetch if any optimization fails

**When Optimizations Activate**:
1. **Coverage check**: Always evaluated first for partial matches
2. **Multi-cache**: Activates when single file coverage < MIN_CACHE_COVERAGE and ENABLE_MULTI_CACHE=1
3. **Range merging**: Always applied to missing date ranges before API fetch

### Configuration Files

**Primary Configs**:
- `configs/atft/config_production_optimized.yaml` - Latest optimized settings
- `configs/atft/train/production_improved.yaml` - Improved training config
- `configs/atft/feature_categories.yaml` - Feature grouping (189 features → categories)

**Key Environment Variables**:
```bash
# JQuants API credentials (required)
JQUANTS_AUTH_EMAIL=xxx
JQUANTS_AUTH_PASSWORD=xxx

# Performance optimization
MAX_CONCURRENT_FETCH=75      # JQuants API parallel requests
MAX_PARALLEL_WORKERS=20       # CPU parallel processing
USE_GPU_ETL=1                # GPU-accelerated ETL (RAPIDS/cuDF)
RMM_POOL_SIZE=70GB           # GPU memory pool for A100 80GB
CUDA_VISIBLE_DEVICES=0       # GPU device selection

# Cache optimization (REQUIRED - major performance impact)
USE_CACHE=1                  # Enable price data caching (default: 1)
                            # CRITICAL: Without this, multi-GB price data
                            # is re-fetched from API every time (45-60s waste)
CACHE_MAX_AGE_DAYS=7        # Maximum cache age in days (default: 7)
GCS_SYNC_AFTER_SAVE=1       # Sync cache to GCS after save (default: 1)

# Phase 2: Smart Partial Match Cache optimizations
MIN_CACHE_COVERAGE=0.3      # Minimum cache coverage threshold (0.0-1.0)
                            # Reject partial cache if coverage < 30%
ENABLE_MULTI_CACHE=1        # Enable multi-file cache combination (1=on, 0=off)
                            # Automatically merges up to 3 cache files to maximize coverage

# Training optimization
ALLOW_UNSAFE_DATALOADER=1    # Enable multi-worker DataLoader
NUM_WORKERS=8                # DataLoader workers
PERSISTENT_WORKERS=1         # Reuse workers across epochs
PREFETCH_FACTOR=4           # Prefetch batches
PIN_MEMORY=1                # GPU memory pinning

# Loss function optimization
USE_RANKIC=1                 # RankIC loss component
RANKIC_WEIGHT=0.2           # RankIC weight
CS_IC_WEIGHT=0.15           # Cross-sectional IC weight
SHARPE_WEIGHT=0.3           # Sharpe ratio weight

# Memory optimization
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # GPU memory fragmentation fix
```

### Makefile Organization

The project uses a modular Makefile structure for better maintainability:

**`Makefile`** (635 lines) - Main entry point:
- General setup, testing, and training commands
- Includes `Makefile.dataset` for all dataset operations
- Simplified main `help` target

**`Makefile.dataset`** (318 lines) - Dataset generation module:
- **Layer 1 (User-Friendly)**: `dataset-bg`, `go`, `dataset`
  - SSH-safe background execution
  - Sensible defaults (last 5 years)
  - Automatic preflight checks
- **Layer 2 (Detailed Control)**: `dataset-gpu`, `dataset-cpu`, `dataset-prod`, `dataset-research`
  - Explicit date range control
  - GPU/CPU selection
  - Production/research configurations
- **Layer 3 (Utilities)**: `dataset-check`, `dataset-clean`, `cache-stats`, `cache-prune`
  - Environment validation
  - Cache management
  - Cleanup and rebuild

**Key Features**:
- Updated `RMM_POOL_SIZE=40GB` (from 0) for OOM prevention
- Monthly cache sharding: `output/graph_cache/YYYYMM/w{WINDOW}-t{THRESHOLD}-k{K}/`
- All legacy commands (`dataset-full-gpu`, etc.) remain supported for backward compatibility

## Common Issues & Solutions

### Training Errors

**Hydra argument errors**:
```bash
# If "unrecognized arguments" error occurs:
# Use CLI flags for integrated_ml_training_pipeline.py:
--data-path, --batch-size, --lr, --max-epochs
# Not as Hydra overrides
```

**CUDA OOM**:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Or reduce batch size in config
```

**No dataset found**:
```bash
# Build dataset first (use new simplified commands)
make dataset-bg                    # Background build (recommended)
# Or with specific dates:
make dataset-gpu START=2020-09-06 END=2025-09-06
# Check symlink
ls -la output/ml_dataset_latest_full.parquet
```

### Data Pipeline Issues

**JQuants API errors**:
- Check `.env` credentials
- Rate limiting: max 75 concurrent requests
- Use date ranges, not per-stock fetches

**Feature mismatch warnings**:
- Check `configs/atft/feature_categories.yaml`
- Ensure all active features properly categorized (~307 features)
- VSN/FAN/SAN require correct grouping
- **Note**: Futures features (88-92 columns) are disabled by default

**Graph builder warnings (normal for early dates)**:
```
WARNING - No valid codes for graph building on 2015-11-10
WARNING - Insufficient codes (0) for graph building
```
- **Cause**: Early dates (2015-2016) have limited data
- **Impact**: None - graph features set to 0 for those dates
- **Solution**: Expected behavior, processing continues normally

**NumPy RuntimeWarning in correlation**:
```
RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
```
- **Cause**: Zero standard deviation (no price movement)
- **Impact**: None - NaN values handled automatically
- **Solution**: Normal behavior for inactive stocks

**Futures features (currently disabled)**:
- **Status**: Disabled due to JQuants `/derivatives/futures` API unavailability
- **Missing features**: 88-92 columns
  - ON (T+0): 20 columns (5 features × 4 categories: TOPIXF, NK225F, JN400F, REITF)
  - EOD (T+1): 68 columns (17 features × 4 categories)
  - Continuous (optional): 4 columns (with `--futures-continuous` flag)
- **Impact**: "395 features" is theoretical maximum including futures; actual count ~303-307
- **Code locations**:
  - `run_full_dataset.py:665,775` - API fetch disabled with `if False`
  - `run_full_dataset.py:879-882` - Force `enable_futures=False` in enrich_and_save
- **Workaround**: Re-enable experimentally with offline parquet data:
  ```bash
  python scripts/pipelines/run_full_dataset.py \
    --futures-parquet output/futures_daily.parquet \
    --futures-categories "TOPIXF,NK225F"
  ```
- **Future columns reference**: `src/gogooku3/features/futures_features.py`
- **Historical context**: See `FUTURES_INTEGRATION_COMPLETE.md` for past integration

**Note on data-dependent features**:
Daily margin (`dmi_*`), short selling, and sector short selling features are enabled by default but **depend on actual data availability**. If data cannot be fetched or does not exist for the date range, these columns may be NULL or not generated. This is why the actual column count can vary slightly (~303-307) even excluding futures.

### Cache Issues

**Cache not being used (always seeing "CACHE MISS")**:
```bash
# 1. Verify USE_CACHE is set in .env
grep USE_CACHE .env
# Expected: USE_CACHE=1  # 価格データのキャッシュを有効化

# 2. Run comprehensive verification
make cache-verify

# 3. Check cache directory exists after first run
ls -lh output/raw/prices/
# Expected: daily_quotes_YYYYMMDD_YYYYMMDD.parquet (~2-3GB)

# 4. Check cache file date range matches request
# If requesting 2020-2025 but cache is 2015-2025, it should hit
# If requesting 2020-2026 but cache is 2020-2025, it will miss
```

**Cache file too small (only a few MB)**:
```bash
# Verify cache content
python -c "import polars as pl; df = pl.read_parquet('output/raw/prices/daily_quotes_*.parquet'); print(f'Records: {len(df):,}, Stocks: {df[\"Code\"].n_unique()}, Date range: {df[\"Date\"].min()} to {df[\"Date\"].max()}')"

# Possible causes:
# 1. Very short date range (e.g., 1-2 weeks)
# 2. Market filter applied (only specific stocks)
# 3. Data fetch partially failed
```

**Cache performance not improving**:
```bash
# Check logs for cache hit confirmation
make dataset-gpu START=2020-09-06 END=2025-09-06 2>&1 | grep -E "CACHE (HIT|MISS)|💾"
# Expected on first run: "🌐 CACHE MISS" then "💾 Saved to cache"
# Expected on second run: "📦 CACHE HIT: Daily Quotes (saved ~45s)"

# If cache miss persists:
# - Date range might differ from cached range
# - Cache may be expired (check CACHE_MAX_AGE_DAYS)
# - Cache file may be corrupted (delete and rebuild)
```

**GCS upload fails after cache save**:
```bash
# Check GCS credentials
ls -lh gogooku-b3b34bc07639.json
# Verify GCS enabled in .env
grep GCS_ENABLED .env

# Note: Cache still saves locally even if GCS upload fails
# You can manually upload later with:
python scripts/upload_output_to_gcs.py
```

**Phase 2 optimization not activating**:
```bash
# Check Phase 2 environment variables
grep -E "MIN_CACHE_COVERAGE|ENABLE_MULTI_CACHE" .env

# Expected:
# MIN_CACHE_COVERAGE=0.3
# ENABLE_MULTI_CACHE=1

# Verify multi-cache is trying to activate (look for these log messages):
make dataset-gpu START=2020-09-06 END=2025-09-06 2>&1 | grep -E "Multi-cache|multi-cache"

# Expected logs on partial match:
# "⚠️  Single file coverage too low (25.0% < 30% threshold)"
# "   Trying multi-cache file combination..."
# "✅ Multi-cache improves coverage: 25.0% → 85.0%"

# If multi-cache never activates:
# 1. All requests are perfect matches (good!)
# 2. Coverage always above MIN_CACHE_COVERAGE (adjust threshold lower to test)
# 3. ENABLE_MULTI_CACHE=0 (check .env)
```

**Phase 2 performance debugging**:
```bash
# Enable detailed Phase 2 logging (look for these patterns):

# Coverage threshold check:
grep "coverage too low" /tmp/phase2_*.log
# Expected: "⚠️  Single file coverage too low (X% < 30% threshold)"

# Multi-cache combination:
grep "Multi-cache" /tmp/phase2_*.log
# Expected: "✅ Multi-cache improves coverage: X% → Y%"

# Range merging:
grep "Merging.*ranges" /tmp/phase2_*.log
# Expected: "Merging 3 missing ranges into 2 API calls"

# Cache hit confirmation:
grep "CACHE HIT" /tmp/phase2_*.log
# Expected: "📦 CACHE HIT: Daily Quotes (saved ~45s)"
```

## Performance Benchmarks

### Current Results
- **Dataset**: 10.6M samples, 3,973 stocks, 5 years
- **RankIC@1d**: 0.180 (+20% improvement)
- **Pipeline Speed**: 1.9s for 7 components
- **Memory**: 7GB peak (Polars optimized)
- **GPU**: 5130 samples/sec throughput

### Target Metrics
- **Sharpe Ratio**: 0.849
- **Model Size**: ~5.6M parameters
- **Training**: 75-120 epochs
- **Batch Size**: 2048-4096

### Parallel Processing Performance
- **API Fetching**: 75 concurrent requests (asyncio)
- **Data Pipeline**: 50-stock batches
- **CPU Utilization**: 20 workers (80% of 24 cores)
- **GPU-ETL**: 10-100x faster with RAPIDS/cuDF
- **Expected Dataset Generation**: 30-60 minutes for 10 years

## Key Implementation Patterns

### Hydra Overrides
```bash
# Common patterns
train.optimizer.lr=2e-4
train.trainer.max_epochs=120
model.hidden_size=256
train.batch.train_batch_size=2048
```

### Advanced Graph Training
```bash
# Enable with flag
--adv-graph-train
# Uses: EWM demean, halflife=30, shrinkage=0.1, K=15 neighbors
```

### Walk-Forward Validation
```python
# Always use with embargo
n_splits=5, embargo_days=20, min_train_days=252
```

## Migration Notes

**v2.0.0 Package Structure**:
- Modern imports: `from gogooku3.training import SafeTrainingPipeline`
- Legacy compatible: `from scripts.run_safe_training import SafeTrainingPipeline`
- CLI available: `gogooku3 train --config xxx` (in progress)

**Backward Compatibility**:
- All `scripts/` still work
- Use `gogooku3.compat` for gradual migration
- No breaking changes for existing workflows
- どんなタスクでもちゃんと分析し、熟考し上で正しい判断をしてから実行してください。