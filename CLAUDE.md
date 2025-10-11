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