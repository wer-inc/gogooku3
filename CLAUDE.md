# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

ATFT-GAT-FAN: Advanced financial ML system for Japanese stock market prediction with Graph Attention Networks and multi-horizon forecasting. Modern Python package (v2.0.0) implementing production-grade time-series ML with strict data safety protocols.

## Essential Commands

### Quick Start
```bash
# Install package in development mode
pip install -e .

# Set up environment and credentials
cp .env.example .env
# Edit .env with your JQuants API credentials:
# JQUANTS_AUTH_EMAIL, JQUANTS_AUTH_PASSWORD

# Verify installation
python -c "import gogooku3; print(f'✅ Gogooku3 v{gogooku3.__version__}')"
```

### Testing & Validation
```bash
# Single test execution
pytest tests/unit/test_specific.py::test_function_name -v

# Test suite by category
pytest tests/ -v                # All tests
pytest tests/unit/ -v           # Unit tests only
pytest tests/integration/ -v    # Integration tests only
pytest tests/ -k "smoke" -v     # Smoke tests only
pytest -m "not slow"            # Skip slow tests

# Quick validation
python scripts/smoke_test.py    # 1-epoch basic functionality test
python scripts/validate_improvements.py --detailed  # Performance validation

# Feature testing
python scripts/test_phase1_features.py  # J-Quants Phase 1 features
python scripts/test_phase2_features.py  # J-Quants Phase 2 features
```

### Data Pipeline
```bash
# Recommended: Full dataset with JQuants API
python scripts/pipelines/run_full_dataset.py --jquants --start-date 2020-09-06 --end-date 2025-09-06

# Alternative: Make command
make dataset-full START=2020-09-06 END=2025-09-06

# Raw data fetching only
make fetch-all START=2020-09-06 END=2025-09-06

# Build ML dataset from existing data
python scripts/data/ml_dataset_builder.py
```

### Model Training
```bash
# 🚀 RECOMMENDED: Safe training pipeline (7-step validation)
python scripts/run_safe_training.py --verbose --n-splits 2 --memory-limit 6

# Complete ATFT-GAT-FAN training with integrated pipeline
python scripts/integrated_ml_training_pipeline.py

# Hydra-configured training with config management
python scripts/train_atft.py --config-path configs/atft --config-name config

# Quick start training script (shows available commands)
python start_training.py
```

### Modern CLI Interface (New v2.0.0)
```bash
# Use the new CLI (framework ready, implementation in progress)
gogooku3 --version
gogooku3 train --config configs/atft/train/production.yaml
gogooku3 data --build-dataset
gogooku3 infer --model-path models/best_model.pth

# Legacy script execution (maintained for compatibility)
python -m gogooku3.compat.script_wrappers train_atft
```

### Code Quality & Linting
```bash
# Quick quality check
ruff check src/ --fix           # Auto-fix linting issues
ruff format src/                # Format code
mypy src/gogooku3              # Type checking

# Full quality gate
pre-commit run --all-files      # Run all hooks
bandit -r src/                 # Security scanning

# Pre-commit setup
pre-commit install              # Install git hooks
pre-commit autoupdate          # Update hook versions
```

### Docker Services
```bash
# Start all services (MinIO, ClickHouse, Redis, Dagster)
make docker-up
# Access points:
# - MinIO Console: http://localhost:9001 (minioadmin/minioadmin123)
# - Dagster UI: http://localhost:3001  
# - Grafana: http://localhost:3000 (admin/gogooku123)
# - Prometheus: http://localhost:9090

make docker-down              # Stop all services
make docker-logs              # View service logs
```

## High-Level Architecture

### Package Structure Overview
The repository has undergone a major migration (v2.0.0) from scripts to a modern Python package structure while maintaining backward compatibility.

### Core Package (src/gogooku3/)
```
src/gogooku3/
├── __init__.py              # Public API exports
├── cli.py                   # Modern command-line interface  
├── data/                    # Data processing layer
│   ├── loaders/            # ProductionDatasetV3, MLDatasetBuilder
│   ├── safety/             # CrossSectionalNormalizerV2, WalkForwardSplitterV2
│   └── samplers/           # DayBatchSampler, temporal sampling
├── features/               # Feature engineering
│   ├── quality_features.py # QualityFinancialFeaturesGenerator
│   └── sector_mappings.py  # Industry sector classification
├── models/                 # ML architecture
│   ├── atft_gat_fan.py    # ATFT-GAT-FAN model architecture
│   └── lightgbm_baseline.py # Financial baseline models  
├── graph/                  # Graph neural networks
│   └── graph_builder.py   # Financial correlation graphs
├── training/               # Training pipeline
│   └── safe_training_pipeline.py # 7-step integrated pipeline
├── utils/                  # Core utilities
│   ├── settings.py         # Pydantic settings management
│   └── deduplication.py    # Safe data deduplication
├── inference/              # Model inference (future)
└── compat/                 # 🆕 Backward compatibility layer
    ├── aliases.py          # Legacy component aliases  
    └── script_wrappers.py  # Script compatibility wrappers
```

### Legacy Scripts Directory
```
scripts/
├── run_safe_training.py              # ✅ 7-step safe training pipeline
├── integrated_ml_training_pipeline.py # ✅ Complete ATFT-GAT-FAN training
├── train_atft.py                     # ✅ Hydra-configured training
├── data/
│   ├── ml_dataset_builder.py        # Enhanced dataset construction
│   └── fetch_jquants_history.py     # JQuants API data fetching
├── pipelines/
│   └── run_full_dataset.py          # Full data pipeline execution
└── maintenance/
    └── cleanup_deprecated.py        # Legacy cleanup utilities
```

### Critical Data Safety Architecture
The system implements strict temporal validation to prevent data leakage in financial ML:

**Data Layer** (`data/`)
- **ProductionDatasetV3**: Polars-based lazy loading with memory efficiency
- **CrossSectionalNormalizerV2**: Daily Z-score normalization with fit-transform separation
- **WalkForwardSplitterV2**: Time-series validation with 20-day embargo periods
- **JQuants Integration**: Real-time Japanese stock data with rate limiting

**Feature Engineering** (`features/`)
- **Technical Indicators**: 50+ indicators via pandas-ta (RSI, MACD, Bollinger Bands, ATR)
- **Market Features**: TOPIX correlation (26 mkt_* features)
- **Investment Flows**: Institutional flow analysis (17 flow_* features)
- **Financial Statements**: T+1 as-of joins with 15:00 cutoff safety (17 stmt_* features)
- **J-Quants Phase 1**: Earnings events (5), short positions (6), listed info (5)
- **J-Quants Phase 2**: Margin trading (9), option sentiment (10), enhanced flows (9)

**Model Architecture** (`models/`)
- **ATFT-GAT-FAN**: Multi-horizon prediction (1d, 5d, 10d, 20d horizons)
- **Graph Attention**: Stock correlation networks with dynamic attention
- **Model Scale**: ~5.6M parameters optimized for financial Sharpe ratio

**Training Pipeline** (`training/`)
- **SafeTrainingPipeline**: 7-step integrated validation pipeline
- **Walk-Forward Validation**: Strict temporal splitting with embargo enforcement
- **Cross-Sectional Normalization**: Daily standardization to prevent lookahead bias

### Critical Implementation Patterns

**Data Safety Architecture**
- Walk-Forward validation with 20-day embargo between train/test
- Cross-sectional normalization (daily Z-score) with fit-transform separation
- No BatchNorm in time-series models (prevents cross-sample leakage)
- T+1 as-of joins for financial statements (15:00 JST cutoff)
- Temporal overlap detection and validation

**J-Quants API Integration**
- Async fetcher with rate limiting (75 concurrent requests max)
- Batch API calls by date range (not per stock)
- Forward-fill weekly data to daily frequency
- Graceful fallback to null features when data unavailable

**Performance Patterns**
- Polars lazy evaluation with columnar projection
- GPU expandable segments: `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- Mixed precision training (bf16) for memory efficiency
- Batch size auto-adjustment on OOM

## Key Configuration Files

- `configs/atft/config.yaml`: Main model configuration
- `configs/atft/train/production.yaml`: Production training settings
- `configs/atft/data/jpx_safe.yaml`: Safe data handling
- `pyproject.toml`: Package dependencies and tools configuration
- `.pre-commit-config.yaml`: Code quality automation

## Common Workflows & Patterns

### Development Workflow
1. **Environment Setup**: `pip install -e .` → `cp .env.example .env` → configure credentials
2. **Code Changes**: Make changes → `pre-commit run --all-files` → run tests
3. **Testing**: `python scripts/smoke_test.py` → `pytest tests/integration/ -v` for validation
4. **Training**: `python scripts/run_safe_training.py --verbose --n-splits 2 --memory-limit 6`

### Data Processing Workflow
1. **Raw Data**: `make fetch-all START=2020-09-06 END=2025-09-06`
2. **Dataset Build**: `python scripts/pipelines/run_full_dataset.py --jquants --start-date 2020-09-06 --end-date 2025-09-06`
3. **Validation**: `pytest tests/integration/test_migration_smoke.py -v`
4. **ML Ready**: Output in `output/` directory as `.parquet` files

### Training Workflow
1. **Quick Test**: `python scripts/smoke_test.py` (1-epoch validation)
2. **Safe Pipeline**: `python scripts/run_safe_training.py` (7-step validation)
3. **Full Training**: `python scripts/integrated_ml_training_pipeline.py`
4. **Monitor**: Check logs in `logs/` directory and TensorBoard/W&B dashboards

## J-Quants Feature Pipeline

### Feature Addition Order (Recommended)
```python
# Optimal order for feature dependencies
df = builder.add_enhanced_listed_features(df, fetcher)  # Provides market_cap
df = builder.add_earnings_features(df, fetcher)         # Phase 1
df = builder.add_short_position_features(df, fetcher)   # Phase 1
df = builder.add_enhanced_margin_features(df, fetcher)  # Phase 2 (uses market_cap)
df = builder.add_option_sentiment_features(df, fetcher) # Phase 2
df = builder.add_enhanced_flow_features(df, fetcher)    # Phase 2
```

### Feature Categories Summary
- **Phase 1 (16 features)**: Basic events and positions
- **Phase 2 (28 features)**: Advanced market microstructure
- **Total**: 44 new J-Quants derived features

## Common Issues & Solutions

### Environment & Installation Issues
**Package Import Errors**
```bash
# Ensure package is installed in development mode
pip install -e .

# Add to Python path if needed
export PYTHONPATH="/home/ubuntu/gogooku3-standalone:$PYTHONPATH"

# Verify installation
python -c "import gogooku3; print(f'✅ Version: {gogooku3.__version__}')"
```

**Missing Dependencies**
```bash
# Reinstall with dev dependencies  
pip install -e ".[dev]"

# Check specific dependencies
pip list | grep -E "(torch|polars|pandas)"
```

### CUDA & GPU Issues
**CUDA Out of Memory (OOM)**
```bash
# Set expandable segments (recommended)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Alternative: Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Reduce batch size in config files
# Edit configs/atft/train/production.yaml: batch_size: 256
```

**GPU Not Detected**
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"

# Check GPU memory
nvidia-smi
```

### Data Issues
**Data Files Not Found**
```bash
# Check output directory
ls -la output/*.parquet

# Validate data shape
python -c "import polars as pl; print(pl.scan_parquet('output/*.parquet').collect().shape)"

# Run data pipeline if missing
make dataset-full START=2020-09-06 END=2025-09-06
```

**JQuants API Issues**
```bash
# Verify credentials in .env file
cat .env | grep JQUANTS

# Test API connection
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
print('Email:', os.getenv('JQUANTS_AUTH_EMAIL'))
print('Password configured:', bool(os.getenv('JQUANTS_AUTH_PASSWORD')))
"
```

### Memory & Performance Issues
**High Memory Usage**
```bash
# Check system memory
free -h

# Monitor during execution
python scripts/run_safe_training.py --memory-limit 4 --verbose

# Use smaller data samples for testing
python scripts/smoke_test.py
```

**Slow Performance**  
```bash
# Enable performance optimizations
export PERF_POLARS_STREAM=1
export PERF_MEMORY_OPTIMIZATION=1

# Check number of CPU cores for parallel processing
nproc
```

## Resource Requirements

- **Memory**: 8-16GB (training), up to 200GB (batch processing)
- **GPU**: A100/V100 recommended for production
- **API Limits**: 75 concurrent JQuants requests
- **Storage**: 100GB+ for data and models

## Performance Benchmarks

- **Dataset**: 10.6M samples, 3,973 stocks, 5 years
- **Pipeline Speed**: 1.9s for 7-component pipeline
- **Memory Usage**: 7GB peak (Polars optimized)
- **Training**: RankIC@1d: 0.180 (+20% improvement)
- **GPU Throughput**: 5130 samples/sec

## Migration Status & Compatibility

### v2.0.0 Migration Complete ✅
The repository has successfully migrated from a script-based architecture to a modern Python package:

**✅ Completed (Production Ready)**
- Modern package structure in `src/gogooku3/`
- CLI interface with `gogooku3` command
- Pydantic settings management
- Pre-commit hooks and code quality automation
- Comprehensive test suite with pytest markers
- Compatibility layer for smooth transition

**🔄 Backward Compatibility Maintained**
- Legacy scripts in `scripts/` directory fully functional
- Compatibility aliases in `gogooku3.compat.aliases`
- Script wrappers for gradual migration
- No breaking changes for existing workflows

**🚀 New Features (v2.0.0)**
```bash
# Modern imports (recommended for new code)
from gogooku3.training import SafeTrainingPipeline
from gogooku3.data.loaders import ProductionDatasetV3
from gogooku3.utils.settings import settings

# Legacy compatibility (deprecated but functional)
from scripts.run_safe_training import SafeTrainingPipeline  # Still works
```

**⚠️ Migration Guidelines for Developers**
1. **New Development**: Use `gogooku3.*` imports and modern CLI
2. **Existing Scripts**: Continue to work, but consider gradual migration
3. **Testing**: Use `pytest tests/integration/test_migration_smoke.py` to validate compatibility
4. **Configuration**: Prefer environment variables and `settings` over hardcoded paths

### ATFT-GAT-FAN Model Architecture

**Core Components**
- **ATFT**: Adaptive Temporal Fusion Transformer for multi-horizon prediction
- **GAT**: Graph Attention Networks for stock correlation modeling
- **FAN**: Frequency Adaptive Normalization for time-series stability
- **Model Scale**: ~5.6M parameters, optimized for Sharpe ratio
- **Horizons**: [1d, 5d, 10d, 20d] simultaneous prediction

**Training Optimizations**
- **Small-init + LayerScale**: Output head stability
- **FreqDropout**: Frequency domain regularization
- **EMA Teacher**: Exponential moving average for stability
- **Huber Loss**: Outlier robustness
- **ParamGroup**: Layer-specific learning rates

### Data Pipeline Architecture

**Three-Stage Processing**
1. **Raw Data Fetching**: J-Quants API → parquet files
2. **Feature Engineering**: Technical + fundamental + J-Quants features
3. **ML Dataset Build**: Cross-sectional normalization + safety validation

**Key Components**
- `JQuantsAsyncFetcher`: Rate-limited async API client
- `MLDatasetBuilder`: Feature orchestration and integration
- `ProductionDatasetV3`: Polars-based lazy loading
- `SafeTrainingPipeline`: 7-step validation pipeline
