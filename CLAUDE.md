# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Advanced financial ML system for Japanese stock market prediction using ATFT-GAT-FAN architecture. Modern Python package (v2.0.0) with focus on safety (壊れず), strength (強く), and speed (速く).

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
# Full test suite
pytest tests/ -v                # All tests
pytest tests/unit/ -v           # Unit tests only  
pytest tests/integration/ -v    # Integration tests only
pytest tests/ -k "smoke" -v     # Smoke tests only

# Quick smoke test (basic functionality)
python scripts/smoke_test.py

# Migration compatibility tests
pytest tests/integration/test_migration_smoke.py -v
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
gogooku3 train --config configs/training/production.yaml
gogooku3 data --build-dataset
gogooku3 infer --model-path models/best_model.pth

# Legacy script execution (maintained for compatibility)
python -m gogooku3.compat.script_wrappers train_atft
```

### Code Quality & Development
```bash
# Automated code quality (pre-commit hooks configured)
pre-commit run --all-files
ruff check src/ --fix           # Linting and auto-fixes
ruff format src/                # Code formatting
mypy src/gogooku3              # Type checking
bandit -r src/                 # Security scanning

# Manual install of pre-commit hooks
pre-commit install
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

**Model Architecture** (`models/`)
- **ATFT-GAT-FAN**: Multi-horizon prediction (1d, 5d, 10d, 20d horizons)
- **Graph Attention**: Stock correlation networks with dynamic attention
- **Model Scale**: ~5.6M parameters optimized for financial Sharpe ratio

**Training Pipeline** (`training/`)
- **SafeTrainingPipeline**: 7-step integrated validation pipeline
- **Walk-Forward Validation**: Strict temporal splitting with embargo enforcement
- **Cross-Sectional Normalization**: Daily standardization to prevent lookahead bias

### Critical Design Patterns

**Data Safety (壊れず)**
- No BatchNorm in time-series models (prevents leakage)
- Fit normalizers on training data only
- T+1 as-of joins for statements (15:00 cutoff)
- Temporal overlap validation between train/test splits

**Configuration Management**
- Hydra framework for hierarchical configs
- Main configs in `configs/atft/`
- Environment-specific settings via `.env`

**Performance Optimization**
- Polars for 3-5x faster data processing
- GPU memory management with expandable segments
- Batch size auto-adjustment for OOM prevention

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

### Key Architectural Insights

**Financial ML Safety Design**
- **No BatchNorm**: Prevents cross-sample information leakage in time series
- **Walk-Forward Validation**: Temporal splitting with 20-day embargo periods
- **Fit-Transform Separation**: Normalizers trained only on historical data
- **T+1 As-Of Joins**: Financial statements use 15:00 cutoff for realistic availability

**Performance Optimization Stack**
- **Polars Engine**: 3-5x faster data processing vs pandas
- **Lazy Loading**: Memory-mapped files with columnar projection
- **GPU Memory Management**: Expandable segments for CUDA OOM prevention
- **Mixed Precision**: bf16 training for memory efficiency

**Production Safety Features**  
- **Data Quality Gates**: Great Expectations integration
- **Automated Testing**: Unit, integration, E2E, and smoke tests
- **Security Scanning**: Bandit, pre-commit hooks, dependency audits
- **Monitoring**: Prometheus metrics, health checks, log rotation