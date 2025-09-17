# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

ATFT-GAT-FAN: Advanced financial ML system for Japanese stock market prediction with Graph Attention Networks and multi-horizon forecasting. Modern Python package (v2.0.0) implementing production-grade time-series ML with strict data safety protocols.

## Essential Commands

### Quick Start & Development Setup
```bash
# Install package in development mode
pip install -e .

# Environment setup
cp .env.example .env
# Edit .env with credentials (JQUANTS_AUTH_EMAIL, JQUANTS_AUTH_PASSWORD)

# Verify installation
python -c "import gogooku3; print(f'✅ Gogooku3 v{gogooku3.__version__}')"

# Pre-commit hooks setup
pre-commit install
pre-commit install -t commit-msg
```

### Core Training Commands
```bash
# 🚀 PRIMARY: Integrated ML Training Pipeline
python scripts/integrated_ml_training_pipeline.py  # Complete ATFT-GAT-FAN training

# With options
python scripts/integrated_ml_training_pipeline.py \
  --run-safe-pipeline      # Run SafeTrainingPipeline validation first
  --data-path output/ml_dataset_latest_full.parquet
  --adv-graph-train        # Enable advanced graph training
  train.optimizer.lr=2e-4  # Hydra overrides

# Alternative approaches
python scripts/train_atft.py --config-path configs/atft --config-name config
python scripts/run_safe_training.py --verbose --n-splits 2 --memory-limit 6

# Make targets (convenience wrappers)
make train-integrated       # Full integrated pipeline
make train-integrated-safe  # With SafeTrainingPipeline
make train-atft            # Direct ATFT training
make smoke                  # Quick 1-epoch test
```

### Testing & Validation
```bash
# Test execution patterns
pytest tests/unit/test_specific.py::test_function_name -v  # Single test
pytest tests/ -v                # All tests
pytest tests/unit/ -v           # Unit tests only
pytest tests/integration/ -v    # Integration tests
pytest -m "not slow"            # Skip slow tests
pytest --ignore=tests/exploratory/  # Skip exploratory tests

# Validation scripts
python scripts/smoke_test.py    # 1-epoch basic test
python scripts/validate_improvements.py --detailed  # Performance validation
python scripts/test_phase1_features.py  # J-Quants Phase 1
python scripts/test_phase2_features.py  # J-Quants Phase 2
```

### Data Pipeline
```bash
# Full dataset pipeline (recommended)
make dataset-full START=2020-09-06 END=2025-09-06
# Or directly:
python scripts/pipelines/run_full_dataset.py --jquants --start-date 2020-09-06 --end-date 2025-09-06

# Research configuration with indices
make dataset-full-research START=2020-09-06 END=2025-09-06

# Raw data fetching
make fetch-all START=2020-09-06 END=2025-09-06

# Build ML dataset from existing data
python scripts/data/ml_dataset_builder.py

# Check indices features
make check-indices DATASET=output/ml_dataset_latest_full.parquet
```

### Research & Analysis Workflows
```bash
# Complete research bundle
make research-plus  # Runs baseline + lag audit + report

# Individual research tasks
make research-baseline DATASET=output/ml_dataset_latest_full.parquet
make research-lags PATTERN="output/*.parquet"
make research-report FACTORS=returns_5d,ret_1d_vs_sec HORIZONS=1,5,10,20
make research-folds SPLITS=output/eval_splits_5fold_20d.json

# HPO (Hyperparameter Optimization)
make hpo-setup        # Setup HPO environment
make hpo-run HPO_TRIALS=20 HPO_STUDY=atft_hpo_production
make hpo-status       # Check study status
make hpo-resume       # Resume existing study
```

### Service Management
```bash
# Docker services
make docker-up        # Start all services (MinIO, ClickHouse, Redis, Dagster)
make docker-down      # Stop all services
make docker-logs      # View service logs

# Service access points:
# - MinIO Console: http://localhost:9001 (minioadmin/minioadmin123)
# - Dagster UI: http://localhost:3001
# - Grafana: http://localhost:3000 (admin/gogooku123)
# - Prometheus: http://localhost:9090

# API server
gogooku3-api         # Start FastAPI server
# Or: python scripts/run_api.py
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

# Cleanup & Maintenance
make clean-deprecated APPLY=1   # Remove deprecated scripts
python scripts/audit_unused.py  # Audit unused code
```

### Modern CLI Interface (v2.0.0)
```bash
# New CLI (framework ready, implementation in progress)
gogooku3 --version
gogooku3 train --config configs/atft/train/production.yaml
gogooku3 data --build-dataset
gogooku3 infer --model-path models/best_model.pth

# Legacy compatibility
python -m gogooku3.compat.script_wrappers train_atft
```

## High-Level Architecture

### Training Pipeline Overview
The primary training flow uses `scripts/integrated_ml_training_pipeline.py` which orchestrates:
1. **Data Loading**: From `output/batch/` or specified paths
2. **Optional SafeTrainingPipeline**: 7-step validation (if `--run-safe-pipeline`)
3. **Graph Construction**: Advanced graph training with EWM correlation
4. **ATFT Training**: Via Hydra-configured `train_atft.py`
5. **HPO Integration**: Optional hyperparameter optimization

### Key Integration Points
- **Data Flow**: JQuants API → Raw Data → ML Dataset → Training
- **Model Training**: Hydra configs → ATFT-GAT-FAN → Checkpoints
- **Validation**: SafeTrainingPipeline → Walk-Forward Splits → Metrics

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
│   └── deduplication.py   # Safe data deduplication
└── compat/                 # Backward compatibility layer
    ├── aliases.py          # Legacy component aliases
    └── script_wrappers.py  # Script compatibility wrappers
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

### Model Configurations
- `configs/atft/config.yaml`: Main model configuration
- `configs/atft/train/production.yaml`: Production training settings
- `configs/atft/data/jpx_safe.yaml`: Safe data handling
- `configs/atft/data/jpx_large_scale.yaml`: Large-scale data config

### Pipeline Configurations
- `configs/pipeline/full_dataset.yaml`: Production dataset pipeline
- `configs/pipeline/research_full_indices.yaml`: Research with indices

### Project Configuration
- `pyproject.toml`: Package dependencies and tools
- `.pre-commit-config.yaml`: Code quality automation
- `.env`: Environment variables (create from .env.example)

## Common Workflows & Patterns

### Complete Training Workflow (Recommended)
```bash
# 1. Build dataset
make dataset-full START=2020-09-06 END=2025-09-06

# 2. Run integrated training pipeline
python scripts/integrated_ml_training_pipeline.py \
  --run-safe-pipeline \
  --adv-graph-train

# 3. Monitor training
tensorboard --logdir logs/
```

### Research Workflow
```bash
# 1. Generate dataset with research config
make dataset-full-research START=2020-09-06 END=2025-09-06

# 2. Run research analysis
make research-plus DATASET=output/ml_dataset_latest_full.parquet

# 3. Review reports
cat reports/research_report.md
```

### HPO Workflow
```bash
# 1. Setup HPO
make hpo-setup

# 2. Run optimization
make hpo-run HPO_TRIALS=20

# 3. Check results
make hpo-status

# 4. Apply best params to training
python scripts/integrated_ml_training_pipeline.py \
  --run-hpo \
  --hpo-n-trials 20
```

### Development Best Practices
1. **Environment Setup**: `pip install -e .` → `cp .env.example .env` → configure credentials
2. **Before Commits**: `pre-commit run --all-files` → `pytest tests/` → `make smoke`
3. **Testing Changes**: Use `python scripts/smoke_test.py` for quick validation
4. **Full Validation**: `make train-integrated-safe` with SafeTrainingPipeline

### Pipeline Integration Patterns

#### Integrated ML Training Pipeline
The main entry point `scripts/integrated_ml_training_pipeline.py` supports:
- **SafeTrainingPipeline Integration**: `--run-safe-pipeline` for validation
- **Advanced Graph Training**: `--adv-graph-train` for EWM correlation graphs
- **Hydra Overrides**: Pass `train.*` namespace configs directly
- **HPO Integration**: `--run-hpo` with automatic best params application

#### Data Pipeline Integration
- **JQuants Fetcher**: Async with rate limiting (75 concurrent max)
- **Feature Builder**: Technical + fundamental + J-Quants features
- **Dataset Builder**: Polars-based with memory optimization

## Important Configuration Patterns

### Hydra Configuration Override
When using `integrated_ml_training_pipeline.py`, pass Hydra overrides directly:
```bash
# Examples of common overrides
train.optimizer.lr=2e-4
train.trainer.max_epochs=10
data.batch.batch_size=256
model.gat.num_heads=4
```

### Environment Variables
Key variables from `.env`:
- `JQUANTS_AUTH_EMAIL`, `JQUANTS_AUTH_PASSWORD`: API credentials
- `OUTPUT_BASE`: Default `/home/ubuntu/gogooku3-standalone/output/batch`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: GPU optimization
- `ADV_GRAPH_TRAIN`: Enable advanced graph training

### Graph Training Configuration
Advanced graph training (`--adv-graph-train`) uses:
- EWM demean with halflife=30
- Shrinkage gamma=0.1
- K=15 nearest neighbors
- Edge threshold=0.25

## Common Issues & Solutions

### Training Pipeline Issues
**Integrated Pipeline Fails**
```bash
# Check data availability
ls -la output/batch/*.parquet || ls -la output/*.parquet

# Run with verbose logging
python scripts/integrated_ml_training_pipeline.py --verbose

# Use fallback direct training
python scripts/train_atft.py --config-path configs/atft --config-name config
```

**SafeTrainingPipeline Errors**
```bash
# Run standalone first to debug
python scripts/run_safe_training.py --verbose --n-splits 1 --memory-limit 4

# Then integrate
python scripts/integrated_ml_training_pipeline.py --run-safe-pipeline
```

### Installation & Environment Issues
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

## Resource Requirements & Optimization

### Hardware Requirements
- **Memory**: 8-16GB (training), up to 200GB (batch processing)
- **GPU**: A100/V100 recommended, 8GB+ VRAM minimum
- **CPU**: 24+ cores recommended for data processing
- **Storage**: 100GB+ for data and models

### Performance Optimization Tips
```bash
# GPU memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Polars optimization
export PERF_POLARS_STREAM=1
export PERF_MEMORY_OPTIMIZATION=1

# Reduce batch size if OOM
# Edit configs/atft/train/production.yaml
batch_size: 256  # Reduce from default 512/1024
```

## Performance Benchmarks & Targets

### Current Performance
- **Dataset**: 10.6M samples, 3,973 stocks, 5 years
- **Pipeline Speed**: 1.9s for 7-component pipeline
- **Memory Usage**: 7GB peak (Polars optimized)
- **Training**: RankIC@1d: 0.180 (+20% improvement)
- **GPU Throughput**: 5130 samples/sec

### ATFT-GAT-FAN Target Metrics
- **Expected Sharpe**: 0.849
- **Model Parameters**: 5.6M
- **Batch Size**: 2048 (bf16 mixed precision)
- **Max Epochs**: 75
- **Learning Rate**: 5e-5

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

## Debugging Commands

### Quick Diagnostics
```bash
# Check system status
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
free -h  # Memory status
df -h    # Disk space

# Validate data
python -c "import polars as pl; print(pl.scan_parquet('output/*.parquet').collect().shape)"

# Test components
python scripts/smoke_test.py --max-epochs 1
python scripts/validate_improvements.py --detailed
```

### Log Analysis
```bash
# Check recent errors
tail -n 100 logs/ml_training.log | grep ERROR

# Monitor training progress
tail -f logs/$(ls -t logs/ | head -1)/train.log

# TensorBoard monitoring
tensorboard --logdir logs/
```

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.