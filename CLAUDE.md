# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Gogooku3-standalone is an advanced financial ML system for Japanese stock market prediction featuring ATFT-GAT-FAN (Adaptive Temporal Fusion Transformer + Graph Attention Networks) architecture with robust safety features and modern Python package structure (v2.0.0).

## Key Commands

### Installation & Setup
```bash
# Install package
pip install -e .

# Or use requirements
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env with JQuants credentials: JQUANTS_AUTH_EMAIL, JQUANTS_AUTH_PASSWORD
```

### Docker Services
```bash
make docker-up    # Start MinIO, ClickHouse, Redis, Dagster
make docker-down  # Stop all services
make docker-logs  # View service logs
```

### Data Pipeline Commands
```bash
# Full 5-year dataset with all features (recommended)
python scripts/pipelines/run_full_dataset.py \
  --jquants \
  --start-date 2020-09-06 \
  --end-date 2025-09-06

# Or using Make
make dataset-full START=2020-09-06 END=2025-09-06

# Basic pipeline (without full enrichment)
python scripts/pipelines/run_pipeline_v4_optimized.py --jquants

# Fetch specific components
make fetch-all START=2020-09-06 END=2025-09-06
```

### Model Training
```bash
# Complete ATFT-GAT-FAN training pipeline
python scripts/integrated_ml_training_pipeline.py

# Hydra-configured training
python scripts/train_atft.py --config-path configs/atft --config-name config

# Safe training pipeline (7-step validation)
python scripts/run_safe_training.py --verbose --n-splits 2 --memory-limit 6

# Hyperparameter tuning
python scripts/hyperparameter_tuning.py --trials 20
```

### Testing & Validation
```bash
# All tests
make test
pytest tests/ -v

# Specific test suites
pytest tests/integration/test_migration_smoke.py -v  # Migration validation
pytest tests/integration/test_safety_components.py   # Safety components
pytest tests/unit/ -v                                # Unit tests only

# Quick functionality test
python scripts/smoke_test.py

# Performance validation
python scripts/validate_improvements.py --detailed

# Production validation
python scripts/production_validation.py --scenario medium_scale
```

### Code Quality
```bash
# Pre-commit hooks (auto-configured)
pre-commit run --all-files

# Manual checks
ruff check src/ --fix
mypy src/gogooku3
```

### Monitoring
```bash
# Start monitoring dashboard
python scripts/monitoring_dashboard.py --start-tensorboard

# Setup monitoring system
python scripts/setup_monitoring.py --install-deps --create-config --create-alerts
```

## Architecture & Structure

### Package Architecture (v2.0.0)
```
src/gogooku3/             # Modern Python package (main development)
├── cli.py               # CLI interface (gogooku3 command)
├── utils/               # Settings (Pydantic), deduplication utilities
├── data/                # Data loaders, scalers, safety components
│   ├── loaders/        # ProductionDatasetV3, MLDatasetBuilder
│   └── safety/         # CrossSectionalNormalizerV2, WalkForwardSplitterV2
├── features/            # Feature engineering
│   ├── market_features.py      # TOPIX market features
│   ├── flow_joiner.py         # Investment flow features
│   └── safe_joiner.py         # Statement features with T+1 safety
├── models/              # Model architectures
│   ├── architectures/   # ATFT-GAT-FAN implementation
│   └── baseline/        # LightGBM baseline
├── graph/               # Financial graph construction
├── training/            # SafeTrainingPipeline (7-step integrated)
└── compat/              # Backward compatibility layer
```

### Key Components & Design Patterns

**Data Safety (壊れず)**
- Walk-Forward validation with 20-day embargo
- Cross-sectional normalization (daily Z-score)
- T+1 as-of joins for statements (15:00 cutoff)
- No BatchNorm in time-series models (prevents data leakage)

**Feature Engineering Pipeline**
1. `MLDatasetBuilder.create_technical_features()` - Core price features
2. `MLDatasetBuilder.add_pandas_ta_features()` - Technical indicators
3. `MLDatasetBuilder.add_topix_features()` - Market features (26 mkt_* features)
4. `MLDatasetBuilder.add_flow_features()` - Investment flow (17 flow_* features)
5. `MLDatasetBuilder.add_statements_features()` - Financial statements (17 stmt_* features)

**ATFT-GAT-FAN Model**
- Multi-horizon prediction (1d, 5d, 10d, 20d)
- Graph attention for stock relationships
- ~5.6M parameters, target Sharpe 0.849
- Frequency Adaptive Normalization
- EMA Teacher for stability

### Configuration (Hydra Framework)

Main configs in `configs/atft/`:
- `config.yaml` - Main configuration
- `train/production.yaml` - Production training settings
- `data/jpx_safe.yaml` - Safe data configuration

Key parameters:
- `batch_size`: 256-2048 (GPU memory dependent)
- `embargo_days`: 20 (match max prediction horizon)
- `memory_limit_gb`: 4-8 (system dependent)
- `max_concurrent_fetch`: 75 (JQuants API limit)

## Data Flow & Pipelines

### Complete Data Pipeline
```
1. JQuants API → Raw data fetch (prices, statements, trades_spec, TOPIX)
2. run_pipeline_v4_optimized.py → Base features (157 columns)
3. run_full_dataset.py → Full enrichment (239+ columns)
4. Output: ml_dataset_latest_full.parquet
```

### Feature Coverage (DATASET.md compliance)
- **Current**: 97.2% (140/142 documented features)
- **Missing**: shares_outstanding, turnover_rate (external data required)
- **Extra**: 100+ additional features (Sharpe ratios, momentum crosses, etc.)

## Critical Constraints & Safety

### Resource Requirements
- **Memory**: 8-16GB for training, up to 200GB for batch processing
- **GPU**: A100/V100 recommended for production
- **Storage**: 100GB+ for data and models
- **API Rate Limits**: 75 concurrent requests (JQuants)

### Data Safety Rules
- Always use Walk-Forward validation with 20-day embargo
- Never use BatchNorm in time-series models
- Fit normalizers on training data only
- Validate no temporal overlap between train/test splits

### Common Issues & Solutions

**CUDA OOM**
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Reduce batch size in training config
```

**Import Errors**
```bash
export PYTHONPATH="/home/ubuntu/gogooku3-standalone:$PYTHONPATH"
pip install -e .  # Reinstall package
```

**Data Loading Issues**
```bash
# Check data exists
ls -la output/*.parquet
# Verify with Polars
python -c "import polars as pl; print(pl.scan_parquet('output/*.parquet').collect().shape)"
```

## Performance Benchmarks

### Latest Results (Production Data)
- **Dataset**: 10.6M samples, 3,973 stocks, 5 years
- **Pipeline Speed**: 1.9s for 7-component pipeline
- **Memory Usage**: 7GB peak (optimized with Polars)
- **Training**: RankIC@1d: 0.180 (+20% improvement)
- **GPU Throughput**: 5130 samples/sec

### Model Performance
- **Target Sharpe**: 0.849
- **Parameters**: 5.6M
- **Training Time**: 9.8s/epoch
- **Inference**: <100ms per batch

## Development Workflow

1. **Feature Development**: Add to `MLDatasetBuilder` in `scripts/data/ml_dataset_builder.py`
2. **Model Changes**: Update `src/models/architectures/atft_gat_fan.py`
3. **Testing**: Run `pytest tests/unit/` then `tests/integration/`
4. **Validation**: `python scripts/smoke_test.py`
5. **Quality Check**: `pre-commit run --all-files`
6. **Full Pipeline**: `make dataset-full` to regenerate dataset

## Migration Notes

The repository underwent major migration from scripts to modern Python package (v2.0.0):
- Legacy scripts in `scripts/` maintain backward compatibility
- New development should use `gogooku3.*` imports
- Compatibility layer in `gogooku3.compat` for gradual migration
- See `MIGRATION.md` for complete migration guide