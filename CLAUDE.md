# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Gogooku3-standalone is an enhanced financial ML system for Japanese stock market prediction, featuring ATFT-GAT-FAN (Adaptive Temporal Fusion Transformer + Graph Attention Networks) architecture with robust safety features.

## Key Commands

### Development & Testing
```bash
# Setup environment
make setup                                      # Install dependencies
make docker-up                                  # Start services (MinIO, ClickHouse, Redis)
make docker-down                                # Stop services

# Testing
make test                                       # Run all tests
pytest tests/unit/ -v                          # Unit tests only
pytest tests/integration/test_migration_smoke.py -v  # Migration validation
pytest tests/integration/test_safety_components.py   # Safety components test

# Code quality
pre-commit run --all-files                     # Run pre-commit hooks
ruff check src/ --fix                          # Linting
mypy src/gogooku3                              # Type checking
```

### ML Training & Pipeline
```bash
# Modern package CLI (recommended)
pip install -e .                               # Install package
gogooku3 train --config configs/training/production.yaml
gogooku3 data --build-dataset
gogooku3 infer --model-path models/best_model.pth

# Key training scripts
python scripts/run_safe_training.py --verbose --n-splits 2   # 7-step safe pipeline
python scripts/integrated_ml_training_pipeline.py            # Complete ATFT training
python scripts/train_atft.py                                 # Hydra-configured training
python scripts/hyperparameter_tuning.py --trials 20          # Hyperparameter optimization

# Validation & monitoring
python scripts/smoke_test.py                                 # Basic functionality test
python scripts/validate_improvements.py --detailed           # Performance validation
python scripts/monitoring_dashboard.py --start-tensorboard   # Start monitoring
```

### Data Processing
```bash
# Build ML dataset from batch output
python scripts/data/ml_dataset_builder.py

# Direct API data collection
python scripts/data/direct_api_dataset_builder.py

# Data validation
python scripts/validate_data.py
```

## Architecture Overview

The system uses a dual structure: modern `src/gogooku3` package (v2.0.0) with backward compatibility for legacy scripts.

### Core Components

**ATFT-GAT-FAN Model** (`src/models/architectures/atft_gat_fan.py`)
- Multi-horizon prediction (1d, 5d, 10d, 20d)
- Graph attention for stock relationships
- Frequency adaptive normalization
- ~5.6M parameters, target Sharpe 0.849

**SafeTrainingPipeline** (`src/gogooku3/training/safe_training_pipeline.py`)
- 7-step integrated pipeline with comprehensive validation
- Walk-Forward validation with 20-day embargo
- Cross-sectional normalization
- Memory-efficient Polars processing

**Data Safety Components** (`src/data/safety/`)
- `CrossSectionalNormalizerV2`: Daily Z-score normalization
- `WalkForwardSplitterV2`: Temporal separation with embargo
- `ProductionDatasetV3`: Lazy loading with column projection

### Configuration Structure

Training configurations use Hydra framework:
- Main config: `configs/atft/config.yaml`
- Production training: `configs/atft/train/production.yaml`
- Safe data config: `configs/atft/data/jpx_safe.yaml`

Key parameters to adjust:
- `batch_size`: 256-2048 (GPU memory dependent)
- `learning_rate`: 1e-4 to 1e-3
- `embargo_days`: 20 (match max prediction horizon)
- `memory_limit_gb`: 4-8 (system dependent)

## Critical Constraints

### Resource Requirements
- **Memory**: 8-16GB for training, up to 200GB for batch processing
- **GPU**: A100/V100 recommended for production training
- **Storage**: 100GB+ for data and models

### Data Safety Rules
- Always use Walk-Forward validation with 20-day embargo
- Never use BatchNorm in time-series models (causes data leakage)
- Fit normalizers on training data only, then transform test data
- Validate no temporal overlap between train/test splits

### Environment Variables
Required in `.env`:
```bash
JQUANTS_AUTH_EMAIL=<email>
JQUANTS_AUTH_PASSWORD=<password>
WANDB_API_KEY=<optional-for-monitoring>
```

## Development Workflow

1. **Package development**: Use `src/gogooku3/` structure with proper imports
2. **Legacy compatibility**: Scripts in `scripts/` maintain backward compatibility
3. **Testing**: Run unit tests before integration tests
4. **Validation**: Always run `scripts/smoke_test.py` after changes
5. **Memory monitoring**: Use `--memory-limit` flag to prevent OOM

## Common Issues & Solutions

**CUDA OOM errors**:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/train_atft.py --batch-size 256  # Reduce batch size
```

**Import errors**:
```bash
export PYTHONPATH="/home/ubuntu/gogooku3-standalone:$PYTHONPATH"
pip install -e .  # Reinstall package
```

**Data loading issues**:
```bash
ls -la data/raw/large_scale/  # Check data files exist
python -c "import polars as pl; print(pl.scan_parquet('data/raw/large_scale/*.parquet').collect().shape)"
```

## File Organization

- `src/gogooku3/`: Modern package structure (use for new development)
- `scripts/`: Training and utility scripts (compatibility maintained)  
- `configs/`: Hydra configuration files
- `data/raw/large_scale/`: Input data location
- `output/`: Training outputs and models
- `tests/`: Test suite with unit/integration/smoke tests