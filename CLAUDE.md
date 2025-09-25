# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

ATFT-GAT-FAN: Advanced financial ML system for Japanese stock market prediction. Multi-horizon forecasting using Graph Attention Networks with production-grade time-series safety protocols. Target Sharpe ratio: 0.849.

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
```bash
# Build full dataset (required before training)
make dataset-full START=2020-09-06 END=2025-09-06

# Alternative dataset commands
make dataset-full-research START=2020-09-06 END=2025-09-06  # With research features
make fetch-all START=2020-09-06 END=2025-09-06             # Raw data only
python scripts/data/ml_dataset_builder.py                   # Build from existing
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

**`scripts/integrated_ml_training_pipeline.py`** - Main orchestrator:
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
- `features/`: QualityFinancialFeaturesGenerator, 189+ features
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
JQUANTS_AUTH_EMAIL=xxx
JQUANTS_AUTH_PASSWORD=xxx
ALLOW_UNSAFE_DATALOADER=1    # Enable multi-worker
USE_RANKIC=1                  # RankIC optimization
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # GPU memory
```

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
# Build dataset first
make dataset-full START=2020-09-06 END=2025-09-06
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
- Ensure 189 features properly categorized
- VSN/FAN/SAN require correct grouping

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