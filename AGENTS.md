# Gogooku3 Standalone - Production Financial ML System

## 🎯 System Overview

**Gogooku3** is a production-grade financial machine learning system for Japanese stock market prediction, featuring state-of-the-art ATFT-GAT-FAN (Adaptive Temporal Fusion Transformer + Graph Attention Networks) architecture with comprehensive safety mechanisms.

### Key Characteristics
- **Design Philosophy**: 壊れず（Unbreakable）・強く（Strong）・速く（Fast）
- **Architecture**: ATFT-GAT-FAN with 5.6M parameters, target Sharpe ratio 0.849
- **Scale**: 632+ stocks, 606K+ samples, 713+ technical indicators
- **Performance**: 1.9s pipeline execution, 7GB memory usage (optimized from 17GB)
- **Safety**: Walk-Forward validation with 20-day embargo, cross-sectional normalization
- **Version**: v2.0.0 - Modern Python package with backward compatibility

## Project Structure & Modules

### Modern Package Architecture (v2.0.0)
```
src/gogooku3/              # Core package - Modern Python structure
├── __init__.py            # Public API exports
├── cli.py                 # Command-line interface
├── utils/                 # Utilities
│   ├── settings.py        # Pydantic settings management
│   ├── calendar_utils.py  # Trading calendar utilities
│   └── deduplication.py   # Safe deduplication utility
├── data/                  # Data processing
│   ├── loaders/          # ProductionDatasetV3, MLDatasetBuilder
│   └── scalers/          # CrossSectionalNormalizerV2, WalkForwardSplitterV2
├── features/             # Feature engineering
│   ├── ta_core.py        # 713+ technical indicators
│   ├── financial_features.py  # Financial feature generation
│   ├── cross_features.py      # Cross-sectional features
│   └── alias_registry.py      # Feature name management
├── models/               # ML models
│   ├── atft_gat_fan.py   # ATFT-GAT-FAN architecture
│   └── lightgbm_baseline.py  # LightGBM baseline
├── graph/                # Graph processing
│   └── financial_graph_builder.py  # Correlation graphs
├── training/             # Training pipelines
│   └── safe_training_pipeline.py   # 7-step integrated pipeline
└── compat/               # Backward compatibility
    ├── aliases.py        # Legacy component aliases
    └── script_wrappers.py # Script compatibility

scripts/                  # Production scripts
├── run_safe_training.py  # Complete 7-step safe pipeline
├── integrated_ml_training_pipeline.py  # ATFT training
├── train_atft.py        # Hydra-configured training
├── pipelines/           # Data pipelines
│   ├── run_pipeline_v4_optimized.py  # J-Quants optimized
│   └── run_pipeline_v3.py            # Legacy pipeline
└── data/                # Data processing scripts
    └── ml_dataset_builder.py  # Dataset construction

configs/                 # Configuration files
├── atft/               # ATFT model configs
│   ├── config.yaml     # Main config
│   ├── data/          # Data configs
│   │   └── jpx_safe.yaml  # Safe data configuration
│   └── train/         # Training configs
│       ├── production.yaml  # Production settings
│       └── walk_forward.yaml # Walk-Forward validation
├── model/              # Model configurations
├── training/           # Training configurations
└── hardware/           # Hardware-specific configs

- Data & artifacts: `data/`, `output/`, `_logs/` (keep logs here; don't commit)
- Orchestration: `dagster_repo/`, `docker-compose.yml`
- Documentation: `MIGRATION.md`, `CLAUDE.md`, `TODO.md`

## 🚀 Core ML Components

### ATFT-GAT-FAN Model Architecture
- **Architecture**: Adaptive Temporal Fusion Transformer + Graph Attention Networks + Frequency Adaptive Normalization
- **Parameters**: ~5.6M trainable parameters
- **Multi-horizon**: Simultaneous prediction for 1d, 5d, 10d, 20d horizons
- **Input**: Sequences of L=60 days with 145+ features
- **Graph Integration**: 50 nodes, 266+ correlation edges
- **Performance**: Target Sharpe 0.849, IC > 0.05

### SafeTrainingPipeline (7-step Integration)
1. **Data Loading**: ProductionDatasetV3 with lazy loading
2. **Feature Engineering**: QualityFinancialFeaturesGenerator (+6 quality features)
3. **Cross-sectional Normalization**: Daily Z-score normalization
4. **Walk-Forward Splitting**: 20-day embargo, no data leakage
5. **LightGBM Baseline**: Multi-horizon baseline model
6. **Graph Construction**: Financial correlation graphs
7. **Performance Reporting**: Comprehensive metrics and validation

### Data Safety Mechanisms
- **Walk-Forward Validation**: Strict temporal separation with configurable embargo
- **Cross-sectional Normalization**: Fit on train only, transform both train/test
- **No BatchNorm**: Prevents cross-sample information leakage
- **Overlap Detection**: Automatic validation of train/test separation
- **Memory Management**: Configurable limits, lazy loading, column projection

## Build, Test, and Development

### Environment Setup
```bash
# Package installation (recommended)
pip install -e .                    # Development mode
pip install -e ".[dev]"             # With dev dependencies

# Alternative setup
make setup                          # Creates venv + installs requirements
pip install -r requirements.txt     # Direct requirements installation
```

### Running the System

#### Modern CLI (Recommended)
```bash
# After package installation
gogooku3 train --config configs/training/production.yaml
gogooku3 data --build-dataset
gogooku3 infer --model-path models/best_model.pth
gogooku3 --version

# Via Python module
python -m gogooku3.cli train
python -m gogooku3.cli --help
```

#### Key Training Scripts
```bash
# Complete 7-step safe pipeline (RECOMMENDED)
python scripts/run_safe_training.py --verbose --n-splits 2 --memory-limit 6

# ATFT-GAT-FAN training
python scripts/integrated_ml_training_pipeline.py  # Complete pipeline
python scripts/train_atft.py                      # Hydra-configured

# Hyperparameter tuning
python scripts/hyperparameter_tuning.py --trials 20 --study-name financial_ml

# Data pipeline
python scripts/pipelines/run_pipeline_v4_optimized.py \
    --jquants --start-date 2020-09-02 --end-date 2025-09-02
```

### Services & Infrastructure
```bash
# Docker services
make docker-up      # Start MinIO, ClickHouse, Redis, Dagster
make docker-down    # Stop all services
make docker-logs    # View service logs
docker stats        # Monitor resource usage

# Database operations
make db-migrate     # Run migrations
make db-seed        # Seed database
make db-shell       # PostgreSQL shell access
```

### Testing & Validation
```bash
# Test suites
pytest                                              # Run all tests
pytest -m "not slow"                               # Quick tests only
pytest tests/unit/ -v                              # Unit tests
pytest tests/integration/test_migration_smoke.py   # Migration validation
pytest tests/integration/test_safety_components.py # Safety validation

# Smoke tests
python scripts/smoke_test.py                       # Basic functionality
python scripts/validate_improvements.py --detailed # Performance validation

# Coverage
pytest --cov=src/gogooku3 --cov-report=term-missing
```

### Code Quality
```bash
# Pre-commit hooks
pre-commit install                  # Setup hooks
pre-commit run --all-files         # Run all checks

# Individual tools
ruff check src/ --fix              # Linting with fixes
black .                            # Code formatting
mypy src/gogooku3                  # Type checking
bandit -r src/                     # Security analysis
```

## Coding Style & Naming
- Python 3.10+, 4-space indent, max line length 88.
- Format/lint: `black .`, `ruff .`, imports via `isort` (black profile).
- Types: add annotations; `mypy` runs in strict-ish mode for `src/gogooku3`.
- Naming: modules `snake_case.py`, classes `CamelCase`, functions/vars `snake_case`. Tests mirror module names: `test_<module>.py`.

## Testing Guidelines
- Framework: `pytest` with markers: `unit`, `integration`, `slow`, `smoke`.
- Quick run: `pytest -m 'not slow'`.
- Targeted: `pytest tests/unit/test_feature_x.py::TestClass::test_case`.
- Keep tests deterministic; use fixtures from `tests/fixtures/`. Avoid network/API in unit tests; gate external calls with `requires_api`.

## Commit & Pull Requests
- Commits: imperative, concise subject, include scope when helpful. English or Japanese acceptable. Examples: `feat(training): add EMA teacher`, `fix(graph): handle isolated nodes`.
- Link issues in body (`Closes #123`). Describe what/why, note risks and rollbacks.
- PRs: clear description, checklist: updated tests, docs/config changes called out, screenshots for UI/metrics where relevant (e.g., Dagster UI, Grafana), CI green.

## Security & Configuration
- Secrets: never commit keys or `.env`. Use `.env.example` as reference.
- Local data: keep logs in `_logs/` and large outputs in `output/`; Git hooks block stray logs.
- Containers: prefer `make docker-up` for local infra; keep changes in `docker/` and `docker-compose*.yml`.

## CI & Release
- Workflows: see `.github/workflows/tests.yml`, `hygiene-ci.yml`, `security.yml`, `release.yml`.
- Commits: Conventional Commits style. Automated release via `semantic-release` (`.releaserc.json`).
- Example: `feat(training): add EMA teacher`.

## Secrets
- Required: `JQUANTS_AUTH_EMAIL`, `JQUANTS_AUTH_PASSWORD` (use `.env.example` as a template).
- Do not commit `.env`; load via dotenv or environment in CI.

## Caching & Data
- Trading calendar cache: `cache/trading_calendar/` (24h TTL). Remove specific files to force refetch (e.g., `calendar_<from>_<to>.json`).
- Artifacts: datasets in `output/`; keep logs under `_logs/` (pre-commit blocks logs elsewhere).

## 📊 Performance & Benchmarks

### System Performance (Production Validated)
- **Dataset Scale**: 606,127 samples × 145 features, 644 stocks, 2021-2025
- **Pipeline Speed**: 1.9s total execution (100x improvement)
- **Memory Usage**: 7.0GB peak (from 17GB baseline)
- **Data Loading**: 163.6MB in 0.1s with lazy loading
- **Processing**: 0.2s normalization, 0.1s graph construction

### ML Model Performance
- **ATFT-GAT-FAN**: Target Sharpe 0.849, 5.6M parameters
- **Multi-horizon IC**: 1d: 0.05+, 5d: 0.04+, 10d: 0.03+
- **Training Time**: ~2 hours on A100 for full dataset
- **Inference**: <100ms per batch (256 samples)

### Component Execution Times
```
Step 1: Data Loading        → 0.1s (ProductionDatasetV3)
Step 2: Feature Engineering → 0.2s (QualityFinancialFeaturesGenerator)
Step 3: Normalization       → 0.2s (CrossSectionalNormalizerV2)
Step 4: Walk-Forward Split  → 0.2s (WalkForwardSplitterV2)
Step 5: GBM Baseline       → 0.6s (LightGBMFinancialBaseline)
Step 6: Graph Construction → 0.1s (FinancialGraphBuilder)
Step 7: Reporting          → 0.0s (Performance metrics)
---
TOTAL PIPELINE TIME        → 1.9s ✅
```

## 🛡️ Financial ML Safety Rules

### Critical DO NOTs (Automatically Prevented)
```python
# ❌ Future information leakage in normalization
scaler.fit(full_data)  # Prevented by CrossSectionalNormalizerV2

# ❌ BatchNorm in time-series models
nn.BatchNorm1d()  # Disabled in all configs

# ❌ No embargo between train/test
train_end = "2024-06-30"; test_start = "2024-07-01"  # WalkForwardSplitterV2 enforces embargo

# ❌ Overlapping train/test periods
# Automatic overlap detection with warnings
```

### Required Safety Practices
- **Walk-Forward Validation**: Always use with 20-day embargo
- **Cross-sectional Normalization**: Daily Z-score, separate fit/transform
- **Temporal Separation**: No data leakage between train/test
- **Memory Limits**: Configure via `--memory-limit` flag
- **Data Validation**: Automatic overlap and leakage detection

## 🔧 Configuration Management

### Key Configuration Files
```yaml
# configs/atft/data/jpx_safe.yaml
time_series:
  sequence_length: 60
  prediction_horizons: [1, 5, 10, 20]
  drop_historical_columns: true

split:
  method: walk_forward
  n_splits: 5
  embargo_days: 20
  min_train_days: 252

# configs/atft/train/production.yaml
model:
  d_model: 256
  n_heads: 8
  n_layers: 6
  dropout: 0.1

training:
  batch_size: 256
  learning_rate: 1e-4
  max_epochs: 100
  early_stopping_patience: 10
```

### Environment Variables (.env)
```bash
# Required for data access
JQUANTS_AUTH_EMAIL=your_email@example.com
JQUANTS_AUTH_PASSWORD=your_password

# Optional monitoring
WANDB_API_KEY=your_wandb_key
TENSORBOARD_LOG_DIR=./runs

# Resource limits
MAX_MEMORY_GB=8
MAX_WORKERS=4
```

## 🚀 Quick Commands

### Production Training Pipeline
```bash
# Complete safe training (RECOMMENDED)
python scripts/run_safe_training.py \
    --verbose \
    --n-splits 2 \
    --memory-limit 6 \
    --experiment-name production_v1

# Quick validation (1 split, 4GB limit)
python scripts/run_safe_training.py --n-splits 1 --memory-limit 4
```

### Data Pipeline Commands
```bash
# Optimized J-Quants pipeline
python scripts/pipelines/run_pipeline_v4_optimized.py \
    --jquants \
    --start-date 2020-09-02 \
    --end-date 2025-09-02 \
    --output-dir data/raw/large_scale

# Build ML dataset from batch output
python scripts/data/ml_dataset_builder.py \
    --input-dir output/batch \
    --output-dir data/processed
```

### Model Training Commands
```bash
# ATFT-GAT-FAN with Hydra config
python scripts/train_atft.py \
    --config-path configs/atft \
    --config-name config \
    data=jpx_safe \
    train=walk_forward

# Hyperparameter optimization
python scripts/hyperparameter_tuning.py \
    --trials 50 \
    --study-name atft_optimization \
    --storage sqlite:///optuna.db
```

### Monitoring & Validation
```bash
# Start TensorBoard
tensorboard --logdir runs/

# Monitor with Dagster UI
dagster dev -f dagster_repo/repository.py

# Validate improvements
python scripts/validate_improvements.py --detailed --save-report

# Check data quality
python scripts/validate_data.py --check-duplicates --check-nulls
```

### Development Workflow
```bash
# Setup and install
make setup && pip install -e ".[dev]"

# Run pre-commit checks
pre-commit run --all-files

# Quick smoke test
python scripts/smoke_test.py

# Full test suite
pytest && python scripts/validate_improvements.py
```

## 🐛 Troubleshooting

### Common Issues & Solutions

#### CUDA Out of Memory
```bash
# Reduce batch size
python scripts/train_atft.py train.batch_size=128

# Enable gradient accumulation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

#### Import Errors
```bash
# Fix Python path
export PYTHONPATH="/home/ubuntu/gogooku3-standalone:$PYTHONPATH"

# Reinstall package
pip install -e . --force-reinstall
```

#### Data Loading Issues
```bash
# Check data files
ls -la data/raw/large_scale/*.parquet

# Validate data integrity
python -c "import polars as pl; print(pl.scan_parquet('data/raw/large_scale/*.parquet').collect().shape)"
```

#### Memory Issues
```bash
# Monitor memory usage
python scripts/run_safe_training.py --memory-limit 4 --verbose

# Clear cache
rm -rf cache/trading_calendar/*
```

## 📈 Migration from Legacy Scripts

### Using the Compatibility Layer
```python
# Legacy imports still work with warnings
from src.data.safety.cross_sectional import CrossSectionalNormalizer  # Deprecated

# Modern equivalent
from gogooku3.data.scalers import CrossSectionalNormalizerV2  # Recommended
```

### Migration Status Check
```bash
# Check compatibility
python -c "
from gogooku3.compat import check_compatibility, print_migration_guide
status = check_compatibility()
print(f'Components available: {sum(status.values())}/{len(status)}')
print_migration_guide()
"

# Run migration smoke tests
pytest tests/integration/test_migration_smoke.py -v
```

## 🎯 Production Deployment Checklist

- [ ] Environment variables configured (.env from .env.example)
- [ ] Data pipeline validated with recent data
- [ ] Walk-Forward validation with 20-day embargo
- [ ] Memory limits configured for production hardware
- [ ] Pre-commit hooks installed and passing
- [ ] All tests passing (unit, integration, smoke)
- [ ] Performance benchmarks met (Speed < 10s, Memory < 8GB)
- [ ] Model metrics validated (IC > 0.05, Sharpe > 0.5)
- [ ] Monitoring configured (TensorBoard/WandB)
- [ ] Backup and recovery procedures tested

## 📚 Additional Resources

- **Migration Guide**: `MIGRATION.md` - Complete v2.0.0 migration instructions
- **Claude Instructions**: `CLAUDE.md` - AI assistant guidance
- **TODO List**: `TODO.md` - Development roadmap
- **API Documentation**: Generated via `make docs`
- **Research Papers**: See `references/` for ATFT-GAT-FAN papers

---

## ML Dataset Enrichment (TOPIX / Trade-Spec / Statements)

This project produces a price-driven base dataset and then enriches it with market (TOPIX), investor flow (trade-spec), and financial statement features. The base pipeline remains unchanged; enrichment is applied afterward via lightweight wrappers.

### What goes into the final dataset
- Price + technical: returns/vol/MA gaps, pandas-ta, targets, validity flags
- Market (TOPIX) features: `mkt_*` (26) + cross features (8)
- Trade-spec flow features: investor activity metrics (interval → daily projection)
- Financial statement features: as-of attached YoY/progress/revisions/ratios

### Join rules (safe, leakage-free)
- TOPIX: `Date` left join (same value for all stocks that day); cross features use market return with t-1 lag.
- Trade-spec (weekly, by section):
  - Effective window: from next business day of `PublishedDate` until the day before the next publication within the same `section_norm`.
  - Daily projection: expand interval to a business-day grid; join on `(section_norm, Date)`.
  - Price side must have `section_norm ∈ {TSEPrime, TSEStandard, TSEGrowth}` attached from listed info.
- Financial statements (event-based, by code):
  - Effective date: `DisclosedDate` (<=15:00) or next business day (>15:00).
  - As-of backward join: for each `(Code, Date)` attach the latest disclosure with `effective_date ≤ Date`.
- Types: normalize `Date` to date type before joins; keep non-covered days as null and gate by validity flags.

### How to run (wrappers)
- Base + TOPIX (no change to base pipeline):
  - `python scripts/pipelines/run_pipeline_with_topix.py [--jquants] [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD]`
  - Outputs: `output/ml_dataset_latest_with_topix.parquet` (+ metadata)
- Base + TOPIX + Trade-Spec + Statements:
  - `python scripts/pipelines/run_pipeline_with_enrichment.py [--jquants] [--trades-spec /path/to.parquet] [--statements /path/to.parquet]`
  - If `--statements` is omitted, searches `output/event_raw_statements*.parquet` automatically.
  - Outputs: `output/ml_dataset_latest_enriched.parquet` (+ metadata)

### Make targets
- `make dataset-with-topix`  (vars: `JQ=1`, `START_DATE`, `END_DATE`)
- `make dataset-with-enrichment`  (vars: `JQ=1`, `TRADES_SPEC=...`, `STATEMENTS=...`)

### Notes
- `run_pipeline_v4_optimized.py` alone does not add TOPIX/flow/statements by default; use the wrappers above or call the builder APIs explicitly.
- J-Quants usage requires tokens/credentials in `.env`. If absent, TOPIX falls back to sample generation (z-score windows still require warm-up).
