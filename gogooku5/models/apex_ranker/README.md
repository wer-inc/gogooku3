# APEX-Ranker for gogooku5

This package hosts the modularised **APEX-Ranker v0** implementation inside the
`gogooku5` repository. It reuses the shared dataset artifacts generated under
`gogooku5/data` and keeps training, inference, and backtesting utilities
self-contained under `models/apex_ranker/`.

## Layout

```
models/apex_ranker/
├── configs/                  # YAML configs (feature groups, training presets)
├── scripts/                  # CLI entry-points (train, inference, backtests)
├── src/apex_ranker/          # Python package (data, models, backtest, API)
├── tests/                    # Unit/integration tests (WIP)
├── Makefile.train            # Convenience targets for training/inference
└── pyproject.toml            # Package metadata & dependencies
```

## Quick start

The commands below assume you run them from the repository root. The defaults
expect a final training dataset under `data/output/datasets/` (for example
`data/output/datasets/ml_dataset_latest_full.parquet`). The helper functions
automatically resolve alternative locations such as `data/output/ml_dataset_latest_full.parquet`
or `ml_dataset_latest_full.parquet` at the workspace root.

```bash
# Full training run (see configs/v0_base.yaml for hyper-parameters)
make -f models/apex_ranker/Makefile.train train

# Smoke-test run with truncated history and epochs
make -f models/apex_ranker/Makefile.train train-quick

# Generate inference rankings using an existing checkpoint
make -f models/apex_ranker/Makefile.train \
    inference APEX_MODEL=output/models/apex_ranker_v0_enhanced.pt
```

Key script entry-points (all relative to this directory):

- `scripts/train_v0.py` – end-to-end training loop with composite ranking loss
- `scripts/inference_v0.py` – batch inference that resolves dataset artifacts
- `scripts/backtest_v0.py` – simple equal-weight portfolio backtest

Refer to `configs/feature_groups.yaml` for column bundles and
`configs/v0_base.yaml` for training defaults. Paths inside the configs are
resolved relative to the config location or the gogooku5 root, so manual
absolute paths are rarely necessary.

## Development notes

- All package dependencies are declared in `pyproject.toml`; install locally via
  `pip install -e models/apex_ranker[dev]` when working on this module.
- Shared helpers in `src/apex_ranker/data/dataset_loader.py` provide resilient
  path resolution for datasets, metadata, and artifacts.
- Tests are staged under `tests/` – add unit/integration coverage as new
  features are migrated.

For additional background see the documentation collection under
`docs/` and the original design notes in `gogooku5/MIGRATION_PLAN.md`.
