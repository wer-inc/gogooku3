# ATFT-GAT-FAN Model Package (Skeleton)

This package will host the modularized ATFT-GAT-FAN implementation. It consumes the shared dataset located at `../../data/output/datasets/ml_dataset_latest.parquet`.

## Planned Modules
- `src/atft_gat_fan/models/`: neural network architectures and layers.
- `src/atft_gat_fan/training/`: trainers, callbacks, and SafeTrainingPipeline integrations.
- `src/atft_gat_fan/config/`: configuration schemas and defaults.
- `scripts/train_atft.py`: entrypoint for Makefile targets.

## Usage (future)
```bash
make train          # full training run
make train-quick    # short smoke test
make lint           # ruff + mypy
make test           # pytest suite
```

Refer to `docs/development/development-guidelines.md` for coding standards and validation requirements.
