# APEX-Ranker Model Package (Skeleton)

This package will contain the modular APEX-Ranker implementation for gogooku5.

## Planned Modules
- `src/apex_ranker/models/`: ranking architectures and losses.
- `src/apex_ranker/training/`: training loops and evaluation routines.
- `src/apex_ranker/config/`: configuration schemas and defaults.
- `scripts/train_v0.py` / `scripts/inference_v0.py`: CLI entrypoints.

## Usage (future)
```bash
make train         # full training run
make train-quick   # smoke test
make inference     # batch inference placeholder
```

Refer to the top-level migration guidelines for testing and validation expectations.
