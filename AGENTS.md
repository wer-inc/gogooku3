# Repository Guidelines

## Project Structure & Module Organization
- `src/gogooku3/` holds the CLI, models, training pipelines, graph utilities, and compatibility layers.
- `scripts/` contains end-to-end pipelines (e.g., `run_safe_training.py`, `train_atft.py`, `pipelines/`).
- `configs/` stores model, data, training, and hardware YAMLs; keep production-ready configs in `configs/atft/`.
- Tests live under `tests/` with reusable fixtures in `tests/fixtures/`; runtime artifacts belong in `data/`, `output/`, `cache/`, and `_logs/` (do not commit `_logs/`).

## Build, Test, and Development Commands
- `pip install -e ".[dev]"` or `make setup` installs the project in editable mode with dev dependencies.
- `pytest -m "not slow"` runs the fast unit and integration suite; add `--cov=src/gogooku3 --cov-report=term-missing` for coverage.
- `gogooku3 train --config configs/atft/train/production.yaml` launches the standard training pipeline.
- `python scripts/run_safe_training.py --n-splits 2 --memory-limit 6` executes the walk-forward safe CV pipeline.
- Use `make train-optimized` to run the latest production-tuned training loop.

## Coding Style & Naming Conventions
- Target Python 3.10+ with 4-space indentation; format using Black (line length 88) and isort (`profile=black`).
- Lint with Ruff, type-check `src/gogooku3/` in strict mode via mypy, and run Bandit for security scans.
- Modules follow `snake_case.py`, classes use `CamelCase`, and functions/variables stay in `snake_case`.

## Testing Guidelines
- Prefer deterministic pytest cases; gate any external calls with `@pytest.mark.requires_api`.
- Mark slow scenarios with `@pytest.mark.slow` so the fast suite stays lean.
- Reuse fixtures from `tests/fixtures/` and keep new fixtures data-free when possible.

## Commit & Pull Request Guidelines
- Use Conventional Commits (e.g., `feat(training): add ema teacher`); one logical change per commit.
- PRs must explain the what/why, reference issues (`Closes #123`), and call out config or docs updates.
- Attach key metrics or plots for training changes and confirm CI passes before requesting review.

## Security & Configuration Tips
- Keep secrets out of the repo; load credentials via environment variables such as `JQUANTS_AUTH_EMAIL`, `JQUANTS_AUTH_PASSWORD`, and `WANDB_API_KEY`.
- Fit normalizers on training folds only, respect the 20-day embargo in walk-forward splits, and store transient logs in `_logs/`.

## Architecture Overview
- The core stack centers on the ATFT-GAT-FAN multi-horizon forecaster orchestrated by the seven-step `SafeTrainingPipeline`.
- Graph attention relies on regularized GAT layers; ensure configs under `configs/atft/` keep `model.gat.regularization` populated to preserve gradient flow.
