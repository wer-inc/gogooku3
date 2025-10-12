# Repository Guidelines

## Project Structure & Module Organization
- `src/gogooku3/`: CLI entrypoints, model definitions, training pipelines, graph utilities, and compatibility layers.
- `scripts/`: End-to-end workflows such as `run_safe_training.py`, `train_atft.py`, and reusable pipeline helpers.
- `configs/`: YAML configs; production-ready variants live under `configs/atft/`.
- `tests/`: Pytest suite with fixtures under `tests/fixtures/`; runtime artifacts belong in `data/`, `output/`, `cache/`, `_logs/` (do not commit `_logs/`).

## Build, Test, and Development Commands
- `pip install -e ".[dev]"` or `make setup`: install project plus development tooling in editable mode.
- `pytest -m "not slow"`: run the fast unit test subset.
- `pytest -m "not slow" --cov=src/gogooku3 --cov-report=term-missing`: execute coverage-focused test run.
- `gogooku3 train --config configs/atft/train/production.yaml`: launch the standard ATFT training pipeline.
- `python scripts/run_safe_training.py --n-splits 2 --memory-limit 6`: perform safe cross-validation with resource controls.
- `make train-optimized`: run the optimized training loop tuned for iteration speed.

## Coding Style & Naming Conventions
- Python 3.10+, four-space indentation, modules and functions in `snake_case`, classes in `CamelCase`.
- Format with Black (line length 88) and isort (`profile=black`); lint via Ruff, security-scan with Bandit.
- Maintain strict mypy types for `src/gogooku3/`; favor concise docstrings for public APIs.

## Testing Guidelines
- Use pytest with deterministic tests named `test_*.py`; reuse fixtures from `tests/fixtures/`.
- Mark slow suites with `@pytest.mark.slow`; external services with `@pytest.mark.requires_api`.
- Target full coverage before PR by running the coverage command above.

## Commit & Pull Request Guidelines
- Follow Conventional Commits (e.g., `feat(training): add ema teacher`); keep each commit logically scoped.
- PRs should explain what and why, link issues with `Closes #123`, and flag config or docs changes.
- Share key training metrics or plots when altering pipelines; ensure CI and coverage remain green.

## Security & Configuration Tips
- Keep secrets out of the repo; provide `JQUANTS_AUTH_EMAIL`, `JQUANTS_AUTH_PASSWORD`, `WANDB_API_KEY` via environment variables.
- Avoid committing runtime artifacts; store transient logs under `_logs/`.
- Fit normalizers on training folds only and preserve the 20-day embargo in walk-forward splits.

## Architecture Overview
- Core system is the ATFT-GAT-FAN multi-horizon forecaster orchestrated by the seven-step `SafeTrainingPipeline`.
- Graph attention relies on regularized GAT layers; ensure `configs/atft/*` keep `model.gat.regularization` populated to protect gradient flow.
