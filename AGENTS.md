# Repository Guidelines

## Project Structure & Module Organization
- Code: `src/gogooku3/` — CLI, utils, data, features, models, graph, training, compat.
- Pipelines: `scripts/` (e.g., `run_safe_training.py`, `train_atft.py`, `pipelines/`).
- Configs: `configs/` for model/data/training/hardware YAMLs.
- Tests: `tests/` with fixtures in `tests/fixtures/`.
- Runtime: `data/`, `output/`, `cache/`, `_logs/` (do not commit `_logs/`).
- Orchestration: `dagster_repo/`, `docker-compose*.yml`.

## Build, Test, and Development Commands
- Setup: `pip install -e ".[dev]"` or `make setup`.
- Train (CLI): `gogooku3 train --config configs/atft/train/production.yaml`.
- Safe training: `python scripts/run_safe_training.py --n-splits 2 --memory-limit 6`.
- Fast tests: `pytest -m "not slow"`.
- Coverage: `pytest --cov=src/gogooku3 --cov-report=term-missing`.
- Quality: `pre-commit run --all-files`; `ruff check src/ --fix`; `black .`; `mypy src/gogooku3`; `bandit -r src/`.

## Coding Style & Naming Conventions
- Python 3.10+, 4-space indents, Black (88 chars), isort (`profile=black`), Ruff.
- Add type annotations; `mypy` is strict on `src/gogooku3/`.
- Names: modules `snake_case.py`; classes `CamelCase`; functions/vars `snake_case`.

## Testing Guidelines
- Framework: `pytest` with markers `unit`, `integration`, `slow`, `smoke`.
- Locations: `tests/unit/`, `tests/integration/`; fixtures in `tests/fixtures/`.
- Naming: `test_<module>.py::TestClass::test_case`.
- Deterministic tests; no network in unit tests; gate external calls with `requires_api`.
- Before PRs: run coverage and quality commands above.

## Commit & Pull Request Guidelines
- Commits: Conventional Commits (e.g., `feat(training): add EMA teacher`).
- PRs: explain what/why, link issues (`Closes #123`), call out config/docs changes, attach key metrics/plots, ensure CI is green.

## Security & Configuration Tips (Optional)
- Secrets via env vars: `JQUANTS_AUTH_EMAIL`, `JQUANTS_AUTH_PASSWORD`, `WANDB_API_KEY` (never commit `.env`).
- Respect safety: walk-forward with 20-day embargo; fit normalization on train only.
- Logs in `_logs/`, caches in `cache/`, artifacts in `output/`.

## Architecture Overview (Optional)
- Core: ATFT-GAT-FAN multi-horizon forecasting.
- Training: 7-step SafeTrainingPipeline with strict leakage prevention.

