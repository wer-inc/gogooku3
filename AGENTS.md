# Repository Guidelines

## Project Structure & Module Organization
- `src/gogooku3/`: core package (CLI, utils, data, features, models, graph, training, compat).
- `scripts/`: runnable pipelines (e.g., `run_safe_training.py`, `train_atft.py`, `pipelines/`).
- `configs/`: YAML configs for models, data, training, hardware.
- `tests/`: unit/integration/smoke; fixtures in `tests/fixtures/`.
- Runtime dirs: `data/`, `output/`, `cache/`, `_logs/` (do not commit `_logs/`).
- Orchestration: `dagster_repo/`, `docker-compose*.yml`.

## Build, Test, and Development Commands
- Setup: `pip install -e ".[dev]"` or `make setup`.
- Train (CLI): `gogooku3 train --config configs/atft/train/production.yaml`.
- Pipeline (safe training): `python scripts/run_safe_training.py --n-splits 2 --memory-limit 6`.
- Tests (fast): `pytest -m "not slow"`.
- Coverage: `pytest --cov=src/gogooku3 --cov-report=term-missing`.
- Quality: `pre-commit run --all-files`; `ruff check src/ --fix`; `black .`; `mypy src/gogooku3`; `bandit -r src/`.

## Coding Style & Naming Conventions
- Python 3.10+, 4-space indents, max line length 88 (Black).
- Format with Black; sort imports via isort (`profile=black`); lint with Ruff.
- Add type annotations; `mypy` runs strict on `src/gogooku3/`.
- Naming: modules `snake_case.py`; classes `CamelCase`; functions/variables `snake_case`.

## Testing Guidelines
- Framework: `pytest` with markers `unit`, `integration`, `slow`, `smoke`.
- Location: `tests/unit/`, `tests/integration/`; fixtures in `tests/fixtures/`.
- Naming: `test_<module>.py::TestClass::test_case`.
- Deterministic tests; no network in unit tests; gate external calls with `requires_api`.
- Run coverage before PRs: `pytest --cov`.

## Commit & Pull Request Guidelines
- Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).
- Examples: `feat(training): add EMA teacher`; `fix(graph): handle isolated nodes`.
- PRs: explain what/why, link issues (e.g., `Closes #123`), call out config/docs changes, attach key metrics/plots, ensure CI is green.

## Security & Configuration Tips
- Store secrets as env vars (do not commit `.env`): `JQUANTS_AUTH_EMAIL/PASSWORD`, `WANDB_API_KEY`.
- Respect safety rules: walk-forward with 20-day embargo; cross-sectional normalization fits on train only.
- Logs in `_logs/`, caches in `cache/`, artifacts in `output/`.

## Architecture Overview
- Core: ATFT‑GAT‑FAN multi‑horizon forecasting.
- Training: 7‑step SafeTrainingPipeline with strict leakage prevention.

