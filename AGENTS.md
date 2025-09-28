# Repository Guidelines

This file is a concise contributor guide for gogooku3. It applies repo‑wide; nested AGENTS.md files may override locally.

## Project Structure & Module Organization
- Code: `src/gogooku3/` (CLI, utils, features, models, training, graph, compat)
- Pipelines: `scripts/` (e.g., `run_safe_training.py`, `train_atft.py`, `pipelines/`)
- Configs: `configs/` (model/data/training/hardware YAMLs)
- Tests: `tests/` with fixtures in `tests/fixtures/`
- Runtime: `data/`, `output/`, `cache/`, `_logs/` (do not commit `_logs/`)
- Ops: `dagster_repo/`, `docker-compose*.yml`

## Build, Test, and Development Commands
- Setup: `pip install -e ".[dev]"` or `make setup` (editable install + dev deps).
- Train: `gogooku3 train --config configs/atft/train/production.yaml`.
- Safe CV: `python scripts/run_safe_training.py --n-splits 2 --memory-limit 6`.
- Tests (fast): `pytest -m "not slow"`.
- Coverage: `pytest --cov=src/gogooku3 --cov-report=term-missing`.
- Quality: `pre-commit run --all-files && ruff check src/ --fix && black . && mypy src/gogooku3 && bandit -r src/`.

## Coding Style & Naming Conventions
- Python 3.10+, 4 spaces. Format with Black (line length 88) and isort (`profile=black`); lint with Ruff.
- Type hints required; `mypy` runs in strict mode on `src/gogooku3/`.
- Naming: modules `snake_case.py`; classes `CamelCase`; functions/variables `snake_case`.
- Example: `src/gogooku3/training/safe_training_pipeline.py::SafeTrainingPipeline`.

## Testing Guidelines
- Framework: `pytest`; markers: `unit`, `integration`, `slow`, `smoke`, `requires_api`.
- Name tests like `tests/test_<module>.py::TestClass::test_case`.
- Deterministic: no network in unit tests; gate external calls with `@pytest.mark.requires_api`.
- Run fast suite: `pytest -m "not slow"`; add coverage with the command above.
- Use shared data from `tests/fixtures/`.

## Commit & Pull Request Guidelines
- Commits: Conventional Commits (e.g., `feat(training): add EMA teacher`).
- PRs: explain what/why, link issues (`Closes #123`), call out config/docs changes, attach key metrics/plots, and ensure CI is green.

## Security & Configuration Tips
- Secrets via env: `JQUANTS_AUTH_EMAIL`, `JQUANTS_AUTH_PASSWORD`, `WANDB_API_KEY` (never commit `.env`).
- Data safety: use walk-forward CV with 20‑day embargo; fit normalization on train only.
- Artifacts: logs in `_logs/`, caches in `cache/`, outputs in `output/`.

## Architecture Overview
- Core: ATFT‑GAT‑FAN multi‑horizon forecasting.
- Pipeline: 7‑step `SafeTrainingPipeline` with strict leakage prevention.

