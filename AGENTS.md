# Repository Guidelines

## Project Structure & Modules
- `src/gogooku3/`: core package (CLI, utils, data, features, models, graph, training, compat).
- `scripts/`: runnable pipelines (`run_safe_training.py`, `train_atft.py`, `pipelines/`; wrapper archived).
- `configs/`: model/data/training/hardware configs.
- `tests/`: unit, integration, smoke (fixtures in `tests/fixtures/`).
- `data/`, `output/`: local data/artifacts; `_logs/`: runtime logs (do not commit).
- `dagster_repo/`, `docker-compose*.yml`: orchestration and services.

## Build, Test, and Development
- Install: `pip install -e ".[dev]"` (or `make setup`).
- CLI: `gogooku3 train --config configs/training/production.yaml`.
- Pipelines: `python scripts/run_safe_training.py --n-splits 2 --memory-limit 6`.
- Tests: `pytest -m "not slow"` and `pytest --cov=src/gogooku3 --cov-report=term-missing`.
- Quality: `pre-commit run --all-files`; `ruff check src/ --fix`; `black .`; `mypy src/gogooku3`; `bandit -r src/`.

## Coding Style & Naming
- Python 3.10+, 4-space indent, max line length 88.
- Formatting: Black; imports via isort (black profile); lint with Ruff.
- Types: add annotations; `mypy` runs strict-ish on `src/gogooku3`.
- Naming: modules `snake_case.py`; classes `CamelCase`; functions/vars `snake_case`.
- Keep changes minimal and focused; avoid unrelated fixes in the same PR.

## Testing Guidelines
- Framework: `pytest` with markers `unit`, `integration`, `slow`, `smoke`.
- Location: `tests/unit/`, `tests/integration/`; fixtures under `tests/fixtures/`.
- Naming: `test_<module>.py::TestClass::test_case`.
- Deterministic tests; no network in unit tests; gate external calls with `requires_api`.
- Aim for meaningful coverage; run `pytest --cov` before PR.

## Commit & PR Guidelines
- Commits: Conventional Commits (e.g., `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).
- Examples: `feat(training): add EMA teacher`, `fix(graph): handle isolated nodes`.
- PRs: clear what/why, link issues (`Closes #123`), call out config/docs changes, attach key metrics/plots where relevant, ensure CI green.
- Do not commit secrets, datasets, or logs; use `.env.example` for reference.

## Security & Configuration
- Secrets via environment (do not commit `.env`): `JQUANTS_AUTH_EMAIL/PASSWORD`, `WANDB_API_KEY`.
- Respect safety rules: walk-forward with 20‑day embargo; cross-sectional normalization fits on train only.
- Keep logs in `_logs/`, caches in `cache/`, artifacts in `output/`.

## Architecture Overview
- Core: ATFT‑GAT‑FAN (multi-horizon); training via 7‑step SafeTrainingPipeline with strict leakage prevention.
