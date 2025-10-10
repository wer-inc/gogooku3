# Repository Guidelines

## Project Structure & Module Organization
- `src/gogooku3/`: CLI, models, training pipelines, graph utils, compatibility layers.
- `scripts/`: end-to-end pipelines (e.g., `run_safe_training.py`, `train_atft.py`, `pipelines/`).
- `configs/`: YAML configs; production-ready live under `configs/atft/`.
- `tests/` with fixtures in `tests/fixtures/`.
- Runtime artifacts: `data/`, `output/`, `cache/`, `_logs/` (do not commit `_logs/`).

## Build, Test, and Development Commands
- Setup: `pip install -e ".[dev]"` or `make setup`.
- Fast tests: `pytest -m "not slow"`.
- Coverage: `pytest -m "not slow" --cov=src/gogooku3 --cov-report=term-missing`.
- Train (standard): `gogooku3 train --config configs/atft/train/production.yaml`.
- Safe CV: `python scripts/run_safe_training.py --n-splits 2 --memory-limit 6`.
- Optimized loop: `make train-optimized`.

## Coding Style & Naming Conventions
- Python 3.10+, 4-space indentation.
- Format: Black (line length 88) and isort (`profile=black`).
- Lint: Ruff; Security: Bandit.
- Types: mypy strict for `src/gogooku3/`.
- Naming: modules `snake_case.py`, classes `CamelCase`, functions/vars `snake_case`.

## Testing Guidelines
- Framework: pytest; keep tests deterministic.
- Mark external calls with `@pytest.mark.requires_api`; slow cases with `@pytest.mark.slow`.
- Name tests `test_*.py`; reuse fixtures from `tests/fixtures/`.
- Run coverage locally before PRs (see command above).

## Commit & Pull Request Guidelines
- Commits: Conventional Commits (e.g., `feat(training): add ema teacher`). One logical change per commit.
- PRs: explain what/why, link issues (e.g., `Closes #123`), call out config/docs updates.
- For training changes, attach key metrics/plots and ensure CI passes.

## Security & Configuration Tips
- Keep secrets out of the repo. Use env vars: `JQUANTS_AUTH_EMAIL`, `JQUANTS_AUTH_PASSWORD`, `WANDB_API_KEY`.
- Fit normalizers on training folds only; enforce a 20‑day embargo in walk‑forward splits.
- Store transient logs in `_logs/`; avoid committing runtime artifacts.

## Architecture Overview
- Core: ATFT‑GAT‑FAN multi‑horizon forecaster orchestrated by the seven‑step `SafeTrainingPipeline`.
- Graph attention uses regularized GAT layers; ensure `configs/atft/*` keep `model.gat.regularization` populated to preserve gradient flow.

