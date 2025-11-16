# Repository Guidelines

## Project Structure & Module Organization
- Core training pipelines live in `scripts/` (e.g. `scripts/integrated_ml_training_pipeline.py`).
- Dataset building and orchestration are under `scripts/pipelines/` and `tools/`.
- Reusable model and data components belong in `common/` and `models/`.
- Keep experiments, notebooks, and one-off scripts out of these core directories.

## Build, Test, and Development Commands
- `tools/project-health-check.sh` — run basic code, env, and data sanity checks.
- `python scripts/integrated_ml_training_pipeline.py` — train the main GAT-based model.
- `python scripts/pipelines/run_full_dataset.py` — build the full dataset locally.
- `make build-chunks`, `make merge-chunks` — generate and merge historical data chunks.

## Coding Style & Naming Conventions
- Use Python 3.10+ with type hints for all functions and docstrings for public APIs.
- Prefer explicit, descriptive names: `jpx_price_loader`, `gat_block`, `feature_normalizer`.
- Follow PEP 8 (4-space indentation, 88–100 char lines) and keep imports ordered and grouped.
- Place configuration in YAML or clearly named Python modules, not hard-coded in scripts.

## Testing Guidelines
- Add unit tests for core financial logic, data transforms, and model components.
- Mirror package layout under a `tests/` tree; name tests like `test_*.py`.
- Run targeted tests before pushing (e.g. `pytest tests/common`), plus the health check script.

## Commit & Pull Request Guidelines
- Write clear, imperative commit messages: `Add GAT layer for sector graph`, `Fix JPX holiday handling`.
- Keep PRs focused and small; describe motivation, key changes, and validation (commands, metrics).
- Link related issues and include screenshots or metric tables for modeling changes when relevant.

## Security & Configuration Notes
- Never commit API keys or proprietary datasets; use `.env` files and local config.
- Validate new data sources carefully; ensure dates, tickers, and currencies are consistent with JPX.
