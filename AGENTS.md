# Repository Guidelines

## Project Structure & Module Organization
- `src/gogooku3/` is the canonical source tree (pipeline builders, training modules, shared utils).
- `scripts/` contains CLI façades (`run_pipeline_v4_optimized.py`, `run_safe_training.py`) that orchestrate `src` code.
- `configs/atft/` hosts Hydra configs for ATFT and safe-mode variants; add new profiles here.
- Tests live under `tests/` (`unit/`, `integration/`, `exploratory/`); mirror this layout when extending coverage.
- Working artefacts land in `output/` and `_logs/`; these are ignored by Git and should stay untracked.

## Build, Test, and Development Commands
- `make dataset-bg START=2020-01-01 END=2024-12-31` – background GPU dataset build (logs in `_logs/dataset/`).
- `make train-safe EPOCHS=75` – run SafeTrainingPipeline with thread limits (`_logs/training/`).
- `make train` or `make train-optimized` – foreground optimized training with full DataLoader features.
- `make test` – execute pytest defaults; use `TEST_TARGET=tests/integration` to scope.
- `python scripts/run_pipeline_v4_optimized.py --start-date 2024-01-01 --end-date 2024-12-31` – smart-cache dataset generation for ad hoc ranges.

## Coding Style & Naming Conventions
- Python 3.10+, 4 spaces, 88-char lines (Black + Ruff). Run `ruff check .` and `black .` pre-commit.
- Add type hints and dataclasses for shared state; avoid implicit Any.
- Stick to snake_case for functions/modules, UpperCamelCase for classes, and SCREAMING_SNAKE_CASE constants (e.g., `SUPPORT_LOOKBACK_DAYS`).
- Hydra config keys remain lower_snake_case; document overrides in-line.

## Testing Guidelines
- Tests use Pytest (`pytest tests/unit` for quick runs, `pytest tests/integration -m gpu` for GPU cases).
- Name files `test_<feature>.py` and functions `test_<scenario>__<expectation>()`.
- Cover new cache or data-fetch flows with integration tests or scripted checks under `scripts/cache/`.
- Tag long jobs with `@pytest.mark.slow`; default CI runs `-k "not slow"`.

## Commit & Pull Request Guidelines
- Use Conventional Commits (`feat(cache):`, `fix(training):`, `docs(setup):`); scopes mirror directories or subsystems.
- Keep commits focused and note API/cache contract changes in the body; reference tickets when applicable.
- PR checklist: short summary, verification commands (`make test`, training log), screenshots for UI/log diffs, and comments on dataset/cache impact.
- Ping module owners in `CLAUDE.md` for reviews and flag anything that changes GPU, credential, or dataset requirements.
