# Development Guidelines for gogooku5 Migration

These guidelines codify how we build, validate, and ship work while migrating from `gogooku3` to the modular `gogooku5` architecture. They complement `docs/development/contributing.md`, `docs/development/testing.md`, and `gogooku5/MIGRATION_PLAN.md`.

## 1. Engineering Principles
- **Modularity first**: keep dataset generation under `data/`, model logic under `models/<name>/`, and shared utilities in `common/` only when reuse is proven.
- **Deterministic outputs**: dataset diffs against `gogooku3` are mandatory before accepting feature parity.
- **GPU-aware design**: assume access to an NVIDIA A100 80GB; default to GPU paths and fall back to CPU explicitly.
- **Observability & safety**: every pipeline must surface health checks, logs, and failure recovery steps.

## 2. Repository & Directory Policy
- Follow the target structure documented in `gogooku5/MIGRATION_PLAN.md`.
- `data/`: standalone Pyproject-managed package for dataset building; avoid cross-imports from `models/`.
- `models/<model>/`: each model owns its dependencies, configs, scripts, and tests.
- `common/`: add modules only after two consumers exist; document rationale in `docs/development/memories.md`.
- Keep symbolic links (e.g., `data/output/ml_dataset_latest.parquet`) intact—models consume only the `*_latest` alias.

## 3. Workflow Expectations
- **Planning**: create a task breakdown referencing migration phases (Week/Day milestones) before coding.
- **Issue tracking**: capture TODOs in `DOC_TASKS.json` or relevant tracking files; update status at handoff.
- **Review readiness**: ensure health checks and relevant tests pass locally; attach logs or summaries in PR descriptions.
- **Decision logging**: record non-obvious trade-offs in `docs/development/memories.md` with timestamps.

## 4. Coding Standards
- Type hints for all functions; `typing.Annotated` when embedding units or constraints is useful.
- Public functions, classes, and modules require docstrings that describe intent, inputs, outputs, and side effects.
- Prefer Polars and vectorized operations; document CPU fallbacks when needed.
- Enforce linting (`ruff check`, `mypy`, `pyright` if applicable) before submission; failures block merge.
- Respect existing style guides (`docs/development/conventions.md`); add concise comments for complex logic only.

## 5. Data Handling & Validation
- `.env.example` must list every environment variable required by new code; keep secrets out of the repo.
- Use caching utilities from `data/src/builder/utils/cache.py`; never roll custom cache logic without review.
- For new features, supply unit tests comparing against `gogooku3` outputs and include fixture updates under `data/tests/fixtures/`.
- Run dataset builds via Make targets (`make dataset-bg`, `make dataset-gpu ...`); share the command and commit hash in validation notes.
- Validate schema changes with an explicit diff report (e.g., column additions, dtype changes).

## 6. Training & Model Guidelines
- Load datasets through shared parquet paths under `data/output/`; do not access raw fetchers from model packages.
- Use Make targets (`make train`, `make train-optimized`, model-specific `Makefile.train`) instead of ad-hoc scripts.
- Monitor GPU utilization with `nvidia-smi` or `tools/project-health-check.sh`; adjust batch sizes rather than leaving memory idle.
- Any new training loop must integrate SafeTrainingPipeline safeguards (checkpointing, OOM recovery).
- Capture experiment metadata in `mlruns/` (MLflow) or the designated tracking system; add summaries to `EXPERIMENT_STATUS.md` when relevant.

## 7. Testing & Verification
- Minimum expectation: unit tests for new modules, integration tests for pipelines, and end-to-end dataset or training smoke tests.
- Run `tools/project-health-check.sh` after significant changes; attach a summary of pass/fail sections.
- For dataset changes, run one-month rebuilds during development and five-year rebuilds before release (align with Phase 1 milestones).
- Maintain >80% coverage in the dataset builder package; track deficits in `docs/development/testing.md`.

## 8. Documentation & Communication
- Update corresponding README files (`data/README.md`, `models/<model>/README.md`) when adding features or commands.
- Keep `gogooku5/MIGRATION_PLAN.md` synchronized—mark completed milestones and adjust timelines with rationale.
- Explain complex or controversial decisions in code review summaries and link back to supporting docs.
- For external communication (stakeholders), use `docs/OPERATIONS` or relevant governance documents.

## 9. Security & Compliance
- Handle financial data per sensitivity guidelines: no raw dumps in logs, respect access scopes, and rotate credentials via secrets management.
- Verify that network calls go through approved clients (`api/` modules); document new providers before integration.
- Ensure temporary files reside in project-sanctioned directories (`/workspace/gogooku3/tmp` or `data/cache/`) and are cleaned up.

## 10. Continuous Improvement
- After each milestone, perform a retrospective: what slowed progress, what automation is needed. Capture actions in `docs/development/clean_up.md`.
- Propose optimizations (GPU kernels, caching, data schema) proactively; validate impact before rolling to main.
- Keep the guideline document current—treat updates as part of delivery definition.

Adhering to these practices keeps the migration predictable, auditable, and aligned with production-readiness goals.
