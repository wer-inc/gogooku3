# gogooku5 Modular Migration Workspace

This workspace tracks the migration from the monolithic `gogooku3` system to the modular `gogooku5` architecture outlined in `MIGRATION_PLAN.md`.

## Goals
- Isolate dataset generation under `data/` and distribute models under `models/<name>/`.
- Maintain a shared, versioned dataset artifact (`data/output/ml_dataset_latest.parquet`) consumed by every model package.
- Keep documentation, tooling, and testing close to the relevant package for fast iteration and review.

## Repository Layout (in-progress)
```
gogooku5/
├── data/               # Dataset builder package (standalone)
├── models/             # Individual model packages (ATFT-GAT-FAN, APEX-Ranker, ...)
├── common/             # Optional shared utilities (only when >1 consumer)
├── tools/              # Agent launchers, health checks, shared scripts
├── Makefile            # Top-level shortcuts delegating into sub-packages
└── MIGRATION_PLAN.md   # Migration roadmap and milestones
```

## Working Agreements
1. Follow `docs/development/development-guidelines.md` for workflow, testing, and documentation expectations.
2. Update `MIGRATION_PLAN.md` as milestones complete or priorities change.
3. Run `tools/health-check.sh` (coming soon) before requesting reviews or merging cross-package changes.

## Getting Started
1. Review the migration plan and the dataset/model package READMEs.
2. Set up required environment variables in `data/.env.example` and copy to `.env` as needed.
3. Use the top-level Make targets (`make dataset`, `make train-atft`, etc.) to dispatch work into each package.

> **Status:** Skeleton structure only. Implementation work is tracked per migration phase.
