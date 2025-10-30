# gogooku5 Dataset Builder (Skeleton)

This package will host the standalone dataset generation pipeline described in `../MIGRATION_PLAN.md`.

## Roadmap
1. Implement configuration and API clients under `src/builder/api/`.
2. Migrate feature engineering modules (`core`, `legacy`, `macro`).
3. Introduce full and optimized pipelines under `src/builder/pipelines/` with caching support.
4. Deliver integration tests that rebuild one month of data and compare with `gogooku3` outputs.

## Usage
```bash
# Ensure environment variables are set
cp .env.example .env

# Build 30-day dataset (START/END default to last 30 days)
make build

# Override date range
make build START=2024-01-01 END=2024-01-31

# Warm caches without writing parquet output
make build-optimized START=2024-01-01 END=2024-01-31 CACHE_ONLY=1
```

Detailed pipeline behavior, feature coverage, and validation routines will be documented as implementation progresses through the migration milestones.
