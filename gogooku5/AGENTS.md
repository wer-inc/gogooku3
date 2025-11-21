# Codex Agent Instructions - gogooku5 Project

You are an autonomous AI developer working on gogooku5, a modular refactoring of the gogooku3 Japanese stock market prediction system.

## Project Context

**Main Goal**: Modular ML pipeline with separation of dataset generation and model training
**Tech Stack**: Python, PyTorch, Polars, RAPIDS/cuDF, Dagster, MLflow
**Hardware**: NVIDIA A100 80GB GPU, 255-core CPU, 1.8TB RAM

## Architecture Overview

gogooku5 separates concerns into independent packages:
- **data/**: Dataset generation (30+ feature modules, GPU-accelerated ETL)
- **models/**: Model-specific packages (APEX-Ranker, ATFT-GAT-FAN)
- **common/**: Shared utilities (minimal, only when >1 consumer)

## Key Project Files

- `data/src/builder/pipelines/dataset_builder.py` - Dataset orchestration
- `data/src/cli/main.py` - CLI interface for dataset generation
- `models/apex_ranker/scripts/train_v0.py` - APEX-Ranker training
- `CLAUDE.md` - Comprehensive project documentation
- `MIGRATION_PLAN.md` - Migration roadmap
- `tools/health-check.sh` - Health diagnostics

## Development Guidelines

1. **Always Read Before Editing**: Use `codex read <file>` to understand context
2. **Test After Changes**: Run tests and validation scripts
3. **Document Changes**: Update relevant documentation
4. **Optimize for GPU**: Leverage A100's 80GB memory for GPU-ETL (RAPIDS/cuDF)
5. **Financial Data Sensitivity**: Handle market data with proper as-of joins (T+1 logic)

## Autonomous Workflow

When working autonomously:
1. Analyze health check reports thoroughly
2. Create detailed todo lists for complex tasks
3. Fix critical issues first (P0 → P1 → P2)
4. Run validation after each major change
5. Document reasoning for non-obvious decisions

## Code Quality Standards

- Type hints for all functions
- Docstrings for public APIs
- Unit tests for core logic
- Memory-efficient data processing (Polars, GPU-ETL)
- Schema validation for datasets

## Useful Commands

```bash
# Health check
tools/health-check.sh

# Dataset generation (from project root)
make -C data build START=2024-01-01 END=2024-12-31

# APEX-Ranker training
make -C models/apex_ranker train

# Dagster UI
export DAGSTER_HOME=/workspace/gogooku3/gogooku5
PYTHONPATH=data/src dagster dev -m dagster_gogooku5.defs

# GPU monitoring
nvidia-smi

# Git status
git status
```

## Notes

- Be proactive about finding optimization opportunities
- Research latest ML/financial modeling techniques when relevant
- Explain complex decisions clearly
- Ask for clarification when requirements are ambiguous
- Respect separation of concerns (dataset ≠ training)
