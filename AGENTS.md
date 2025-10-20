# Codex Agent Instructions - ATFT-GAT-FAN Project

You are an autonomous AI developer working on a Japanese stock market prediction system using Graph Attention Networks (GAT) and deep learning.

## Project Context

**Main Goal**: Build a production-ready financial ML system for Japanese stock market prediction
**Tech Stack**: PyTorch, Graph Neural Networks, Time Series Analysis, Financial Data Processing
**Hardware**: NVIDIA A100 80GB GPU, 24-core CPU, 216GB RAM

## Key Project Files

- `scripts/integrated_ml_training_pipeline.py` - Main training pipeline
- `scripts/pipelines/run_full_dataset.py` - Dataset builder
- `CLAUDE.md` - Comprehensive project documentation
- `tools/project-health-check.sh` - Health diagnostics

## Development Guidelines

1. **Always Read Before Editing**: Use `codex read <file>` to understand context
2. **Test After Changes**: Run health checks and unit tests
3. **Document Changes**: Update relevant documentation
4. **Optimize for GPU**: Leverage A100's 80GB memory for large batch sizes
5. **Financial Data Sensitivity**: Handle market data with proper validation

## Autonomous Workflow

When working autonomously:
1. Analyze health check reports thoroughly
2. Create detailed todo lists for complex tasks
3. Fix critical issues first (P0 → P1 → P2)
4. Run verification after each major change
5. Document reasoning for non-obvious decisions

## Code Quality Standards

- Type hints for all functions
- Docstrings for public APIs
- Unit tests for core logic
- Memory-efficient data processing
- GPU utilization monitoring

## Useful Commands

```bash
# Health check
tools/project-health-check.sh

# Run training
python scripts/integrated_ml_training_pipeline.py

# Dataset building
python scripts/pipelines/run_full_dataset.py

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
