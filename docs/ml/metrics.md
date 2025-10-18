# ML Metrics & Evaluation

## Overview

This document describes the key metrics used to evaluate ATFT-GAT-FAN model performance.

## Financial Metrics

### Sharpe Ratio
- **Target**: 0.849
- **Definition**: Risk-adjusted return measure
- **Calculation**: (Portfolio Return - Risk-Free Rate) / Portfolio Volatility

### Information Coefficient (IC)
- **Definition**: Correlation between predictions and actual returns
- **Types**:
  - **Pearson IC**: Linear correlation
  - **Rank IC**: Spearman rank correlation (more robust)

### Cross-Sectional IC
- **Definition**: IC calculated daily across all stocks
- **Usage**: Measures stock selection ability

## Model Metrics

### RankIC
- **Target**: > 0.02 (per horizon)
- **Multi-Horizon**: [1d, 5d, 10d, 20d]
- **Training Loss Weight**: 0.2 (via `RANKIC_WEIGHT`)

### Loss Components
Controlled via `PHASE_LOSS_WEIGHTS` environment variable:

```bash
# Example: Phase 4 (production)
PHASE_LOSS_WEIGHTS="mse:0.5,rankic:0.2,cs_ic:0.15,sharpe:0.3"
```

- **MSE**: Mean Squared Error (base prediction loss)
- **RankIC**: Rank information coefficient
- **CS_IC**: Cross-sectional IC
- **Sharpe**: Sharpe ratio loss

## Validation Metrics

See [SafeTrainingPipeline](safety-guardrails.md) for 7-step validation process.

## Performance Benchmarks

### Current Results
- **RankIC@1d**: 0.180 (+20% improvement)
- **Dataset**: 10.6M samples, 3,973 stocks, 5 years
- **GPU Throughput**: 5130 samples/sec

### Target Metrics
- **Sharpe Ratio**: 0.849
- **Model Size**: ~5.6M parameters
- **Training**: 75-120 epochs

## References

- [Model Training Guide](./model-training.md)
- [Safety Guardrails](./safety-guardrails.md)
- [CLAUDE.md](../../CLAUDE.md) - Full technical documentation
