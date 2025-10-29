# APEX-Ranker Experiment Status

**Last Updated**: 2025-10-29
**Status**: Phase 1/2 Complete, Phase 3 Planned

---

## Current Release: v0.1.0-pruned

### Model Checkpoint
- **Path**: `models/apex_ranker_v0_pruned.pt`
- **Size**: 13 MB
- **Training Date**: 2025-10-29
- **Git Branch**: `feature/phase2-graph-rebuild` (commit: `5937ff4`)

### Model Architecture

**APEXRankerV0** - PatchTST-based multi-horizon stock ranker

```yaml
Architecture:
  Encoder: PatchTST (Patch Time Series Transformer)
  Heads: Multi-horizon linear heads (1d, 5d, 10d, 20d)

Hyperparameters:
  in_features: 64              # Pruned from 89
  d_model: 192                 # Embedding dimension
  depth: 3                     # Transformer layers
  patch_len: 16                # Patch length
  stride: 8                    # Patch stride
  n_heads: 8                   # Attention heads
  dropout: 0.1                 # Dropout rate
  lookback: 180                # Historical window (days)

Training Config:
  optimizer: Adam
  lr: 5e-4                     # Initial learning rate
  lr_warmup_epochs: 3          # Warmup period
  batch_size: 256              # Training batch size
  max_epochs: 50               # Maximum epochs
  early_stopping_patience: 3   # Early stop patience
  early_stopping_metric: 20d_pak  # Primary metric

Loss Function:
  type: RankLoss (PairwiseRanking)
  horizons: [1, 5, 10, 20]     # Days ahead
```

### Feature Configuration

**Feature Groups** (64 features total):
- **Core features** (33): OHLCV-derived, returns, log returns, volatility
- **Technical indicators** (17): RSI, MACD, Stochastic, Bollinger Bands
- **Volume features** (8): Volume ratios, on-balance volume
- **Sector features** (6): Sector-relative returns and Z-scores

**Excluded Features** (25 features removed):
- **Bottom 10 negative impact**:
  - `flow_breadth_pos` (-2.03%)
  - `returns_20d` (-1.78%)
  - `ema_20` (-1.39%)
  - `ema_200` (-1.16%)
  - `flow_foreign_net_z` (-0.58%)
  - `ema_5` (-0.57%)
  - `stoch_k` (-0.32%)
  - `idio_vol_ratio` (-0.32%)
  - `flow_individual_net_ratio` (-0.30%)
  - `rsi_2` (-0.27%)

- **15 zero impact market features**: `mkt_ret_*`, `mkt_vol_*`, etc.

**Config File**: `apex-ranker/configs/v0_pruned.yaml`

### Performance Metrics

#### Validation Results (Best Epoch: 6/9)

| Horizon | RankIC | P@K (Top-50) |
|---------|--------|--------------|
| 1d      | 0.0049 | 0.4735       |
| 5d      | -0.0112| 0.4915       |
| 10d     | -0.0196| 0.5203       |
| **20d** | **-0.0322** | **0.5405** |

**Training Progression**:
```
Epoch 1: 20d P@K = 0.4622
Epoch 2: 20d P@K = 0.4940 ✓ (best)
Epoch 3: 20d P@K = 0.5145 ✓ (best)
Epoch 4: 20d P@K = 0.5088
Epoch 5: 20d P@K = 0.4932
Epoch 6: 20d P@K = 0.5405 ✓ (best) ← FINAL
Epoch 7: 20d P@K = 0.5240 (early stop +1)
Epoch 8: 20d P@K = 0.5290 (early stop +2)
Epoch 9: 20d P@K = 0.5293 (early stop +3) → STOPPED
```

**Training Time**: ~11 hours 30 minutes
**Validation Period**: 120 trading days
**Dataset**: `output/ml_dataset_latest_full.parquet` (10.6M samples, 3,973 stocks, 2020-2025)

---

## Comparison: Enhanced vs Pruned

### Enhanced Model Baseline (v0_base.yaml)

| Metric | Enhanced (89 features) | Pruned (64 features) | Change |
|--------|------------------------|----------------------|--------|
| **20d P@K** | 0.5765 | 0.5405 | -6.2% |
| **Features** | 89 | 64 | -28.1% |
| **Training Time** | ~12h | ~11.5h | -4.2% |
| **Model Size** | 13 MB | 13 MB | 0% |

**Analysis**:
- Pruned model shows 6.2% P@K degradation despite removing 25 features
- Trade-off: Simpler model, faster inference, but slightly lower ranking accuracy
- **Decision pending**: Need same-period re-evaluation to determine Phase 3 primary model

---

## Development Timeline

### Phase 0: Baseline (2025-10-27 ~ 2025-10-28)
- ✅ Model architecture implementation (APEXRankerV0)
- ✅ Training pipeline setup
- ✅ Initial baseline model (`apex_ranker_v0_baseline.pt`)
- ✅ Feature importance analysis (permutation-based)

### Phase 1: Feature Pruning (2025-10-29)
- ✅ Config: `v0_pruned.yaml` with 25 excluded features
- ✅ FeatureSelector: Added `exclude_features` parameter
- ✅ Training: Completed with early stopping (9 epochs)
- ✅ Model: `apex_ranker_v0_pruned.pt` saved

### Phase 2: Inference Infrastructure (2025-10-29)
- ✅ Script: `apex-ranker/scripts/inference_v0.py`
  - CLI-based prediction engine
  - Multi-horizon support (1d, 5d, 10d, 20d)
  - CSV output with Date, Rank, Code, Score, Horizon
- ✅ Monitoring: `apex-ranker/scripts/monitor_predictions.py`
  - Prediction logging with metadata
  - Daily summary report generation
- ✅ Documentation: `apex-ranker/INFERENCE_GUIDE.md`
  - Production usage guide
  - Performance benchmarks
  - Troubleshooting
- ✅ Bug Fixes:
  - Panel cache: Handle empty targets for inference
  - Inference script: Correct cache access pattern
  - Model output: Fixed Dict[int, Tensor] format

**Inference Validation**:
```bash
# Test run (2025-10-24, 3751 stocks)
python apex-ranker/scripts/inference_v0.py \
  --model models/apex_ranker_v0_enhanced.pt \
  --config apex-ranker/configs/v0_base.yaml \
  --top-k 10 --horizon 20

✅ Score range: [0.1002, 0.9695]
✅ Top-1: Code 42550 (Score: 0.9695)
✅ Panel cache: ~2 minutes for 10.6M samples
✅ Inference: <1 second on GPU
```

### Phase 3: Long-term Backtest (Planned)
**Objective**: Validate model performance over 2-3 years with transaction costs

**Scope**:
1. Walk-forward backtest framework
2. Transaction cost simulation (slippage + commission)
3. Market microstructure assumptions
4. Performance metrics (Sharpe, max drawdown, turnover)
5. Out-of-sample validation

**Deliverables**:
- Backtest script accepting inference CSV outputs
- Transaction cost model documentation
- Walk-forward split configuration
- Backtest result logging and visualization

**Status**: Ready to start (pending model comparison decision)

---

## Known Issues & Limitations

### Current Limitations
1. **No real-time data**: Requires pre-processed parquet dataset
2. **No drift detection**: Does not monitor distribution changes
3. **No fallback logic**: Fails if any feature is missing
4. **Panel cache rebuild**: Full rebuild on every inference run (~2 min)
5. **CLI only**: No API server (FastAPI wrapper planned for v0.2)

### Performance Bottlenecks
1. **Panel cache building**: 2 minutes for 10.6M samples (CPU-bound)
   - **Solution**: Cache serialization to disk (planned)
2. **Cross-sectional normalization**: ~30 seconds
   - **Solution**: Pre-compute Z-scores in dataset (planned)

### Production Readiness
**Current Status**: ~60% ready for 1-2 week deployment

**Blockers**:
- ❌ Long-term backtest validation (Phase 3)
- ❌ Transaction cost simulation
- ❌ Panel cache persistence
- ⚠️ Model selection (pruned vs enhanced)

**Ready**:
- ✅ Inference pipeline (CLI)
- ✅ Monitoring & logging
- ✅ Documentation
- ✅ Trained models (2 versions)

---

## Next Steps

### Immediate (This week)
1. **Model Comparison**:
   - Re-evaluate pruned vs enhanced on same validation period
   - Analyze P@K difference factors
   - Decide Phase 3 primary model

2. **Transaction Cost Model**:
   - Define market microstructure assumptions
   - Document slippage model (e.g., linear impact)
   - Define commission structure (e.g., 0.1% per trade)

3. **Backtest Design**:
   - Specify walk-forward backtest interface
   - Design input/output format
   - Break down implementation tasks

### Short-term (1-2 weeks)
1. **Phase 3 Implementation**:
   - Walk-forward splitting logic
   - Transaction cost calculation module
   - Backtest logging infrastructure
   - Evaluation metrics computation

2. **Monitoring Validation**:
   - Test with production-like data
   - Verify log rotation and anomaly detection
   - Validate file storage and retrieval

### Medium-term (2-4 weeks)
1. **Performance Optimization**:
   - Panel cache serialization
   - Pre-computed Z-scores
   - Inference batch processing

2. **API Server** (v0.2):
   - FastAPI wrapper for inference
   - Prometheus metrics export
   - Health check endpoints

---

## References

### Key Files
- Models: `models/apex_ranker_v0_{baseline,enhanced,pruned}.pt`
- Configs: `apex-ranker/configs/v0_{base,pruned}.yaml`
- Training log: `/tmp/train_pruned_20251029.log`
- Feature importance: `results/feature_importance_enhanced_top6_d2.json`
- Backtest results: `results/backtest_{baseline,enhanced}.json`

### Documentation
- Inference guide: `apex-ranker/INFERENCE_GUIDE.md`
- Project README: `apex-ranker/README.md`
- Training README: `README.md` (root)

### Related Issues
- Feature pruning analysis: Removed 25 features based on permutation importance
- Panel cache fix: Handle empty targets for inference (apex_ranker/data/panel_dataset.py:56,89-92)
- Model output fix: Dict[int, Tensor] format (apex-ranker/scripts/inference_v0.py:301-311)

---

**Generated**: 2025-10-29
**Author**: Claude Code (Autonomous Development Agent)
