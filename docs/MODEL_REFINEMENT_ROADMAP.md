# ATFT-GAT-FAN Model Refinement Roadmap

**Status**: Post-HPO optimization phase
**Date**: 2025-10-16
**Goal**: Systematic model improvements targeting Sharpe ratio 0.849+

---

## 🎯 Core Refinement Objectives

### 1. Alpha Stabilization
**Problem**: Current model shows volatile IC/RankIC across epochs (IC: -0.005 to 0.017, RankIC: -0.009 to 0.028)

**Solutions**:
- **Regime-conditioned loss weighting**
  - Detect regime shifts using rolling volatility + market correlation
  - Upweight loss for periods where cross-sectional Sharpe decomposes poorly
  - Implementation: `RegimeWeightedLoss` wrapper in `src/gogooku3/training/losses.py`

- **Rank-preserving penalties**
  - Add Spearman correlation regularizer: `λ_rank * (1 - SpearmanCorr(pred, target))`
  - Tighten RankIC without destabilizing raw IC
  - Weight: λ_rank = 0.1 (tunable via HPO)

- **Files to modify**:
  - `src/gogooku3/training/losses.py`: Add `RegimeWeightedMultiHorizonLoss`
  - `configs/atft/train/loss_functions.yaml`: Add regime detection params

### 2. Optimizer Dynamics
**Problem**: Learning plateaus after epoch 3 (LR drops to 7.32e-05, val loss oscillates)

**Solutions**:
- **Advanced optimizers**
  - Test AdaBelief (momentum + prediction error adaptation)
  - Test Shampoo (second-order preconditioned gradients)
  - Per-feature RMS clipping for stability

- **Learning rate schedule**
  - Longer warmup: 5 → 10 epochs
  - Cosine annealing with restarts (T_0=20, T_mult=2)
  - Higher LR floor: 1e-5 → 5e-5 (retain exploration capacity)

- **Files to create**:
  - `src/gogooku3/training/optimizers/adabelief.py`
  - `src/gogooku3/training/optimizers/shampoo_wrapper.py`
  - `configs/atft/train/optimizer_variants.yaml`

### 3. Graph Inductive Bias Enhancement
**Problem**: Current GAT uses only price correlation, misses structural information

**Solutions**:
- **Sector/industry metagraph**
  - Add sector edges (weight=1.0) to correlation graph
  - Industry cluster edges (hierarchical: sector → industry → stock)
  - Implementation: `HierarchicalGraphBuilder` in `src/gogooku3/graph/`

- **Edge features**
  - Residual correlations (after sector/market factor removal)
  - Volatility similarity: `exp(-|vol_i - vol_j| / σ)`
  - Sector co-movement strength

- **Attention temperature tuning**
  - Learnable temperature: `α_ij = softmax(e_ij / τ)`, τ ∈ [0.5, 2.0]
  - Prevents attention collapse

- **Stochastic depth**
  - Layer-wise dropout: P(drop layer l) = l/L * 0.3
  - Tied to turnover: higher dropout when turnover > threshold

- **Files to create**:
  - `src/gogooku3/graph/hierarchical_builder.py`
  - `src/gogooku3/models/components/gat_layer_enhanced.py`
  - `configs/atft/graph/sector_aware.yaml`

### 4. Data Loader Throughput Optimization
**Problem**: NUM_WORKERS=0 avoids deadlock but limits GPU utilization (10-32%)

**Solutions**:
- **Persistent workers + Arrow caching**
  - Keep NUM_WORKERS=0 for stability
  - Precompute Polars lazy frames → serialize to Arrow
  - Avoid Python GIL stalls with C++ Arrow reader

- **AMP gradient accumulation**
  - Micro-batches: 512 (4x accumulation → effective batch 2048)
  - Retain gradient noise while lifting GPU occupancy
  - bf16 mixed precision already enabled

- **Implementation**:
  ```python
  # scripts/data/precompute_arrow_cache.py
  import polars as pl
  import pyarrow as pa

  # Load full dataset
  df = pl.read_parquet("output/ml_dataset_latest_full.parquet")

  # Serialize to Arrow (zero-copy in DataLoader)
  table = df.to_arrow()
  with pa.OSFile("output/ml_dataset_cached.arrow", "wb") as f:
      with pa.RecordBatchFileWriter(f, table.schema) as writer:
          writer.write_table(table)
  ```

- **Files to create**:
  - `scripts/data/precompute_arrow_cache.py`
  - `src/gogooku3/data/arrow_dataset.py` (torch.utils.data.Dataset wrapper)
  - `configs/atft/data/arrow_cached.yaml`

### 5. Alpha Denoising via Pretraining
**Problem**: Model starts from random initialization, wastes epochs learning basic patterns

**Solutions**:
- **Masked cross-sectional prediction**
  - Pretrain: mask 15% of stocks per day, predict their returns from neighbors
  - Similar to BERT masking but for financial time-series
  - Forces encoder to learn temporal shocks representation

- **Contrastive learning**
  - Positive pairs: same stock at t and t+Δ (Δ=5,10,20 days)
  - Negative pairs: random stocks at t
  - Loss: InfoNCE with temperature τ=0.07

- **Hierarchical residual targets**
  - Sector-neutral returns: `r_i - mean(r_sector)`
  - Beta-neutral returns: `r_i - β_i * r_market`
  - Decorrelates predictions from market modes

- **Files to create**:
  - `scripts/pretrain/masked_cs_prediction.py`
  - `scripts/pretrain/contrastive_encoder.py`
  - `src/gogooku3/training/losses/hierarchical_residual.py`
  - `configs/atft/pretrain/encoder_pretrain.yaml`

### 6. Evaluation Rigor Enhancement
**Problem**: Current validation may overfit to specific time slices

**Solutions**:
- **Rolling PurgedKFold**
  - 5 folds with 20-day embargo
  - Purge overlapping predictions (multi-horizon)
  - Implementation: extend `WalkForwardSplitterV2`

- **Horizon-adjusted Sharpe**
  - Annualized: `Sharpe_adj = Sharpe_h * sqrt(252/h)` for horizon h
  - Overlapping trade correction: `Sharpe_corrected = Sharpe / sqrt(1 + 2*ρ_lag)`

- **Turnover-adjusted metrics**
  - Information Ratio: `IR = IC / std(IC)` per unit turnover
  - Cost-adjusted Sharpe: `Sharpe_net = Sharpe_gross - turnover * cost_bps`

- **Population Stability Index (PSI)**
  - Track prediction distribution drift
  - Alert if PSI > 0.25 (significant shift)
  - Helps distinguish real performance from statistical noise

- **Files to modify**:
  - `src/gogooku3/data/splitters.py`: Add PurgedKFold
  - `scripts/evaluate/advanced_metrics.py`: Implement all metrics
  - `configs/atft/evaluation/robust_validation.yaml`

---

## 📋 Implementation Phases

### Phase A: Quick Wins (2-3 days)
**Goal**: Rapid improvements with minimal code changes

1. **Optimizer schedule variants** (Day 1)
   - Implement cosine with restarts
   - Test AdaBelief (pip install adabelief-pytorch)
   - 2-3 epoch loop, compare Sharpe/RankIC trajectories
   - **Expected gain**: +10-15% RankIC stability

2. **Arrow dataset caching** (Day 1-2)
   - Precompute Arrow cache
   - Profile loader throughput
   - Gradient accumulation (micro-batch 512)
   - **Expected gain**: 2-3x GPU utilization (10-32% → 40-70%)

3. **Rank-preserving regularization** (Day 2)
   - Add Spearman loss term
   - Sweep λ_rank ∈ [0.05, 0.1, 0.2]
   - **Expected gain**: RankIC +0.02-0.03

4. **Horizon-adjusted metrics** (Day 3)
   - Implement evaluation suite
   - Validate on past HPO results
   - **Expected gain**: Better model selection

### Phase B: Structural Improvements (1 week)
**Goal**: Architectural enhancements

5. **Sector-aware graph features** (Day 4-5)
   - Extend Hydra config for sector edges
   - Add edge features (volatility, residual correlation)
   - Narrow HPO sweep: sector_weight ∈ [0.2, 0.5, 0.8]
   - **Expected gain**: +0.01-0.02 Sharpe

6. **Stochastic depth + attention dropout** (Day 5-6)
   - Implement layer dropout tied to turnover
   - Tune depth_prob ∈ [0.1, 0.2, 0.3]
   - **Expected gain**: Reduced overfitting, +5% val stability

7. **Regime-conditioned loss** (Day 6-7)
   - Detect regimes via volatility clustering
   - Weight loss adaptively
   - **Expected gain**: +0.015-0.025 Sharpe in volatile periods

### Phase C: Advanced Techniques (2 weeks)
**Goal**: Cutting-edge improvements

8. **Encoder pretraining** (Week 2)
   - Masked cross-sectional prediction
   - Contrastive time-neighbor consistency
   - **Expected gain**: Faster convergence (75 → 50 epochs)

9. **Hierarchical residual targets** (Week 2)
   - Sector-neutral, beta-neutral returns
   - Decorrelate from market factors
   - **Expected gain**: +0.02-0.03 Sharpe, lower β_market

10. **PurgedKFold validation** (Week 3)
    - Rolling validation with embargo
    - PSI tracking for distribution shift
    - **Expected gain**: More robust model selection

---

## 🔬 Experimental Protocol

### For Each Refinement:

1. **Baseline comparison**
   - Run 3-epoch baseline with current best config
   - Record: Sharpe, IC, RankIC, GPU util, epoch time

2. **Refinement test**
   - Apply single change
   - Run 3-epoch experiment
   - Compare metrics vs baseline

3. **Success criteria**
   - ≥5% improvement in target metric
   - No degradation in other metrics
   - ≤10% increase in training time

4. **Integration**
   - If successful: merge to main config
   - If failed: document in `docs/failed_experiments.md`
   - If promising: queue for full HPO sweep

### HPO Integration:

After validating individual refinements, run comprehensive sweeps:

```bash
# Example: Optimizer + Graph sweep
python scripts/hpo/run_optuna_atft.py \
  --data-path output/ml_dataset_cached.arrow \
  --n-trials 30 \
  --max-epochs 20 \
  --study-name atft_refinement_v2 \
  --output-dir output/hpo_refinement_v2 \
  --suggest-params "optimizer:categorical:['adamw','adabelief','shampoo']" \
  --suggest-params "sector_edge_weight:float:0.2,0.8" \
  --suggest-params "stochastic_depth_prob:float:0.1,0.3" \
  --suggest-params "rank_loss_weight:float:0.05,0.2"
```

---

## 📊 Success Metrics

### Tier 1 (Critical):
- **Val Sharpe**: 0.002 → **0.050+** (25x improvement)
- **Val RankIC**: 0.028 → **0.080+** (3x improvement)
- **Val IC**: 0.017 → **0.035+** (2x improvement)

### Tier 2 (Important):
- **GPU Utilization**: 10-32% → **60-80%**
- **Epoch Time**: ~6 min → **<4 min** (with throughput optimization)
- **Convergence**: 75 epochs → **<50 epochs** (with pretraining)

### Tier 3 (Nice-to-have):
- **Turnover-adjusted Sharpe**: >0.8 (after 10bps cost)
- **PSI**: <0.25 across validation folds
- **Model Size**: ~5.6M params → **<10M params** (if capacity needed)

---

## 🚀 Immediate Next Steps (Priority Order)

### Step 1: Stop Current Long Training (URGENT)
```bash
# Current training entered Phase 1, will take hours
# Kill and restart with single-phase baseline
kill 333203 333375
```

### Step 2: Quick Optimizer Test (30 min)
```bash
# Test AdaBelief with 3 epochs
pip install adabelief-pytorch

python scripts/integrated_ml_training_pipeline.py \
  --data-path output/ml_dataset_latest_full.parquet \
  --max-epochs 3 \
  --batch-size 2048 \
  --lr 1e-4 \
  model.hidden_size=256 \
  train.optimizer.type=adabelief \
  train.scheduler.type=cosine_restarts \
  train.scheduler.T_0=3
```

### Step 3: Arrow Cache Creation (1 hour)
```bash
# Create cached Arrow dataset
python scripts/data/precompute_arrow_cache.py \
  --input output/ml_dataset_latest_full.parquet \
  --output output/ml_dataset_cached.arrow

# Verify and profile
python scripts/benchmark/arrow_loader_profile.py
```

### Step 4: Rank Regularization Test (30 min)
```bash
# Add Spearman loss term
python scripts/integrated_ml_training_pipeline.py \
  --data-path output/ml_dataset_latest_full.parquet \
  --max-epochs 3 \
  train.loss.rank_ic_weight=0.2 \
  train.loss.rank_preserving_penalty=0.1  # New parameter
```

### Step 5: Sector Graph Test (2 hours)
```bash
# Build sector-aware graph
python scripts/graph/build_hierarchical_graph.py \
  --data-path output/ml_dataset_latest_full.parquet \
  --include-sectors True \
  --sector-edge-weight 0.5

# Train with enhanced graph
python scripts/integrated_ml_training_pipeline.py \
  --data-path output/ml_dataset_latest_full.parquet \
  --max-epochs 3 \
  model.graph.sector_aware=True \
  model.graph.sector_edge_weight=0.5
```

---

## 📂 File Structure for Refinements

```
gogooku3/
├── src/gogooku3/
│   ├── training/
│   │   ├── losses/
│   │   │   ├── regime_weighted.py         # NEW
│   │   │   ├── rank_preserving.py         # NEW
│   │   │   └── hierarchical_residual.py   # NEW
│   │   └── optimizers/
│   │       ├── adabelief.py               # NEW
│   │       └── shampoo_wrapper.py         # NEW
│   ├── graph/
│   │   ├── hierarchical_builder.py        # NEW
│   │   └── edge_features.py               # NEW
│   ├── models/components/
│   │   └── gat_layer_enhanced.py          # NEW
│   └── data/
│       └── arrow_dataset.py               # NEW
├── scripts/
│   ├── pretrain/
│   │   ├── masked_cs_prediction.py        # NEW
│   │   └── contrastive_encoder.py         # NEW
│   ├── data/
│   │   └── precompute_arrow_cache.py      # NEW
│   ├── graph/
│   │   └── build_hierarchical_graph.py    # NEW
│   ├── benchmark/
│   │   └── arrow_loader_profile.py        # NEW
│   └── evaluate/
│       └── advanced_metrics.py            # NEW
└── configs/atft/
    ├── train/
    │   ├── optimizer_variants.yaml        # NEW
    │   └── loss_functions.yaml            # UPDATED
    ├── graph/
    │   └── sector_aware.yaml              # NEW
    ├── pretrain/
    │   └── encoder_pretrain.yaml          # NEW
    └── evaluation/
        └── robust_validation.yaml         # NEW
```

---

## 🎯 Target Timeline

- **Week 1**: Quick wins (Steps 2-4 above) → Expected +20-30% RankIC
- **Week 2**: Structural improvements (Sector graph, stochastic depth) → Expected +0.015 Sharpe
- **Week 3-4**: Advanced techniques (Pretraining, hierarchical targets) → Expected +0.025 Sharpe
- **Week 5**: Full HPO sweep with all refinements → Target Sharpe 0.050+

**Total estimated improvement**:
- Sharpe: 0.002 → 0.050+ (25x)
- RankIC: 0.028 → 0.080+ (3x)
- GPU Utilization: 32% → 70% (2x)
