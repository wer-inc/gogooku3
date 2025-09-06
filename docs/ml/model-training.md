# 🧠 Model Training and ATFT-GAT-FAN Architecture

This document describes the complete machine learning training pipeline in Gogooku3, focusing on the ATFT-GAT-FAN (Adaptive Temporal Fusion Transformer + Graph Attention Network + Frequency Adaptive Normalization) model.

## 🏗️ Model Architecture Overview

### ATFT-GAT-FAN: Next-Generation Financial Time Series Model

The ATFT-GAT-FAN model combines three powerful components for multi-horizon stock return prediction:

1. **ATFT (Adaptive Temporal Fusion Transformer)**: Time-series attention mechanism
2. **GAT (Graph Attention Network)**: Inter-stock correlation modeling  
3. **FAN (Frequency Adaptive Normalization)**: Dynamic feature scaling

#### Key Specifications
```yaml
Model Configuration:
  Name: ATFT-GAT-FAN
  Parameters: 5.6M
  Target Sharpe: 0.849
  Architecture: Transformer + GNN hybrid
  Framework: PyTorch 2.0+
  Mixed Precision: bfloat16 support
  
Input Shape:
  Sequence Length: 60 days (L=60)
  Feature Dimensions: 145 features  
  Prediction Horizons: [1, 5, 10, 20] days
  
Output:
  Multi-horizon returns: 4 target variables
  Attention weights: Interpretable feature importance
  Graph embeddings: Stock relationship representations
```

## 🔧 Training Pipeline Components

### 1. Safe Training Pipeline (SafeTrainingPipeline)

The integrated 7-step training pipeline ensures data safety and model robustness:

```python
from gogooku3.training import SafeTrainingPipeline

# Initialize pipeline with safety configuration
pipeline = SafeTrainingPipeline(
    experiment_name="production_training",
    data_path="data/raw/large_scale/ml_dataset_full.parquet",
    verbose=True
)

# Execute complete training workflow
results = pipeline.run_pipeline(
    n_splits=5,           # 5-fold Walk-Forward CV
    embargo_days=20,      # 20-day embargo period
    memory_limit_gb=8.0   # Memory constraint
)
```

#### Pipeline Steps (Execution Order)

**Step 1: Data Loading** (`ProductionDatasetV3`)
```python
# Polars-optimized lazy loading
loader = ProductionDatasetV3(
    data_path="data/raw/large_scale/ml_dataset_full.parquet",
    use_lazy_scan=True,
    column_projection=True  # Load only required columns
)

data = loader.load_data()
print(f"Loaded: {data.shape[0]:,} samples × {data.shape[1]} features")
# Output: 606,127 samples × 145 features
```

**Step 2: Feature Engineering** (`QualityFinancialFeaturesGenerator`)
```python
# Generate enhanced quality features (+6 additional)
feature_generator = QualityFinancialFeaturesGenerator(
    use_cross_sectional_quantiles=True,
    sigma_threshold=2.0
)

enhanced_data = feature_generator.generate_quality_features(data)
print(f"Features: {data.shape[1]} → {enhanced_data.shape[1]} (+{enhanced_data.shape[1] - data.shape[1]})")
# Output: Features: 139 → 145 (+6)
```

**Step 3: Cross-Sectional Normalization** (`CrossSectionalNormalizerV2`)
```python
# Daily Z-score normalization without temporal leakage
normalizer = CrossSectionalNormalizerV2(
    cache_stats=True,
    robust_clip=5.0
)

# Fit-transform separation for safety
normalizer.fit(train_data)
train_normalized = normalizer.transform(train_data)
test_normalized = normalizer.transform(test_data)  # Uses train stats only
```

**Step 4: Walk-Forward Splitting** (`WalkForwardSplitterV2`)
```python
# Temporal validation with embargo period
splitter = WalkForwardSplitterV2(
    n_splits=5,
    embargo_days=20,        # Match max prediction horizon
    min_train_days=252      # Minimum 1 year training
)

# Generate chronological splits
for fold, (train_idx, test_idx) in enumerate(splitter.split(data)):
    print(f"Fold {fold}: {len(train_idx):,} train, {len(test_idx):,} test")
```

**Step 5: Baseline Model Training** (`LightGBMFinancialBaseline`)
```python
# Multi-horizon gradient boosting baseline
baseline = LightGBMFinancialBaseline(
    prediction_horizons=[1, 5, 10, 20],
    embargo_days=20,
    normalize_features=True
)

baseline.fit(data)
performance = baseline.evaluate_performance()

for horizon, metrics in performance.items():
    print(f"{horizon}d: IC={metrics['mean_ic']:.3f}, RankIC={metrics['mean_rank_ic']:.3f}")
```

**Step 6: Graph Construction** (`FinancialGraphBuilder`)
```python
# Time-series correlation graph for GAT
graph_builder = FinancialGraphBuilder(
    correlation_window=60,
    include_negative_correlation=True,
    max_edges_per_node=10
)

stock_codes = data['code'].unique()[:50]  # Top 50 by market cap
graph = graph_builder.build_graph(data, stock_codes, date_end="2024-12-31")

print(f"Graph: {graph['n_nodes']} nodes, {graph['n_edges']} edges")
# Output: Graph: 50 nodes, 266 edges
```

**Step 7: ATFT-GAT-FAN Training**
```python
# Load model architecture
from gogooku3.models import ATFTGATFANModel

model = ATFTGATFANModel(
    input_dim=145,
    hidden_dim=256,
    num_heads=8,
    num_layers=4,
    prediction_horizons=[1, 5, 10, 20],
    graph_info=graph
)

# Training configuration
trainer = ATFTTrainer(
    model=model,
    train_data=train_normalized,
    val_data=test_normalized,
    config={
        'epochs': 50,
        'batch_size': 512,
        'learning_rate': 1e-3,
        'mixed_precision': True,
        'gradient_clip': 1.0
    }
)

# Execute training
training_results = trainer.fit()
```

## 🎯 Model Components Deep Dive

### ATFT: Adaptive Temporal Fusion Transformer

#### Architecture Details
```python
class ATFTCore(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, sequence_length):
        super().__init__()
        self.sequence_length = sequence_length  # L=60
        
        # Multi-head attention layers
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # LSTM backbone for sequential processing
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Feature fusion layers
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
```

#### Attention Mechanism
- **Query**: Current time step features
- **Key/Value**: Historical sequence (60 days)
- **Output**: Temporally fused representations
- **Innovation**: Adaptive attention weights based on market regime

### GAT: Graph Attention Network

#### Stock Correlation Graph
```python
class FinancialGraphAttention(nn.Module):
    def __init__(self, in_features, out_features, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
        # Attention coefficients for each head
        self.W = nn.Linear(in_features, out_features * num_heads)
        self.a = nn.Linear(2 * out_features, 1)
        
        # Edge weights based on correlation strength
        self.edge_importance = nn.Parameter(torch.randn(1))
        
    def forward(self, features, adjacency_matrix):
        # Multi-head attention over graph neighbors
        attention_weights = self.compute_attention(features, adjacency_matrix)
        
        # Aggregate neighbor information
        aggregated_features = self.aggregate_neighbors(features, attention_weights)
        
        return aggregated_features, attention_weights
```

#### Graph Construction Process
1. **Correlation Calculation**: Rolling 60-day return correlations
2. **Edge Filtering**: Keep top-K correlations per stock (K=10)
3. **Dynamic Updates**: Rebuild graph monthly for regime changes
4. **Negative Correlations**: Include for diversification insights

### FAN: Frequency Adaptive Normalization

#### Dynamic Feature Scaling
```python
class FrequencyAdaptiveNorm(nn.Module):
    def __init__(self, num_features, num_frequencies=5):
        super().__init__()
        self.num_frequencies = num_frequencies
        
        # Learned frequency decomposition
        self.frequency_weights = nn.Parameter(torch.randn(num_frequencies, num_features))
        
        # Adaptive scaling parameters
        self.scale_network = nn.Sequential(
            nn.Linear(num_features, num_features // 2),
            nn.ReLU(),
            nn.Linear(num_features // 2, num_features),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Decompose into frequency components
        frequency_components = self.fft_decompose(x)
        
        # Apply adaptive scaling
        scaling_factors = self.scale_network(x.mean(dim=1))
        
        # Reconstruct with scaled components
        normalized_x = self.fft_reconstruct(frequency_components * scaling_factors)
        
        return normalized_x
```

## 📊 Training Configuration

### Hydra Configuration Files

#### Main Training Config (`configs/model/atft/train.yaml`)
```yaml
# ATFT-GAT-FAN Training Configuration
model:
  name: "atft_gat_fan"
  input_dim: 145
  hidden_dim: 256
  num_heads: 8
  num_layers: 4
  dropout: 0.3
  prediction_horizons: [1, 5, 10, 20]

training:
  epochs: 50
  batch_size: 512
  learning_rate: 1e-3
  weight_decay: 1e-4
  gradient_clip_val: 1.0
  mixed_precision: true
  
scheduler:
  type: "cosine_annealing"
  T_max: 50
  eta_min: 1e-6

loss:
  type: "multi_horizon_mse"
  horizon_weights:
    1: 1.0    # 1-day most important
    5: 0.7    # 5-day important
    10: 0.5   # 10-day moderate
    20: 0.5   # 20-day moderate

early_stopping:
  monitor: "val/rank_ic_5d"
  patience: 10
  min_delta: 0.001
```

#### Data Configuration (`configs/data/jpx_safe.yaml`)
```yaml
# Safe data processing configuration
dataset:
  name: "jpx_production"
  data_path: "data/raw/large_scale/ml_dataset_full.parquet"
  
time_series:
  sequence_length: 60
  prediction_horizons: [1, 5, 10, 20]
  drop_historical_columns: true
  use_tensor_sequences: true

normalization:
  cross_sectional:
    enabled: true
    method: "polars_v2"
    robust_clip: 5.0
  batch_norm:
    enabled: false  # Disabled for safety

split:
  method: "walk_forward"
  n_splits: 5
  embargo_days: 20
  min_train_days: 252

performance:
  use_polars_lazy: true
  memory_limit_gb: 8.0
  column_projection: true
```

## 🚀 Training Execution Commands

### Method 1: Modern CLI (Recommended)
```bash
cd /home/ubuntu/gogooku3-standalone

# Install package in development mode
pip install -e .

# Train with production configuration
gogooku3 train --config configs/model/atft/train.yaml

# Quick training (2 epochs)
gogooku3 train --config configs/model/atft/train.yaml --epochs 2

# Memory-constrained training
gogooku3 train --config configs/model/atft/train.yaml --memory-limit 4
```

### Method 2: Python API (Programmatic)
```python
# Complete training pipeline
from gogooku3.training import SafeTrainingPipeline
from gogooku3 import settings

pipeline = SafeTrainingPipeline(
    experiment_name="atft_production",
    data_path=settings.data_dir / "raw/large_scale/ml_dataset_full.parquet",
    model_config="configs/model/atft/train.yaml",
    verbose=True
)

# Execute full pipeline
results = pipeline.run_pipeline(
    n_splits=5,
    embargo_days=20,
    memory_limit_gb=8.0
)

print(f"Training completed in {results['total_duration']:.1f}s")
print(f"Best Sharpe: {results['best_sharpe']:.3f}")
print(f"Model saved to: {results['model_path']}")
```

### Method 3: Legacy Scripts (Compatibility)
```bash
# Legacy ATFT training (with deprecation warnings)
python scripts/train_atft.py --config-path configs/model/atft --config-name train

# Integrated ML pipeline (legacy)
python scripts/integrated_ml_training_pipeline.py --verbose --n-splits 5

# Safe training pipeline (legacy)
python scripts/run_safe_training.py --memory-limit 8 --n-splits 5
```

## 📈 Performance Monitoring and Evaluation

### Training Metrics

#### Primary Financial Metrics
```python
# Evaluation during training
evaluation_metrics = {
    'ic_1d': information_coefficient(predictions_1d, returns_1d),
    'ic_5d': information_coefficient(predictions_5d, returns_5d),
    'rank_ic_1d': rank_information_coefficient(predictions_1d, returns_1d),
    'rank_ic_5d': rank_information_coefficient(predictions_5d, returns_5d),
    'sharpe_ratio': compute_sharpe_ratio(predictions, returns),
    'max_drawdown': compute_max_drawdown(predictions, returns)
}
```

#### Model Performance Targets
```yaml
Production Targets:
  Sharpe Ratio: 0.849 (primary target)
  IC (1-day): >0.05
  RankIC (5-day): >0.10
  Max Drawdown: <20%
  
Training Speed:
  10 epochs: 45 minutes (A100 GPU)
  Full 50 epochs: 3.75 hours
  Memory usage: <8GB
  
Baseline Comparison:
  LightGBM IC: 0.032
  ATFT-GAT-FAN Target IC: >0.05
  Performance Improvement: 56%+
```

### Real-Time Monitoring
```python
# Training progress callback
class FinancialMetricsCallback:
    def on_epoch_end(self, trainer, pl_module):
        # Compute financial metrics
        val_predictions = trainer.predict(pl_module, val_dataloader)
        
        ic_1d = self.compute_ic(val_predictions[:, 0], val_returns[:, 0])
        ic_5d = self.compute_ic(val_predictions[:, 1], val_returns[:, 1])
        
        # Log to tensorboard/wandb
        pl_module.log("val/ic_1d", ic_1d)
        pl_module.log("val/ic_5d", ic_5d)
        
        # Early stopping based on financial metrics
        if ic_5d > self.best_ic_5d:
            self.best_ic_5d = ic_5d
            trainer.save_checkpoint(f"best_model_ic_{ic_5d:.3f}.ckpt")
```

## 🔧 Advanced Training Features

### Mixed Precision Training
```python
# Automatic mixed precision for A100/H100
trainer = pl.Trainer(
    precision="bf16-mixed",  # bfloat16 mixed precision
    accelerator="gpu",
    devices=1,
    max_epochs=50,
    gradient_clip_val=1.0
)
```

### Graph Dynamic Updates
```python
# Update correlation graph during training
class DynamicGraphCallback:
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % 10 == 0:  # Update every 10 epochs
            # Rebuild graph with latest correlations
            new_graph = self.graph_builder.build_graph(
                recent_data, 
                date_window=60
            )
            pl_module.update_graph(new_graph)
```

### Multi-Horizon Loss Function
```python
class MultiHorizonFinancialLoss(nn.Module):
    def __init__(self, horizon_weights={1: 1.0, 5: 0.7, 10: 0.5, 20: 0.5}):
        super().__init__()
        self.horizon_weights = horizon_weights
        
    def forward(self, predictions, targets):
        """
        predictions: [batch_size, num_horizons]
        targets: [batch_size, num_horizons]
        """
        total_loss = 0
        
        for i, horizon in enumerate([1, 5, 10, 20]):
            horizon_loss = F.mse_loss(predictions[:, i], targets[:, i])
            weighted_loss = horizon_loss * self.horizon_weights[horizon]
            total_loss += weighted_loss
            
        return total_loss / len(self.horizon_weights)
```

## 🐛 Troubleshooting Training Issues

### Common Problems and Solutions

#### 1. Memory Issues
```bash
# Problem: OOM during training
# Solution: Reduce batch size and enable gradient checkpointing

# Memory-efficient training
gogooku3 train \
  --config configs/model/atft/train.yaml \
  --batch-size 256 \
  --gradient-checkpointing \
  --memory-limit 6
```

#### 2. Slow Convergence
```python
# Problem: Training loss plateaus
# Solution: Learning rate scheduling and warmup

scheduler_config = {
    "scheduler": torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=1e-2,
        total_steps=num_training_steps
    ),
    "interval": "step",
    "frequency": 1
}
```

#### 3. Graph Construction Errors
```python
# Problem: Insufficient correlations for graph
# Solution: Lower correlation threshold

graph_builder = FinancialGraphBuilder(
    correlation_threshold=0.3,  # Lower from 0.5
    min_edges_per_node=3,       # Reduce minimum
    include_negative_correlation=True
)
```

## 🔗 Related Documentation

- [Safety Guardrails](safety-guardrails.md) - Data leakage prevention and validation
- [Data Quality](data-quality.md) - Data validation and checks  
- [Data Pipeline](../architecture/data-pipeline.md) - Technical data processing architecture
- [Contributing Guide](../development/contributing.md) - Development and testing procedures

---

**🧠 Key Insight**: The ATFT-GAT-FAN model achieves superior performance by combining temporal attention (ATFT), inter-stock relationships (GAT), and adaptive feature scaling (FAN), while maintaining strict data safety through the SafeTrainingPipeline framework.
