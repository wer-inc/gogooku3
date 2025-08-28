# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Gogooku3 is a standalone next-generation MLOps batch processing system for Japanese stock market analysis. It's a complete financial ML pipeline that integrates JQuants API data processing, technical indicator calculation, Graph Attention Network (GAT) training, and feature store management.

## Architecture

### Core Components

**Data Processing Pipeline**:
- **JQuants API Integration**: Asynchronous data fetching with 150 parallel connections
- **Technical Indicators**: 62+ optimized features (from 713 original indicators)
- **ATFT-GAT-FAN Model**: Adaptive Temporal Fusion Transformer with Graph Attention and Frequency Adaptive Normalization
  - **TFT Core**: LSTM-based temporal fusion with multi-head attention
  - **Adaptive Normalization**: FAN (Frequency Adaptive) + SAN (Slice Adaptive) for dynamic feature scaling
  - **Graph Attention**: Dynamic correlation-based graph construction with GAT layers
  - **Multi-horizon Prediction**: 1d, 2d, 3d, 5d, 10d forecasting with quantile regression
- **Feature Store**: Feast-based feature management with Redis online store

**MLOps Infrastructure**:
- **Dagster**: Orchestration with 8 assets, 7 jobs, 6 schedules, 7 sensors
- **MLflow**: Experiment tracking and model registry
- **Docker Compose**: Multi-service deployment (12+ services)
- **Monitoring**: Grafana + Prometheus + custom metrics

**Storage Layer**:
- **MinIO**: S3-compatible object storage
- **ClickHouse**: OLAP database for analytics
- **Redis**: Caching and online feature store
- **PostgreSQL**: Metadata store for MLflow, Dagster, Feast

## Commands

### Development Environment

```bash
# Environment setup
make setup                    # Create Python venv and install dependencies
cp .env.example .env         # Configure environment variables

# Docker services
make docker-up               # Start all services (MinIO, ClickHouse, etc.)
make docker-down             # Stop all services
make docker-logs             # View logs from all services

# Development
make dev                     # Setup environment and start services
make clean                   # Clean up environment and containers
```

### Data Pipeline Execution

```bash
# Core pipeline
python scripts/pipelines/run_pipeline.py --jquants    # Full JQuants pipeline
python scripts/pipelines/run_pipeline.py --stocks 100 --days 300  # Custom settings

# Individual components
python scripts/core/ml_dataset_builder.py             # Build ML dataset
python scripts/corporate_actions/adjust.py            # Corporate action adjustments
python scripts/quality/price_checks.py                # Data quality validation
```

### ATFT-GAT-FAN ML Training

```bash
# ML Training (requires processed data in output/atft_data/)
make smoke                   # 1-epoch smoke test
make train-cv                # Full training with cross-validation  
make infer                   # Run inference

# Integrated training pipeline
python scripts/integrated_ml_training_pipeline.py
python scripts/train_atft_wrapper.py
bash scripts/complete_atft_training.sh

# Phase Training (Hydra-based structured learning)
python scripts/train.py mode=debug data=jpx_parquet           # Debug mode (fast iteration)
python scripts/train.py phase=baseline                        # Phase 0: Baseline TFT only
python scripts/train.py phase=adaptive_norm                   # Phase 1: + Adaptive normalization
python scripts/train.py phase=gat                             # Phase 2: + Graph attention
python scripts/train.py phase=augmentation                    # Phase 3: + Data augmentation
python scripts/train.py mode=full data=jpx_parquet            # Full training (all phases)

# Data validation and preprocessing
python scripts/validate_data.py                              # Validate Parquet data quality
python scripts/preprocess_data.py                            # Preprocess large-scale data
```

### Dagster Orchestration

```bash
# Dagster UI
make run                     # Start Dagster webserver (localhost:3001)

# CLI execution
dagster asset materialize --select price_data_asset
dagster job execute --job daily_feature_pipeline
```

### Testing

```bash
make test                    # Run all tests
make test-unit               # Unit tests only
make test-integration        # Integration tests only

# Individual test files
pytest tests/unit/test_ml_dataset_builder.py -v
pytest tests/integration/ -v --cov=batch
```

## Project Structure

```
gogooku3-standalone/
├── src/                     # Core ML source code
│   ├── data/               # Data loaders, scalers, validation
│   ├── graph/              # Graph construction (dynamic KNN)
│   ├── models/             # ATFT-GAT-FAN architecture
│   ├── training/           # Training loops and utilities
│   └── utils/              # Config validation, metrics
├── scripts/                # Execution scripts and orchestration  
│   ├── orchestration/      # Dagster assets, jobs, schedules
│   ├── pipelines/          # Data pipeline runners
│   ├── core/               # ML dataset builder
│   ├── feature_store/      # Feast definitions
│   ├── mlflow/             # MLflow integration
│   ├── quality/            # Data quality checks
│   └── corporate_actions/  # Stock split/dividend adjustments
├── configs/                # Configuration files
│   ├── atft/              # ATFT model configurations
│   ├── docker/            # Docker service configs
│   └── dagster/           # Dagster workspace settings
├── output/                 # Processing outputs and datasets
├── models/                 # Saved ML models
└── tests/                  # Test suites
```

## Key Configuration Files

### Environment Variables (`.env`)
```bash
# JQuants API
JQUANTS_AUTH_EMAIL=your_email@example.com
JQUANTS_AUTH_PASSWORD=your_password

# Resource limits
MAX_PARALLEL_WORKERS=200     # Parallel processing limit
JQUANTS_MAX_CONCURRENT=150   # API connection limit
```

### Hydra Configuration System

The system uses Hydra + OmegaConf for hierarchical configuration management:

**Main Config** (`configs/config.yaml`):
- Global project settings, hardware config, logging setup

**Data Config** (`configs/data/jpx_parquet.yaml`):
- Large-scale Parquet data processing (17GB, 195 files)
- Feature definitions (306 total features: 46 basic + 260 historical)
- Time series parameters (sequence_length=20, prediction_horizons=[1,2,3,5,10])
- Preprocessing pipelines (scaling, normalization, graph construction)

**Model Config** (`configs/model/atft_gat_fan_v1.yaml`):
- ATFT architecture (hidden_size=64, optimized for large data)
- TFT settings (LSTM layers, attention heads, variable selection)
- Adaptive normalization (FAN window_sizes=[5,10,20], SAN slices=3)
- GAT configuration (2 layers, dynamic k-NN graph, correlation-based edges)

**Training Config** (`configs/train/adaptive.yaml`):
- Phase training settings (baseline→adaptive_norm→gat→augmentation)
- Batch sizes (train=64, val=128 - optimized for memory efficiency)
- Multi-horizon loss functions with Sharpe ratio optimization
- Early stopping and checkpoint management

## Web Services Access

| Service | URL | Credentials |
|---------|-----|------------|
| Dagster UI | http://localhost:3001 | None |
| MLflow UI | http://localhost:5000 | None |
| Grafana | http://localhost:3000 | admin/gogooku123 |
| MinIO Console | http://localhost:9001 | minioadmin/minioadmin123 |
| ClickHouse | http://localhost:8123 | default/gogooku123 |
| Feast UI | http://localhost:8888 | None |
| Prometheus | http://localhost:9090 | None |

## Development Workflows

### Adding New Features
1. Check existing patterns in `src/` and `scripts/` directories
2. Follow modular ETL design principles (see `docs/specifications/MODULAR_ETL_DESIGN.md`)
3. Add corresponding Dagster assets in `scripts/orchestration/assets.py`
4. Update feature definitions in `scripts/feature_store/defs.py`
5. Write unit tests in `tests/unit/`

### Data Pipeline Development
1. Implement new data sources in `scripts/core/`
2. Add quality checks in `scripts/quality/`
3. Register assets and dependencies in Dagster
4. Test with limited dataset before full execution

### ML Model Development
1. Model architectures go in `src/models/architectures/`
2. Training components in `src/training/`
3. Configuration in `configs/atft/`
4. Use MLflow for experiment tracking

## Critical Constraints

### Performance Requirements
- **Memory**: System requires 16GB+ RAM (production uses 200GB+)
- **Processing**: Optimized for 14,000+ records/second with Polars
- **API**: JQuants rate limiting - max 150 concurrent connections
- **Storage**: Uses Parquet compression (80% space reduction)

### Large-Scale Data Processing Constraints
- **Parquet Data**: 17GB total, 195 files, requiring streaming data loader
- **Chunk Processing**: chunk_size=10000 for memory efficiency
- **Feature Dimensions**: 306 total features (46 current + 260 historical sequences)
- **Sequence Processing**: 20-day lookback windows with 5 prediction horizons
- **Memory Mapping**: Uses memory-mapped files for large datasets
- **Batch Size Limits**: Reduced to 64 (train) / 128 (val) for stability

### Data Integrity
- **No Data Leakage**: Strict temporal splitting enforced
- **Corporate Actions**: Automatic adjustment for stock splits/dividends
- **Quality Validation**: Comprehensive checks via `great-expectations`
- **Backward Compatibility**: Maintains 62 core features from 713 indicators

### Resource Management
- Docker services require significant memory allocation
- Use `docker stats` to monitor resource usage
- Adjust `MAX_PARALLEL_WORKERS` if memory constrained
- ClickHouse requires proper initialization scripts

## Batch Processing Safety & Financial ML Best Practices

### P0: Data Leakage Prevention (最重要)

#### Temporal Data Integrity
- **統計量のfold内fit**: 訓練期間でfit → 検証/テストはtransformのみ
- **クロスセクションZ標準化**: 当日の全銘柄統計を使用（ミニバッチ統計は禁止）
- **BatchNorm禁止**: LayerNorm/GroupNormを使用（`src/models/components/graph_norm.py`）

#### Walk-Forward Validation
- **embargo=20日**: 最大予測ホライズンに合わせた情報遮断
- **時系列順序保持**: train→val→testの厳格な時間順序
- **境界管理**: fold境界をまたぐバッチを作らない

### P1: Sampler/Collate Implementation

#### Date-Bucketed Sampling (実装済み)
```python
# 使用方法: src/data/samplers/day_batch_sampler_fixed.py
from src.data.samplers.day_batch_sampler_fixed import DayBatchSamplerFixed

sampler = DayBatchSamplerFixed(
    dataset=train_dataset,
    max_batch_size=2048,  # Production設定（development: 64-128推奨）
    min_nodes_per_day=20,
    shuffle=True
)
```

#### Walk-Forward with Embargo (実装済み)
**既存実装**: `scripts/train_atft.py:2176+`でPurged K-Fold + embargo実装済み

```python
# 実装済みの使用方法
# train_atft.pyでの設定例
embargo_days = int(os.getenv("EMBARGO_DAYS", "20"))  # デフォルト20日
logger.info(f"Using Purged K-Fold: k={cv_folds}, embargo={embargo_days}d")

# embargo適用例（実際の実装から）
embargo = pd.to_timedelta(embargo_days, unit="D")
train_end_eff = val_start - embargo
```

**参考：自前実装する場合**
```python
def create_walk_forward_folds(dates, n_splits=5, embargo=20):
    """Walk-Forward分割（embargo付き）"""
    unique_dates = sorted(dates.unique())
    fold_size = len(unique_dates) // n_splits
    
    folds = []
    for i in range(n_splits):
        train_end = unique_dates[fold_size * (i + 1)]
        val_start = unique_dates[fold_size * (i + 1) + embargo]
        val_end = unique_dates[min(fold_size * (i + 2), len(unique_dates)-1)]
        
        folds.append({
            'train': dates <= train_end,
            'val': (dates >= val_start) & (dates <= val_end)
        })
    return folds
```

#### Cross-sectional Z-score Normalization
```python
def cs_zscore(df, feature_cols, date_col='date'):
    """日次クロスセクションZ標準化"""
    for date in df[date_col].unique():
        mask = df[date_col] == date
        day_data = df.loc[mask, feature_cols]
        
        # 当日の統計量
        mu = day_data.mean()
        sigma = day_data.std() + 1e-8
        
        # 標準化
        df.loc[mask, feature_cols] = (day_data - mu) / sigma
    return df
```

### P1: GAT/Graph Batch Processing

#### Temporal Graph Consistency
- **スナップショット方式**: 各シーケンス末日での相関グラフ構築
- **近傍サンプリング**: k=10銘柄、週次更新で計算効率化
- **段階的導入**: peer_mean/peer_var特徴量から開始

```python
# グラフ特徴量の段階的導入
def add_peer_features(df, k_neighbors=10):
    """近傍銘柄の統計量を特徴量に追加"""
    for date in df['date'].unique():
        day_data = df[df['date'] == date]
        # k近傍の平均/分散を計算
        peer_mean = day_data.groupby('sector')['return_5d'].transform('mean')
        peer_var = day_data.groupby('sector')['return_5d'].transform('std')
        df.loc[df['date'] == date, 'peer_mean'] = peer_mean
        df.loc[df['date'] == date, 'peer_var'] = peer_var
    return df
```

### P1: Performance Optimization

#### GPU最適化設定

**Production設定** (`configs/atft/train/production.yaml`):
```yaml
batch:
  train_batch_size: 2048    # A100/H100最適化
  val_batch_size: 4096
  num_workers: 8
  prefetch_factor: 4
  persistent_workers: true
  pin_memory: true
  gradient_accumulation_steps: 2
```

**Development設定** (`configs/train/adaptive.yaml`):
```yaml
batch:
  train_batch_size: 64      # メモリ制約環境
  val_batch_size: 128
  num_workers: 8
  pin_memory: true
  gradient_accumulation_steps: 2
```

#### Parquet Lazy Loading
```python
import polars as pl

# 効率的なParquet読み込み
df = pl.scan_parquet("output/atft_data/*.parquet")
    .select(feature_cols + ['date', 'code'])
    .filter(pl.col("date").is_between(start_date, end_date))
    .collect()
```

### P1: Metrics Computation

#### Batch vs Daily Evaluation
- **学習**: バッチ単位の損失計算（回帰、分位点損失）
- **評価**: 日次クロスセクション集計
  - **IC (Information Coefficient)**: 予測値と実際のリターンの相関
  - **RankIC**: ランク相関（外れ値にロバスト）
  - **Decile Analysis**: Long-Short spread分析

```python
def compute_daily_metrics(predictions, targets, dates):
    """日次評価メトリクスの計算"""
    daily_metrics = []
    for date in dates.unique():
        mask = dates == date
        pred_day = predictions[mask]
        target_day = targets[mask]
        
        # IC計算
        ic = np.corrcoef(pred_day, target_day)[0, 1]
        rank_ic = spearmanr(pred_day, target_day)[0]
        
        daily_metrics.append({
            'date': date,
            'ic': ic,
            'rank_ic': rank_ic
        })
    
    return pd.DataFrame(daily_metrics)
```

### P2: Advanced Quality Improvements

#### Importance Sampling
```python
def volatility_weighted_sampling(df, vol_col='return_volatility'):
    """ボラティリティ調整サンプリング"""
    weights = 1.0 / (df[vol_col] + 1e-8)  # 高ボラ銘柄の重みを下げる
    weights = weights / weights.sum()
    return weights
```

#### Curriculum Learning
```python
# 段階的複雑性増加
curriculum_phases = {
    'phase1': {'sequence_length': 40, 'horizons': [1, 5]},
    'phase2': {'sequence_length': 60, 'horizons': [1, 5, 10]},
    'phase3': {'sequence_length': 60, 'horizons': [1, 2, 3, 5, 10, 20]}
}
```

#### Sharpe Ratio Stabilization
```python
def stable_sharpe_ratio(returns, epsilon=1e-12):
    """安定化されたSharpe比計算"""
    mu = returns.mean()
    sigma = returns.std() + epsilon  # ゼロ除算防止
    return mu / sigma
```

### Safety Checklist

- [ ] **統計fit/transform分離**: fold内fit → transform（`RobustZScaler`使用）
- [ ] **Walk-Forward + embargo**: 20日embargo実施
- [ ] **LayerNorm使用**: BatchNorm不使用確認
- [ ] **Sharpe安定化**: 分母事前計算（epsilon=1e-12）
- [ ] **グラフ時点整合**: スナップショット管理
- [ ] **メモリ最適化**: pin_memory/persistent_workers設定
- [ ] **日次評価**: IC/RankIC/Decileの日次集計
- [ ] **クロスセクション正規化**: 当日統計のみ使用
- [ ] **embargo期間**: 最大予測ホライズンと整合

### Production Configuration Reference

**Production設定** (`configs/atft/train/production.yaml`):
- **バッチサイズ**: 2048 (A100/H100最適化)
- **混合精度**: bf16-mixed
- **勾配クリッピング**: 0.8 (安定性重視)
- **学習率**: 0.001 (AdamW)
- **Early Stopping**: patience=9, monitor=val/total_loss
- **損失関数**: MSE + multi-horizon重み [1.0, 0.8, 0.6, 0.4, 0.2]

**Development設定** (`configs/train/adaptive.yaml`):
- **バッチサイズ**: 64 (メモリ制約環境)
- **Phase Training**: 段階的学習対応
- **Quantile Loss**: 分位点回帰
- **Sharpe比最適化**: 補助損失として使用

## Troubleshooting

### Common Issues

**Memory/Performance**:
```bash
# Reduce parallel workers
export MAX_PARALLEL_WORKERS=10

# Monitor resource usage  
docker stats

# Restart services if memory issues
make docker-down && make docker-up

# For large Parquet data processing
# Reduce batch size in configs/train/adaptive.yaml:
# train_batch_size: 32 (instead of 64)
# Reduce sequence length: sequence_length: 10 (instead of 20)
# Reduce chunk size: chunk_size: 5000 (instead of 10000)
```

**JQuants API Issues**:
```bash
# Reduce API concurrency
export JQUANTS_MAX_CONCURRENT=50

# Check API credentials in .env file
```

**Data Pipeline Failures**:
```bash
# Check Dagster UI for detailed error logs
# Validate data quality issues in output/
# Examine individual asset execution in logs/
```

**Service Startup Problems**:
```bash
# Check individual service logs
docker-compose logs [service-name]

# Verify port availability
lsof -i :3001  # Dagster
lsof -i :5000  # MLflow

# Full service restart
docker-compose down -v && docker-compose up -d
```

## Testing Strategy

- **Unit Tests**: Focus on data processing logic and calculations
- **Integration Tests**: End-to-end pipeline validation  
- **Smoke Tests**: Quick validation of core functionality
- **Data Quality Tests**: Automated validation of output datasets

Run comprehensive validation before production deployment:
```bash
make test && python scripts/test_atft_training.py
```

## Important Notes

- **Production Data**: Handle Japanese stock market data with extreme care
- **Time Zones**: All scheduling uses JST (Japan Standard Time) 
- **Parallel Processing**: Optimized for high-memory environments
- **Model Persistence**: Models saved in PyTorch format in `models/` directory
- **Feature Engineering**: Based on financial domain expertise and empirical validation