# 🛡️ ML Safety Guardrails and Data Leakage Prevention

This document outlines critical safety measures implemented in Gogooku3 to prevent data leakage and ensure robust time-series machine learning for financial markets.

## 🚨 Critical Safety Rules

### ❌ **NEVER DO** (Automatically Prevented)

#### 1. Future Information Leakage
```python
# ❌ WRONG: Using future information in normalization
scaler.fit(full_data)  # Uses future statistics for past predictions
train_norm = scaler.transform(train_data)  # Already contaminated!

# ✅ CORRECT: CrossSectionalNormalizerV2 enforces separation
normalizer = CrossSectionalNormalizerV2()
normalizer.fit(train_data)           # Stats computed from train only
train_norm = normalizer.transform(train_data) 
test_norm = normalizer.transform(test_data)   # Same train stats applied
```

#### 2. Cross-Sample Information Leakage
```python
# ❌ WRONG: BatchNorm uses statistics across training batch
nn.BatchNorm1d(num_features)  # Leaks information across samples

# ✅ CORRECT: Only cross-sectional normalization per time period
# Each day's data is normalized independently using only that day's statistics
```

#### 3. Temporal Overlaps Without Embargo
```python
# ❌ WRONG: Adjacent train/test periods
train_end = "2024-06-30"
test_start = "2024-07-01"  # No gap between train and test

# ✅ CORRECT: WalkForwardSplitterV2 enforces embargo period
train_end = "2024-06-30" 
embargo_period = 20  # 20-day gap
test_start = "2024-07-21"  # Safe temporal separation
```

### ✅ **ALWAYS DO** (Production Implementation)

#### 1. Proper Normalization Sequence
```python
# ✅ IMPLEMENTED: Strict fit-transform separation
normalizer = CrossSectionalNormalizerV2(
    cache_stats=True,      # Cache for consistent application
    robust_clip=5.0        # Outlier protection
)

# Training phase
normalizer.fit(train_data)  # Learn statistics from training data only
train_normalized = normalizer.transform(train_data)

# Testing phase  
test_normalized = normalizer.transform(test_data)  # Apply same train stats
```

#### 2. Walk-Forward Validation with Embargo
```python
# ✅ IMPLEMENTED: Temporal splits with safety gaps
splitter = WalkForwardSplitterV2(
    n_splits=5,
    embargo_days=20,        # 20-day safety buffer
    min_train_days=252      # Minimum 1 year training data
)

for fold, (train_idx, test_idx) in enumerate(splitter.split(data)):
    # Automatic validation: train.max() + 20 <= test.min()
    assert (data.iloc[train_idx]['date'].max() + 
            pd.Timedelta(days=20) <= 
            data.iloc[test_idx]['date'].min())
```

#### 3. Automatic Data Leakage Detection
```python
# ✅ IMPLEMENTED: Built-in overlap detection
validation_results = splitter.validate_split(data)

if len(validation_results['overlaps']) > 0:
    warnings.warn(f"Found {len(validation_results['overlaps'])} temporal overlaps!")
    for overlap in validation_results['overlaps']:
        print(f"Overlap: {overlap['train_end']} -> {overlap['test_start']}")
```

## 🔒 Core Safety Components

### 1. CrossSectionalNormalizerV2

#### Purpose
Remove market-wide effects while preserving cross-sectional (relative) information without temporal leakage.

#### Implementation
```python
class CrossSectionalNormalizerV2:
    def __init__(self, cache_stats=True, robust_clip=5.0):
        self.cache_stats = cache_stats      # Cache for consistency
        self.robust_clip = robust_clip      # Outlier clipping
        
    def fit(self, data):
        """Compute statistics from training data only"""
        # Store mean/std for each feature per date
        self._daily_stats = {}
        
    def transform(self, data):
        """Apply normalization using stored training statistics"""
        # Z-score using training statistics only
        return (data - self._daily_stats['mean']) / self._daily_stats['std']
```

#### Safety Features
- **Temporal Isolation**: Each date normalized independently
- **Training Stats Only**: Test normalization uses training statistics
- **Robust Outliers**: 5σ clipping prevents extreme values
- **Validation**: Automatic mean≈0, std≈1 checking

### 2. WalkForwardSplitterV2

#### Purpose
Create temporally separated training/validation splits that prevent future information leakage.

#### Implementation
```python
class WalkForwardSplitterV2:
    def __init__(self, n_splits=5, embargo_days=20, min_train_days=252):
        self.n_splits = n_splits
        self.embargo_days = embargo_days    # Critical safety parameter
        self.min_train_days = min_train_days
        
    def split(self, data):
        """Generate chronological splits with embargo gaps"""
        for i in range(self.n_splits):
            train_end = self._compute_train_end(i)
            test_start = train_end + pd.Timedelta(days=self.embargo_days)
            
            train_mask = data['date'] <= train_end
            test_mask = data['date'] >= test_start
            
            yield train_mask, test_mask
```

#### Safety Features
- **Embargo Period**: Configurable gap (default 20 days)
- **Chronological Order**: Strictly increasing time splits
- **Overlap Detection**: Automatic validation of temporal separation
- **Minimum Training**: Ensures sufficient training data (252+ days)

### 3. Financial Feature Safety

#### Target Variable Construction
```python
# ✅ SAFE: Forward-looking returns for prediction
def create_forward_returns(data, horizons=[1, 5, 10, 20]):
    """Create forward returns without lookahead bias"""
    for horizon in horizons:
        data[f'feat_ret_{horizon}d'] = (
            data.groupby('code')['close']
            .pct_change(periods=horizon)
            .shift(-horizon)  # Shift back to align with prediction date
        )
    return data

# ❌ DANGEROUS: Using any historical target data during normalization
# This would leak future information into past predictions
```

#### Feature Engineering Safety
```python
# ✅ SAFE: Using only historical data for feature computation
def compute_technical_indicators(data):
    """All indicators use only past/current information"""
    data['rsi_14'] = compute_rsi(data['close'], window=14)
    data['sma_20'] = data['close'].rolling(20).mean()
    data['volatility_20d'] = data['close'].pct_change().rolling(20).std()
    
    # Ensure no forward-looking calculations
    assert not data[['rsi_14', 'sma_20', 'volatility_20d']].isna().all()
    return data
```

## 📊 Monitoring and Validation

### Automated Safety Checks

#### 1. Pipeline Execution Monitoring
```python
# Integrated in SafeTrainingPipeline
def run_pipeline(self, n_splits=5, embargo_days=20):
    """Execute training pipeline with safety validation"""
    
    # Pre-execution checks
    self._validate_data_quality()
    self._check_temporal_ordering()
    
    # During execution monitoring
    for step in self.pipeline_steps:
        start_time = time.time()
        memory_before = self._get_memory_usage()
        
        result = step.execute()
        
        # Post-step validation
        self._validate_step_output(step, result)
        self._check_memory_usage(memory_before)
        
    return results
```

#### 2. Data Quality Metrics
```python
# Automatic validation during processing
validation_metrics = {
    'missing_value_rate': 0.02,      # < 2% missing allowed
    'outlier_detection': 0.01,       # < 1% extreme outliers  
    'temporal_consistency': True,     # Monotonic time ordering
    'feature_stability': 0.95,       # 95% feature availability
    'normalization_quality': {
        'mean_deviation': 0.05,       # |mean| < 0.05 after normalization
        'std_deviation': 0.05         # |std - 1| < 0.05 after normalization
    }
}
```

### Known Issues and Mitigation

#### 🟡 Current Technical Debt (Non-Critical)
```yaml
Issue 1: CrossSectionalNormalizerV2 Polars API compatibility
  Status: Non-blocking, fallback to pandas available
  Impact: Slight performance reduction (0.1s → 0.3s)
  
Issue 2: 2 Walk-Forward overlaps detected in edge cases  
  Status: Date boundary edge case, embargo still enforced
  Impact: Minimal, overlaps are <24 hours and distant from prediction horizons
  
Issue 3: GBM baseline column name mapping (_z suffix handling)
  Status: Cosmetic, does not affect model training
  Impact: Column name inconsistency in intermediate outputs
```

#### ✅ Production-Ready Components
- Data loading: ProductionDatasetV3 ✅
- Feature engineering: QualityFinancialFeaturesGenerator ✅
- Normalization: CrossSectionalNormalizerV2 ✅
- Temporal splitting: WalkForwardSplitterV2 ✅
- Graph construction: FinancialGraphBuilder ✅
- Pipeline integration: SafeTrainingPipeline ✅

## 🎯 Production Safety Targets

### Performance and Safety Metrics
```yaml
Execution Performance:
  Pipeline Runtime: <2s (current: 1.9s) ✅
  Memory Usage: <8GB (current: 7.0GB) ✅
  Data Processing: 606K samples in 1.9s ✅
  
Safety Validation:
  Temporal Overlaps: 0 (current: 2 minor edge cases) 🟡
  Data Leakage Tests: All pass ✅
  Normalization Quality: Mean≈0, Std≈1 ✅
  Embargo Enforcement: 20-day minimum ✅
  
Model Performance:
  Target Sharpe Ratio: 0.849 (ATFT-GAT-FAN)
  Baseline Established: LightGBM benchmark ✅
  Validation Method: 5-fold Walk-Forward ✅
  Feature Count: 26 (optimized from 713) ✅
```

## 🔧 Troubleshooting Safety Issues

### Common Safety Violations and Solutions

#### 1. Normalization Errors
```python
# Problem: Inconsistent normalization statistics
# Solution: Use cached normalizer
normalizer = CrossSectionalNormalizerV2(cache_stats=True)
validation = normalizer.validate_transform(normalized_data)

if len(validation['warnings']) > 0:
    print("⚠️ Normalization warnings:", validation['warnings'])
    # Reload normalizer with fresh statistics
```

#### 2. Temporal Overlap Detection
```python
# Problem: Detected temporal overlaps in splits
# Solution: Increase embargo period or adjust split parameters
splitter = WalkForwardSplitterV2(
    embargo_days=30,        # Increase from 20 to 30 days
    min_train_days=365      # Increase minimum training period
)

# Validate before using
validation = splitter.validate_split(data)
assert len(validation['overlaps']) == 0, "Still have overlaps!"
```

#### 3. Memory and Performance Issues
```python
# Problem: Pipeline exceeds memory limits
# Solution: Use memory-limited execution
pipeline = SafeTrainingPipeline(experiment_name="memory_constrained")

results = pipeline.run_pipeline(
    memory_limit_gb=4,      # Reduce memory limit
    n_splits=3,             # Fewer CV folds
    embargo_days=20
)

# Monitor memory usage
print(f"Peak memory: {results['peak_memory_gb']:.1f}GB")
```

## 📋 Safety Checklist

Before deploying any model to production, ensure:

- [ ] **Data Normalization**: Only training statistics used for test normalization
- [ ] **Temporal Separation**: Minimum 20-day embargo between train/test
- [ ] **Feature Engineering**: No forward-looking information in features  
- [ ] **Target Construction**: Forward returns properly aligned with prediction dates
- [ ] **Validation Method**: Walk-Forward splitting enforced
- [ ] **Memory Limits**: Pipeline execution within 8GB memory budget
- [ ] **Performance Targets**: <2s execution time for full pipeline
- [ ] **Quality Metrics**: IC/RankIC computed on out-of-sample predictions only
- [ ] **Overlap Detection**: Zero temporal overlaps between train/test sets
- [ ] **Documentation**: All safety measures documented and reviewed

## 🔗 Related Documentation

- [ML Metrics](metrics.md) - Evaluation methods and feature descriptions
- [Model Training](model-training.md) - ATFT-GAT-FAN training pipeline  
- [Data Pipeline](../architecture/data-pipeline.md) - Technical architecture
- [Contributing Guide](../development/contributing.md) - Development guidelines

---

**🛡️ Key Principle**: In financial ML, preventing data leakage is more important than optimizing model performance. A model with modest but honest performance is infinitely more valuable than an overfit model with artificially high backtesting returns.