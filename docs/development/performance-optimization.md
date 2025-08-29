# ⚡ Performance Optimization Guide

## Overview

gogooku3-standalone provides comprehensive performance optimization features that can be enabled via environment variables. All optimizations are opt-in and maintain backward compatibility with existing workflows.

## Optimization Categories

### 1. Data Processing Optimization

#### Polars Streaming (`PERF_POLARS_STREAM`)
- **Purpose**: Optimize large dataset processing using Polars lazy evaluation
- **Benefits**: 30%+ performance improvement for datasets >100MB
- **Memory Usage**: Reduced memory footprint through streaming
- **Compatibility**: Fully backward compatible

**Usage:**
```bash
export PERF_POLARS_STREAM=1
python main.py safe-training --mode full
```

**Implementation:**
```python
@performance_optimizer.polars_streaming
def process_large_dataset(df):
    """Process large datasets with streaming optimization."""
    return (
        df.lazy()
        .filter(pl.col("close") > 0)
        .group_by("symbol")
        .agg([
            pl.col("close").mean().alias("avg_close"),
            pl.col("volume").sum().alias("total_volume")
        ])
        .collect(streaming=True)  # Enable streaming
    )
```

#### Parallel Processing (`PERF_PARALLEL_PROCESSING`)
- **Purpose**: Utilize multiple CPU cores for data processing
- **Benefits**: Linear scaling with CPU cores
- **Memory Usage**: Increased due to parallel execution
- **Compatibility**: Safe for CPU-bound operations

**Usage:**
```bash
export PERF_PARALLEL_PROCESSING=1
python scripts/performance_optimizer.py
```

**Implementation:**
```python
@performance_optimizer.parallel_processing
def parallel_feature_engineering(df):
    """Parallel feature engineering across CPU cores."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp

    # Split data by symbols for parallel processing
    symbols = df['symbol'].unique()
    chunks = [df[df['symbol'] == symbol] for symbol in symbols]

    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        results = list(executor.map(process_symbol_chunk, chunks))

    return pd.concat(results)
```

### 2. Memory Optimization

#### Memory Optimization (`PERF_MEMORY_OPTIMIZATION`)
- **Purpose**: Reduce memory usage through intelligent memory management
- **Benefits**: 20%+ memory reduction, prevents OOM errors
- **Performance Impact**: Minimal overhead
- **Compatibility**: Safe for memory-constrained environments

**Usage:**
```bash
export PERF_MEMORY_OPTIMIZATION=1
python main.py ml-dataset
```

**Implementation:**
```python
@performance_optimizer.memory_optimized
def memory_efficient_processing(df):
    """Memory-efficient data processing."""
    # Process in chunks to reduce memory usage
    chunk_size = 10000
    results = []

    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size].copy()

        # Process chunk
        processed_chunk = process_chunk(chunk)

        # Force garbage collection between chunks
        import gc
        gc.collect()

        results.append(processed_chunk)

    return pd.concat(results, ignore_index=True)
```

### 3. Caching Optimization

#### Intelligent Caching (`PERF_CACHING_ENABLED`)
- **Purpose**: Cache expensive computations and data transformations
- **Benefits**: Significant speedup for repeated operations
- **Storage**: Automatic cache management with size limits
- **Compatibility**: Transparent caching layer

**Usage:**
```bash
export PERF_CACHING_ENABLED=1
python main.py safe-training --mode full
```

**Implementation:**
```python
@performance_optimizer.cached(cache_key="feature_engineering")
def cached_feature_engineering(df, config):
    """Cached feature engineering with automatic key generation."""
    # Expensive computation
    features = engineer_features(df, config)

    # Automatic caching based on function signature
    return features

# Manual cache key
@performance_optimizer.cached(cache_key="custom_cache_key")
def custom_cached_function(data):
    """Function with custom cache key."""
    return expensive_computation(data)
```

### 4. GPU Acceleration (Future)

#### GPU Acceleration (`PERF_GPU_ACCELERATION`)
- **Purpose**: Utilize GPU for ML computations
- **Benefits**: 10-100x speedup for compatible operations
- **Requirements**: CUDA-compatible GPU, PyTorch GPU support
- **Compatibility**: Opt-in for GPU environments

**Usage:**
```bash
export PERF_GPU_ACCELERATION=1
python main.py complete-atft
```

## Performance Monitoring

### Real-time Metrics

```bash
# View current performance metrics
python ops/metrics_exporter.py --once | grep -E "(optimization|performance)"

# Monitor optimization status
python scripts/performance_optimizer.py metrics
```

### Benchmarking

```bash
# Run performance benchmarks
pytest tests/ -k "performance" --benchmark-only --benchmark-save=benchmark.json

# Compare with/without optimizations
PERF_POLARS_STREAM=0 python main.py safe-training --mode quick  # Baseline
PERF_POLARS_STREAM=1 python main.py safe-training --mode quick  # Optimized
```

## Configuration Examples

### Development Environment
```bash
# Development with basic optimizations
export PERF_POLARS_STREAM=1
export PERF_MEMORY_OPTIMIZATION=1
export PERF_CACHING_ENABLED=1
```

### Production Environment
```bash
# Production with full optimizations
export PERF_POLARS_STREAM=1
export PERF_PARALLEL_PROCESSING=1
export PERF_MEMORY_OPTIMIZATION=1
export PERF_CACHING_ENABLED=1
export PERF_GPU_ACCELERATION=1
```

### Memory-Constrained Environment
```bash
# Memory-optimized configuration
export PERF_POLARS_STREAM=1
export PERF_MEMORY_OPTIMIZATION=1
# Disable parallel processing to save memory
unset PERF_PARALLEL_PROCESSING
```

### CI/CD Environment
```bash
# CI/CD with performance testing
export PERF_POLARS_STREAM=1
export PERF_MEMORY_OPTIMIZATION=1
export CI_PERFORMANCE_TESTING=1
```

## Performance Tuning Guide

### Memory Tuning

#### Large Dataset Processing
```python
# For datasets >1GB
@performance_optimizer.polars_streaming
@performance_optimizer.memory_optimized
def process_large_dataset(df):
    """Optimized processing for large datasets."""
    return (
        df.lazy()
        .filter(pl.col("volume") > 0)
        .with_columns([
            (pl.col("close") / pl.col("open") - 1).alias("return"),
            pl.col("high").max().over("symbol").alias("symbol_high"),
            pl.col("low").min().over("symbol").alias("symbol_low")
        ])
        .collect(streaming=True)
    )
```

#### Memory Usage Monitoring
```python
import psutil
import os

def monitor_memory_usage():
    """Monitor memory usage during processing."""
    process = psutil.Process(os.getpid())

    # Memory before
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Memory before: {memory_before:.1f} MB")

    # Your processing code here
    result = expensive_operation()

    # Memory after
    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    memory_delta = memory_after - memory_before
    print(f"Memory after: {memory_after:.1f} MB (Δ{memory_delta:+.1f} MB)")

    return result
```

### CPU Optimization

#### Parallel Feature Engineering
```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

@performance_optimizer.parallel_processing
def parallel_feature_computation(df):
    """Parallel computation of features."""
    def compute_features_for_symbol(symbol_data):
        """Compute features for a single symbol."""
        # Your feature computation logic
        return compute_technical_indicators(symbol_data)

    # Group by symbol
    symbol_groups = df.groupby('symbol')

    # Parallel processing
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        results = list(executor.map(
            compute_features_for_symbol,
            [group for _, group in symbol_groups]
        ))

    return pd.concat(results)
```

#### CPU Usage Optimization
```python
import psutil
import time

def adaptive_processing(df):
    """Adaptive processing based on system load."""
    cpu_percent = psutil.cpu_percent()
    cpu_count = psutil.cpu_count()

    if cpu_percent > 80:
        # High CPU usage - use sequential processing
        return sequential_processing(df)
    elif cpu_count > 4:
        # Multi-core system - use parallel processing
        return parallel_processing(df)
    else:
        # Default processing
        return standard_processing(df)
```

### I/O Optimization

#### Efficient Data Loading
```python
@performance_optimizer.polars_streaming
def optimized_data_loading(file_path):
    """Optimized data loading with Polars."""
    return (
        pl.scan_parquet(file_path)  # Lazy loading
        .filter(pl.col("date") >= pl.date(2023, 1, 1))  # Filter early
        .select([
            "date", "symbol", "open", "high", "low", "close", "volume"
        ])  # Select only needed columns
        .collect(streaming=True)  # Streaming collection
    )
```

#### Batch Processing
```python
def batch_process_large_file(file_path, batch_size=100000):
    """Process large files in batches to optimize memory."""
    import pandas as pd

    # Process in chunks
    chunks = pd.read_csv(file_path, chunksize=batch_size)

    results = []
    for chunk in chunks:
        # Process each chunk
        processed_chunk = process_chunk(chunk)
        results.append(processed_chunk)

        # Memory cleanup
        del chunk
        import gc
        gc.collect()

    return pd.concat(results, ignore_index=True)
```

## Troubleshooting Performance Issues

### Common Performance Problems

#### High Memory Usage
```
Symptoms: OOM errors, slow processing
Solutions:
1. Enable PERF_MEMORY_OPTIMIZATION=1
2. Use streaming processing for large datasets
3. Process data in smaller chunks
4. Monitor memory usage with psutil
```

#### Slow Processing
```
Symptoms: Long execution times, high CPU usage
Solutions:
1. Enable PERF_PARALLEL_PROCESSING=1 for CPU-bound tasks
2. Use Polars instead of pandas for large datasets
3. Enable PERF_POLARS_STREAM=1 for streaming processing
4. Implement intelligent caching
```

#### I/O Bottlenecks
```
Symptoms: High disk I/O, slow data loading
Solutions:
1. Use Parquet format instead of CSV
2. Enable streaming for large files
3. Cache frequently accessed data
4. Use SSD storage for data directories
```

### Performance Debugging

#### Profiling Code Execution
```python
import cProfile
import pstats

def profile_function(func, *args, **kwargs):
    """Profile function execution."""
    profiler = cProfile.Profile()
    profiler.enable()

    result = func(*args, **kwargs)

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions

    return result

# Usage
result = profile_function(my_expensive_function, arg1, arg2)
```

#### Memory Profiling
```python
from memory_profiler import profile

@profile
def memory_intensive_function():
    """Function with memory profiling."""
    # Your code here
    large_data = create_large_dataset()
    processed_data = process_data(large_data)
    return processed_data

# Run with memory profiling
memory_intensive_function()
```

#### System Monitoring
```bash
# Monitor system resources during execution
top -p $(pgrep -f python)  # CPU and memory usage

# Disk I/O monitoring
iostat -x 1  # Disk I/O statistics

# Network monitoring (if applicable)
nload  # Network load monitoring
```

## Best Practices

### Optimization Strategy

1. **Measure First**: Always benchmark before and after optimizations
2. **Start Small**: Enable one optimization at a time
3. **Monitor Impact**: Track performance metrics and system resources
4. **Test Thoroughly**: Ensure optimizations don't break functionality
5. **Document Changes**: Record optimization settings and their impact

### Environment-Specific Optimization

#### Development Environment
- Focus on developer productivity
- Use basic optimizations
- Enable comprehensive logging

#### Staging Environment
- Mirror production configuration
- Enable all safe optimizations
- Extensive performance testing

#### Production Environment
- Enable all optimizations
- Monitor performance continuously
- Automated performance regression detection

### Continuous Optimization

#### Automated Performance Testing
```yaml
# .github/workflows/performance.yml
name: Performance Testing

on:
  push:
    branches: [ main, develop ]
  schedule:
    - cron: '0 2 * * *'  # Daily performance check

jobs:
  performance-test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Run performance benchmarks
        run: |
          export PERF_POLARS_STREAM=1
          export PERF_MEMORY_OPTIMIZATION=1
          pytest tests/ -k "performance" --benchmark-only

      - name: Compare with baseline
        run: |
          # Compare current performance with stored baseline
          # Alert if performance regression detected
```

#### Performance Regression Detection
```python
def detect_performance_regression(current_metrics, baseline_metrics, threshold=0.1):
    """Detect performance regressions."""
    regressions = []

    for metric_name, current_value in current_metrics.items():
        if metric_name in baseline_metrics:
            baseline_value = baseline_metrics[metric_name]
            change = (current_value - baseline_value) / baseline_value

            if change > threshold:  # 10% degradation
                regressions.append({
                    'metric': metric_name,
                    'baseline': baseline_value,
                    'current': current_value,
                    'change_percent': change * 100
                })

    return regressions
```

## Integration with Existing Code

### Decorator Pattern
```python
from scripts.performance_optimizer import performance_optimizer

class MyDataProcessor:
    @performance_optimizer.polars_streaming
    @performance_optimizer.memory_optimized
    @performance_optimizer.cached(cache_key="data_processing")
    def process_data(self, df):
        """Optimized data processing method."""
        return (
            df.lazy()
            .filter(pl.col("close") > 0)
            .with_columns([
                (pl.col("close") - pl.col("open")).alias("price_change"),
                (pl.col("close") / pl.col("open") - 1).alias("return")
            ])
            .collect(streaming=True)
        )
```

### Configuration-Driven Optimization
```python
import os

class ConfigurableOptimizer:
    def __init__(self):
        self.config = {
            'polars_stream': os.getenv('PERF_POLARS_STREAM', '0') == '1',
            'memory_opt': os.getenv('PERF_MEMORY_OPTIMIZATION', '0') == '1',
            'parallel': os.getenv('PERF_PARALLEL_PROCESSING', '0') == '1',
            'caching': os.getenv('PERF_CACHING_ENABLED', '0') == '1'
        }

    def optimize_function(self, func):
        """Apply optimizations based on configuration."""
        optimized_func = func

        if self.config['polars_stream']:
            optimized_func = performance_optimizer.optimize_polars_processing(optimized_func)

        if self.config['memory_opt']:
            optimized_func = performance_optimizer.optimize_memory_usage(optimized_func)

        if self.config['parallel']:
            optimized_func = performance_optimizer.optimize_parallel_processing(optimized_func)

        if self.config['caching']:
            optimized_func = performance_optimizer.enable_caching(optimized_func)

        return optimized_func
```

## Metrics and Monitoring

### Performance Metrics
```python
# Export performance metrics to Prometheus
def export_performance_metrics():
    """Export performance metrics for monitoring."""
    metrics = performance_optimizer.get_performance_metrics()

    prometheus_metrics = f"""
# HELP gogooku3_performance_optimizations_active Active performance optimizations
# TYPE gogooku3_performance_optimizations_active gauge
gogooku3_performance_optimizations_active {len(metrics['optimizations_active'])}

# HELP gogooku3_system_cpu_percent Current CPU usage percentage
# TYPE gogooku3_system_cpu_percent gauge
gogooku3_system_cpu_percent {metrics['system']['cpu_percent']}

# HELP gogooku3_system_memory_percent Current memory usage percentage
# TYPE gogooku3_system_memory_percent gauge
gogooku3_system_memory_percent {metrics['system']['memory_percent']}
"""

    return prometheus_metrics
```

### Grafana Dashboard
```json
{
  "dashboard": {
    "title": "Performance Optimization Dashboard",
    "panels": [
      {
        "title": "Active Optimizations",
        "type": "stat",
        "targets": [
          {
            "expr": "gogooku3_performance_optimizations_active",
            "legendFormat": "Active Optimizations"
          }
        ]
      },
      {
        "title": "System Resources",
        "type": "graph",
        "targets": [
          {
            "expr": "gogooku3_system_cpu_percent",
            "legendFormat": "CPU %"
          },
          {
            "expr": "gogooku3_system_memory_percent",
            "legendFormat": "Memory %"
          }
        ]
      }
    ]
  }
}
```

## Support and Resources

### Getting Help
- **Performance Issues**: Check system resources and optimization settings
- **Memory Problems**: Enable memory optimization and monitor usage
- **Slow Processing**: Review CPU usage and consider parallel processing
- **I/O Bottlenecks**: Optimize data formats and access patterns

### Additional Resources
- **Polars Documentation**: https://pola.rs/
- **Memory Profiling**: https://github.com/pythonprofilers/memory_profiler
- **Performance Benchmarking**: https://pytest-benchmark.readthedocs.io/

---

## Contact & Support

- **Performance Team**: Performance optimization and tuning
- **Infrastructure Team**: System resource management
- **DevOps Team**: CI/CD and monitoring integration

---

*Last Updated: 2024-01-XX*
*Version: 2.0.0*
*Document Owner: Performance Engineering Team*
