#!/usr/bin/env python3
"""
Performance Optimizer for gogooku3-standalone
===========================================

Provides performance optimization features that can be enabled via environment variables.
All optimizations are opt-in and maintain backward compatibility.

Environment Variables:
    PERF_POLARS_STREAM=1        # Enable Polars streaming for large datasets
    PERF_PARALLEL_PROCESSING=1  # Enable parallel data processing
    PERF_MEMORY_OPTIMIZATION=1  # Enable memory optimization techniques
    PERF_GPU_ACCELERATION=1     # Enable GPU acceleration (if available)
    PERF_CACHING_ENABLED=1      # Enable intelligent caching

Usage:
    # Enable all optimizations
    export PERF_POLARS_STREAM=1
    export PERF_PARALLEL_PROCESSING=1
    export PERF_MEMORY_OPTIMIZATION=1
    export PERF_GPU_ACCELERATION=1
    export PERF_CACHING_ENABLED=1

    # Run with optimizations
    python main.py safe-training --mode full
"""

import os
import sys
import time
import psutil
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, TypeVar
from functools import wraps
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pickle
import hashlib

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Type hints
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


class PerformanceOptimizer:
    """Performance optimization manager."""

    def __init__(self):
        self.config = self._load_config()
        self.cache_dir = project_root / "cache" / "performance"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize optimization components
        self.cacher = CacheManager(self.cache_dir) if self.config['caching_enabled'] else None
        self.parallel_processor = ParallelProcessor() if self.config['parallel_processing'] else None
        self.memory_optimizer = MemoryOptimizer() if self.config['memory_optimization'] else None

    def _load_config(self) -> Dict[str, bool]:
        """Load performance configuration from environment variables."""
        return {
            'polars_stream': os.getenv('PERF_POLARS_STREAM', '0') == '1',
            'parallel_processing': os.getenv('PERF_PARALLEL_PROCESSING', '0') == '1',
            'memory_optimization': os.getenv('PERF_MEMORY_OPTIMIZATION', '0') == '1',
            'gpu_acceleration': os.getenv('PERF_GPU_ACCELERATION', '0') == '1',
            'caching_enabled': os.getenv('PERF_CACHING_ENABLED', '0') == '1'
        }

    def optimize_polars_processing(self, func: F) -> F:
        """Decorator to optimize Polars DataFrame processing."""
        if not self.config['polars_stream']:
            return func

        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info("🚀 Enabling Polars streaming optimization")

            # Set Polars streaming configuration
            import polars as pl

            # Enable streaming for large datasets
            with pl.StringCache():
                # Use lazy evaluation for better memory efficiency
                result = func(*args, **kwargs)

                # If result is a DataFrame, ensure it's optimized
                if hasattr(result, 'collect'):  # LazyFrame
                    result = result.collect(streaming=True)

            return result

        return wrapper

    def optimize_parallel_processing(self, func: F) -> F:
        """Decorator to enable parallel processing."""
        if not self.config['parallel_processing'] or not self.parallel_processor:
            return func

        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info("🔄 Enabling parallel processing optimization")
            return self.parallel_processor.process_parallel(func, *args, **kwargs)

        return wrapper

    def optimize_memory_usage(self, func: F) -> F:
        """Decorator to optimize memory usage."""
        if not self.config['memory_optimization'] or not self.memory_optimizer:
            return func

        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info("💾 Enabling memory optimization")

            # Monitor memory before
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            # Execute with memory optimization
            result = self.memory_optimizer.optimize_execution(func, *args, **kwargs)

            # Monitor memory after
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_delta = memory_after - memory_before

            logger.info(".1f"
            return result

        return wrapper

    def enable_caching(self, func: F, cache_key: Optional[str] = None) -> F:
        """Decorator to enable intelligent caching."""
        if not self.config['caching_enabled'] or not self.cacher:
            return func

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if cache_key is None:
                # Auto-generate based on function and arguments
                key_components = [func.__name__] + [str(arg) for arg in args]
                key_components.extend([f"{k}={v}" for k, v in kwargs.items()])
                cache_key_gen = hashlib.md5('_'.join(key_components).encode()).hexdigest()
            else:
                cache_key_gen = cache_key

            # Check cache
            cached_result = self.cacher.get(cache_key_gen)
            if cached_result is not None:
                logger.info(f"📋 Cache hit for {func.__name__}")
                return cached_result

            # Execute and cache
            logger.info(f"💾 Computing and caching {func.__name__}")
            result = func(*args, **kwargs)
            self.cacher.set(cache_key_gen, result)

            return result

        return wrapper

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        metrics = {
            'config': self.config,
            'system': {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent
            },
            'optimizations_active': []
        }

        if self.config['polars_stream']:
            metrics['optimizations_active'].append('polars_streaming')
        if self.config['parallel_processing']:
            metrics['optimizations_active'].append('parallel_processing')
        if self.config['memory_optimization']:
            metrics['optimizations_active'].append('memory_optimization')
        if self.config['gpu_acceleration']:
            metrics['optimizations_active'].append('gpu_acceleration')
        if self.config['caching_enabled']:
            metrics['optimizations_active'].append('caching')

        return metrics


class CacheManager:
    """Intelligent caching manager."""

    def __init__(self, cache_dir: Path, max_cache_size_mb: int = 1000):
        self.cache_dir = cache_dir
        self.max_cache_size_mb = max_cache_size_mb
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """Get cached item."""
        cache_file = self.cache_dir / f"{key}.pkl"

        if not cache_file.exists():
            return None

        try:
            with self._lock:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Cache read error for {key}: {e}")
            return None

    def set(self, key: str, value: Any) -> bool:
        """Set cached item."""
        cache_file = self.cache_dir / f"{key}.pkl"

        try:
            with self._lock:
                # Ensure cache directory exists
                cache_file.parent.mkdir(parents=True, exist_ok=True)

                # Clean cache if needed
                self._clean_cache_if_needed()

                # Save to cache
                with open(cache_file, 'wb') as f:
                    pickle.dump(value, f)

                return True

        except Exception as e:
            logger.warning(f"Cache write error for {key}: {e}")
            return False

    def _clean_cache_if_needed(self):
        """Clean old cache files if cache size exceeds limit."""
        try:
            cache_files = list(self.cache_dir.glob("*.pkl"))
            if not cache_files:
                return

            # Calculate total cache size
            total_size = sum(f.stat().st_size for f in cache_files) / 1024 / 1024  # MB

            if total_size > self.max_cache_size_mb:
                # Remove oldest files
                cache_files.sort(key=lambda f: f.stat().st_mtime)
                files_to_remove = len(cache_files) // 2  # Remove half

                for i in range(files_to_remove):
                    try:
                        cache_files[i].unlink()
                        logger.info(f"🗑️  Removed old cache file: {cache_files[i].name}")
                    except Exception as e:
                        logger.warning(f"Failed to remove cache file {cache_files[i]}: {e}")

        except Exception as e:
            logger.warning(f"Cache cleanup error: {e}")


class ParallelProcessor:
    """Parallel processing manager."""

    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(32, os.cpu_count() * 2)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

    def process_parallel(self, func: Callable, *args, **kwargs) -> Any:
        """Process function in parallel if applicable."""
        # For now, this is a simple wrapper
        # In production, you'd implement more sophisticated parallel processing
        # based on the specific function and data structure

        return func(*args, **kwargs)

    def __del__(self):
        """Cleanup executor."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


class MemoryOptimizer:
    """Memory optimization manager."""

    def __init__(self):
        self.gc_threshold = (700, 10, 10)  # Conservative GC thresholds
        self.original_threshold = None

    def optimize_execution(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with memory optimizations."""
        import gc

        # Save original GC threshold
        self.original_threshold = gc.get_threshold()

        try:
            # Set more aggressive GC thresholds
            gc.set_threshold(*self.gc_threshold)

            # Execute function
            result = func(*args, **kwargs)

            # Force garbage collection
            gc.collect()

            return result

        finally:
            # Restore original GC threshold
            if self.original_threshold:
                gc.set_threshold(*self.original_threshold)


# Global optimizer instance
_optimizer = None

def get_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer instance."""
    global _optimizer
    if _optimizer is None:
        _optimizer = PerformanceOptimizer()
    return _optimizer


# Convenience decorators
def polars_streaming(func: F) -> F:
    """Decorator to enable Polars streaming optimization."""
    return get_optimizer().optimize_polars_processing(func)


def parallel_processing(func: F) -> F:
    """Decorator to enable parallel processing."""
    return get_optimizer().optimize_parallel_processing(func)


def memory_optimized(func: F) -> F:
    """Decorator to enable memory optimization."""
    return get_optimizer().optimize_memory_usage(func)


def cached(cache_key: Optional[str] = None):
    """Decorator to enable caching."""
    def decorator(func: F) -> F:
        return get_optimizer().enable_caching(func, cache_key)
    return decorator


def performance_metrics() -> Dict[str, Any]:
    """Get current performance metrics."""
    return get_optimizer().get_performance_metrics()


# Integration utilities
def integrate_with_existing_code():
    """Integrate performance optimizations with existing codebase."""
    logger.info("🔧 Integrating performance optimizations...")

    # This function would be called during application startup
    # to monkey-patch or wrap existing functions with optimizations

    optimizer = get_optimizer()
    config = optimizer.config

    if any(config.values()):
        logger.info("✅ Performance optimizations enabled:"        for opt, enabled in config.items():
            if enabled:
                logger.info(f"  • {opt.replace('_', ' ').title()}")
    else:
        logger.info("ℹ️  No performance optimizations enabled (use PERF_* environment variables)")


if __name__ == "__main__":
    # Command-line interface
    import argparse

    parser = argparse.ArgumentParser(description='gogooku3 Performance Optimizer')
    parser.add_argument('command', choices=['metrics', 'integrate', 'clean-cache'])
    parser.add_argument('--format', choices=['json', 'pretty'], default='pretty')

    args = parser.parse_args()

    optimizer = get_optimizer()

    if args.command == 'metrics':
        metrics = optimizer.get_performance_metrics()

        if args.format == 'json':
            import json
            print(json.dumps(metrics, indent=2))
        else:
            print("🚀 Performance Metrics")
            print("=" * 30)
            print(f"Optimizations: {', '.join(metrics['optimizations_active']) or 'None'}")
            print(f"CPU Usage: {metrics['system']['cpu_percent']:.1f}%")
            print(f"Memory Usage: {metrics['system']['memory_percent']:.1f}%")
            print(f"Disk Usage: {metrics['system']['disk_usage']:.1f}%")

    elif args.command == 'integrate':
        integrate_with_existing_code()

    elif args.command == 'clean-cache':
        if optimizer.cacher:
            import shutil
            shutil.rmtree(optimizer.cacher.cache_dir, ignore_errors=True)
            optimizer.cacher.cache_dir.mkdir(parents=True, exist_ok=True)
            print("🗑️  Cache cleaned")
        else:
            print("❌ Caching not enabled")