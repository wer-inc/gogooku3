"""
DayBatchSampler - Bridge module for backward compatibility.

This module provides a compatibility layer for DayBatchSampler imports.
"""

# Import from the actual location
from src.gogooku3.data.samplers.day_batch_sampler import DayBatchSampler

# Export for compatibility
__all__ = ["DayBatchSampler"]
