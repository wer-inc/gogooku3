"""
Production Loader V2 - Bridge module for backward compatibility.

This module provides ProductionDataModuleV2 and fallback ProductionDatasetV2
for the legacy train_atft.py script.
"""

# Import ProductionDataModuleV2 from the modern location
from src.gogooku3.training.atft.data_module import ProductionDataModuleV2

# Import the optimized dataset as fallback for ProductionDatasetV2
# This is only used when USE_OPTIMIZED_LOADER=0
from src.data.loaders.production_loader_v2_optimized import ProductionDatasetOptimized as ProductionDatasetV2

# Export both classes for compatibility
__all__ = ["ProductionDatasetV2", "ProductionDataModuleV2"]