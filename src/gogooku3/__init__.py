"""Gogooku3: 壊れず・強く・速く 金融ML システム

Next-generation MLOps batch processing system for Japanese stock market
with enhanced safety, performance, and reproducibility.
"""

__version__ = "2.0.0"
__author__ = "Gogooku Team"

import os
from .utils.settings import settings

# Main components (lazy: avoid importing heavy training stack on package import)
if os.getenv("GOGOOKU3_IMPORT_TRAINING", "0") == "1":
    try:
        from .training.safe_training_pipeline import SafeTrainingPipeline
        from .data.loaders import ProductionDatasetV3, MLDatasetBuilder
        from .data.scalers import CrossSectionalNormalizerV2, WalkForwardSplitterV2
        from .features import QualityFinancialFeaturesGenerator
        from .models import ATFTGATFANModel, LightGBMFinancialBaseline
        from .graph import FinancialGraphBuilder
    except Exception:
        # Expose names but set to None if training stack is unavailable
        SafeTrainingPipeline = None
        ProductionDatasetV3 = None
        MLDatasetBuilder = None
        CrossSectionalNormalizerV2 = None
        WalkForwardSplitterV2 = None
        QualityFinancialFeaturesGenerator = None
        ATFTGATFANModel = None
        LightGBMFinancialBaseline = None
        FinancialGraphBuilder = None
else:
    SafeTrainingPipeline = None
    ProductionDatasetV3 = None
    MLDatasetBuilder = None
    CrossSectionalNormalizerV2 = None
    WalkForwardSplitterV2 = None
    QualityFinancialFeaturesGenerator = None
    ATFTGATFANModel = None
    LightGBMFinancialBaseline = None
    FinancialGraphBuilder = None

__all__ = [
    "__version__",
    "__author__", 
    "settings",
    "SafeTrainingPipeline",
    "ProductionDatasetV3",
    "MLDatasetBuilder",
    "CrossSectionalNormalizerV2",
    "WalkForwardSplitterV2",
    "QualityFinancialFeaturesGenerator",
    "ATFTGATFANModel",
    "LightGBMFinancialBaseline",
    "FinancialGraphBuilder",
]
