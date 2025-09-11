"""Compatibility aliases for backward compatibility with legacy code."""

import warnings
from typing import Any, Dict, Optional, Callable

from ..models.atft_gat_fan import ATFTGATFANModel
from ..models.lightgbm_baseline import LightGBMFinancialBaseline
from ..data.loaders import MLDatasetBuilder, ProductionDatasetV3
from ..training.safe_training_pipeline import SafeTrainingPipeline
from ..features.quality_generator import QualityFinancialFeaturesGenerator
from ..graph.builder import FinancialGraphBuilder


def _deprecation_warning(old_name: str, new_name: str) -> None:
    """Issue deprecation warning for legacy imports."""
    warnings.warn(
        f"{old_name} is deprecated. Use {new_name} instead.",
        DeprecationWarning,
        stacklevel=3
    )


class LegacyATFTGATFANModel(ATFTGATFANModel):
    """Legacy alias for ATFTGATFANModel."""
    
    def __init__(self, *args, **kwargs):
        _deprecation_warning(
            "LegacyATFTGATFANModel", 
            "gogooku3.models.ATFTGATFANModel"
        )
        super().__init__(*args, **kwargs)


class LegacyMLDatasetBuilder(MLDatasetBuilder):
    """Legacy alias for MLDatasetBuilder."""
    
    def __init__(self, *args, **kwargs):
        _deprecation_warning(
            "LegacyMLDatasetBuilder",
            "gogooku3.data.loaders.MLDatasetBuilder"
        )
        super().__init__(*args, **kwargs)


def get_ml_dataset_builder() -> Callable:
    """Get ML dataset builder class for backward compatibility."""
    _deprecation_warning(
        "get_ml_dataset_builder()",
        "from gogooku3.data.loaders import MLDatasetBuilder"
    )
    return MLDatasetBuilder


def get_safe_training_pipeline() -> Callable:
    """Get safe training pipeline class for backward compatibility."""
    _deprecation_warning(
        "get_safe_training_pipeline()",
        "from gogooku3.training import SafeTrainingPipeline"
    )
    return SafeTrainingPipeline


def check_compatibility() -> Dict[str, bool]:
    """Check compatibility status of all components."""
    _deprecation_warning(
        "check_compatibility()",
        "Direct imports from gogooku3 modules"
    )
    
    status = {}
    
    try:
        from ..models import ATFTGATFANModel, LightGBMFinancialBaseline
        status["models"] = True
    except ImportError:
        status["models"] = False
    
    try:
        from ..data.loaders import MLDatasetBuilder, ProductionDatasetV3
        status["data_loaders"] = True
    except ImportError:
        status["data_loaders"] = False
    
    try:
        from ..training import SafeTrainingPipeline
        status["training"] = True
    except ImportError:
        status["training"] = False
    
    try:
        from ..features import QualityFinancialFeaturesGenerator
        status["features"] = True
    except ImportError:
        status["features"] = False
    
    try:
        from ..graph import FinancialGraphBuilder
        status["graph"] = True
    except ImportError:
        status["graph"] = False
    
    return status


__all__ = [
    "ATFTGATFANModel",
    "LightGBMFinancialBaseline",
    "MLDatasetBuilder", 
    "ProductionDatasetV3",
    "SafeTrainingPipeline",
    "QualityFinancialFeaturesGenerator",
    "FinancialGraphBuilder",
    "LegacyATFTGATFANModel",
    "LegacyMLDatasetBuilder",
    "get_ml_dataset_builder",
    "get_safe_training_pipeline",
    "check_compatibility"
]
