"""Compatibility layer for backward compatibility."""

from .aliases import (
    ATFTGATFANModel,
    LightGBMFinancialBaseline,
    MLDatasetBuilder,
    ProductionDatasetV3,
    SafeTrainingPipeline,
    QualityFinancialFeaturesGenerator,
    FinancialGraphBuilder,
    get_ml_dataset_builder,
    get_safe_training_pipeline,
    check_compatibility
)

__all__ = [
    "ATFTGATFANModel",
    "LightGBMFinancialBaseline", 
    "MLDatasetBuilder",
    "ProductionDatasetV3",
    "SafeTrainingPipeline",
    "QualityFinancialFeaturesGenerator",
    "FinancialGraphBuilder",
    "get_ml_dataset_builder",
    "get_safe_training_pipeline",
    "check_compatibility"
]
