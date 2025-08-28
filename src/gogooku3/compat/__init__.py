"""Backward compatibility layer.

This module provides backward compatibility with existing scripts and APIs.
It contains aliases and wrappers to ensure existing code continues to work
during the migration process.
"""

# Import compatibility components
from .aliases import (
    # Legacy component getters
    get_ml_dataset_builder,
    get_safe_training_pipeline, 
    get_cross_sectional_normalizer_v2,
    get_walk_forward_splitter_v2,
    get_quality_features_generator,
    get_atft_gat_fan_model,
    get_lightgbm_baseline,
    get_financial_graph_builder,
    
    # Direct aliases
    MLDatasetBuilder,
    SafeTrainingPipeline,
    CrossSectionalNormalizerV2,
    WalkForwardSplitterV2, 
    QualityFinancialFeaturesGenerator,
    ATFTGATFANModel,
    LightGBMFinancialBaseline,
    FinancialGraphBuilder,
    
    # Legacy script functions
    run_safe_training_legacy,
    build_ml_dataset_legacy,
    
    # Configuration mapping
    get_config_path,
    CONFIG_PATH_MAPPING,
)

from .script_wrappers import (
    # Script wrappers
    run_safe_training_wrapper,
    ml_dataset_builder_wrapper,
    train_atft_wrapper,
    
    # Entry points
    safe_training_main,
    ml_dataset_main, 
    train_atft_main,
    
    # Utilities
    check_compatibility,
    print_migration_guide,
)

__all__ = [
    # Component getters
    "get_ml_dataset_builder",
    "get_safe_training_pipeline", 
    "get_cross_sectional_normalizer_v2",
    "get_walk_forward_splitter_v2",
    "get_quality_features_generator",
    "get_atft_gat_fan_model",
    "get_lightgbm_baseline",
    "get_financial_graph_builder",
    
    # Direct aliases (for imports)
    "MLDatasetBuilder",
    "SafeTrainingPipeline",
    "CrossSectionalNormalizerV2",
    "WalkForwardSplitterV2", 
    "QualityFinancialFeaturesGenerator",
    "ATFTGATFANModel",
    "LightGBMFinancialBaseline",
    "FinancialGraphBuilder",
    
    # Legacy functions
    "run_safe_training_legacy",
    "build_ml_dataset_legacy",
    
    # Script wrappers
    "run_safe_training_wrapper",
    "ml_dataset_builder_wrapper", 
    "train_atft_wrapper",
    
    # Entry points
    "safe_training_main",
    "ml_dataset_main",
    "train_atft_main", 
    
    # Utilities
    "check_compatibility",
    "print_migration_guide",
    "get_config_path",
    "CONFIG_PATH_MAPPING",
]