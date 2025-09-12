"""
Backward Compatibility Aliases
Provides aliases for existing scripts and functions to maintain compatibility during migration.
"""

import warnings
from pathlib import Path
import sys

# Add the project root to the path for backward compatibility
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def deprecation_warning(old_name: str, new_name: str):
    """Issue a deprecation warning for renamed components."""
    warnings.warn(
        f"{old_name} is deprecated. Use {new_name} instead.",
        DeprecationWarning,
        stacklevel=3
    )


# ============================================================================
# Data Loaders - Backward Compatibility
# ============================================================================

# Legacy imports from scripts.data.*
def get_ml_dataset_builder():
    """Legacy alias for MLDatasetBuilder."""
    deprecation_warning("scripts.data.ml_dataset_builder", "gogooku3.data.loaders.MLDatasetBuilder")
    from gogooku3.data.loaders import MLDatasetBuilder
    return MLDatasetBuilder

def get_production_dataset_v3():
    """Legacy alias for ProductionDatasetV3.""" 
    deprecation_warning("src.data.loaders.production_loader_v3", "gogooku3.data.loaders.ProductionDatasetV3")
    from gogooku3.data.loaders import ProductionDatasetV3
    return ProductionDatasetV3


# ============================================================================
# Safety Components - Backward Compatibility
# ============================================================================

def get_cross_sectional_normalizer_v2():
    """Legacy alias for CrossSectionalNormalizerV2."""
    deprecation_warning("src.data.safety.cross_sectional_v2", "gogooku3.data.scalers.CrossSectionalNormalizerV2")
    from gogooku3.data.scalers import CrossSectionalNormalizerV2
    return CrossSectionalNormalizerV2

def get_walk_forward_splitter_v2():
    """Legacy alias for WalkForwardSplitterV2."""
    deprecation_warning("src.data.safety.walk_forward_v2", "gogooku3.data.scalers.WalkForwardSplitterV2")
    from gogooku3.data.scalers import WalkForwardSplitterV2
    return WalkForwardSplitterV2


# ============================================================================
# Feature Engineering - Backward Compatibility
# ============================================================================

def get_quality_features_generator():
    """Legacy alias for QualityFinancialFeaturesGenerator."""
    deprecation_warning("src.features.quality_features", "gogooku3.features.QualityFinancialFeaturesGenerator")
    from gogooku3.features import QualityFinancialFeaturesGenerator
    return QualityFinancialFeaturesGenerator


# ============================================================================
# Models - Backward Compatibility
# ============================================================================

def get_atft_gat_fan_model():
    """Legacy alias for ATFTGATFANModel."""
    deprecation_warning("src.models.architectures.atft_gat_fan", "gogooku3.models.ATFTGATFANModel")
    from gogooku3.models import ATFTGATFANModel
    return ATFTGATFANModel

def get_lightgbm_baseline():
    """Legacy alias for LightGBMFinancialBaseline."""
    deprecation_warning("src.models.baseline.lightgbm_baseline", "gogooku3.models.LightGBMFinancialBaseline")
    from gogooku3.models import LightGBMFinancialBaseline
    return LightGBMFinancialBaseline


# ============================================================================
# Graph Components - Backward Compatibility  
# ============================================================================

def get_financial_graph_builder():
    """Legacy alias for FinancialGraphBuilder."""
    deprecation_warning("src.data.utils.graph_builder", "gogooku3.graph.FinancialGraphBuilder")
    from gogooku3.graph import FinancialGraphBuilder
    return FinancialGraphBuilder


# ============================================================================
# Training Pipeline - Backward Compatibility
# ============================================================================

def get_safe_training_pipeline():
    """Legacy alias for SafeTrainingPipeline.""" 
    deprecation_warning("scripts.run_safe_training", "gogooku3.training.SafeTrainingPipeline")
    from gogooku3.training import SafeTrainingPipeline
    return SafeTrainingPipeline


# ============================================================================
# Legacy Script Wrappers
# ============================================================================

class LegacyScriptWrapper:
    """Wrapper for legacy scripts to provide compatibility."""
    
    def __init__(self, script_name: str, new_component: str):
        self.script_name = script_name
        self.new_component = new_component
    
    def __call__(self, *args, **kwargs):
        deprecation_warning(self.script_name, self.new_component)
        # Import and execute the new component
        # This is a placeholder - specific implementations would go here
        print(f"Legacy script {self.script_name} called. Please migrate to {self.new_component}")


# ============================================================================
# Direct Legacy Aliases (for import statements)
# ============================================================================

# These allow direct imports of legacy names
MLDatasetBuilder = get_ml_dataset_builder()
ProductionDatasetV3 = get_production_dataset_v3()
CrossSectionalNormalizerV2 = get_cross_sectional_normalizer_v2() 
WalkForwardSplitterV2 = get_walk_forward_splitter_v2()
QualityFinancialFeaturesGenerator = get_quality_features_generator()
ATFTGATFANModel = get_atft_gat_fan_model()
LightGBMFinancialBaseline = get_lightgbm_baseline()
FinancialGraphBuilder = get_financial_graph_builder()
SafeTrainingPipeline = get_safe_training_pipeline()


# ============================================================================
# Legacy Script Functions (for scripts that call functions directly)
# ============================================================================

def run_safe_training_legacy(*args, **kwargs):
    """Legacy function for scripts/run_safe_training.py compatibility."""
    deprecation_warning("scripts.run_safe_training.main", "gogooku3.training.SafeTrainingPipeline.run_pipeline")
    from gogooku3.training import SafeTrainingPipeline
    pipeline = SafeTrainingPipeline()
    return pipeline.run_pipeline(*args, **kwargs)

def build_ml_dataset_legacy(*args, **kwargs):
    """Legacy function for scripts/data/ml_dataset_builder.py compatibility."""
    deprecation_warning("scripts.data.ml_dataset_builder.main", "gogooku3.data.loaders.MLDatasetBuilder")
    from gogooku3.data.loaders import MLDatasetBuilder
    builder = MLDatasetBuilder()
    return builder.build_dataset(*args, **kwargs)


# ============================================================================
# Configuration Path Mapping
# ============================================================================

CONFIG_PATH_MAPPING = {
    # Old path -> New path
    "configs/atft/data/jpx_safe.yaml": "configs/data/jpx_safe.yaml", 
    "configs/atft/model/atft_gat_fan.yaml": "configs/model/atft_gat_fan.yaml",
    "configs/atft/train/production.yaml": "configs/training/production.yaml",
    "configs/atft/train/walk_forward.yaml": "configs/training/walk_forward.yaml",
    "configs/atft/hardware/default.yaml": "configs/hardware/default.yaml",
}

def get_config_path(legacy_path: str) -> str:
    """Map legacy config paths to new paths."""
    new_path = CONFIG_PATH_MAPPING.get(legacy_path, legacy_path)
    if new_path != legacy_path:
        deprecation_warning(f"config path {legacy_path}", f"config path {new_path}")
    return new_path


# ============================================================================
# Import Path Mapping
# ============================================================================

def setup_legacy_imports():
    """Set up legacy import paths for backward compatibility."""
    import sys
    from types import ModuleType
    
    # Create pseudo-modules for legacy imports
    legacy_modules = {
        'scripts.data.ml_dataset_builder': ModuleType('ml_dataset_builder'),
        'scripts.run_safe_training': ModuleType('run_safe_training'),
        'src.data.safety.cross_sectional_v2': ModuleType('cross_sectional_v2'),
        'src.data.safety.walk_forward_v2': ModuleType('walk_forward_v2'),
        'src.data.loaders.production_loader_v3': ModuleType('production_loader_v3'),
        'src.features.quality_features': ModuleType('quality_features'),
        'src.models.baseline.lightgbm_baseline': ModuleType('lightgbm_baseline'),
        'src.data.utils.graph_builder': ModuleType('graph_builder'),
    }
    
    # Add legacy attributes to the modules
    legacy_modules['scripts.data.ml_dataset_builder'].MLDatasetBuilder = MLDatasetBuilder
    legacy_modules['scripts.run_safe_training'].SafeTrainingPipeline = SafeTrainingPipeline
    legacy_modules['scripts.run_safe_training'].main = run_safe_training_legacy
    legacy_modules['src.data.safety.cross_sectional_v2'].CrossSectionalNormalizerV2 = CrossSectionalNormalizerV2
    legacy_modules['src.data.safety.walk_forward_v2'].WalkForwardSplitterV2 = WalkForwardSplitterV2
    legacy_modules['src.data.loaders.production_loader_v3'].ProductionDatasetV3 = ProductionDatasetV3
    legacy_modules['src.features.quality_features'].QualityFinancialFeaturesGenerator = QualityFinancialFeaturesGenerator
    legacy_modules['src.models.baseline.lightgbm_baseline'].LightGBMFinancialBaseline = LightGBMFinancialBaseline
    legacy_modules['src.data.utils.graph_builder'].FinancialGraphBuilder = FinancialGraphBuilder
    
    # Register legacy modules in sys.modules
    for module_name, module_obj in legacy_modules.items():
        sys.modules[module_name] = module_obj


# ============================================================================
# Auto-setup (runs when module is imported)
# ============================================================================

# Automatically set up legacy imports when this module is imported
setup_legacy_imports()

# Show a general deprecation notice
warnings.warn(
    "You are using gogooku3 compatibility layer. "
    "Please migrate to the new gogooku3 package structure. "
    "See MIGRATION.md for details.",
    DeprecationWarning,
    stacklevel=2
)