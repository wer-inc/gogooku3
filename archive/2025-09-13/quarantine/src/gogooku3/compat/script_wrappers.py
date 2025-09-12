"""
Script Wrappers for Legacy Compatibility
Provides wrapper functions for existing scripts to maintain compatibility.
"""

import sys
import warnings
from pathlib import Path
from typing import Any, Dict

# Import the compatibility layer
from . import aliases


def wrap_script_execution(
    script_name: str, 
    new_component: str, 
    wrapper_func: callable,
    *args, **kwargs
) -> Any:
    """Generic wrapper for script execution with deprecation warning."""
    aliases.deprecation_warning(script_name, new_component)
    
    try:
        return wrapper_func(*args, **kwargs)
    except Exception as e:
        print(f"Error executing {script_name}: {e}")
        print(f"Please migrate to {new_component}")
        raise


def run_safe_training_wrapper(*args, **kwargs):
    """Wrapper for scripts/run_safe_training.py"""
    from gogooku3.training import SafeTrainingPipeline
    
    # Parse command line arguments if needed
    if args and isinstance(args[0], list):
        # Command line arguments passed
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--verbose", action="store_true")
        parser.add_argument("--n-splits", type=int, default=5)
        parser.add_argument("--memory-limit", type=float, default=8.0)
        parser.add_argument("--experiment-name", type=str, default="safe_training")
        
        parsed_args = parser.parse_args(args[0])
        
        pipeline = SafeTrainingPipeline(
            experiment_name=parsed_args.experiment_name,
            verbose=parsed_args.verbose
        )
        
        return pipeline.run_pipeline(
            n_splits=parsed_args.n_splits,
            memory_limit_gb=parsed_args.memory_limit
        )
    
    else:
        # Direct function call
        pipeline = SafeTrainingPipeline(**kwargs)
        return pipeline.run_pipeline(*args, **kwargs)


def ml_dataset_builder_wrapper(*args, **kwargs):
    """Wrapper for scripts/data/ml_dataset_builder.py"""
    from gogooku3.data.loaders import MLDatasetBuilder
    
    builder = MLDatasetBuilder(**kwargs)
    return builder.build_dataset(*args, **kwargs)


def train_atft_wrapper(*args, **kwargs):
    """Wrapper for scripts/train_atft.py"""
    from gogooku3.training import SafeTrainingPipeline
    from gogooku3.models import ATFTGATFANModel
    
    # This would need specific implementation based on train_atft.py structure
    pipeline = SafeTrainingPipeline(**kwargs)
    return pipeline.run_pipeline(*args, **kwargs)


# ============================================================================
# Script Entry Points (can be called from command line)
# ============================================================================

def safe_training_main():
    """Main entry point for run_safe_training.py wrapper."""
    import sys
    return wrap_script_execution(
        "scripts/run_safe_training.py",
        "gogooku3.training.SafeTrainingPipeline",
        run_safe_training_wrapper,
        sys.argv[1:]
    )


def ml_dataset_main(): 
    """Main entry point for ml_dataset_builder.py wrapper."""
    import sys
    return wrap_script_execution(
        "scripts/data/ml_dataset_builder.py", 
        "gogooku3.data.loaders.MLDatasetBuilder",
        ml_dataset_builder_wrapper,
        sys.argv[1:]
    )


def train_atft_main():
    """Main entry point for train_atft.py wrapper.""" 
    import sys
    return wrap_script_execution(
        "scripts/train_atft.py",
        "gogooku3.models.ATFTGATFANModel + gogooku3.training.SafeTrainingPipeline", 
        train_atft_wrapper,
        sys.argv[1:]
    )


# ============================================================================
# Utility Functions
# ============================================================================

def check_compatibility() -> Dict[str, bool]:
    """Check if all compatibility components are available."""
    compatibility_status = {}
    
    components = {
        "MLDatasetBuilder": ("gogooku3.data.loaders", "MLDatasetBuilder"),
        "SafeTrainingPipeline": ("gogooku3.training", "SafeTrainingPipeline"),  
        "CrossSectionalNormalizerV2": ("gogooku3.data.scalers", "CrossSectionalNormalizerV2"),
        "WalkForwardSplitterV2": ("gogooku3.data.scalers", "WalkForwardSplitterV2"),
        "QualityFinancialFeaturesGenerator": ("gogooku3.features", "QualityFinancialFeaturesGenerator"),
        "ATFTGATFANModel": ("gogooku3.models", "ATFTGATFANModel"),
        "LightGBMFinancialBaseline": ("gogooku3.models", "LightGBMFinancialBaseline"),
        "FinancialGraphBuilder": ("gogooku3.graph", "FinancialGraphBuilder"),
    }
    
    for component_name, (module_name, class_name) in components.items():
        try:
            module = __import__(module_name, fromlist=[class_name])
            component = getattr(module, class_name)
            compatibility_status[component_name] = True
        except (ImportError, AttributeError) as e:
            compatibility_status[component_name] = False
            print(f"⚠️ {component_name} not available: {e}")
    
    return compatibility_status


def print_migration_guide():
    """Print a migration guide for users."""
    print("""
🚀 Gogooku3 Migration Guide

Your scripts are using the compatibility layer. To fully migrate:

1. Update import statements:
   OLD: from scripts.run_safe_training import SafeTrainingPipeline  
   NEW: from gogooku3.training import SafeTrainingPipeline

2. Update script calls:
   OLD: python scripts/run_safe_training.py
   NEW: python -m gogooku3.cli train

3. Update configuration paths:
   OLD: configs/atft/data/jpx_safe.yaml
   NEW: configs/data/jpx_safe.yaml

4. Use new package structure:
   - Data components: gogooku3.data.loaders, gogooku3.data.scalers
   - Models: gogooku3.models  
   - Training: gogooku3.training
   - Features: gogooku3.features
   - Graph: gogooku3.graph

See MIGRATION.md for detailed migration instructions.
""")


if __name__ == "__main__":
    # Show compatibility status
    status = check_compatibility() 
    
    available = sum(status.values())
    total = len(status)
    
    print(f"🔧 Compatibility Status: {available}/{total} components available")
    
    if available < total:
        print("⚠️ Some components are missing. Migration may be incomplete.")
        print_migration_guide()
    else:
        print("✅ All components available. Ready for migration!")
        print_migration_guide()