"""
Smoke test for basic functionality verification.
Tests core imports and basic pipeline initialization.
"""

import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_core_imports():
    """Test that core modules can be imported."""
    logger.info("Testing core imports...")
    
    try:
        from src.gogooku3.models import ATFTGATFANModel, LightGBMFinancialBaseline
        logger.info("✅ Model imports successful")
        
        from src.gogooku3.data.loaders import MLDatasetBuilder, ProductionDatasetV3
        logger.info("✅ Data loader imports successful")
        
        from src.gogooku3.training import SafeTrainingPipeline
        logger.info("✅ Training pipeline import successful")
        
        from src.gogooku3.features import QualityFinancialFeaturesGenerator
        logger.info("✅ Feature generator import successful")
        
        from src.gogooku3.graph import FinancialGraphBuilder
        logger.info("✅ Graph builder import successful")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Core import failed: {e}")
        return False


def test_model_instantiation():
    """Test that models can be instantiated."""
    logger.info("Testing model instantiation...")
    
    try:
        from src.gogooku3.models import ATFTGATFANModel, LightGBMFinancialBaseline
        
        atft_model = ATFTGATFANModel(input_dim=145, hidden_dim=256)
        logger.info(f"✅ ATFT-GAT-FAN model created: {type(atft_model).__name__}")
        
        lgb_model = LightGBMFinancialBaseline(n_estimators=10)
        logger.info(f"✅ LightGBM baseline created: {type(lgb_model).__name__}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model instantiation failed: {e}")
        return False


def test_pipeline_initialization():
    """Test that pipelines can be initialized."""
    logger.info("Testing pipeline initialization...")
    
    try:
        from src.gogooku3.training import SafeTrainingPipeline
        pipeline = SafeTrainingPipeline(experiment_name="smoke_test")
        logger.info(f"✅ SafeTrainingPipeline initialized: {pipeline.experiment_name}")
        
        from src.gogooku3.data.loaders import MLDatasetBuilder
        builder = MLDatasetBuilder()
        logger.info("✅ MLDatasetBuilder initialized")
        
        from src.gogooku3.features import QualityFinancialFeaturesGenerator
        generator = QualityFinancialFeaturesGenerator()
        logger.info("✅ QualityFinancialFeaturesGenerator initialized")
        
        from src.gogooku3.graph import FinancialGraphBuilder
        graph_builder = FinancialGraphBuilder()
        logger.info("✅ FinancialGraphBuilder initialized")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline initialization failed: {e}")
        return False


def test_legacy_compatibility():
    """Test legacy compatibility layer."""
    logger.info("Testing legacy compatibility...")
    
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            
            from src.gogooku3.compat import (
                get_ml_dataset_builder,
                get_safe_training_pipeline,
                check_compatibility
            )
            
            ml_builder_cls = get_ml_dataset_builder()
            pipeline_cls = get_safe_training_pipeline()
            status = check_compatibility()
            
            logger.info(f"✅ Compatibility check: {sum(status.values())}/{len(status)} components available")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Legacy compatibility test failed: {e}")
        return False


def test_core_script_imports():
    """Test that core scripts can import their dependencies."""
    logger.info("Testing core script imports...")
    
    try:
        from src.data.jquants.fetcher import JQuantsAsyncFetcher
        logger.info("✅ JQuantsAsyncFetcher import successful")
        
        from src.data.loaders.production_loader_v2 import ProductionDatasetV2
        logger.info("✅ ProductionDatasetV2 import successful")
        
        from src.data.samplers import DayBatchSampler
        logger.info("✅ DayBatchSampler import successful")
        
        from src.data.validation import NormalizationValidator
        logger.info("✅ NormalizationValidator import successful")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Core script import failed: {e}")
        return False


def main():
    """Run all smoke tests."""
    logger.info("🔍 Starting smoke tests...")
    
    tests = [
        ("Core Imports", test_core_imports),
        ("Model Instantiation", test_model_instantiation),
        ("Pipeline Initialization", test_pipeline_initialization),
        ("Legacy Compatibility", test_legacy_compatibility),
        ("Core Script Imports", test_core_script_imports)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} ---")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("SMOKE TEST SUMMARY")
    logger.info("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        logger.info("🎉 All smoke tests passed!")
        return 0
    else:
        logger.error("❌ Some smoke tests failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
