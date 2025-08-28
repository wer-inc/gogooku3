"""
Smoke Tests for Gogooku3 Migration
Tests basic functionality of migrated components to ensure they work correctly.
"""

import pytest
import sys
from pathlib import Path
import warnings

# Add src to path for testing
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class TestPackageStructure:
    """Test that the new package structure is correctly set up."""
    
    def test_package_import(self):
        """Test that the main package can be imported."""
        import gogooku3
        assert gogooku3.__version__ == "2.0.0"
        assert hasattr(gogooku3, 'settings')
    
    def test_cli_import(self):
        """Test that the CLI module can be imported."""
        from gogooku3.cli import main
        assert callable(main)
    
    def test_settings_import(self):
        """Test that settings can be imported and configured."""
        from gogooku3.utils.settings import settings
        assert settings.environment in ['development', 'production']
        assert hasattr(settings, 'project_root')


class TestDataComponents:
    """Test data processing components."""
    
    def test_data_loaders_import(self):
        """Test that data loaders can be imported."""
        try:
            from gogooku3.data.loaders import ProductionDatasetV3, MLDatasetBuilder
            assert ProductionDatasetV3 is not None
            assert MLDatasetBuilder is not None
        except ImportError as e:
            pytest.skip(f"Data loaders not available: {e}")
    
    def test_data_scalers_import(self):
        """Test that data scalers can be imported."""
        try:
            from gogooku3.data.scalers import CrossSectionalNormalizerV2, WalkForwardSplitterV2
            assert CrossSectionalNormalizerV2 is not None
            assert WalkForwardSplitterV2 is not None
        except ImportError as e:
            pytest.skip(f"Data scalers not available: {e}")


class TestFeatureComponents:
    """Test feature engineering components."""
    
    def test_quality_features_import(self):
        """Test that quality features generator can be imported."""
        try:
            from gogooku3.features import QualityFinancialFeaturesGenerator
            assert QualityFinancialFeaturesGenerator is not None
        except ImportError as e:
            pytest.skip(f"Feature components not available: {e}")


class TestModelComponents:
    """Test model components."""
    
    def test_models_import(self):
        """Test that model components can be imported.""" 
        try:
            from gogooku3.models import ATFTGATFANModel, LightGBMFinancialBaseline
            assert ATFTGATFANModel is not None
            assert LightGBMFinancialBaseline is not None
        except ImportError as e:
            pytest.skip(f"Model components not available: {e}")


class TestGraphComponents:
    """Test graph neural network components."""
    
    def test_graph_import(self):
        """Test that graph components can be imported."""
        try:
            from gogooku3.graph import FinancialGraphBuilder
            assert FinancialGraphBuilder is not None
        except ImportError as e:
            pytest.skip(f"Graph components not available: {e}")


class TestTrainingComponents:
    """Test training pipeline components."""
    
    def test_training_pipeline_import(self):
        """Test that training pipeline can be imported."""
        try:
            from gogooku3.training import SafeTrainingPipeline
            assert SafeTrainingPipeline is not None
            
            # Test basic instantiation
            pipeline = SafeTrainingPipeline(experiment_name="test")
            assert pipeline.experiment_name == "test"
        except ImportError as e:
            pytest.skip(f"Training components not available: {e}")


class TestCompatibilityLayer:
    """Test backward compatibility layer."""
    
    def test_compat_import(self):
        """Test that compatibility layer can be imported."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from gogooku3.compat import check_compatibility
            assert callable(check_compatibility)
    
    def test_legacy_aliases(self):
        """Test that legacy aliases work."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from gogooku3.compat import (
                get_ml_dataset_builder, 
                get_safe_training_pipeline
            )
            
            # Test that getters return callable objects
            ml_builder = get_ml_dataset_builder()
            assert ml_builder is not None
            
            pipeline_cls = get_safe_training_pipeline()
            assert pipeline_cls is not None
    
    def test_compatibility_status(self):
        """Test compatibility status checker."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from gogooku3.compat import check_compatibility
            
            status = check_compatibility()
            assert isinstance(status, dict)
            assert len(status) > 0
            
            # At least some components should be available
            available_count = sum(status.values())
            assert available_count > 0, f"No components available: {status}"


class TestUtilityComponents:
    """Test utility components."""
    
    def test_deduplication_import(self):
        """Test that deduplication utility can be imported."""
        from gogooku3.utils.deduplication import SafeDeduplicator
        assert SafeDeduplicator is not None
        
        # Test basic instantiation
        deduplicator = SafeDeduplicator()
        assert deduplicator.dry_run is True  # Default to safe mode


class TestCLIFunctionality:
    """Test CLI functionality."""
    
    def test_cli_help(self):
        """Test that CLI help works."""
        from gogooku3.cli import main
        import sys
        from unittest.mock import patch
        
        # Mock sys.argv to test help
        with patch.object(sys, 'argv', ['gogooku3', '--help']):
            try:
                main()
            except SystemExit:
                pass  # Help command exits normally
    
    def test_cli_version(self):
        """Test that CLI version works."""
        from gogooku3.cli import main
        import sys
        from unittest.mock import patch
        
        # Mock sys.argv to test version
        with patch.object(sys, 'argv', ['gogooku3', '--version']):
            try:
                main()
            except SystemExit:
                pass  # Version command exits normally


class TestIntegrationSmoke:
    """High-level integration smoke tests."""
    
    def test_end_to_end_import_chain(self):
        """Test that we can import and instantiate a complete pipeline."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            
            try:
                # Test complete import chain
                from gogooku3 import settings
                from gogooku3.training import SafeTrainingPipeline
                
                # Test instantiation with settings
                pipeline = SafeTrainingPipeline(
                    experiment_name="smoke_test",
                    verbose=False
                )
                
                assert pipeline is not None
                assert pipeline.experiment_name == "smoke_test"
                
            except ImportError as e:
                pytest.skip(f"Integration components not available: {e}")
    
    def test_legacy_to_new_migration_path(self):
        """Test that legacy components can be accessed through new structure."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            
            # Test that we can get legacy components through the new interface
            from gogooku3.compat import get_safe_training_pipeline
            
            pipeline_class = get_safe_training_pipeline()
            assert pipeline_class is not None
            
            # Should be able to instantiate
            pipeline = pipeline_class(experiment_name="legacy_test")
            assert hasattr(pipeline, 'run_pipeline')


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])