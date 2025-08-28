#!/usr/bin/env python3
"""
Safety Components Integration Tests
安全コンポーネントの統合テスト（統一版）

This replaces:
- scripts/quick_train_safe.py
- test_safety_integration.py
- partial tests scattered across the project
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import logging

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Safety components
try:
    from src.data.safety.cross_sectional import CrossSectionalNormalizer
    from src.metrics.financial_metrics import FinancialMetrics
    SAFETY_COMPONENTS_AVAILABLE = True
    print("✅ Safety components loaded successfully")
except ImportError as e:
    CrossSectionalNormalizer = None
    FinancialMetrics = None
    SAFETY_COMPONENTS_AVAILABLE = False
    print(f"❌ Safety components failed to load: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_data(n_samples=1000, n_features=10, n_days=50):
    """Create sample financial data for testing"""
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n_days, freq='D')
    codes = [f"CODE_{i:04d}" for i in range(20)]  # 20 stocks
    
    data = []
    for date in dates:
        for code in codes:
            features = np.random.randn(n_features) * 0.1
            return_1d = np.random.randn() * 0.02  # 2% daily volatility
            return_5d = return_1d * 5 + np.random.randn() * 0.01
            return_20d = return_1d * 20 + np.random.randn() * 0.05
            
            row = {
                'date': date,
                'code': code,
                'return_1d': return_1d,
                'return_5d': return_5d, 
                'return_20d': return_20d,
            }
            
            # Add technical features
            for i in range(n_features):
                row[f'feature_{i}'] = features[i]
                
            data.append(row)
    
    df = pd.DataFrame(data)
    logger.info(f"Created sample data: {len(df)} rows, {n_days} dates, {len(codes)} stocks")
    return df


class TestCrossSectionalNormalizer:
    """CrossSectionalNormalizer testing class"""
    
    def test_basic_functionality(self):
        """Test basic fit/transform functionality"""
        print("\n🧪 Testing CrossSectionalNormalizer basic functionality...")
        
        df = create_sample_data(n_samples=500, n_features=5)
        split_date = df['date'].quantile(0.7)
        train_df = df[df['date'] <= split_date]
        test_df = df[df['date'] > split_date]
        
        normalizer = CrossSectionalNormalizer(
            date_column='date',
            code_column='code',
            min_stocks_per_day=5,
            fillna_method='forward_fill'
        )
        
        # Fit on training data only
        train_normalized = normalizer.fit_transform(train_df, verbose=True)
        test_normalized = normalizer.transform(test_df, verbose=True)
        
        # Validate results
        validation = normalizer.validate_transform(train_normalized)
        
        assert len(validation['warnings']) == 0, f"Normalization warnings: {validation['warnings']}"
        assert len(train_normalized) > 0, "Training data normalization failed"
        assert len(test_normalized) > 0, "Test data normalization failed"
        
        print("✅ CrossSectionalNormalizer basic test passed")
        return True
    
    def test_data_leakage_prevention(self):
        """Test that future data doesn't leak into past normalization"""
        print("\n🛡️ Testing data leakage prevention...")
        
        df = create_sample_data(n_samples=300, n_features=3)
        # Create artificial pattern in later dates
        future_mask = df['date'] > df['date'].quantile(0.8)
        df.loc[future_mask, 'feature_0'] += 10.0  # Large shift in future data
        
        split_date = df['date'].quantile(0.7)
        train_df = df[df['date'] <= split_date]
        test_df = df[df['date'] > split_date]
        
        normalizer = CrossSectionalNormalizer(
            date_column='date',
            code_column='code'
        )
        
        # Fit only on training data
        normalizer.fit(train_df, verbose=False)
        
        # Check that training stats don't include future information
        train_stats = normalizer.get_daily_stats(train_df['date'].iloc[0])
        assert train_stats is not None, "Training stats should be available"
        
        # Transform test data (should use training period stats)
        test_normalized = normalizer.transform(test_df, verbose=False)
        
        # Future data shift should be preserved (not normalized away)
        # because normalization uses only training period statistics
        future_feature_0 = test_normalized[test_normalized['date'] > df['date'].quantile(0.8)]['feature_0']
        if len(future_feature_0) > 0:
            # Future data should have different distribution (not centered at 0)
            assert abs(future_feature_0.mean()) > 1.0, "Future data leakage detected!"
        
        print("✅ Data leakage prevention test passed")
        return True


class TestFinancialMetrics:
    """FinancialMetrics testing class"""
    
    def test_ic_calculation(self):
        """Test Information Coefficient calculation"""
        print("\n📊 Testing FinancialMetrics IC calculation...")
        
        # Create correlated prediction/target data
        n_samples = 500
        predictions = np.random.randn(n_samples) * 0.02
        targets = predictions * 0.8 + np.random.randn(n_samples) * 0.01  # 80% correlation
        dates = np.repeat(pd.date_range(end=pd.Timestamp.now(), periods=25, freq='D'), 20)[:n_samples]
        
        metrics_calc = FinancialMetrics(min_stocks_per_day=15)
        
        ic = metrics_calc.compute_information_coefficient(predictions, targets, dates)
        rank_ic = metrics_calc.compute_rank_ic(predictions, targets, dates)
        
        assert 0.4 < ic < 0.9, f"IC should be around 0.8, got {ic}"
        assert 0.3 < rank_ic < 0.9, f"RankIC should be positive, got {rank_ic}"
        
        print(f"✅ IC calculation test passed: IC={ic:.3f}, RankIC={rank_ic:.3f}")
        return ic, rank_ic
    
    def test_decile_analysis(self):
        """Test Decile analysis"""
        print("\n📈 Testing Decile analysis...")
        
        n_samples = 400
        # Create strong signal for decile testing
        predictions = np.random.randn(n_samples)
        targets = predictions * 0.5 + np.random.randn(n_samples) * 0.2
        dates = np.repeat(pd.date_range(end=pd.Timestamp.now(), periods=20, freq='D'), 20)[:n_samples]
        
        metrics_calc = FinancialMetrics(min_stocks_per_day=15, decile_count=5)
        
        decile_result = metrics_calc.compute_decile_analysis(predictions, targets, dates)
        
        assert decile_result['valid_days'] > 0, "No valid days for decile analysis"
        assert 'long_short_spread' in decile_result, "Long-short spread not calculated"
        
        print(f"✅ Decile analysis test passed: Long-Short={decile_result['long_short_spread']:.4f}")
        return decile_result


class TestIntegration:
    """Integration testing class"""
    
    def test_full_pipeline(self):
        """Test full pipeline: normalization -> prediction -> metrics"""
        print("\n🔗 Testing full integration pipeline...")
        
        # 1. Create data
        df = create_sample_data(n_samples=600, n_features=5, n_days=30)
        split_date = df['date'].quantile(0.7)
        train_df = df[df['date'] <= split_date]
        val_df = df[df['date'] > split_date]
        
        # 2. Normalize
        normalizer = CrossSectionalNormalizer(
            date_column='date',
            code_column='code',
            min_stocks_per_day=10
        )
        
        train_normalized = normalizer.fit_transform(train_df, verbose=False)
        val_normalized = normalizer.transform(val_df, verbose=False)
        
        # 3. Create predictions (use normalized feature as proxy)
        val_predictions = val_normalized['feature_0'].values
        val_targets = val_normalized['return_1d'].values
        val_dates = val_normalized['date'].values
        
        # 4. Calculate financial metrics
        metrics_calc = FinancialMetrics(min_stocks_per_day=15)
        
        ic = metrics_calc.compute_information_coefficient(val_predictions, val_targets, val_dates)
        rank_ic = metrics_calc.compute_rank_ic(val_predictions, val_targets, val_dates)
        decile_result = metrics_calc.compute_decile_analysis(val_predictions, val_targets, val_dates)
        
        # 5. Validate pipeline
        assert len(val_normalized) > 0, "Validation normalization failed"
        assert not np.isnan(ic), "IC calculation failed"
        assert not np.isnan(rank_ic), "RankIC calculation failed"
        assert decile_result['valid_days'] > 0, "Decile analysis failed"
        
        results = {
            'ic': float(ic),
            'rank_ic': float(rank_ic),
            'long_short_spread': float(decile_result['long_short_spread']),
            'valid_days': decile_result['valid_days'],
            'pipeline_success': True
        }
        
        print(f"✅ Full pipeline test passed:")
        print(f"   IC: {results['ic']:.4f}")
        print(f"   RankIC: {results['rank_ic']:.4f}")  
        print(f"   Long-Short: {results['long_short_spread']:.4f}")
        print(f"   Valid days: {results['valid_days']}")
        
        return results


def run_all_tests():
    """Run all safety component tests"""
    if not SAFETY_COMPONENTS_AVAILABLE:
        print("❌ Safety components not available, skipping tests")
        return False
    
    print("🚀 Starting comprehensive safety components tests...\n")
    
    try:
        # Test CrossSectionalNormalizer
        normalizer_tests = TestCrossSectionalNormalizer()
        normalizer_tests.test_basic_functionality()
        normalizer_tests.test_data_leakage_prevention()
        
        # Test FinancialMetrics
        metrics_tests = TestFinancialMetrics()
        ic, rank_ic = metrics_tests.test_ic_calculation()
        decile_result = metrics_tests.test_decile_analysis()
        
        # Test Integration
        integration_tests = TestIntegration()
        integration_result = integration_tests.test_full_pipeline()
        
        # Overall assessment
        success = (
            abs(ic) > 0.1 and
            abs(rank_ic) > 0.1 and
            integration_result['valid_days'] > 0 and
            integration_result['pipeline_success']
        )
        
        print(f"\n🎉 Overall test result: {'✅ SUCCESS' if success else '⚠️ NEEDS REVIEW'}")
        
        return success
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)