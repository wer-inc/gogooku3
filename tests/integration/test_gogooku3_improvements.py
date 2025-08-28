#!/usr/bin/env python3
"""
Gogooku3 Improvements Integration Test
gogooku3改善点の統合テスト

テスト対象:
- Cross-sectional Z-score正規化（Polarsベース）
- Walk-Forward + Embargo=20
- Parquet lazy scan + 列射影
- 系列相関グラフ構築
- GBMベースライン
- 高品質特徴量生成
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import polars as pl
import torch
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Any
import tempfile
import shutil

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 改善コンポーネントをインポート
try:
    from src.data.safety.cross_sectional_v2 import CrossSectionalNormalizerV2
    from src.data.safety.walk_forward_v2 import WalkForwardSplitterV2
    from src.data.loaders.production_loader_v3 import ProductionDatasetV3
    from src.data.utils.graph_builder import FinancialGraphBuilder
    from src.models.baseline.lightgbm_baseline import LightGBMFinancialBaseline
    from src.features.quality_features import QualityFinancialFeaturesGenerator
    COMPONENTS_AVAILABLE = True
    print("✅ All improvement components loaded successfully")
except ImportError as e:
    COMPONENTS_AVAILABLE = False
    print(f"❌ Component loading failed: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_synthetic_financial_data(n_stocks=50, n_days=300) -> pd.DataFrame:
    """合成金融データを作成"""
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n_days, freq='D')
    codes = [f"STOCK_{i:04d}" for i in range(n_stocks)]
    
    data = []
    
    # 各銘柄のランダムウォーク
    for code in codes:
        price = 1000.0  # 初期価格
        volume_base = np.random.uniform(100000, 1000000)
        
        for date in dates:
            # 価格のランダムウォーク
            daily_return = np.random.normal(0.0001, 0.02)  # 平均0.01%、標準偏差2%
            price *= (1 + daily_return)
            
            # 出来高（価格変動と相関）
            volume = volume_base * (1 + np.abs(daily_return) * 5 + np.random.normal(0, 0.3))
            volume = max(volume, 1000)  # 最小出来高
            
            # OHLC
            high = price * (1 + np.abs(np.random.normal(0, 0.005)))
            low = price * (1 - np.abs(np.random.normal(0, 0.005)))
            open_price = price * (1 + np.random.normal(0, 0.003))
            
            # テクニカル指標（簡易版）
            rsi = np.random.uniform(20, 80)
            ema_20 = price * (1 + np.random.normal(0, 0.01))
            volatility = np.abs(np.random.normal(0.02, 0.01))
            
            # リターン系
            returns_1d = daily_return
            returns_5d = daily_return * 5 + np.random.normal(0, 0.01)
            returns_20d = daily_return * 20 + np.random.normal(0, 0.05)
            
            row = {
                'date': date,
                'code': code,
                'adjustment_close': price,
                'adjustment_open': open_price,
                'adjustment_high': high,
                'adjustment_low': low,
                'adjustment_volume': volume,
                'turnover_value': price * volume,
                'return_1d': returns_1d,
                'return_5d': returns_5d,
                'return_10d': returns_5d * 2 + np.random.normal(0, 0.02),
                'return_20d': returns_20d,
                'rsi14': rsi,
                'ema20': ema_20,
                'volatility_20d': volatility,
                # 特徴量候補
                'feat_ret_1d': daily_return,
                'feat_vol': volatility,
                'feat_momentum': rsi / 100.0
            }
            
            data.append(row)
    
    df = pd.DataFrame(data)
    logger.info(f"Created synthetic data: {len(df)} rows, {n_days} dates, {n_stocks} stocks")
    return df


class TestCrossSectionalNormalizerV2:
    """CrossSectionalNormalizerV2テストクラス"""
    
    def test_polars_performance(self):
        """Polars版の性能テスト"""
        print("\n🧪 Testing CrossSectionalNormalizerV2 (Polars-based)...")
        
        df = create_synthetic_financial_data(n_stocks=100, n_days=200)
        
        # 訓練/テスト分割
        split_date = df['date'].quantile(0.7)
        train_df = df[df['date'] <= split_date]
        test_df = df[df['date'] > split_date]
        
        normalizer = CrossSectionalNormalizerV2(
            cache_stats=True,
            robust_clip=5.0,
            verbose=True
        )
        
        # フィット＆変換
        start_time = datetime.now()
        train_norm = normalizer.fit_transform(train_df)
        test_norm = normalizer.transform(test_df)
        elapsed = (datetime.now() - start_time).total_seconds()
        
        # 検証
        validation = normalizer.validate_transform(train_norm)
        
        print(f"✅ Polars normalization completed in {elapsed:.2f}s")
        print(f"   Train samples: {len(train_norm)}, Test samples: {len(test_norm)}")
        print(f"   Warnings: {len(validation['warnings'])}")
        
        assert len(validation['warnings']) <= 2, "Too many normalization warnings"
        assert len(train_norm) > 0, "Training normalization failed"
        assert len(test_norm) > 0, "Test normalization failed"
        
        return True


class TestWalkForwardV2:
    """WalkForwardSplitterV2テストクラス"""
    
    def test_embargo_20_splits(self):
        """embargo=20の分割テスト"""
        print("\n📅 Testing Walk-Forward with embargo=20...")
        
        df = create_synthetic_financial_data(n_stocks=30, n_days=500)
        
        splitter = WalkForwardSplitterV2(
            n_splits=3,
            embargo_days=20,
            min_train_days=100,
            min_test_days=30,
            verbose=True
        )
        
        # 分割検証
        validation = splitter.validate_split(df)
        splits = list(splitter.split(df))
        
        print(f"✅ Generated {len(splits)} splits")
        print(f"   Total overlaps: {len(validation['overlaps'])}")
        print(f"   Average gap: {np.mean([g['gap_days'] for g in validation['gaps']]):.1f} days")
        
        # 各分割のギャップをチェック
        for i, (train_idx, test_idx) in enumerate(splits):
            train_dates = pd.to_datetime(df.iloc[train_idx]['date'])
            test_dates = pd.to_datetime(df.iloc[test_idx]['date'])
            
            gap_days = (test_dates.min() - train_dates.max()).days
            print(f"   Fold {i}: Gap = {gap_days} days (embargo requirement: ≥{splitter.embargo_days})")
            
            assert gap_days >= splitter.embargo_days, f"Insufficient embargo in fold {i}"
        
        return True


class TestProductionLoaderV3:
    """ProductionDatasetV3テストクラス"""
    
    def test_lazy_loading(self):
        """Lazy loading & 列射影テスト"""
        print("\n⚡ Testing ProductionDatasetV3 (Lazy Loading)...")
        
        # 一時ファイル作成
        temp_dir = Path(tempfile.mkdtemp())
        try:
            df = create_synthetic_financial_data(n_stocks=20, n_days=100)
            
            # Parquetファイルとして保存
            parquet_files = []
            for i in range(3):
                chunk = df[i*len(df)//3:(i+1)*len(df)//3].copy()
                file_path = temp_dir / f"data_chunk_{i}.parquet"
                chunk.to_parquet(file_path)
                parquet_files.append(file_path)
            
            # データセット作成
            dataset = ProductionDatasetV3(
                data_files=parquet_files,
                config=type('Config', (), {
                    'data': type('Data', (), {
                        'time_series': type('TimeSeries', (), {
                            'prediction_horizons': [1, 5, 10, 20]
                        })()
                    })()
                })(),
                sequence_length=20,
                use_lazy_loading=True,
                memory_limit_gb=4.0
            )
            
            # テスト
            data_info = dataset.get_data_info()
            
            print(f"✅ Dataset created: {data_info['num_sequences']} sequences")
            print(f"   Feature dim: {data_info['feature_dim']}")
            print(f"   Memory usage: {data_info['memory_usage_mb']:.1f} MB")
            print(f"   Date range: {data_info['date_range']['min']} to {data_info['date_range']['max']}")
            
            # サンプルアクセステスト
            if len(dataset) > 0:
                sample = dataset[0]
                assert 'features' in sample, "Features not found in sample"
                assert sample['features'].shape[0] == 20, "Incorrect sequence length"
                print(f"   Sample features shape: {sample['features'].shape}")
            
            return True
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestGraphBuilder:
    """FinancialGraphBuilderテストクラス"""
    
    def test_time_series_correlation(self):
        """時系列相関グラフ構築テスト"""
        print("\n🕸️ Testing FinancialGraphBuilder (Time Series Correlation)...")
        
        df = create_synthetic_financial_data(n_stocks=25, n_days=150)
        codes = df['code'].unique()[:20]  # 20銘柄でテスト
        
        graph_builder = FinancialGraphBuilder(
            correlation_window=60,
            correlation_threshold=0.3,
            max_edges_per_node=5,
            include_negative_correlation=True,
            verbose=True
        )
        
        # グラフ構築
        end_date = df['date'].max()
        graph_result = graph_builder.build_graph(
            data=df,
            codes=codes,
            date_end=end_date,
            return_column='return_1d'
        )
        
        print(f"✅ Graph built: {graph_result['n_nodes']} nodes, {graph_result['n_edges']} edges")
        
        # peer特徴量テスト
        peer_features = graph_builder.get_peer_features_for_codes(codes[:5], end_date)
        
        for code, features in list(peer_features.items())[:3]:
            print(f"   {code}: peer_count={features['peer_count']}, "
                  f"peer_correlation_mean={features['peer_correlation_mean']:.3f}")
        
        # グラフ統計
        stats = graph_builder.analyze_graph_statistics(end_date)
        if stats:
            print(f"   Graph density: {stats.get('network_stats', {}).get('density', 'N/A')}")
        
        assert graph_result['n_nodes'] > 0, "No nodes in graph"
        assert len(peer_features) > 0, "No peer features generated"
        
        return True


class TestLightGBMBaseline:
    """LightGBMBaselineテストクラス"""
    
    def test_multi_horizon_baseline(self):
        """多ホライズンGBMベースラインテスト"""
        print("\n🌲 Testing LightGBM Multi-Horizon Baseline...")
        
        df = create_synthetic_financial_data(n_stocks=30, n_days=200)
        
        # 軽量設定でテスト
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 15,
            'learning_rate': 0.1,
            'n_estimators': 10,  # 高速化のため少なく
            'verbosity': -1,
            'seed': 42
        }
        
        baseline = LightGBMFinancialBaseline(
            prediction_horizons=[1, 5],  # 2ホライズンのみでテスト
            lgb_params=lgb_params,
            n_splits=2,  # 2分割で高速化
            embargo_days=20,
            normalize_features=True,
            verbose=True
        )
        
        # 学習実行
        baseline.fit(df)
        
        # 結果評価
        performance = baseline.evaluate_performance()
        results_summary = baseline.get_results_summary()
        
        print(f"✅ Baseline training completed")
        print(f"   Performance summary:")
        
        for horizon, metrics in performance.items():
            print(f"   {horizon}: mean_ic={metrics['mean_ic']:.3f}, "
                  f"mean_rank_ic={metrics['mean_rank_ic']:.3f}, "
                  f"positive_rate={metrics['ic_positive_rate']:.1%}")
        
        # 特徴量重要度テスト
        if 1 in baseline.feature_importance:
            importance_df = baseline.get_feature_importance(horizon=1, top_k=5)
            print(f"   Top 5 features for 1d horizon:")
            for _, row in importance_df.iterrows():
                print(f"     {row['feature']}: {row['importance']:.1f}")
        
        assert len(performance) > 0, "No performance metrics generated"
        assert len(results_summary) > 0, "No results summary generated"
        
        return True


class TestQualityFeatures:
    """QualityFinancialFeaturesGeneratorテストクラス"""
    
    def test_quality_features_generation(self):
        """高品質特徴量生成テスト"""
        print("\n✨ Testing Quality Features Generation...")
        
        df = create_synthetic_financial_data(n_stocks=20, n_days=100)
        
        generator = QualityFinancialFeaturesGenerator(
            use_cross_sectional_quantiles=True,
            sigma_threshold=2.0,
            verbose=True
        )
        
        # 特徴量生成
        original_cols = len(df.columns)
        enhanced_df = generator.generate_quality_features(df)
        final_cols = len(enhanced_df.columns)
        
        print(f"✅ Features enhanced: {original_cols} → {final_cols} columns (+{final_cols - original_cols})")
        
        # 特徴量品質検証
        validation = generator.validate_features(enhanced_df)
        
        print(f"   Quality validation:")
        print(f"     Zero variance features: {len(validation['zero_variance_features'])}")
        print(f"     High missing features: {len(validation['high_missing_features'])}")
        print(f"     Infinite features: {len(validation['infinite_features'])}")
        
        # 特徴量カテゴリ
        categories = validation['feature_categories']
        for category, features in categories.items():
            if features:
                print(f"     {category}: {len(features)} features")
        
        # daily_vol列の存在確認
        assert not validation['missing_daily_vol'], "daily_vol column not generated"
        assert 'sharpe_features' in categories, "Sharpe features not generated"
        assert final_cols > original_cols, "No new features added"
        
        return True


class TestIntegration:
    """統合テストクラス"""
    
    def test_full_pipeline(self):
        """フルパイプライン統合テスト"""
        print("\n🔗 Testing Full Pipeline Integration...")
        
        # 1. データ生成
        df = create_synthetic_financial_data(n_stocks=40, n_days=250)
        print(f"   Step 1: Generated {len(df)} samples")
        
        # 2. 高品質特徴量生成
        generator = QualityFinancialFeaturesGenerator(verbose=False)
        df_enhanced = generator.generate_quality_features(df)
        print(f"   Step 2: Enhanced to {len(df_enhanced.columns)} features")
        
        # 3. Cross-sectional正規化
        normalizer = CrossSectionalNormalizerV2(verbose=False)
        split_date = df_enhanced['date'].quantile(0.7)
        
        if isinstance(df_enhanced, pl.DataFrame):
            train_mask = df_enhanced['date'] <= pl.lit(split_date)
            test_mask = df_enhanced['date'] > pl.lit(split_date)
            train_df = df_enhanced.filter(train_mask)
            test_df = df_enhanced.filter(test_mask)
        else:
            train_df = df_enhanced[df_enhanced['date'] <= split_date]
            test_df = df_enhanced[df_enhanced['date'] > split_date]
        
        train_norm = normalizer.fit_transform(train_df)
        test_norm = normalizer.transform(test_df)
        print(f"   Step 3: Normalized {len(train_norm)} train + {len(test_norm)} test samples")
        
        # 4. グラフ構築
        codes = df['code'].unique()[:15]  # 15銘柄で軽量化
        graph_builder = FinancialGraphBuilder(verbose=False)
        graph_result = graph_builder.build_graph(
            data=df, codes=codes, date_end=df['date'].max()
        )
        print(f"   Step 4: Built graph with {graph_result['n_nodes']} nodes, {graph_result['n_edges']} edges")
        
        # 5. 軽量GBMベースライン
        df_pandas = df_enhanced.to_pandas() if isinstance(df_enhanced, pl.DataFrame) else df_enhanced
        baseline = LightGBMFinancialBaseline(
            prediction_horizons=[1, 5],
            n_splits=2,
            embargo_days=10,  # テスト用に短縮
            verbose=False
        )
        
        # 最小限のデータで学習
        sample_df = df_pandas.sample(n=min(1000, len(df_pandas)))
        baseline.fit(sample_df)
        performance = baseline.evaluate_performance()
        print(f"   Step 5: GBM baseline trained, performance: {len(performance)} horizons")
        
        # 6. 最終検証
        success_indicators = [
            len(df_enhanced.columns) > len(df.columns),  # 特徴量拡張
            len(train_norm) > 0,  # 正規化成功
            graph_result['n_nodes'] > 0,  # グラフ構築成功
            len(performance) > 0,  # ベースライン学習成功
        ]
        
        success_rate = sum(success_indicators) / len(success_indicators)
        
        print(f"✅ Full pipeline integration: {success_rate:.1%} success rate")
        
        pipeline_result = {
            'original_features': len(df.columns),
            'enhanced_features': len(df_enhanced.columns),
            'normalized_samples': len(train_norm) + len(test_norm),
            'graph_nodes': graph_result['n_nodes'],
            'graph_edges': graph_result['n_edges'],
            'model_horizons': len(performance),
            'success_rate': success_rate
        }
        
        return pipeline_result


def run_all_tests():
    """全テストを実行"""
    if not COMPONENTS_AVAILABLE:
        print("❌ Components not available, skipping tests")
        return False
    
    print("🚀 Starting Gogooku3 Improvements Integration Tests...\n")
    
    test_results = {}
    
    try:
        # 個別コンポーネントテスト
        test_results['normalizer_v2'] = TestCrossSectionalNormalizerV2().test_polars_performance()
        test_results['walk_forward_v2'] = TestWalkForwardV2().test_embargo_20_splits()
        test_results['loader_v3'] = TestProductionLoaderV3().test_lazy_loading()
        test_results['graph_builder'] = TestGraphBuilder().test_time_series_correlation()
        test_results['gbm_baseline'] = TestLightGBMBaseline().test_multi_horizon_baseline()
        test_results['quality_features'] = TestQualityFeatures().test_quality_features_generation()
        
        # 統合テスト
        test_results['full_pipeline'] = TestIntegration().test_full_pipeline()
        
        # 結果サマリー
        passed_tests = sum(1 for result in test_results.values() if result)
        total_tests = len(test_results)
        
        print(f"\n🎉 Test Results: {passed_tests}/{total_tests} tests passed")
        
        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {test_name}: {status}")
        
        # 統合テスト結果の詳細
        if isinstance(test_results['full_pipeline'], dict):
            pipeline_result = test_results['full_pipeline']
            print(f"\n📊 Pipeline Integration Details:")
            print(f"   Features: {pipeline_result['original_features']} → {pipeline_result['enhanced_features']}")
            print(f"   Normalized samples: {pipeline_result['normalized_samples']:,}")
            print(f"   Graph: {pipeline_result['graph_nodes']} nodes, {pipeline_result['graph_edges']} edges")
            print(f"   Models: {pipeline_result['model_horizons']} horizons")
            print(f"   Success rate: {pipeline_result['success_rate']:.1%}")
        
        overall_success = passed_tests == total_tests
        
        print(f"\n🏆 Overall Result: {'✅ ALL TESTS PASSED' if overall_success else '⚠️ SOME TESTS FAILED'}")
        
        return overall_success
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)